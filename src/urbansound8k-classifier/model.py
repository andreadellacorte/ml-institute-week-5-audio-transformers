import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class AudioFeatureExtractor(nn.Module):
    """
    Extracts features from raw audio waveforms using 1D convolutions.
    This acts as a learnable alternative to traditional feature extraction methods.
    """
    def __init__(
        self, 
        in_channels: int = 1, 
        base_filters: int = 32, 
        kernel_sizes: Tuple[int, ...] = (128, 64, 32, 16),
        stride: int = 4
    ):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        current_channels = in_channels
        
        # Create a stack of convolutional layers with increasing filter counts
        for i, kernel_size in enumerate(kernel_sizes):
            out_channels = base_filters * (2 ** i)
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(current_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )
            current_channels = out_channels
            
        self.output_dim = current_channels
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw audio waveform [batch_size, channels, time]
            
        Returns:
            Audio features [batch_size, sequence_length, feature_dim]
        """
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
            
        # Transpose to [batch_size, sequence_length, features]
        return x.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings.
    """
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings [batch_size, seq_length, embedding_dim]
            
        Returns:
            Embeddings with positional encoding added [batch_size, seq_length, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    A single transformer encoder block with multi-head self-attention
    followed by a position-wise feed-forward network.
    """
    def __init__(
        self, 
        d_model: int = 256, 
        nhead: int = 8, 
        dim_feedforward: int = 1024, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            mask: Optional mask for self-attention
            
        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        # Self-attention with residual connection and normalization
        attn_output, _ = self.self_attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            attn_mask=mask
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class AudioTransformerEncoder(nn.Module):
    """
    An encoder-only transformer for audio classification.
    """
    def __init__(
        self,
        num_classes: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        feature_extractor_base_filters: int = 32
    ):
        super().__init__()
        
        # Audio feature extraction from raw waveform
        self.feature_extractor = AudioFeatureExtractor(
            base_filters=feature_extractor_base_filters
        )
        
        # Project extracted features to transformer dimension
        self.input_projection = nn.Linear(self.feature_extractor.output_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw audio waveform [batch_size, channels, time]
            
        Returns:
            Class logits [batch_size, num_classes]
        """
        # Extract features
        x = self.feature_extractor(x)
        
        # Project to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        # Global pooling (mean over sequence dimension)
        x = torch.mean(x, dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class AudioClassifier:
    """
    Wrapper class for training and evaluating the AudioTransformerEncoder model.
    """
    def __init__(
        self,
        num_classes: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        feature_extractor_base_filters: int = 32,
        learning_rate: float = 1e-4
    ):
        self.model = AudioTransformerEncoder(
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            feature_extractor_base_filters=feature_extractor_base_filters
        )
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, batch: dict) -> dict:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing 'waveform' and 'label'
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        waveform = batch['waveform']
        labels = batch['label']
        
        # Forward pass
        logits = self.model(waveform)
        loss = self.criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            
        Returns:
            Dictionary of metrics (loss, accuracy)
        """
        device = next(self.model.parameters()).device  # Get the device the model is on
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Ensure data is on the correct device
                waveform = batch['waveform'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = self.model(waveform)
                loss = self.criterion(logits, labels)
                
                # Update metrics
                total_loss += loss.item() * waveform.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += waveform.size(0)
        
        return {
            'loss': total_loss / total if total > 0 else float('inf'),
            'accuracy': correct / total if total > 0 else 0
        }
    
    def predict(self, waveform: torch.Tensor) -> dict:
        """
        Make a prediction for a single waveform.
        
        Args:
            waveform: Audio waveform [channels, time]
            
        Returns:
            Dictionary with prediction results
        """
        self.model.eval()
        
        # Add batch dimension if needed
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(waveform)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
            
        return {
            'class_id': pred_class,
            'confidence': confidence,
            'probabilities': probs[0].cpu().numpy()
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, filepath)
    
    def load(self, filepath: str, device: str = 'cpu') -> None:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to load the model from
            device: Device to load the model to
        """
        checkpoint = torch.load(filepath, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])