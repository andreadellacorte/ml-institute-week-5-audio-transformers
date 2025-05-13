import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchaudio
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
                    nn.GELU(),  # Replace ReLU with GELU for better performance
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
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            
        # Transpose to [batch_size, sequence_length, features]
        return x.transpose(1, 2)


class SpectrogramFeatureExtractor(nn.Module):
    """
    Extracts features from audio by converting it to a spectrogram and processing with 2D convolutions.
    """
    def __init__(
        self,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,  # Reduced from 128 or 80 to prevent filterbank issues
        base_filters: int = 32,
        sample_rate: int = 16000,
        normalize: bool = True
    ):
        super().__init__()
        
        # Spectrogram transformation parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.normalize = normalize
        
        # Mel spectrogram transform with adjusted parameters
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,  # Add minimum frequency threshold
            f_max=sample_rate/2 - 100,  # Add maximum frequency
            power=2.0,
        )
        
        # Amplitude to DB conversion
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        # 2D convolutional layers for feature extraction
        self.conv_layers = nn.ModuleList()
        
        # First Conv2D layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(1, base_filters, kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.BatchNorm2d(base_filters),
                nn.GELU(),
                nn.Dropout2d(0.1)
            )
        )
        
        # Additional Conv2D layers with increasing filters
        layer_filters = [base_filters, base_filters * 2, base_filters * 4]
        for i in range(len(layer_filters) - 1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(layer_filters[i], layer_filters[i+1], kernel_size=(3, 3), stride=(2, 2), padding=1),
                    nn.BatchNorm2d(layer_filters[i+1]),
                    nn.GELU(),
                    nn.Dropout2d(0.1)
                )
            )
        
        # Store output dimension for later use
        self.output_dim = layer_filters[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw audio waveform [batch_size, channels, time]
            
        Returns:
            Spectrogram features [batch_size, sequence_length, feature_dim]
        """
        batch_size = x.size(0)
        
        # Convert to mono if needed
        if x.size(1) > 1:
            x = torch.mean(x, dim=1, keepdim=True)
        
        # Apply Mel spectrogram transform
        x = self.mel_spectrogram(x)
        
        # Add a small epsilon to avoid log(0)
        x = x + 1e-10
        
        # Convert to decibels
        x = self.amplitude_to_db(x)
        
        # Add channel dimension if not present
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Apply normalization if enabled - improved implementation
        if self.normalize:
            # More stable normalization with fixed epsilon
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True) + 1e-5
            x = (x - mean) / std
        
        # Check for NaN values and replace with zeros
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply 2D convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Reshape to sequence form: [batch, channels, height, width] -> [batch, time, features]
        # Treat the frequency dimension as features and time as sequence
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2)  # [batch, width, channels, height]
        x = x.reshape(batch_size, width, channels * height)  # [batch, time, features]
        
        return x


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as a buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            
        Returns:
            Output tensor with positional encoding added [batch_size, seq_length, d_model]
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
            nn.GELU(),  # Use GELU instead of ReLU for better performance
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
        # Pre-LayerNorm pattern (tends to be more stable) instead of Post-LayerNorm
        x_norm = self.norm1(x)
        
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=mask
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
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


class SpectrogramTransformerEncoder(nn.Module):
    """
    An encoder-only transformer that processes audio spectrograms for classification.
    """
    def __init__(
        self,
        num_classes: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        base_filters: int = 32,
        sample_rate: int = 16000
    ):
        super().__init__()
        
        # Spectrogram feature extraction
        self.feature_extractor = SpectrogramFeatureExtractor(
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            base_filters=base_filters,
            sample_rate=sample_rate
        )
        
        # Project extracted features to transformer dimension
        self.input_projection = nn.Linear(self.feature_extractor.output_dim * n_mels // 8, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers (use the same TransformerEncoderBlock as the raw audio model)
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
        # Extract features using spectrogram
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
        
        # Initialize with optimizer that has weight decay (exclude norm layers and biases)
        # This helps with generalization and training stability
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if 'norm' in name or 'bias' in name:
                no_decay.append(param)
            else:
                decay.append(param)
                
        optimizer_grouped_parameters = [
            {'params': decay, 'weight_decay': 0.01},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        # Learning rate scheduler with warm-up and cosine decay
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
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
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
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


class SpectrogramClassifier:
    """
    Wrapper class for training and evaluating the SpectrogramTransformerEncoder model.
    """
    def __init__(
        self,
        num_classes: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        base_filters: int = 32,
        sample_rate: int = 16000,
        learning_rate: float = 1e-4
    ):
        self.model = SpectrogramTransformerEncoder(
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            base_filters=base_filters,
            sample_rate=sample_rate
        )
        
        # Initialize optimizer with weight decay
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if 'norm' in name or 'bias' in name:
                no_decay.append(param)
            else:
                decay.append(param)
                
        optimizer_grouped_parameters = [
            {'params': decay, 'weight_decay': 0.01},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
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
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
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
        device = next(self.model.parameters()).device
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