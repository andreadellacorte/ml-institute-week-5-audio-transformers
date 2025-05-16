# filepath: /Users/andreadellacorte/Documents/Workspace/GitHub/ml-institute-week-5-audio-transformers/src/maestro-transformer-v2/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional, Dict, Tuple, List, Union


class SpecEncoder(nn.Module):
    """
    Encoder for processing mel spectrograms into a latent representation.
    Uses convolutional layers to process the 2D spectrogram data.
    """
    def __init__(
        self, 
        n_mels: int = 128,
        d_model: int = 512,
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [2, 2, 2, 2],
        channels: List[int] = [32, 64, 128, 256]
    ):
        """
        Initialize the spectrogram encoder.
        
        Args:
            n_mels: Number of mel bands in the input spectrogram
            d_model: Dimension of the transformer model
            kernel_sizes: Kernel sizes for each convolutional layer
            strides: Strides for each convolutional layer
            channels: Number of channels for each convolutional layer
        """
        super().__init__()
        
        # Input shape: [batch_size, 1, n_mels, time]
        self.conv_layers = nn.ModuleList()
        
        # First layer: from 1 channel to first number of channels
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=kernel_sizes[0]//2),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU()
            )
        )
        
        # Remaining convolutional layers
        for i in range(1, len(channels)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i-1], channels[i], kernel_size=kernel_sizes[i], 
                             stride=strides[i], padding=kernel_sizes[i]//2),
                    nn.BatchNorm2d(channels[i]),
                    nn.ReLU()
                )
            )
        
        # Calculate output shape after convolutions for projection
        self.feature_dim = self._get_conv_output_dim(n_mels)
        
        # Final projection to d_model
        self.projection = nn.Linear(self.feature_dim, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def _get_conv_output_dim(self, n_mels: int) -> int:
        """
        Calculate the output dimension after convolutions.
        """
        # Start with a dummy input to pass through the convolutional layers
        dummy_input = torch.zeros(1, 1, n_mels, 100)  # Arbitrary time dimension
        
        # Pass through convolutional layers
        x = dummy_input
        for conv in self.conv_layers:
            x = conv(x)
        
        # Get the final feature dimension
        return x.shape[1] * x.shape[2]
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the spectrogram encoder.
        
        Args:
            x: Input spectrogram tensor [batch_size, n_mels, time]
            
        Returns:
            Encoded representation [batch_size, time', d_model]
        """
        # Add channel dimension: [batch_size, 1, n_mels, time]
        x = x.unsqueeze(1)
        
        # Pass through convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape: [batch_size, channels, height, time] -> [batch_size, time, channels*height]
        batch_size, channels, height, time = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time, channels * height)
        
        # Project to d_model dimension
        x = self.projection(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Dimension of the model
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to the input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):
    """
    Custom transformer decoder layer with cross-attention to the encoded spectrogram.
    """
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize the transformer decoder layer.
        
        Args:
            d_model: Dimension of the model
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention with encoder output
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for the decoder layer.
        
        Args:
            tgt: Target sequence [batch_size, tgt_len, d_model]
            memory: Memory from encoder [batch_size, src_len, d_model]
            tgt_mask: Target sequence mask [tgt_len, tgt_len]
            tgt_key_padding_mask: Target key padding mask [batch_size, tgt_len]
            memory_key_padding_mask: Memory key padding mask [batch_size, src_len]
            
        Returns:
            Output tensor [batch_size, tgt_len, d_model]
        """
        # Self-attention
        tgt2 = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2 = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class SpectrogramToMIDITransformer(nn.Module):
    """
    Transformer model that converts spectrograms to MIDI tokens.
    Consists of a spectrogram encoder and a transformer decoder.
    """
    def __init__(
        self,
        tokenizer,
        n_mels: int = 128,
        vocab_size: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        pad_token_id: int = 0
    ):
        """
        Initialize the spectrogram-to-MIDI transformer.
        
        Args:
            tokenizer: Tokenizer for mapping tokens
            n_mels: Number of mel bands in the input spectrogram
            vocab_size: Size of the MIDI token vocabulary
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            pad_token_id: Padding token ID
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        
        # Spectrogram encoder
        self.spec_encoder = SpecEncoder(
            n_mels=n_mels,
            d_model=d_model
        )
        
        # Token embedding for MIDI tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Final output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def _generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generate a square mask for the sequence to prevent attending to future tokens.
        
        Args:
            sz: Size of the square mask
            
        Returns:
            Mask tensor
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _create_padding_mask(self, seq: Tensor, pad_token_id: int) -> Tensor:
        """
        Create padding mask for the sequence.
        
        Args:
            seq: Input sequence [batch_size, seq_len]
            pad_token_id: Padding token ID
            
        Returns:
            Padding mask [batch_size, seq_len]
        """
        return seq == pad_token_id
        
    def forward(
        self,
        spectrogram: Tensor,
        target_tokens: Optional[Tensor] = None,
        target_padding_mask: Optional[Tensor] = None,
        generate: bool = False,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through the transformer.
        
        Args:
            spectrogram: Input spectrogram tensor [batch_size, n_mels, time]
            target_tokens: Target MIDI tokens (for training) [batch_size, tgt_len]
            target_padding_mask: Target padding mask [batch_size, tgt_len]
            generate: Whether to generate MIDI tokens autoregressively
            max_length: Maximum length for generation
            temperature: Sampling temperature (higher = more random)
            top_k: Sample from top k most likely tokens
            top_p: Sample from top tokens with cumulative probability p (nucleus sampling)
            
        Returns:
            If generate=False: Output logits [batch_size, tgt_len, vocab_size]
            If generate=True: Dictionary with generated tokens and other info
        """
        # Encode the spectrogram
        memory = self.spec_encoder(spectrogram)
        
        # If generating, perform autoregressive generation
        if generate:
            return self.generate_tokens(memory, max_length, temperature, top_k, top_p)
        
        # Regular forward pass (training/validation)
        batch_size, tgt_len = target_tokens.shape
        
        # Embed target tokens
        tgt_embeddings = self.token_embedding(target_tokens)
        tgt_embeddings = self.pos_encoder(tgt_embeddings)
        
        # Create target mask to prevent attending to future tokens
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(spectrogram.device)
        
        # Convert padding mask to float type to match attn_mask
        # This fixes the "mismatched key_padding_mask and attn_mask" warning
        if target_padding_mask is not None:
            target_padding_mask = target_padding_mask.float().masked_fill(
                target_padding_mask, float('-inf')).masked_fill(~target_padding_mask, float(0.0))
        
        # Process through decoder layers
        output = tgt_embeddings
        for layer in self.decoder_layers:
            output = layer(
                output, 
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=target_padding_mask,
                memory_key_padding_mask=None  # We don't use memory padding mask for spectrograms
            )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def generate_tokens(
        self, 
        memory: Tensor, 
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Tensor]:
        """
        Generate MIDI tokens autoregressively.
        
        Args:
            memory: Encoded spectrogram [batch_size, src_len, d_model]
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Sample from top k most likely tokens
            top_p: Sample from top tokens with cumulative probability p (nucleus sampling)
            
        Returns:
            Dictionary with generated tokens and other information
        """
        batch_size = memory.shape[0]
        device = memory.device
        
        # Initialize with start token (assuming token ID 1 is the start token)
        # In a real implementation, this should be a proper START token specific to your vocabulary
        start_token_id = 1
        curr_tokens = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        # Keep track of all generated tokens
        all_tokens = []
        all_probs = []
        
        # Generate tokens one by one
        for step in range(max_length):
            # Embed the current tokens
            token_embeddings = self.token_embedding(curr_tokens)
            token_embeddings = self.pos_encoder(token_embeddings)
            
            # Create mask to prevent attending to future tokens
            tgt_mask = self._generate_square_subsequent_mask(curr_tokens.size(1)).to(device)
            
            # Process through decoder layers
            output = token_embeddings
            for layer in self.decoder_layers:
                output = layer(
                    output, 
                    memory,
                    tgt_mask=tgt_mask
                )
            
            # Project to vocabulary - only need the last token prediction
            next_token_logits = self.output_projection(output[:, -1])
            
            # Apply temperature
            next_token_logits = next_token_logits / max(temperature, 1e-8)
            
            # Apply top-k sampling
            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) sampling
            if top_p is not None and top_p > 0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated tokens
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
            
            # Store probabilities of selected tokens
            next_token_probs = torch.gather(probs, 1, next_token)
            all_probs.append(next_token_probs)
            
            # Check if we've generated an end token (assuming token ID 2 is the end token)
            # In a real implementation, this should be a proper END token specific to your vocabulary
            end_token_id = 2
            if (next_token == end_token_id).all():
                break
        
        # Validate token alignment with tokenizer's vocabulary
        if hasattr(self.tokenizer, 'vocab'):
            token_mapping = {v: k for k, v in self.tokenizer.vocab.items()}
            mapped_tokens = []
            unmapped_tokens = []

            for token in curr_tokens.view(-1):
                token_str = token_mapping.get(token.item(), None)
                if token_str:
                    mapped_tokens.append(token_str)
                else:
                    unmapped_tokens.append(token.item())
                    mapped_tokens.append(f"Unknown_{token.item()}")

            print(f"  - Mapped tokens: {mapped_tokens}")
            if unmapped_tokens:
                print(f"  - Unmapped tokens: {unmapped_tokens}")
        
        # Stack probabilities
        all_probs = torch.cat(all_probs, dim=1) if all_probs else torch.empty(batch_size, 0, device=device)
        
        return {
            "tokens": curr_tokens,
            "probs": all_probs
        }


# Loss function for training
class MIDIGenerationLoss(nn.Module):
    """
    Loss function for MIDI token generation.
    """
    def __init__(self, ignore_index: int = 0):
        """
        Initialize the loss function.
        
        Args:
            ignore_index: Index to ignore in the loss calculation (usually padding)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self, 
        logits: Tensor, 
        target: Tensor
    ) -> Tensor:
        """
        Calculate loss.
        
        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            target: Target tokens [batch_size, seq_len]
            
        Returns:
            Loss value
        """
        # Reshape logits to [batch_size * seq_len, vocab_size]
        logits = logits.contiguous().view(-1, logits.size(-1))
        # Reshape target to [batch_size * seq_len]
        target = target.contiguous().view(-1)
        
        return self.criterion(logits, target)


# Example of how to use the model
def example_usage():
    # Parameters
    batch_size = 4
    n_mels = 128
    time_steps = 100
    vocab_size = 512
    max_seq_len = 50
    
    # Create model
    tokenizer = None  # Replace with actual tokenizer
    model = SpectrogramToMIDITransformer(
        tokenizer=tokenizer,
        n_mels=n_mels,
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_decoder_layers=6
    )
    
    # Create dummy inputs
    spectrogram = torch.randn(batch_size, n_mels, time_steps)
    target_tokens = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    target_padding_mask = torch.zeros_like(target_tokens).bool()
    
    # Forward pass (training)
    logits = model(spectrogram, target_tokens, target_padding_mask)
    
    # Calculate loss
    loss_fn = MIDIGenerationLoss()
    loss = loss_fn(logits, target_tokens)
    print(f"Training logits shape: {logits.shape}")
    print(f"Loss: {loss.item()}")
    
    # Generate tokens
    with torch.no_grad():
        output = model(spectrogram, generate=True, max_length=100)
    
    generated_tokens = output["tokens"]
    print(f"Generated tokens shape: {generated_tokens.shape}")


if __name__ == "__main__":
    example_usage()