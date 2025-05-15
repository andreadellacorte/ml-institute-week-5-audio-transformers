import torch
import torch.nn as nn
import torchaudio.transforms as T
import math

class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for Transformer models.
    Expects input shape [seq_len, batch_size, embedding_dim].
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SpectrogramTransformer(nn.Module):
    """
    Transformer model to convert audio waveforms to MIDI token sequences.
    Input waveforms are converted to MelSpectrograms, then processed by an
    Encoder-Decoder architecture.
    """
    def __init__(self,
                 midi_vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 sample_rate: int = 16000, # For MelSpectrogram
                 n_fft: int = 2048,        # For MelSpectrogram
                 hop_length: int = 512,    # For MelSpectrogram
                 n_mels: int = 128,        # Number of Mel bins
                 max_spectrogram_len: int = 2000, # Max frames in spectrogram for PE
                 max_midi_len: int = 1024         # Max MIDI tokens for PE
                ):
        super().__init__()
        self.d_model = d_model

        self.spectrogram_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        # Project Mel bins to d_model for the encoder input
        self.encoder_input_proj = nn.Linear(n_mels, d_model)

        self.midi_embedding = nn.Embedding(midi_vocab_size, d_model)

        self.pos_encoder_spectrogram = PositionalEncoding(d_model, dropout, max_len=max_spectrogram_len)
        self.pos_encoder_midi = PositionalEncoding(d_model, dropout, max_len=max_midi_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.output_linear = nn.Linear(d_model, midi_vocab_size)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.midi_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder_input_proj.weight.data.uniform_(-initrange, initrange)
        if self.encoder_input_proj.bias is not None:
            self.encoder_input_proj.bias.data.zero_()
        self.output_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a square causal boolean mask for the decoder's self-attention.
           True values indicate positions that should be masked.
        """
        # Create a boolean mask where True means "masked".
        # For causal attention, an upper triangular matrix (where j > i) is masked.
        mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self,
                src_waveforms: torch.Tensor,    # Shape: [batch_size, num_samples] or [batch_size, 1, num_samples]
                tgt_midi_tokens: torch.Tensor,  # Shape: [batch_size, tgt_seq_len]
                src_padding_mask: torch.Tensor = None, # Shape: [batch_size, src_spec_len], True for pad
                tgt_padding_mask: torch.Tensor = None, # Shape: [batch_size, tgt_seq_len], True for pad
                memory_key_padding_mask: torch.Tensor = None # Shape: [batch_size, src_spec_len], True for pad
               ):
        """
        Forward pass of the SpectrogramTransformer.

        Args:
            src_waveforms: Raw audio waveforms.
            tgt_midi_tokens: Target MIDI token sequences (for teacher forcing).
            src_padding_mask: Padding mask for the source spectrogram sequences.
            tgt_padding_mask: Padding mask for the target MIDI sequences.
            memory_key_padding_mask: Padding mask for the encoder output (memory),
                                     typically same as src_padding_mask.
        Returns:
            Output logits over the MIDI vocabulary, shape [batch_size, tgt_seq_len, midi_vocab_size].
        """
        # 1. Compute and Prepare Spectrograms (Encoder Input)
        if src_waveforms.ndim == 3 and src_waveforms.size(1) == 1: # [B, 1, T_audio] -> [B, T_audio]
            src_waveforms = src_waveforms.squeeze(1)
        
        # src_spec: [batch_size, n_mels, time_frames]
        src_spec = self.spectrogram_transform(src_waveforms)
        
        # Reshape for projection & transformer: [batch_size, time_frames, n_mels]
        src_spec = src_spec.permute(0, 2, 1)
        
        # Project n_mels to d_model: [batch_size, time_frames, d_model]
        src_projected = self.encoder_input_proj(src_spec) * math.sqrt(self.d_model)
        
        # Add positional encoding. Current PE expects [seq_len, batch_size, dim].
        # Permute: [time_frames, batch_size, d_model]
        src_pe_input = src_projected.permute(1, 0, 2)
        src_pe_output = self.pos_encoder_spectrogram(src_pe_input)
        # Permute back for batch_first Transformer: [batch_size, time_frames, d_model]
        encoder_input = src_pe_output.permute(1, 0, 2)

        # 2. Encode Spectrograms
        # memory: [batch_size, time_frames, d_model]
        memory = self.transformer_encoder(encoder_input, src_key_padding_mask=src_padding_mask)

        # 3. Prepare Target MIDI (Decoder Input)
        # tgt_embedded: [batch_size, tgt_seq_len, d_model]
        tgt_embedded = self.midi_embedding(tgt_midi_tokens) * math.sqrt(self.d_model)
        
        # Add positional encoding.
        # Permute: [tgt_seq_len, batch_size, d_model]
        tgt_pe_input = tgt_embedded.permute(1, 0, 2)
        tgt_pe_output = self.pos_encoder_midi(tgt_pe_input)
        # Permute back for batch_first Transformer: [batch_size, tgt_seq_len, d_model]
        decoder_input = tgt_pe_output.permute(1, 0, 2)

        # 4. Create masks for Decoder
        tgt_seq_len = tgt_midi_tokens.size(1)
        # tgt_mask (look-ahead mask): [tgt_seq_len, tgt_seq_len]
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, tgt_midi_tokens.device)
        
        # tgt_padding_mask: [batch_size, tgt_seq_len] (True for pad tokens)
        # memory_key_padding_mask: [batch_size, time_frames] (True for pad tokens in memory)
        # This is often the same as src_padding_mask.

        # 5. Decode
        # decoder_output: [batch_size, tgt_seq_len, d_model]
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # 6. Final Output Layer
        # predicted_midi_logits: [batch_size, tgt_seq_len, midi_vocab_size]
        predicted_midi_logits = self.output_linear(decoder_output)

        return predicted_midi_logits

# Example Usage (Illustrative - requires actual data and vocab size)
if __name__ == '__main__':
    MIDI_VOCAB_SIZE = 30000 # Example vocab size
    D_MODEL = 512
    MAX_MIDI_LEN = 512
    MAX_SPEC_LEN = 1000
    SAMPLE_RATE = 16000
    N_MELS = 128

    model = SpectrogramTransformer(
        midi_vocab_size=MIDI_VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=8,
        num_encoder_layers=3, # Smaller for quick test
        num_decoder_layers=3, # Smaller for quick test
        dim_feedforward=2048,
        dropout=0.1,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        max_spectrogram_len=MAX_SPEC_LEN,
        max_midi_len=MAX_MIDI_LEN
    )
    model.eval()

    # Dummy data
    batch_size = 2
    num_audio_samples = SAMPLE_RATE * 5 # 5 seconds of audio
    tgt_seq_length = 100 # 100 MIDI tokens

    dummy_waveforms = torch.randn(batch_size, num_audio_samples)
    dummy_midi_tokens = torch.randint(0, MIDI_VOCAB_SIZE, (batch_size, tgt_seq_length))
    
    # Dummy padding masks (optional, True means pad)
    # For simplicity, assume no padding in this dummy example for src
    # For tgt, let's say last 10 tokens are padding for the first sample
    dummy_tgt_padding_mask = torch.zeros(batch_size, tgt_seq_length, dtype=torch.bool)
    if batch_size > 0:
        dummy_tgt_padding_mask[0, -10:] = True 
    
    print(f"Input waveform shape: {dummy_waveforms.shape}")
    print(f"Input MIDI tokens shape: {dummy_midi_tokens.shape}")
    if dummy_tgt_padding_mask is not None:
        print(f"Target padding mask shape: {dummy_tgt_padding_mask.shape}")

    with torch.no_grad():
        logits = model(
            src_waveforms=dummy_waveforms,
            tgt_midi_tokens=dummy_midi_tokens,
            tgt_padding_mask=dummy_tgt_padding_mask
            # src_padding_mask and memory_key_padding_mask left as None for this example
        )
    
    print(f"Output logits shape: {logits.shape}") # Expected: [batch_size, tgt_seq_length, MIDI_VOCAB_SIZE]

    # To get predicted tokens (e.g., during inference or evaluation):
    # predicted_tokens = torch.argmax(logits, dim=-1)
    # print(f"Predicted tokens shape: {predicted_tokens.shape}")

