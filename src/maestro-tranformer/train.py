import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import os
from pathlib import Path
from tqdm import tqdm  # Added tqdm
import sys  # Required for sys.platform in play_midi_from_tokens

# Project imports
from model import SpectrogramTransformer
from dataset import MaestroDataset
from src.config import CHECKPOINTS_DATA_DIR, PROJ_ROOT  # Added PROJ_ROOT for temp file path

# For REMI tokenizer vocab size and special token IDs
from miditok import REMI, TokenizerConfig

# --- Constants ---

# Model Hyperparameters
D_MODEL = 256  # Dimensionality of the model
NHEAD = 4      # Number of attention heads
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 1024  # Dimension of feedforward network
DROPOUT = 0.1

# Spectrogram parameters (should align with model defaults and dataset)
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
CHUNK_DURATION_SEC = 4.0  # Duration of audio chunks in seconds (e.g., 10 seconds)
MAX_MIDI_TOKENS_PER_CHUNK = 1024  # Max MIDI tokens for a chunk (e.g., 1024)

# Calculate PE lengths based on chunking parameters
MAX_SPECTROGRAM_LEN_PE = math.floor((CHUNK_DURATION_SEC * SAMPLE_RATE) / HOP_LENGTH) + 1
MAX_MIDI_LEN_PE = MAX_MIDI_TOKENS_PER_CHUNK  # Or a bit more as a buffer if needed

# Training Hyperparameters
NUM_EPOCHS = 10  # Number of training epochs
BATCH_SIZE = 4  # Batch size (adjust based on GPU memory)
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_FILENAME = "spectrogram_transformer_maestro.pth"
MODEL_SAVE_PATH = CHECKPOINTS_DATA_DIR / MODEL_FILENAME
CHUNK_PLAYBACK_INTERVAL = 10  # Play MIDI every 10 batches
TEMP_MIDI_PLAYBACK_FILE = PROJ_ROOT / "temp_playback.mid"  # Temporary MIDI file path

# Initialize tokenizer (consistent with MaestroDataset to get vocab size and PAD_ID)
tokenizer_config_for_setup = TokenizerConfig(
    num_velocities=16, use_chords=True, use_programs=False, use_sustain_pedals=True
)
remi_tokenizer_for_setup = REMI(tokenizer_config_for_setup)
MIDI_VOCAB_SIZE = len(remi_tokenizer_for_setup)
PAD_TOKEN_ID = remi_tokenizer_for_setup.pad_token_id  # This is 0 for default REMI

# --- MIDI Playback Function ---
def play_midi_from_tokens(token_ids, tokenizer, temp_filepath):
    """Converts token IDs to a MIDI file and plays it using system default."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().tolist()

    filtered_tokens = [token for token in token_ids if token != tokenizer.pad_token_id]

    if not filtered_tokens:
        print("No non-padding tokens to play.")
        return

    try:
        midi_score = tokenizer.tokens_to_midi([filtered_tokens])
        midi_score.dump_midi(temp_filepath)
        print(f"Playing MIDI file: {temp_filepath}")
        if os.name == 'posix':
            os.system(f"open \"{temp_filepath}\"" if sys.platform == 'darwin' else f"xdg-open \"{temp_filepath}\"")
        elif os.name == 'nt':
            os.startfile(temp_filepath)
        else:
            print(f"Cannot automatically play MIDI on this OS. File saved at {temp_filepath}")
    except Exception as e:
        print(f"Error during MIDI playback: {e}")

# --- Collate Function ---
def collate_fn(batch):
    waveforms = [item['waveform'].squeeze(0) for item in batch if item is not None and 'waveform' in item]
    midi_tokens = [torch.tensor(item['midi_tokens'][0]) for item in batch if item is not None and 'midi_tokens' in item and item['midi_tokens']]

    if not waveforms or not midi_tokens:
        return None

    original_waveform_lengths = [wf.shape[0] for wf in waveforms]
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    padded_midi_tokens = pad_sequence(midi_tokens, batch_first=True, padding_value=PAD_TOKEN_ID)

    tgt_padding_mask = (padded_midi_tokens == PAD_TOKEN_ID)
    max_frames_in_batch = math.floor(padded_waveforms.shape[1] / HOP_LENGTH) + 1

    src_padding_mask_list = []
    for length in original_waveform_lengths:
        actual_num_frames = math.floor(length / HOP_LENGTH) + 1
        padding_needed = max_frames_in_batch - actual_num_frames
        if padding_needed < 0:
            padding_needed = 0
        mask_row = torch.cat((
            torch.zeros(actual_num_frames, dtype=torch.bool, device=DEVICE),
            torch.ones(padding_needed, dtype=torch.bool, device=DEVICE)
        ))
        if mask_row.shape[0] < max_frames_in_batch:
            mask_row = torch.cat((mask_row, torch.ones(max_frames_in_batch - mask_row.shape[0], dtype=torch.bool, device=DEVICE)))
        elif mask_row.shape[0] > max_frames_in_batch:
            mask_row = mask_row[:max_frames_in_batch]
        src_padding_mask_list.append(mask_row)

    src_padding_mask = torch.stack(src_padding_mask_list) if src_padding_mask_list else None

    return {
        'waveforms': padded_waveforms.to(DEVICE),
        'midi_tokens': padded_midi_tokens.to(DEVICE),
        'tgt_padding_mask': tgt_padding_mask.to(DEVICE),
        'src_padding_mask': src_padding_mask.to(DEVICE) if src_padding_mask is not None else None
    }

# --- Main Training Function ---
def train():
    print(f"Using device: {DEVICE}")
    print(f"MIDI Vocab Size (from REMI tokenizer): {MIDI_VOCAB_SIZE}")
    print(f"Padding Token ID for MIDI: {PAD_TOKEN_ID}")

    print("Loading MAESTRO dataset...")
    try:
        train_dataset = MaestroDataset(
            split="train",
            sample_rate=SAMPLE_RATE,
            chunk_duration_sec=CHUNK_DURATION_SEC,
            max_midi_tokens_per_chunk=MAX_MIDI_TOKENS_PER_CHUNK
        )
        val_dataset = MaestroDataset(
            split="validation",
            sample_rate=SAMPLE_RATE,
            chunk_duration_sec=CHUNK_DURATION_SEC,
            max_midi_tokens_per_chunk=MAX_MIDI_TOKENS_PER_CHUNK
        )

        print("Attempting to fetch one item from train_dataset to confirm it's working...")
        first_train_item = next(iter(train_dataset))
        print(f"Successfully fetched first training item. Waveform shape: {first_train_item['waveform'].shape}, MIDI (first track, 10 tokens): {first_train_item['midi_tokens'][0][:10]}")

        train_dataset = MaestroDataset(
            split="train",
            sample_rate=SAMPLE_RATE,
            chunk_duration_sec=CHUNK_DURATION_SEC,
            max_midi_tokens_per_chunk=MAX_MIDI_TOKENS_PER_CHUNK
        )
        val_dataset = MaestroDataset(
            split="validation",
            sample_rate=SAMPLE_RATE,
            chunk_duration_sec=CHUNK_DURATION_SEC,
            max_midi_tokens_per_chunk=MAX_MIDI_TOKENS_PER_CHUNK
        )

    except StopIteration:
        print("Error: A dataset split (train or validation) is empty or failed to load. Please check dataset integrity and paths.")
        return
    except Exception as e:
        print(f"Error initializing or checking datasets: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    model = SpectrogramTransformer(
        midi_vocab_size=MIDI_VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        max_spectrogram_len=MAX_SPECTROGRAM_LEN_PE,
        max_midi_len=MAX_MIDI_LEN_PE
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Saving model checkpoints to: {MODEL_SAVE_PATH.parent}")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        train_batch_count = 0
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader) if hasattr(train_loader, '__len__') else None, desc=f"Epoch {epoch+1} Training")
        for i, batch in progress_bar:
            if batch is None:
                print(f"Warning: Skipping empty batch {i} in training.")
                continue

            waveforms = batch['waveforms']
            midi_tokens = batch['midi_tokens']
            tgt_padding_mask = batch['tgt_padding_mask']
            src_padding_mask = batch['src_padding_mask']

            optimizer.zero_grad()

            logits = model(
                src_waveforms=waveforms,
                tgt_midi_tokens=midi_tokens,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )

            loss = criterion(logits.reshape(-1, MIDI_VOCAB_SIZE), midi_tokens.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batch_count += 1

            progress_bar.set_postfix(loss=loss.item())

            if (i + 1) % CHUNK_PLAYBACK_INTERVAL == 0:
                print(f"\nEpoch {epoch+1}, Batch {i+1}: Generating and playing sample MIDI...")
                predicted_batch_tokens = torch.argmax(logits, dim=-1)
                first_predicted_tokens = predicted_batch_tokens[0].detach()
                play_midi_from_tokens(first_predicted_tokens, remi_tokenizer_for_setup, TEMP_MIDI_PLAYBACK_FILE)

        avg_train_loss = total_train_loss / train_batch_count if train_batch_count > 0 else 0
        print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader) if hasattr(val_loader, '__len__') else None, desc=f"Epoch {epoch+1} Validation")
        with torch.no_grad():
            for i, batch in val_progress_bar:
                if batch is None:
                    print(f"Warning: Skipping empty batch {i} in validation.")
                    continue

                waveforms = batch['waveforms']
                midi_tokens = batch['midi_tokens']
                tgt_padding_mask = batch['tgt_padding_mask']
                src_padding_mask = batch['src_padding_mask']

                logits = model(
                    src_waveforms=waveforms,
                    tgt_midi_tokens=midi_tokens,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
                )
                loss = criterion(logits.reshape(-1, MIDI_VOCAB_SIZE), midi_tokens.reshape(-1))
                total_val_loss += loss.item()
                val_batch_count += 1
                val_progress_bar.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else 0
        print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model checkpoint saved to {MODEL_SAVE_PATH}")

    print("Training complete.")
    print(f"Final model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()
