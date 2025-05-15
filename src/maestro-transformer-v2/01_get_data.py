# filepath: /Users/andreadellacorte/Documents/Workspace/GitHub/ml-institute-week-5-audio-transformers/src/maestro-transformer-v2/01_get_data.py

import pandas as pd
import json
import os
from pathlib import Path
import librosa
import pickle
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm  # For progress bar

from src.config import PROCESSED_DATA_DIR

# --- Configuration ---
DATASET_ID = "ddPn08/maestro-v3.0.0"
# Specific directory for this dataset's processed files
PROCESSED_DIR = PROCESSED_DATA_DIR / DATASET_ID.replace("/", "-")
MAESTRO_CSV_NAME = "maestro-v3.0.0.csv"
MAESTRO_JSON_NAME = "maestro-v3.0.0.json"

# Spectrogram parameters (can be adjusted based on your model's needs)
SR = 22050  # Sample rate to resample audio to
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for STFT
N_MELS = 128  # Number of Mel bands

def ensure_dir_exists(path: Path):
    """Ensures that a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)

def download_metadata_files():
    """
    Downloads MAESTRO metadata CSV and JSON from Hugging Face
    and saves them to the PROCESSED_DIR.
    Returns paths to the local CSV and JSON files.
    """
    print(f"Ensuring {PROCESSED_DIR} directory exists...")
    ensure_dir_exists(PROCESSED_DIR)

    # Define local paths for metadata files
    csv_local_path = PROCESSED_DIR / MAESTRO_CSV_NAME
    json_local_path = PROCESSED_DIR / MAESTRO_JSON_NAME

    # Download CSV
    if not csv_local_path.exists():
        print(f"Downloading {MAESTRO_CSV_NAME} from {DATASET_ID} to {csv_local_path}...")
        try:
            hf_hub_download(
                repo_id=DATASET_ID,
                filename=MAESTRO_CSV_NAME,
                repo_type="dataset",
                local_dir=PROCESSED_DIR,  # Download directly into this folder
                local_dir_use_symlinks=False  # Make sure it's a copy
            )
            print(f"{MAESTRO_CSV_NAME} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading {MAESTRO_CSV_NAME}: {e}")
            return None, None
    else:
        print(f"{MAESTRO_CSV_NAME} already exists at {csv_local_path}")

    # Download JSON
    if not json_local_path.exists():
        print(f"Downloading {MAESTRO_JSON_NAME} from {DATASET_ID} to {json_local_path}...")
        try:
            hf_hub_download(
                repo_id=DATASET_ID,
                filename=MAESTRO_JSON_NAME,
                repo_type="dataset",
                local_dir=PROCESSED_DIR,  # Download directly into this folder
                local_dir_use_symlinks=False  # Make sure it's a copy
            )
            print(f"{MAESTRO_JSON_NAME} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading {MAESTRO_JSON_NAME}: {e}")
            # If CSV downloaded but JSON failed, we might still proceed with CSV
            return csv_local_path, None
    else:
        print(f"{MAESTRO_JSON_NAME} already exists at {json_local_path}")

    return csv_local_path, json_local_path

def process_songs(maestro_df: pd.DataFrame):
    """
    Processes each song in the MAESTRO dataframe:
    - Downloads audio and MIDI if not already processed.
    - Computes spectrogram from audio.
    - Saves spectrogram and MIDI data as .pkl files.
    """
    if maestro_df is None:
        print("MAESTRO DataFrame is None. Cannot process songs.")
        return

    print(f"\nProcessing {len(maestro_df)} songs from the dataset...")

    for index, row in tqdm(maestro_df.iterrows(), total=maestro_df.shape[0], desc="Processing songs"):
        # These are relative paths on the Hugging Face Hub dataset
        audio_filename_on_hub = row['audio_filename']
        midi_filename_on_hub = row['midi_filename']

        # Create a unique ID from the audio filename (without extension and directory)
        song_id = index

        spectrogram_filename = f"{song_id}_spectrogram.pkl"
        midi_filename = f"{song_id}.midi"

        # Output paths for the processed files
        spectrogram_output_path = PROCESSED_DIR / spectrogram_filename
        midi_output_path = PROCESSED_DIR / midi_filename

        # Check if both processed files already exist
        if spectrogram_output_path.exists() and midi_output_path.exists():
            continue  # Skip if already processed

        # --- Process Audio (Download, Load, Compute Spectrogram, Save) ---
        if not spectrogram_output_path.exists():
            try:
                # Download audio file
                cached_audio_path_str = hf_hub_download(
                    repo_id=DATASET_ID,
                    filename=audio_filename_on_hub,
                    repo_type="dataset",
                )

                # Load audio with librosa
                y, sr_orig = librosa.load(cached_audio_path_str, sr=None)  # Load with original sample rate

                # Resample if necessary to the target sample rate SR
                if sr_orig != SR:
                    y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)

                # Calculate Mel spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
                S_db = librosa.power_to_db(S, ref=np.max)  # Convert to decibels

                # Save spectrogram
                with open(spectrogram_output_path, 'wb') as f_spec:
                    pickle.dump(S_db, f_spec)

            except Exception as e:
                print(f"  Error processing audio for {song_id} ({audio_filename_on_hub}): {e}")

        # --- Process MIDI (Download, Save as .midi) ---
        if not midi_output_path.exists():
            try:
                # Download MIDI file
                cached_midi_path_str = hf_hub_download(
                    repo_id=DATASET_ID,
                    filename=midi_filename_on_hub,
                    repo_type="dataset",
                )

                # Read MIDI data as bytes
                with open(cached_midi_path_str, 'rb') as f_midi_in:
                    midi_data_bytes = f_midi_in.read()  # Read raw bytes of the MIDI file

                # Save MIDI data as .midi file directly
                with open(midi_output_path, 'wb') as f_midi_out:
                    f_midi_out.write(midi_data_bytes)  # Write raw bytes directly, no pickle

            except Exception as e:
                print(f"  Error processing MIDI for {song_id} ({midi_filename_on_hub}): {e}")

def main():
    print("--- MAESTRO Dataset Processing Script ---")

    # Ensure the base output directory exists
    ensure_dir_exists(PROCESSED_DATA_DIR)

    # 1. Download metadata files (CSV and JSON)
    local_csv_path, local_json_path = download_metadata_files()

    if not local_csv_path or not local_csv_path.exists():
        print("MAESTRO CSV metadata could not be downloaded or found. Exiting.")
        return

    # 2. Load the MAESTRO CSV into a pandas DataFrame
    maestro_df = None
    try:
        print(f"Loading MAESTRO song data from {local_csv_path}...")
        maestro_df = pd.read_csv(local_csv_path)
        print(f"Loaded {len(maestro_df)} song entries from the MAESTRO CSV.")
    except Exception as e:
        print(f"Error loading MAESTRO CSV ({local_csv_path}): {e}")
        return

    # 3. Process each song (download audio/MIDI, compute spectrogram, save)
    process_songs(maestro_df)

    print("\n--- MAESTRO Dataset Processing Complete ---")

if __name__ == "__main__":
    main()