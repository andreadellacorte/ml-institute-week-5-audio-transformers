# filepath: /Users/andreadellacorte/Documents/Workspace/GitHub/ml-institute-week-5-audio-transformers/src/maestro-transformer-v2/01_get_data.py

import pandas as pd
import json
import os
import sys
import random
from pathlib import Path
import librosa
import pickle
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm  # For progress bar
import miditoolkit  # For MIDI processing
from miditok import REMI, TokenizerConfig  # For MIDI tokenization

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

# Chunking parameters
CHUNK_SIZE_SECONDS = 4  # Number of seconds per chunk
# Minimum duration for the last chunk (to avoid very short chunks)
MIN_CHUNK_DURATION_SECONDS = CHUNK_SIZE_SECONDS / 1

# Processing parameters
MAX_SONGS_TO_PROCESS = 100  # Maximum number of songs to process
TRAIN_SPLIT_PERCENTAGE = 80  # Percentage of songs to use for training (the rest for evaluation)

# Seed for reproducibility
RANDOM_SEED = 42

def ensure_dir_exists(path: Path):
    """Ensures that a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)

def extract_midi_chunk(midi_data, ticks_per_beat, start_tick, end_tick):
    """
    Extract a chunk of MIDI data between start_tick and end_tick.
    
    Args:
        midi_data: A miditoolkit.MidiFile object
        ticks_per_beat: Ticks per beat of the MIDI file
        start_tick: Start tick for the chunk
        end_tick: End tick for the chunk
        
    Returns:
        A new miditoolkit.MidiFile object containing only the specified chunk
    """
    
    # Create a new MIDI object with the same ticks_per_beat
    new_midi = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
    
    # Copy tempo changes within the chunk
    for tempo_change in midi_data.tempo_changes:
        if start_tick <= tempo_change.time < end_tick:
            # Adjust time to be relative to the new chunk's start
            new_time = tempo_change.time - start_tick
            tc = miditoolkit.TempoChange(tempo=tempo_change.tempo, time=new_time)
            new_midi.tempo_changes.append(tc)
    
    # Copy time signature changes within the chunk
    for ts_change in midi_data.time_signature_changes:
        if start_tick <= ts_change.time < end_tick:
            # Adjust time to be relative to the new chunk's start
            new_time = ts_change.time - start_tick
            tsc = miditoolkit.TimeSignature(
                numerator=ts_change.numerator, 
                denominator=ts_change.denominator, 
                time=new_time
            )
            new_midi.time_signature_changes.append(tsc)
    
    # Copy key signature changes within the chunk
    for ks_change in midi_data.key_signature_changes:
        if start_tick <= ks_change.time < end_tick:
            # Adjust time to be relative to the new chunk's start
            new_time = ks_change.time - start_tick
            ksc = miditoolkit.KeySignature(
                key_number=ks_change.key_number, 
                time=new_time
            )
            new_midi.key_signature_changes.append(ksc)
    
    # Process each track
    for track in midi_data.instruments:
        new_track = miditoolkit.Instrument(program=track.program, is_drum=track.is_drum, name=track.name)
        
        # Only include notes within the chunk
        for note in track.notes:
            # Skip notes that end before our chunk starts or start after our chunk ends
            if note.end <= start_tick or note.start >= end_tick:
                continue
            
            # Create a new note with adjusted timings
            new_start = max(0, note.start - start_tick)
            new_end = min(end_tick - start_tick, note.end - start_tick)
            
            new_note = miditoolkit.Note(
                pitch=note.pitch,
                velocity=note.velocity,
                start=new_start,
                end=new_end
            )
            new_track.notes.append(new_note)
        
        # Only add tracks that have notes
        if new_track.notes:
            new_midi.instruments.append(new_track)
    
    # If no tempo changes were copied, add default tempo
    if not new_midi.tempo_changes and midi_data.tempo_changes:
        tempo = miditoolkit.TempoChange(tempo=midi_data.tempo_changes[0].tempo, time=0)
        new_midi.tempo_changes.append(tempo)
    
    return new_midi

def split_midi_into_chunks(midi_data, chunk_size_seconds):
    """
    Split a MIDI file into chunks of a specified duration.
    
    Args:
        midi_data: A miditoolkit.MidiFile object
        chunk_size_seconds: Size of each chunk in seconds
        
    Returns:
        List of (chunk_number, miditoolkit.MidiFile) tuples
    """
    
    # Get ticks per beat and tempo
    ticks_per_beat = midi_data.ticks_per_beat
    
    if not midi_data.tempo_changes:
        print(f"Warning: No tempo changes found in MIDI. Using default tempo of 120 BPM")
        # Default tempo: 120 BPM (500000 microseconds per beat)
        tempo = 500000
        is_bpm = False
    else:
        tempo = midi_data.tempo_changes[0].tempo
        # Check if tempo is in BPM (most likely under 1000) or microseconds per beat (much larger)
        is_bpm = tempo < 1000
        print(f"DEBUG: MIDI tempo value: {tempo}, interpreted as {'BPM' if is_bpm else 'microseconds per beat'}")
    
    # Calculate ticks per second
    if is_bpm:
        # If tempo is in BPM, convert to microseconds per beat
        microseconds_per_beat = 60_000_000 / tempo
    else:
        # Already in microseconds per beat format
        microseconds_per_beat = tempo
        
    beats_per_second = 1_000_000 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second
    
    print(f"DEBUG: Ticks per beat: {ticks_per_beat}, Beats per sec: {beats_per_second:.2f}, " +
          f"Ticks per sec: {ticks_per_second:.2f}")
    
    # Find the end time (max tick among all notes)
    max_tick = 0
    for track in midi_data.instruments:
        for note in track.notes:
            max_tick = max(max_tick, note.end)
    
    # Calculate total duration in seconds
    total_duration_seconds = max_tick / ticks_per_second
    
    # Calculate number of complete chunks
    num_complete_chunks = int(total_duration_seconds / chunk_size_seconds)
    
    # Check if we have a remaining partial chunk that's at least half of chunk_size_seconds
    remaining_seconds = total_duration_seconds % chunk_size_seconds
    include_last_chunk = remaining_seconds >= MIN_CHUNK_DURATION_SECONDS
    
    # Calculate total chunks
    total_chunks = num_complete_chunks + (1 if include_last_chunk else 0)
    
    chunks = []
    
    # Create each chunk
    for chunk_num in range(total_chunks):
        chunk_start_tick = int(chunk_num * chunk_size_seconds * ticks_per_second)
        
        # For the last chunk, use the actual end tick if it's a partial chunk
        if chunk_num == num_complete_chunks and include_last_chunk:
            chunk_end_tick = max_tick
        else:
            chunk_end_tick = int((chunk_num + 1) * chunk_size_seconds * ticks_per_second)
        
        # Extract the chunk
        chunk_midi = extract_midi_chunk(midi_data, ticks_per_beat, chunk_start_tick, chunk_end_tick)
        
        # Add to list if the chunk has any notes
        has_notes = any(len(track.notes) > 0 for track in chunk_midi.instruments)
        if has_notes:
            chunks.append((chunk_num + 1, chunk_midi))
    
    return chunks

def split_audio_into_chunks(audio_data, sr, chunk_size_seconds):
    """
    Split an audio file into chunks of a specified duration.
    
    Args:
        audio_data: Audio data array
        sr: Sample rate
        chunk_size_seconds: Size of each chunk in seconds
        
    Returns:
        List of (chunk_number, audio_data) tuples
    """
    
    # Calculate chunk size in samples
    chunk_size_samples = int(chunk_size_seconds * sr)
    
    # Calculate total number of complete chunks
    num_complete_chunks = int(len(audio_data) / chunk_size_samples)
    
    # Check if we have a remaining partial chunk that's at least half of chunk_size_seconds
    remaining_samples = len(audio_data) % chunk_size_samples
    include_last_chunk = remaining_samples >= (chunk_size_samples // 2)
    
    # Calculate total chunks
    total_chunks = num_complete_chunks + (1 if include_last_chunk else 0)
    
    chunks = []
    
    # Create each chunk
    for chunk_num in range(total_chunks):
        start_idx = chunk_num * chunk_size_samples
        
        # For the last chunk, use the actual end if it's a partial chunk
        if chunk_num == num_complete_chunks and include_last_chunk:
            end_idx = len(audio_data)
        else:
            end_idx = (chunk_num + 1) * chunk_size_samples
        
        # Extract the chunk
        chunk_audio = audio_data[start_idx:end_idx]
        
        # Add to list if the chunk has enough audio data
        if len(chunk_audio) >= (chunk_size_samples // 2):  # At least half of the expected samples
            chunks.append((chunk_num + 1, chunk_audio))
    
    return chunks

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

def create_dataset_split(song_ids, train_percent=TRAIN_SPLIT_PERCENTAGE):
    """
    Splits a list of song IDs into training and evaluation sets based on the given percentage.
    
    Args:
        song_ids: List of song IDs to split
        train_percent: Percentage of songs to allocate to the training set (default: TRAIN_SPLIT_PERCENTAGE)
        
    Returns:
        (train_ids, eval_ids): Tuple of lists containing the IDs for training and evaluation
    """
    # Ensure reproducibility
    random.seed(RANDOM_SEED)
    
    # Shuffle the song IDs
    shuffled_ids = song_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Calculate the split point
    train_count = max(1, int(len(shuffled_ids) * train_percent / 100))
    
    # Split the shuffled IDs
    train_ids = shuffled_ids[:train_count]
    eval_ids = shuffled_ids[train_count:]
    
    return train_ids, eval_ids

def setup_split_directories():
    """
    Creates train and evaluation directories within the processed directory.
    
    Returns:
        (train_dir, eval_dir): Tuple of paths for train and evaluation directories
    """
    train_dir = PROCESSED_DIR / "train"
    eval_dir = PROCESSED_DIR / "eval"
    
    # Create directories if they don't exist
    ensure_dir_exists(train_dir)
    ensure_dir_exists(eval_dir)
    
    return train_dir, eval_dir

def tokenize_midi(midi_data):
    """
    Tokenize MIDI data using miditok.REMI tokenizer.

    Args:
        midi_data: A miditoolkit.MidiFile object

    Returns:
        List of tokens
    """
    # Create a TokenizerConfig with the desired settings
    config = TokenizerConfig(
        num_velocities=32, use_chords=True, use_programs=False, use_sustain_pedals=True
    )

    # Initialize the REMI tokenizer with the configuration
    tokenizer = REMI(config)

    # Tokenize MIDI data
    try:
        # Check if MIDI data is valid
        if not midi_data or not hasattr(midi_data, 'instruments') or not midi_data.instruments:
            print(f"Warning: Invalid MIDI data (no instruments)")
            return None
            
        # Check if there are any notes
        note_count = sum(len(track.notes) for track in midi_data.instruments)
        if note_count == 0:
            print(f"Warning: MIDI data has no notes")
            return None
            
        # Tokenize
        tokens = tokenizer(midi_data)
        
        # Verify tokens were created
        if not tokens or len(tokens) == 0:
            print(f"Warning: Tokenization produced no tokens")
            return None
            
        return tokens
    except Exception as e:
        print(f"Error tokenizing MIDI: {e}")
        return None

def process_songs(maestro_df: pd.DataFrame, max_songs=MAX_SONGS_TO_PROCESS, split_pct=TRAIN_SPLIT_PERCENTAGE):
    """
    Processes each song in the MAESTRO dataframe:
    - Downloads audio and MIDI if not already processed.
    - Splits audio and MIDI into chunks of CHUNK_SIZE_SECONDS.
    - Computes spectrogram from each audio chunk.
    - Saves spectrogram and MIDI data for each chunk.
    - Separates files into train and evaluation splits.
    
    Args:
        maestro_df: DataFrame containing MAESTRO metadata
        max_songs: Maximum number of songs to process (default: MAX_SONGS_TO_PROCESS)
        split_pct: Percentage for train/eval split (default: TRAIN_SPLIT_PERCENTAGE)
    """
    if maestro_df is None:
        print("MAESTRO DataFrame is None. Cannot process songs.")
        return

    # Limit to maximum number of songs
    if max_songs and max_songs < len(maestro_df):
        print(f"\nLimiting processing to {max_songs} songs from the dataset...")
        maestro_df = maestro_df.head(max_songs)
    else:
        print(f"\nProcessing all {len(maestro_df)} songs from the dataset...")
        
    print(f"Each song will be split into chunks of {CHUNK_SIZE_SECONDS} seconds")
    
    # Get song IDs and create train/eval split
    song_ids = maestro_df['index'].unique().tolist()
    train_song_ids, eval_song_ids = create_dataset_split(song_ids, split_pct)
    
    print(f"Split songs into {len(train_song_ids)} for training ({split_pct}%) " +
          f"and {len(eval_song_ids)} for evaluation ({100-split_pct}%)")
    
    # Set up train and eval directories
    train_dir, eval_dir = setup_split_directories()
    
    # Create a cache directory for temporary files
    cache_dir = PROCESSED_DIR / ".cache"
    ensure_dir_exists(cache_dir)
    
    # Process each song in the DataFrame
    for _, row in tqdm(maestro_df.iterrows(), total=len(maestro_df), desc="Processing songs"):
        # Use the original index from the CSV as the song_id for consistent naming
        song_id = row['index']
        # These are relative paths on the Hugging Face Hub dataset
        audio_filename_on_hub = row['audio_filename']
        midi_filename_on_hub = row['midi_filename']

        # Check if we need to process this song
        existing_midi_files = list(PROCESSED_DIR.glob(f"{song_id}_*.midi"))
        existing_tokenized_files = list(PROCESSED_DIR.glob(f"{song_id}_*_midi_tokenised.pkl"))
        
        # If we already have MIDI files and we've tokenized all of them
        if existing_midi_files and len(existing_midi_files) == len(existing_tokenized_files):
            print(f"  Song {song_id} already has all chunks processed and tokenized. Skipping.")
            continue
            
        # If we have MIDI chunks but need to tokenize them
        if existing_midi_files and len(existing_midi_files) > len(existing_tokenized_files):
            print(f"  Song {song_id} has {len(existing_midi_files)} chunks but only {len(existing_tokenized_files)} tokenized. Tokenizing remaining chunks.")
            
            # For each existing MIDI file that doesn't have a corresponding tokenized file
            for midi_file in existing_midi_files:
                # Extract chunk number from filename (format: song_id_chunk_number.midi)
                chunk_num = int(midi_file.stem.split('_')[1])
                
                # Check if tokenized file already exists
                chunk_midi_tokenized_filename = f"{song_id}_{chunk_num}_midi_tokenised.pkl"
                chunk_midi_tokenized_path = PROCESSED_DIR / chunk_midi_tokenized_filename
                
                if not chunk_midi_tokenized_path.exists():
                    try:
                        # Load the MIDI file
                        midi_data = miditoolkit.MidiFile(str(midi_file))
                        
                        # Tokenize MIDI data
                        midi_tokens = tokenize_midi(midi_data)
                        
                        if midi_tokens is not None:
                            # Save tokens as pickle file
                            with open(chunk_midi_tokenized_path, 'wb') as f_tokens:
                                pickle.dump(midi_tokens, f_tokens)
                            print(f"    Tokenized {midi_file.name} -> {chunk_midi_tokenized_path.name}")
                        else:
                            print(f"    Failed to tokenize {midi_file.name}")
                    except Exception as e:
                        print(f"    Error tokenizing {midi_file.name}: {e}")
            
            # Skip the rest of the processing for this song since we only needed to tokenize
            continue
            
        try:
            # --- Download and process audio ---
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
            
            # --- Download and process MIDI ---
            cached_midi_path_str = hf_hub_download(
                repo_id=DATASET_ID,
                filename=midi_filename_on_hub,
                repo_type="dataset",
            )
            
            # Load MIDI data as a miditoolkit object
            midi_data = miditoolkit.MidiFile(cached_midi_path_str)
            
            # Debug MIDI file contents
            note_count = sum(len(track.notes) for track in midi_data.instruments)
            max_tick = 0
            if midi_data.instruments:
                for track in midi_data.instruments:
                    for note in track.notes:
                        max_tick = max(max_tick, note.end)
            
            # Get tempo information
            if midi_data.tempo_changes:
                tempo = midi_data.tempo_changes[0].tempo
                print(f"  MIDI file has {len(midi_data.instruments)} tracks, {note_count} notes, " +
                     f"max tick: {max_tick}, tempo: {tempo}")
            else:
                print(f"  MIDI file has {len(midi_data.instruments)} tracks, {note_count} notes, " +
                     f"max tick: {max_tick}, no tempo information")
            
            # Split audio and MIDI into chunks
            audio_chunks = split_audio_into_chunks(y, SR, CHUNK_SIZE_SECONDS)
            midi_chunks = split_midi_into_chunks(midi_data, CHUNK_SIZE_SECONDS)
            
            # Process and save each chunk
            print(f"  Processing {len(audio_chunks)} audio chunks and {len(midi_chunks)} MIDI chunks for song {song_id}")
            
            # We'll only process chunks that exist in both audio and MIDI
            # Use set intersection to find common chunk numbers
            audio_chunk_numbers = {a[0] for a in audio_chunks}
            midi_chunk_numbers = {m[0] for m in midi_chunks}
            common_chunk_numbers = audio_chunk_numbers.intersection(midi_chunk_numbers)
            
            # Create a dictionary for easy lookup
            audio_chunks_dict = {a[0]: a[1] for a in audio_chunks}
            midi_chunks_dict = {m[0]: m[1] for m in midi_chunks}
            
            # Determine which split this song belongs to
            output_dir = train_dir if song_id in train_song_ids else eval_dir
            
            for chunk_num in sorted(common_chunk_numbers):
                # Generate filenames for this chunk
                chunk_spectrogram_filename = f"{song_id}_{chunk_num}_spectrogram.pkl"
                chunk_midi_filename = f"{song_id}_{chunk_num}.midi"
                chunk_midi_tokenized_filename = f"{song_id}_{chunk_num}_midi_tokenised.pkl"
                
                # Output paths
                chunk_spectrogram_path = output_dir / chunk_spectrogram_filename
                chunk_midi_path = output_dir / chunk_midi_filename
                chunk_midi_tokenized_path = output_dir / chunk_midi_tokenized_filename
                
                # Skip if all files already exist
                if chunk_spectrogram_path.exists() and chunk_midi_path.exists() and chunk_midi_tokenized_path.exists():
                    continue
                
                # Process and save audio spectrogram
                if not chunk_spectrogram_path.exists():
                    try:
                        # Get audio chunk
                        chunk_audio = audio_chunks_dict[chunk_num]
                        
                        # Calculate Mel spectrogram
                        S = librosa.feature.melspectrogram(y=chunk_audio, sr=SR, n_fft=N_FFT, 
                                                        hop_length=HOP_LENGTH, n_mels=N_MELS)
                        S_db = librosa.power_to_db(S, ref=np.max)  # Convert to decibels
                        
                        # Save spectrogram
                        with open(chunk_spectrogram_path, 'wb') as f_spec:
                            pickle.dump(S_db, f_spec)
                            
                    except Exception as e:
                        print(f"  Error processing audio chunk {chunk_num} for song {song_id}: {e}")
                
                # Save MIDI chunk
                if not chunk_midi_path.exists():
                    try:
                        # Get MIDI chunk
                        chunk_midi = midi_chunks_dict[chunk_num]
                        
                        # Save MIDI as .midi file
                        chunk_midi.dump(chunk_midi_path)
                        
                    except Exception as e:
                        print(f"  Error processing MIDI chunk {chunk_num} for song {song_id}: {e}")
                
                # Create tokenized MIDI file
                if not chunk_midi_tokenized_path.exists():
                    try:
                        # Get MIDI chunk
                        chunk_midi = midi_chunks_dict[chunk_num]
                        
                        # Tokenize MIDI data
                        midi_tokens = tokenize_midi(chunk_midi)
                        
                        if midi_tokens is not None:
                            # Save tokens as pickle file
                            with open(chunk_midi_tokenized_path, 'wb') as f_tokens:
                                pickle.dump(midi_tokens, f_tokens)
                        else:
                            print(f"  Failed to tokenize MIDI chunk {chunk_num} for song {song_id}")
                            
                    except Exception as e:
                        print(f"  Error tokenizing MIDI chunk {chunk_num} for song {song_id}: {e}")
            
        except Exception as e:
            print(f"  Error processing song {song_id}: {e}")

def main():
    print("--- MAESTRO Dataset Processing Script ---")
    print(f"Chunk size: {CHUNK_SIZE_SECONDS} seconds")
    print(f"Minimum chunk duration: {MIN_CHUNK_DURATION_SECONDS} seconds")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

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
        
        # Shuffle the DataFrame but preserve the original index for file naming
        maestro_df = maestro_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=False)
        print(f"Shuffled song order using random seed {RANDOM_SEED}")
    except Exception as e:
        print(f"Error loading MAESTRO CSV ({local_csv_path}): {e}")
        return

    # 3. Process each song (download audio/MIDI, compute spectrogram, save)
    # Process songs with our specified parameters
    process_songs(maestro_df, max_songs=MAX_SONGS_TO_PROCESS, split_pct=TRAIN_SPLIT_PERCENTAGE)
    
    # Set up directories
    train_dir, eval_dir = setup_split_directories()
    
    # Count the generated chunks in both train and eval directories
    train_midi_chunks = list(train_dir.glob("*_*.midi"))
    train_spectrogram_chunks = list(train_dir.glob("*_*_spectrogram.pkl"))
    train_tokenized_midi_chunks = list(train_dir.glob("*_*_midi_tokenised.pkl"))
    
    eval_midi_chunks = list(eval_dir.glob("*_*.midi"))
    eval_spectrogram_chunks = list(eval_dir.glob("*_*_spectrogram.pkl"))
    eval_tokenized_midi_chunks = list(eval_dir.glob("*_*_midi_tokenised.pkl"))
    
    # Count total files
    midi_chunks = train_midi_chunks + eval_midi_chunks
    spectrogram_chunks = train_spectrogram_chunks + eval_spectrogram_chunks
    tokenized_midi_chunks = train_tokenized_midi_chunks + eval_tokenized_midi_chunks
    
    print(f"\nGenerated:")
    print(f"  - {len(midi_chunks)} total MIDI chunks")
    print(f"  - {len(spectrogram_chunks)} total spectrogram chunks")
    print(f"  - {len(tokenized_midi_chunks)} total tokenized MIDI chunks")
    
    # Print breakdown by split
    print(f"\nTraining set ({TRAIN_SPLIT_PERCENTAGE}%):")
    print(f"  - {len(train_midi_chunks)} MIDI chunks")
    print(f"  - {len(train_spectrogram_chunks)} spectrogram chunks")
    print(f"  - {len(train_tokenized_midi_chunks)} tokenized MIDI chunks")
    
    print(f"\nEvaluation set ({100-TRAIN_SPLIT_PERCENTAGE}%):")
    print(f"  - {len(eval_midi_chunks)} MIDI chunks")
    print(f"  - {len(eval_spectrogram_chunks)} spectrogram chunks")
    print(f"  - {len(eval_tokenized_midi_chunks)} tokenized MIDI chunks")
    
    # Show some examples of the generated files
    if midi_chunks:
        print("\nExample generated files:")
        
        # Training examples
        if train_midi_chunks:
            print("\nTraining files:")
            for file_type, files in [
                ("MIDI", sorted(train_midi_chunks)[:2]),
                ("Spectrogram", sorted(train_spectrogram_chunks)[:2]),
                ("Tokenized MIDI", sorted(train_tokenized_midi_chunks)[:2])
            ]:
                print(f"\n{file_type} files:")
                for file in files:
                    print(f"  {file.name}")
                if len(files) > 2:
                    print(f"  ...and more")
                    
        # Evaluation examples
        if eval_midi_chunks:
            print("\nEvaluation files:")
            for file_type, files in [
                ("MIDI", sorted(eval_midi_chunks)[:2]),
                ("Spectrogram", sorted(eval_spectrogram_chunks)[:2]),
                ("Tokenized MIDI", sorted(eval_tokenized_midi_chunks)[:2])
            ]:
                print(f"\n{file_type} files:")
                for file in files:
                    print(f"  {file.name}")
                if len(files) > 2:
                    print(f"  ...and more")

    print("\n--- MAESTRO Dataset Processing Complete ---")

if __name__ == "__main__":
    main()