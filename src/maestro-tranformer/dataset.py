import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, Audio
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
import os
from miditok import REMI, TokenizerConfig
from symusic import Score

class MaestroDataset(IterableDataset):
    """
    A PyTorch IterableDataset for the MAESTRO dataset.
    It iterates over a metadata CSV, loads specified audio files directly
    from Hugging Face, downloads corresponding MIDI files, and processes them.
    Can yield full sequences or fixed-size chunks.
    """

    def __init__(self, split="train", sample_rate=16000,
                 dataset_repo_id="ddPn08/maestro-v3.0.0",
                 metadata_filename="maestro-v3.0.0.csv",
                 chunk_duration_sec: float = None,  # Duration of audio chunks in seconds
                 max_midi_tokens_per_chunk: int = None  # Max MIDI tokens for a chunk
                 ):
        """
        Initialize the MAESTRO dataset.

        Args:
            split (str): Dataset split to use ("train", "validation", or "test").
                         Filters the metadata CSV by this split.
            sample_rate (int): Target sample rate for audio.
            dataset_repo_id (str): Hugging Face dataset repository ID.
            metadata_filename (str): Name of the metadata CSV file in the dataset repository.
            chunk_duration_sec (float): Duration of audio chunks in seconds.
            max_midi_tokens_per_chunk (int): Maximum number of MIDI tokens per chunk.
        """
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.dataset_repo_id = dataset_repo_id
        self.metadata_filename = metadata_filename
        self.metadata_df = pd.DataFrame()

        self.chunk_duration_sec = chunk_duration_sec
        self.max_midi_tokens_per_chunk = max_midi_tokens_per_chunk
        if self.chunk_duration_sec is not None:
            self.chunk_samples = int(self.chunk_duration_sec * self.sample_rate)
            if self.max_midi_tokens_per_chunk is None:
                # Provide a default if chunking audio but not specifying max MIDI tokens
                print("Warning: chunk_duration_sec is set, but max_midi_tokens_per_chunk is not. Defaulting to 512.")
                self.max_midi_tokens_per_chunk = 512
        else:
            self.chunk_samples = None
            if self.max_midi_tokens_per_chunk is not None:
                print("Warning: max_midi_tokens_per_chunk is set, but chunk_duration_sec is not. max_midi_tokens_per_chunk will be ignored.")

        # Configure and initialize the REMI tokenizer
        tokenizer_config = TokenizerConfig(
            num_velocities=16, 
            use_chords=True, 
            use_programs=False,  # MAESTRO is piano, program changes not essential
            use_sustain_pedals=True, # Sustain pedals are important for piano
        )
        self.tokenizer = REMI(tokenizer_config)

        try:
            metadata_path = hf_hub_download(
                repo_id=self.dataset_repo_id,
                filename=self.metadata_filename,
                repo_type="dataset"
            )
            full_metadata_df = pd.read_csv(metadata_path)

            if 'split' in full_metadata_df.columns:
                self.metadata_df = full_metadata_df[full_metadata_df['split'] == self.split].reset_index(drop=True)
                if self.metadata_df.empty:
                    print(f"Warning: No entries found for split '{self.split}' in {self.metadata_filename}.")
            else:
                print(f"Warning: 'split' column not found in {self.metadata_filename}. Using all entries.")
                self.metadata_df = full_metadata_df

            if 'audio_filename' not in self.metadata_df.columns or 'midi_filename' not in self.metadata_df.columns:
                print(f"Warning: 'audio_filename' or 'midi_filename' column not found in the filtered metadata. Processing might fail.")
                self.metadata_df = pd.DataFrame() # Ensure it's empty

        except Exception as e:
            print(f"Error downloading or parsing metadata CSV {self.metadata_filename}: {e}")
            # self.metadata_df remains an empty DataFrame

    def __iter__(self):
        if self.metadata_df.empty:
            print("Metadata is empty, cannot iterate.")
            return iter([]) # Return an empty iterator

        for index, row in self.metadata_df.iterrows():
            csv_audio_path_in_repo = row.get('audio_filename')
            csv_midi_path_in_repo = row.get('midi_filename') # Get MIDI filename from CSV

            if not csv_audio_path_in_repo or not isinstance(csv_audio_path_in_repo, str):
                print(f"Skipping row {index} due to missing or invalid 'audio_filename': {csv_audio_path_in_repo}")
                continue
            
            if not csv_midi_path_in_repo or not isinstance(csv_midi_path_in_repo, str):
                print(f"Skipping row {index} due to missing or invalid 'midi_filename': {csv_midi_path_in_repo}")
                continue

            # Construct the direct Hugging Face URL for this audio file
            clean_csv_audio_path = csv_audio_path_in_repo.lstrip('/')
            hf_audio_url = f"https://huggingface.co/datasets/{self.dataset_repo_id}/resolve/main/{clean_csv_audio_path}"

            downloaded_midi_path = None
            try:
                clean_csv_midi_path = csv_midi_path_in_repo.lstrip('/')
                downloaded_midi_path = hf_hub_download(
                    repo_id=self.dataset_repo_id,
                    filename=clean_csv_midi_path,
                    repo_type="dataset",
                    # cache_dir can be specified if needed, e.g., "./midi_cache"
                )
            except Exception as e_midi:
                print(f"Error downloading MIDI file {csv_midi_path_in_repo} for row {index}: {e_midi}")
                # Decide whether to skip the item or proceed without MIDI
                continue # For now, skip if MIDI download fails as it's the label

            try:
                # Load the single audio file using 'audiofolder'
                single_file_ds = load_dataset(
                    "audiofolder",
                    data_files={"item": [hf_audio_url]},
                    streaming=True,
                    # trust_remote_code=True # May be needed depending on HF datasets version and config
                )
                processed_ds = single_file_ds.cast_column(
                    "audio",
                    Audio(sampling_rate=self.sample_rate, mono=True)
                )
                audio_stream_item = next(iter(processed_ds['item']))
                audio_array = audio_stream_item['audio']['array']
                loaded_audio_path = audio_stream_item['audio']['path']

                if os.path.basename(loaded_audio_path) != os.path.basename(csv_audio_path_in_repo):
                    print(f"Warning: Basename mismatch! CSV: {os.path.basename(csv_audio_path_in_repo)}, Loaded: {os.path.basename(loaded_audio_path)}")

                waveform = torch.tensor(audio_array, dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0) # Ensure [1, num_samples]

                full_midi_score = None
                if downloaded_midi_path and os.path.exists(downloaded_midi_path):
                    try:
                        full_midi_score = Score(downloaded_midi_path)
                    except Exception as e_score:
                        print(f"Error loading MIDI Score {downloaded_midi_path} for row {index}: {e_score}")
                        continue
                else:
                    print(f"Skipping row {index} as MIDI path is invalid or file doesn't exist.")
                    continue
                
                # If not chunking, process as before
                if self.chunk_samples is None:
                    try:
                        midi_tokens = self.tokenizer(full_midi_score)
                        if not midi_tokens or not midi_tokens[0]: # Ensure tokens were produced
                            print(f"Skipping row {index} due to empty tokenization for {downloaded_midi_path}")
                            continue
                    except Exception as e_tok:
                        print(f"Error tokenizing full MIDI {downloaded_midi_path} for row {index}: {e_tok}")
                        continue
                    
                    output_item = {
                        "waveform": waveform,
                        "sample_rate": self.sample_rate,
                        "source_audio_url": loaded_audio_path,
                        "downloaded_midi_path": downloaded_midi_path,
                        "midi_tokens": midi_tokens
                    }
                    csv_metadata = row.to_dict()
                    for key, value in csv_metadata.items():
                        if key not in output_item:
                            output_item[key] = value
                        elif key == 'midi_filename' and downloaded_midi_path:
                            output_item['original_midi_filename_from_csv'] = value
                    yield output_item

                else: # Implement chunking
                    num_total_samples = waveform.shape[1]
                    current_sample_offset = 0
                    
                    while current_sample_offset < num_total_samples:
                        start_sample = current_sample_offset
                        end_sample = min(current_sample_offset + self.chunk_samples, num_total_samples)
                        
                        # Ensure audio_chunk is not too short (e.g., less than 1 sec)
                        min_chunk_samples = self.sample_rate // 2 # 0.5 second
                        if (end_sample - start_sample) < min_chunk_samples and start_sample > 0 : # Avoid tiny trailing chunks unless it's the only chunk
                             current_sample_offset = end_sample # Effectively skip tiny trailing chunk
                             continue

                        audio_chunk = waveform[:, start_sample:end_sample]
                        
                        start_time_sec = start_sample / self.sample_rate
                        end_time_sec = end_sample / self.sample_rate
                        
                        # Slice the MIDI score for the current chunk
                        # clip method arguments are (start, end, unit='s', clip_notes=True)
                        try:
                            chunk_midi_score = full_midi_score.clip(start_time_sec, end_time_sec, unit='s', clip_notes=True)
                            # Tokenize the sliced MIDI score
                            chunk_midi_tokens = self.tokenizer(chunk_midi_score)
                        except Exception as e_midi_chunk:
                            print(f"Error slicing/tokenizing MIDI chunk for {downloaded_midi_path} ({start_time_sec:.2f}s-{end_time_sec:.2f}s): {e_midi_chunk}")
                            current_sample_offset = end_sample
                            continue

                        # Validate MIDI tokens
                        if not chunk_midi_tokens or not chunk_midi_tokens[0]: # No MIDI events in this chunk
                            current_sample_offset = end_sample
                            continue # Skip chunks with no MIDI

                        if len(chunk_midi_tokens[0]) > self.max_midi_tokens_per_chunk:
                            current_sample_offset = end_sample
                            continue # Skipping for now

                        output_item = {
                            "waveform": audio_chunk,
                            "sample_rate": self.sample_rate,
                            "source_audio_url": loaded_audio_path, # This refers to the original full audio
                            "downloaded_midi_path": downloaded_midi_path, # Original full MIDI
                            "midi_tokens": chunk_midi_tokens,
                            "chunk_start_sec": start_time_sec,
                            "chunk_end_sec": end_time_sec
                        }
                        csv_metadata = row.to_dict()
                        for key, value in csv_metadata.items():
                            if key not in output_item:
                                output_item[key] = value
                            elif key == 'midi_filename' and downloaded_midi_path:
                                output_item['original_midi_filename_from_csv'] = value
                        yield output_item
                        current_sample_offset = end_sample
            except StopIteration: # From next(iter(processed_ds['item'])) if dataset is empty
                print(f"Warning: Could not load audio stream for {hf_audio_url}. It might be empty or inaccessible.")
                continue
            except Exception as e:
                print(f"Error processing audio file {csv_audio_path_in_repo} (URL: {hf_audio_url}): {e}")
                continue

if __name__ == "__main__":
    print("Starting MAESTRO dataset demonstration (CSV-driven, with MIDI download)...")

    TARGET_SAMPLE_RATE = 16000
    DATASET_SPLIT = "validation"
    # --- Test with chunking ---
    CHUNK_DURATION_SECONDS = 10.0 # 10 seconds audio chunks
    MAX_MIDI_PER_CHUNK = 512    # Max 512 MIDI tokens for each chunk

    print(f"Initializing MaestroDataset:")
    print(f"  Split: {DATASET_SPLIT}")
    print(f"  Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    if CHUNK_DURATION_SECONDS is not None:
        print(f"  Audio Chunk Duration: {CHUNK_DURATION_SECONDS}s")
        print(f"  Max MIDI Tokens per Chunk: {MAX_MIDI_PER_CHUNK}")
    else:
        print("  Chunking: Disabled (processing full sequences)")

    try:
        maestro_dataset = MaestroDataset(
            split=DATASET_SPLIT,
            sample_rate=TARGET_SAMPLE_RATE,
            chunk_duration_sec=CHUNK_DURATION_SECONDS,
            max_midi_tokens_per_chunk=MAX_MIDI_PER_CHUNK
        )
        dataset_iterator = iter(maestro_dataset)

        print("\nFetching a sample from the dataset (ordered by CSV)...")
        for i in range(2): # Try to get 2 samples
            try:
                sample = next(dataset_iterator)
                print(f"\n--- Sample {i+1} Information ---")
                for key, value in sample.items():
                    if key == "waveform":
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, length={value.shape[1]/sample.get('sample_rate', TARGET_SAMPLE_RATE):.2f}s")
                    elif key == "downloaded_midi_path":
                        print(f"  {key}: {value}")
                        if value and os.path.exists(value):
                            print(f"    MIDI file size: {os.path.getsize(value)} bytes")
                        elif value:
                            print(f"    Warning: MIDI file path listed but not found at {value}")
                    elif key == "midi_tokens":
                        if value is not None:
                            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                                print(f"  {key}: (List[List[int]]) {len(value)} track(s)")
                                print(f"    First track tokens (sample): {value[0][:10]}...")
                            elif isinstance(value, list):
                                print(f"  {key}: (List[int]) {len(value)} tokens")
                                print(f"    Tokens (sample): {value[:10]}...")
                            else:
                                print(f"  {key}: {type(value)} {str(value)[:60]}...")
                        else:
                            print(f"  {key}: None")
                    else:
                        str_value = str(value)
                        if len(str_value) > 70: str_value = str_value[:67] + "..."
                        print(f"  {key}: {str_value}")
                
                if i == 0:
                    print("\n--- Audio Playback (First Sample) ---")
                    waveform_to_play = sample["waveform"]
                    sr_to_play = sample["sample_rate"]
                    try:
                        import sounddevice as sd
                        audio_to_play_np = waveform_to_play.squeeze().numpy()
                        if audio_to_play_np.ndim == 1 and audio_to_play_np.size > 0:
                            print(f"Playing audio ({audio_to_play_np.shape[0]/sr_to_play:.2f} seconds)...")
                            sd.play(audio_to_play_np, samplerate=sr_to_play)
                            sd.wait()
                            print("Playback finished.")
                        else:
                            print("Cannot play audio: waveform is empty or has unexpected shape.")
                    except ImportError:
                        print("Sounddevice not installed. Skipping audio playback. pip install sounddevice")
                    except Exception as e_play:
                        print(f"Error during playback: {e_play}")

            except StopIteration:
                print(f"\nNo more samples available in the '{DATASET_SPLIT}' split after {i} samples.")
                break
            except Exception as e_sample:
                print(f"\nError fetching sample {i+1}: {e_sample}")
                break

    except Exception as e:
        print(f"\nAn unexpected error occurred during dataset initialization or iteration: {e}")
        import traceback
        traceback.print_exc()

    print("\nMAESTRO dataset demonstration finished.")
