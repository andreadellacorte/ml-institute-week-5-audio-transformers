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
    It loads data from the 'train' split of the source, then internally
    splits it into training and validation/test sets.
    Iterates over a metadata CSV, loads audio, downloads MIDI, and processes them.
    Can yield full sequences or fixed-size chunks.
    Can be limited to a maximum number of items from its designated internal split.
    """

    def __init__(self, 
                 mode: str = "train", # "train" or "validation" (or "test")
                 train_split_percentage: float = 0.8, # Percentage of source 'train' data for internal training
                 sample_rate=16000,
                 dataset_repo_id="ddPn08/maestro-v3.0.0",
                 metadata_filename="maestro-v3.0.0.csv",
                 chunk_duration_sec: float = None,
                 max_midi_tokens_per_chunk: int = None,
                 max_items: int = 10,
                 random_seed_for_split: int = 42 # Seed for reproducible train/val split
                 ):
        """
        Initialize the MAESTRO dataset.

        Args:
            mode (str): "train" or "validation". Determines which internal split to use.
            train_split_percentage (float): Percentage of the data to allocate to the internal training set.
            sample_rate (int): Target sample rate for audio.
            dataset_repo_id (str): Hugging Face dataset repository ID.
            metadata_filename (str): Name of the metadata CSV file in the dataset repository.
            chunk_duration_sec (float): Duration of audio chunks in seconds.
            max_midi_tokens_per_chunk (int): Maximum number of MIDI tokens per chunk.
            max_items (int, optional): Maximum number of items to yield from the selected internal split.
            random_seed_for_split (int): Seed used for shuffling before creating the internal train/validation split.
        """
        super().__init__()
        self.mode = mode
        self.train_split_percentage = train_split_percentage
        self.sample_rate = sample_rate
        self.dataset_repo_id = dataset_repo_id
        self.metadata_filename = metadata_filename
        self.internal_metadata_df = pd.DataFrame() # This will hold the metadata for the current mode

        self.chunk_duration_sec = chunk_duration_sec
        self.max_midi_tokens_per_chunk = max_midi_tokens_per_chunk
        if self.chunk_duration_sec is not None:
            self.chunk_samples = int(self.chunk_duration_sec * self.sample_rate)
            if self.max_midi_tokens_per_chunk is None:
                print("Warning: chunk_duration_sec is set, but max_midi_tokens_per_chunk is not. Defaulting to 512.")
                self.max_midi_tokens_per_chunk = 512
        else:
            self.chunk_samples = None
            if self.max_midi_tokens_per_chunk is not None:
                print("Warning: max_midi_tokens_per_chunk is set, but chunk_duration_sec is not. max_midi_tokens_per_chunk will be ignored.")

        self.max_items = max_items
        self.random_seed_for_split = random_seed_for_split
        self._warned_no_sec_to_tick_files = set() # Initialize set for suppressing warnings

        tokenizer_config = TokenizerConfig(
            num_velocities=32, use_chords=True, use_programs=False, use_sustain_pedals=True
        )
        self.tokenizer = REMI(tokenizer_config)

        try:
            metadata_path = hf_hub_download(
                repo_id=self.dataset_repo_id,
                filename=self.metadata_filename,
                repo_type="dataset"
            )
            source_full_metadata_df = pd.read_csv(metadata_path)

            if 'split' in source_full_metadata_df.columns:
                # Always use the 'train' split from the source for our internal splitting
                source_train_df = source_full_metadata_df[source_full_metadata_df['split'] == 'train'].reset_index(drop=True)
                if source_train_df.empty:
                    print(f"Warning: No entries found for source split 'train' in {self.metadata_filename}.")
                    return # internal_metadata_df remains empty
            else:
                print(f"Warning: 'split' column not found in {self.metadata_filename}. Using all entries as source for internal split.")
                source_train_df = source_full_metadata_df
            
            if 'audio_filename' not in source_train_df.columns or 'midi_filename' not in source_train_df.columns:
                print(f"Warning: 'audio_filename' or 'midi_filename' column not found in the source 'train' metadata. Processing might fail.")
                return # internal_metadata_df remains empty

            # Shuffle the source_train_df for unbiased internal splitting
            shuffled_source_train_df = source_train_df.sample(frac=1, random_state=self.random_seed_for_split).reset_index(drop=True)
            
            split_index = int(len(shuffled_source_train_df) * self.train_split_percentage)

            if self.mode == "train":
                self.internal_metadata_df = shuffled_source_train_df.iloc[:split_index].reset_index(drop=True)
                print(f"Initialized for 'train' mode: {len(self.internal_metadata_df)} files.")
            elif self.mode == "validation" or self.mode == "test":
                self.internal_metadata_df = shuffled_source_train_df.iloc[split_index:].reset_index(drop=True)
                print(f"Initialized for '{self.mode}' mode: {len(self.internal_metadata_df)} files.")
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Choose 'train', 'validation', or 'test'.")
            
            if self.internal_metadata_df.empty:
                 print(f"Warning: Internal metadata for mode '{self.mode}' is empty after splitting.")


        except Exception as e:
            print(f"Error downloading or parsing metadata CSV {self.metadata_filename} for internal splitting: {e}")
            # self.internal_metadata_df remains an empty DataFrame

    def __len__(self):
        """Return the effective length of the dataset for the current mode."""
        if self.internal_metadata_df.empty:
            return 0
        
        actual_metadata_length = len(self.internal_metadata_df)
        
        if self.max_items is not None and self.max_items >= 0:
            return min(self.max_items, actual_metadata_length) if not self.chunk_samples else self.max_items if self.max_items < actual_metadata_length else actual_metadata_length

        return actual_metadata_length

    def __iter__(self):
        if self.internal_metadata_df.empty:
            print(f"Internal metadata for mode '{self.mode}' is empty, cannot iterate.")
            return iter([]) 

        items_yielded = 0 

        for index, row in self.internal_metadata_df.iterrows(): # Iterate over the mode-specific dataframe
            if self.max_items is not None and items_yielded >= self.max_items:
                print(f"Reached max_items ({self.max_items}) for mode '{self.mode}', stopping iteration.")
                break 

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
                )
            except Exception as e_midi:
                print(f"Error downloading MIDI file {csv_midi_path_in_repo} for row {index}: {e_midi}")
                continue

            try:
                single_file_ds = load_dataset(
                    "audiofolder",
                    data_files={"item": [hf_audio_url]},
                    streaming=True,
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
                    items_yielded += 1

                else:
                    num_total_samples = waveform.shape[1]
                    current_sample_offset = 0
                    
                    while current_sample_offset < num_total_samples:
                        start_sample = current_sample_offset
                        end_sample = min(current_sample_offset + self.chunk_samples, num_total_samples)
                        
                        min_chunk_samples = self.sample_rate // 2
                        if (end_sample - start_sample) < min_chunk_samples and start_sample > 0:
                             current_sample_offset = end_sample
                             continue

                        audio_chunk = waveform[:, start_sample:end_sample]
                        
                        start_time_sec = start_sample / self.sample_rate
                        end_time_sec = end_sample / self.sample_rate
                        
                        try:
                            start_tick = -1
                            end_tick = -1

                            if hasattr(full_midi_score, 'sec_to_tick') and callable(full_midi_score.sec_to_tick):
                                start_tick = full_midi_score.sec_to_tick(start_time_sec)
                                end_tick = full_midi_score.sec_to_tick(end_time_sec)
                            elif hasattr(full_midi_score, 'ticks_per_quarter') and hasattr(full_midi_score, 'tempos'):
                                if downloaded_midi_path not in self._warned_no_sec_to_tick_files:
                                    print(f"Warning: MIDI file {downloaded_midi_path} (type {type(full_midi_score)}) lacks 'sec_to_tick'. Attempting manual time conversion. (Further warnings for this file path will be suppressed for this dataset worker/instance)")
                                    self._warned_no_sec_to_tick_files.add(downloaded_midi_path)
                                
                                tpq = full_midi_score.ticks_per_quarter
                                if callable(tpq):
                                    tpq = tpq()

                                active_mspq = 500000.0 
                                tempos_list = full_midi_score.tempos
                                if callable(tempos_list):
                                    tempos_list = tempos_list()

                                if tempos_list:
                                    sorted_tempos = sorted(tempos_list, key=lambda t: t.time)
                                    initial_tempo_event = next((t for t in sorted_tempos if t.time <= 0), None)
                                    if initial_tempo_event:
                                        active_mspq = initial_tempo_event.mspq
                                    elif sorted_tempos:
                                        active_mspq = sorted_tempos[0].mspq
                                
                                if active_mspq <= 0:
                                    print(f"Warning: Invalid mspq ({active_mspq}) for {downloaded_midi_path}. Using default 500000.0.")
                                    active_mspq = 500000.0
                                
                                tps_approx = (float(tpq) * 1_000_000.0) / float(active_mspq)
                                
                                start_tick = int(start_time_sec * tps_approx)
                                end_tick = int(end_time_sec * tps_approx)
                            else:
                                raise AttributeError(f"MIDI object for {downloaded_midi_path} (type {type(full_midi_score)}) lacks 'sec_to_tick' and also 'ticks_per_quarter'/'tempos' for fallback conversion.")

                            chunk_midi_score = full_midi_score.clip(start_tick, end_tick, clip_end=True)
                            
                            chunk_midi_tokens = self.tokenizer(chunk_midi_score)
                        except Exception as e_midi_chunk:
                            print(f"Error slicing/tokenizing MIDI chunk for {downloaded_midi_path} ({start_time_sec:.2f}s-{end_time_sec:.2f}s): {e_midi_chunk}")
                            current_sample_offset = end_sample
                            continue

                        if not chunk_midi_tokens or not chunk_midi_tokens[0]:
                            current_sample_offset = end_sample
                            continue

                        if len(chunk_midi_tokens[0]) > self.max_midi_tokens_per_chunk:
                            current_sample_offset = end_sample
                            continue

                        output_item = {
                            "waveform": audio_chunk,
                            "sample_rate": self.sample_rate,
                            "source_audio_url": loaded_audio_path,
                            "downloaded_midi_path": downloaded_midi_path,
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
                        items_yielded += 1
                        if self.max_items is not None and items_yielded >= self.max_items:
                            print(f"Reached max_items ({self.max_items}) during chunking for mode '{self.mode}', stopping iteration.")
                            return
                        current_sample_offset = end_sample
            except StopIteration:
                print(f"Warning: Could not load audio stream for {hf_audio_url}. It might be empty or inaccessible.")
                continue
            except Exception as e:
                print(f"Error processing audio file {csv_audio_path_in_repo} (URL: {hf_audio_url}): {e}")
                continue

if __name__ == "__main__":
    print("Starting MAESTRO dataset demonstration (CSV-driven, with MIDI download)...")

    TARGET_SAMPLE_RATE = 16000
    DATASET_SPLIT = "validation"
    CHUNK_DURATION_SECONDS = 10.0
    MAX_MIDI_PER_CHUNK = 512
    MAX_DATASET_ITEMS = None

    print(f"Initializing MaestroDataset:")
    print(f"  Split: {DATASET_SPLIT}")
    print(f"  Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    if CHUNK_DURATION_SECONDS is not None:
        print(f"  Audio Chunk Duration: {CHUNK_DURATION_SECONDS}s")
        print(f"  Max MIDI Tokens per Chunk: {MAX_MIDI_PER_CHUNK}")
    else:
        print("  Chunking: Disabled (processing full sequences)")
    if MAX_DATASET_ITEMS is not None:
        print(f"  Max Dataset Items: {MAX_DATASET_ITEMS}")

    try:
        maestro_dataset = MaestroDataset(
            mode="train",
            train_split_percentage=0.8,
            sample_rate=TARGET_SAMPLE_RATE,
            chunk_duration_sec=CHUNK_DURATION_SECONDS,
            max_midi_tokens_per_chunk=MAX_MIDI_PER_CHUNK,
            max_items=MAX_DATASET_ITEMS
        )
        dataset_iterator = iter(maestro_dataset)

        print("\nFetching a sample from the dataset (ordered by CSV)...")
        print(f"Reported dataset length: {len(maestro_dataset)}")

        for i in range(2):
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
