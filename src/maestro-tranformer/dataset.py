import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, Audio
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
import os

class MaestroDataset(IterableDataset):
    """
    A PyTorch IterableDataset for the MAESTRO dataset.
    It iterates over a metadata CSV, loads specified audio files directly
    from Hugging Face, and processes them on the fly.
    """

    def __init__(self, split="train", sample_rate=16000,
                 dataset_repo_id="ddPn08/maestro-v3.0.0",
                 metadata_filename="maestro-v3.0.0.csv"):
        """
        Initialize the MAESTRO dataset.

        Args:
            split (str): Dataset split to use ("train", "validation", or "test").
                         Filters the metadata CSV by this split.
            sample_rate (int): Target sample rate for audio.
            dataset_repo_id (str): Hugging Face dataset repository ID.
            metadata_filename (str): Name of the metadata CSV file in the dataset repository.
        """
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.dataset_repo_id = dataset_repo_id
        self.metadata_filename = metadata_filename
        self.metadata_df = pd.DataFrame()

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

            if 'audio_filename' not in self.metadata_df.columns:
                print(f"Warning: 'audio_filename' column not found in the filtered metadata. Audio loading will fail.")
                self.metadata_df = pd.DataFrame() # Ensure it's empty to prevent further errors

        except Exception as e:
            print(f"Error downloading or parsing metadata CSV {self.metadata_filename}: {e}")
            # self.metadata_df remains an empty DataFrame

    def __iter__(self):
        if self.metadata_df.empty:
            print("Metadata is empty, cannot iterate.")
            return iter([]) # Return an empty iterator

        for index, row in self.metadata_df.iterrows():
            csv_audio_path_in_repo = row.get('audio_filename')

            if not csv_audio_path_in_repo or not isinstance(csv_audio_path_in_repo, str):
                print(f"Skipping row {index} due to missing or invalid 'audio_filename': {csv_audio_path_in_repo}")
                continue

            # Construct the direct Hugging Face URL for this audio file
            # Ensure csv_audio_path_in_repo does not start with a slash if dataset_repo_id already implies root
            clean_csv_audio_path = csv_audio_path_in_repo.lstrip('/')
            hf_audio_url = f"https://huggingface.co/datasets/{self.dataset_repo_id}/resolve/main/{clean_csv_audio_path}"

            try:
                # Load the single audio file using 'audiofolder'
                # The key for data_files (e.g., 'item') becomes the split name in the loaded dataset
                single_file_ds = load_dataset(
                    "audiofolder",
                    data_files={"item": [hf_audio_url]},
                    streaming=True
                )

                # Apply resampling and mono conversion
                # The 'audiofolder' loader creates an 'audio' column by default.
                processed_ds = single_file_ds.cast_column(
                    "audio",
                    Audio(sampling_rate=self.sample_rate, mono=True)
                )

                # Get the single item from this dataset stream
                audio_stream_item = next(iter(processed_ds['item']))

                audio_array = audio_stream_item['audio']['array']
                loaded_audio_path = audio_stream_item['audio']['path'] # This will be the URL

                # Optional: Path check for sanity
                if os.path.basename(loaded_audio_path) != os.path.basename(csv_audio_path_in_repo):
                    print(f"Warning: Basename mismatch! CSV: {os.path.basename(csv_audio_path_in_repo)}, Loaded: {os.path.basename(loaded_audio_path)}")

                waveform = torch.tensor(audio_array, dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0) # Ensure [1, num_samples]

                output_item = {
                    "waveform": waveform,
                    "sample_rate": self.sample_rate,
                    "source_audio_url": loaded_audio_path
                }
                # Add all metadata from the CSV row
                output_item.update(row.to_dict())

                if index == 0: # Print details for the first successfully processed item
                    print("\n--- Details for the First Processed Item ---")
                    print(f"CSV Index: {index}")
                    print(f"Audio Filename (from CSV): {csv_audio_path_in_repo}")
                    print(f"Loaded Audio URL: {loaded_audio_path}")
                    print(f"Waveform shape: {waveform.shape}, dtype: {waveform.dtype}")
                    print("Metadata from CSV:")
                    for key, value in row.to_dict().items():
                        print(f"  {key}: {value}")
                    print("--------------------------------------------")

                yield output_item

            except Exception as e:
                print(f"Error processing file {csv_audio_path_in_repo} (URL: {hf_audio_url}): {e}")
                # Optionally, yield a placeholder or skip
                continue

if __name__ == "__main__":
    print("Starting MAESTRO dataset demonstration (CSV-driven)...")

    # --- Configuration ---
    TARGET_SAMPLE_RATE = 16000
    DATASET_SPLIT = "validation"  # Changed to validation for a potentially smaller set for quick testing

    print(f"Initializing MaestroDataset:")
    print(f"  Split: {DATASET_SPLIT}")
    print(f"  Sample Rate: {TARGET_SAMPLE_RATE} Hz")

    try:
        maestro_dataset = MaestroDataset(
            split=DATASET_SPLIT,
            sample_rate=TARGET_SAMPLE_RATE
        )

        dataset_iterator = iter(maestro_dataset)

        print("\nFetching the first sample from the dataset (ordered by CSV)...")
        # Fetch a few samples to demonstrate
        for i in range(2): # Try to get 2 samples
            try:
                sample = next(dataset_iterator)
                print(f"\n--- Sample {i+1} Information ---")
                for key, value in sample.items():
                    if key == "waveform":
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, length={value.shape[1]/sample['sample_rate']:.2f}s")
                    else:
                        print(f"  {key}: {value}")

                # Play the first sample
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
                break # Stop if there's an error fetching a sample

    except Exception as e:
        print(f"\nAn unexpected error occurred during dataset initialization or iteration: {e}")
        import traceback
        traceback.print_exc()

    print("\nMAESTRO dataset demonstration finished.")
