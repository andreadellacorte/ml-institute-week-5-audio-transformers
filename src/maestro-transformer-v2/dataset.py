# filepath: /Users/andreadellacorte/Documents/Workspace/GitHub/ml-institute-week-5-audio-transformers/src/maestro-transformer-v2/dataset.py

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import List, Tuple, Dict, Optional, Union
from collections import defaultdict

from src.config import PROCESSED_DATA_DIR

# Dataset paths 
MAESTRO_PROCESSED_DIR = PROCESSED_DATA_DIR / "ddPn08-maestro-v3.0.0"
TRAIN_DIR = MAESTRO_PROCESSED_DIR / "train"
EVAL_DIR = MAESTRO_PROCESSED_DIR / "eval"

class MaestroMIDISpectrogramDataset(Dataset):
    """
    Dataset for loading paired MIDI tokens and audio spectrograms from the processed MAESTRO data.
    """
    def __init__(self, 
                 data_dir: Union[str, Path], 
                 max_seq_length: Optional[int] = None,
                 random_seed: int = 42):
        """
        Initialize the MAESTRO dataset.
        
        Args:
            data_dir: Directory containing processed data (train or eval folder)
            max_seq_length: Maximum sequence length for MIDI tokens (None for no limit)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.max_seq_length = max_seq_length
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Find all unique sample IDs (song_id_chunk)
        self.sample_ids = self._get_sample_ids()
        print(f"Found {len(self.sample_ids)} paired samples in {data_dir}")
    
    def _get_sample_ids(self) -> List[str]:
        """
        Get all unique sample IDs from the data directory.
        
        Returns:
            List of sample IDs (song_id_chunk format)
        """
        # Find all MIDI tokenized files
        tokenized_files = list(self.data_dir.glob("*_*_midi_tokenised.pkl"))
        
        # Find all spectrogram files
        spectrogram_files = list(self.data_dir.glob("*_*_spectrogram.pkl"))
        
        # Extract unique sample IDs (song_id_chunk) from tokenized files
        tokenized_ids = set()
        for file in tokenized_files:
            # Extract song_id_chunk from filename (e.g., "101_3_midi_tokenised.pkl" -> "101_3")
            parts = file.stem.split("_midi_tokenised")[0]
            tokenized_ids.add(parts)
        
        # Extract unique sample IDs from spectrogram files
        spectrogram_ids = set()
        for file in spectrogram_files:
            # Extract song_id_chunk from filename (e.g., "101_3_spectrogram.pkl" -> "101_3")
            parts = file.stem.split("_spectrogram")[0]
            spectrogram_ids.add(parts)
        
        # Find IDs that have both MIDI tokens and spectrograms
        common_ids = tokenized_ids.intersection(spectrogram_ids)
        
        # Sort for reproducibility
        return sorted(list(common_ids))
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample by index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'midi_tokens': Tensor of MIDI tokens
                - 'spectrogram': Tensor of spectrogram
                - 'sample_id': String identifier of the sample
        """
        # Get sample ID
        sample_id = self.sample_ids[idx]
        
        # Load MIDI tokens
        midi_tokens_path = self.data_dir / f"{sample_id}_midi_tokenised.pkl"
        with open(midi_tokens_path, 'rb') as f:
            midi_tokens = pickle.load(f)
        
        # Debug - uncomment to check the structure
        # print(f"Sample {sample_id}, MIDI token type: {type(midi_tokens)}, shape: {np.shape(midi_tokens) if isinstance(midi_tokens, np.ndarray) else len(midi_tokens)}")
        
        # Handle different token structures
        if isinstance(midi_tokens, dict):
            # REMI tokenizer might return a dictionary with 'ids' as the token sequence
            if 'ids' in midi_tokens:
                midi_tokens = midi_tokens['ids']
            else:
                # If it's another structure, flatten the values
                all_tokens = []
                for key in sorted(midi_tokens.keys()):
                    if isinstance(midi_tokens[key], (list, np.ndarray)):
                        all_tokens.extend(midi_tokens[key])
                midi_tokens = all_tokens
        
        # Convert to tensor (handling different input types)
        if isinstance(midi_tokens, np.ndarray):
            midi_tokens = torch.from_numpy(midi_tokens)
            # Convert to long for token IDs or keep as float for embeddings
            if midi_tokens.dtype in [np.int32, np.int64]:
                midi_tokens = midi_tokens.long()
            else:
                midi_tokens = midi_tokens.float()
        elif isinstance(midi_tokens, list):
            # Check if inner elements are lists/tuples (2D structure)
            if midi_tokens and isinstance(midi_tokens[0], (list, tuple, np.ndarray)):
                midi_tokens = torch.tensor(midi_tokens, dtype=torch.float)
            else:
                midi_tokens = torch.tensor(midi_tokens, dtype=torch.long)
        
        # Important: Keep the tensor in its original shape
        # From the debug output, it seems the REMI tokenizer returns shape [1, seq_len]
        # Do NOT squeeze the tensor as this will be handled in the collate function
        
        # Trim MIDI tokens if needed - handle different tensor shapes
        if self.max_seq_length:
            if len(midi_tokens.shape) == 1:
                if midi_tokens.size(0) > self.max_seq_length:
                    midi_tokens = midi_tokens[:self.max_seq_length]
            elif len(midi_tokens.shape) == 2 and midi_tokens.size(0) == 1:
                # Shape is [1, seq_len]
                if midi_tokens.size(1) > self.max_seq_length:
                    midi_tokens = midi_tokens[:, :self.max_seq_length]
            elif len(midi_tokens.shape) == 2:
                # Shape is [seq_len, features]
                if midi_tokens.size(0) > self.max_seq_length:
                    midi_tokens = midi_tokens[:self.max_seq_length, :]
        
        # Load spectrogram
        spec_path = self.data_dir / f"{sample_id}_spectrogram.pkl"
        with open(spec_path, 'rb') as f:
            spectrogram = pickle.load(f)
        
        # Convert spectrogram to tensor
        spectrogram = torch.from_numpy(spectrogram).float()
        
        return {
            'midi_tokens': midi_tokens,
            'spectrogram': spectrogram,
            'sample_id': sample_id
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for handling variable length MIDI token sequences.
    
    Args:
        batch: List of dictionaries containing 'midi_tokens', 'spectrogram', 'sample_id'
        
    Returns:
        Dictionary containing batched tensors:
            - 'midi_tokens': Padded tensor of MIDI tokens [batch_size, max_seq_len] or [batch_size, seq_dim, feature_dim]
            - 'lengths': Lengths of original sequences [batch_size]
            - 'spectrogram': Batched spectrograms [batch_size, freq, time]
            - 'sample_ids': List of sample IDs
    """
    # Get sample IDs
    sample_ids = [item['sample_id'] for item in batch]
    
    # Check the shape of the first item's midi_tokens to determine how to pad
    first_item = batch[0]['midi_tokens']
    
    # Based on the debug output, MIDI tokens have shape [1, seq_len] 
    # or maybe [seq_len, features] - need to handle both cases
    
    if len(first_item.shape) == 1:
        # Case 1: 1D tokens (just a sequence of token IDs)
        lengths = [item['midi_tokens'].size(0) for item in batch]
        max_len = max(lengths)
        
        # Padded MIDI tokens
        padded_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, item in enumerate(batch):
            seq_len = item['midi_tokens'].size(0)
            padded_tokens[i, :seq_len] = item['midi_tokens']
    
    elif len(first_item.shape) == 2:
        # Case 2: 2D tokens - we need to determine if it's [1, seq_len] or [seq_len, features]
        if first_item.shape[0] == 1:
            # Shape is [1, seq_len] - REMI tokens are in this format based on debug output
            lengths = [item['midi_tokens'].size(1) for item in batch]
            max_len = max(lengths)
            
            # Create padded tensor with shape [batch_size, 1, max_seq_len]
            padded_tokens = torch.zeros(len(batch), 1, max_len, dtype=first_item.dtype)
            for i, item in enumerate(batch):
                seq_len = item['midi_tokens'].size(1)
                padded_tokens[i, 0, :seq_len] = item['midi_tokens'][0, :seq_len]
            
            # Reshape to [batch_size, max_seq_len] to make it easier to use in models
            padded_tokens = padded_tokens.squeeze(1)
        else:
            # Shape is [seq_len, features]
            num_features = first_item.shape[1]
            lengths = [item['midi_tokens'].size(0) for item in batch]
            max_len = max(lengths)
            
            # Padded MIDI tokens with feature dimension
            padded_tokens = torch.zeros(len(batch), max_len, num_features, dtype=first_item.dtype)
            for i, item in enumerate(batch):
                seq_len = item['midi_tokens'].size(0)
                padded_tokens[i, :seq_len, :] = item['midi_tokens']
    
    else:
        raise ValueError(f"Unexpected MIDI token shape: {first_item.shape}")
    
    # Stack spectrograms
    spectrograms = torch.stack([item['spectrogram'] for item in batch], dim=0)
    
    return {
        'midi_tokens': padded_tokens,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'spectrogram': spectrograms,
        'sample_ids': sample_ids
    }

def create_maestro_dataloaders(
    batch_size: int = 32,
    max_seq_length: Optional[int] = None,
    train_dir: Union[str, Path] = TRAIN_DIR,
    eval_dir: Union[str, Path] = EVAL_DIR,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and evaluation.
    
    Args:
        batch_size: Batch size for DataLoaders
        max_seq_length: Maximum sequence length for MIDI tokens (None for no limit)
        train_dir: Directory containing training data
        eval_dir: Directory containing evaluation data
        num_workers: Number of workers for DataLoader
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_dataloader, eval_dataloader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    # Create datasets
    train_dataset = MaestroMIDISpectrogramDataset(
        data_dir=train_dir,
        max_seq_length=max_seq_length,
        random_seed=random_seed
    )
    
    eval_dataset = MaestroMIDISpectrogramDataset(
        data_dir=eval_dir,
        max_seq_length=max_seq_length,
        random_seed=random_seed
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, eval_loader

def get_dataset_statistics(data_dir: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """
    Get statistics about the processed dataset files.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Dictionary of statistics
    """
    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
    
    # Get all files by type
    midi_files = list(data_dir.glob("*_*.midi"))
    spectrogram_files = list(data_dir.glob("*_*_spectrogram.pkl"))
    tokenized_files = list(data_dir.glob("*_*_midi_tokenised.pkl"))
    
    # Count unique songs
    song_ids = defaultdict(int)
    for file in midi_files:
        song_id = file.stem.split("_")[0]
        song_ids[song_id] += 1
    
    return {
        "files": {
            "midi": len(midi_files),
            "spectrogram": len(spectrogram_files),
            "tokenized_midi": len(tokenized_files)
        },
        "songs": {
            "count": len(song_ids),
            "avg_chunks_per_song": round(len(midi_files) / len(song_ids) if song_ids else 0, 2)
        }
    }

def inspect_midi_tokenized_file(file_path):
    """
    Inspect the structure of a tokenized MIDI file to understand its format.
    
    Args:
        file_path: Path to the tokenized MIDI file (.pkl)
        
    Returns:
        Dictionary containing structure information
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        info = {
            'type': type(data).__name__,
            'structure': None,
            'shape': None,
            'sample': None
        }
        
        if isinstance(data, dict):
            info['structure'] = 'dict'
            info['keys'] = list(data.keys())
            
            # Sample from first key
            if data and info['keys']:
                first_key = info['keys'][0]
                first_value = data[first_key]
                info['first_key_type'] = type(first_value).__name__
                
                if isinstance(first_value, (list, np.ndarray)):
                    info['first_key_length'] = len(first_value)
                    if len(first_value) > 0:
                        info['first_item_type'] = type(first_value[0]).__name__
                        info['sample'] = first_value[:10]  # First 10 elements
        
        elif isinstance(data, list):
            info['structure'] = 'list'
            info['shape'] = len(data)
            
            # Check if it's a 2D list
            if data and isinstance(data[0], (list, np.ndarray)):
                info['nested'] = True
                info['nested_shape'] = [len(data), len(data[0]) if data[0] else 0]
                info['sample'] = data[0][:10] if data[0] else []
            else:
                info['nested'] = False
                info['sample'] = data[:10]  # First 10 elements
        
        elif isinstance(data, np.ndarray):
            info['structure'] = 'numpy.ndarray'
            info['shape'] = data.shape
            info['dtype'] = str(data.dtype)
            
            if data.size > 0:
                if data.ndim == 1:
                    info['sample'] = data[:10].tolist()  # First 10 elements
                elif data.ndim == 2:
                    info['sample'] = data[0, :10].tolist() if data.shape[1] >= 10 else data[0].tolist()
        
        return info
    
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # Example usage and testing
    print("MAESTRO Dataset Statistics:")
    
    # Get statistics for train and eval sets
    train_stats = get_dataset_statistics(TRAIN_DIR)
    eval_stats = get_dataset_statistics(EVAL_DIR)
    
    print(f"\nTraining set statistics:")
    print(f"Files: {train_stats['files']}")
    print(f"Songs: {train_stats['songs']['count']}, " +
          f"Avg {train_stats['songs']['avg_chunks_per_song']} chunks per song")
    
    print(f"\nEvaluation set statistics:")
    print(f"Files: {eval_stats['files']}")
    print(f"Songs: {eval_stats['songs']['count']}, " +
          f"Avg {eval_stats['songs']['avg_chunks_per_song']} chunks per song")
    
    # Inspect a sample tokenized MIDI file structure
    print("\nInspecting tokenized MIDI file structure:")
    tokenized_files = list(TRAIN_DIR.glob("*_*_midi_tokenised.pkl"))
    if tokenized_files:
        sample_file = tokenized_files[0]
        print(f"Sample file: {sample_file}")
        info = inspect_midi_tokenized_file(sample_file)
        print(f"Structure: {info}")
    
    # Create example dataloaders with small batch size for testing
    print("\nCreating example dataloaders with batch_size=2:")
    train_loader, eval_loader = create_maestro_dataloaders(batch_size=2)
    
    # Get one batch from train loader to test
    print("\nLoading one batch from training dataloader:")
    try:
        batch = next(iter(train_loader))
        print(f"MIDI tokens shape: {batch['midi_tokens'].shape}")
        print(f"Spectrogram shape: {batch['spectrogram'].shape}")
        print(f"Sequence lengths: {batch['lengths']}")
        print(f"Sample IDs: {batch['sample_ids']}")
        print("Successfully loaded batch!")
    except Exception as e:
        print(f"Error loading batch: {e}")
        
        # If there's an error, try to load individual items directly
        print("\nDebug: Trying to load individual samples:")
        try:
            dataset = MaestroMIDISpectrogramDataset(TRAIN_DIR)
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Sample 0 loaded successfully:")
                print(f"- MIDI tokens type: {type(sample['midi_tokens'])}")
                print(f"- MIDI tokens shape: {sample['midi_tokens'].shape}")
                print(f"- Spectrogram shape: {sample['spectrogram'].shape}")
                
                # Try to manually collate 2 samples to verify our collate function
                if len(dataset) > 1:
                    print("\nTesting manual collation of 2 samples:")
                    sample1 = dataset[0]
                    sample2 = dataset[1]
                    mini_batch = [sample1, sample2]
                    
                    # Print shapes before collation
                    print(f"Sample 1 MIDI shape: {sample1['midi_tokens'].shape}")
                    print(f"Sample 2 MIDI shape: {sample2['midi_tokens'].shape}")
                    
                    # Try collation
                    collated = collate_fn(mini_batch)
                    print("Manual collation successful!")
                    print(f"Collated MIDI tokens shape: {collated['midi_tokens'].shape}")
                    print(f"Collated lengths: {collated['lengths']}")
        except Exception as inner_e:
            print(f"Error loading individual sample: {inner_e}")