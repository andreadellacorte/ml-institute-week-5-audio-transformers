import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
from typing import List, Optional, Tuple
import numpy as np
import functools


class UrbanSoundDataset(Dataset):
    """
    A PyTorch Dataset for the UrbanSound8K dataset from Hugging Face.
    
    This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes:
    air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, 
    gun_shot, jackhammer, siren, and street_music.
    """
    
    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 22050,
        max_length: Optional[int] = None,
        fold: Optional[int] = None,
        target_length: Optional[int] = None,
        augment: bool = False,
        split_ratio = 0.8,
        cache_size: int = 1000  # Number of samples to cache in memory
    ):
        """
        Initialize the UrbanSound8K dataset.
        
        Args:
            split: Dataset split to use ("train" or "test")
            sample_rate: Target sample rate for audio
            max_length: Maximum number of samples to include
            fold: If provided, only includes data from this fold
            target_length: If provided, pad/trim all audio to this length
            augment: Whether to apply data augmentation
            cache_size: Maximum number of samples to cache in memory
        """
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.augment = augment
        self.split_ratio = split_ratio
        self._cache = {}  # Initialize cache dictionary
        self._cache_size = cache_size
        self._cache_keys = []  # LRU tracking
        
        # Load dataset from Hugging Face
        self.dataset = load_dataset("danavery/urbansound8K", split="train")
        
        # Add audio with resampling
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        
        # Handle folds: UrbanSound8K comes with 10 predefined folds
        if fold is not None:
            self.dataset = self.dataset.filter(lambda x: x["fold"] == fold)
        else:
            # Use standard split: 80% train, 20% test based on indices
            total_size = len(self.dataset)
            indices = list(range(total_size))
            
            split_idx = int(self.split_ratio * total_size)
            
            if split == "train":
                selected_indices = indices[:split_idx]
            else:  # test
                selected_indices = indices[split_idx:]
            
            self.dataset = self.dataset.select(selected_indices)
        
        # Limit dataset size if specified
        if max_length is not None:
            self.dataset = self.dataset.select(range(min(max_length, len(self.dataset))))
        
        # Create label to index mapping
        self.class_names = sorted(list(set(self.dataset["classID"])))
        self.class_to_idx = {cls_id: i for i, cls_id in enumerate(self.class_names)}
    
    def __len__(self):
        return len(self.dataset)
    
    @functools.lru_cache(maxsize=8)  # Cache the last 8 items returned
    def __getitem__(self, idx):
        # Check if item is in cache
        if idx in self._cache:
            return self._cache[idx]
        
        item = self.dataset[idx]
        
        # Load audio and convert to tensor
        audio_data = item["audio"]["array"]
        waveform = torch.tensor(audio_data).float()
        
        # Handle stereo to mono conversion if needed
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        
        # Apply target length constraint if specified
        if self.target_length is not None:
            if waveform.size(1) < self.target_length:
                # Pad
                padding = self.target_length - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.size(1) > self.target_length:
                # Trim (take a random segment for training data)
                if self.augment:
                    start = torch.randint(0, waveform.size(1) - self.target_length + 1, (1,)).item()
                    waveform = waveform[:, start:start + self.target_length]
                else:
                    waveform = waveform[:, :self.target_length]
        
        # Apply augmentation if enabled
        if self.augment:
            waveform = self._augment_audio(waveform)
        
        # Get label
        label = self.class_to_idx[item["classID"]]
        
        result = {
            "waveform": waveform,
            "sample_rate": self.sample_rate,
            "label": label,
            "class_name": item["class"],
        }
        
        # Cache the result if cache isn't full
        if len(self._cache) < self._cache_size:
            self._cache[idx] = result
            self._cache_keys.append(idx)
        elif self._cache_keys:  # If cache is full, remove oldest item
            old_idx = self._cache_keys.pop(0)
            del self._cache[old_idx]
            self._cache[idx] = result
            self._cache_keys.append(idx)
        
        return result
    
    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to the audio waveform more efficiently."""
        # Apply a single augmentation type based on random choice
        # This is faster than checking each augmentation separately
        aug_type = torch.randint(0, 3, (1,)).item()
        
        if aug_type == 0:  # Random gain adjustment (volume)
            gain = 0.5 + torch.rand(1) * 1.0  # Random gain between 0.5 and 1.5
            waveform = waveform * gain
        elif aug_type == 1:  # Random time shift
            shift_amount = int(waveform.shape[1] * 0.1 * torch.rand(1))  # Up to 10% shift
            direction = 1 if torch.rand(1) > 0.5 else -1  # Left or right shift
            waveform = torch.roll(waveform, shifts=shift_amount * direction, dims=1)
        elif aug_type == 2:  # Random noise
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            
        # Ensure values are in the valid range
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform


def get_datasets(
    sample_rate: int = 22050,
    max_length: Optional[int] = None,
    fold_split: Optional[Tuple[List[int], List[int]]] = None,
    target_length: Optional[int] = None,
    augment: bool = False,
    split_ratio: float = 0.8
):
    """
    Get train and test datasets for the UrbanSound8K dataset.
    
    Args:
        sample_rate: Target sample rate for audio
        max_length: Maximum number of samples to include in each dataset
        fold_split: Optional tuple of (train_folds, test_folds) to use specific folds
        target_length: If provided, pad/trim all audio to this length
        augment: Whether to apply data augmentation (train only)
        split_ratio: Ratio for train/test split if not using folds
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if fold_split is not None:
        train_folds, test_folds = fold_split
        
        # Create datasets for each train fold
        train_datasets = []
        for fold in train_folds:
            train_datasets.append(
                UrbanSoundDataset(
                    split="train",
                    sample_rate=sample_rate,
                    max_length=max_length,
                    fold=fold,
                    target_length=target_length,
                    augment=augment,
                    split_ratio=split_ratio
                )
            )
        
        test_datasets = []
        for fold in test_folds:
            test_datasets.append(
                UrbanSoundDataset(
                    split="test",
                    sample_rate=sample_rate,
                    max_length=max_length,
                    fold=fold,
                    target_length=target_length,
                    augment=False,  # No augmentation for test data
                    split_ratio=split_ratio
                )
            )
        
        # For simplicity, just use one fold for now
        train_dataset = train_datasets[0] if train_datasets else None
        test_dataset = test_datasets[0] if test_datasets else None
    else:
        # Create train/test splits based on indexing
        train_dataset = UrbanSoundDataset(
            split="train", 
            sample_rate=sample_rate,
            max_length=max_length,
            target_length=target_length, 
            augment=augment,
            split_ratio=split_ratio
        )
        
        test_dataset = UrbanSoundDataset(
            split="test", 
            sample_rate=sample_rate,
            max_length=max_length,
            target_length=target_length, 
            augment=False,  # No augmentation for test data
            split_ratio=split_ratio
        )
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Example usage with dataset
    print("=== UrbanSound8K Dataset Example ===")
    train_dataset, test_dataset = get_datasets(
        sample_rate=16000,
        target_length=16000 * 4,  # 4 seconds of audio at 16kHz
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Get a sample from the dataset
    sample = train_dataset[0]
    print(f"Sample waveform shape: {sample['waveform'].shape}")
    print(f"Sample label: {sample['label']} ({sample['class_name']})")
    
    # Test with data loader
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,
        shuffle=True,
        num_workers=2
    )
    
    for batch in train_loader:
        print(f"Batch waveform shape: {batch['waveform'].shape}")
        print(f"Batch labels: {batch['label']}")
        print(f"Batch processed successfully!")
        break