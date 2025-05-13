import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
from typing import List, Optional, Tuple
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
        num_augmentations: int = 0,  # Number of augmented copies to generate (0 means no augmentation)
        split_ratio = 0.8,
        cache_size: int = 1024,  # Reduced default cache size to save memory
        prefetch_factor: int = 2,  # Controls how many samples to prefetch per worker
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
            num_augmentations: Number of augmented copies to generate per sample (0 means no augmentation)
            split_ratio: Ratio for train/test split if not using folds
            cache_size: Maximum number of samples to cache in memory
            prefetch_factor: Controls how many samples to prefetch per worker
        """
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.augment = augment
        self.num_augmentations = num_augmentations
        self.split_ratio = split_ratio
        self._cache = {}  # Initialize cache dictionary
        self._cache_size = cache_size
        self._cache_keys = []  # LRU tracking
        self.prefetch_factor = prefetch_factor
        
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
        
        # Flag to determine if we're returning augmented variants
        self.use_multiple_augmentations = augment and num_augmentations > 0
        
        # Calculate actual dataset length based on augmentations
        self._effective_length = len(self.dataset)
        if self.use_multiple_augmentations:
            # Each original sample + num_augmentations additional samples
            self._effective_length = len(self.dataset) * (1 + num_augmentations)
    
    def __len__(self):
        return self._effective_length
    
    def _get_original_index_and_aug_id(self, idx):
        """
        Map global index to original dataset index and augmentation ID.
        
        For example, if num_augmentations=2:
        - Global indices 0, 1, 2 map to original sample 0 with aug_ids 0, 1, 2
        - Global indices 3, 4, 5 map to original sample 1 with aug_ids 0, 1, 2
        - etc.
        
        Where aug_id 0 means "no augmentation" (original sample)
        """
        if not self.use_multiple_augmentations:
            return idx, 0
        
        # Each original sample has (1 + num_augmentations) entries
        items_per_original = 1 + self.num_augmentations
        original_idx = idx // items_per_original
        aug_id = idx % items_per_original  # 0 means original, 1+ means augmented
        
        return original_idx, aug_id
    
    @functools.lru_cache(maxsize=128)  # Cache the last 128 items returned
    def __getitem__(self, idx):
        # Map to original index and augmentation ID
        original_idx, aug_id = self._get_original_index_and_aug_id(idx)
        
        # Check if item is in cache
        cache_key = (original_idx, aug_id)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get the original item data
        item = self.dataset[original_idx]
        
        # Load audio and convert to tensor (always use float32)
        audio_data = item["audio"]["array"]
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        
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
                # Trim
                if aug_id > 0:  # Take a random segment for augmented samples
                    start = torch.randint(0, waveform.size(1) - self.target_length + 1, (1,)).item()
                    waveform = waveform[:, start:start + self.target_length]
                else:  # Take the beginning for original samples (deterministic)
                    waveform = waveform[:, :self.target_length]
        
        # Apply augmentation for non-zero aug_id if using multiple augmentations
        if self.use_multiple_augmentations and aug_id > 0:
            # Set a unique seed for each (sample_idx, aug_id) combination for reproducibility
            # But still have different augmentations for each aug_id
            seed = hash((original_idx, aug_id)) % (2**32)
            torch.manual_seed(seed)
            waveform = self._augment_audio(waveform)
        # Apply augmentation if enabled but not using multiple augmentations
        elif self.augment and not self.use_multiple_augmentations:
            waveform = self._augment_audio(waveform)
        
        # Get label
        label = self.class_to_idx[item["classID"]]
        
        # Create a minimal result dictionary to save memory
        result = {
            "waveform": waveform,
            "label": label,
        }
        
        # Only add these fields if not in training/eval loop (for analysis/debugging)
        if not hasattr(torch.utils.data, '_DataLoader__initialized'):
            result.update({
                "sample_rate": self.sample_rate,
                "class_name": item["class"],
                "is_augmented": aug_id > 0,
                "original_index": original_idx
            })
        
        # Cache management with memory limits
        if len(self._cache) < self._cache_size:
            self._cache[cache_key] = result
            self._cache_keys.append(cache_key)
        elif self._cache_keys:  # If cache is full, remove oldest item
            old_key = self._cache_keys.pop(0)
            del self._cache[old_key]
            self._cache[cache_key] = result
            self._cache_keys.append(cache_key)
        
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

    def clear_cache(self):
        """Clear the sample cache to free memory"""
        self._cache = {}
        self._cache_keys = []


def get_datasets(
    sample_rate: int = 22050,
    max_length: Optional[int] = None,
    fold_split: Optional[Tuple[List[int], List[int]]] = None,
    target_length: Optional[int] = None,
    num_augmentations: int = 0,  # Number of augmented copies per original sample (0 = no augmentation)
    split_ratio: float = 0.8
):
    """
    Get train and test datasets for the UrbanSound8K dataset.
    
    Args:
        sample_rate: Target sample rate for audio
        max_length: Maximum number of samples to include in each dataset
        fold_split: Optional tuple of (train_folds, test_folds) to use specific folds
        target_length: If provided, pad/trim all audio to this length
        num_augmentations: Number of augmented copies to generate per sample (0 = no augmentation)
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
                    augment=num_augmentations > 0,  # Enable augmentation if num_augmentations > 0
                    num_augmentations=num_augmentations,
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
                    num_augmentations=0,  # No augmented copies for test data
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
            augment=num_augmentations > 0,  # Enable augmentation if num_augmentations > 0
            num_augmentations=num_augmentations,
            split_ratio=split_ratio
        )
        
        test_dataset = UrbanSoundDataset(
            split="test", 
            sample_rate=sample_rate,
            max_length=max_length,
            target_length=target_length, 
            augment=False,  # No augmentation for test data
            num_augmentations=0,  # No augmented copies for test data
            split_ratio=split_ratio
        )
    
    return train_dataset, test_dataset