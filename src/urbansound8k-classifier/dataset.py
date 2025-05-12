import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
from typing import List, Optional, Tuple


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
        split_ratio = 0.8
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
        """
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.augment = augment
        self.split_ratio = split_ratio
        
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
    
    def __getitem__(self, idx):
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
                # Trim
                waveform = waveform[:, :self.target_length]
        
        # Apply augmentation if enabled
        if self.augment:
            waveform = self._augment_audio(waveform)
        
        # Get label
        label = self.class_to_idx[item["classID"]]
        
        return {
            "waveform": waveform,
            "sample_rate": self.sample_rate,
            "label": label,
            "class_name": item["class"],
        }
    
    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to the audio waveform."""
        # Random gain adjustment (volume)
        if torch.rand(1) > 0.5:
            gain = 0.5 + torch.rand(1) * 1.0  # Random gain between 0.5 and 1.5
            waveform = waveform * gain
        
        # Random time shift
        if torch.rand(1) > 0.5:
            shift_amount = int(waveform.shape[1] * 0.1 * torch.rand(1))  # Up to 10% shift
            if torch.rand(1) > 0.5:  # Left or right shift
                waveform = torch.roll(waveform, shifts=shift_amount, dims=1)
            else:
                waveform = torch.roll(waveform, shifts=-shift_amount, dims=1)
        
        # Ensure values are in the valid range
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform


def get_datasets(
    sample_rate: int = 22050,
    max_length: Optional[int] = None,
    fold_split: Optional[Tuple[List[int], List[int]]] = None,
    target_length: Optional[int] = None,
    augment: bool = False,
):
    """
    Create train and test dataset splits.
    
    Args:
        sample_rate: Target sample rate
        max_length: Maximum samples per split
        fold_split: Tuple of (train_folds, test_folds) if using fold-based splits
        target_length: Pad/trim audio to this length
        augment: Whether to apply data augmentation to training data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if fold_split:
        train_folds, test_folds = fold_split
        
        # Create datasets using specific folds
        train_datasets = []
        for fold in train_folds:
            train_datasets.append(
                UrbanSoundDataset(
                    split="train",
                    sample_rate=sample_rate,
                    max_length=max_length,
                    fold=fold,
                    target_length=target_length,
                    augment=augment
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
                    augment=False  # No augmentation for test data
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
            augment=augment
        )
        
        test_dataset = UrbanSoundDataset(
            split="test", 
            sample_rate=sample_rate,
            max_length=max_length,
            target_length=target_length, 
            augment=False  # No augmentation for test data
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