import wandb
import torch
import argparse
import os
import sys
import time
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm  # Import tqdm for progress bars

# Add the parent directory to the path to allow importing local modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import CHECKPOINTS_DATA_DIR
from dataset import get_datasets
from model import AudioClassifier, SpectrogramClassifier, WhisperStyleAudioClassifier  # Added WhisperStyleAudioClassifier
from torch.amp import autocast, GradScaler  # Updated import to use torch.amp instead of torch.cuda.amp

# Set default seed for reproducibility
def set_seed(seed=42):
    """
    Set seed for all random number generators for reproducibility.
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model_with_config():
    """
    Train a model with configuration parameters from wandb.
    This function is called for each sweep run.
    """
    # Initialize a new wandb run
    with wandb.init() as run:
        # Get hyperparameters from wandb
        config = wandb.config
        
        # Create timestamp for run identification
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Set up device
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Set PyTorch memory allocation settings to reduce fragmentation
        if device.type == "cuda":
            # Set environment variable for expandable segments to avoid memory fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            # Set memory management parameters
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
        # Set random seed
        set_seed(config.seed)
                
        # Modify dataset loading to use fold 1 for testing and folds 2-10 for training
        print("Loading datasets with fold-based split...")
        train_dataset, test_dataset = get_datasets(
            sample_rate=config.sample_rate,
            target_length=config.sample_rate * 4,  # 4 seconds of audio
            num_augmentations=config.num_augmentations,  # Use num_augmentations parameter (0 = no augmentation)
            max_length=None,  # Use all available data
            train_folds=list(range(2, 10)),  # Use folds 2-10 for training
            test_folds=[1]  # Use fold 1 for testing
        )

        # Print number of items per class in train and test datasets
        from collections import Counter
        train_labels = [item['label'] for item in train_dataset]
        test_labels = [item['label'] for item in test_dataset]
        print("Train class counts:", dict(Counter(train_labels)))
        print("Test class counts:", dict(Counter(test_labels)))
        
        # Configure cache size if specified
        if hasattr(config, 'dataset_cache_size'):
            train_dataset._cache_size = config.dataset_cache_size
            test_dataset._cache_size = config.dataset_cache_size
        
        print(f"Datasets loaded. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        print(f"Dataset cache size: {train_dataset._cache_size}")

        # Adjust batch size based on available memory
        # Start with the configured batch size, but be ready to reduce it
        effective_batch_size = config.batch_size
        grad_accum_steps = config.gradient_accumulation_steps
        
        # If we have a very large model or using spectrogram approach with large dimensions
        adjust_batch_size = False
        if config.model_type in ['raw', 'spectrogram']:
            if hasattr(config, 'd_model') and hasattr(config, 'num_encoder_layers') and \
               (config.d_model >= 512 and config.num_encoder_layers >= 6):
                adjust_batch_size = True
            if config.model_type == 'spectrogram' and \
               hasattr(config, 'n_fft') and hasattr(config, 'n_mels') and \
               (config.n_fft >= 512 and config.n_mels >= 128):
                adjust_batch_size = True
        elif config.model_type == 'whisperstyle':
            if hasattr(config, 'n_state') and hasattr(config, 'n_layer') and \
               (config.n_state >= 512 and config.n_layer >= 6):
                adjust_batch_size = True
            # Optionally, add checks for n_fft and n_mels for whisperstyle if they can become very large
            # For current whisperstyle sweep, n_fft=400, n_mels=80, so not triggering large spectrogram part
            # if hasattr(config, 'n_fft') and hasattr(config, 'n_mels') and \
            #    (config.n_fft >= 512 and config.n_mels >= 128):
            #     adjust_batch_size = True
                
        if adjust_batch_size:
            effective_batch_size = max(16, effective_batch_size // 2)  # Reduced min from 32 to 16 for safety
            grad_accum_steps = max(2, grad_accum_steps * 2)
            print(f"Adjusted batch size for {config.model_type} model to {effective_batch_size} with {grad_accum_steps} gradient accumulation steps due to large model/feature parameters.")
        
        # Create dataloaders
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=effective_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
        except RuntimeError as e:
            # If we hit a CUDA error during dataloader creation, try with a smaller batch size
            print(f"Error creating dataloaders: {e}")
            print("Trying with smaller batch size and more accumulation steps...")
            effective_batch_size = max(16, effective_batch_size // 4)
            grad_accum_steps = max(4, grad_accum_steps * 4)
            print(f"Re-adjusted to batch size {effective_batch_size}, accumulation steps {grad_accum_steps}")
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=effective_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
        
        # Log dataset info to wandb, including original vs augmented counts
        original_count = len(train_dataset.dataset)
        total_count = len(train_dataset)
        augmented_count = total_count - original_count
        
        # Log actual batch size and gradient accumulation steps used
        wandb.config.update({
            "train_size_total": len(train_dataset),
            "train_size_original": original_count,
            "train_size_augmented": augmented_count,
            "test_size": len(test_dataset),
            "timestamp": timestamp,
            "effective_batch_size": effective_batch_size,
            "effective_gradient_accumulation_steps": grad_accum_steps,
            "effective_total_batch_size": effective_batch_size * grad_accum_steps,
            "dataset_cache_size": train_dataset._cache_size
        })
        
        # Create model based on model_type
        if config.model_type == 'raw':
            print("Creating raw audio transformer model...")
            model = AudioClassifier(
                num_classes=10,
                d_model=config.d_model,
                nhead=config.nhead,
                num_encoder_layers=config.num_encoder_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                feature_extractor_base_filters=config.feature_extractor_base_filters,
                learning_rate=config.learning_rate  # This will be overridden below
            ).model
        elif config.model_type == 'spectrogram':
            print("Creating spectrogram transformer model...")
            model = SpectrogramClassifier(
                num_classes=10,
                d_model=config.d_model,
                nhead=config.nhead,
                num_encoder_layers=config.num_encoder_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                base_filters=config.feature_extractor_base_filters,
                sample_rate=config.sample_rate,
                learning_rate=config.learning_rate  # This will be overridden below
            ).model
        elif config.model_type == 'whisperstyle':
            print("Creating Whisper-style audio classifier model...")
            model = WhisperStyleAudioClassifier(
                num_classes=10,
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                n_state=config.n_state,
                n_head=config.n_head,
                n_layer=config.n_layer,
                transformer_n_ctx=config.transformer_n_ctx,
                dropout=config.dropout,
                normalize_spectrogram=config.normalize_spectrogram
            )
        else:
            raise ValueError(f"Unsupported model_type: {config.model_type}")
        
        # Move model to device
        model.to(device)
        
        # Create custom optimizer with configurable parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        # Create criterion (loss function) with label smoothing
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Create learning rate scheduler based on sweep config
        if config.scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                min_lr=config.scheduler_min_lr,
                verbose=True
            )
        elif config.scheduler_type == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.scheduler_t_max * len(train_loader),
                eta_min=config.scheduler_min_lr
            )
        elif config.scheduler_type == 'one_cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.learning_rate * 10,
                steps_per_epoch=len(train_loader),
                epochs=config.num_epochs,
                pct_start=0.3
            )
        else:
            raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
            
        # Create a classifier wrapper to maintain the same API as before
        class ClassifierWrapper:
            def __init__(self, model, optimizer, criterion, scheduler):
                self.model = model
                self.optimizer = optimizer
                self.criterion = criterion
                self.scheduler = scheduler
                
            def evaluate(self, dataloader):
                """Evaluate the model on a dataset"""
                self.model.eval()
                total_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in dataloader:
                        waveform = batch['waveform'].to(device)
                        labels = batch['label'].to(device)
                        
                        # Forward pass
                        logits = self.model(waveform)
                        loss = self.criterion(logits, labels)
                        
                        # Update metrics
                        total_loss += loss.item() * waveform.size(0)
                        preds = torch.argmax(logits, dim=1)
                        correct += (preds == labels).sum().item()
                        total += waveform.size(0)
                
                return {
                    'loss': total_loss / total if total > 0 else float('inf'),
                    'accuracy': correct / total if total > 0 else 0
                }
                
            def save(self, filepath):
                """Save model checkpoint"""
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                }, filepath)
        
        # Create the classifier wrapper
        classifier = ClassifierWrapper(model, optimizer, criterion, scheduler)
        
        # Enable mixed precision
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Initialize gradient scaler for mixed precision training
        use_mixed_precision = (device.type == "cuda")
        scaler = GradScaler(enabled=use_mixed_precision)  # Fix: remove incorrect device_type parameter
        
        # Count and log number of parameters
        num_params = count_parameters(classifier.model)
        wandb.config.update({"num_parameters": num_params})
        print(f"Model has {num_params:,} trainable parameters")
        
        # Log model graph and optimizer config to wandb
        wandb.watch(classifier.model, log="all", log_freq=10)
        wandb.log({
            "optimizer/beta1": config.beta1,
            "optimizer/beta2": config.beta2,
            "optimizer/weight_decay": config.weight_decay,
            "optimizer/eps": config.eps,
            "scheduler/type": config.scheduler_type
        })
        
        # Training loop
        best_eval_accuracy = 0.0
        best_eval_loss = float('inf')
        early_stop_patience = config.early_stop_patience
        early_stop_counter = 0
        
        print(f"Training for {config.num_epochs} epochs...")
        
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            
            # Clear dataset caches before each epoch to free memory
            if hasattr(train_dataset, 'clear_cache'):
                print("Clearing dataset caches...")
                train_dataset.clear_cache()
                test_dataset.clear_cache()
                # Force garbage collection
                import gc
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Training phase
            classifier.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Track epoch training time
            epoch_start_time = time.time()
            
            # Add tqdm progress bar for training loop
            progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{config.num_epochs}", 
                               leave=True, ncols=100)
            
            for i, batch in enumerate(progress_bar):
                try:
                    # Move data to device
                    waveform = batch['waveform'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                    
                    # Mixed precision training - updated to use the new API format
                    with autocast('cuda', enabled=use_mixed_precision):
                        # Forward pass
                        logits = classifier.model(waveform)
                        loss = classifier.criterion(logits, labels)
                        
                        # Scale loss for gradient accumulation
                        if grad_accum_steps > 1:
                            loss = loss / grad_accum_steps
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Only step optimizer after accumulating gradients
                    if (i + 1) % grad_accum_steps == 0:
                        # Unscale gradients for gradient clipping
                        scaler.unscale_(classifier.optimizer)
                        
                        # Add gradient clipping to prevent exploding gradients and NaN loss
                        torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), max_norm=1.0)
                        
                        # Update weights with gradient scaling for mixed precision
                        scaler.step(classifier.optimizer)
                        scaler.update()
                        classifier.optimizer.zero_grad(set_to_none=True)
                        
                        # Update OneCycleLR scheduler if used (it's step-based)
                        if config.scheduler_type == 'one_cycle':
                            classifier.scheduler.step()
                    
                    # Calculate metrics
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=1)
                        batch_correct = (preds == labels).sum().item()
                        batch_total = labels.size(0)
                        train_correct += batch_correct
                        train_total += batch_total
                    
                    # Scale loss back for reporting if needed
                    current_loss = loss.item()
                    if grad_accum_steps > 1:
                        current_loss *= grad_accum_steps
                    
                    train_loss += current_loss
                    
                    # Update progress bar with current metrics
                    batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0
                    progress_bar.set_postfix({
                        'loss': f"{current_loss:.4f}", 
                        'acc': f"{batch_accuracy:.4f}"
                    })
                except RuntimeError as e:
                    # Detect CUDA OOM errors
                    if "CUDA out of memory" in str(e):
                        print(f"CUDA OOM error: {e}")
                        print("This run is aborting due to GPU memory limitations.")
                        # Log the OOM error to wandb
                        wandb.log({"error": "CUDA OOM", "error_message": str(e)})
                        # Return a negative score to indicate failure
                        return -1.0
                    else:
                        # Re-raise other runtime errors
                        raise
            
            # Calculate epoch metrics
            epoch_loss = train_loss / len(train_loader)
            epoch_accuracy = train_correct / train_total if train_total > 0 else 0
            
            # Calculate epoch training time
            epoch_time = time.time() - epoch_start_time
            
            # Log training metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy,
                "learning_rate": classifier.optimizer.param_groups[0]['lr'],
                "epoch_training_time": epoch_time  # Log training time per epoch
            })
            
            # Evaluation phase
            classifier.model.eval()
            # Add tqdm progress bar for evaluation loop
            eval_progress = tqdm(test_loader, desc=f"Eval Epoch {epoch+1}/{config.num_epochs}", leave=True, ncols=100)
            eval_loss = 0.0
            eval_correct = 0
            eval_total = 0
            
            with torch.no_grad():
                for batch in eval_progress:
                    # Move data to device
                    waveform = batch['waveform'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                    
                    # Forward pass
                    logits = classifier.model(waveform)
                    loss = classifier.criterion(logits, labels)
                    
                    # Calculate metrics
                    preds = torch.argmax(logits, dim=1)
                    batch_correct = (preds == labels).sum().item()
                    batch_total = labels.size(0)
                    eval_correct += batch_correct
                    eval_total += batch_total
                    eval_loss += loss.item() * batch_total
                    
                    # Update progress bar
                    batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0
                    eval_progress.set_postfix({
                        'loss': f"{loss.item():.4f}", 
                        'acc': f"{batch_accuracy:.4f}"
                    })
            
            # Calculate evaluation metrics
            eval_epoch_loss = eval_loss / eval_total if eval_total > 0 else float('inf')
            eval_epoch_accuracy = eval_correct / eval_total if eval_total > 0 else 0
            
            # Store metrics in a format compatible with classifier.evaluate
            eval_metrics = {
                'loss': eval_epoch_loss,
                'accuracy': eval_epoch_accuracy
            }
            
            wandb.log({
                "epoch": epoch,
                "eval_loss": eval_metrics['loss'],
                "eval_accuracy": eval_metrics['accuracy']
            })
            
            # Update learning rate based on validation loss for plateau scheduler
            if config.scheduler_type == 'reduce_on_plateau':
                classifier.scheduler.step(eval_metrics['loss'])
            # Update cosine scheduler if used (it's epoch-based)
            elif config.scheduler_type == 'cosine_annealing':
                classifier.scheduler.step()
            
            # Print epoch results
            print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f} (in {epoch_time:.2f}s)")
            print(f"Evaluation - Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"Learning rate: {classifier.optimizer.param_groups[0]['lr']:.8f}")
            
            # Save best model based on validation accuracy with accuracy in filename
            if eval_metrics['accuracy'] > best_eval_accuracy:
                best_eval_accuracy = eval_metrics['accuracy']
                # Format accuracy as a percentage in the filename
                acc_str = f"{best_eval_accuracy:.2f}".replace('.', '_')
                model_type_str = config.model_type
                best_model_path = CHECKPOINTS_DATA_DIR / f"{model_type_str}_acc_{acc_str}_model_{run.id}.pt"
                CHECKPOINTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
                classifier.save(str(best_model_path))
                print(f"Saved best model with accuracy: {best_eval_accuracy:.4f}")
                early_stop_counter = 0
            
            # Early stopping based on evaluation loss
            if eval_metrics['loss'] < best_eval_loss:
                best_eval_loss = eval_metrics['loss']
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"Validation loss did not improve for {early_stop_counter} epochs")
                
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Final evaluation
        # Clear caches before final evaluation
        if hasattr(train_dataset, 'clear_cache'):
            train_dataset.clear_cache()
            test_dataset.clear_cache()
            # Force garbage collection
            import gc
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        final_metrics = classifier.evaluate(test_loader)
        print(f"\nFinal Evaluation - Loss: {final_metrics['loss']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")
        
        # Calculate parameter efficiency
        param_efficiency = best_eval_accuracy / (num_params / 1e6)  # Accuracy per million parameters
        
        # Calculate memory efficiency - accuracy per MB of peak memory
        peak_memory_mb = 0
        if device.type == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
        
        memory_efficiency = best_eval_accuracy / max(peak_memory_mb, 1) if peak_memory_mb > 0 else 0
        
        # Log final metrics
        wandb.log({
            "final_eval_loss": final_metrics['loss'],
            "final_eval_accuracy": final_metrics['accuracy'],
            "best_eval_accuracy": best_eval_accuracy,
            "parameter_efficiency": param_efficiency,  # Accuracy per million parameters
            "peak_memory_mb": peak_memory_mb,
            "memory_efficiency": memory_efficiency,  # Accuracy per MB
            "model_type": config.model_type
        })
        
        # Return metric to optimize (higher accuracy is better)
        return final_metrics['accuracy']

def create_sweep_config():
    """
    Creates the W&B sweep configuration.
    This sweep focuses on the impact of 'sample_rate' across different model types.
    """
    sweep_config = {
        'method': 'grid',  # Use 'grid' to test all sample_rate combinations for each model
        'metric': {
            'name': 'best_eval_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'model_type': {
                'values': ["raw", "spectrogram", "whisperstyle"]
            },
            'sample_rate': {
                'values': [8000, 16000, 32000] # Key variable for this sweep
            },

            # --- Fixed Architectural & Training Params to isolate sample_rate effect ---
            'd_model': {
                'value': 128
            },
            'nhead': {
                'value': 4
            },
            'num_encoder_layers': {
                'value': 4
            },
            'dim_feedforward': { # Typically d_model * 4
                'value': 512
            },
            'dropout': {
                'value': 0.1
            },
            'learning_rate': {
                'value': 0.0001
            },
            'batch_size': { # Effective batch size per device before accumulation
                'value': 32 # A reasonable starting point, can be adapted by the script
            },
            'gradient_accumulation_steps': {
                'value': 1 # Start with 1, can be adapted by the script
            },
            'optimizer_type': {
                'value': "adamw"
            },
            'beta1': {'value': 0.9},
            'beta2': {'value': 0.999},
            'eps': {'value': 1e-8},
            'weight_decay': {
                'value': 0.01
            },
            'scheduler_type': {
                'value': "reduce_on_plateau"
            },
            'scheduler_factor': {'value': 0.2}, # Renamed from reduce_on_plateau_factor
            'scheduler_patience': {'value': 3}, # Renamed from reduce_on_plateau_patience
            'scheduler_min_lr': {'value': 1e-7},
            'num_epochs': { # Max epochs; early stopping will likely trigger sooner
                'value': 10
            },
            'early_stop_patience': {
                'value': 10
            },
            'label_smoothing': {
                'value': 0.1
            },
            'use_mixup': { # Disable augmentations to focus on sample_rate
                'value': False
            },
            'mixup_alpha': { # Irrelevant if use_mixup is false
                'value': 0.4
            },
            'split_ratio': { # For train/test split by get_datasets
                'value': 0.8
            },
            'num_augmentations': { # Dataset-level augmentations (0 for none)
                'value': 0
            },
            'num_workers': { # Dataloader workers
                'value': 2
            },
            'seed': { # Random seed for reproducibility
                'value': 42
            },
            'device': { # Device to use
                'value': 'cuda' # Will fallback to 'cpu' if cuda not available
            },
            'dataset_cache_size': { # For UrbanSoundDataset caching
                'value': 128
            },
            'n_fft': {
                'value': 1024
            },
            'hop_length': {
                'value': 256
            },
            'n_mels': {
                'value': 128
            },
            'feature_extractor_base_filters': { # For raw_transformer's Conv1D stem
                 'value': 16 # Example value, adjust as needed
            },
            'n_state': {'value': 128}, # Corresponds to d_model for Whisper
            'n_head': {'value': 4},   # Corresponds to nhead for Whisper
            'n_layer': {'value': 4},  # Corresponds to num_encoder_layers for Whisper
            'transformer_n_ctx': {'value': 1024}, # Max sequence length for Whisper's transformer
            'normalize_spectrogram': {'value': True},
            'target_length_multiplier': { # Duration of audio clips in seconds (4s * sample_rate)
                 'value': 4
            },
            'use_wandb_logging': { # This is implicit if wandb.init() is called
                'value': True
            }
        }
    }
    return sweep_config

def main():
    """
    Main function to create and run the W&B sweep.
    """
    parser = argparse.ArgumentParser(description="Run a W&B sweep for UrbanSound8K classification.")
    parser.add_argument(
        '--count', 
        type=int, 
        default=None,  # Default to None to run all combinations in a grid sweep
        help='Maximum number of runs for the agent. If None, runs all combinations for grid sweeps.'
    )
    parser.add_argument(
        '--project', 
        type=str, 
        default="urbansound8k_sample_rate_analysis",
        help='W&B project name for the sweep.'
    )
    parser.add_argument(
        '--entity', 
        type=str, 
        default=None,  # Uses default W&B entity if None
        help='W&B entity (username or organization).'
    )
    parser.add_argument(
        '--initialize_sweep_only',
        action='store_true',
        help='If set, only initializes the sweep and prints the sweep ID, then exits.'
    )
    args = parser.parse_args()

    # Create the single, comprehensive sweep configuration
    sweep_config = create_sweep_config()

    # Initialize the sweep
    print(f"Initializing sweep for project '{args.project}' under entity '{args.entity if args.entity else 'default'}'...")
    sweep_id = wandb.sweep(sweep_config, entity=args.entity, project=args.project)
    
    print(f"\nSweep created successfully!")
    print(f"  Sweep ID: {sweep_id}")
    print(f"  Project: {args.project}")
    if args.entity:
        print(f"  Entity: {args.entity}")
    
    # Construct the sweep URL carefully
    entity_str = args.entity if args.entity else '[YOUR_WANDB_ENTITY]' # Placeholder if no entity provided
    sweep_url = f"https://wandb.ai/{entity_str}/{args.project}/sweeps/{sweep_id}"
    print(f"  Sweep URL: {sweep_url}")
    
    print(f"\nTo run the agent, use the following command:")
    agent_command_entity = args.entity if args.entity else '[YOUR_WANDB_ENTITY]'
    print(f"  wandb agent {agent_command_entity}/{args.project}/{sweep_id}")

    if args.initialize_sweep_only:
        print("\nSweep initialized. Exiting as per --initialize_sweep_only flag.")
        return

    # Start the sweep agent
    print(f"\nStarting W&B agent for sweep ID: {sweep_id}...")
    try:
        wandb.agent(sweep_id, function=train_model_with_config, count=args.count)
        print(f"\nSweep agent finished processing runs.")
    except KeyboardInterrupt:
        print("\nSweep agent interrupted by user. Exiting.")
    except Exception as e:
        print(f"\nAn error occurred while running the sweep agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()