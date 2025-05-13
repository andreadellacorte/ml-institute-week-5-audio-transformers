import wandb
import torch
import argparse
import os
import time
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm  # Import tqdm for progress bars

from src.config import CHECKPOINTS_DATA_DIR
from dataset import get_datasets
from model import AudioClassifier, SpectrogramClassifier
from torch.amp import autocast  # Updated import from torch.amp instead of torch.cuda.amp
from torch.cuda.amp import GradScaler

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
                
        # Create datasets with num_augmentations (replaces use_augmentation)
        print("Loading datasets...")
        train_dataset, test_dataset = get_datasets(
            sample_rate=config.sample_rate,
            target_length=config.sample_rate * 4,  # 4 seconds of audio
            num_augmentations=config.num_augmentations,  # Use num_augmentations parameter (0 = no augmentation)
            max_length=None,  # Use all available data
            split_ratio=config.split_ratio
        )
        
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
        if (config.d_model >= 512 and config.num_encoder_layers >= 6) or \
           (config.model_type == 'spectrogram' and config.n_fft >= 512 and config.n_mels >= 128):
            # Reduce batch size for larger models to prevent OOM
            effective_batch_size = max(32, effective_batch_size // 2)
            # Increase gradient accumulation steps to compensate
            grad_accum_steps = max(2, grad_accum_steps * 2)
            print(f"Adjusted batch size to {effective_batch_size} with {grad_accum_steps} gradient accumulation steps")
        
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
        else:  # spectrogram
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
        
        # Create criterion (loss function)
        criterion = torch.nn.CrossEntropyLoss()
        
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
        scaler = GradScaler(enabled=use_mixed_precision)
        
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

def create_sweep_config(model_type=None):
    """
    Create a configuration for the wandb sweep with Bayesian optimization.
    Memory-optimized to prevent CUDA OOM errors.
    
    Args:
        model_type: If specified, creates a config for only this model type ('raw' or 'spectrogram').
                   If None, randomly selects between both model types.
    
    Returns:
        Dictionary with sweep configuration
    """
    # If model_type is not specified, select it randomly to create a balanced set of runs
    if model_type is None:
        import random
        model_type = random.choice(['raw', 'spectrogram'])
    
    print(f"Creating sweep configuration for model_type: {model_type}")
    
    # Common parameters for both model types
    common_params = {
        # Model architecture common parameters
        'd_model': {
            'values': [64, 128, 256]  # Reduced maximum model dimension 
        },
        'nhead': {
            'values': [4, 8]  # Reduced maximum heads
        },
        'num_encoder_layers': {
            'values': [2, 4, 6]  # Reduced maximum layers
        },
        'dim_feedforward': {
            'values': [256, 512, 1024]  # Reduced maximum feedforward dimension
        },
        'dropout': {
            'values': [0.05, 0.1, 0.2]  # Dropout rate
        },
        'feature_extractor_base_filters': {
            'values': [8, 16, 32]  # Base filters for feature extractor
        },
        
        # Memory optimization parameters
        'use_mixed_precision': {
            'values': [True, False]  # Whether to use mixed precision (fp16)
        },
        'dataset_cache_size': {
            'values': [512, 1024]  # Number of samples to cache in dataset
        },
        
        # Training parameters
        'batch_size': {
            'values': [32, 64, 128]  # Reduced batch sizes
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'num_epochs': {
            'value': 20  # Fixed to avoid wasting resources
        },
        'gradient_accumulation_steps': {
            'values': [2, 4, 8]  # Increased accumulation steps to compensate for smaller batches
        },
        
        # Adam optimizer parameters
        'beta1': {
            'values': [0.9, 0.95, 0.99]  # First momentum coefficient
        },
        'beta2': {
            'values': [0.990, 0.995, 0.999]  # Second momentum coefficient
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3
        },
        'eps': {
            'values': [1e-8, 1e-7, 1e-6]  # Numerical stability term
        },
        
        # Learning rate scheduler parameters
        'scheduler_type': {
            'values': ['reduce_on_plateau', 'cosine_annealing', 'one_cycle']  # Type of scheduler
        },
        'scheduler_patience': {
            'values': [2, 3, 5]  # Patience for ReduceLROnPlateau
        },
        'scheduler_factor': {
            'values': [0.1, 0.2, 0.5]  # Reduction factor for ReduceLROnPlateau
        },
        'scheduler_min_lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-7,
            'max': 1e-5
        },
        'scheduler_t_max': {
            'values': [5, 10]  # For CosineAnnealingLR, cycles before reset
        },
        
        # Early stopping parameters
        'early_stop_patience': {
            'value': 5  # Fixed to avoid excessive training
        },
        
        # Dataset parameters
        'sample_rate': {
            'value': 16000  # Fixed for consistency
        },
        'split_ratio': {
            'value': 0.9  # Train/test split ratio
        },
        'num_augmentations': {
            'values': [0, 1, 2]  # Reduced maximum augmentations to save memory
        },
        
        # System parameters
        'device': {
            'value': 'cuda'  # Fixed to use GPU
        },
        'num_workers': {
            'value': 4  # Number of data loading workers
        },
        'seed': {
            'value': 42  # Fixed for reproducibility
        }
    }
    
    # Model type specific configuration
    if model_type == 'raw':
        # Raw audio specific parameters
        model_specific_params = {
            'model_type': {'value': 'raw'}  # Fixed to raw audio model
        }
    else:  # spectrogram
        # Spectrogram specific parameters
        model_specific_params = {
            'model_type': {'value': 'spectrogram'},  # Fixed to spectrogram model
            'n_fft': {
                'values': [256, 400]  # FFT size for spectrogram
            },
            'hop_length': {
                'values': [128, 160]  # Hop length for spectrogram
            },
            'n_mels': {
                'values': [64, 80]  # Number of mel bands
            }
        }
    
    # Create the full sweep configuration
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'best_eval_accuracy',
            'goal': 'maximize'  # We want to maximize accuracy
        },
        'parameters': {**common_params, **model_specific_params}  # Merge common and specific parameters
    }
    
    return sweep_config

def main():
    """
    Main function to create and run the sweep.
    """
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep with wandb")
    parser.add_argument('--count', type=int, default=120, help='Number of runs to perform in the sweep')
    parser.add_argument('--project', type=str, default="mlx7-week-5-urbansound8k-classifier", 
                        help='wandb project name')
    parser.add_argument('--model_type', type=str, choices=['raw', 'spectrogram', 'both'], default='',
                        help='Model type to sweep: raw, spectrogram, or both')
    args = parser.parse_args()
    
    if args.model_type == 'both':
        # Create two sweeps, one for each model type
        raw_sweep_config = create_sweep_config('raw')
        spec_sweep_config = create_sweep_config('spectrogram')
        
        # Launch raw sweep
        raw_sweep_id = wandb.sweep(raw_sweep_config, project=f"{args.project}-raw")
        print(f"Created raw sweep with ID: {raw_sweep_id}")
        
        # Launch spectrogram sweep
        spec_sweep_id = wandb.sweep(spec_sweep_config, project=f"{args.project}-spectrogram")
        print(f"Created spectrogram sweep with ID: {spec_sweep_id}")
        
        # Distribute the runs between the two sweeps
        raw_count = args.count // 2
        spec_count = args.count - raw_count
        
        print(f"Running {raw_count} raw model sweeps...")
        wandb.agent(raw_sweep_id, function=train_model_with_config, count=raw_count)
        
        print(f"Running {spec_count} spectrogram model sweeps...")
        wandb.agent(spec_sweep_id, function=train_model_with_config, count=spec_count)
        
        print(f"Sweep completed with {raw_count} raw model runs and {spec_count} spectrogram model runs")
    else:
        # Create a sweep for the specified model type
        sweep_config = create_sweep_config(args.model_type)
        sweep_id = wandb.sweep(sweep_config, project=f"{args.project}-{args.model_type}")
        print(f"Created {args.model_type} sweep with ID: {sweep_id}")
        
        # Run the sweep
        wandb.agent(sweep_id, function=train_model_with_config, count=args.count)
        print(f"Sweep completed after {args.count} {args.model_type} model runs")

if __name__ == "__main__":
    main()