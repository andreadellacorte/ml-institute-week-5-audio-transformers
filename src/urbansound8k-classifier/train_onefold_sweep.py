import wandb
import torch
import argparse
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm  # Import tqdm for progress bars

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINTS_DATA_DIR
from dataset import get_datasets
from model import AudioClassifier
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
        
        # Set random seed
        set_seed(config.seed)
        
        # Create datasets
        print("Loading datasets...")
        train_dataset, test_dataset = get_datasets(
            sample_rate=config.sample_rate,
            target_length=config.sample_rate * 4,  # 4 seconds of audio
            augment=config.use_augmentation,
            max_length=None,  # Use all available data
            split_ratio=config.split_ratio
        )
        
        print(f"Datasets loaded. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Log dataset info to wandb
        wandb.config.update({
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "timestamp": timestamp
        })
        
        # Create model
        classifier = AudioClassifier(
            num_classes=10,
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            feature_extractor_base_filters=config.feature_extractor_base_filters,
            learning_rate=config.learning_rate
        )
        
        # Move model to device
        classifier.model.to(device)
        
        # Enable mixed precision
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Initialize gradient scaler for mixed precision training
        use_mixed_precision = (device.type == "cuda")
        scaler = GradScaler(enabled=use_mixed_precision)
        
        # Log model graph to wandb
        wandb.watch(classifier.model, log="all", log_freq=10)
        
        # Training loop
        best_eval_accuracy = 0.0
        best_eval_loss = float('inf')
        early_stop_patience = config.early_stop_patience
        early_stop_counter = 0
        
        print(f"Training for {config.num_epochs} epochs...")
        
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            
            # Training phase
            classifier.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Add tqdm progress bar for training loop
            progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{config.num_epochs}", 
                               leave=True, ncols=100)
            
            for i, batch in enumerate(progress_bar):
                # Move data to device
                waveform = batch['waveform'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                # Mixed precision training - updated to use the new API format
                with autocast('cuda', enabled=use_mixed_precision):
                    # Forward pass
                    logits = classifier.model(waveform)
                    loss = classifier.criterion(logits, labels)
                    
                    # Scale loss for gradient accumulation
                    if config.gradient_accumulation_steps > 1:
                        loss = loss / config.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Only step optimizer after accumulating gradients
                if (i + 1) % config.gradient_accumulation_steps == 0:
                    # Update weights with gradient scaling for mixed precision
                    scaler.step(classifier.optimizer)
                    scaler.update()
                    classifier.optimizer.zero_grad(set_to_none=True)
                
                # Calculate metrics
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    batch_correct = (preds == labels).sum().item()
                    batch_total = labels.size(0)
                    train_correct += batch_correct
                    train_total += batch_total
                
                # Scale loss back for reporting if needed
                current_loss = loss.item()
                if config.gradient_accumulation_steps > 1:
                    current_loss *= config.gradient_accumulation_steps
                
                train_loss += current_loss
                
                # Update progress bar with current metrics
                batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0
                progress_bar.set_postfix({
                    'loss': f"{current_loss:.4f}", 
                    'acc': f"{batch_accuracy:.4f}"
                })
            
            # Calculate epoch metrics
            epoch_loss = train_loss / len(train_loader)
            epoch_accuracy = train_correct / train_total if train_total > 0 else 0
            
            # Log training metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy,
                "learning_rate": classifier.optimizer.param_groups[0]['lr']
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
            
            # Update learning rate based on validation loss
            if hasattr(classifier, 'scheduler'):
                classifier.scheduler.step(eval_metrics['loss'])
            
            # Print epoch results
            print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            print(f"Evaluation - Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}")
            
            # Save best model based on validation accuracy with accuracy in filename
            if eval_metrics['accuracy'] > best_eval_accuracy:
                best_eval_accuracy = eval_metrics['accuracy']
                # Format accuracy as a percentage in the filename
                acc_str = f"{best_eval_accuracy:.2f}".replace('.', '_')
                best_model_path = CHECKPOINTS_DATA_DIR / f"acc_{acc_str}_model_{run.id}.pt"
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
        final_metrics = classifier.evaluate(test_loader)
        print(f"\nFinal Evaluation - Loss: {final_metrics['loss']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")
        
        # Log final metrics
        wandb.log({
            "final_eval_loss": final_metrics['loss'],
            "final_eval_accuracy": final_metrics['accuracy'],
            "best_eval_accuracy": best_eval_accuracy
        })
        
        # Return metric to optimize (higher accuracy is better)
        return final_metrics['accuracy']

def create_sweep_config():
    """
    Create a configuration for the wandb sweep with Bayesian optimization.
    """
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'final_eval_accuracy',
            'goal': 'maximize'  # We want to maximize accuracy
        },
        'parameters': {
            # Model architecture parameters
            'd_model': {
                'values': [64, 128, 256, 512]  # Model dimension
            },
            'nhead': {
                'values': [2, 4, 8]  # Number of attention heads
            },
            'num_encoder_layers': {
                'values': [2, 3, 4]  # Number of encoder layers
            },
            'dim_feedforward': {
                'values': [64, 128, 256, 512]  # Feed-forward dimension
            },
            'dropout': {
                'values': [0.1, 0.2, 0.3]  # Dropout rate
            },
            'feature_extractor_base_filters': {
                'values': [16, 32, 64]  # Base filters for feature extractor
            },
            
            # Training parameters
            'batch_size': {
                'values': [256, 512]  # Batch size
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3
            },
            'num_epochs': {
                'value': 10  # Fixed to avoid wasting resources
            },
            'gradient_accumulation_steps': {
                'values': [1, 2, 4, 8]  # Gradient accumulation steps
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
            'use_augmentation': {
                'values': [True, False]  # Whether to use data augmentation
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
    }
    
    return sweep_config

def main():
    """
    Main function to create and run the sweep.
    """
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep with wandb")
    parser.add_argument('--count', type=int, default=120, help='Number of runs to perform in the sweep')
    parser.add_argument('--project', type=str, default="mlx7-week-5-urbansound8k-sweep", 
                        help='wandb project name')
    args = parser.parse_args()
    
    # Create the sweep
    sweep_config = create_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"Created sweep with ID: {sweep_id}")
    
    # Run the sweep
    wandb.agent(sweep_id, function=train_model_with_config, count=args.count)
    print(f"Sweep completed after {args.count} runs")

if __name__ == "__main__":
    main()