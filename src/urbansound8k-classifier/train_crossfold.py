import argparse
import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINTS_DATA_DIR, FIGURES_DIR
from dataset import UrbanSoundDataset
from model import AudioClassifier

# Set default seed for reproducibility
def set_seed(seed=42):
    """
    Set seed for all random number generators for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")

# Constants
NUM_FOLDS = 10
SAMPLE_RATE = 16000
TARGET_LENGTH = SAMPLE_RATE * 4  # 4 seconds of audio at 16kHz

# Default configuration dictionary
DEFAULT_CONFIG = {
    # Model parameters
    "d_model": 128,
    "nhead": 4,
    "num_encoder_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "feature_extractor_base_filters": 32,
    
    # Training parameters
    "batch_size": 32,
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "num_workers": 4,
    "augment": False,
    
    # Cross-validation parameters
    "max_folds": 10,  # Maximum number of folds to use (out of 10)
    "use_folds": None,  # Specific folds to use, if None use all folds up to max_folds
    
    # Dataset information
    "model_type": "encoder_only_transformer",
    "dataset": "urbansound8k",
    "sample_rate": SAMPLE_RATE,
    "target_length": TARGET_LENGTH,
    "num_folds": NUM_FOLDS,
}

# Function to get configuration with device information
def get_config(args=None):
    """
    Get configuration dictionary, optionally overriding defaults with command-line args.
    """
    config = DEFAULT_CONFIG.copy()
    
    if args:
        # Override defaults with args
        for key, value in vars(args).items():
            config[key] = value
    
    # Add device information
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    return config


def setup_wandb(config):
    """
    Initialize Weights & Biases with the given config.
    """
    run = wandb.init(
        project="mlx7-week-5-urbansound8k-classifier",
        config=config,
        job_type="training",
    )
    
    # Create a unique experiment name
    experiment_name = f"encoder-only_{config['d_model']}d_{config['num_encoder_layers']}layers_{wandb.run.id}"
    wandb.run.name = experiment_name
    
    return run


def save_fold_results_plot(fold_train_losses, fold_eval_losses, fold_eval_accuracies):
    """
    Save a plot of training and evaluation losses/accuracies for each fold.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    for fold in range(len(fold_train_losses)):
        plt.plot(fold_train_losses[fold], label=f'Fold {fold+1} Train Loss')
    plt.title('Training Loss per Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot evaluation metrics
    plt.subplot(2, 1, 2)
    for fold in range(len(fold_eval_losses)):
        plt.plot(fold_eval_losses[fold], label=f'Fold {fold+1} Eval Loss')
        plt.plot(fold_eval_accuracies[fold], label=f'Fold {fold+1} Accuracy', linestyle='--')
    plt.title('Evaluation Metrics per Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig(FIGURES_DIR / f"urbansound8k_folds_results_{timestamp}.png")
    plt.close()


def load_fold_datasets(config):
    """
    Load datasets for each fold based on configuration.
    
    Args:
        config: Configuration dictionary with fold settings
        
    Returns:
        List of datasets for each fold
    """
    fold_datasets = []
    
    # Determine which folds to use
    if config["use_folds"] is not None:
        folds_to_use = config["use_folds"]
    else:
        folds_to_use = range(1, config["max_folds"] + 1)
    
    for fold in folds_to_use:
        print(f"Loading fold {fold}...")
        fold_dataset = UrbanSoundDataset(
            split="train",  # We'll handle the splits manually for cross-validation
            sample_rate=config["sample_rate"],
            target_length=config["target_length"],
            fold=fold,
            augment=config["augment"]
        )
        fold_datasets.append(fold_dataset)
        
    return fold_datasets


def train_model_with_cross_validation(config):
    """
    Train the audio transformer model using k-fold cross-validation.
    
    Args:
        config: Dictionary of hyperparameters
    """
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed()
    
    # Load all fold datasets
    fold_datasets = load_fold_datasets(config)
    active_folds = len(fold_datasets)
    print(f"Loaded {active_folds} folds with sizes: {[len(ds) for ds in fold_datasets]}")
    
    # Initialize arrays to store results
    all_fold_train_losses = []
    all_fold_eval_losses = []
    all_fold_eval_accuracies = []
    
    # For each fold, use it as validation and the rest as training
    for test_fold_idx in range(active_folds):
        print(f"\n{'='*40}\nTraining with fold {test_fold_idx+1}/{active_folds} as validation\n{'='*40}")
        
        # Create train and validation datasets
        train_datasets = [fold_datasets[i] for i in range(active_folds) if i != test_fold_idx]
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = fold_datasets[test_fold_idx]
        
        print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
        
        # Create data loaders (no shuffling as per requirement)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"]
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"]
        )
        
        # Initialize model
        classifier = AudioClassifier(
            num_classes=10,  # UrbanSound8K has 10 classes
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_encoder_layers=config["num_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            feature_extractor_base_filters=config["feature_extractor_base_filters"],
            learning_rate=config["learning_rate"]
        )
        classifier.model.to(device)
        
        # Store metrics for this fold
        train_losses = []
        eval_losses = []
        eval_accuracies = []
        
        # Training loop for specific number of epochs
        for epoch in range(config["num_epochs"]):
            print(f"\nEpoch {epoch+1}/{config['num_epochs']} - Fold {test_fold_idx+1}/{active_folds}")
            
            # Training
            classifier.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                # Move data to device
                waveform = batch['waveform'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                classifier.optimizer.zero_grad()
                logits = classifier.model(waveform)
                loss = classifier.criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                classifier.optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })
            
            # Calculate average epoch loss
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Evaluation
            classifier.model.eval()
            eval_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc="Evaluation")
                for batch in progress_bar:
                    # Move data to device
                    waveform = batch['waveform'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Forward pass
                    logits = classifier.model(waveform)
                    loss = classifier.criterion(logits, labels)
                    
                    # Update statistics
                    eval_loss += loss.item() * waveform.size(0)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += waveform.size(0)
            
            # Calculate average evaluation metrics
            avg_eval_loss = eval_loss / total
            accuracy = correct / total
            eval_losses.append(avg_eval_loss)
            eval_accuracies.append(accuracy)
            
            # Update learning rate scheduler
            classifier.scheduler.step(avg_eval_loss)
            
            # Print statistics
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_eval_loss:.4f} | Accuracy: {accuracy:.4f}")
            
            # Log to wandb
            wandb.log({
                f"fold_{test_fold_idx+1}/train_loss": avg_train_loss,
                f"fold_{test_fold_idx+1}/val_loss": avg_eval_loss,
                f"fold_{test_fold_idx+1}/accuracy": accuracy,
                f"fold_{test_fold_idx+1}/learning_rate": classifier.optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })
        
        # Save model checkpoint for this fold
        checkpoint_path = CHECKPOINTS_DATA_DIR / f"model_fold_{test_fold_idx+1}.pt"
        CHECKPOINTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        classifier.save(checkpoint_path)
        
        # Log model to wandb
        wandb.save(str(checkpoint_path))
        
        # Store fold results
        all_fold_train_losses.append(train_losses)
        all_fold_eval_losses.append(eval_losses)
        all_fold_eval_accuracies.append(eval_accuracies)
    
    # Calculate and log cross-validation results
    mean_train_loss = np.mean([losses[-1] for losses in all_fold_train_losses])
    mean_eval_loss = np.mean([losses[-1] for losses in all_fold_eval_losses])
    mean_accuracy = np.mean([accs[-1] for accs in all_fold_eval_accuracies])
    std_accuracy = np.std([accs[-1] for accs in all_fold_eval_accuracies])
    
    print("\n" + "="*60)
    print(f"Cross-Validation Results (over {active_folds} folds):")
    print(f"Mean Train Loss: {mean_train_loss:.4f}")
    print(f"Mean Evaluation Loss: {mean_eval_loss:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print("="*60)
    
    # Final wandb logging
    wandb.log({
        "cross_val/mean_train_loss": mean_train_loss,
        "cross_val/mean_eval_loss": mean_eval_loss,
        "cross_val/mean_accuracy": mean_accuracy,
        "cross_val/std_accuracy": std_accuracy,
        "cross_val/num_folds": active_folds
    })
    
    # Save fold results plot
    save_fold_results_plot(all_fold_train_losses, all_fold_eval_losses, all_fold_eval_accuracies)
    
    # Upload the plot to wandb
    wandb.log({"fold_results": wandb.Image(str(FIGURES_DIR / f"urbansound8k_folds_results_{time.strftime('%Y%m%d_%H%M%S')}.png"))})


def main():
    parser = argparse.ArgumentParser(description="Train audio transformer model on UrbanSound8K dataset")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=DEFAULT_CONFIG["d_model"], help="Model dimension")
    parser.add_argument("--nhead", type=int, default=DEFAULT_CONFIG["nhead"], help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=DEFAULT_CONFIG["num_encoder_layers"], 
                       help="Number of transformer encoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=DEFAULT_CONFIG["dim_feedforward"], 
                       help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"], help="Dropout rate")
    parser.add_argument("--feature_extractor_base_filters", type=int, 
                       default=DEFAULT_CONFIG["feature_extractor_base_filters"],
                       help="Base filters for feature extractor")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG["num_epochs"], help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"], help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"], 
                       help="Number of data loader workers")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    
    # Cross-validation parameters
    parser.add_argument("--max_folds", type=int, default=DEFAULT_CONFIG["max_folds"],
                       help="Maximum number of folds to use (1-10)")
    parser.add_argument("--folds", type=str, default=None,
                       help="Specific folds to use (comma-separated, e.g., '1,3,5,7,9')")
    
    # Add option to use faster configuration for testing
    parser.add_argument("--fast", action="store_true", 
                       help="Use a smaller, faster configuration for testing")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle fast mode by updating args
    if args.fast:
        args.d_model = 64
        args.num_encoder_layers = 2
        args.dim_feedforward = 256
        args.num_epochs = 5
    
    # Convert folds string to list if provided
    if args.folds is not None:
        try:
            args.use_folds = [int(fold) for fold in args.folds.split(',')]
            print(f"Using specific folds: {args.use_folds}")
        except ValueError:
            print(f"Error parsing folds: {args.folds}. Format should be comma-separated integers (e.g., '1,3,5,7,9')")
            return
    
    # Get config from defaults and args
    config = get_config(args)
    
    # Initialize wandb
    run = setup_wandb(config)
    
    try:
        # Train with cross-validation
        train_model_with_cross_validation(config)
        
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Finish wandb run
        wandb.finish()


if __name__ == "__main__":
    main()