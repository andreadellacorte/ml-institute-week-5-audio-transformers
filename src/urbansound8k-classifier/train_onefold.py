from model import AudioClassifier
import wandb
import argparse
import torch
from datetime import datetime
from src.config import CHECKPOINTS_DATA_DIR

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

def get_config():
    """
    Get configuration dictionary from arguments
    """
    # Default configuration
    config = {
        # Model parameters
        "d_model": 256,
        "nhead": 4,
        "num_encoder_layers": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "feature_extractor_base_filters": 32,
        "split_ratio": 0.8,
        
        # Training parameters
        "batch_size": 128,
        "num_epochs": 5,
        "learning_rate": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        
        # Dataset parameters
        "sample_rate": 16000,
        "target_length": 16000 * 4,
        "use_augmentation": False
    }
            
    return config

if __name__ == "__main__":
    # Get config from args
    config = get_config()
    
    # Example usage
    from torch.utils.data import DataLoader
    from dataset import get_datasets
    
    # Initialize wandb
    run = setup_wandb(config)
    
    # Create datasets with timeout protection for loading only
    print("Loading datasets...")
    try:
        
        train_dataset, test_dataset = get_datasets(
            sample_rate=config["sample_rate"],
            target_length=config["target_length"],
            augment=config["use_augmentation"],
            max_length=None,  # Use all available data
            split_ratio=config["split_ratio"],
        )

        print(f"Datasets loaded. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True, 
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config["batch_size"], 
            shuffle=False, 
            num_workers=2
        )
        
        # Log dataset info to wandb
        wandb.config.update({
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        })
        
        # Initialize model
        device = torch.device(config["device"])
        classifier = AudioClassifier(
            num_classes=10,  # UrbanSound8K has 10 classes
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_encoder_layers=config["num_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"]
        )
        classifier.model.to(device)
        
        # Log model graph to wandb
        wandb.watch(classifier.model, log="all", log_freq=10)
        
        print(f"Training for {config['num_epochs']} epochs (using all available data)...")
        try:
            # Training loop
            for epoch in range(config['num_epochs']):
                print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                num_batches = 0
                
                for i, batch in enumerate(train_loader):
                    # Move data to device
                    batch['waveform'] = batch['waveform'].to(device)
                    batch['label'] = batch['label'].to(device)
                    
                    # Train step
                    metrics = classifier.train_step(batch)
                    epoch_loss += metrics['loss']
                    epoch_accuracy += metrics['accuracy']
                    num_batches += 1
                    
                    # Log individual steps to wandb
                    if (i+1) % 10 == 0:
                        wandb.log({
                            "step_loss": metrics['loss'],
                            "step_accuracy": metrics['accuracy'],
                            "step": i + epoch * len(train_loader)
                        })
                    
                    # Print progress every 10 batches
                    if (i+1) % 10 == 0:
                        print(f"  Step {i+1}/{len(train_loader)}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
                
                # Print epoch summary
                avg_loss = epoch_loss / num_batches
                avg_accuracy = epoch_accuracy / num_batches
                print(f"Epoch {epoch+1} complete - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
                
                # Log epoch metrics to wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_accuracy": avg_accuracy,
                    "learning_rate": classifier.optimizer.param_groups[0]['lr']
                })
            
                # Evaluate on test data after each epoch
                classifier.model.eval()
                eval_metrics = classifier.evaluate(test_loader)
                print(f"Evaluation: Loss = {eval_metrics['loss']:.4f}, Accuracy = {eval_metrics['accuracy']:.4f}")
                
                # Log evaluation metrics to wandb
                wandb.log({
                    "epoch": epoch,
                    "eval_loss": eval_metrics['loss'],
                    "eval_accuracy": eval_metrics['accuracy']
                })
            
            # Final evaluation
            print("\nFinal Evaluation...")
            final_metrics = classifier.evaluate(test_loader)
            print(f"Final Evaluation: Loss = {final_metrics['loss']:.4f}, Accuracy = {final_metrics['accuracy']:.4f}")
            
            # Log final metrics to wandb
            wandb.log({
                "final_eval_loss": final_metrics['loss'],
                "final_eval_accuracy": final_metrics['accuracy']
            })
            
            # Make a sample prediction
            sample = next(iter(test_loader))
            waveform = sample['waveform'][0].unsqueeze(0).to(device)  # Get first waveform
            prediction = classifier.predict(waveform)
            print(f"Prediction: Class {prediction['class_id']} with confidence {prediction['confidence']:.4f}")
            
            # Create the directory if it doesn't exist
            CHECKPOINTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = CHECKPOINTS_DATA_DIR / f"model_{wandb.run.id}.pt"
            classifier.save(str(model_path))
            
            # Log model to wandb
            wandb.save(str(model_path))
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Stopping gracefully...")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()
            print("Example complete")
    
    except TimeoutError:
        print("Dataset loading timed out. The dataset might be temporarily unavailable.")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up...")
        wandb.finish()