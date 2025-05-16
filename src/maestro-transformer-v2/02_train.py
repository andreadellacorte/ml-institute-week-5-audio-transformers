
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
import random
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Local imports
from model import SpectrogramToMIDITransformer, MIDIGenerationLoss
from dataset import create_maestro_dataloaders, MaestroMIDISpectrogramDataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_DATA_DIR, CHECKPOINTS_DATA_DIR
import miditoolkit
from miditok import REMI


# Configuration
class TrainingConfig:
    """Training configuration"""
    
    # Model parameters
    n_mels: int = 128
    vocab_size: int = 282  # Matches REMI tokenizer vocabulary size
    d_model: int = 512
    nhead: int = 8
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 1024
    pad_token_id: int = 0
    
    # Training parameters
    batch_size: int = 16
    eval_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 50
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    
    # Data parameters
    num_workers: int = 4
    max_train_examples: Optional[int] = None  # Set to None to use all examples
    
    # Evaluation parameters
    eval_every: int = 1  # Evaluate every N steps
    generate_midi_every: int = 1  # Generate MIDI every N evals
    num_eval_examples: int = 16  # Number of examples to use for evaluation
    
    # Generation parameters
    generation_temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    max_generation_length: int = 1024
    
    # Paths
    maestro_processed_dir: Path = PROCESSED_DATA_DIR / "ddPn08-maestro-v3.0.0"
    checkpoint_dir: Path = CHECKPOINTS_DATA_DIR / "maestro-transformer-v2"
    
    # Tokenizationres
    start_token: int = 1
    end_token: int = 2
    
    # Random seed
    seed: int = 42


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    step: int,
    loss: float,
    filename: str
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss
    }
    torch.save(checkpoint, filename)


def load_checkpoint(
    filename: str, 
    model: nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, int, float]:
    """Load model from checkpoint."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['step'], checkpoint['loss']


def create_remi_tokenizer():
    """
    Create and return a REMI tokenizer instance.
    The tokenizer is configured to match the one used during data preprocessing.
    
    Returns:
        REMI tokenizer instance
    """
    # Initialize the REMI tokenizer with the same configuration as in data preprocessing
    tokenizer = REMI()
    return tokenizer


def tokens_to_midi_file(
    tokens: torch.Tensor,
    tokenizer,  # REMI tokenizer instance
    output_path: str,
    start_token_id: int = 1,
    end_token_id: int = 2,
    pad_token_id: int = 0
) -> None:
    """
    Convert generated tokens to a MIDI file using the REMI tokenizer.
    
    Args:
        tokens: Tensor of token IDs [seq_len] or [batch_size, seq_len]
        tokenizer: REMI tokenizer instance
        output_path: Path to save the MIDI file
        start_token_id: ID of the start token to remove
        end_token_id: ID of the end token to remove
        pad_token_id: ID of the padding token to remove
    """
    try:
        # First, create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Make sure tokens is 1D (take first sample if batched)
        if len(tokens.shape) > 1:
            tokens = tokens[0]  # Take first sample in batch
        
        # Convert to numpy array
        tokens = tokens.cpu().numpy()
        
        # Filter out special tokens (start, end, pad)
        filtered_tokens = []
        for token in tokens:
            if token != start_token_id and token != end_token_id and token != pad_token_id:
                filtered_tokens.append(token)
        
        # Debug info
        print(f"  - Original token count: {len(tokens)}")
        print(f"  - Filtered token count: {len(filtered_tokens)}")
        
        # Check if we have any tokens left
        if not filtered_tokens:
            print(f"  - Warning: No valid tokens remaining after filtering special tokens")
            # Create an empty MIDI file as fallback
            print(f"  - Creating empty MIDI file at {output_path}")
            midi_obj = miditoolkit.MidiFile()
            midi_obj.instruments.append(miditoolkit.Instrument(program=0, is_drum=False, name="Empty"))
            midi_obj.dump(output_path)
            return
        
        # Convert tokens back to MIDI using the tokenizer
        print(f"  - Decoding {len(filtered_tokens)} tokens to MIDI")
        midi_obj = tokenizer.decode(filtered_tokens)
        
        # Save the MIDI file
        print(f"  - Writing MIDI to {output_path}")
        midi_obj.dump(output_path)
        
        # Verify file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"  - Successfully saved MIDI file to {output_path} ({file_size} bytes)")
        else:
            print(f"  - Error: MIDI file not found at {output_path} after writing")
        
    except Exception as e:
        print(f"Error converting tokens to MIDI: {e}")
        import traceback
        traceback.print_exc()
        # Create an empty MIDI file as fallback
        try:
            print(f"  - Creating empty MIDI file as fallback at {output_path}")
            midi_obj = miditoolkit.MidiFile()
            midi_obj.instruments.append(miditoolkit.Instrument(program=0, is_drum=False, name="Empty"))
            midi_obj.dump(output_path)
        except Exception as e2:
            print(f"  - Failed to create empty MIDI fallback: {e2}")


class WarmupLinearLR:
    """
    Learning rate scheduler with linear warmup and linear decay.
    """
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        warmup_steps: int, 
        total_steps: int,
        min_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, step: int) -> None:
        """Update the learning rate based on the current step."""
        if step < self.warmup_steps:
            # Linear warmup
            lr_factor = step / max(1, self.warmup_steps)
        else:
            # Linear decay
            lr_factor = max(
                self.min_lr_ratio,
                (self.total_steps - step) / max(1, (self.total_steps - self.warmup_steps))
            )
            
        for idx, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.initial_lrs[idx] * lr_factor


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupLinearLR,
    criterion: nn.Module,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
) -> Tuple[float, int]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # Get batch data
        spectrograms = batch['spectrogram'].to(device)
        midi_tokens = batch['midi_tokens'].to(device)
        padding_mask = (midi_tokens == config.pad_token_id)
        
        # Forward pass
        logits = model(spectrograms, midi_tokens, padding_mask)
        
        # Calculate loss (shift targets so we predict next token)
        # Here we're using the current token to predict the next one (auto-regressive)
        loss = criterion(logits[:, :-1, :], midi_tokens[:, 1:])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate
        scheduler.step(global_step)
        
        # Update metrics
        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1
        global_step += 1
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{batch_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        # Log to wandb
        wandb.log({
            "train/loss": batch_loss,
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch,
            "global_step": global_step
        }, step=global_step)
        
        # Evaluate model periodically
        if global_step % config.eval_every == 0:
            eval_loss = evaluate(model, eval_loader, criterion, device, config, global_step)
            wandb.log({"eval/loss": eval_loss}, step=global_step)

            run_name = wandb.run.name
            
            # Save checkpoint
            checkpoint_dir = config.checkpoint_dir / run_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
            checkpoint_path = checkpoint_dir / f"step_{global_step}.pt"
            save_checkpoint(model, optimizer, epoch, global_step, eval_loss, str(checkpoint_path))
            
            # Log best checkpoint to wandb
            wandb.save(str(checkpoint_path))
            
            # Generate MIDI examples for visualization
            if global_step % config.generate_midi_every == 0:
                generate_and_save_midi_examples(model, eval_loader, device, config, global_step)
            
            # Switch back to training mode
            model.train()
    
    avg_loss = total_loss / num_batches
    return avg_loss, global_step


def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainingConfig,
    global_step: int,
) -> float:
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(eval_loader, desc="Evaluation")
        
        for batch in progress_bar:
            # Get batch data - limited to a subset for evaluation
            if num_batches >= config.num_eval_examples / config.eval_batch_size:
                break
                
            spectrograms = batch['spectrogram'].to(device)
            midi_tokens = batch['midi_tokens'].to(device)
            padding_mask = (midi_tokens == config.pad_token_id)
            
            # Forward pass
            logits = model(spectrograms, midi_tokens, padding_mask)
            
            # Calculate loss (shift targets so we predict next token)
            loss = criterion(logits[:, :-1, :], midi_tokens[:, 1:])
            
            # Update metrics
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"eval_loss": f"{batch_loss:.4f}"})
    
    avg_loss = total_loss / num_batches
    print(f"Evaluation completed at step {global_step}. Loss: {avg_loss:.4f}")
    return avg_loss


def generate_and_save_midi_examples(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    global_step: int,
    num_examples: int = 4
) -> None:
    """Generate and save MIDI examples for visualization."""
    # Create directory for generated MIDI files
    run_name = wandb.run.name
    
    print(f"\n=== Starting MIDI generation at step {global_step} ===")
    
    # Save to checkpoint directory
    midi_dir = config.checkpoint_dir / run_name / "generated_midi" / f"step_{global_step}"
    midi_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving MIDI files to:")
    print(f"  - {midi_dir}")
    
    # Create REMI tokenizer for decoding generated tokens
    tokenizer = create_remi_tokenizer()
    
    # Get a batch of examples
    model.eval()
    batch = next(iter(eval_loader))
    spectrograms = batch['spectrogram'][:num_examples].to(device)
    original_tokens = batch['midi_tokens'][:num_examples].to(device)
    sample_ids = batch['sample_ids'][:num_examples]
    
    print(f"Generating {num_examples} MIDI examples at step {global_step}")
    
    # Generate tokens
    with torch.no_grad():
        generated_outputs = model(
            spectrograms,
            generate=True,
            max_length=config.max_generation_length,
            temperature=config.generation_temperature,
            top_k=config.top_k,
            top_p=config.top_p
        )
        generated_tokens = generated_outputs["tokens"]
    
    # Save original and generated MIDI files
    for i, (sample_id, orig_token, gen_token) in enumerate(zip(sample_ids, original_tokens, generated_tokens)):
        # Get paths for the files
        orig_path = midi_dir / f"{sample_id}_original.midi"
        gen_path = midi_dir / f"{sample_id}_generated.midi"
        
        # Save the original MIDI file from the tokenized version
        original_midi_path = config.maestro_processed_dir / "eval" / f"{sample_id}.midi"
        if original_midi_path.exists():
            shutil.copy(original_midi_path, orig_path)
        else:
            original_midi_path = config.maestro_processed_dir / "train" / f"{sample_id}.midi"
            if original_midi_path.exists():
                shutil.copy(original_midi_path, orig_path)
            else:
                # If the original MIDI file doesn't exist, convert tokens to MIDI
                try:
                    tokens_to_midi_file(
                        tokens=orig_token,
                        tokenizer=tokenizer,
                        output_path=str(orig_path),
                        start_token_id=config.start_token,
                        end_token_id=config.end_token,
                        pad_token_id=config.pad_token_id
                    )
                except Exception as e:
                    print(f"Error creating original MIDI file: {e}")
        
        # Convert generated tokens to MIDI
        try:
            print(f"Generating MIDI for sample {sample_id}...")
            
            print(f"  - Saving to: {gen_path}")
            tokens_to_midi_file(
                tokens=gen_token,
                tokenizer=tokenizer,
                output_path=str(gen_path),
                start_token_id=config.start_token,
                end_token_id=config.end_token,
                pad_token_id=config.pad_token_id
            )
            
            # Verify file was created
            if not os.path.exists(str(gen_path)):
                print(f"  - Warning: File not found at {gen_path} after save attempt")
                
            # Log these MIDI files to wandb
            wandb.log({
                f"midi_examples/original_{i}": wandb.Audio(str(orig_path), sample_rate=44100, caption=f"Original {sample_id}"),
                f"midi_examples/generated_{i}": wandb.Audio(str(gen_path), sample_rate=44100, caption=f"Generated {sample_id}")
            }, step=global_step)
            
            print(f"  - Successfully saved MIDI file for sample {sample_id}")
            
        except Exception as e:
            print(f"Error saving generated MIDI file: {e}")
            import traceback
            traceback.print_exc()
    
    # Also log some token sequences as text (for easy inspection)
    for i, (sample_id, gen_token) in enumerate(zip(sample_ids[:2], generated_tokens[:2])):
        token_text = " ".join([str(t.item()) for t in gen_token[:50]])
        wandb.log({
            f"token_sequences/sample_{i}": wandb.Html(f"<pre>Sample ID: {sample_id}\nTokens: {token_text}</pre>")
        }, step=global_step)


def main(args):
    # Setup configuration
    config = TrainingConfig()
    config.batch_size = args.batch_size if args.batch_size else config.batch_size
    config.learning_rate = args.lr if args.lr else config.learning_rate
    config.epochs = args.epochs if args.epochs else config.epochs
    config.eval_every = args.eval_every if args.eval_every else config.eval_every
    config.generate_midi_every = args.generate_midi_every if args.generate_midi_every else config.generate_midi_every
    
    # Create REMI tokenizer
    print("Initializing REMI tokenizer...")
    tokenizer = create_remi_tokenizer()
    
    # Get tokenizer vocabulary size
    vocab_size = len(tokenizer.vocab)
    if vocab_size != config.vocab_size:
        print(f"Warning: Updating model vocabulary size from {config.vocab_size} to {vocab_size} to match REMI tokenizer")
        config.vocab_size = vocab_size
        
    # Setup W&B
    run_name = f"maestro_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="maestro-transformer",
        name=run_name,
        config={
            "architecture": "SpectrogramToMIDITransformer",
            "dataset": "MAESTRO",
            "tokenizer": "REMI",
            "vocab_size": config.vocab_size,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "d_model": config.d_model,
            "num_decoder_layers": config.num_decoder_layers,
            "eval_every": config.eval_every,
            "generate_midi_every": config.generate_midi_every,
        }
    )
    
    # Set random seeds
    set_seed(config.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, eval_loader = create_maestro_dataloaders(
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        max_seq_length=config.max_seq_len,
        num_workers=config.num_workers,
        random_seed=config.seed
    )
    
    # Calculate training steps
    train_examples = len(train_loader.dataset)
    steps_per_epoch = (train_examples + config.batch_size - 1) // config.batch_size
    total_steps = steps_per_epoch * config.epochs
    
    print(f"Training examples: {train_examples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    
    # Create model
    print("Creating model...")
    model = SpectrogramToMIDITransformer(
        n_mels=config.n_mels,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        pad_token_id=config.pad_token_id
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = WarmupLinearLR(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps
    )
    
    # Create loss function
    criterion = MIDIGenerationLoss(ignore_index=config.pad_token_id)
    
    # Training loop
    global_step = 0
    best_eval_loss = float('inf')
    
    # Check for checkpoint
    resume_from = args.resume
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model, optimizer, start_epoch, global_step, _ = load_checkpoint(
            resume_from, model, optimizer
        )
    else:
        start_epoch = 0
    
    print("Starting training...")
    for epoch in range(start_epoch, config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Train for one epoch
        train_loss, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            config=config,
            epoch=epoch,
            global_step=global_step
        )
        
        print(f"Epoch {epoch+1} completed. Train loss: {train_loss:.4f}")
        
        # Save checkpoint at the end of each epoch
        run_name = wandb.run.name
        epoch_checkpoint_dir = config.checkpoint_dir / run_name
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        checkpoint_path = epoch_checkpoint_dir / f"epoch_{epoch+1}.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            loss=train_loss,
            filename=str(checkpoint_path)
        )
    
    # Final evaluation
    final_eval_loss = evaluate(
        model=model,
        eval_loader=eval_loader,
        criterion=criterion,
        device=device,
        config=config,
        global_step=global_step
    )
    
    print(f"Training completed. Final evaluation loss: {final_eval_loss:.4f}")
    
    # Save final model
    run_name = wandb.run.name
    final_checkpoint_dir = config.checkpoint_dir / run_name
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    final_path = final_checkpoint_dir / "final_model.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.epochs,
        step=global_step,
        loss=final_eval_loss,
        filename=str(final_path)
    )
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spectrogram-to-MIDI Transformer")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--eval-every", type=int, help="Evaluate every N steps")
    parser.add_argument("--generate-midi-every", type=int, help="Generate MIDI every N evaluation steps")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    main(args)
