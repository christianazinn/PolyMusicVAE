import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import argparse
import os
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

from model import MusicVAE


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    
    Args:
        batch: List of (sequence, length) tuples
        
    Returns:
        sequences: (batch_size, max_seq_len) padded sequences
        lengths: (batch_size,) actual sequence lengths
    """
    sequences, lengths = zip(*batch)
    
    # Convert to tensors
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    # Pad sequences to same length
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    
    for seq in sequences:
        if len(seq) < max_len:
            # Pad with pad_token_id (assuming 0)
            padded = seq + [0] * (max_len - len(seq))
        else:
            padded = seq
        padded_sequences.append(padded)
    
    sequences = torch.tensor(padded_sequences, dtype=torch.long)
    
    return sequences, lengths


def get_kl_weight(step, total_steps, schedule='cosine', max_beta=1.0):
    """
    KL annealing schedule for β-VAE training.
    
    Args:
        step: Current training step
        total_steps: Total training steps
        schedule: Annealing schedule ('linear', 'cosine', 'cyclical')
        max_beta: Maximum β value
    """
    if schedule == 'linear':
        return min(max_beta, step / (total_steps * 0.5))
    elif schedule == 'cosine':
        return max_beta * (1 - np.cos(np.pi * min(step / (total_steps * 0.5), 1.0))) / 2
    elif schedule == 'cyclical':
        cycle_length = total_steps // 4
        cycle_progress = (step % cycle_length) / cycle_length
        return max_beta * (1 - np.cos(2 * np.pi * cycle_progress)) / 2
    else:
        return max_beta


def evaluate_model(model, dataloader, device, max_batches=None):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (sequences, lengths) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            outputs = model(sequences, lengths, target_sequences=sequences, sample=True)
            
            # Compute loss
            loss, recon_loss, kl_loss = model.compute_loss(
                outputs['logits'], sequences, outputs['latent_dist'], beta=1.0
            )
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
    
    return {
        'val_loss': total_loss / num_batches,
        'val_recon_loss': total_recon_loss / num_batches,
        'val_kl_loss': total_kl_loss / num_batches
    }


def save_checkpoint(model, optimizer, scheduler, step, loss, save_dir, filename=None):
    """Save model checkpoint."""
    if filename is None:
        filename = f'checkpoint_step_{step}.pt'
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    step = checkpoint['step']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: step {step}, loss {loss:.4f}")
    return step, loss


def generate_samples(model, device, num_samples=4, max_length=256, temperature=1.0):
    """Generate sample sequences for logging."""
    model.eval()
    with torch.no_grad():
        samples = model.generate(
            batch_size=num_samples,
            max_length=max_length,
            temperature=temperature
        )
    return samples.cpu().numpy()


def train(args):
    """Main training loop."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"music_vae_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"music_vae_{timestamp}",
            config=vars(args)
        )
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MusicDataset(
        data_path=args.train_data_path,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size
    )
    
    val_dataset = MusicDataset(
        data_path=args.val_data_path,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = MusicVAE(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        latent_dim=args.latent_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        pad_token_id=args.pad_token_id,
        sos_token_id=args.sos_token_id,
        eos_token_id=args.eos_token_id,
        device=device
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Initialize scheduler
    total_steps = len(train_loader) * args.num_epochs
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=args.learning_rate * 0.01
        )
    else:
        scheduler = None
    
    # Load checkpoint if specified
    start_step = 0
    if args.resume_from:
        start_step, _ = load_checkpoint(model, optimizer, scheduler, args.resume_from, device)
    
    # Training loop
    print("Starting training...")
    model.train()
    step = start_step
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, (sequences, lengths) in enumerate(progress_bar):
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            outputs = model(sequences, lengths, target_sequences=sequences, sample=True)
            
            # Compute loss with KL annealing
            beta = get_kl_weight(step, total_steps, args.kl_schedule, args.max_beta)
            loss, recon_loss, kl_loss = model.compute_loss(
                outputs['logits'], sequences, outputs['latent_dist'], beta=beta
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Logging
            epoch_losses.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'beta': f'{beta:.3f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log to wandb
            if args.use_wandb and step % args.log_interval == 0:
                wandb.log({
                    'step': step,
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'train_recon_loss': recon_loss.item(),
                    'train_kl_loss': kl_loss.item(),
                    'beta': beta,
                    'learning_rate': current_lr
                })
            
            step += 1
            
            # Validation and checkpointing
            if step % args.val_interval == 0:
                print("\nRunning validation...")
                val_metrics = evaluate_model(model, val_loader, device, max_batches=args.max_val_batches)
                
                print(f"Validation - Loss: {val_metrics['val_loss']:.4f}, "
                      f"Recon: {val_metrics['val_recon_loss']:.4f}, "
                      f"KL: {val_metrics['val_kl_loss']:.4f}")
                
                if args.use_wandb:
                    wandb.log({**val_metrics, 'step': step})
                
                # Save checkpoint if best validation loss
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    save_checkpoint(model, optimizer, scheduler, step, val_metrics['val_loss'], 
                                  save_dir, 'best_model.pt')
                
                model.train()  # Back to training mode
            
            # Generate samples
            if step % args.sample_interval == 0:
                print("\nGenerating samples...")
                samples = generate_samples(model, device, num_samples=4, temperature=args.sample_temperature)
                
                if args.use_wandb:
                    # Log first few tokens of each sample
                    sample_table = wandb.Table(columns=["Sample", "Tokens"])
                    for i, sample in enumerate(samples):
                        tokens_str = " ".join(map(str, sample[:20]))  # First 20 tokens
                        sample_table.add_data(i, tokens_str)
                    wandb.log({"samples": sample_table, "step": step})
            
            # Regular checkpointing
            if step % args.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scheduler, step, loss.item(), save_dir)
        
        # End of epoch logging
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1} completed - Average loss: {avg_epoch_loss:.4f}")
        
        if args.use_wandb:
            wandb.log({'epoch_avg_loss': avg_epoch_loss, 'epoch': epoch})
    
    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, step, loss.item(), save_dir, 'final_model.pt')
    
    print(f"Training completed! Models saved in: {save_dir}")
    
    if args.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Music VAE')
    
    # Data arguments
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--val_data_path', type=str, required=True,
                        help='Path to validation data')
    parser.add_argument('--vocab_size', type=int, default=512,
                        help='Vocabulary size')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--pad_token_id', type=int, default=0,
                        help='Padding token ID')
    parser.add_argument('--sos_token_id', type=int, default=1,
                        help='Start of sequence token ID')
    parser.add_argument('--eos_token_id', type=int, default=2,
                        help='End of sequence token ID')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_encoder_layers', type=int, default=6,
                        help='Number of encoder layers')
    parser.add_argument('--n_decoder_layers', type=int, default=6,
                        help='Number of decoder layers')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use learning rate scheduler')
    
    # VAE-specific arguments
    parser.add_argument('--kl_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine', 'cyclical'],
                        help='KL annealing schedule')
    parser.add_argument('--max_beta', type=float, default=1.0,
                        help='Maximum beta for KL annealing')
    
    # Logging and checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval (steps)')
    parser.add_argument('--val_interval', type=int, default=1000,
                        help='Validation interval (steps)')
    parser.add_argument('--checkpoint_interval', type=int, default=5000,
                        help='Checkpointing interval (steps)')
    parser.add_argument('--sample_interval', type=int, default=2000,
                        help='Sample generation interval (steps)')
    parser.add_argument('--max_val_batches', type=int, default=50,
                        help='Maximum validation batches (None for all)')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='music-vae',
                        help='Wandb project name')
    
    # Generation arguments
    parser.add_argument('--sample_temperature', type=float, default=1.0,
                        help='Temperature for sample generation')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Start training
    train(args)


if __name__ == '__main__':
    main()