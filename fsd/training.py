import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from data_loader import load_dataset
from model import MixClearNet
import torch.nn as nn
import numpy as np
import os

def curriculum_learning_scheduler(epoch: int, total_epochs: int = 100) -> float:
    """
    Curriculum learning as described in the paper
    Returns difficulty factor (0-1, where 1 is most difficult)
    """
    return min(1.0, epoch / (total_epochs * 0.3))  # Reach full difficulty at 30% of training

def compute_si_snr(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio
    """
    # Ensure same length
    min_len = min(predicted.size(-1), target.size(-1))
    predicted = predicted[..., :min_len]
    target = target[..., :min_len]
    
    # Zero-mean
    predicted = predicted - torch.mean(predicted, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # SI-SNR computation
    s_target = torch.sum(predicted * target, dim=-1, keepdim=True) * target / (torch.sum(target**2, dim=-1, keepdim=True) + 1e-8)
    e_noise = predicted - s_target
    
    si_snr = 10 * torch.log10((torch.sum(s_target**2, dim=-1) + 1e-8) / (torch.sum(e_noise**2, dim=-1) + 1e-8))
    
    return torch.mean(si_snr)

# Training loop with automatic mixed precision, gradient accumulation, and curriculum learning
def train_model(model, dataloader, optimizer, scheduler, device, num_epochs=100, accumulation_steps=4):
    model.train()
    scaler = GradScaler()
    writer = SummaryWriter(log_dir="runs/MixClearNet")
    
    best_si_snr = float('-inf')
    patience = 0
    max_patience = 10
    
    print("Starting MixClearNet training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_si_snr = 0.0
        num_batches = 0
        
        # Curriculum learning
        difficulty = curriculum_learning_scheduler(epoch, num_epochs)
        
        model.train()
        for batch_idx, (mixed_audio, target_speakers) in enumerate(dataloader):
            try:
                mixed_audio = mixed_audio.to(device)
                target_speakers = target_speakers.to(device)
                
                # Ensure proper tensor shapes
                if mixed_audio.dim() == 2:
                    mixed_audio = mixed_audio.float()
                if target_speakers.dim() == 3:
                    target_speakers = target_speakers.float()
                
                with autocast():
                    # Forward pass
                    separated_speakers = model(mixed_audio)
                    
                    # Compute composite loss
                    loss = model.compute_composite_loss(separated_speakers, target_speakers)
                    loss = loss / accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    # Gradient clipping as mentioned in paper
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                # Compute SI-SNR for monitoring
                with torch.no_grad():
                    si_snr = 0
                    for i in range(separated_speakers.size(1)):
                        si_snr += compute_si_snr(separated_speakers[:, i], target_speakers[:, i])
                    si_snr /= separated_speakers.size(1)
                    total_si_snr += si_snr.item()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Log to TensorBoard
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar("Loss/train_step", loss.item(), global_step)
                writer.add_scalar("SI-SNR/train_step", si_snr.item(), global_step)
                writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar("Curriculum_Difficulty", difficulty, global_step)
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, SI-SNR: {si_snr.item():.2f} dB")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Epoch statistics
        avg_loss = total_loss / max(num_batches, 1)
        avg_si_snr = total_si_snr / max(num_batches, 1)
        
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average SI-SNR: {avg_si_snr:.2f} dB")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Curriculum Difficulty: {difficulty:.2f}")
        print("-" * 50)
        
        # Log epoch statistics
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("SI-SNR/train_epoch", avg_si_snr, epoch)
        
        # Learning rate scheduling based on SI-SNR
        scheduler.step(avg_si_snr)
        
        # Early stopping and model saving
        if avg_si_snr > best_si_snr:
            best_si_snr = avg_si_snr
            patience = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_si_snr': best_si_snr,
            }, "mixclearnet_best.pth")
            print(f"New best model saved! SI-SNR: {best_si_snr:.2f} dB")
        else:
            patience += 1
            
        if patience >= max_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'si_snr': avg_si_snr,
            }, f"mixclearnet_checkpoint_epoch_{epoch}.pth")

    writer.close()
    print("Training completed!")
    return best_si_snr

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_name = "si_dt_05"
    try:
        dataloader = load_dataset(dataset_name)
        print(f"Dataset loaded successfully: {len(dataloader)} batches")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    # Initialize MixClearNet model
    model = MixClearNet(num_speakers=2).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.1f}M")
    
    # Define optimizer with paper's settings
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-3,  # As mentioned in paper
        weight_decay=1e-4  # As mentioned in paper
    )
    
    # Learning rate scheduler - reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Maximize SI-SNR
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    # Train the model
    try:
        best_si_snr = train_model(model, dataloader, optimizer, scheduler, device)
        print(f"Training completed. Best SI-SNR achieved: {best_si_snr:.2f} dB")
        
        # Save final model
        torch.save(model.state_dict(), "mixclearnet_final.pth")
        print("Final model saved as mixclearnet_final.pth")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()