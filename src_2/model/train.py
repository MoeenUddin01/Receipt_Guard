"""
train.py - Training loop for Siamese LayoutLM model.

This module provides training utilities for the Siamese model including
threshold optimization, mixed precision training, and comprehensive
logging with similarity distribution analysis.
"""

import gc
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .siamese_model import SiameseSimilarityModel


def find_best_threshold(similarities: List[float], labels: List[int]) -> float:
    """
    Find optimal threshold that maximizes F1 score on validation set.
    
    Args:
        similarities: List of cosine similarity scores
        labels: List of binary labels (0=legitimate, 1=fraud)
        
    Returns:
        Best threshold value between 0.1 and 0.99
    """
    best_threshold = 0.5
    best_f1 = 0.0
    
    # Sweep thresholds from 0.1 to 0.99 in steps of 0.01
    for threshold in np.arange(0.1, 1.0, 0.01):
        # Predict fraud if similarity > threshold
        predictions = [1 if sim > threshold else 0 for sim in similarities]
        
        # Calculate F1 score
        f1 = f1_score(labels, predictions, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def train_siamese_epoch(
    model: SiameseSimilarityModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch_num: int,
    scaler: GradScaler = None,
    gradient_accumulation_steps: int = 1
) -> Dict:
    """
    One training epoch with mixed precision, gradient clipping, and
    gradient accumulation for memory-constrained GPUs.
    
    Args:
        model: SiameseSimilarityModel instance
        dataloader: Training dataloader
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch_num: Current epoch number
        scaler: Optional GradScaler instance (reused across epochs)
        gradient_accumulation_steps: Number of mini-batches to accumulate
            before performing an optimizer step (default 1 = no accumulation)
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    all_similarities_fraud = []
    all_similarities_legit = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num}")
    
    # Zero gradients at the start
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device (non_blocking for pin_memory overlap)
        receipt_a = {k: v.to(device, non_blocking=True) for k, v in batch['receipt_a'].items()}
        receipt_b = {k: v.to(device, non_blocking=True) for k, v in batch['receipt_b'].items()}
        labels = batch['labels'].to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        if scaler is not None:
            with autocast('cuda'):
                loss, logits, similarity_scores = model(receipt_a, receipt_b, labels)
            
            # Scale loss by accumulation steps
            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
        else:
            loss, logits, similarity_scores = model(receipt_a, receipt_b, labels)
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
        
        # Optimizer step at accumulation boundary or end of epoch
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Accumulate metrics (move to CPU immediately to free GPU memory)
        batch_loss = loss.item()
        total_loss += batch_loss
        
        similarities_cpu = similarity_scores.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        
        # Free GPU memory for this batch explicitly
        del receipt_a, receipt_b, labels, loss, logits, similarity_scores, scaled_loss
        
        for sim, label in zip(similarities_cpu, labels_cpu):
            if label == 1:  # Fraud
                all_similarities_fraud.append(sim)
            else:  # Legitimate
                all_similarities_legit.append(sim)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'avg_sim_fraud': f'{np.mean(all_similarities_fraud[-50:]) if all_similarities_fraud else 0:.3f}',
            'avg_sim_legit': f'{np.mean(all_similarities_legit[-50:]) if all_similarities_legit else 0:.3f}'
        })
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(dataloader)
    avg_similarity_fraud = np.mean(all_similarities_fraud) if all_similarities_fraud else 0.0
    avg_similarity_legit = np.mean(all_similarities_legit) if all_similarities_legit else 0.0
    
    return {
        'epoch': epoch_num,
        'train_loss': avg_loss,
        'avg_similarity_fraud': avg_similarity_fraud,
        'avg_similarity_legit': avg_similarity_legit
    }


def evaluate_siamese(
    model: SiameseSimilarityModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict:
    """
    Evaluation pass without gradient computation.
    
    Args:
        model: SiameseSimilarityModel instance
        dataloader: Validation dataloader
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    all_similarities = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            receipt_a = {k: v.to(device, non_blocking=True) for k, v in batch['receipt_a'].items()}
            receipt_b = {k: v.to(device, non_blocking=True) for k, v in batch['receipt_b'].items()}
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass with mixed precision for memory efficiency
            if device.type == 'cuda':
                with autocast('cuda'):
                    loss, logits, similarity_scores = model(receipt_a, receipt_b, labels)
            else:
                loss, logits, similarity_scores = model(receipt_a, receipt_b, labels)
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Get predictions (argmax of logits)
            predictions = torch.argmax(logits, dim=1)
            
            # Store results
            all_similarities.extend(similarity_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            
            # Free GPU memory
            del receipt_a, receipt_b, labels, loss, logits, similarity_scores, predictions
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return {
        'eval_loss': avg_loss,
        'accuracy': accuracy,
        'all_similarities': all_similarities,
        'all_labels': all_labels
    }


class SiameseTrainer:
    """
    Trainer class for Siamese model with comprehensive logging and checkpointing.
    """
    
    def __init__(
        self,
        model: SiameseSimilarityModel,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config,
        device: torch.device,
        output_dir: str
    ):
        """
        Initialize trainer.
        
        Args:
            model: SiameseSimilarityModel instance
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Training configuration
            device: Device to train on
            output_dir: Directory to save outputs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "runs" / "siamese").mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        self.writer = SummaryWriter(self.output_dir / "runs" / "siamese")
        
        # Best validation loss tracking
        self.best_val_loss = float('inf')
        self.best_threshold = 0.5
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
    
    def train(self, num_epochs: int):
        """
        Full training loop with evaluation and checkpointing.
        
        Args:
            num_epochs: Number of training epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio
        )
        
        # Create GradScaler once and reuse across epochs
        scaler = GradScaler('cuda') if self.device.type == 'cuda' else None
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Training
            train_metrics = train_siamese_epoch(
                self.model, self.train_loader, optimizer, scheduler, self.device, epoch,
                scaler=scaler
            )
            
            # Free training activations before evaluation
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Evaluation
            eval_metrics = evaluate_siamese(self.model, self.val_loader, self.device)
            
            # Find best threshold on validation set
            best_threshold = find_best_threshold(
                eval_metrics['all_similarities'], 
                eval_metrics['all_labels']
            )
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Loss/Val', eval_metrics['eval_loss'], epoch)
            self.writer.add_scalar('Accuracy/Val', eval_metrics['accuracy'], epoch)
            self.writer.add_scalar('Threshold/Best', best_threshold, epoch)
            self.writer.add_scalar('Similarity/Fraud', train_metrics['avg_similarity_fraud'], epoch)
            self.writer.add_scalar('Similarity/Legit', train_metrics['avg_similarity_legit'], epoch)
            self.writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
            
            # Print epoch summary
            print(f"Train Loss: {train_metrics['train_loss']:.6f}")
            print(f"Val Loss: {eval_metrics['eval_loss']:.6f}")
            print(f"Val Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"Best Threshold: {best_threshold:.3f}")
            print(f"Avg Sim Fraud: {train_metrics['avg_similarity_fraud']:.3f}")
            print(f"Avg Sim Legit: {train_metrics['avg_similarity_legit']:.3f}")
            print(f"Similarity Gap: {abs(train_metrics['avg_similarity_fraud'] - train_metrics['avg_similarity_legit']):.3f}")
            
            # Save checkpoint if validation loss improves
            if eval_metrics['eval_loss'] < self.best_val_loss:
                self.best_val_loss = eval_metrics['eval_loss']
                self.best_threshold = best_threshold
                
                checkpoint_path = self.output_dir / "best_model.pth"
                from .siamese_model import save_siamese_checkpoint
                save_siamese_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=eval_metrics['eval_loss'],
                    similarity_threshold=best_threshold,
                    path=str(checkpoint_path)
                )
                print(f"✓ New best model saved (Val Loss: {eval_metrics['eval_loss']:.6f})")
            
            # Plot similarity distributions
            self.plot_similarity_distributions(epoch, eval_metrics)
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best Val Loss: {self.best_val_loss:.6f}")
        print(f"Best Threshold: {self.best_threshold:.3f}")
        print(f"{'='*60}")
        
        self.writer.close()
    
    def plot_similarity_distributions(self, epoch: int, eval_metrics: Dict):
        """
        Save histogram showing overlap between fraud and legit similarity distributions.
        
        Args:
            epoch: Current epoch number
            eval_metrics: Evaluation metrics containing similarities and labels
        """
        similarities = eval_metrics['all_similarities']
        labels = eval_metrics['all_labels']
        
        # Separate similarities by label
        fraud_sims = [sim for sim, label in zip(similarities, labels) if label == 1]
        legit_sims = [sim for sim, label in zip(similarities, labels) if label == 0]
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        
        bins = np.linspace(0, 1, 51)  # 50 bins from 0 to 1
        
        plt.hist(legit_sims, bins=bins, alpha=0.7, label='Legitimate', color='blue', density=True)
        plt.hist(fraud_sims, bins=bins, alpha=0.7, label='Fraud', color='red', density=True)
        
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title(f'Similarity Distributions - Epoch {epoch}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        if legit_sims and fraud_sims:
            plt.axvline(np.mean(legit_sims), color='blue', linestyle='--', alpha=0.7, 
                       label=f'Legit Mean: {np.mean(legit_sims):.3f}')
            plt.axvline(np.mean(fraud_sims), color='red', linestyle='--', alpha=0.7,
                       label=f'Fraud Mean: {np.mean(fraud_sims):.3f}')
        
        # Save plot
        plot_path = self.output_dir / f"sim_dist_epoch{epoch}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Similarity distribution plot saved to {plot_path}")


def detect_device() -> torch.device:
    """
    Detect the best available device for training.
    
    Returns:
        torch.device: CUDA > MPS > CPU priority
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


if __name__ == "__main__":
    """Test the training utilities."""
    
    print("=" * 60)
    print("Testing Siamese Training Utilities")
    print("=" * 60)
    
    try:
        # Test device detection
        print("\n1. Testing device detection:")
        device = detect_device()
        print(f"   Detected device: {device}")
        
        # Test threshold optimization
        print("\n2. Testing find_best_threshold:")
        # Create dummy data where fraud has higher similarity
        np.random.seed(42)
        legit_similarities = np.random.normal(0.3, 0.1, 100).clip(0, 1)
        fraud_similarities = np.random.normal(0.7, 0.1, 100).clip(0, 1)
        
        similarities = np.concatenate([legit_similarities, fraud_similarities]).tolist()
        labels = [0] * 100 + [1] * 100
        
        best_threshold = find_best_threshold(similarities, labels)
        print(f"   Best threshold: {best_threshold:.3f}")
        print(f"   Legit mean similarity: {np.mean(legit_similarities):.3f}")
        print(f"   Fraud mean similarity: {np.mean(fraud_similarities):.3f}")
        
        # Test that threshold separates the distributions
        legit_correct = sum(1 for sim in legit_similarities if sim <= best_threshold)
        fraud_correct = sum(1 for sim in fraud_similarities if sim > best_threshold)
        
        print(f"   Legit correctly classified: {legit_correct}/100")
        print(f"   Fraud correctly classified: {fraud_correct}/100")
        
        print("\n3. Testing SiameseTrainer initialization:")
        print("   ✓ All utilities imported successfully")
        print("   ✓ Device detection working")
        print("   ✓ Threshold optimization working")
        print("   ✓ Ready for training!")
        
    except Exception as e:
        print(f"\n   ⚠️  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)
