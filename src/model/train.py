"""
Training loop for LayoutLM-based NER model.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.model.model import (
    ModelConfig,
    ReceiptFieldExtractor,
    save_checkpoint,
)


def get_device() -> torch.device:
    """Detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_one_epoch(
    model: ReceiptFieldExtractor,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    epoch_num: int,
    max_grad_norm: float = 1.0,
) -> dict:
    """
    Run one training epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch_num: Current epoch number (for logging)
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Dictionary with {epoch, train_loss, train_loss_per_step: list}
    """
    model.train()
    total_loss = 0.0
    loss_per_step = []
    num_batches = len(dataloader)
    
    # Check if CUDA is available for mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        bbox = batch["bbox"].to(device)
        labels = batch["labels"].to(device) if "labels" in batch else None
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if available
        if use_amp:
            with torch.cuda.amp.autocast():
                loss, _ = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    bbox,
                    labels,
                )
        else:
            loss, _ = model(
                input_ids,
                attention_mask,
                token_type_ids,
                bbox,
                labels,
            )
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Track loss
        loss_value = loss.item()
        total_loss += loss_value
        loss_per_step.append(loss_value)
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss_value:.4f}"})
    
    pbar.close()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        "epoch": epoch_num,
        "train_loss": avg_loss,
        "train_loss_per_step": loss_per_step,
    }


def evaluate_on_loader(
    model: ReceiptFieldExtractor,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run evaluation on a data loader.
    
    Args:
        model: The model to evaluate
        dataloader: Evaluation data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary with {eval_loss, token_accuracy}
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            bbox = batch["bbox"].to(device)
            labels = batch["labels"].to(device) if "labels" in batch else None
            
            # Forward pass
            loss, logits = model(
                input_ids,
                attention_mask,
                token_type_ids,
                bbox,
                labels,
            )
            
            if loss is not None:
                total_loss += loss.item()
            
            # Get predictions and compute accuracy (excluding -100 padding)
            if labels is not None:
                preds = torch.argmax(logits, dim=-1)
                
                # Mask for valid tokens (not padding)
                valid_mask = labels != -100
                
                # Count correct predictions
                correct = (preds == labels) & valid_mask
                total_correct += correct.sum().item()
                total_tokens += valid_mask.sum().item()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return {
        "eval_loss": avg_loss,
        "token_accuracy": accuracy,
    }


def build_optimizer_and_scheduler(
    model: ReceiptFieldExtractor,
    config: ModelConfig,
    num_training_steps: int,
) -> tuple[AdamW, Optional]:
    """
    Build AdamW optimizer and linear warmup scheduler.
    
    Args:
        model: The model to optimize
        config: Model configuration
        num_training_steps: Total number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Separate parameters for weight decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm"]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    return optimizer, scheduler


class Trainer:
    """Trainer class for receipt field extraction model."""
    
    def __init__(
        self,
        model: ReceiptFieldExtractor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ModelConfig,
        device: torch.device,
        output_dir: str,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Model configuration
            device: Device to train on
            output_dir: Directory for outputs (checkpoints, logs, plots)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            "epochs": [],
            "train_loss": [],
            "eval_loss": [],
            "token_accuracy": [],
        }
        
        # Best model tracking
        self.best_eval_loss = float("inf")
        self.best_checkpoint_path: Optional[Path] = None
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.output_dir / "runs")
        
        print(f"Trainer initialized on device: {device}")
        print(f"Output directory: {self.output_dir}")
    
    def train(self, num_epochs: int) -> None:
        """
        Run full training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        # Calculate total training steps
        num_training_steps = num_epochs * len(self.train_loader)
        
        # Build optimizer and scheduler
        optimizer, scheduler = build_optimizer_and_scheduler(
            self.model,
            self.config,
            num_training_steps,
        )
        
        print(f"\nTraining for {num_epochs} epochs")
        print(f"Total training steps: {num_training_steps}")
        print(f"Steps per epoch: {len(self.train_loader)}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            # Training phase
            train_results = train_one_epoch(
                self.model,
                self.train_loader,
                optimizer,
                scheduler,
                self.device,
                epoch,
            )
            
            train_loss = train_results["train_loss"]
            
            # Evaluation phase
            eval_results = evaluate_on_loader(
                self.model,
                self.val_loader,
                self.device,
            )
            
            eval_loss = eval_results["eval_loss"]
            token_accuracy = eval_results["token_accuracy"]
            
            # Update history
            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["eval_loss"].append(eval_loss)
            self.history["token_accuracy"].append(token_accuracy)
            
            # Log to console
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Eval Loss: {eval_loss:.4f} | "
                f"Token Acc: {token_accuracy:.4f}"
            )
            
            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/eval", eval_loss, epoch)
            self.writer.add_scalar("Accuracy/token", token_accuracy, epoch)
            self.writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)
            
            # Save best checkpoint
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.best_checkpoint_path = self.output_dir / "best_model.pt"
                save_checkpoint(
                    self.model,
                    optimizer,
                    epoch,
                    eval_loss,
                    str(self.best_checkpoint_path),
                )
                print(f"  -> Best model saved (eval_loss: {eval_loss:.4f})")
        
        print("=" * 60)
        print("Training completed!")
        print(f"Best eval loss: {self.best_eval_loss:.4f}")
        
        # Save final checkpoint
        final_path = self.output_dir / "final_model.pt"
        save_checkpoint(
            self.model,
            optimizer,
            num_epochs,
            eval_loss,
            str(final_path),
        )
        
        # Save training history
        self._save_training_log()
        
        # Plot training curves
        self.plot_training_curves()
        
        # Close TensorBoard writer
        self.writer.close()
    
    def _save_training_log(self) -> None:
        """Save training history to JSON file."""
        log_path = self.output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training log saved to {log_path}")
    
    def plot_training_curves(self) -> None:
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = self.history["epochs"]
        
        # Loss subplot
        ax1 = axes[0]
        ax1.plot(epochs, self.history["train_loss"], "b-", label="Train Loss", linewidth=2)
        ax1.plot(epochs, self.history["eval_loss"], "r-", label="Eval Loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Evaluation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy subplot
        ax2 = axes[1]
        ax2.plot(
            epochs,
            self.history["token_accuracy"],
            "g-",
            label="Token Accuracy",
            linewidth=2,
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Token-Level Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Training curves saved to {plot_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("train.py - Training utilities for ReceiptGuard-ML")
    print("=" * 60)
    
    # Test device detection
    device = get_device()
    print(f"\nDevice detection: {device}")
    
    print("\nTrainer class components:")
    print("- train_one_epoch(): Single epoch training with mixed precision")
    print("- evaluate_on_loader(): Evaluation with token accuracy")
    print("- build_optimizer_and_scheduler(): AdamW + linear warmup")
    print("- Trainer: Full training loop with TensorBoard logging")
    
    print("\nUsage:")
    print("  from src.model.train import Trainer, get_device")
    print("  from src.model.model import build_model, ModelConfig")
    print("  ")
    print("  device = get_device()")
    print("  model = build_model(config)")
    print("  trainer = Trainer(model, train_loader, val_loader, config, device, output_dir)")
    print("  trainer.train(num_epochs=10)")
