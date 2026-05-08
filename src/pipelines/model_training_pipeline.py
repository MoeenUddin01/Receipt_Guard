"""
Model training pipeline for ReceiptGuard-ML.
End-to-end training orchestration for LayoutLM-based NER model.
"""

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from dataset import ReceiptDataset, collate_fn, LABEL2ID
from dataloader import get_tokenizer
from model import ModelConfig, build_model
from train import Trainer, get_device

# Import configuration
from src.config import CFG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training pipeline."""

    # Model settings
    model_path: str = None
    num_labels: int = None
    dropout: float = None

    # Training settings
    output_dir: str = None
    num_epochs: int = None
    batch_size: int = None
    max_length: int = None
    learning_rate: float = None
    weight_decay: float = None
    warmup_ratio: float = None
    seed: int = None

    # Data paths
    data_path: str = None
    
    def __post_init__(self):
        """Set default values from CFG if not provided and resolve paths."""
        if self.model_path is None:
            self.model_path = CFG.model.model_path
        if self.num_labels is None:
            self.num_labels = CFG.model.num_labels
        if self.dropout is None:
            self.dropout = CFG.model.dropout
        if self.output_dir is None:
            self.output_dir = str(CFG.resolve_path(CFG.training.output_dir))
        if self.num_epochs is None:
            self.num_epochs = CFG.training.num_epochs
        if self.batch_size is None:
            self.batch_size = CFG.training.batch_size
        if self.max_length is None:
            self.max_length = CFG.data.max_length
        if self.learning_rate is None:
            self.learning_rate = CFG.training.learning_rate
        if self.weight_decay is None:
            self.weight_decay = CFG.training.weight_decay
        if self.warmup_ratio is None:
            self.warmup_ratio = CFG.training.warmup_ratio
        if self.seed is None:
            self.seed = CFG.training.seed
        if self.data_path is None:
            self.data_path = str(CFG.resolve_path(CFG.data.raw_data_path))
        else:
            # Resolve if provided as relative
            self.data_path = str(CFG.resolve_path(self.data_path))

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save_config(self, path: str) -> None:
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        logger.info(f"Config saved to {path}")

    @classmethod
    def load_config(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def to_model_config(self) -> ModelConfig:
        """Convert TrainingConfig to ModelConfig for model building."""
        # Calculate warmup steps based on warmup_ratio
        # This is an estimate; actual calculation needs dataset size
        return ModelConfig(
            model_path=self.model_path,
            num_labels=self.num_labels,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=0,  # Will be calculated later when dataloader is known
            max_length=self.max_length,
        )


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_dataloaders(
    config: TrainingConfig,
    tokenizer
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test dataloaders.

    Args:
        config: Training configuration
        tokenizer: LayoutLM tokenizer instance

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Build datasets
    logger.info("Building datasets...")

    # Data paths are already resolved in TrainingConfig normalization
    data_path = config.data_path
    model_path = config.model_path
    batch_size = config.batch_size
    max_length = config.max_length

    train_dataset = ReceiptDataset(
        data_path=data_path,
        split="train",
        tokenizer_name=model_path,
        max_length=max_length,
    )

    test_dataset = ReceiptDataset(
        data_path=data_path,
        split="test",
        tokenizer_name=model_path,
        max_length=max_length,
    )

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Log statistics
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    logger.info(f"Train batches: {len(train_loader)} (batch_size={config.training.batch_size})")
    logger.info(f"Test batches: {len(test_loader)} (batch_size={config.training.batch_size})")

    return train_loader, test_loader


def normalize_config(config: Union[TrainingConfig, Any]) -> TrainingConfig:
    """Ensure the config object is a TrainingConfig instance with resolved paths."""
    if isinstance(config, TrainingConfig):
        return config
    
    # If it's the global CFG or a Config instance
    return TrainingConfig(
        model_path=config.model.model_path,
        num_labels=config.model.num_labels,
        dropout=config.model.dropout,
        output_dir=str(CFG.resolve_path(config.training.output_dir)),
        num_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        max_length=config.data.max_length,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        seed=config.training.seed,
        data_path=str(CFG.resolve_path(config.data.raw_data_path))
    )


def run_training_pipeline(config: Union[TrainingConfig, Any]) -> Dict:
    """
    Run the complete model training pipeline.

    Steps:
    1. Setup: Set seed, create output directory, save config, detect device
    2. Data: Load tokenizer, build dataloaders
    3. Model: Build model, move to device, log parameter count
    4. Train: Instantiate Trainer and run training
    5. Report: Plot training curves and return summary

    Args:
        config: TrainingConfig with all training parameters

    Returns:
        Training summary dictionary
    """
    logger.info("=" * 70)
    logger.info("Starting ReceiptGuard-ML Model Training Pipeline")
    logger.info("=" * 70)

    # Ensure we have a TrainingConfig with absolute paths
    config = normalize_config(config)

    # =========================================================================
    # Step 1 — Setup
    # =========================================================================
    logger.info("\nStep 1: Setup...")

    # Set seed for reproducibility
    set_seed(config.seed)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")

    # Save config
    import json
    config_path = output_dir / "training_config.json"
    config_data = config.to_dict()
        
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2, default=str)

    # Detect device
    device = get_device()
    logger.info(f"Device: {device}")

    # =========================================================================
    # Step 2 — Data
    # =========================================================================
    logger.info("\nStep 2: Loading data...")

    # Load tokenizer
    tokenizer = get_tokenizer(config.model_path)
    logger.info(f"Tokenizer loaded from {config.model_path}")

    # Build dataloaders
    train_loader, val_loader = get_dataloaders(config, tokenizer)

    # =========================================================================
    # Step 3 — Model
    # =========================================================================
    logger.info("\nStep 3: Building model...")

    # Build model config with calculated warmup steps
    from src.model.model import ModelConfig
    # Calculate warmup steps based on actual train loader size
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    model_config = ModelConfig(
        model_path=config.model_path,
        num_labels=config.num_labels,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        max_length=config.max_length
    )
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {model_config.warmup_steps} ({config.warmup_ratio*100:.0f}%)")

    # Build model
    model = build_model(model_config)

    # Move to device
    model = model.to(device)
    logger.info(f"Model moved to {device}")

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # Step 4 — Train
    # =========================================================================
    logger.info("\nStep 4: Training...")

    # Instantiate trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=model_config,
        device=device,
        output_dir=config.output_dir,
    )

    # Run training with CUDA OOM handling
    try:
        trainer.train(num_epochs=config.num_epochs)
        training_completed = True
        oom_error = None
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            logger.error("=" * 70)
            logger.error("CUDA Out of Memory Error!")
            logger.error("=" * 70)
            logger.error(str(e))
            logger.error("\nSuggestions to resolve OOM:")
            logger.error("  1. Reduce batch_size (current: %d)", config.batch_size)
            logger.error("  2. Reduce max_length (current: %d)", config.max_length)
            logger.error("  3. Use a smaller model or enable gradient checkpointing")
            logger.error("  4. Use a GPU with more memory")
            training_completed = False
            oom_error = str(e)
        else:
            raise

    # =========================================================================
    # Step 5 — Report
    # =========================================================================
    logger.info("\nStep 5: Generating report...")

    if training_completed:
        # Plot training curves
        trainer.plot_training_curves()
        logger.info("Training curves plotted")

        # Build summary
        summary = {
            'status': 'completed',
            'config': asdict(config),
            'device': str(device),
            'model_parameters': {
                'total': total_params,
                'trainable': trainable_params,
            },
            'training_history': trainer.history,
            'best_eval_loss': trainer.best_eval_loss,
            'best_checkpoint': str(trainer.best_checkpoint_path) if trainer.best_checkpoint_path else None,
            'output_dir': config.output_dir,
        }

        # Save training report to text file
        report_path = output_dir / "training_report.txt"
        with open(report_path, "w", encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ReceiptGuard-ML Training Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Status: Completed Successfully\n")
            f.write(f"Device: {device}\n")
            f.write(f"Model: {config.model_path}\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
            f.write("-" * 70 + "\n")
            f.write("Training Configuration\n")
            f.write("-" * 70 + "\n")
            f.write(f"Epochs: {config.num_epochs}\n")
            f.write(f"Batch Size: {config.batch_size}\n")
            f.write(f"Learning Rate: {config.learning_rate}\n")
            f.write(f"Weight Decay: {config.weight_decay}\n")
            f.write(f"Warmup Ratio: {config.warmup_ratio}\n")
            f.write(f"Max Length: {config.max_length}\n")
            f.write(f"Dropout: {config.dropout}\n\n")
            f.write("-" * 70 + "\n")
            f.write("Training Results\n")
            f.write("-" * 70 + "\n")
            f.write(f"Best Eval Loss: {trainer.best_eval_loss:.4f}\n")
            f.write(f"Best Checkpoint: {trainer.best_checkpoint_path}\n\n")
            f.write("-" * 70 + "\n")
            f.write("Per-Epoch Results\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<14}{'Eval Loss':<14}{'Eval Acc':<14}\n")
            f.write("-" * 70 + "\n")
            for i, epoch in enumerate(trainer.history['epochs']):
                train_loss = trainer.history['train_loss'][i]
                train_acc = trainer.history['train_accuracy'][i]
                eval_loss = trainer.history['eval_loss'][i]
                eval_acc = trainer.history['eval_accuracy'][i]
                f.write(f"{epoch:<8}{train_loss:<14.4f}{train_acc:<14.4f}{eval_loss:<14.4f}{eval_acc:<14.4f}\n")
            f.write("=" * 70 + "\n")
        logger.info(f"Training report saved to: {report_path}")

        logger.info("=" * 70)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info("=" * 70)
        logger.info(f"Best eval loss: {trainer.best_eval_loss:.4f}")
        logger.info(f"Best checkpoint: {trainer.best_checkpoint_path}")
    else:
        summary = {
            'status': 'failed',
            'error': oom_error,
            'config': asdict(config),
            'device': str(device),
            'suggestion': 'Reduce batch_size or max_length and retry',
        }

        logger.info("=" * 70)
        logger.info("Training Pipeline Failed (OOM)")
        logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReceiptGuard-ML Model Training Pipeline"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="dataset/raw/SROIE2019/layoutlm-base-uncased",
        help="Path to pretrained LayoutLM model"
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=9,
        help="Number of NER labels (default: 9)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)"
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/processed/checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio of total steps (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/raw/SROIE2019",
        help="Path to dataset"
    )

    # Config file option
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (overrides other arguments)"
    )

    args = parser.parse_args()

    # Load from config file if provided
    if args.config:
        config = TrainingConfig.load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = TrainingConfig(
            model_path=args.model_path,
            num_labels=args.num_labels,
            dropout=args.dropout,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            data_path=args.data_path,
        )

    # Run pipeline
    summary = run_training_pipeline(config)

    # Exit with appropriate code
    if summary['status'] == 'completed':
        sys.exit(0)
    else:
        sys.exit(1)
