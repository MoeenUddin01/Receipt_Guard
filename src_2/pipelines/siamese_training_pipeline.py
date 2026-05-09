"""
siamese_training_pipeline.py - End-to-end training orchestration for Model 2.

This pipeline orchestrates the complete training process for the Siamese fraud detection model,
including data loading, model setup, training, and final diagnostics.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMTokenizer, get_linear_schedule_with_warmup

from ..data.dataloader import get_siamese_dataloaders
from ..model.siamese_model import SiameseConfig, SiameseSimilarityModel, build_siamese_model
from ..model.train import SiameseTrainer, detect_device


@dataclass
class SiameseTrainingConfig:
    """Configuration for Siamese model training."""
    model_path: str = "dataset/raw/SROIE2019/layoutlm-base-uncased"
    processed_path: str = "dataset/processed"
    output_dir: str = "artifacts/siamese/checkpoints"
    num_epochs: int = 10
    batch_size: int = 8
    max_length: int = 512
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'model_path': self.model_path,
            'processed_path': self.processed_path,
            'output_dir': self.output_dir,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio,
            'seed': self.seed
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SiameseTrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer_with_no_decay(model: torch.nn.Module, learning_rate: float, weight_decay: float):
    """
    Create optimizer with no weight decay for bias and LayerNorm parameters.
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        AdamW optimizer
    """
    # Separate parameters with and without weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def run_siamese_training_pipeline(config: SiameseTrainingConfig) -> Dict:
    """
    Run complete Siamese training pipeline.
    
    Args:
        config: SiameseTrainingConfig instance
        
    Returns:
        Training summary dictionary
    """
    print("=" * 80)
    print("Siamese Training Pipeline - Model 2")
    print("=" * 80)
    
    # Step 1: Setup
    print(f"\n{'─' * 60}")
    print("STEP 1: SETUP")
    print(f"{'─' * 60}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    print(f"✓ Random seed set to {config.seed}")
    
    # Create output directories
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    print(f"✓ Output directory created: {output_dir}")
    
    # Save configuration
    config_path = output_dir / "siamese_config.json"
    config.save(config_path)
    print(f"✓ Configuration saved to {config_path}")
    
    # Detect and log device
    device = detect_device()
    print(f"✓ Device detected: {device}")
    
    # Step 2: Data
    print(f"\n{'─' * 60}")
    print("STEP 2: DATA LOADING")
    print(f"{'─' * 60}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {config.model_path}...")
    tokenizer = LayoutLMTokenizer.from_pretrained(config.model_path)
    print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")
    
    # Load dataloaders
    print(f"Loading dataloaders from {config.processed_path}...")
    
    try:
        train_loader, test_loader = get_siamese_dataloaders(
            processed_path=config.processed_path,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            max_length=config.max_length
        )
        
        print(f"✓ Train dataloader: {len(train_loader)} batches, {len(train_loader.dataset)} samples")
        print(f"✓ Test dataloader: {len(test_loader)} batches, {len(test_loader.dataset)} samples")
        
    except FileNotFoundError as e:
        print(f"❌ Data loading failed: {e}")
        print("Please run the fraud data pipeline first to generate the required data.")
        return {'error': str(e), 'status': 'failed'}
    
    # Step 3: Model
    print(f"\n{'─' * 60}")
    print("STEP 3: MODEL SETUP")
    print(f"{'─' * 60}")
    
    # Create Siamese config
    siamese_config = SiameseConfig(
        model_path=config.model_path,
        dropout=0.1,
        projection_dim=256,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio
    )
    
    # Build model
    print(f"Building Siamese model...")
    model = build_siamese_model(siamese_config)
    model = model.to(device)
    print(f"✓ Model moved to {device}")
    
    # Step 4: Training
    print(f"\n{'─' * 60}")
    print("STEP 4: TRAINING")
    print(f"{'─' * 60}")
    
    try:
        # Create optimizer with proper weight decay handling
        optimizer = create_optimizer_with_no_decay(
            model, config.learning_rate, config.weight_decay
        )
        print(f"✓ Optimizer created: AdamW (lr={config.learning_rate}, wd={config.weight_decay})")
        
        # Create scheduler
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print(f"✓ Scheduler created: Linear warmup ({warmup_steps} steps) + decay")
        
        # Create trainer
        trainer = SiameseTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            config=siamese_config,
            device=device,
            output_dir=str(output_dir)
        )
        print(f"✓ Trainer initialized")
        
        # Start training
        print(f"\n🚀 Starting training for {config.num_epochs} epochs...")
        trainer.train(config.num_epochs)
        
        print(f"✓ Training completed successfully!")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA out of memory: {e}")
        print(f"💡 Try reducing batch_size from {config.batch_size} to {config.batch_size // 2}")
        return {'error': 'CUDA OOM', 'status': 'failed', 'suggestion': f'reduce batch_size to {config.batch_size // 2}'}
    
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'status': 'failed'}
    
    # Step 5: Final diagnostics
    print(f"\n{'─' * 60}")
    print("STEP 5: FINAL DIAGNOSTICS")
    print(f"{'─' * 60}")
    
    # Plot final similarity distributions
    final_eval_metrics = evaluate_siamese(model, test_loader, device)
    trainer.plot_similarity_distributions(config.num_epochs, final_eval_metrics)
    
    # Load best checkpoint to get best threshold
    checkpoint_path = output_dir / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        best_threshold = checkpoint.get('similarity_threshold', 0.5)
        best_epoch = checkpoint.get('epoch', config.num_epochs)
        best_loss = checkpoint.get('loss', 0.0)
        
        print(f"✓ Best model:")
        print(f"  Epoch: {best_epoch}")
        print(f"  Validation loss: {best_loss:.6f}")
        print(f"  Best threshold: {best_threshold:.3f}")
    
    # Create training summary
    summary = {
        'config': config.__dict__,
        'status': 'completed',
        'output_dir': str(output_dir),
        'best_threshold': best_threshold if checkpoint_path.exists() else 0.5,
        'total_epochs': config.num_epochs,
        'train_samples': len(train_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'device': str(device),
        'model_parameters': {
            'total': sum(p.numel() for p in model.parameters()),
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }
    
    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Training summary saved to {summary_path}")
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📁 Results saved to: {output_dir}")
    
    return summary


def evaluate_siamese(model, dataloader, device):
    """Quick evaluation for final diagnostics."""
    model.eval()
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            receipt_a = {k: v.to(device) for k, v in batch['receipt_a'].items()}
            receipt_b = {k: v.to(device) for k, v in batch['receipt_b'].items()}
            labels = batch['labels'].to(device)
            
            similarities = model.get_similarity(receipt_a, receipt_b)
            
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return {
        'all_similarities': all_similarities,
        'all_labels': all_labels
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Siamese fraud detection model (Model 2)")
    
    # Model configuration
    parser.add_argument(
        "--model-path",
        default="dataset/raw/SROIE2019/layoutlm-base-uncased",
        help="Path to pretrained LayoutLM model"
    )
    
    parser.add_argument(
        "--output-dir",
        default="artifacts/siamese/checkpoints",
        help="Output directory for checkpoints and logs"
    )
    
    # Training configuration
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = SiameseTrainingConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed
    )
    
    try:
        # Run pipeline
        summary = run_siamese_training_pipeline(config)
        
        if summary.get('status') == 'completed':
            print(f"\n{'=' * 80}")
            print("🚀 Siamese training pipeline completed successfully!")
            print(f"📁 Results saved to: {config.output_dir}")
            print(f"🎯 Best threshold: {summary.get('best_threshold', 0.5):.3f}")
            print(f"{'=' * 80}")
            return 0
        else:
            print(f"\n❌ Pipeline failed: {summary.get('error', 'Unknown error')}")
            if 'suggestion' in summary:
                print(f"💡 Suggestion: {summary['suggestion']}")
            return 1
    
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
