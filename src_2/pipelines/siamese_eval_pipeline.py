"""
siamese_eval_pipeline.py - Evaluation pipeline for trained Siamese model.

This pipeline loads a trained Siamese model and runs comprehensive evaluation
including overall metrics, per-fraud-type analysis, and visualization.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from transformers import LayoutLMTokenizer

from ..data.dataloader import get_siamese_dataloaders
from ..model.evaluation import (compute_fraud_metrics, compute_per_type_metrics,
                               run_siamese_evaluation)
from ..model.siamese_model import (SiameseSimilarityModel,
                                  load_siamese_checkpoint)
from ..model.train import detect_device


@dataclass
class SiameseEvalConfig:
    """Configuration for Siamese model evaluation."""
    checkpoint_path: str  # required
    model_path: str = "dataset/raw/SROIE2019/layoutlm-base-uncased"
    output_dir: str = "artifacts/siamese/evaluation"
    batch_size: int = 8


def run_siamese_eval_pipeline(config: SiameseEvalConfig) -> Dict:
    """
    Run complete Siamese model evaluation pipeline.
    
    Args:
        config: SiameseEvalConfig instance
        
    Returns:
        Full evaluation results dictionary
    """
    print("=" * 80)
    print("Siamese Model Evaluation Pipeline")
    print("=" * 80)
    
    # Validate checkpoint exists
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Load model
    print(f"\n{'─' * 60}")
    print("STEP 1: LOAD MODEL")
    print(f"{'─' * 60}")
    
    # Detect device
    device = detect_device()
    print(f"Device: {device}")
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint...")
    checkpoint_info = load_siamese_checkpoint(str(checkpoint_path))
    model = checkpoint_info['model']
    model = model.to(device)
    model.eval()
    
    # Extract saved threshold
    similarity_threshold = checkpoint_info['similarity_threshold']
    epoch = checkpoint_info['epoch']
    val_loss = checkpoint_info['loss']
    
    print(f"✓ Model loaded successfully")
    print(f"  Epoch: {epoch}")
    print(f"  Validation loss: {val_loss:.6f}")
    print(f"  Similarity threshold: {similarity_threshold:.3f}")
    
    # Step 2: Load test data
    print(f"\n{'─' * 60}")
    print("STEP 2: LOAD TEST DATA")
    print(f"{'─' * 60}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {config.model_path}...")
    tokenizer = LayoutLMTokenizer.from_pretrained(config.model_path)
    
    # Load test dataloader
    processed_path = "dataset/processed"
    print(f"Loading test dataloader from {processed_path}...")
    
    try:
        _, test_loader = get_siamese_dataloaders(
            processed_path=processed_path,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            max_length=512
        )
        
        print(f"✓ Test dataloader loaded: {len(test_loader)} batches, {len(test_loader.dataset)} samples")
        
    except FileNotFoundError as e:
        print(f"❌ Data loading failed: {e}")
        print("Please run fraud data pipeline first to generate required data.")
        return {'error': str(e), 'status': 'failed'}
    
    # Step 3: Run evaluation
    print(f"\n{'─' * 60}")
    print("STEP 3: RUN EVALUATION")
    print(f"{'─' * 60}")
    
    print(f"Running comprehensive evaluation...")
    evaluation_results = run_siamese_evaluation(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=similarity_threshold,
        output_dir=str(output_dir)
    )
    
    print(f"✓ Evaluation completed")
    print(f"  ROC curve saved to: {output_dir / 'roc_curve.png'}")
    print(f"  Confusion matrix saved to: {output_dir / 'confusion_matrix.png'}")
    print(f"  Results saved to: {output_dir / 'evaluation_results.json'}")
    
    # Step 4: Per-type breakdown
    print(f"\n{'─' * 60}")
    print("STEP 4: PER-TYPE BREAKDOWN")
    print(f"{'─' * 60}")
    
    # Get per-type metrics
    overall_metrics = evaluation_results['overall_metrics']
    per_type_metrics = evaluation_results['per_type_metrics']
    
    print(f"Per-fraud-type performance:")
    print(f"{'Type':<18} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Count':<6}")
    print(f"{'─' * 60}")
    
    # Find best and worst performing types
    best_type = None
    worst_type = None
    best_f1 = -1
    worst_f1 = 2
    
    for type_name, metrics in per_type_metrics.items():
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        count = metrics['count']
        
        print(f"{type_name:<18} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {count:<6}")
        
        # Track best and worst
        if f1 > best_f1 and count > 0:  # Only consider types with samples
            best_f1 = f1
            best_type = type_name
        
        if f1 < worst_f1 and count > 0:  # Only consider types with samples
            worst_f1 = f1
            worst_type = type_name
    
    print(f"\n🏆 Best performing type: {best_type} (F1: {best_f1:.3f})")
    print(f"⚠️  Worst performing type: {worst_type} (F1: {worst_f1:.3f})")
    
    # Step 5: Print final report
    print(f"\n{'─' * 60}")
    print("STEP 5: FINAL REPORT")
    print(f"{'─' * 60}")
    
    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"{'─' * 30}")
    print(f"Precision: {overall_metrics['precision']:.3f}")
    print(f"Recall: {overall_metrics['recall']:.3f}")
    print(f"F1 Score: {overall_metrics['f1']:.3f}")
    print(f"Accuracy: {overall_metrics['accuracy']:.3f}")
    print(f"Threshold used: {overall_metrics['threshold_used']:.3f}")
    
    print(f"\n📈 CONFUSION MATRIX")
    print(f"{'─' * 30}")
    print(f"True Positives: {overall_metrics['true_positives']}")
    print(f"False Positives: {overall_metrics['false_positives']}")
    print(f"True Negatives: {overall_metrics['true_negatives']}")
    print(f"False Negatives: {overall_metrics['false_negatives']}")
    
    print(f"\n📁 OUTPUT FILES")
    print(f"{'─' * 30}")
    print(f"ROC Curve: {output_dir / 'roc_curve.png'}")
    print(f"Confusion Matrix: {output_dir / 'confusion_matrix.png'}")
    print(f"Results JSON: {output_dir / 'evaluation_results.json'}")
    
    # Create comprehensive results dictionary
    results = {
        'config': {
            'checkpoint_path': config.checkpoint_path,
            'model_path': config.model_path,
            'output_dir': config.output_dir,
            'batch_size': config.batch_size
        },
        'model_info': {
            'epoch': epoch,
            'validation_loss': val_loss,
            'similarity_threshold': similarity_threshold,
            'device': str(device)
        },
        'evaluation_results': evaluation_results,
        'summary': {
            'best_fraud_type': best_type,
            'worst_fraud_type': worst_type,
            'overall_f1': overall_metrics['f1'],
            'overall_precision': overall_metrics['precision'],
            'overall_recall': overall_metrics['recall']
        }
    }
    
    # Save comprehensive results
    comprehensive_results_path = output_dir / "comprehensive_evaluation_results.json"
    with open(comprehensive_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Comprehensive results saved to: {comprehensive_results_path}")
    
    print(f"\n🎉 Evaluation completed successfully!")
    print(f"📁 All results saved to: {output_dir}")
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained Siamese fraud detection model")
    
    # Required argument
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to trained model checkpoint"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model-path",
        default="dataset/raw/SROIE2019/layoutlm-base-uncased",
        help="Path to base LayoutLM model"
    )
    
    parser.add_argument(
        "--output-dir",
        default="artifacts/siamese/evaluation",
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = SiameseEvalConfig(
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    try:
        # Run evaluation pipeline
        results = run_siamese_eval_pipeline(config)
        
        print(f"\n{'=' * 80}")
        print("🚀 Siamese evaluation pipeline completed successfully!")
        print(f"📁 Results saved to: {config.output_dir}")
        print(f"🎯 Overall F1 Score: {results['summary']['overall_f1']:.3f}")
        print(f"{'=' * 80}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Evaluation pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
