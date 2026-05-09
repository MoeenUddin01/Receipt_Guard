"""
evaluation.py - Full evaluation metrics for Siamese fraud detection model.

This module provides comprehensive evaluation utilities including precision/recall/F1,
per-fraud-type analysis, ROC curves, confusion matrices, and detailed reporting.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from tqdm import tqdm

from .siamese_model import SiameseSimilarityModel


def compute_fraud_metrics(
    similarities: List[float],
    labels: List[int],
    threshold: float
) -> Dict:
    """
    Compute comprehensive fraud detection metrics.
    
    Args:
        similarities: List of cosine similarity scores
        labels: List of binary labels (0=legitimate, 1=fraud)
        threshold: Similarity threshold for fraud prediction
        
    Returns:
        Dictionary with all metrics and confusion matrix components
    """
    # Predict fraud where similarity > threshold
    predictions = [1 if sim > threshold else 0 for sim in similarities]
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    # Calculate metrics
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'threshold_used': threshold
    }


def compute_per_type_metrics(
    similarities: List[float],
    labels: List[int],
    pair_types: List[int],
    threshold: float
) -> Dict:
    """
    Break down metrics by FraudPairType.
    
    Args:
        similarities: List of cosine similarity scores
        labels: List of binary labels (0=legitimate, 1=fraud)
        pair_types: List of pair type integers (FraudPairType values)
        threshold: Similarity threshold for fraud prediction
        
    Returns:
        Dictionary with metrics for each FraudPairType
    """
    from ..data.fraud_dataset import FraudPairType
    
    # Predict fraud where similarity > threshold
    predictions = [1 if sim > threshold else 0 for sim in similarities]
    
    # Initialize results dictionary
    results = {}
    
    # Process each fraud type
    for fraud_type in FraudPairType:
        type_indices = [i for i, pt in enumerate(pair_types) if pt == fraud_type.value]
        
        if not type_indices:
            continue
        
        # Extract data for this type
        type_labels = [labels[i] for i in type_indices]
        type_predictions = [predictions[i] for i in type_indices]
        
        # Calculate metrics
        precision = precision_score(type_labels, type_predictions, zero_division=0)
        recall = recall_score(type_labels, type_predictions, zero_division=0)
        f1 = f1_score(type_labels, type_predictions, zero_division=0)
        
        results[fraud_type.name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'count': len(type_indices)
        }
    
    return results


def plot_roc_curve(similarities: List[float], labels: List[int], output_path: str):
    """
    Plot ROC curve with AUC score and mark chosen threshold.
    
    Args:
        similarities: List of cosine similarity scores
        labels: List of binary labels
        output_path: Path to save the plot
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Find the threshold point closest to 0.5 (or use Youden's J statistic)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    
    # Mark the optimal threshold point
    plt.scatter(best_fpr, best_tpr, color='red', s=100, zorder=5,
               label=f'Optimal threshold = {best_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Fraud Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to {output_path}")
    print(f"  AUC: {roc_auc:.3f}")
    print(f"  Optimal threshold: {best_threshold:.3f}")


def plot_confusion_matrix(predictions: List[int], labels: List[int], output_path: str):
    """
    Plot 2x2 confusion matrix for fraud detection.
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        output_path: Path to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Create plot
    plt.figure(figsize=(6, 5))
    
    # Use seaborn for better visualization
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Add text annotations
    plt.text(0.5, -0.15, f'TN: {cm[0,0]}', ha='center', transform=plt.gca().transAxes)
    plt.text(1.5, -0.15, f'FP: {cm[0,1]}', ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, 1.15, f'FN: {cm[1,0]}', ha='center', transform=plt.gca().transAxes)
    plt.text(1.5, 1.15, f'TP: {cm[1,1]}', ha='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")


def run_siamese_evaluation(
    model: SiameseSimilarityModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float,
    output_dir: str
) -> Dict:
    """
    Full evaluation run with all metrics and plots.
    
    Args:
        model: SiameseSimilarityModel instance
        dataloader: Test/evaluation dataloader
        device: Device to run evaluation on
        threshold: Similarity threshold for fraud prediction
        output_dir: Directory to save results and plots
        
    Returns:
        Dictionary with all evaluation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Siamese evaluation...")
    print(f"  Device: {device}")
    print(f"  Threshold: {threshold}")
    print(f"  Output directory: {output_dir}")
    
    # Collect all predictions and similarities
    model.eval()
    all_similarities = []
    all_labels = []
    all_pair_types = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            receipt_a = {k: v.to(device) for k, v in batch['receipt_a'].items()}
            receipt_b = {k: v.to(device) for k, v in batch['receipt_b'].items()}
            labels = batch['labels'].to(device)
            pair_types = batch['pair_types']
            
            # Get similarity scores
            similarity_scores = model.get_similarity(receipt_a, receipt_b)
            
            # Make predictions
            predictions = (similarity_scores > threshold).long()
            
            # Collect results
            all_similarities.extend(similarity_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_pair_types.extend(pair_types)
            all_predictions.extend(predictions.cpu().numpy())
    
    print(f"  Evaluated {len(all_labels)} samples")
    
    # Compute overall metrics
    overall_metrics = compute_fraud_metrics(all_similarities, all_labels, threshold)
    
    # Compute per-type metrics
    per_type_metrics = compute_per_type_metrics(
        all_similarities, all_labels, all_pair_types, threshold
    )
    
    # Generate plots
    plot_roc_curve(all_similarities, all_labels, str(output_dir / "roc_curve.png"))
    plot_confusion_matrix(all_predictions, all_labels, str(output_dir / "confusion_matrix.png"))
    
    # Prepare results dictionary
    results = {
        'overall_metrics': overall_metrics,
        'per_type_metrics': per_type_metrics,
        'threshold_used': threshold,
        'total_samples': len(all_labels)
    }
    
    # Save results to JSON
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Print clean summary table
    print_summary_table(overall_metrics, per_type_metrics)
    
    return results


def print_summary_table(overall_metrics: Dict, per_type_metrics: Dict):
    """Print a clean summary table to console."""
    
    # Overall metrics table
    print("\n" + "─" * 40)
    print("│" + " " * 10 + "Fraud Detection Results" + " " * 9 + "│")
    print("─" * 40)
    
    metrics_to_show = [
        ('Precision', overall_metrics['precision']),
        ('Recall', overall_metrics['recall']),
        ('F1', overall_metrics['f1']),
        ('Accuracy', overall_metrics['accuracy']),
        ('Threshold', overall_metrics['threshold_used'])
    ]
    
    for metric_name, value in metrics_to_show:
        if metric_name == 'Threshold':
            print(f"│  {metric_name:<15} │   {value:.2f}   │")
        else:
            print(f"│  {metric_name:<15} │   {value:.2f}   │")
    
    print("─" * 40)
    
    # Confusion matrix summary
    print(f"│  True Positives   │   {overall_metrics['true_positives']:>4}   │")
    print(f"│  False Positives  │   {overall_metrics['false_positives']:>4}   │")
    print(f"│  True Negatives   │   {overall_metrics['true_negatives']:>4}   │")
    print(f"│  False Negatives  │   {overall_metrics['false_negatives']:>4}   │")
    print("─" * 40)
    
    # Per-type metrics
    print("\nPer-Type Performance:")
    print("─" * 50)
    print(f"{'Type':<18} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Count':<6}")
    print("─" * 50)
    
    for type_name, metrics in per_type_metrics.items():
        print(f"{type_name:<18} {metrics['precision']:<10.3f} "
              f"{metrics['recall']:<10.3f} {metrics['f1']:<10.3f} "
              f"{metrics['count']:<6}")
    
    print("─" * 50)


if __name__ == "__main__":
    """Test the evaluation utilities."""
    
    print("=" * 60)
    print("Testing Siamese Evaluation Utilities")
    print("=" * 60)
    
    try:
        # Test with dummy data
        print("\n1. Testing compute_fraud_metrics:")
        
        # Create dummy data where fraud has higher similarity
        np.random.seed(42)
        legit_similarities = np.random.normal(0.3, 0.1, 100).clip(0, 1)
        fraud_similarities = np.random.normal(0.7, 0.1, 100).clip(0, 1)
        
        similarities = np.concatenate([legit_similarities, fraud_similarities]).tolist()
        labels = [0] * 100 + [1] * 100
        
        threshold = 0.5
        metrics = compute_fraud_metrics(similarities, labels, threshold)
        
        print(f"   Threshold: {threshold}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   F1: {metrics['f1']:.3f}")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   TP/FP/TN/FN: {metrics['true_positives']}/{metrics['false_positives']}/{metrics['true_negatives']}/{metrics['false_negatives']}")
        
        # Test per-type metrics
        print("\n2. Testing compute_per_type_metrics:")
        
        # Create dummy pair types
        from ..data.fraud_dataset import FraudPairType
        pair_types = []
        
        # Add legitimate pairs
        pair_types.extend([FraudPairType.DIFFERENT_RECEIPT.value] * 50)
        
        # Add fraud pairs of different types
        pair_types.extend([FraudPairType.EXACT_DUPLICATE.value] * 25)
        pair_types.extend([FraudPairType.DATE_TAMPERED.value] * 25)
        pair_types.extend([FraudPairType.TOTAL_TAMPERED.value] * 25)
        pair_types.extend([FraudPairType.COMPANY_TYPO.value] * 25)
        
        # Adjust labels to match
        labels = [0] * 50 + [1] * 100
        
        # Adjust similarities to match
        legit_sims = np.random.normal(0.3, 0.1, 50).clip(0, 1)
        fraud_sims = np.random.normal(0.7, 0.1, 100).clip(0, 1)
        similarities = np.concatenate([legit_sims, fraud_sims]).tolist()
        
        per_type = compute_per_type_metrics(similarities, labels, pair_types, threshold)
        
        for type_name, metrics in per_type.items():
            print(f"   {type_name}: F1={metrics['f1']:.3f}, Count={metrics['count']}")
        
        # Test plotting functions
        print("\n3. Testing plotting functions:")
        
        output_dir = Path("/tmp/test_evaluation")
        output_dir.mkdir(exist_ok=True)
        
        # Test ROC curve
        plot_roc_curve(similarities, labels, str(output_dir / "test_roc.png"))
        
        # Test confusion matrix
        predictions = [1 if sim > threshold else 0 for sim in similarities]
        plot_confusion_matrix(predictions, labels, str(output_dir / "test_cm.png"))
        
        print(f"   Test plots saved to {output_dir}")
        
        # Test summary table
        print("\n4. Testing summary table:")
        print_summary_table(metrics, per_type)
        
        print("\n✓ All evaluation utilities working correctly!")
        
    except Exception as e:
        print(f"\n   ⚠️  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)
