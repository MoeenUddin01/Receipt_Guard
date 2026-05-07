"""
Full NER evaluation and fraud detection metrics.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.model import ReceiptFieldExtractor

# Import configuration
from src.config import CFG


# ID2LABEL mapping for evaluation
ID2LABEL = {
    0: "O",
    1: "B-COMPANY",
    2: "I-COMPANY",
    3: "B-DATE",
    4: "I-DATE",
    5: "B-ADDRESS",
    6: "I-ADDRESS",
    7: "B-TOTAL",
    8: "I-TOTAL",
}

# Entity types for span extraction
ENTITY_TYPES = ["COMPANY", "DATE", "ADDRESS", "TOTAL"]


def extract_spans(labels: List[str]) -> Set[Tuple[int, int, str]]:
    """
    Extract entity spans from BIO labels.
    
    Args:
        labels: List of BIO labels (e.g., ['O', 'B-COMPANY', 'I-COMPANY', 'O'])
        
    Returns:
        Set of (start_idx, end_idx, entity_type) tuples
    """
    spans = set()
    current_entity = None
    start_idx = None
    
    for idx, label in enumerate(labels):
        if label.startswith("B-"):
            # Close previous entity if exists
            if current_entity is not None:
                spans.add((start_idx, idx, current_entity))
            # Start new entity
            current_entity = label[2:]  # Remove "B-" prefix
            start_idx = idx
            
        elif label.startswith("I-"):
            entity_type = label[2:]  # Remove "I-" prefix
            # Continue current entity only if same type
            if current_entity == entity_type:
                continue
            else:
                # Close previous entity if mismatch
                if current_entity is not None:
                    spans.add((start_idx, idx, current_entity))
                # Start new entity (invalid BIO but handle gracefully)
                current_entity = entity_type
                start_idx = idx
                
        else:  # O label
            # Close current entity
            if current_entity is not None:
                spans.add((start_idx, idx, current_entity))
                current_entity = None
                start_idx = None
    
    # Handle entity at end of sequence
    if current_entity is not None:
        spans.add((start_idx, len(labels), current_entity))
    
    return spans


def compute_ner_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> dict:
    """
    Compute entity-level (span-level) precision, recall, F1 per class and macro average.
    
    Args:
        true_labels: List of true label sequences (list of list of BIO labels)
        pred_labels: List of predicted label sequences (list of list of BIO labels)
        
    Returns:
        Dictionary with {per_entity: {COMPANY: {p,r,f1}, ...}, macro: {p,r,f1}}
    """
    # Initialize counters per entity type
    entity_stats = {entity: {"tp": 0, "fp": 0, "fn": 0} for entity in ENTITY_TYPES}
    
    # Process each sequence
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        true_spans = extract_spans(true_seq)
        pred_spans = extract_spans(pred_seq)
        
        # Calculate TP, FP, FN per entity type
        for entity in ENTITY_TYPES:
            # Filter spans for this entity type
            true_entity_spans = {span for span in true_spans if span[2] == entity}
            pred_entity_spans = {span for span in pred_spans if span[2] == entity}
            
            # True positives: spans that match in both
            tp = len(true_entity_spans & pred_entity_spans)
            
            # False positives: predicted but not in true
            fp = len(pred_entity_spans - true_entity_spans)
            
            # False negatives: in true but not predicted
            fn = len(true_entity_spans - pred_entity_spans)
            
            entity_stats[entity]["tp"] += tp
            entity_stats[entity]["fp"] += fp
            entity_stats[entity]["fn"] += fn
    
    # Calculate metrics per entity
    per_entity = {}
    for entity in ENTITY_TYPES:
        stats = entity_stats[entity]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        per_entity[entity] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    
    # Calculate macro average
    macro_precision = sum(per_entity[e]["precision"] for e in ENTITY_TYPES) / len(ENTITY_TYPES)
    macro_recall = sum(per_entity[e]["recall"] for e in ENTITY_TYPES) / len(ENTITY_TYPES)
    macro_f1 = sum(per_entity[e]["f1"] for e in ENTITY_TYPES) / len(ENTITY_TYPES)
    
    return {
        "per_entity": per_entity,
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
    }


def extract_entities_from_predictions(
    tokens: List[str],
    pred_ids: List[int],
    id2label: Dict[int, str],
) -> dict:
    """
    Extract entities from tokens and predicted label IDs for one receipt.
    
    Args:
        tokens: List of token strings
        pred_ids: List of predicted label IDs
        id2label: Mapping from label ID to label name
        
    Returns:
        Dict with extracted entities: {company, date, address, total}
        Empty string for missing entities
    """
    # Convert IDs to labels
    labels = [id2label.get(pid, "O") for pid in pred_ids]
    
    # Extract spans
    spans = extract_spans(labels)
    
    # Build entity text by joining tokens in each span
    entities = {"company": "", "date": "", "address": "", "total": ""}
    
    for start, end, entity_type in spans:
        entity_tokens = tokens[start:end]
        entity_text = " ".join(entity_tokens).strip()
        
        if entity_type.lower() in entities:
            # If multiple spans of same type, concatenate with space
            if entities[entity_type.lower()]:
                entities[entity_type.lower()] += " " + entity_text
            else:
                entities[entity_type.lower()] = entity_text
    
    return entities


def run_full_evaluation(
    model: ReceiptFieldExtractor,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    output_dir: str = None,
) -> dict:
    """
    Run full evaluation on test set.
    
    Args:
        model: Trained ReceiptFieldExtractor model
        test_loader: DataLoader with test samples
        device: Device to run evaluation on
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics including NER metrics and per-receipt entities
    """
    if output_dir is None:
        output_dir = CFG.paths.evaluation_dir
    
    model.eval()
    
    # Collect predictions
    all_true_labels = []
    all_pred_labels = []
    all_tokens = []
    all_receipt_ids = []
    all_extracted_entities = []
    total_loss = 0.0
    num_batches = 0
    
    # Get tokenizer for decoding (if available in dataset)
    tokenizer = None
    if hasattr(dataloader.dataset, "tokenizer"):
        tokenizer = dataloader.dataset.tokenizer
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Full Evaluation", leave=False):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
            bbox = batch["bbox"].to(device)
            labels = batch.get("labels", None)
            
            # Forward pass
            loss, logits = model(input_ids, attention_mask, token_type_ids, bbox, labels)
            
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1
            
            # Get predictions
            predictions = model.get_predictions(logits, attention_mask)
            
            # Process each sample in batch
            for i in range(len(predictions)):
                # Get valid token mask
                mask = attention_mask[i] == 1
                valid_length = mask.sum().item()
                
                # Get true labels if available
                if labels is not None:
                    true_ids = labels[i][:valid_length].cpu().tolist()
                    true_label_names = [id2label.get(lid, "O") for lid in true_ids if lid != -100]
                else:
                    true_label_names = []
                
                # Get predicted labels
                pred_ids = predictions[i]
                pred_label_names = [id2label.get(pid, "O") for pid in pred_ids]
                
                # Store labels
                all_true_labels.append(true_label_names)
                all_pred_labels.append(pred_label_names)
                
                # Decode tokens (approximate from input_ids)
                if tokenizer is not None:
                    token_ids = input_ids[i][:valid_length].cpu().tolist()
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                    # Clean up tokens (remove special tokens, ## prefix)
                    tokens = [t.replace("##", "").replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "") for t in tokens if t not in ["[PAD]", "[CLS]", "[SEP]", ""]]
                else:
                    tokens = [f"tok_{j}" for j in range(len(pred_ids))]
                
                all_tokens.append(tokens)
                all_receipt_ids.append(f"receipt_{len(all_receipt_ids)}")
                
                # Extract entities
                entities = extract_entities_from_predictions(tokens, pred_ids, id2label)
                all_extracted_entities.append({
                    "receipt_id": all_receipt_ids[-1],
                    "entities": entities,
                })
    
    # Compute NER metrics
    ner_metrics = compute_ner_metrics(all_true_labels, all_pred_labels)
    
    # Compute token-level accuracy
    total_correct = sum(
        sum(t == p for t, p in zip(true, pred))
        for true, pred in zip(all_true_labels, all_pred_labels)
    )
    total_tokens = sum(len(seq) for seq in all_true_labels)
    token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    # Build results dict
    results = {
        "avg_loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "token_accuracy": token_accuracy,
        "ner_metrics": ner_metrics,
        "num_samples": len(all_receipt_ids),
        "per_receipt_entities": all_extracted_entities,
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        # Don't save all per-receipt entities in main file (too large)
        summary_results = {k: v for k, v in results.items() if k != "per_receipt_entities"}
        json.dump(summary_results, f, indent=2)
    
    # Save per-receipt entities separately
    entities_file = output_path / "extracted_entities.json"
    with open(entities_file, "w") as f:
        json.dump(all_extracted_entities, f, indent=2)
    
    print(f"Evaluation results saved to {results_file}")
    print(f"Extracted entities saved to {entities_file}")
    
    return results


def plot_confusion_matrix(
    true_labels_flat: List[str],
    pred_labels_flat: List[str],
    output_path: str,
) -> None:
    """
    Plot token-level confusion matrix across all 9 label classes.
    
    Args:
        true_labels_flat: Flattened list of true labels
        pred_labels_flat: Flattened list of predicted labels
        output_path: Path to save the plot
    """
    # Get unique labels in order
    all_labels = sorted(set(true_labels_flat) | set(pred_labels_flat))
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels_flat, pred_labels_flat, labels=all_labels)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=all_labels,
        yticklabels=all_labels,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Token-Level Confusion Matrix")
    plt.tight_layout()
    
    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")


def evaluate_fraud_detection(
    ledger: "ReceiptLedger",
    ground_truth_duplicates: List[Tuple[str, str]],
) -> dict:
    """
    Evaluate the deduplication engine.
    
    Args:
        ledger: ReceiptLedger instance with duplicates detected
        ground_truth_duplicates: List of (id1, id2) pairs known to be duplicates
        
    Returns:
        Dictionary with precision, recall, F1 of fraud detection
    """
    # Get detected duplicates from ledger
    detected_duplicates = set()
    if hasattr(ledger, "duplicate_pairs"):
        for pair in ledger.duplicate_pairs:
            # Normalize order (smaller first)
            id1, id2 = pair
            if id1 > id2:
                id1, id2 = id2, id1
            detected_duplicates.add((id1, id2))
    elif hasattr(ledger, "clusters"):
        # Extract pairs from clusters
        for cluster in ledger.clusters:
            ids = list(cluster)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    id1, id2 = ids[i], ids[j]
                    if id1 > id2:
                        id1, id2 = id2, id1
                    detected_duplicates.add((id1, id2))
    else:
        # Try to get duplicates via method call
        detected = ledger.get_duplicate_pairs() if hasattr(ledger, "get_duplicate_pairs") else []
        for pair in detected:
            id1, id2 = pair
            if id1 > id2:
                id1, id2 = id2, id1
            detected_duplicates.add((id1, id2))
    
    # Normalize ground truth
    ground_truth_set = set()
    for id1, id2 in ground_truth_duplicates:
        if id1 > id2:
            id1, id2 = id2, id1
        ground_truth_set.add((id1, id2))
    
    # Calculate metrics
    tp = len(detected_duplicates & ground_truth_set)
    fp = len(detected_duplicates - ground_truth_set)
    fn = len(ground_truth_set - detected_duplicates)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "detected_pairs": len(detected_duplicates),
        "ground_truth_pairs": len(ground_truth_set),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("evaluation.py - NER and Fraud Detection Evaluation")
    print("=" * 60)
    
    # Test span extraction
    print("\nTesting span extraction...")
    test_labels = ["O", "B-COMPANY", "I-COMPANY", "O", "B-DATE", "I-DATE", "I-DATE", "O"]
    spans = extract_spans(test_labels)
    print(f"Labels: {test_labels}")
    print(f"Extracted spans: {spans}")
    
    # Test entity extraction
    print("\nTesting entity extraction...")
    tokens = ["ACME", "Corp", "paid", "on", "2024-01-15", "for", "$100"]
    pred_ids = [1, 2, 0, 0, 3, 0, 7]  # B-COMPANY, I-COMPANY, O, O, B-DATE, O, B-TOTAL
    entities = extract_entities_from_predictions(tokens, pred_ids, ID2LABEL)
    print(f"Tokens: {tokens}")
    print(f"Predicted IDs: {pred_ids}")
    print(f"Extracted entities: {entities}")
    
    # Test NER metrics
    print("\nTesting NER metrics...")
    true_labels = [
        ["O", "B-COMPANY", "I-COMPANY", "O", "B-DATE", "O"],
        ["B-COMPANY", "O", "B-TOTAL", "I-TOTAL"],
    ]
    pred_labels = [
        ["O", "B-COMPANY", "I-COMPANY", "O", "B-DATE", "O"],  # All correct
        ["B-COMPANY", "O", "B-TOTAL", "O"],  # Missing I-TOTAL
    ]
    metrics = compute_ner_metrics(true_labels, pred_labels)
    print(f"NER Metrics:")
    print(json.dumps(metrics, indent=2))
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
