"""
Evaluation pipeline for ReceiptGuard-ML.
NER evaluation + fraud detection orchestration.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from dataset import ReceiptDataset, collate_fn, ID2LABEL
from dataloader import get_tokenizer, ReceiptLedger
from model import ModelConfig, ReceiptFieldExtractor, load_checkpoint
from train import get_device
from evaluation import (
    run_full_evaluation,
    compute_ner_metrics,
    extract_entities_from_predictions,
    plot_confusion_matrix,
)

# Import configuration
from src.config import CFG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline."""
    checkpoint_path: str  # Required - path to best checkpoint
    model_path: str = None
    processed_data_path: str = None
    output_dir: str = None
    batch_size: int = None
    run_fraud_detection: bool = True

    def __post_init__(self):
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required")
        if self.model_path is None:
            self.model_path = CFG.model.model_path
        if self.processed_data_path is None:
            self.processed_data_path = CFG.paths.processed_data_path
        if self.output_dir is None:
            self.output_dir = CFG.paths.evaluation_dir
        if self.batch_size is None:
            self.batch_size = CFG.inference.batch_size


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> ReceiptFieldExtractor:
    """
    Step 1 — Load model from checkpoint.

    Loads ModelConfig from checkpoint, rebuilds model, loads weights.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model onto

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Infer model config from checkpoint path or use defaults
    # Since checkpoint only has state dict, we need to reconstruct config
    # We'll use the default model path and 9 labels for SROIE2019
    model_config = ModelConfig(
        model_path="dataset/raw/SROIE2019/layoutlm-base-uncased",
        num_labels=9,  # O, B-COMPANY, I-COMPANY, B-DATE, I-DATE, B-ADDRESS, I-ADDRESS, B-TOTAL, I-TOTAL
        dropout=0.1,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=0,
        max_length=512,
    )

    # Build model
    model = ReceiptFieldExtractor(
        model_path=model_config.model_path,
        num_labels=model_config.num_labels,
        dropout=model_config.dropout,
    )

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device and set eval mode
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Model is on device: {device}")

    return model


def build_test_dataloader(config: EvaluationConfig) -> DataLoader:
    """
    Build test dataloader for evaluation.

    Args:
        config: Evaluation configuration

    Returns:
        Test DataLoader
    """
    logger.info("Building test dataloader...")

    # Try to load processed data first
    processed_test_file = Path(config.processed_data_path) / "test_samples.json"
    data_path = config.processed_data_path if processed_test_file.exists() else "dataset/raw/SROIE2019"

    # Build dataset
    test_dataset = ReceiptDataset(
        data_path=data_path,
        split="test",
        tokenizer_name=config.model_path,
        max_length=512,
    )

    # Build dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Test dataset: {len(test_dataset)} samples")
    logger.info(f"Test batches: {len(test_loader)} (batch_size={config.batch_size})")

    return test_loader


def run_ner_evaluation(model: ReceiptFieldExtractor, test_loader: DataLoader,
                       device: torch.device, output_dir: Path) -> Dict:
    """
    Step 2 — Run NER evaluation on test set.

    Args:
        model: Loaded model
        test_loader: Test dataloader
        device: Device
        output_dir: Output directory for results

    Returns:
        Evaluation results dictionary
    """
    logger.info("Running NER evaluation on test set...")

    # Run full evaluation
    results = run_full_evaluation(
        model=model,
        dataloader=test_loader,
        device=device,
        id2label=ID2LABEL,
        output_dir=str(output_dir),
    )

    # Print metrics table
    print("\n" + "=" * 70)
    print("NER Evaluation Results")
    print("=" * 70)

    ner_metrics = results["ner_metrics"]

    # Per-entity scores
    print("\nPer-Entity Metrics:")
    print("-" * 70)
    print(f"{'Entity':<12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("-" * 70)

    for entity, metrics in ner_metrics["per_entity"].items():
        print(f"{entity:<12} {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} {metrics['f1']:>12.4f}")

    # Macro average
    print("-" * 70)
    macro = ner_metrics["macro"]
    print(f"{'MACRO':<12} {macro['precision']:>12.4f} {macro['recall']:>12.4f} {macro['f1']:>12.4f}")
    print("=" * 70)

    # Additional metrics
    print(f"\nToken Accuracy: {results['token_accuracy']:.4f}")
    print(f"Average Loss: {results['avg_loss']:.4f}")

    return results


def run_fraud_detection_pipeline(config: EvaluationConfig, test_loader: DataLoader,
                                 model: ReceiptFieldExtractor, device: torch.device) -> Dict:
    """
    Step 3 — Run fraud detection on test set.

    For each test receipt, extracts entities and checks through ReceiptLedger.

    Args:
        config: Evaluation configuration
        test_loader: Test dataloader
        model: Loaded model
        device: Device

    Returns:
        Fraud detection report
    """
    logger.info("Running fraud detection pipeline...")

    # Initialize ledger
    ledger_path = Path(config.output_dir) / "evaluation_ledger.json"
    ledger = ReceiptLedger(str(ledger_path))

    # Collect predictions and detect duplicates
    duplicate_results = []
    processed_count = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
            bbox = batch["bbox"].to(device)

            # Forward pass
            _, logits = model(input_ids, attention_mask, token_type_ids, bbox)

            # Get predictions
            predictions = model.get_predictions(logits, attention_mask)

            # Process each sample
            for i in range(len(predictions)):
                receipt_id = f"test_receipt_{processed_count}"
                processed_count += 1

                # Get tokens
                mask = attention_mask[i] == 1
                valid_length = mask.sum().item()
                token_ids = input_ids[i][:valid_length].cpu().tolist()

                # Get tokenizer from dataset
                tokenizer = test_loader.dataset.tokenizer if hasattr(test_loader.dataset, "tokenizer") else None
                if tokenizer:
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                    tokens = [t.replace("##", "").replace("[CLS]", "").replace("[SEP]", "") for t in tokens if t not in ["", "[CLS]", "[SEP]"]]
                else:
                    tokens = [f"tok_{j}" for j in range(len(predictions[i]))]

                # Extract entities
                pred_ids = predictions[i]
                entities = extract_entities_from_predictions(tokens, pred_ids, ID2LABEL)

                # Check in ledger
                result = ledger.check_and_register(receipt_id, entities)

                if result["is_duplicate"]:
                    duplicate_results.append({
                        "receipt_id": receipt_id,
                        "fingerprint": result["fingerprint"],
                        "existing_record": result["existing_record"],
                        "extracted_entities": entities,
                    })

    # Save ledger
    ledger.save()

    # Build fraud report
    fraud_report = {
        "total_processed": processed_count,
        "duplicates_found": len(duplicate_results),
        "duplicate_rate": len(duplicate_results) / processed_count if processed_count > 0 else 0.0,
        "duplicates": duplicate_results,
        "ledger_path": str(ledger_path),
    }

    # Print fraud report
    print("\n" + "=" * 70)
    print("Fraud Detection Report")
    print("=" * 70)
    print(f"Total receipts processed: {processed_count}")
    print(f"Duplicates detected: {len(duplicate_results)}")
    print(f"Duplicate rate: {fraud_report['duplicate_rate']:.2%}")

    if duplicate_results:
        print("\nDuplicate Receipts:")
        print("-" * 70)
        for dup in duplicate_results[:10]:  # Show first 10
            print(f"  {dup['receipt_id']} matches {dup['existing_record']['receipt_id']}")
            print(f"    Fingerprint: {dup['fingerprint'][:16]}...")
            print(f"    Entities: {dup['extracted_entities']}")
        if len(duplicate_results) > 10:
            print(f"  ... and {len(duplicate_results) - 10} more")

    print("=" * 70)

    return fraud_report


def save_evaluation_outputs(output_dir: Path, ner_results: Dict, fraud_report: Dict,
                            test_loader: DataLoader, model: ReceiptFieldExtractor,
                            device: torch.device) -> None:
    """
    Step 4 — Save evaluation outputs.

    Args:
        output_dir: Output directory
        ner_results: NER evaluation results
        fraud_report: Fraud detection report
        test_loader: Test dataloader
        model: Loaded model
        device: Device
    """
    logger.info("Saving evaluation outputs...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save full evaluation results JSON
    results_file = output_dir / "evaluation_results.json"
    full_results = {
        "ner_metrics": ner_results["ner_metrics"],
        "token_accuracy": ner_results["token_accuracy"],
        "avg_loss": ner_results["avg_loss"],
        "num_samples": ner_results["num_samples"],
        "fraud_report": {
            "total_processed": fraud_report["total_processed"],
            "duplicates_found": fraud_report["duplicates_found"],
            "duplicate_rate": fraud_report["duplicate_rate"],
        },
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation results saved to {results_file}")

    # 2. Save confusion matrix PNG
    # Collect all true and predicted labels
    all_true_labels = []
    all_pred_labels = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
            bbox = batch["bbox"].to(device)
            labels = batch.get("labels", None)

            _, logits = model(input_ids, attention_mask, token_type_ids, bbox, labels)
            predictions = model.get_predictions(logits, attention_mask)

            # Collect labels
            for i in range(len(predictions)):
                mask = attention_mask[i] == 1
                valid_length = mask.sum().item()

                if labels is not None:
                    true_ids = labels[i][:valid_length].cpu().tolist()
                    true_labels = [ID2LABEL.get(lid, "O") for lid in true_ids if lid != -100]
                else:
                    true_labels = []

                pred_ids = predictions[i]
                pred_labels = [ID2LABEL.get(pid, "O") for pid in pred_ids]

                all_true_labels.extend(true_labels)
                all_pred_labels.extend(pred_labels)

    # Plot confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(all_true_labels, all_pred_labels, str(cm_path))

    # 3. Save fraud report JSON
    fraud_file = output_dir / "fraud_report.json"
    with open(fraud_file, "w", encoding="utf-8") as f:
        json.dump(fraud_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Fraud report saved to {fraud_file}")


def run_evaluation_pipeline(config: EvaluationConfig) -> Dict:
    """
    Run the complete evaluation pipeline.

    Steps:
    1. Load model from checkpoint
    2. Run NER evaluation on test set
    3. Run fraud detection (optional)
    4. Save outputs
    5. Return summary

    Args:
        config: EvaluationConfig

    Returns:
        Summary dictionary with ner_metrics, fraud_report, output_dir
    """
    logger.info("=" * 70)
    logger.info("Starting ReceiptGuard-ML Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {config.checkpoint_path}")
    logger.info(f"Fraud detection: {'enabled' if config.run_fraud_detection else 'disabled'}")
    logger.info(f"Output directory: {config.output_dir}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1 — Load model from checkpoint
    # =========================================================================
    logger.info("\nStep 1: Loading model from checkpoint...")
    device = get_device()
    model = load_model_from_checkpoint(config.checkpoint_path, device)

    # =========================================================================
    # Step 2 — Build test dataloader
    # =========================================================================
    logger.info("\nStep 2: Building test dataloader...")
    test_loader = build_test_dataloader(config)

    # =========================================================================
    # Step 3 — Run NER evaluation
    # =========================================================================
    logger.info("\nStep 3: Running NER evaluation...")
    ner_results = run_ner_evaluation(model, test_loader, device, output_dir)

    # =========================================================================
    # Step 4 — Run fraud detection (optional)
    # =========================================================================
    fraud_report = {
        "total_processed": 0,
        "duplicates_found": 0,
        "duplicate_rate": 0.0,
        "duplicates": [],
        "ledger_path": "",
        "status": "skipped",
    }

    if config.run_fraud_detection:
        logger.info("\nStep 4: Running fraud detection...")
        fraud_report = run_fraud_detection_pipeline(config, test_loader, model, device)
        fraud_report["status"] = "completed"
    else:
        logger.info("\nStep 4: Skipping fraud detection (disabled)")

    # =========================================================================
    # Step 5 — Save outputs
    # =========================================================================
    logger.info("\nStep 5: Saving outputs...")
    save_evaluation_outputs(output_dir, ner_results, fraud_report, test_loader, model, device)

    # =========================================================================
    # Step 6 — Print final summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Evaluation Pipeline Complete!")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"NER Macro F1: {ner_results['ner_metrics']['macro']['f1']:.4f}")
    if config.run_fraud_detection:
        print(f"Duplicates detected: {fraud_report['duplicates_found']}")
    print("=" * 70)

    # Build summary
    summary = {
        "ner_metrics": ner_results["ner_metrics"],
        "fraud_report": fraud_report,
        "output_dir": str(output_dir),
        "checkpoint_path": config.checkpoint_path,
        "config": asdict(config),
    }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReceiptGuard-ML Evaluation Pipeline"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (required)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="dataset/raw/SROIE2019/layoutlm-base-uncased",
        help="Path to LayoutLM model"
    )
    parser.add_argument(
        "--processed_data_path",
        type=str,
        default="dataset/processed",
        help="Path to processed data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/processed/evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--no_fraud_detection",
        action="store_true",
        help="Skip fraud detection"
    )

    args = parser.parse_args()

    config = EvaluationConfig(
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        processed_data_path=args.processed_data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        run_fraud_detection=not args.no_fraud_detection,
    )

    summary = run_evaluation_pipeline(config)

    # Exit with success
    sys.exit(0)
