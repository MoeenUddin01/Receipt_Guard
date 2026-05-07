"""
ReceiptGuard-ML — Main CLI Entry Point

Single CLI entry point for the entire ReceiptGuard-ML project.
Provides subcommands for preprocessing, training, evaluation, and full pipeline.

Usage examples:
    # Full pipeline
    receiptguard full

    # Just train with custom settings
    receiptguard train --num_epochs 10 --batch_size 4 --learning_rate 3e-5

    # Evaluate a checkpoint
    receiptguard evaluate --checkpoint_path dataset/processed/checkpoints/best_model.pt

    # Preprocess only
    receiptguard preprocess --raw_path /path/to/data --processed_path /path/to/output
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src" / "pipelines"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "data"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "model"))

from preprocessing_pipeline import (
    run_preprocessing_pipeline,
    PreprocessingConfig,
)
from model_training_pipeline import (
    run_training_pipeline,
    TrainingConfig,
)
from evaluation_pipeline import (
    run_evaluation_pipeline,
    EvaluationConfig,
)

__version__ = "0.1.0"


def print_banner(title: str) -> None:
    """Print a formatted header banner."""
    width = 60
    print()
    print("=" * width)
    print(f"  ReceiptGuard-ML — {title}")
    print("=" * width)
    print()


def print_summary(pipeline_name: str, elapsed: float, status: str = "completed") -> None:
    """Print a one-line summary with elapsed time."""
    print(f"\n[{pipeline_name}] {status} in {elapsed:.2f}s")


def cmd_preprocess(args) -> int:
    """Run the preprocessing pipeline."""
    print_banner("Preprocessing Pipeline")

    start_time = time.time()

    config = PreprocessingConfig(
        raw_data_path=args.raw_path,
        processed_data_path=args.processed_path,
        splits=args.splits,
        verify_images=args.verify_images,
    )

    try:
        summary = run_preprocessing_pipeline(config)
        elapsed = time.time() - start_time

        # Check status
        if summary.get('failed_samples_count', 0) > 0:
            status = f"completed with {summary['failed_samples_count']} failures"
        else:
            status = "completed successfully"

        print_summary("Preprocess", elapsed, status)
        print(f"  Processed: {summary.get('total_samples', 0)} samples")
        print(f"  Output: {config.processed_data_path}")

        return 0

    except Exception as e:
        elapsed = time.time() - start_time
        print_summary("Preprocess", elapsed, f"failed: {e}")
        return 1


def cmd_train(args) -> int:
    """Run the training pipeline."""
    print_banner("Training Pipeline")

    start_time = time.time()

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

    try:
        summary = run_training_pipeline(config)
        elapsed = time.time() - start_time

        if summary.get('status') == 'completed':
            print_summary("Train", elapsed, "completed")
            print(f"  Best eval loss: {summary.get('best_eval_loss', 'N/A'):.4f}")
            print(f"  Checkpoint: {summary.get('best_checkpoint', 'N/A')}")
            return 0
        else:
            print_summary("Train", elapsed, f"failed: {summary.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        elapsed = time.time() - start_time
        print_summary("Train", elapsed, f"failed: {e}")
        return 1


def cmd_evaluate(args) -> int:
    """Run the evaluation pipeline."""
    print_banner("Evaluation Pipeline")

    start_time = time.time()

    config = EvaluationConfig(
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        processed_data_path=args.processed_data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        run_fraud_detection=not args.no_fraud_detection,
    )

    try:
        summary = run_evaluation_pipeline(config)
        elapsed = time.time() - start_time

        ner_metrics = summary.get('ner_metrics', {})
        macro_f1 = ner_metrics.get('macro', {}).get('f1', 0.0)

        fraud_report = summary.get('fraud_report', {})
        duplicates = fraud_report.get('duplicates_found', 0)

        print_summary("Evaluate", elapsed, "completed")
        print(f"  Macro F1: {macro_f1:.4f}")
        if config.run_fraud_detection:
            print(f"  Duplicates detected: {duplicates}")
        print(f"  Output: {config.output_dir}")

        return 0

    except Exception as e:
        elapsed = time.time() - start_time
        print_summary("Evaluate", elapsed, f"failed: {e}")
        return 1


def cmd_full(args) -> int:
    """Run full pipeline: preprocess → train → evaluate."""
    print_banner("Full Pipeline (Preprocess → Train → Evaluate)")

    overall_start = time.time()
    exit_code = 0

    # Step 1: Preprocess
    print("\n" + "-" * 60)
    print("STEP 1/3: Preprocessing")
    print("-" * 60)

    preprocess_args = argparse.Namespace(
        raw_path=args.raw_path,
        processed_path=args.processed_path,
        splits=args.splits,
        verify_images=args.verify_images,
    )

    result = cmd_preprocess(preprocess_args)
    if result != 0:
        print("\n[Full Pipeline] Stopping: Preprocessing failed")
        return 1

    # Step 2: Train
    print("\n" + "-" * 60)
    print("STEP 2/3: Training")
    print("-" * 60)

    train_args = argparse.Namespace(
        model_path=args.model_path,
        num_labels=args.num_labels,
        dropout=args.dropout,
        output_dir=args.checkpoint_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        data_path=args.data_path,
    )

    result = cmd_train(train_args)
    if result != 0:
        print("\n[Full Pipeline] Stopping: Training failed")
        return 1

    # Step 3: Evaluate
    print("\n" + "-" * 60)
    print("STEP 3/3: Evaluation")
    print("-" * 60)

    # Infer checkpoint path from training output
    checkpoint_path = Path(args.checkpoint_dir) / "best_model.pt"

    evaluate_args = argparse.Namespace(
        checkpoint_path=str(checkpoint_path),
        model_path=args.model_path,
        processed_data_path=args.processed_path,
        output_dir=args.eval_output_dir,
        batch_size=args.batch_size,
        no_fraud_detection=not args.run_fraud_detection,
    )

    result = cmd_evaluate(evaluate_args)
    if result != 0:
        print("\n[Full Pipeline] Stopping: Evaluation failed")
        return 1

    # Final summary
    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total elapsed time: {overall_elapsed:.2f}s")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluation: {args.eval_output_dir}")
    print("=" * 60)

    return 0


def main() -> int:
    """Main entry point for ReceiptGuard-ML CLI."""
    parser = argparse.ArgumentParser(
        prog="receiptguard",
        description="ReceiptGuard-ML — Duplicate receipt detection using LayoutLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  receiptguard full

  # Just train with custom settings
  receiptguard train --num_epochs 10 --batch_size 4 --learning_rate 3e-5

  # Evaluate a checkpoint
  receiptguard evaluate --checkpoint_path dataset/processed/checkpoints/best_model.pt

  # Preprocess only
  receiptguard preprocess --raw_path /path/to/data
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"ReceiptGuard-ML v{__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -------------------------------------------------------------------------
    # preprocess subcommand
    # -------------------------------------------------------------------------
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Run data preprocessing pipeline",
    )
    preprocess_parser.add_argument(
        "--raw_path",
        type=str,
        default="dataset/raw/SROIE2019",
        help="Path to raw dataset",
    )
    preprocess_parser.add_argument(
        "--processed_path",
        type=str,
        default="dataset/processed",
        help="Path for processed output",
    )
    preprocess_parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to process",
    )
    preprocess_parser.add_argument(
        "--verify_images",
        action="store_true",
        default=True,
        help="Verify image files during preprocessing",
    )
    preprocess_parser.set_defaults(func=cmd_preprocess)

    # -------------------------------------------------------------------------
    # train subcommand
    # -------------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Run model training pipeline",
    )
    train_parser.add_argument(
        "--model_path",
        type=str,
        default="dataset/raw/SROIE2019/layoutlm-base-uncased",
        help="Path to pretrained LayoutLM model",
    )
    train_parser.add_argument(
        "--num_labels",
        type=int,
        default=9,
        help="Number of NER labels",
    )
    train_parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    train_parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/processed/checkpoints",
        help="Output directory for checkpoints",
    )
    train_parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    train_parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    train_parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    train_parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio of total steps",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    train_parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/raw/SROIE2019",
        help="Path to dataset",
    )
    train_parser.set_defaults(func=cmd_train)

    # -------------------------------------------------------------------------
    # evaluate subcommand
    # -------------------------------------------------------------------------
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation pipeline",
    )
    evaluate_parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (required)",
    )
    evaluate_parser.add_argument(
        "--model_path",
        type=str,
        default="dataset/raw/SROIE2019/layoutlm-base-uncased",
        help="Path to LayoutLM model",
    )
    evaluate_parser.add_argument(
        "--processed_data_path",
        type=str,
        default="dataset/processed",
        help="Path to processed data",
    )
    evaluate_parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/processed/evaluation",
        help="Output directory for evaluation results",
    )
    evaluate_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    evaluate_parser.add_argument(
        "--no_fraud_detection",
        action="store_true",
        help="Skip fraud detection",
    )
    evaluate_parser.set_defaults(func=cmd_evaluate)

    # -------------------------------------------------------------------------
    # full subcommand
    # -------------------------------------------------------------------------
    full_parser = subparsers.add_parser(
        "full",
        help="Run full pipeline: preprocess → train → evaluate",
    )
    # Preprocess args
    full_parser.add_argument(
        "--raw_path",
        type=str,
        default="dataset/raw/SROIE2019",
        help="Path to raw dataset",
    )
    full_parser.add_argument(
        "--processed_path",
        type=str,
        default="dataset/processed",
        help="Path for processed output",
    )
    full_parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to process",
    )
    # Train args
    full_parser.add_argument(
        "--model_path",
        type=str,
        default="dataset/raw/SROIE2019/layoutlm-base-uncased",
        help="Path to pretrained LayoutLM model",
    )
    full_parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    full_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    full_parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    full_parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    full_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    full_parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/raw/SROIE2019",
        help="Path to dataset",
    )
    full_parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="dataset/processed/checkpoints",
        help="Directory for saving checkpoints",
    )
    # Evaluate args
    full_parser.add_argument(
        "--eval_output_dir",
        type=str,
        default="dataset/processed/evaluation",
        help="Directory for evaluation results",
    )
    full_parser.add_argument(
        "--run_fraud_detection",
        action="store_true",
        default=True,
        help="Run fraud detection during evaluation",
    )
    full_parser.set_defaults(func=cmd_full)

    # Parse arguments
    args = parser.parse_args()

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return 0

    # Execute the selected command with top-level error handling
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
