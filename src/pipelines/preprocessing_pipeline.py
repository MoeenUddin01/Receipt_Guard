import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from preprocessing import build_processed_sample

# Import configuration
from src.config import CFG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""
    raw_data_path: str = None
    processed_data_path: str = None
    splits: list = None
    verify_images: bool = True

    def __post_init__(self):
        if self.raw_data_path is None:
            self.raw_data_path = str(CFG.resolve_path(CFG.paths.raw_data_path))
        else:
            self.raw_data_path = str(CFG.resolve_path(self.raw_data_path))
            
        if self.processed_data_path is None:
            self.processed_data_path = str(CFG.resolve_path(CFG.paths.processed_data_path))
        else:
            self.processed_data_path = str(CFG.resolve_path(self.processed_data_path))
            
        if self.splits is None:
            self.splits = [CFG.data.train_split, CFG.data.test_split]


def verify_dataset_integrity(config: PreprocessingConfig) -> Dict[str, Dict]:
    """
    Step 1 — Verify dataset integrity for each split.

    For each split, confirm that img/, box/, entities/ directories exist.
    Find and report any receipts with missing files.
    Log counts of total images, box files, and entity files per split.

    Args:
        config: PreprocessingConfig with paths and settings

    Returns:
        Dictionary with verification results per split
    """
    integrity_report = {}

    for split in config.splits:
        logger.info(f"Verifying dataset integrity for split: {split}")

        split_path = Path(config.raw_data_path) / split
        img_dir = split_path / "img"
        box_dir = split_path / "box"
        entities_dir = split_path / "entities"

        # Check if directories exist
        dirs_exist = {
            'img': img_dir.exists(),
            'box': box_dir.exists(),
            'entities': entities_dir.exists()
        }

        missing_dirs = [name for name, exists in dirs_exist.items() if not exists]
        if missing_dirs:
            logger.error(f"Missing directories for {split}: {missing_dirs}")
            integrity_report[split] = {
                'valid': False,
                'missing_directories': missing_dirs,
                'total_images': 0,
                'total_boxes': 0,
                'total_entities': 0,
                'missing_files': []
            }
            continue

        # Get all files
        img_files = set(f.stem for f in img_dir.glob("*.jpg"))
        box_files = set(f.stem for f in box_dir.glob("*.txt"))
        entity_files = set(f.stem for f in entities_dir.glob("*.txt"))

        # Find missing files for each receipt
        all_ids = img_files | box_files | entity_files
        missing_files = []

        for receipt_id in all_ids:
            missing = []
            if receipt_id not in img_files:
                missing.append("img")
            if receipt_id not in box_files:
                missing.append("box")
            if receipt_id not in entity_files:
                missing.append("entities")
            if missing:
                missing_files.append({
                    'receipt_id': receipt_id,
                    'missing': missing
                })

        # Log counts
        logger.info(f"  Total images: {len(img_files)}")
        logger.info(f"  Total box files: {len(box_files)}")
        logger.info(f"  Total entity files: {len(entity_files)}")

        if missing_files:
            logger.warning(f"  Found {len(missing_files)} receipts with missing files")
            for mf in missing_files[:5]:  # Show first 5
                logger.warning(f"    - {mf['receipt_id']}: missing {mf['missing']}")

        # Find complete receipts (have all three files)
        complete_ids = img_files & box_files & entity_files

        integrity_report[split] = {
            'valid': True,
            'total_images': len(img_files),
            'total_boxes': len(box_files),
            'total_entities': len(entity_files),
            'complete_receipts': len(complete_ids),
            'missing_files': missing_files,
            'receipt_ids': sorted(list(complete_ids))
        }

    return integrity_report


def process_all_samples(config: PreprocessingConfig, integrity_report: Dict) -> Dict:
    """
    Step 2 — Process all samples.

    Call build_processed_sample for every receipt ID in both splits.
    Collect valid samples, log failures.
    Track label distribution across all samples.

    Args:
        config: PreprocessingConfig with paths and settings
        integrity_report: Output from verify_dataset_integrity

    Returns:
        Dictionary with processed samples and statistics
    """
    all_samples = []
    failed_samples = []
    all_labels = []
    token_lengths = []

    split_samples = {}

    for split in config.splits:
        if not integrity_report.get(split, {}).get('valid', False):
            logger.warning(f"Skipping {split} due to invalid integrity check")
            continue

        receipt_ids = integrity_report[split]['receipt_ids']
        logger.info(f"Processing {len(receipt_ids)} samples for split: {split}")

        split_data = []
        split_failed = []

        for receipt_id in receipt_ids:
            try:
                sample = build_processed_sample(
                    receipt_id,
                    split,
                    config.raw_data_path
                )

                if sample:
                    split_data.append(sample)
                    all_samples.append(sample)

                    # Collect labels and token lengths
                    all_labels.extend(sample['labels'])
                    token_lengths.append(len(sample['tokens']))
                else:
                    split_failed.append(receipt_id)
                    failed_samples.append({
                        'receipt_id': receipt_id,
                        'split': split,
                        'reason': 'build_processed_sample returned None'
                    })

            except Exception as e:
                logger.error(f"Failed to process {receipt_id} in {split}: {e}")
                split_failed.append(receipt_id)
                failed_samples.append({
                    'receipt_id': receipt_id,
                    'split': split,
                    'reason': str(e)
                })

        split_samples[split] = split_data

        logger.info(f"  Successfully processed: {len(split_data)}")
        logger.info(f"  Failed: {len(split_failed)}")

    # Calculate label distribution
    label_counts = Counter(all_labels)

    return {
        'samples': all_samples,
        'split_samples': split_samples,
        'failed_samples': failed_samples,
        'label_distribution': dict(label_counts),
        'token_lengths': token_lengths
    }


def save_processed_data(config: PreprocessingConfig, processing_results: Dict) -> None:
    """
    Step 3 — Save processed data to JSON files.

    Save all processed samples to dataset/processed/train_samples.json and test_samples.json.
    Save label statistics to dataset/processed/label_stats.json.

    Args:
        config: PreprocessingConfig with paths and settings
        processing_results: Output from process_all_samples
    """
    processed_dir = Path(config.processed_data_path)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save split samples
    for split in config.splits:
        split_data = processing_results['split_samples'].get(split, [])
        output_file = processed_dir / f"{split}_samples.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(split_data)} samples to {output_file}")

    # Calculate and save label statistics
    label_dist = processing_results['label_distribution']
    total_labels = sum(label_dist.values())

    label_stats = {
        'total_samples': len(processing_results['samples']),
        'total_labels': total_labels,
        'label_counts': label_dist,
        'label_percentages': {
            label: round((count / total_labels) * 100, 2) if total_labels > 0 else 0
            for label, count in label_dist.items()
        }
    }

    stats_file = processed_dir / "label_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(label_stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved label statistics to {stats_file}")


def generate_data_report(config: PreprocessingConfig, integrity_report: Dict,
                         processing_results: Dict) -> str:
    """
    Step 4 — Generate data report.

    Print and save to dataset/processed/preprocessing_report.txt:
    - Total samples processed per split
    - Failed samples list
    - Token length distribution (min, max, mean, p95)
    - Label class distribution with percentages
    - Class imbalance warning if any class < 1%

    Args:
        config: PreprocessingConfig with paths and settings
        integrity_report: Output from verify_dataset_integrity
        processing_results: Output from process_all_samples

    Returns:
        Report text as string
    """
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("ReceiptGuard-ML Preprocessing Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Dataset integrity summary
    lines.append("-" * 70)
    lines.append("Dataset Integrity Summary")
    lines.append("-" * 70)

    for split in config.splits:
        report = integrity_report.get(split, {})
        if not report.get('valid', False):
            lines.append(f"\n{split.upper()}:")
            lines.append(f"  Status: INVALID - Missing directories: {report.get('missing_directories', [])}")
            continue

        lines.append(f"\n{split.upper()}:")
        lines.append(f"  Total images: {report['total_images']}")
        lines.append(f"  Total box files: {report['total_boxes']}")
        lines.append(f"  Total entity files: {report['total_entities']}")
        lines.append(f"  Complete receipts: {report['complete_receipts']}")
        lines.append(f"  Missing files: {len(report['missing_files'])}")

    lines.append("")

    # Processing summary
    lines.append("-" * 70)
    lines.append("Processing Summary")
    lines.append("-" * 70)

    for split in config.splits:
        split_data = processing_results['split_samples'].get(split, [])
        lines.append(f"\n{split.upper()}:")
        lines.append(f"  Successfully processed: {len(split_data)} samples")

    # Failed samples
    failed = processing_results['failed_samples']
    lines.append(f"\nFailed Samples: {len(failed)}")
    if failed:
        for fs in failed[:10]:  # Show first 10
            lines.append(f"  - {fs['receipt_id']} ({fs['split']}): {fs['reason']}")
        if len(failed) > 10:
            lines.append(f"  ... and {len(failed) - 10} more")

    lines.append("")

    # Token length distribution
    token_lengths = processing_results['token_lengths']
    if token_lengths:
        lines.append("-" * 70)
        lines.append("Token Length Distribution")
        lines.append("-" * 70)

        import statistics
        sorted_lengths = sorted(token_lengths)
        p95_idx = int(len(sorted_lengths) * 0.95)
        p95 = sorted_lengths[min(p95_idx, len(sorted_lengths) - 1)]

        lines.append(f"  Min: {min(token_lengths)}")
        lines.append(f"  Max: {max(token_lengths)}")
        lines.append(f"  Mean: {statistics.mean(token_lengths):.2f}")
        lines.append(f"  Median: {statistics.median(token_lengths):.2f}")
        lines.append(f"  P95: {p95}")

    lines.append("")

    # Label class distribution
    label_dist = processing_results['label_distribution']
    total_labels = sum(label_dist.values())

    lines.append("-" * 70)
    lines.append("Label Class Distribution")
    lines.append("-" * 70)

    # Sort by count descending
    sorted_labels = sorted(label_dist.items(), key=lambda x: x[1], reverse=True)

    imbalance_warnings = []
    for label, count in sorted_labels:
        percentage = (count / total_labels) * 100 if total_labels > 0 else 0
        lines.append(f"  {label:15s}: {count:8d} ({percentage:6.2f}%)")

        if percentage < 1.0:
            imbalance_warnings.append(label)

    lines.append("")

    # Class imbalance warning
    if imbalance_warnings:
        lines.append("-" * 70)
        lines.append("⚠️  CLASS IMBALANCE WARNING")
        lines.append("-" * 70)
        lines.append(f"The following classes represent less than 1% of all labels:")
        for label in imbalance_warnings:
            count = label_dist[label]
            percentage = (count / total_labels) * 100 if total_labels > 0 else 0
            lines.append(f"  - {label}: {percentage:.2f}%")
        lines.append("")
        lines.append("Consider data augmentation or class weighting during training.")

    lines.append("")
    lines.append("=" * 70)
    lines.append("End of Report")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    # Print to console
    print("\n" + report_text)

    # Save to file
    processed_dir = Path(config.processed_data_path)
    report_file = processed_dir / "preprocessing_report.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    logger.info(f"Report saved to {report_file}")

    return report_text


def run_preprocessing_pipeline(config: PreprocessingConfig) -> Dict:
    """
    Run the complete preprocessing pipeline.

    Steps:
    1. Verify dataset integrity
    2. Process all samples
    3. Save processed data
    4. Generate data report

    Args:
        config: PreprocessingConfig with paths and settings

    Returns:
        Summary dictionary with all statistics
    """
    logger.info("=" * 70)
    logger.info("Starting ReceiptGuard-ML Preprocessing Pipeline")
    logger.info("=" * 70)
    logger.info(f"Raw data path: {config.raw_data_path}")
    logger.info(f"Processed data path: {config.processed_data_path}")
    logger.info(f"Splits: {config.splits}")
    logger.info("")

    # Step 1: Verify dataset integrity
    logger.info("Step 1: Verifying dataset integrity...")
    integrity_report = verify_dataset_integrity(config)
    logger.info("")

    # Step 2: Process all samples
    logger.info("Step 2: Processing all samples...")
    processing_results = process_all_samples(config, integrity_report)
    logger.info("")

    # Step 3: Save processed data
    logger.info("Step 3: Saving processed data...")
    save_processed_data(config, processing_results)
    logger.info("")

    # Step 4: Generate data report
    logger.info("Step 4: Generating data report...")
    report_text = generate_data_report(config, integrity_report, processing_results)
    logger.info("")

    logger.info("=" * 70)
    logger.info("Preprocessing Pipeline Complete!")
    logger.info("=" * 70)

    # Build summary dictionary
    summary = {
        'config': asdict(config),
        'integrity_report': integrity_report,
        'total_samples': len(processing_results['samples']),
        'failed_samples_count': len(processing_results['failed_samples']),
        'failed_samples': [fs['receipt_id'] for fs in processing_results['failed_samples']],
        'label_distribution': processing_results['label_distribution'],
        'token_length_stats': {
            'min': min(processing_results['token_lengths']) if processing_results['token_lengths'] else 0,
            'max': max(processing_results['token_lengths']) if processing_results['token_lengths'] else 0,
            'mean': sum(processing_results['token_lengths']) / len(processing_results['token_lengths']) if processing_results['token_lengths'] else 0
        },
        'report_path': str(Path(config.processed_data_path) / "preprocessing_report.txt")
    }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReceiptGuard-ML Preprocessing Pipeline"
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default="dataset/raw/SROIE2019",
        help="Path to raw dataset (default: dataset/raw/SROIE2019)"
    )
    parser.add_argument(
        "--processed_path",
        type=str,
        default="dataset/processed",
        help="Path for processed output (default: dataset/processed)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to process (default: train test)"
    )

    args = parser.parse_args()

    config = PreprocessingConfig(
        raw_data_path=args.raw_path,
        processed_data_path=args.processed_path,
        splits=args.splits
    )

    summary = run_preprocessing_pipeline(config)

    # Exit with error code if there were failures
    if summary['failed_samples_count'] > 0:
        logger.warning(f"Pipeline completed with {summary['failed_samples_count']} failed samples")
        exit(1)
    else:
        logger.info("Pipeline completed successfully with no failures")
        exit(0)
