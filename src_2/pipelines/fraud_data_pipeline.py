"""
fraud_data_pipeline.py - Generates and validates all training pairs for Model 2.

This pipeline creates fraud detection pairs from processed SROIE data,
validates pair generation, and saves metadata for inspection.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from transformers import LayoutLMTokenizer

from ..data.fraud_dataset import FraudPairType, SROIEPairDataset, load_pair_dataset


@dataclass
class FraudDataConfig:
    """Configuration for fraud data pipeline."""
    processed_path: str = "dataset/processed"
    output_path: str = "artifacts/siamese"
    fraud_ratio: float = 0.5
    max_length: int = 512


def run_fraud_data_pipeline(config: FraudDataConfig) -> Dict:
    """
    Run the complete fraud data pipeline.
    
    Args:
        config: FraudDataConfig instance
        
    Returns:
        Summary dictionary with pipeline results
    """
    print("=" * 60)
    print("Fraud Data Pipeline - Model 2")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    
    summary = {
        'config': {
            'processed_path': config.processed_path,
            'output_path': config.output_path,
            'fraud_ratio': config.fraud_ratio,
            'max_length': config.max_length
        },
        'train': {},
        'test': {},
        'validation_issues': []
    }
    
    # Process train and test splits
    for split in ['train', 'test']:
        print(f"\n{'─' * 40}")
        print(f"Processing {split.upper()} split")
        print(f"{'─' * 40}")
        
        # Step 1: Load processed samples
        print(f"\nStep 1: Loading processed samples...")
        samples = load_processed_samples(config.processed_path, split)
        summary[split]['samples_loaded'] = len(samples)
        print(f"   Loaded {len(samples)} receipts from {split}_samples.json")
        
        # Step 2: Generate pairs
        print(f"\nStep 2: Generating pairs...")
        dataset = SROIEPairDataset(
            samples=samples,
            tokenizer=tokenizer,
            max_length=config.max_length,
            fraud_ratio=config.fraud_ratio
        )
        
        summary[split]['total_pairs'] = len(dataset)
        summary[split]['fraud_pairs'] = sum(1 for p in dataset.pairs if p['label'] == 1)
        summary[split]['legit_pairs'] = sum(1 for p in dataset.pairs if p['label'] == 0)
        
        print(f"   Total pairs: {len(dataset)}")
        print(f"   Fraud pairs: {summary[split]['fraud_pairs']}")
        print(f"   Legitimate pairs: {summary[split]['legit_pairs']}")
        
        # Step 3: Validate pairs
        print(f"\nStep 3: Validating pairs...")
        validation_issues = validate_pairs(dataset.pairs, samples)
        summary[split]['validation_issues'] = validation_issues
        summary['validation_issues'].extend(validation_issues)
        
        if validation_issues:
            print(f"   ⚠️  Found {len(validation_issues)} validation issues:")
            for issue in validation_issues[:5]:  # Show first 5
                print(f"     - {issue}")
            if len(validation_issues) > 5:
                print(f"     ... and {len(validation_issues) - 5} more")
        else:
            print(f"   ✓ All pairs validated successfully")
        
        # Step 4: Save pair metadata
        print(f"\nStep 4: Saving pair metadata...")
        metadata_path = output_dir / f"{split}_pairs_meta.json"
        save_pair_metadata(dataset.pairs, samples, metadata_path)
        print(f"   Saved metadata to {metadata_path}")
        
        # Step 5: Collect statistics for final report
        pair_type_counts = {}
        for pair in dataset.pairs:
            ptype = pair['pair_type']
            pair_type_counts[ptype] = pair_type_counts.get(ptype, 0) + 1
        
        summary[split]['pair_type_distribution'] = pair_type_counts
    
    # Step 5: Print final data report
    print(f"\n{'─' * 40}")
    print("FINAL DATA REPORT")
    print(f"{'─' * 40}")
    
    print_data_report(summary)
    
    # Save summary
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Pipeline completed successfully!")
    print(f"✓ Summary saved to {summary_path}")
    
    return summary


def load_processed_samples(processed_path: str, split: str) -> List[Dict]:
    """
    Load processed samples from Model 1 pipeline.
    
    Args:
        processed_path: Path to processed directory
        split: 'train' or 'test'
        
    Returns:
        List of processed receipt samples
    """
    processed_path = Path(processed_path)
    samples_file = processed_path / f"{split}_samples.json"
    
    if not samples_file.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_file}")
    
    with open(samples_file, 'r') as f:
        samples = json.load(f)
    
    return samples


def validate_pairs(pairs: List[Dict], samples: List[Dict]) -> List[str]:
    """
    Validate generated pairs for correctness.
    
    Args:
        pairs: List of generated pairs
        samples: Original samples list
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    # Create sample ID mapping
    sample_ids = [sample.get('id', i) for i, sample in enumerate(samples)]
    
    for i, pair in enumerate(pairs):
        pair_type = FraudPairType(pair['pair_type'])
        
        # Check legitimate pairs: no receipt should be paired with itself
        if pair_type == FraudPairType.DIFFERENT_RECEIPT:
            receipt_a_id = pair['receipt_a'].get('id', 'unknown')
            receipt_b_id = pair['receipt_b'].get('id', 'unknown')
            
            if receipt_a_id == receipt_b_id:
                issues.append(f"Pair {i}: Legitimate pair has same receipt ID ({receipt_a_id})")
        
        # Check fraud pairs: verify modifications are applied correctly
        else:
            # For fraud pairs, receipts should be related (same base receipt)
            receipt_a_entities = pair['receipt_a'].get('entities', [])
            receipt_b_entities = pair['receipt_b'].get('entities', [])
            
            if pair_type == FraudPairType.DATE_TAMPERED:
                # Check that date was actually changed
                date_a = next((e['text'] for e in receipt_a_entities if e['label'] == 'DATE'), None)
                date_b = next((e['text'] for e in receipt_b_entities if e['label'] == 'DATE'), None)
                
                if date_a and date_b and date_a == date_b:
                    issues.append(f"Pair {i}: DATE_TAMPERED pair has unchanged date")
            
            elif pair_type == FraudPairType.TOTAL_TAMPERED:
                # Check that total was actually changed
                total_a = next((e['text'] for e in receipt_a_entities if e['label'] == 'TOTAL'), None)
                total_b = next((e['text'] for e in receipt_b_entities if e['label'] == 'TOTAL'), None)
                
                if total_a and total_b and total_a == total_b:
                    issues.append(f"Pair {i}: TOTAL_TAMPERED pair has unchanged total")
            
            elif pair_type == FraudPairType.COMPANY_TYPO:
                # Check that company was actually changed
                company_a = next((e['text'] for e in receipt_a_entities if e['label'] == 'COMPANY'), None)
                company_b = next((e['text'] for e in receipt_b_entities if e['label'] == 'COMPANY'), None)
                
                if company_a and company_b and company_a == company_b:
                    issues.append(f"Pair {i}: COMPANY_TYPO pair has unchanged company name")
    
    return issues


def save_pair_metadata(pairs: List[Dict], samples: List[Dict], output_path: Path):
    """
    Save pair metadata for inspection (not tensors).
    
    Args:
        pairs: List of generated pairs
        samples: Original samples list
        output_path: Path to save metadata
    """
    metadata = []
    
    for i, pair in enumerate(pairs):
        # Extract receipt IDs
        receipt_a_id = pair['receipt_a'].get('id', f"receipt_{i}_a")
        receipt_b_id = pair['receipt_b'].get('id', f"receipt_{i}_b")
        
        metadata.append({
            'pair_index': i,
            'receipt_a_id': receipt_a_id,
            'receipt_b_id': receipt_b_id,
            'label': pair['label'],
            'pair_type': pair['pair_type'],
            'pair_type_name': FraudPairType(pair['pair_type']).name
        })
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def print_data_report(summary: Dict):
    """
    Print comprehensive data report.
    
    Args:
        summary: Pipeline summary dictionary
    """
    print(f"\n📊 DATASET SUMMARY")
    print(f"{'─' * 50}")
    
    for split in ['train', 'test']:
        if split not in summary:
            continue
            
        split_data = summary[split]
        print(f"\n{split.upper()} Split:")
        print(f"  Samples loaded: {split_data['samples_loaded']:,}")
        print(f"  Total pairs: {split_data['total_pairs']:,}")
        print(f"  Fraud pairs: {split_data['fraud_pairs']:,} ({split_data['fraud_pairs']/split_data['total_pairs']*100:.1f}%)")
        print(f"  Legitimate pairs: {split_data['legit_pairs']:,} ({split_data['legit_pairs']/split_data['total_pairs']*100:.1f}%)")
        
        # Pair type distribution
        print(f"\n  Pair Type Distribution:")
        pair_type_counts = split_data['pair_type_distribution']
        for ptype, count in sorted(pair_type_counts.items()):
            type_name = FraudPairType(ptype).name
            percentage = count / split_data['total_pairs'] * 100
            print(f"    {type_name}: {count:,} ({percentage:.1f}%)")
        
        # Validation issues
        issues = split_data.get('validation_issues', [])
        if issues:
            print(f"\n  ⚠️  Validation Issues: {len(issues)}")
        else:
            print(f"\n  ✓ All pairs validated")
    
    # Overall validation issues
    total_issues = len(summary.get('validation_issues', []))
    print(f"\n{'─' * 50}")
    print(f"Total validation issues across all splits: {total_issues}")
    
    if total_issues == 0:
        print("🎉 All data generated successfully!")
    else:
        print(f"⚠️  Please review {total_issues} validation issues above")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate fraud detection pairs for Model 2")
    
    parser.add_argument(
        "--processed-path",
        default="dataset/processed",
        help="Path to processed data from Model 1"
    )
    
    parser.add_argument(
        "--output-path",
        default="artifacts/siamese",
        help="Output directory for generated pairs"
    )
    
    parser.add_argument(
        "--fraud-ratio",
        type=float,
        default=0.5,
        help="Ratio of fraud pairs in dataset (default: 0.5)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = FraudDataConfig(
        processed_path=args.processed_path,
        output_path=args.output_path,
        fraud_ratio=args.fraud_ratio,
        max_length=args.max_length
    )
    
    try:
        # Run pipeline
        summary = run_fraud_data_pipeline(config)
        
        print(f"\n{'=' * 60}")
        print("🚀 Fraud data pipeline completed successfully!")
        print(f"📁 Results saved to: {config.output_path}")
        print(f"{'=' * 60}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
