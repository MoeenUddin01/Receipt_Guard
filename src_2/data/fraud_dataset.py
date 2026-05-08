"""
fraud_dataset.py - Generates synthetic fraud pairs from SROIE for Siamese model training.

This module creates pairs of receipts for fraud detection:
- Fraud pairs: Modified versions of the same receipt (duplicate, tampered)
- Legitimate pairs: Two completely different receipts
"""

import json
import random
import re
import string
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class FraudPairType(IntEnum):
    """Types of receipt pairs for fraud detection training."""
    DIFFERENT_RECEIPT = 0   # Two completely different receipts (legitimate)
    EXACT_DUPLICATE = 1     # Same receipt twice (fraud - exact duplicate)
    DATE_TAMPERED = 2       # Same receipt, date ±1-3 days (fraud)
    TOTAL_TAMPERED = 3      # Same receipt, total ±0.10-2.00 (fraud)
    COMPANY_TYPO = 4        # Same receipt, one char changed in company (fraud)


def tamper_date(date_str: str) -> str:
    """
    Adds or subtracts 1-3 days randomly from date string DD/MM/YYYY.
    
    Args:
        date_str: Date string in format DD/MM/YYYY
        
    Returns:
        New date string in same format DD/MM/YYYY
    """
    # Parse the date
    date_obj = datetime.strptime(date_str, "%d/%m/%Y")
    
    # Randomly add or subtract 1-3 days
    days_offset = random.choice([-1, 1]) * random.randint(1, 3)
    new_date = date_obj + timedelta(days=days_offset)
    
    # Return in same format
    return new_date.strftime("%d/%m/%Y")


def tamper_total(total_str: str) -> str:
    """
    Adds or subtracts a random float 0.10-2.00 from total.
    
    Args:
        total_str: Total amount as string (e.g., "125.50" or "$125.50")
        
    Returns:
        Modified total string with 2 decimal places, never below 0
    """
    # Extract numeric value (handle currency symbols)
    # Remove common currency symbols and whitespace
    cleaned = re.sub(r'[$€£¥\s]', '', total_str)
    
    try:
        total_val = float(cleaned)
    except ValueError:
        # If can't parse, return original
        return total_str
    
    # Random offset between 0.10 and 2.00
    offset = random.uniform(0.10, 2.00)
    
    # Randomly add or subtract
    if random.random() < 0.5:
        new_total = total_val - offset
    else:
        new_total = total_val + offset
    
    # Ensure never goes below 0
    new_total = max(0.0, new_total)
    
    # Format with 2 decimal places
    return f"{new_total:.2f}"


def tamper_company(company_str: str) -> str:
    """
    Randomly changes one character in the company name.
    Operations: swap two adjacent chars, delete one char, or insert one letter.
    
    Args:
        company_str: Original company name
        
    Returns:
        Modified company name with one character change
    """
    if len(company_str) < 2:
        return company_str
    
    company_chars = list(company_str)
    operation = random.choice(["swap", "delete", "insert"])
    
    if operation == "swap" and len(company_chars) >= 2:
        # Swap two adjacent characters
        idx = random.randint(0, len(company_chars) - 2)
        company_chars[idx], company_chars[idx + 1] = company_chars[idx + 1], company_chars[idx]
        
    elif operation == "delete" and len(company_chars) >= 2:
        # Delete one random character
        idx = random.randint(0, len(company_chars) - 1)
        company_chars.pop(idx)
        
    elif operation == "insert":
        # Insert one random letter at random position
        idx = random.randint(0, len(company_chars))
        random_letter = random.choice(string.ascii_letters)
        company_chars.insert(idx, random_letter)
    
    return "".join(company_chars)


class SROIEPairDataset(Dataset):
    """
    Dataset that generates pairs of receipts for fraud detection.
    
    Fraud pairs (50% of data, split evenly across types 1-4):
        - EXACT_DUPLICATE: Same receipt twice
        - DATE_TAMPERED: Same receipt with modified date
        - TOTAL_TAMPERED: Same receipt with modified total
        - COMPANY_TYPO: Same receipt with company name typo
    
    Legitimate pairs (50% of data):
        - DIFFERENT_RECEIPT: Two completely different receipts
    """
    
    def __init__(
        self,
        samples: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        fraud_ratio: float = 0.5
    ):
        """
        Initialize the pair dataset.
        
        Args:
            samples: List of processed receipts from train_samples.json
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            fraud_ratio: Ratio of fraud pairs (default 0.5)
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fraud_ratio = fraud_ratio
        
        # Generate all pairs at initialization
        self.pairs = self._generate_pairs()
        
        # Shuffle pairs
        random.shuffle(self.pairs)
        
        print(f"Generated {len(self.pairs)} pairs:")
        fraud_count = sum(1 for p in self.pairs if p['label'] == 1)
        legit_count = sum(1 for p in self.pairs if p['label'] == 0)
        print(f"  Fraud pairs: {fraud_count} ({fraud_count/len(self.pairs)*100:.1f}%)")
        print(f"  Legitimate pairs: {legit_count} ({legit_count/len(self.pairs)*100:.1f}%)")
    
    def _generate_pairs(self) -> List[Dict]:
        """Generate all receipt pairs for the dataset."""
        pairs = []
        n_samples = len(self.samples)
        
        # Total pairs = len(samples) × 2
        total_pairs = n_samples * 2
        
        # Number of fraud pairs
        n_fraud = int(total_pairs * self.fraud_ratio)
        n_legit = total_pairs - n_fraud
        
        # Split fraud pairs evenly across types 1-4
        fraud_per_type = n_fraud // 4
        
        # Generate fraud pairs
        for fraud_type in [FraudPairType.EXACT_DUPLICATE, 
                          FraudPairType.DATE_TAMPERED,
                          FraudPairType.TOTAL_TAMPERED, 
                          FraudPairType.COMPANY_TYPO]:
            for _ in range(fraud_per_type):
                # Pick random receipt
                receipt = random.choice(self.samples)
                
                # Create tampered version
                receipt_a = receipt.copy()
                receipt_b = self._create_tampered_copy(receipt, fraud_type)
                
                pairs.append({
                    'receipt_a': receipt_a,
                    'receipt_b': receipt_b,
                    'label': 1,  # Fraud
                    'pair_type': int(fraud_type)
                })
        
        # Generate legitimate pairs (different receipts)
        for _ in range(n_legit):
            # Pick two different receipts
            idx_a = random.randint(0, n_samples - 1)
            idx_b = random.randint(0, n_samples - 1)
            
            # Ensure they're different
            while idx_a == idx_b and n_samples > 1:
                idx_b = random.randint(0, n_samples - 1)
            
            pairs.append({
                'receipt_a': self.samples[idx_a],
                'receipt_b': self.samples[idx_b],
                'label': 0,  # Legitimate
                'pair_type': int(FraudPairType.DIFFERENT_RECEIPT)
            })
        
        return pairs
    
    def _create_tampered_copy(self, receipt: Dict, fraud_type: FraudPairType) -> Dict:
        """Create a tampered copy of a receipt based on fraud type."""
        copy = receipt.copy()
        
        if fraud_type == FraudPairType.EXACT_DUPLICATE:
            # Exact duplicate - just return the same
            return copy
            
        elif fraud_type == FraudPairType.DATE_TAMPERED:
            # Modify the date if present
            entities = copy.get('entities', [])
            for entity in entities:
                if entity.get('label') == 'DATE':
                    entity['text'] = tamper_date(entity['text'])
                    break
            return copy
            
        elif fraud_type == FraudPairType.TOTAL_TAMPERED:
            # Modify the total if present
            entities = copy.get('entities', [])
            for entity in entities:
                if entity.get('label') == 'TOTAL':
                    entity['text'] = tamper_total(entity['text'])
                    break
            return copy
            
        elif fraud_type == FraudPairType.COMPANY_TYPO:
            # Modify the company name if present
            entities = copy.get('entities', [])
            for entity in entities:
                if entity.get('label') == 'COMPANY':
                    entity['text'] = tamper_company(entity['text'])
                    break
            return copy
        
        return copy
    
    def _encode_receipt(self, receipt: Dict) -> Dict[str, torch.Tensor]:
        """Encode a receipt into tensors for the model."""
        tokens = receipt.get('tokens', [])
        bboxes = receipt.get('bboxes', [])
        
        # Get text from tokens
        texts = [t.get('text', '') for t in tokens]
        
        # Tokenize
        encoding = self.tokenizer(
            texts,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Pad bboxes to max_length
        padded_bboxes = bboxes + [[0, 0, 0, 0]] * (self.max_length - len(bboxes))
        padded_bboxes = padded_bboxes[:self.max_length]
        
        encoding['bbox'] = torch.tensor([padded_bboxes], dtype=torch.long)
        
        # Squeeze batch dimension
        return {k: v.squeeze(0) for k, v in encoding.items()}
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a pair of receipts.
        
        Returns:
            dict with keys:
                - receipt_a: dict of tensors for receipt A
                - receipt_b: dict of tensors for receipt B
                - label: LongTensor (1=fraud, 0=legitimate)
                - pair_type: int (FraudPairType value)
        """
        pair = self.pairs[idx]
        
        return {
            'receipt_a': self._encode_receipt(pair['receipt_a']),
            'receipt_b': self._encode_receipt(pair['receipt_b']),
            'label': torch.tensor(pair['label'], dtype=torch.long),
            'pair_type': pair['pair_type']
        }


def load_pair_dataset(
    processed_path: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_length: int = 512
) -> SROIEPairDataset:
    """
    Load processed samples and create pair dataset.
    
    Args:
        processed_path: Path to dataset/processed directory
        tokenizer: Tokenizer for encoding
        split: "train" or "test"
        max_length: Maximum sequence length
        
    Returns:
        SROIEPairDataset instance
    """
    processed_path = Path(processed_path)
    samples_file = processed_path / f"{split}_samples.json"
    
    if not samples_file.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_file}")
    
    print(f"Loading {split} samples from {samples_file}...")
    
    with open(samples_file, 'r') as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} receipts")
    
    # Create dataset
    dataset = SROIEPairDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Print distribution
    fraud_count = sum(1 for p in dataset.pairs if p['label'] == 1)
    legit_count = len(dataset.pairs) - fraud_count
    
    print(f"\nDataset statistics:")
    print(f"  Total pairs: {len(dataset)}")
    print(f"  Fraud pairs: {fraud_count}")
    print(f"  Legitimate pairs: {legit_count}")
    
    # Count by fraud type
    type_counts = {}
    for p in dataset.pairs:
        ptype = p['pair_type']
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    
    print(f"\nFraud type distribution:")
    for ptype, count in sorted(type_counts.items()):
        type_name = FraudPairType(ptype).name
        print(f"  {type_name}: {count}")
    
    return dataset


if __name__ == "__main__":
    """Test the fraud dataset generation."""
    
    print("=" * 60)
    print("Testing Fraud Dataset Generation")
    print("=" * 60)
    
    # Test tamper functions
    print("\n1. Testing tamper functions:")
    
    # Test date tampering
    test_date = "15/03/2024"
    for _ in range(3):
        print(f"   Date: {test_date} -> {tamper_date(test_date)}")
    
    # Test total tampering
    test_total = "125.50"
    for _ in range(3):
        print(f"   Total: ${test_total} -> ${tamper_total(test_total)}")
    
    # Test company tampering
    test_company = "ACME Corporation"
    for _ in range(3):
        print(f"   Company: '{test_company}' -> '{tamper_company(test_company)}'")
    
    # Try to load and test dataset
    print("\n2. Loading dataset and testing first 5 pairs:")
    
    try:
        from transformers import LayoutLMTokenizer
        
        # Use default tokenizer
        tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        
        # Load dataset
        processed_dir = Path(__file__).parent.parent.parent / "dataset" / "processed"
        
        if (processed_dir / "train_samples.json").exists():
            dataset = load_pair_dataset(processed_dir, tokenizer, split="train")
            
            # Print first 5 pairs
            print("\n" + "=" * 60)
            print("First 5 pairs:")
            print("=" * 60)
            
            for i in range(min(5, len(dataset))):
                pair = dataset[i]
                label = pair['label'].item()
                pair_type = pair['pair_type']
                type_name = FraudPairType(pair_type).name
                
                # Get token lengths
                len_a = (pair['receipt_a']['input_ids'] != 0).sum().item()
                len_b = (pair['receipt_b']['input_ids'] != 0).sum().item()
                
                print(f"\nPair {i+1}:")
                print(f"  Label: {label} ({'FRAUD' if label == 1 else 'LEGITIMATE'})")
                print(f"  Type: {pair_type} ({type_name})")
                print(f"  Receipt A tokens: {len_a}")
                print(f"  Receipt B tokens: {len_b}")
        else:
            print(f"\n   ⚠️  train_samples.json not found at {processed_dir}")
            print("   Run Model 1 preprocessing first to generate this file.")
            
    except ImportError as e:
        print(f"\n   ⚠️  Could not import required packages: {e}")
        print("   Install with: pip install transformers torch")
        
    except Exception as e:
        print(f"\n   ⚠️  Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)
