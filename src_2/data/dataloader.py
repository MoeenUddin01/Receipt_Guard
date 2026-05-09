"""
dataloader.py - Batching utilities for Siamese pair data.

This module provides custom collate functions and dataloader utilities
for handling SROIEPairDataset in Siamese model training.
"""

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from torch import LongTensor, Tensor
from transformers import PreTrainedTokenizer

from .fraud_dataset import SROIEPairDataset


def collate_pairs(batch: List[dict]) -> dict:
    """
    Custom collate function for SROIEPairDataset.
    Stacks all tensors inside receipt_a and receipt_b separately.
    
    Args:
        batch: List of samples from SROIEPairDataset
        
    Returns:
        dict with structure:
        {
            receipt_a: {input_ids, attention_mask, token_type_ids, bbox},
            receipt_b: {input_ids, attention_mask, token_type_ids, bbox},
            labels: LongTensor shape [batch_size],
            pair_types: list[int]
        }
    """
    # Initialize containers for batched data
    receipt_a_batch = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'bbox': []
    }
    
    receipt_b_batch = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'bbox': []
    }
    
    labels = []
    pair_types = []
    
    # Process each sample in the batch
    for sample in batch:
        # Stack receipt_a tensors
        receipt_a_batch['input_ids'].append(sample['receipt_a']['input_ids'])
        receipt_a_batch['attention_mask'].append(sample['receipt_a']['attention_mask'])
        receipt_a_batch['token_type_ids'].append(sample['receipt_a']['token_type_ids'])
        receipt_a_batch['bbox'].append(sample['receipt_a']['bbox'])
        
        # Stack receipt_b tensors
        receipt_b_batch['input_ids'].append(sample['receipt_b']['input_ids'])
        receipt_b_batch['attention_mask'].append(sample['receipt_b']['attention_mask'])
        receipt_b_batch['token_type_ids'].append(sample['receipt_b']['token_type_ids'])
        receipt_b_batch['bbox'].append(sample['receipt_b']['bbox'])
        
        # Collect labels and pair types
        labels.append(sample['label'])
        pair_types.append(sample['pair_type'])
    
    # Stack all tensors
    receipt_a_batch['input_ids'] = torch.stack(receipt_a_batch['input_ids'])
    receipt_a_batch['attention_mask'] = torch.stack(receipt_a_batch['attention_mask'])
    receipt_a_batch['token_type_ids'] = torch.stack(receipt_a_batch['token_type_ids'])
    receipt_a_batch['bbox'] = torch.stack(receipt_a_batch['bbox'])
    
    receipt_b_batch['input_ids'] = torch.stack(receipt_b_batch['input_ids'])
    receipt_b_batch['attention_mask'] = torch.stack(receipt_b_batch['attention_mask'])
    receipt_b_batch['token_type_ids'] = torch.stack(receipt_b_batch['token_type_ids'])
    receipt_b_batch['bbox'] = torch.stack(receipt_b_batch['bbox'])
    
    # Convert labels to tensor
    labels = torch.stack(labels)
    
    return {
        'receipt_a': receipt_a_batch,
        'receipt_b': receipt_b_batch,
        'labels': labels,
        'pair_types': pair_types
    }


def get_siamese_dataloaders(
    processed_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for Siamese model training.
    
    Args:
        processed_path: Path to dataset/processed directory
        tokenizer: Tokenizer for encoding
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Import here to avoid circular imports
    from .fraud_dataset import load_pair_dataset
    
    # Load datasets
    train_dataset = load_pair_dataset(
        processed_path=processed_path,
        tokenizer=tokenizer,
        split="train",
        max_length=max_length
    )
    
    test_dataset = load_pair_dataset(
        processed_path=processed_path,
        tokenizer=tokenizer,
        split="test",
        max_length=max_length
    )
    
    # Create train dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_pairs
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_pairs
    )
    
    return train_loader, test_loader


def get_class_weights(dataset: SROIEPairDataset) -> Tensor:
    """
    Computes class weights for imbalanced labels.
    Returns tensor of shape [2] for use in loss function.
    
    Args:
        dataset: SROIEPairDataset instance
        
    Returns:
        Tensor of shape [2] with class weights [weight_class_0, weight_class_1]
    """
    # Count labels
    total_samples = len(dataset)
    fraud_count = sum(1 for p in dataset.pairs if p['label'] == 1)
    legit_count = total_samples - fraud_count
    
    # Compute inverse frequency weights
    # weight = total_samples / (2 * class_count)
    weight_legit = total_samples / (2.0 * legit_count) if legit_count > 0 else 1.0
    weight_fraud = total_samples / (2.0 * fraud_count) if fraud_count > 0 else 1.0
    
    # Normalize weights so they sum to 2 (average weight = 1)
    total_weight = weight_legit + weight_fraud
    weight_legit = (weight_legit / total_weight) * 2.0
    weight_fraud = (weight_fraud / total_weight) * 2.0
    
    return torch.tensor([weight_legit, weight_fraud], dtype=torch.float32)


if __name__ == "__main__":
    """Test the dataloader utilities."""
    
    print("=" * 60)
    print("Testing Siamese Dataloader")
    print("=" * 60)
    
    try:
        from transformers import LayoutLMTokenizer
        
        # Use default tokenizer
        tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        
        # Load dataset
        processed_dir = Path(__file__).parent.parent.parent / "dataset" / "processed"
        
        if (processed_dir / "train_samples.json").exists():
            # Test collate function
            print("\n1. Testing collate_pairs function:")
            
            # Create a small dataset for testing
            from fraud_dataset import load_pair_dataset
            dataset = load_pair_dataset(processed_dir, tokenizer, split="train", max_length=128)
            
            # Get a small batch
            batch_samples = [dataset[i] for i in range(min(4, len(dataset)))]
            
            # Apply collate function
            batched = collate_pairs(batch_samples)
            
            print(f"   Batch size: {len(batch_samples)}")
            print(f"   Receipt A shapes:")
            print(f"     input_ids: {batched['receipt_a']['input_ids'].shape}")
            print(f"     attention_mask: {batched['receipt_a']['attention_mask'].shape}")
            print(f"     token_type_ids: {batched['receipt_a']['token_type_ids'].shape}")
            print(f"     bbox: {batched['receipt_a']['bbox'].shape}")
            
            print(f"   Receipt B shapes:")
            print(f"     input_ids: {batched['receipt_b']['input_ids'].shape}")
            print(f"     attention_mask: {batched['receipt_b']['attention_mask'].shape}")
            print(f"     token_type_ids: {batched['receipt_b']['token_type_ids'].shape}")
            print(f"     bbox: {batched['receipt_b']['bbox'].shape}")
            
            print(f"   Labels shape: {batched['labels'].shape}")
            print(f"   Labels: {batched['labels'].tolist()}")
            print(f"   Pair types: {batched['pair_types']}")
            
            # Test class weights
            print("\n2. Testing get_class_weights function:")
            class_weights = get_class_weights(dataset)
            print(f"   Class weights: {class_weights}")
            print(f"   Weight for legitimate (0): {class_weights[0]:.4f}")
            print(f"   Weight for fraud (1): {class_weights[1]:.4f}")
            
            # Test dataloader creation
            print("\n3. Testing get_siamese_dataloaders function:")
            try:
                train_loader, test_loader = get_siamese_dataloaders(
                    processed_dir, tokenizer, batch_size=2, max_length=128
                )
                
                print(f"   Train loader batches: {len(train_loader)}")
                print(f"   Test loader batches: {len(test_loader)}")
                
                # Load one batch from train loader
                for batch in train_loader:
                    print(f"\n   First train batch:")
                    print(f"     Batch size: {batch['labels'].shape[0]}")
                    print(f"     Receipt A input_ids: {batch['receipt_a']['input_ids'].shape}")
                    print(f"     Receipt B input_ids: {batch['receipt_b']['input_ids'].shape}")
                    print(f"     Labels: {batch['labels'].tolist()}")
                    break
                    
            except FileNotFoundError as e:
                print(f"   ⚠️  Test samples not found: {e}")
                print("   Only train samples available for testing.")
            
        else:
            print(f"\n   ⚠️  train_samples.json not found at {processed_dir}")
            print("   Run preprocessing first to generate this file.")
            
    except ImportError as e:
        print(f"\n   ⚠️  Could not import required packages: {e}")
        print("   Install with: pip install transformers torch")
        
    except Exception as e:
        print(f"\n   ⚠️  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)
