#!/usr/bin/env python3
"""
CHECK 3: Test Model 1 on known SROIE sample to compare predictions vs ground truth.
"""

import sys
import torch
import json
import os
from pathlib import Path
sys.path.append('/home/moeen/projects/ReceiptGuard-ML')

from src.model.model import ReceiptFieldExtractor
from src.data.dataset import ReceiptDataset
from src.model.evaluation import ID2LABEL

def test_model_on_sample():
    print("=" * 60)
    print("CHECK 3: TESTING MODEL 1 ON SROIE SAMPLE")
    print("=" * 60)
    
    # Paths
    test_samples_path = "dataset/processed/test_samples.json"
    model_path = "artifacts/best_model.pt"
    base_model_path = "dataset/raw/SROIE2019/layoutlm-base-uncased"
    
    if not os.path.exists(test_samples_path):
        print(f"❌ Test samples not found: {test_samples_path}")
        return False
    
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint not found: {model_path}")
        return False
    
    # Load test samples
    with open(test_samples_path, 'r') as f:
        test_samples = json.load(f)
    
    if not test_samples:
        print("❌ No test samples found")
        return False
    
    # Get first sample
    sample = test_samples[0]
    print(f"Testing on sample: {sample.get('id', 'unknown')}")
    
    # Load model
    print("Loading Model 1...")
    try:
        model = ReceiptFieldExtractor.load_from_checkpoint(
            model_path, 
            model_path=base_model_path, 
            num_labels=9
        )
        model.eval()
        device = torch.device('cpu')
        model.to(device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Create dataset-like input (same tokenization as training)
    print("\n🔍 PREDICTION ANALYSIS")
    print("-" * 50)
    
    # Extract sample data
    tokens = sample.get('tokens', [])
    bboxes = sample.get('bboxes', [])
    labels = sample.get('labels', [])
    
    print(f"Sample has {len(tokens)} tokens, {len(bboxes)} bboxes, {len(labels)} labels")
    
    # Create dataset-style input
    dataset = ReceiptDataset(
        samples=[sample],
        tokenizer_path=base_model_path,
        max_length=512
    )
    
    # Get the first (and only) item from dataset
    item = dataset[0]
    
    # Move to device
    for key in item:
        if isinstance(item[key], torch.Tensor):
            item[key] = item[key].unsqueeze(0).to(device)  # Add batch dimension
    
    # Run prediction
    with torch.no_grad():
        loss, logits = model(
            item['input_ids'],
            item['attention_mask'], 
            item['token_type_ids'],
            item['bbox'],
            item['labels']
        )
        
        # Get predictions
        predictions = model.get_predictions(logits, item['attention_mask'])
        pred_ids = predictions[0].cpu().tolist()
        true_ids = item['labels'][0].cpu().tolist()
    
    # Convert to label names
    pred_labels = [ID2LABEL.get(pid, 'O') for pid in pred_ids]
    true_labels = [ID2LABEL.get(tid, 'O') if tid != -100 else 'PAD' for tid in true_ids]
    
    # Get attention mask to find valid tokens
    attention_mask = item['attention_mask'][0].cpu().tolist()
    valid_positions = [i for i, mask in enumerate(attention_mask) if mask == 1]
    
    # Get word_ids for alignment
    word_ids = dataset.tokenizer.convert_ids_to_tokens(item['input_ids'][0])
    word_id_mapping = dataset.tokenizer(item['tokens'], is_split_into_words=True, return_tensors="pt").word_ids()
    
    print(f"\nFirst 20 valid tokens comparison:")
    print("Token          True label     Predicted")
    print("─" * 45)
    
    valid_count = 0
    correct_predictions = 0
    
    for i, pos in enumerate(valid_positions):
        if valid_count >= 20:
            break
            
        # Get word id for this position
        word_id = word_id_mapping[pos] if pos < len(word_id_mapping) else None
        
        # Only print for actual words (not special tokens)
        if word_id is not None and word_id < len(tokens):
            token = tokens[word_id]
            true_label = true_labels[pos] if pos < len(true_labels) else 'O'
            pred_label = pred_labels[pos] if pos < len(pred_labels) else 'O'
            
            print(f"{token:<15} {true_label:<14} {pred_label}")
            
            if true_label == pred_label:
                correct_predictions += 1
            
            valid_count += 1
    
    # Calculate accuracy on valid tokens
    total_valid = sum(1 for pos in valid_positions 
                    if pos < len(word_id_mapping) and 
                    word_id_mapping[pos] is not None and 
                    word_id_mapping[pos] < len(tokens))
    
    accuracy = correct_predictions / total_valid if total_valid > 0 else 0
    
    print(f"\n📊 ACCURACY ANALYSIS:")
    print(f"Valid tokens analyzed: {total_valid}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Check if model predicts all O
    entity_predictions = [label for label in pred_labels if label != 'O']
    if not entity_predictions:
        print("\n❌ MODEL PREDICTS ALL O LABELS!")
        print("   This confirms the issue - model did not learn entities")
    else:
        print(f"\n✅ Model predicts {len(entity_predictions)} entity labels")
        entity_counter = {}
        for label in entity_predictions:
            entity_counter[label] = entity_counter.get(label, 0) + 1
        for label, count in entity_counter.items():
            print(f"   {label}: {count}")
    
    # Check ground truth entities
    true_entities = [label for label in true_labels if label not in ['O', 'PAD']]
    if not true_entities:
        print("\n❌ GROUND TRUTH HAS NO ENTITY LABELS!")
        print("   This indicates broken preprocessing")
    else:
        print(f"\n✅ Ground truth has {len(true_entities)} entity labels")
    
    print(f"\n✅ CHECK 3 COMPLETED")
    return True

if __name__ == "__main__":
    success = test_model_on_sample()
    sys.exit(0 if success else 1)
