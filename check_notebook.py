# Cell 2
# Install required packages
!pip install transformers torch torchvision tqdm tensorboard scikit-learn
!pip install sentencepiece tiktoken

import sys
import os
import json
import logging
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")

# Cell 4
# Kaggle dataset paths
KAGGLE_INPUT = '/kaggle/input'
DATASET_PATH = '/kaggle/input/datasets/urbikn/sroie-datasetv2/SROIE2019'
RAW_DATA_PATH = DATASET_PATH
WORKING_DIR = '/kaggle/working'

print(f"Raw data path: {RAW_DATA_PATH}")
print(f"Working directory: {WORKING_DIR}")

# Verify dataset structure
if os.path.exists(RAW_DATA_PATH):
    print("\nDataset structure:")
    for split in ['train', 'test']:
        split_path = f'{RAW_DATA_PATH}/{split}'
        if os.path.exists(split_path):
            img_count = len(os.listdir(f'{split_path}/img')) if os.path.exists(f'{split_path}/img') else 0
            box_count = len(os.listdir(f'{split_path}/box')) if os.path.exists(f'{split_path}/box') else 0
            entity_count = len(os.listdir(f'{split_path}/entities')) if os.path.exists(f'{split_path}/entities') else 0
            print(f"  {split}: {img_count} images, {box_count} box files, {entity_count} entity files")
        else:
            print(f"  {split}: directory not found")
else:
    print(f"❌ Dataset not found at {RAW_DATA_PATH}")
    print("Please make sure the SROIE dataset path is correct")

# Cell 6
# Create working directory structure
!mkdir -p {WORKING_DIR}/src/data {WORKING_DIR}/src/model {WORKING_DIR}/src/pipelines {WORKING_DIR}/src/config {WORKING_DIR}/artifacts

# Download the fixed code from GitHub
!git clone https://github.com/MoeenUddin01/Receipt_Guard.git {WORKING_DIR}/Receipt_Guard

# Copy the fixed source code
!cp -r {WORKING_DIR}/Receipt_Guard/src/* {WORKING_DIR}/src/
!cp -r {WORKING_DIR}/Receipt_Guard/dataset/raw {WORKING_DIR}/dataset/

# Create config file
config_content = '''
# Model Configuration
model:
  model_path: "{RAW_DATA_PATH}/layoutlm-base-uncased"
  num_labels: 9
  dropout: 0.1

# Training Configuration  
training:
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  batch_size: 8
  num_epochs: 10

# Data Configuration
data:
  max_length: 512
  data_path: "{WORKING_DIR}/dataset/processed"
'''

with open(f'{WORKING_DIR}/src/config/config.yaml', 'w') as f:
    f.write(config_content)

print("✅ Code and config setup complete")

# Cell 8
# Add paths to Python path
sys.path.insert(0, f'{WORKING_DIR}/src')
sys.path.insert(0, f'{WORKING_DIR}/src/data')
sys.path.insert(0, f'{WORKING_DIR}/src/model')

# Change to working directory
os.chdir(WORKING_DIR)

# Debug: Check if dataset path exists
print(f"Dataset path exists: {os.path.exists(RAW_DATA_PATH)}")
print(f"Dataset path contents: {os.listdir(RAW_DATA_PATH) if os.path.exists(RAW_DATA_PATH) else 'Not found'}")

# Create dataset/processed directory if it doesn't exist
os.makedirs("dataset/processed", exist_ok=True)

# Run preprocessing with correct dataset path and verbose output
print("🔄 Running preprocessing pipeline...")
print(f"Raw data path: {RAW_DATA_PATH}")
print(f"Processed path: dataset/processed")

# Run with error handling
import subprocess
result = subprocess.run([
    "python", "-m", "src.pipelines.preprocessing_pipeline", 
    "--raw_path", RAW_DATA_PATH, 
    "--processed_path", "dataset/processed"
], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

# Check if files were created
print(f"\nFiles in dataset/processed/:")
if os.path.exists("dataset/processed"):
    files = os.listdir("dataset/processed")
    print(files)
else:
    print("Directory does not exist")

print("\n✅ Preprocessing complete!")

# Cell 10
# Check the processed data
import json

try:
    with open('dataset/processed/train_samples.json', 'r') as f:
        train_samples = json.load(f)
    
    with open('dataset/processed/label_stats.json', 'r') as f:
        label_stats = json.load(f)
    
    print(f"✅ Training samples: {len(train_samples)}")
    print(f"\nLabel distribution:")
    
    for label, count in sorted(label_stats['label_counts'].items()):
        percentage = (count / label_stats['total_labels']) * 100
        print(f"  {label:<12}: {count:>6} ({percentage:>5.1f}%)")
    
    # Check first sample for entity labels
    sample = train_samples[0]
    entity_labels = [l for l in sample['labels'] if l != 'O']
    
    print(f"\nFirst sample analysis:")
    print(f"  Total tokens: {len(sample['labels'])}")
    print(f"  Entity labels: {len(entity_labels)}")
    print(f"  Entity types: {set([l.split('-')[1] for l in entity_labels])}")
    print(f"  Entity values: {sample.get('entity_values', {})}")
    
    if len(entity_labels) > 0:
        print("\n🎉 SUCCESS: Entity labels found in processed data!")
    else:
        print("\n❌ WARNING: No entity labels found - check preprocessing")
        
except Exception as e:
    print(f"❌ Error loading processed data: {e}")

# Cell 12
# Backup old model if exists
!mv artifacts/best_model.pt artifacts/best_model_broken_backup.pt 2>/dev/null || true

# Start training
print("🚀 Starting Model 1 training...")
print("Expected behavior:")
print("  - Epoch 1-2: Model starts predicting non-O labels")
print("  - Epoch 3: Token accuracy > 85%")
print("  - Loss decreases each epoch")
print("\nTraining output:")
print("=" * 50)

!python -m src.pipelines.model_training_pipeline --num_epochs 10

# Cell 14
# Check if training completed successfully
import os
import glob

# Look for new checkpoint
checkpoint_files = glob.glob('artifacts/model1/checkpoints/best_model.pt')
checkpoint_files.extend(glob.glob('dataset/processed/checkpoints/best_model.pt'))

if checkpoint_files:
    print(f"✅ Training completed! Checkpoint found: {checkpoint_files[0]}")
    
    # Check file size
    file_size = os.path.getsize(checkpoint_files[0]) / (1024*1024)  # MB
    print(f"   File size: {file_size:.1f} MB")
    
    if file_size > 100:  # Should be around 400+ MB
        print("   ✅ File size looks correct")
    else:
        print("   ⚠️  File size seems small - check if training completed")
else:
    print("❌ No checkpoint found - training may have failed")
    
    # Check for any training logs
    log_files = glob.glob('**/training.log') + glob.glob('**/tensorboard_logs/**')
    if log_files:
        print(f"   Log files found: {log_files}")

# Cell 16
# Quick test of the trained model
try:
    import torch
    from src.model.model import ReceiptFieldExtractor
    from transformers import LayoutLMTokenizer
    
    # Find checkpoint
    checkpoint_path = None
    for path in ['artifacts/model1/checkpoints/best_model.pt', 'dataset/processed/checkpoints/best_model.pt']:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path:
        print(f"🧪 Testing model from: {checkpoint_path}")
        
        # Load model
        model = ReceiptFieldExtractor.load_from_checkpoint(
            checkpoint_path,
            model_path=f"{RAW_DATA_PATH}/layoutlm-base-uncased",
            num_labels=9
        )
        model.eval()
        
        # Load tokenizer
        tokenizer = LayoutLMTokenizer.from_pretrained(f"{RAW_DATA_PATH}/layoutlm-base-uncased")
        
        # Test with sample data
        sample_tokens = ["BOOK", "TA", ".K", "(TAMAN", "DAYA)", "SDN", "BHD"]
        sample_bboxes = [[0, 0, 100, 20] for _ in sample_tokens]
        
        # Tokenize
        encoding = tokenizer(
            sample_tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Create input tensors
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = torch.zeros_like(input_ids)
        bbox = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
        
        # Run inference
        with torch.no_grad():
            loss, logits = model(input_ids, attention_mask, token_type_ids, bbox)
            predictions = model.get_predictions(logits, attention_mask)
        
        # Check predictions
        pred_ids = predictions[0][:len(sample_tokens)]
        
        # Convert to labels (simplified)
        id2label = {0: "O", 1: "B-COMPANY", 2: "I-COMPANY", 3: "B-DATE", 4: "I-DATE", 
                  5: "B-ADDRESS", 6: "I-ADDRESS", 7: "B-TOTAL", 8: "I-TOTAL"}
        
        pred_labels = [id2label.get(pid, "O") for pid in pred_ids]
        
        print("\nTest results:")
        for token, label in zip(sample_tokens, pred_labels):
            print(f"  {token:<12} -> {label}")
        
        # Check if model predicts entities
        entity_predictions = [label for label in pred_labels if label != "O"]
        if entity_predictions:
            print(f"\n🎉 SUCCESS! Model predicts {len(entity_predictions)} entity labels")
            print(f"   Entity types: {set([l.split('-')[1] for l in entity_predictions])}")
        else:
            print(f"\n⚠️  Model still predicts only O labels - may need more training")
            
    else:
        print("❌ No checkpoint found for testing")
        
except Exception as e:
    print(f"❌ Error testing model: {e}")
    import traceback
    traceback.print_exc()

