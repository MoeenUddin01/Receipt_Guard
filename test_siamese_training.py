#!/usr/bin/env python3
"""
Quick test script for Siamese Model (Model 2) to identify errors before full Kaggle run.
Uses minimal data and epochs for fast debugging.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from src_2.pipelines.siamese_training_pipeline import run_siamese_training_pipeline, SiameseTrainingConfig

def main():
    print("=" * 60)
    print("QUICK SIAMESE TRAINING TEST - Error Detection")
    print("=" * 60)
    
    # Force CPU for local test due to incompatibility
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Create minimal test config
    test_config = SiameseTrainingConfig(
        model_path="microsoft/layoutlm-base-uncased", # Use online model for test
        output_dir="artifacts/siamese/test_run",
        num_epochs=1,
        batch_size=2,
        max_length=128, # Smaller max_length for faster test
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        seed=42
    )
    
    print(f"Test config:")
    print(f"  Model path: {test_config.model_path}")
    print(f"  Epochs: {test_config.num_epochs}")
    print(f"  Batch size: {test_config.batch_size}")
    print(f"  Max length: {test_config.max_length}")
    print(f"  Output dir: {test_config.output_dir}")
    
    # Check if processed data exists
    processed_dir = PROJECT_ROOT / "dataset" / "processed"
    if not (processed_dir / "train_samples.json").exists():
        print("\n" + "!" * 60)
        print("ERROR: Processed data not found!")
        print(f"Please run the Model 1 preprocessing first.")
        print("!" * 60)
        return False

    try:
        print("\nStarting quick Siamese training test...")
        summary = run_siamese_training_pipeline(test_config)
        
        if summary.get('status') == 'completed':
            print("\n" + "=" * 60)
            print("SIAMESE TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Best threshold: {summary.get('best_threshold', 'N/A')}")
            print("Ready for full Kaggle training!")
            return True
        else:
            print("\n" + "=" * 60)
            print("SIAMESE TEST FAILED!")
            print("=" * 60)
            print(f"Error: {summary.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print("\n" + "=" * 60)
        print("SIAMESE TEST FAILED - Exception Raised!")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
