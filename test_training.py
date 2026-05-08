#!/usr/bin/env python3
"""
Quick test script to identify training errors before full Kaggle run.
Uses minimal data and epochs for fast debugging.
"""

import sys
import os
sys.path.append('/home/moeen/projects/ReceiptGuard-ML')

from src.pipelines.model_training_pipeline import run_training_pipeline
from src.config import CFG

def main():
    print("=" * 60)
    print("QUICK TRAINING TEST - Error Detection")
    print("=" * 60)
    
    # Create minimal test config by overriding the global CFG
    from src.config import override_config
    
    # Override with minimal test settings
    test_overrides = {
        'training.num_epochs': 1,
        'training.batch_size': 2,
        'model.model_path': 'microsoft/layoutlm-base-uncased'
    }
    override_config(test_overrides)
    test_config = CFG
    
    print(f"Test config:")
    print(f"  Epochs: {test_config.training.num_epochs}")
    print(f"  Batch size: {test_config.training.batch_size}")
    print(f"  Train samples: {getattr(test_config.data, 'train_samples', 'N/A')}")
    print(f"  Val samples: {getattr(test_config.data, 'val_samples', 'N/A')}")
    
    try:
        print("\nStarting quick training test...")
        training_summary = run_training_pipeline(test_config)
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final training loss: {training_summary.get('final_train_loss', 'N/A')}")
        print(f"Final validation loss: {training_summary.get('final_val_loss', 'N/A')}")
        print("Ready for full Kaggle training!")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED - Error Found!")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
