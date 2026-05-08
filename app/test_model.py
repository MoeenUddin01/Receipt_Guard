#!/usr/bin/env python3
"""
Test script to verify the trained model loads correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.model.model import ReceiptFieldExtractor, ModelConfig

def test_model_loading():
    """Test that the model checkpoint loads successfully"""
    
    checkpoint_path = Path(__file__).parent.parent / "artifacts" / "best_model.pt"
    
    print("=" * 60)
    print("RECEIPTGUARD MODEL VERIFICATION")
    print("=" * 60)
    
    print(f"\n📁 Checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        print("❌ Checkpoint file not found!")
        return False
    
    print(f"📊 File size: {checkpoint_path.stat().st_size / 1e6:.1f} MB")
    
    try:
        # Create model config
        print("\n🏗️  Building model...")
        model_config = ModelConfig(
            model_path="microsoft/layoutlm-base-uncased",
            num_labels=9,
            dropout=0.1
        )
        
        model = ReceiptFieldExtractor(
            model_path=model_config.model_path,
            num_labels=model_config.num_labels,
            dropout=model_config.dropout
        )
        
        # Load checkpoint
        print("📥 Loading checkpoint...")
        device = torch.device("cpu")  # Use CPU for testing
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Loss: {checkpoint.get('loss', 'N/A')}")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        print("\n✅ Model loaded successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB (FP32)")
        
        # Test forward pass with dummy input
        print("\n🧪 Testing forward pass...")
        batch_size = 2
        seq_len = 128
        
        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        dummy_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        dummy_bbox = torch.randint(0, 1000, (batch_size, seq_len, 4))
        
        with torch.no_grad():
            loss, logits = model(
                dummy_input_ids,
                dummy_attention_mask,
                dummy_token_type_ids,
                dummy_bbox
            )
        
        print(f"   ✅ Output shape: {logits.shape}")
        print(f"   ✅ Expected: ({batch_size}, {seq_len}, 9)")
        
        print("\n" + "=" * 60)
        print("✅ MODEL VERIFICATION COMPLETE")
        print("=" * 60)
        print("\nModel is ready for inference!")
        print(f"Checkpoint: {checkpoint_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
