#!/usr/bin/env python3
"""
STEP 1: Verify the BIO fix works on raw samples manually.
"""

import sys
import json
from pathlib import Path
sys.path.append('/home/moeen/projects/ReceiptGuard-ML')

from src.data.preprocessing import parse_box_file, assign_bio_labels

def verify_fix_on_samples():
    print("=" * 80)
    print("STEP 1: VERIFYING BIO FIX ON RAW SAMPLES")
    print("=" * 80)
    
    # Get training samples
    raw_train_dir = Path("dataset/raw/SROIE2019/train")
    box_dir = raw_train_dir / "box"
    entities_dir = raw_train_dir / "entities"
    
    if not box_dir.exists() or not entities_dir.exists():
        print(f"❌ Training directories not found")
        print(f"Box dir: {box_dir}")
        print(f"Entities dir: {entities_dir}")
        return False
    
    # Get first few box files
    box_files = list(box_dir.glob("*.txt"))[:3]
    
    if not box_files:
        print(f"❌ No box files found in {box_dir}")
        return False
    
    print(f"Testing on {len(box_files)} samples from training set")
    
    all_passed = True
    
    for i, box_file in enumerate(box_files):
        sample_id = box_file.stem
        entity_file = entities_dir / f"{sample_id}.txt"
        
        print(f"\n--- Sample {i+1}: {sample_id} ---")
        
        if not entity_file.exists():
            print(f"❌ Entity file not found: {entity_file}")
            all_passed = False
            continue
        
        try:
            # Parse box file
            tokens = parse_box_file(str(box_file))
            print(f"✅ Parsed {len(tokens)} tokens")
            
            # Parse entity file
            with open(entity_file, 'r', encoding='utf-8') as f:
                entity_lines = f.readlines()
            
            entities = {}
            for line in entity_lines:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    entities[key.strip().lower()] = value.strip()
            
            print(f"✅ Parsed entities: {list(entities.keys())}")
            
            # Run BIO assignment
            labels = assign_bio_labels(tokens, entities)
            
            if labels is None:
                print(f"❌ BIO assignment returned None")
                all_passed = False
                continue
            
            # Count labels
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            entity_labels = [label for label in labels if label != 'O']
            
            print(f"Label distribution:")
            for label, count in sorted(label_counts.items()):
                percentage = (count / len(labels)) * 100
                print(f"  {label:<12}: {count:>3} ({percentage:>5.1f}%)")
            
            if not entity_labels:
                print(f"❌ NO ENTITY LABELS ASSIGNED!")
                all_passed = False
            else:
                print(f"✅ {len(entity_labels)} entity labels assigned")
                
                # Show entity tokens
                print(f"Entity tokens:")
                entity_count = 0
                for token_idx, (token, label) in enumerate(zip(tokens, labels)):
                    if label != 'O' and entity_count < 10:
                        print(f"  {token_idx:2d}: '{token['text']:<30}' -> {label}")
                        entity_count += 1
                
                # Check for expected entities
                entity_types_found = set()
                for label in entity_labels:
                    entity_type = label.split('-')[1]
                    entity_types_found.add(entity_type)
                
                print(f"Entity types found: {sorted(entity_types_found)}")
                
                if len(entity_types_found) >= 2:
                    print(f"✅ Multiple entity types detected")
                else:
                    print(f"⚠️  Only one entity type detected")
            
        except Exception as e:
            print(f"❌ Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print(f"\n" + "=" * 80)
    if all_passed:
        print(f"✅ STEP 1 PASSED - BIO fix verified on training samples")
        print(f"   Ready to proceed to preprocessing pipeline")
    else:
        print(f"❌ STEP 1 FAILED - Issues detected in BIO assignment")
        print(f"   STOP - Do not proceed to retraining")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = verify_fix_on_samples()
    sys.exit(0 if success else 1)
