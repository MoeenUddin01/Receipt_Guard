#!/usr/bin/env python3
"""
Debug script to test box file parsing
"""

import sys
sys.path.append('/home/moeen/projects/ReceiptGuard-ML')

from src.data.preprocessing import parse_box_file

def main():
    print("=" * 60)
    print("DEBUG BOX FILE PARSING")
    print("=" * 60)
    
    # Test parsing a sample box file
    box_file = "dataset/raw/SROIE2019/train/box/X00016469612.txt"
    print(f"Testing file: {box_file}")
    
    try:
        tokens = parse_box_file(box_file)
        print(f"Number of tokens parsed: {len(tokens)}")
        
        if tokens:
            print("\nFirst few tokens:")
            for i, token in enumerate(tokens[:3]):
                print(f"  {i+1}: '{token['text']}' -> bbox: {token['bbox_normalized']}")
        else:
            print("ERROR: No tokens parsed!")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
