"""
src_2/inference/seed_database.py - Populate the known_receipts database with test images.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src_2.inference.image_fraud_detector import ImageFraudDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_database(
    image_dir: str,
    model_checkpoint: str,
    output_path: str = "artifacts/known_receipts.json",
    limit: int = 20
):
    """
    Process images from image_dir and save embeddings to output_path.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return

    # Initialize detector
    detector = ImageFraudDetector(
        model2_checkpoint=model_checkpoint,
        known_db_path=output_path,
        used_db_path="/tmp/unused_db.json"
    )

    known_receipts = {}
    image_files = sorted(list(image_dir.glob("*.jpg"))) + sorted(list(image_dir.glob("*.png")))
    
    if limit > 0:
        image_files = image_files[:limit]

    logger.info(f"Seeding database with {len(image_files)} images...")

    for img_path in tqdm(image_files):
        try:
            embedding = detector.get_embedding(str(img_path))
            # Use filename as ID for easier tracking in demo
            receipt_id = img_path.name
            
            known_receipts[receipt_id] = {
                "embedding": embedding,
                "original_filename": img_path.name,
                "seeded_at": Path(img_path).stat().st_mtime
            }
        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}")

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(known_receipts, f, indent=2)

    logger.info(f"Successfully seeded {len(known_receipts)} receipts to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed ReceiptGuard Database")
    parser.add_argument("--img_dir", default="dataset/raw/SROIE2019/test/img", help="Directory with test images")
    parser.add_argument("--model", default="artifacts/siamese/2_model/best_model.pth", help="Model checkpoint")
    parser.add_argument("--output", default="artifacts/known_receipts.json", help="Output JSON path")
    parser.add_argument("--limit", type=int, default=20, help="Max images to process")

    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        # Try fallback
        fallback = "artifacts/siamese/best_model.pth"
        if Path(fallback).exists():
            args.model = fallback
        else:
            logger.warning(f"Model {args.model} not found. Seeding might use random weights if siamese_model.py logic triggers.")

    seed_database(args.img_dir, args.model, args.output, args.limit)
