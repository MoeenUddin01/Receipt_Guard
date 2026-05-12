"""
src_2/inference/image_fraud_detector.py - Standalone image-based fraud detection.

Uses Siamese LayoutLM (Model 2) to compare receipt images directly without NER.
Supports a 'Claim' workflow with two databases:
1. known_receipts.json: Master list of valid receipts (seeded).
2. used_receipts.json: Receipts already submitted.
"""

import os
import json
import time
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from transformers import LayoutLMTokenizer

# Internal imports
from src_2.model.siamese_model import load_siamese_checkpoint, SiameseSimilarityModel
from src.inference.ocr import run_ocr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ImageFraudDetector:
    """
    Standalone fraud detection system using Siamese LayoutLM embeddings.
    Supports a 'Claim' workflow with two databases.
    """

    def __init__(
        self,
        model2_checkpoint: str,
        model_path: str = "microsoft/layoutlm-base-uncased",
        known_db_path: str = "artifacts/known_receipts.json",
        used_db_path: str = "artifacts/used_receipts.json"
    ):
        """
        Initialize the detector with a two-database system.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load Siamese model
        logger.info(f"Loading Siamese model from {model2_checkpoint}...")
        checkpoint_info = load_siamese_checkpoint(model2_checkpoint)
        self.model = checkpoint_info['model'].to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_path)

        # Database paths
        self.known_db_path = Path(known_db_path)
        self.used_db_path = Path(used_db_path)

        # Load known (master) database
        self.known_db = {}
        if self.known_db_path.exists():
            with open(self.known_db_path, "r") as f:
                self.known_db = json.load(f)
            logger.info(f"Loaded {len(self.known_db)} known receipts from {known_db_path}")
        else:
            logger.warning(f"Known database {known_db_path} not found. Please run seed_database.py")

        # Load used (claimed) database
        self.used_db = {}
        if self.used_db_path.exists():
            with open(self.used_db_path, "r") as f:
                self.used_db = json.load(f)
            logger.info(f"Loaded {len(self.used_db)} used receipts from {used_db_path}")

    def get_embedding(self, image_path: str) -> List[float]:
        """Extract CLS embedding from receipt image."""
        tokens_data = run_ocr(image_path)
        if not tokens_data:
            return [0.0] * self.model.projection_dim

        # Get actual dimensions used during OCR (run_ocr resizes to max 2000)
        with Image.open(image_path) as img:
            w, h = img.size
            max_dim = max(w, h)
            if max_dim > 2000:
                scale = 2000 / max_dim
                width, height = int(w * scale), int(h * scale)
            else:
                width, height = w, h

        words = [t['text'] for t in tokens_data]
        bboxes = [t['bbox_normalized'] for t in tokens_data]

        normalized_bboxes = []
        for bbox in bboxes:
            normalized_bboxes.append([
                int(1000 * (bbox[0] / width)),
                int(1000 * (bbox[1] / height)),
                int(1000 * (bbox[2] / width)),
                int(1000 * (bbox[3] / height))
            ])

        encoding = self.tokenizer(words, is_split_into_words=True, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        word_ids = encoding.word_ids()
        aligned_bboxes = []
        for word_idx in word_ids:
            if word_idx is None:
                aligned_bboxes.append([0, 0, 0, 0])
            else:
                idx = min(word_idx, len(normalized_bboxes) - 1)
                aligned_bboxes.append(normalized_bboxes[idx])

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids)).to(self.device)
        bbox_tensor = torch.tensor([aligned_bboxes], dtype=torch.long).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, bbox=bbox_tensor)

        return embedding[0].cpu().tolist()

    def _find_match(self, embedding: List[float], database: Dict) -> Tuple[Optional[str], float]:
        """Helper to find match in a specific database using cosine similarity."""
        if not database:
            return None, 0.0

        emb_np = np.array(embedding)
        best_id = None
        best_score = -1.0

        for rid, data in database.items():
            db_emb = np.array(data['embedding'])
            norm_a = np.linalg.norm(emb_np)
            norm_b = np.linalg.norm(db_emb)
            if norm_a == 0 or norm_b == 0:
                score = 0.0
            else:
                score = np.dot(emb_np, db_emb) / (norm_a * norm_b)
            
            if score > best_score:
                best_score = score
                best_id = rid

        return best_id, float(best_score)

    def check_receipt(self, image_path: str, threshold: float = 0.85) -> Dict:
        """
        Main logic: Check BOTH databases and pick the best match.
        The database with the higher similarity score wins.
        """
        logger.info(f"Processing receipt: {image_path}")
        embedding = self.get_embedding(image_path)

        # Search both databases
        matched_known_id, known_score = self._find_match(embedding, self.known_db)
        matched_used_id, used_score = self._find_match(embedding, self.used_db)

        logger.info(f"Scores — known_db best: {known_score:.4f} ({matched_known_id}), used_db best: {used_score:.4f} ({matched_used_id})")

        # Neither database has a match above threshold → unknown receipt
        if (known_score < threshold) and (used_score < threshold):
            logger.warning(f"No match found in either database")
            return {
                "verdict": "FRAUD",
                "confidence": 1.0 - max(used_score, known_score),
                "message": "Not generated by us"
            }

        # Compare: whichever database has the HIGHER score wins
        if known_score >= used_score:
            # Best match is in known_db → LEGITIMATE first-time claim
            logger.info(f"Legitimate claim: {matched_known_id} (known={known_score:.4f} > used={used_score:.4f})")

            receipt_data = self.known_db.pop(matched_known_id)
            receipt_data['claimed_at'] = datetime.now().isoformat()
            receipt_data['claimed_image'] = str(image_path)

            self.used_db[matched_known_id] = receipt_data
            self.save_databases()

            return {
                "verdict": "LEGITIMATE",
                "confidence": known_score,
                "receipt_id": matched_known_id,
                "message": "Valid receipt recognized and successfully claimed!"
            }
        else:
            # Best match is in used_db → FRAUD duplicate
            logger.warning(f"Fraud: {matched_used_id} (used={used_score:.4f} > known={known_score:.4f})")
            return {
                "verdict": "FRAUD",
                "confidence": used_score,
                "matched_receipt": matched_used_id,
                "matched_at": self.used_db[matched_used_id].get('claimed_at'),
                "message": "This receipt has already been claimed and used."
            }

    def save_databases(self):
        """Persist both databases to JSON."""
        try:
            # Ensure artifacts directory exists
            self.known_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.known_db_path, "w") as f:
                json.dump(self.known_db, f, indent=2)
            with open(self.used_db_path, "w") as f:
                json.dump(self.used_db, f, indent=2)
            logger.info("Databases successfully updated and saved.")
        except Exception as e:
            logger.error(f"Failed to save databases: {e}")


def print_verdict_box(result: Dict):
    """
    Prints a formatted box with the verdict results.
    """
    verdict = result['verdict']
    confidence = result['confidence']
    receipt_id = result.get('receipt_id', result.get('matched_receipt', 'N/A'))
    
    # Format timestamp
    raw_date = result.get('matched_at', datetime.now().isoformat())
    try:
        first_seen = datetime.fromisoformat(raw_date).strftime("%Y-%m-%d")
    except:
        first_seen = raw_date.split('T')[0]

    symbol = "🚨 FRAUD" if verdict == "FRAUD" else "✅ LEGITIMATE"
    
    print("\n  ┌─────────────────────────────────┐")
    print("  │  ReceiptGuard Verdict           │")
    print("  ├─────────────────┬───────────────┤")
    print(f"  │  Verdict        │ {symbol:<13} │")
    print(f"  │  Confidence     │ {confidence:<13.2f} │")
    print(f"  │  First seen     │ {first_seen:<13} │")
    print(f"  │  Receipt ID     │ {receipt_id[:10]}...    │")
    print("  └─────────────────┴───────────────┘\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReceiptGuard-ML Image Fraud Detector")
    parser.add_argument("--image", required=True, help="Path to receipt image")
    parser.add_argument("--model2", required=True, help="Path to Siamese model checkpoint")
    parser.add_argument("--model_path", default="microsoft/layoutlm-base-uncased", help="Base model for tokenizer")
    parser.add_argument("--known_db", default="artifacts/known_receipts.json", help="Path to known database")
    parser.add_argument("--used_db", default="artifacts/used_receipts.json", help="Path to used database")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (default: 0.85)")

    args = parser.parse_args()

    try:
        detector = ImageFraudDetector(
            model2_checkpoint=args.model2,
            model_path=args.model_path,
            known_db_path=args.known_db,
            used_db_path=args.used_db
        )
        
        result = detector.check_receipt(args.image, threshold=args.threshold)
        print_verdict_box(result)
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
