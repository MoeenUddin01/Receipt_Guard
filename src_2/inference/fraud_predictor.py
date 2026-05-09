"""
fraud_predictor.py - Final combined inference engine for fraud detection.

This module combines Model 1 (NER), Model 2 (Siamese), and ReceiptLedger
to produce comprehensive fraud verdicts for real receipts.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import LayoutLMTokenizer

from ..model.siamese_model import SiameseSimilarityModel, load_siamese_checkpoint
from ..receipt_ledger import ReceiptLedger
from ..train import detect_device
from src.inference.ocr import run_ocr
from src.model.receipt_field_extractor import ReceiptFieldExtractor


class ReceiptGuardPredictor:
    """
    Combined inference engine using both models + ledger for fraud detection.
    """
    
    def __init__(
        self,
        model1_checkpoint: str,  # NER model from src/
        model2_checkpoint: str,  # Siamese model from src_2/
        model_path: str = "dataset/raw/SROIE2019/layoutlm-base-uncased",
        ledger_path: str = "artifacts/ledger.json",
        device: str = "auto"
    ):
        """
        Initialize the ReceiptGuard predictor.
        
        Args:
            model1_checkpoint: Path to Model 1 (NER) checkpoint
            model2_checkpoint: Path to Model 2 (Siamese) checkpoint
            model_path: Path to base LayoutLM model
            ledger_path: Path to receipt ledger file
            device: Device to use ('auto', 'cuda', 'cpu', etc.)
        """
        # Detect device
        if device == "auto":
            self.device = detect_device()
        else:
            self.device = torch.device(device)
        
        print(f"Initializing ReceiptGuardPredictor on {self.device}")
        
        # Load Model 1 (NER)
        print(f"Loading Model 1 (NER) from {model1_checkpoint}...")
        self.model1 = ReceiptFieldExtractor.load_from_checkpoint(model1_checkpoint)
        self.model1.to(self.device)
        self.model1.eval()
        print(f"✓ Model 1 loaded")
        
        # Load Model 2 (Siamese)
        print(f"Loading Model 2 (Siamese) from {model2_checkpoint}...")
        checkpoint_info = load_siamese_checkpoint(model2_checkpoint)
        self.model2 = checkpoint_info['model']
        self.model2.to(self.device)
        self.model2.eval()
        
        # Extract similarity threshold from Model 2 checkpoint
        self.similarity_threshold = checkpoint_info['similarity_threshold']
        print(f"✓ Model 2 loaded with threshold: {self.similarity_threshold:.3f}")
        
        # Load tokenizer
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_path)
        print(f"✓ Tokenizer loaded")
        
        # Load ledger
        print(f"Loading ledger from {ledger_path}...")
        self.ledger = ReceiptLedger(ledger_path)
        print(f"✓ Ledger loaded with {len(self.ledger.receipts)} stored receipts")
        
        print("✓ ReceiptGuardPredictor initialized successfully")
    
    def extract_fields(self, box_file_path: str) -> Dict:
        """
        Extract fields from a box file using Model 1.
        
        Args:
            box_file_path: Path to box file
            
        Returns:
            Dictionary with extracted fields {company, date, address, total}
        """
        # Load and process box file
        with open(box_file_path, 'r') as f:
            box_data = json.load(f)
        
        # Extract fields using Model 1
        extracted_fields = self.model1.extract_fields(box_data)
        
        return extracted_fields
    
    def predict(self, box_file_path: str) -> Dict:
        """
        Full prediction for one receipt using both models and ledger.
        
        Args:
            box_file_path: Path to box file
            
        Returns:
            Comprehensive prediction dictionary
        """
        # Step 1: Extract fields via Model 1
        extracted_fields = self.extract_fields(box_file_path)
        
        # Step 2: Fingerprint check via ReceiptLedger
        fingerprint = self._create_fingerprint(extracted_fields)
        is_new_receipt = not self.ledger.has_fingerprint(fingerprint)
        
        # Step 3: Siamese similarity check if previous receipts exist
        similarity_score = 0.0
        most_similar_receipt = None
        
        if not is_new_receipt:
            # Find most similar stored receipt by fingerprint proximity
            most_similar_receipt = self._find_most_similar_receipt(extracted_fields)
            
            if most_similar_receipt:
                # Run Model 2 to get cosine similarity score
                similarity_score = self._compute_similarity(box_file_path, most_similar_receipt['box_file_path'])
        
        # Step 4: Apply rule engine for verdict
        verdict, confidence = self._apply_rule_engine(
            similarity_score, fingerprint, is_new_receipt
        )
        
        # Step 5: Update ledger
        if verdict == "LEGITIMATE":
            self.ledger.add_receipt(
                fingerprint=fingerprint,
                box_file_path=box_file_path,
                extracted_fields=extracted_fields
            )
        elif verdict == "FRAUD":
            self.ledger.increment_fraud_count(fingerprint)
        
        # Save ledger
        self.ledger.save()
        
        return {
            'company': extracted_fields.get('company', ''),
            'date': extracted_fields.get('date', ''),
            'address': extracted_fields.get('address', ''),
            'total': extracted_fields.get('total', ''),
            'verdict': verdict,
            'confidence': confidence,
            'similarity_score': similarity_score,
            'fingerprint': fingerprint,
            'is_new_receipt': is_new_receipt
        }
    
    def predict_from_image(self, image_path: str) -> Dict:
        """
        Run prediction on an image file (OCR first, then prediction).
        
        Args:
            image_path: Path to receipt image
            
        Returns:
            Prediction dictionary
        """
        # Create temporary box file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            temp_box_path = tmp_file.name
        
        try:
            # Run OCR
            print(f"Running OCR on {image_path}...")
            ocr_result = run_ocr(image_path, temp_box_path)
            
            if not ocr_result['success']:
                raise Exception(f"OCR failed: {ocr_result.get('error', 'Unknown error')}")
            
            # Run prediction on generated box file
            prediction = self.predict(temp_box_path)
            
            return prediction
            
        finally:
            # Clean up temporary file
            if Path(temp_box_path).exists():
                Path(temp_box_path).unlink()
    
    def batch_predict(self, box_file_paths: List[str]) -> List[Dict]:
        """
        Run prediction on multiple receipts.
        
        Args:
            box_file_paths: List of box file paths
            
        Returns:
            List of prediction dictionaries
        """
        print(f"Running batch prediction on {len(box_file_paths)} receipts...")
        
        predictions = []
        fraud_count = 0
        suspicious_count = 0
        
        for i, box_file_path in enumerate(box_file_paths):
            print(f"Processing receipt {i+1}/{len(box_file_paths)}: {box_file_path}")
            
            prediction = self.predict(box_file_path)
            predictions.append(prediction)
            
            # Count verdicts
            if prediction['verdict'] == 'FRAUD':
                fraud_count += 1
            elif prediction['verdict'] == 'SUSPICIOUS':
                suspicious_count += 1
        
        # Print summary
        print(f"\n{'─' * 50}")
        print("BATCH PREDICTION SUMMARY")
        print(f"{'─' * 50}")
        print(f"Total receipts: {len(box_file_paths)}")
        print(f"Fraud detected: {fraud_count}")
        print(f"Suspicious: {suspicious_count}")
        print(f"Legitimate: {len(box_file_paths) - fraud_count - suspicious_count}")
        print(f"{'─' * 50}")
        
        return predictions
    
    def _create_fingerprint(self, extracted_fields: Dict) -> str:
        """Create SHA-256 fingerprint from extracted fields."""
        # Create normalized string from key fields
        fingerprint_data = f"{extracted_fields.get('company', '')}|{extracted_fields.get('date', '')}|{extracted_fields.get('total', '')}"
        fingerprint_data = fingerprint_data.lower().strip()
        
        # Create SHA-256 hash
        fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()
        return fingerprint
    
    def _find_most_similar_receipt(self, extracted_fields: Dict) -> Optional[Dict]:
        """Find most similar stored receipt by fingerprint proximity."""
        current_fingerprint = self._create_fingerprint(extracted_fields)
        current_company = extracted_fields.get('company', '').lower().strip()
        current_date = extracted_fields.get('date', '').strip()
        
        most_similar = None
        best_score = 0
        
        for receipt in self.ledger.receipts.values():
            # Check for company or date proximity
            stored_company = receipt.get('extracted_fields', {}).get('company', '').lower().strip()
            stored_date = receipt.get('extracted_fields', {}).get('date', '').strip()
            
            score = 0
            if current_company and stored_company and current_company == stored_company:
                score += 0.5
            
            if current_date and stored_date and current_date == stored_date:
                score += 0.5
            
            if score > best_score:
                best_score = score
                most_similar = receipt
        
        return most_similar if best_score > 0 else None
    
    def _compute_similarity(self, current_box_path: str, stored_box_path: str) -> float:
        """Compute cosine similarity between two receipts using Model 2."""
        # Load and tokenize both receipts
        with open(current_box_path, 'r') as f:
            current_data = json.load(f)
        
        with open(stored_box_path, 'r') as f:
            stored_data = json.load(f)
        
        # Convert to model input format
        current_receipt = self._prepare_receipt_for_model(current_data)
        stored_receipt = self._prepare_receipt_for_model(stored_data)
        
        # Move to device
        current_receipt = {k: v.to(self.device) for k, v in current_receipt.items()}
        stored_receipt = {k: v.to(self.device) for k, v in stored_receipt.items()}
        
        # Compute similarity
        with torch.no_grad():
            similarity = self.model2.get_similarity(current_receipt, stored_receipt)
            similarity_score = similarity.item()
        
        return similarity_score
    
    def _prepare_receipt_for_model(self, box_data: Dict) -> Dict:
        """Prepare box data for Model 2 input."""
        # Extract tokens and bboxes
        tokens = box_data.get('tokens', [])
        bboxes = box_data.get('bboxes', [])
        
        # Get text from tokens
        texts = [t.get('text', '') for t in tokens]
        
        # Tokenize
        encoding = self.tokenizer(
            texts,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Pad bboxes to max_length
        padded_bboxes = bboxes + [[0, 0, 0, 0]] * (512 - len(bboxes))
        padded_bboxes = padded_bboxes[:512]
        
        encoding['bbox'] = torch.tensor([padded_bboxes], dtype=torch.long)
        
        # Squeeze batch dimension
        return {k: v.squeeze(0) for k, v in encoding.items()}
    
    def _apply_rule_engine(self, similarity_score: float, fingerprint: str, is_new_receipt: bool) -> tuple:
        """Apply rule engine to determine verdict and confidence."""
        fingerprint_match = not is_new_receipt
        
        if similarity_score > self.similarity_threshold and fingerprint_match:
            verdict = "FRAUD"
            confidence = similarity_score
        elif similarity_score > self.similarity_threshold and not fingerprint_match:
            verdict = "SUSPICIOUS"
            confidence = similarity_score
        else:
            verdict = "LEGITIMATE"
            confidence = 1.0 - similarity_score
        
        return verdict, confidence


def print_verdict_table(prediction: Dict):
    """Print verdict in clean table format."""
    verdict_emoji = {
        "FRAUD": "🚨",
        "SUSPICIOUS": "⚠️",
        "LEGITIMATE": "✅"
    }
    
    emoji = verdict_emoji.get(prediction['verdict'], "❓")
    
    print("┌" + "─" * 32 + "┐")
    print("│" + " " * 8 + "ReceiptGuard Verdict" + " " * 8 + "│")
    print("├" + "─" * 14 + "┼" + "─" * 17 + "┤")
    print(f"│  Company     │ {prediction['company']:<15} │")
    print(f"│  Date        │ {prediction['date']:<15} │")
    print(f"│  Total       │ {prediction['total']:<15} │")
    print(f"│  Verdict     │ {emoji} {prediction['verdict']:<12} │")
    print(f"│  Confidence  │ {prediction['confidence']:<15.2f} │")
    print("└" + "─" * 14 + "┴" + "─" * 17 + "┘")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ReceiptGuard fraud detection predictor")
    
    parser.add_argument(
        "--image",
        required=True,
        help="Path to receipt image"
    )
    
    parser.add_argument(
        "--model1",
        required=True,
        help="Path to Model 1 (NER) checkpoint"
    )
    
    parser.add_argument(
        "--model2",
        required=True,
        help="Path to Model 2 (Siamese) checkpoint"
    )
    
    parser.add_argument(
        "--model-path",
        default="dataset/raw/SROIE2019/layoutlm-base-uncased",
        help="Path to base LayoutLM model"
    )
    
    parser.add_argument(
        "--ledger-path",
        default="artifacts/ledger.json",
        help="Path to receipt ledger"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cuda, cpu, etc.)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = ReceiptGuardPredictor(
            model1_checkpoint=args.model1,
            model2_checkpoint=args.model2,
            model_path=args.model_path,
            ledger_path=args.ledger_path,
            device=args.device
        )
        
        # Run prediction
        print(f"\nAnalyzing receipt: {args.image}")
        prediction = predictor.predict_from_image(args.image)
        
        # Print verdict
        print_verdict_table(prediction)
        
        return 0
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
