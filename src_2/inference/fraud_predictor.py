"""
fraud_predictor.py - Combined inference engine for fraud detection.

Combines Model 1 (NER), Model 2 (Siamese), and ReceiptLedger
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

from src_2.model.siamese_model import SiameseSimilarityModel, load_siamese_checkpoint
from src_2.model.train import detect_device
import importlib
_ocr_mod = importlib.import_module("src.inference.ocr")
run_ocr = _ocr_mod.run_ocr
from src.model.model import ReceiptFieldExtractor
from src.data.dataloader import ReceiptLedger, build_receipt_fingerprint
from src.data.preprocessing import normalize_bbox_for_layoutlm
from src.model.evaluation import extract_entities_from_predictions, ID2LABEL


class ReceiptGuardPredictor:
    """
    Combined inference engine using both models + ledger for fraud detection.
    """

    def __init__(
        self,
        model1_checkpoint: str,
        model2_checkpoint: str,
        model_path: str = "dataset/raw/SROIE2019/layoutlm-base-uncased",
        ledger_path: str = "artifacts/ledger.json",
        device: str = "auto",
    ):
        if device == "auto":
            self.device = detect_device()
        else:
            self.device = torch.device(device)

        print(f"Initializing ReceiptGuardPredictor on {self.device}")

        # Load Model 1 (NER)
        print(f"Loading Model 1 (NER) from {model1_checkpoint}...")
        self.model1 = ReceiptFieldExtractor.load_from_checkpoint(
            model1_checkpoint, model_path=model_path, num_labels=9
        )
        self.model1.to(self.device)
        self.model1.eval()
        print(f"Model 1 loaded")

        # Load Model 2 (Siamese)
        print(f"Loading Model 2 (Siamese) from {model2_checkpoint}...")
        checkpoint_info = load_siamese_checkpoint(model2_checkpoint)
        self.model2 = checkpoint_info["model"]
        self.model2.to(self.device)
        self.model2.eval()
        self.similarity_threshold = checkpoint_info["similarity_threshold"]
        print(f"Model 2 loaded with threshold: {self.similarity_threshold:.3f}")

        # Load tokenizer
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_path)
        print(f"Tokenizer loaded")

        # Load ledger
        print(f"Loading ledger from {ledger_path}...")
        self.ledger = ReceiptLedger(ledger_path)
        print(f"Ledger loaded with {len(self.ledger)} stored receipts")

        print("ReceiptGuardPredictor initialized successfully")

    def extract_fields(self, box_file_path: str) -> Dict:
        """
        Extract fields from a box file using Model 1.

        Args:
            box_file_path: Path to box file (JSON with tokens & bboxes)

        Returns:
            Dictionary with extracted fields {company, date, address, total}
        """
        with open(box_file_path, "r") as f:
            box_data = json.load(f)

        tokens = box_data.get("tokens", [])
        bboxes = box_data.get("bboxes", [])

        if not tokens:
            return {"company": "", "date": "", "address": "", "total": ""}

        words = [t.get("text", "") if isinstance(t, dict) else str(t) for t in tokens]

        # Compute image dimensions from max bbox values
        if bboxes:
            max_x = max(b[2] if len(b) > 2 else b[0] + 100 for b in bboxes)
            max_y = max(b[3] if len(b) > 3 else b[1] + 100 for b in bboxes)
        else:
            max_x, max_y = 1000, 1000

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids()
        token_bboxes = []
        for word_idx in word_ids:
            if word_idx is None:
                token_bboxes.append([0, 0, 0, 0])
            else:
                raw_bbox = bboxes[word_idx] if word_idx < len(bboxes) else [0, 0, 0, 0]
                token_bboxes.append(
                    normalize_bbox_for_layoutlm(raw_bbox, max_x, max_y)
                )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        bbox_tensor = (
            torch.tensor([token_bboxes], dtype=torch.long).to(self.device)
        )
        token_type_ids = torch.zeros_like(input_ids).to(self.device)

        with torch.no_grad():
            _, logits = self.model1(input_ids, attention_mask, token_type_ids, bbox_tensor)
            pred_ids = self.model1.get_predictions(logits, attention_mask)

        valid_positions = torch.where(attention_mask[0] == 1)[0].tolist()
        word_pred_map = {}
        for i, pos in enumerate(valid_positions):
            wid = word_ids[pos]
            if wid is not None and i < len(pred_ids[0]):
                word_pred_map[wid] = pred_ids[0][i]

        aligned_pred_ids = [word_pred_map.get(i, 0) for i in range(len(words))]

        entities = extract_entities_from_predictions(words, aligned_pred_ids, ID2LABEL)
        
        # If all fields are empty, it likely means model predicted all O labels
        # This is the expected behavior when Model 1 doesn't detect entities
        return entities

    def predict(self, box_file_path: str) -> Dict:
        """
        Full prediction for one receipt using both models and ledger.
        """
        # Step 1: Extract fields via Model 1
        extracted_fields = self.extract_fields(box_file_path)

        # Step 2: Register in ledger & check for duplicates
        receipt_id = Path(box_file_path).stem
        # Include box_file_path so ledger stores it for future Siamese comparisons
        ledger_entry_fields = {**extracted_fields, "box_file_path": box_file_path}
        ledger_result = self.ledger.check_and_register(receipt_id, ledger_entry_fields)
        fingerprint = ledger_result["fingerprint"]
        is_new_receipt = not ledger_result["is_duplicate"]

        # Step 3: Siamese similarity check if duplicate exists
        similarity_score = 0.0
        if not is_new_receipt:
            sim = self._find_and_compare_similar(box_file_path, extracted_fields)
            if sim is not None:
                similarity_score = sim

        # Step 4: Apply rule engine for verdict
        verdict, confidence = self._apply_rule_engine(
            similarity_score, fingerprint, is_new_receipt
        )

        # Step 5: Save ledger
        self.ledger.save()

        return {
            "company": extracted_fields.get("company", ""),
            "date": extracted_fields.get("date", ""),
            "address": extracted_fields.get("address", ""),
            "total": extracted_fields.get("total", ""),
            "verdict": verdict,
            "confidence": confidence,
            "similarity_score": similarity_score,
            "fingerprint": fingerprint,
            "is_new_receipt": is_new_receipt,
        }

    def predict_from_image(self, image_path: str) -> Dict:
        """
        Run prediction on an image file (OCR first, then prediction).

        Args:
            image_path: Path to receipt image

        Returns:
            Prediction dictionary
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            temp_box_path = tmp_file.name

        try:
            print(f"Running OCR on {image_path}...")
            ocr_tokens = run_ocr(image_path)

            # Save OCR output as box JSON
            tokens_list = []
            bboxes_list = []
            for tok in ocr_tokens:
                tokens_list.append({"text": tok["text"]})
                bboxes_list.append(tok["bbox_normalized"])

            box_data = {"tokens": tokens_list, "bboxes": bboxes_list}
            with open(temp_box_path, "w") as f:
                json.dump(box_data, f)

            prediction = self.predict(temp_box_path)
            return prediction

        finally:
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

            if prediction["verdict"] == "FRAUD":
                fraud_count += 1
            elif prediction["verdict"] == "SUSPICIOUS":
                suspicious_count += 1

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
        fingerprint_data = (
            f"{extracted_fields.get('company', '')}|"
            f"{extracted_fields.get('date', '')}|"
            f"{extracted_fields.get('total', '')}"
        )
        fingerprint_data = fingerprint_data.lower().strip()
        fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()
        return fingerprint

    def _find_and_compare_similar(
        self, current_box_path: str, extracted_fields: Dict
    ) -> Optional[float]:
        """Find most similar stored receipt and compute Siamese similarity."""
        current_company = extracted_fields.get("company", "").lower().strip()
        current_date = extracted_fields.get("date", "").strip()

        best_score = 0
        best_stored_path = None

        for receipt in self.ledger.ledger.values():
            stored_company = receipt.get("company", "").lower().strip()
            stored_date = receipt.get("date", "").strip()

            score = 0
            if current_company and stored_company and current_company == stored_company:
                score += 0.5
            if current_date and stored_date and current_date == stored_date:
                score += 0.5

            if score > best_score:
                best_score = score
                best_stored_path = receipt.get("box_file_path")

        if best_score == 0 or not best_stored_path or not Path(best_stored_path).exists():
            return None

        with open(current_box_path, "r") as f:
            current_data = json.load(f)
        with open(best_stored_path, "r") as f:
            stored_data = json.load(f)

        current_receipt = self._prepare_receipt_for_model(current_data)
        stored_receipt = self._prepare_receipt_for_model(stored_data)

        current_receipt = {k: v.to(self.device) for k, v in current_receipt.items()}
        stored_receipt = {k: v.to(self.device) for k, v in stored_receipt.items()}

        with torch.no_grad():
            similarity = self.model2.get_similarity(current_receipt, stored_receipt)
            return similarity.item()

    def _prepare_receipt_for_model(self, box_data: Dict) -> Dict:
        """Prepare box data for Model 2 input."""
        tokens = box_data.get("tokens", [])
        bboxes = box_data.get("bboxes", [])

        texts = [t.get("text", "") if isinstance(t, dict) else str(t) for t in tokens]

        encoding = self.tokenizer(
            texts,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        padded_bboxes = bboxes + [[0, 0, 0, 0]] * (512 - len(bboxes))
        padded_bboxes = padded_bboxes[:512]
        encoding["bbox"] = torch.tensor([padded_bboxes], dtype=torch.long)

        return {k: v.squeeze(0) for k, v in encoding.items()}

    def _apply_rule_engine(
        self, similarity_score: float, fingerprint: str, is_new_receipt: bool
    ) -> tuple:
        """Apply rule engine to determine verdict and confidence.
        
        Logic:
        - Fingerprint match (same company+date+total hash) → FRAUD (confirmed duplicate)
        - No fingerprint match, high Siamese similarity → SUSPICIOUS (visually similar)
        - No match, low similarity → LEGITIMATE
        """
        fingerprint_match = not is_new_receipt

        if fingerprint_match:
            verdict = "FRAUD"
            confidence = max(similarity_score, 0.85)
        elif similarity_score > self.similarity_threshold:
            verdict = "SUSPICIOUS"
            confidence = similarity_score
        else:
            verdict = "LEGITIMATE"
            confidence = 1.0 - similarity_score

        return verdict, confidence


def print_verdict_table(prediction: Dict):
    """Print verdict in clean table format."""
    verdict_emoji = {
        "FRAUD": "\U0001f6a8",
        "SUSPICIOUS": "\u26a0\ufe0f",
        "LEGITIMATE": "\u2705",
    }

    emoji = verdict_emoji.get(prediction["verdict"], "\u2753")

    # Helper function to display "not found" for empty fields
    def display_field(value: str) -> str:
        return value if value.strip() else "not found"

    print("\u250c" + "\u2500" * 32 + "\u2510")
    print(
        "\u2502" + " " * 8 + "ReceiptGuard Verdict" + " " * 8 + "\u2502"
    )
    print("\u251c" + "\u2500" * 14 + "\u253c" + "\u2500" * 17 + "\u2524")
    print(f"\u2502  Company     \u2502 {display_field(prediction['company']):<15} \u2502")
    print(f"\u2502  Date        \u2502 {display_field(prediction['date']):<15} \u2502")
    print(f"\u2502  Total       \u2502 {display_field(prediction['total']):<15} \u2502")
    print(
        f"\u2502  Verdict     \u2502 {emoji} {prediction['verdict']:<12} \u2502"
    )
    print(
        f"\u2502  Confidence  \u2502 {prediction['confidence']:<15.2f} \u2502"
    )
    print("\u2514" + "\u2500" * 14 + "\u2534" + "\u2500" * 17 + "\u2518")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ReceiptGuard fraud detection predictor"
    )

    parser.add_argument("--image", required=True, help="Path to receipt image")
    parser.add_argument(
        "--model1", required=True, help="Path to Model 1 (NER) checkpoint"
    )
    parser.add_argument(
        "--model2", required=True, help="Path to Model 2 (Siamese) checkpoint"
    )
    parser.add_argument(
        "--model-path",
        default="dataset/raw/SROIE2019/layoutlm-base-uncased",
        help="Path to base LayoutLM model",
    )
    parser.add_argument(
        "--ledger-path",
        default="artifacts/ledger.json",
        help="Path to receipt ledger",
    )
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto, cuda, cpu, etc.)"
    )

    args = parser.parse_args()

    try:
        predictor = ReceiptGuardPredictor(
            model1_checkpoint=args.model1,
            model2_checkpoint=args.model2,
            model_path=args.model_path,
            ledger_path=args.ledger_path,
            device=args.device,
        )

        print(f"\nAnalyzing receipt: {args.image}")
        prediction = predictor.predict_from_image(args.image)

        print_verdict_table(prediction)

        return 0

    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
