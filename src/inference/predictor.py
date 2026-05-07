"""
ReceiptGuard-ML Inference Predictor.

Loads trained model, processes receipt images or box files,
extracts entities, and performs duplicate detection.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoTokenizer

# Add src paths
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from dataset import ID2LABEL
from dataloader import ReceiptLedger, build_receipt_fingerprint
from evaluation import extract_entities_from_predictions
from model import ModelConfig, ReceiptFieldExtractor, load_checkpoint
from preprocessing import normalize_bbox_for_layoutlm, parse_box_file
from train import get_device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import pytesseract, handle gracefully if not installed
try:
    import pytesseract
    from pytesseract import Output

    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. OCR functionality will not be available.")

# Default paths
DEFAULT_CHECKPOINT_DIR = Path("dataset/processed/checkpoints")
DEFAULT_LEDGER_PATH = Path("dataset/processed/inference_ledger.json")
DEFAULT_MODEL_PATH = "dataset/raw/SROIE2019/layoutlm-base-uncased"


def run_ocr_on_image(image_path: str) -> List[Dict]:
    """
    Run OCR on an image using pytesseract and return tokens with bounding boxes.

    Output format matches parse_box_file():
    [{'text': str, 'x1': int, 'y1': int, 'x2': int, 'y2': int, 'x3': int, 'y3': int,
      'x4': int, 'y4': int, 'bbox_normalized': [x_min, y_min, x_max, y_max]}, ...]

    Args:
        image_path: Path to the receipt image

    Returns:
        List of token dictionaries with OCR text and bounding boxes

    Raises:
        RuntimeError: If pytesseract is not installed
    """
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError(
            "pytesseract is required for OCR but not installed. "
            "Install with: pip install pytesseract"
        )

    logger.info(f"Running OCR on {image_path}")

    # Open image and get dimensions
    with Image.open(image_path) as img:
        width, height = img.size

    # Run OCR with bounding box data
    ocr_data = pytesseract.image_to_data(image_path, output_type=Output.DICT)

    tokens = []
    n_boxes = len(ocr_data["text"])

    for i in range(n_boxes):
        text = ocr_data["text"][i].strip()
        conf = int(ocr_data["conf"][i])

        # Skip empty text and low confidence boxes
        if not text or conf < 30:
            continue

        # Get bounding box (pytesseract returns left, top, width, height)
        left = ocr_data["left"][i]
        top = ocr_data["top"][i]
        box_width = ocr_data["width"][i]
        box_height = ocr_data["height"][i]

        # Calculate all 4 corners (axis-aligned, so simplified)
        x1, y1 = left, top
        x2, y2 = left + box_width, top
        x3, y3 = left + box_width, top + box_height
        x4, y4 = left, top + box_height

        # Normalized bbox for LayoutLM [x_min, y_min, x_max, y_max]
        bbox_normalized = [left, top, left + box_width, top + box_height]

        tokens.append({
            "text": text,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "x3": x3,
            "y3": y3,
            "x4": x4,
            "y4": y4,
            "bbox_normalized": bbox_normalized,
            "confidence": conf,
        })

    logger.info(f"OCR extracted {len(tokens)} tokens from image")
    return tokens


def load_model_for_inference(
    checkpoint_path: Optional[str] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    num_labels: int = 9,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file. If None, looks for best_model.pt in default dir
        model_path: Path to pretrained LayoutLM model
        num_labels: Number of NER labels
        device: Device to load model on (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer, device)
    """
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_DIR / "best_model.pt"
        if not checkpoint_path.exists():
            # Try to find any .pt file
            pt_files = list(DEFAULT_CHECKPOINT_DIR.glob("*.pt"))
            if pt_files:
                checkpoint_path = pt_files[0]
            else:
                raise FileNotFoundError(
                    f"No checkpoint found in {DEFAULT_CHECKPOINT_DIR}. "
                    "Please train a model first or specify checkpoint_path."
                )
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    # Auto-detect device
    if device is None:
        device = get_device()
    logger.info(f"Using device: {device}")

    # Build model
    config = ModelConfig(
        model_path=model_path,
        num_labels=num_labels,
        dropout=0.1,
    )
    model = ReceiptFieldExtractor(
        model_path=config.model_path,
        num_labels=config.num_labels,
        dropout=config.dropout,
    )

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device and set eval mode
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")

    return model, tokenizer, device


def tokenize_for_layoutlm(
    tokens: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize OCR tokens for LayoutLM input.

    Each word token is tokenized and paired with its bounding box.
    The bbox is repeated for all subword tokens.

    Args:
        tokens: List of token dicts from OCR or box file
        tokenizer: LayoutLM tokenizer
        max_length: Maximum sequence length

    Returns:
        Dictionary with input_ids, attention_mask, bbox tensors
    """
    # Collect words and their bboxes
    words = [t["text"] for t in tokens]
    bboxes = [t["bbox_normalized"] for t in tokens]

    # Tokenize with LayoutLM (handles word-level tokenization)
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Build bbox tensor aligned with tokens
    # Each word token gets its bbox, subword tokens inherit parent's bbox
    word_ids = encoding.word_ids()  # Maps token index to word index
    token_bboxes = []

    # Get image dimensions from normalized bboxes (approximate)
    max_x = max(b[2] for b in bboxes) if bboxes else 1000
    max_y = max(b[3] for b in bboxes) if bboxes else 1000

    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens [CLS], [SEP], [PAD] get default bbox
            token_bboxes.append([0, 0, 0, 0])
        else:
            # Get bbox for this word and normalize to 0-1000
            word_bbox = bboxes[word_idx]
            normalized = normalize_bbox_for_layoutlm(word_bbox, max_x, max_y)
            token_bboxes.append(normalized)

    # Convert to tensor [batch_size=1, seq_len, 4]
    bbox_tensor = torch.tensor([token_bboxes], dtype=torch.long)

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox": bbox_tensor,
    }


def run_inference(
    model: ReceiptFieldExtractor,
    tokenizer: AutoTokenizer,
    tokens: List[Dict],
    device: torch.device,
    max_length: int = 512,
) -> tuple:
    """
    Run model inference on tokenized input.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        tokens: List of token dictionaries
        device: Device
        max_length: Max sequence length

    Returns:
        Tuple of (predictions_list, extracted_entities_dict)
    """
    # Tokenize
    inputs = tokenize_for_layoutlm(tokens, tokenizer, max_length)

    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    bbox = inputs["bbox"].to(device)
    token_type_ids = torch.zeros_like(input_ids).to(device)

    # Run inference
    with torch.no_grad():
        _, logits = model(input_ids, attention_mask, token_type_ids, bbox)
        predictions = model.get_predictions(logits, attention_mask)

    # Get predictions for valid tokens (not padding)
    pred_ids = predictions[0].cpu().tolist()

    # Get words for each token
    words = [t["text"] for t in tokens]
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    # Extract word-level predictions
    word_ids = encoding.word_ids()
    word_predictions = {}

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx not in word_predictions:
            word_predictions[word_idx] = pred_ids[token_idx]

    # Build prediction list aligned with words
    word_pred_ids = [word_predictions.get(i, 0) for i in range(len(words))]

    # Extract entities
    extracted_entities = extract_entities_from_predictions(
        words, word_pred_ids, ID2LABEL
    )

    return word_pred_ids, extracted_entities


class ReceiptPredictor:
    """
    High-level predictor class for receipt inference.

    Handles model loading, input processing, and duplicate detection.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_path: str = DEFAULT_MODEL_PATH,
        ledger_path: str = str(DEFAULT_LEDGER_PATH),
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the predictor.

        Args:
            checkpoint_path: Path to model checkpoint
            model_path: Path to pretrained LayoutLM
            ledger_path: Path to fraud detection ledger
            device: Device to run on
        """
        self.model, self.tokenizer, self.device = load_model_for_inference(
            checkpoint_path=checkpoint_path,
            model_path=model_path,
            device=device,
        )
        self.ledger = ReceiptLedger(ledger_path)
        self.model_path = model_path

    def predict_from_box_file(self, box_file_path: str, receipt_id: Optional[str] = None) -> Dict:
        """
        Predict entities from a pre-made box file.

        Args:
            box_file_path: Path to box file
            receipt_id: Optional receipt ID (auto-generated if None)

        Returns:
            Result dict with entities and fraud detection info
        """
        # Parse box file
        tokens = parse_box_file(box_file_path)

        if not tokens:
            raise ValueError(f"No tokens found in box file: {box_file_path}")

        # Generate receipt ID from filename if not provided
        if receipt_id is None:
            receipt_id = Path(box_file_path).stem

        # Run inference
        _, entities = run_inference(
            self.model, self.tokenizer, tokens, self.device
        )

        # Check for duplicates
        return self._build_result(receipt_id, entities)

    def predict_from_image(self, image_path: str, receipt_id: Optional[str] = None) -> Dict:
        """
        Predict entities from a raw image (runs OCR first).

        Args:
            image_path: Path to receipt image
            receipt_id: Optional receipt ID (auto-generated if None)

        Returns:
            Result dict with entities and fraud detection info
        """
        # Run OCR
        tokens = run_ocr_on_image(image_path)

        if not tokens:
            raise ValueError(f"No text detected in image: {image_path}")

        # Generate receipt ID from filename if not provided
        if receipt_id is None:
            receipt_id = Path(image_path).stem

        # Run inference
        _, entities = run_inference(
            self.model, self.tokenizer, tokens, self.device
        )

        # Check for duplicates
        return self._build_result(receipt_id, entities)

    def predict(
        self,
        input_path: str,
        receipt_id: Optional[str] = None,
    ) -> Dict:
        """
        Auto-detect input type and predict.

        Args:
            input_path: Path to box file (.txt) or image (.jpg, .png, etc.)
            receipt_id: Optional receipt ID

        Returns:
            Result dict with entities and fraud detection info
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Determine input type by extension
        if input_path.suffix.lower() == ".txt":
            return self.predict_from_box_file(str(input_path), receipt_id)
        else:
            # Assume image file
            return self.predict_from_image(str(input_path), receipt_id)

    def _build_result(self, receipt_id: str, entities: Dict) -> Dict:
        """
        Build final result dict with fraud detection.

        Args:
            receipt_id: Receipt identifier
            entities: Extracted entity dict

        Returns:
            Complete result dictionary
        """
        # Check ledger for duplicates
        ledger_result = self.ledger.check_and_register(receipt_id, entities)

        # Build fraud message
        if ledger_result["is_duplicate"]:
            existing = ledger_result["existing_record"]
            fraud_message = (
                f"DUPLICATE DETECTED: This receipt matches '{existing['receipt_id']}' "
                f"(submitted {existing['submission_count']} times). "
                f"Potential fraudulent submission."
            )
        else:
            fraud_message = "Receipt appears to be new (no duplicate detected)."

        # Build result
        result = {
            "company": entities.get("company", ""),
            "date": entities.get("date", ""),
            "address": entities.get("address", ""),
            "total": entities.get("total", ""),
            "is_duplicate": ledger_result["is_duplicate"],
            "fraud_message": fraud_message,
            "receipt_id": receipt_id,
            "fingerprint": ledger_result["fingerprint"],
        }

        return result

    def save_ledger(self) -> None:
        """Save the receipt ledger to disk."""
        self.ledger.save()


def predict_receipt(
    input_path: str,
    checkpoint_path: Optional[str] = None,
    receipt_id: Optional[str] = None,
    ledger_path: str = str(DEFAULT_LEDGER_PATH),
) -> Dict:
    """
    Convenience function for one-off predictions.

    Args:
        input_path: Path to box file or image
        checkpoint_path: Path to model checkpoint
        receipt_id: Optional receipt ID
        ledger_path: Path to ledger file

    Returns:
        Result dict with entities and fraud detection info
    """
    predictor = ReceiptPredictor(
        checkpoint_path=checkpoint_path,
        ledger_path=ledger_path,
    )

    try:
        result = predictor.predict(input_path, receipt_id)
        predictor.save_ledger()
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="ReceiptGuard-ML Predictor")
    parser.add_argument("input_path", help="Path to box file (.txt) or image")
    parser.add_argument("--checkpoint", help="Path to checkpoint (optional)")
    parser.add_argument("--receipt_id", help="Receipt ID (optional)")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), help="Ledger path")
    parser.add_argument("--save", action="store_true", help="Save result to JSON")

    args = parser.parse_args()

    # Run prediction
    result = predict_receipt(
        input_path=args.input_path,
        checkpoint_path=args.checkpoint,
        receipt_id=args.receipt_id,
        ledger_path=args.ledger,
    )

    # Print result
    print("\n" + "=" * 60)
    print("Prediction Result")
    print("=" * 60)
    print(f"Receipt ID: {result['receipt_id']}")
    print(f"Company:    {result['company']}")
    print(f"Date:       {result['date']}")
    print(f"Address:    {result['address']}")
    print(f"Total:      {result['total']}")
    print(f"Duplicate:  {'YES' if result['is_duplicate'] else 'NO'}")
    print(f"Message:    {result['fraud_message']}")
    print("=" * 60)

    # Save to JSON if requested
    if args.save:
        import json

        output_path = Path(args.input_path).with_suffix(".prediction.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Result saved to: {output_path}")
