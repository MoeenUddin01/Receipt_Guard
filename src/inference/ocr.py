"""
OCR module for ReceiptGuard-ML.

Extracts words and bounding boxes from receipt images using pytesseract.
Outputs data in SROIE box file format for direct compatibility with
parse_box_file().

Features:
- Image resizing for optimal OCR performance
- Grayscale conversion
- Confidence filtering (default threshold: 60)
- Box format compatible with parse_box_file()
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Try to import pytesseract
try:
    import pytesseract
    from pytesseract import Output

    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIDENCE_THRESHOLD = 60
MAX_IMAGE_DIMENSION = 2000  # Resize if larger than this
TARGET_DPI = 300


def preprocess_image(image: Image.Image, max_dim: int = MAX_IMAGE_DIMENSION) -> Image.Image:
    """
    Preprocess image for optimal OCR performance.

    Steps:
    1. Convert to grayscale
    2. Resize if dimensions exceed max_dim (preserving aspect ratio)
    3. Enhance contrast

    Args:
        image: PIL Image
        max_dim: Maximum dimension (width or height)

    Returns:
        Preprocessed PIL Image
    """
    # Convert to grayscale
    if image.mode != "L":
        image = image.convert("L")
        logger.debug("Converted image to grayscale")

    # Check if resizing is needed
    width, height = image.size
    max_current = max(width, height)

    if max_current > max_dim:
        # Calculate scale factor
        scale = max_dim / max_current
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize using LANCZOS for best quality
        image = image.resize((new_width, new_height), Image.LANCZOS)
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")

    # Enhance contrast using autocontrast
    from PIL import ImageOps
    image = ImageOps.autocontrast(image, cutoff=1)

    return image


def calculate_bounding_box(
    left: int, top: int, width: int, height: int
) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Calculate 8-point bounding box from x,y,w,h rectangle.

    Args:
        left: Left coordinate
        top: Top coordinate
        width: Width
        height: Height

    Returns:
        Tuple of (x1, y1, x2, y2, x3, y3, x4, y4)
        Order: top-left, top-right, bottom-right, bottom-left
    """
    right = left + width
    bottom = top + height

    # x1,y1 = top-left, x2,y2 = top-right
    # x3,y3 = bottom-right, x4,y4 = bottom-left
    return (left, top, right, top, right, bottom, left, bottom)


def run_ocr(
    image_path: str,
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
    preprocess: bool = True,
) -> List[Dict]:
    """
    Run OCR on an image and return tokens in SROIE box file format.

    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score (0-100). Words below
            this threshold are filtered out. Default: 60
        preprocess: Whether to apply image preprocessing (grayscale, resize)

    Returns:
        List of token dictionaries with keys:
        - text: The recognized word
        - x1, y1, x2, y2, x3, y3, x4, y4: Bounding box coordinates
        - bbox_normalized: [x_min, y_min, x_max, y_max]
        - confidence: OCR confidence score

    Raises:
        RuntimeError: If pytesseract is not installed
        FileNotFoundError: If image file doesn't exist
    """
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError(
            "pytesseract is required but not installed. "
            "Install with: pip install pytesseract"
        )

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info(f"Running OCR on {image_path}")

    # Load image
    image = Image.open(image_path)
    original_size = image.size
    logger.debug(f"Original image size: {original_size}")

    # Preprocess if requested
    if preprocess:
        image = preprocess_image(image)
        processed_size = image.size
        if processed_size != original_size:
            logger.info(f"Processed image size: {processed_size}")

    # Run pytesseract with bounding box data
    ocr_data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config="--psm 6",  # Assume a single uniform block of text
    )

    tokens = []
    n_boxes = len(ocr_data["text"])

    for i in range(n_boxes):
        text = ocr_data["text"][i].strip()
        conf = int(ocr_data["conf"][i])

        # Skip empty text
        if not text:
            continue

        # Filter by confidence threshold
        if conf < confidence_threshold:
            logger.debug(f"Skipping low confidence word '{text}' (conf={conf})")
            continue

        # Get bounding box
        left = ocr_data["left"][i]
        top = ocr_data["top"][i]
        width = ocr_data["width"][i]
        height = ocr_data["height"][i]

        # Calculate 8-point bounding box
        x1, y1, x2, y2, x3, y3, x4, y4 = calculate_bounding_box(
            left, top, width, height
        )

        # Calculate normalized bbox [x_min, y_min, x_max, y_max]
        bbox_normalized = [left, top, left + width, top + height]

        token = {
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
        }

        tokens.append(token)

    logger.info(
        f"OCR complete: extracted {len(tokens)} tokens "
        f"(filtered {n_boxes - len(tokens)} by confidence threshold {confidence_threshold})"
    )

    return tokens


def save_as_box_file(
    tokens: List[Dict],
    output_path: str,
    include_confidence: bool = False,
) -> None:
    """
    Save OCR tokens to a box file in SROIE format.

    Format: x1,y1,x2,y2,x3,y3,x4,y4,text

    Args:
        tokens: List of token dictionaries from run_ocr()
        output_path: Path to save the box file
        include_confidence: If True, append confidence as extra field
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for token in tokens:
        x1, y1, x2, y2, x3, y3, x4, y4 = (
            token["x1"],
            token["y1"],
            token["x2"],
            token["y2"],
            token["x3"],
            token["y3"],
            token["x4"],
            token["y4"],
        )
        text = token["text"]

        line = f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{text}"

        if include_confidence:
            conf = token.get("confidence", 0)
            line += f",{conf}"

        lines.append(line)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved box file to {output_path} ({len(tokens)} tokens)")


def extract_and_save(
    image_path: str,
    output_path: Optional[str] = None,
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
) -> List[Dict]:
    """
    Convenience function: run OCR and optionally save to box file.

    Args:
        image_path: Path to input image
        output_path: Optional path to save box file (auto-generated if None)
        confidence_threshold: Minimum confidence score

    Returns:
        List of token dictionaries
    """
    # Run OCR
    tokens = run_ocr(image_path, confidence_threshold=confidence_threshold)

    # Auto-generate output path if not provided
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.with_suffix(".box.txt")

    # Save to box file
    save_as_box_file(tokens, str(output_path))

    return tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ReceiptGuard-ML OCR Tool")
    parser.add_argument("image_path", help="Path to receipt image")
    parser.add_argument(
        "--output", "-o", help="Output box file path (default: <image>.box.txt)"
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=int,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (0-100, default: {DEFAULT_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable image preprocessing",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        tokens = extract_and_save(
            image_path=args.image_path,
            output_path=args.output,
            confidence_threshold=args.confidence,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("OCR Results Summary")
        print("=" * 60)
        print(f"Image: {args.image_path}")
        print(f"Tokens extracted: {len(tokens)}")
        if tokens:
            avg_conf = sum(t["confidence"] for t in tokens) / len(tokens)
            print(f"Average confidence: {avg_conf:.1f}")
            print("\nSample tokens (first 5):")
            for token in tokens[:5]:
                print(f"  '{token['text']}' (conf={token['confidence']}, "
                      f"bbox={token['bbox_normalized']})")
        print("=" * 60)

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise
