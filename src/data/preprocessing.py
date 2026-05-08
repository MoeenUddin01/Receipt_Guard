import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# Import configuration
from src.config import CFG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_box_file(path: str) -> List[Dict]:
    """
    Reads a .txt box file line by line and extracts text with bounding box coordinates.
    
    Each line format: x1,y1,x2,y2,x3,y3,x4,y4,text
    
    Args:
        path: Path to the box file
        
    Returns:
        List of dicts with keys: text, x1, y1, x2, y2, x3, y3, x4, y4, bbox_normalized
    """
    tokens = []
    
    try:
        # Try UTF-8 first, fall back to latin-1 if it fails
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {path}, falling back to latin-1")
            with open(path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(',', 8)
            if len(parts) < 9:
                logger.warning(f"Malformed line {line_num} in {path}: {line}")
                continue
                
            try:
                x1, y1, x2, y2, x3, y3, x4, y4, text = parts
                coords = [int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)]
                
                # Calculate normalized bbox [x_min, y_min, x_max, y_max]
                x_coords = [int(x1), int(x2), int(x3), int(x4)]
                y_coords = [int(y1), int(y2), int(y3), int(y4)]
                bbox_normalized = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                
                tokens.append({
                    'text': text.strip(),
                    'x1': int(x1), 'y1': int(y1),
                    'x2': int(x2), 'y2': int(y2),
                    'x3': int(x3), 'y3': int(y3),
                    'x4': int(x4), 'y4': int(y4),
                    'bbox_normalized': bbox_normalized
                })
                
            except ValueError as e:
                logger.warning(f"Invalid coordinates in line {line_num} in {path}: {e}")
                continue
                    
    except FileNotFoundError:
        logger.error(f"Box file not found: {path}")
        return []
    except Exception as e:
        logger.error(f"Error reading box file {path}: {e}")
        return []
        
    return tokens


def parse_entity_file(path: str) -> Dict:
    """
    Reads the JSON entity file and normalizes date and total values.
    
    Args:
        path: Path to the entity file
        
    Returns:
        Dict with keys: company, date, address, total
    """
    entities = {'company': '', 'date': '', 'address': '', 'total': ''}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract entities with fallback to empty string
        entities['company'] = data.get('company', '').strip()
        entities['address'] = data.get('address', '').strip()
        
        # Normalize date to DD/MM/YYYY format
        raw_date = data.get('date', '').strip()
        entities['date'] = normalize_date(raw_date)
        
        # Normalize total to float with 2 decimal places
        raw_total = data.get('total', '').strip()
        entities['total'] = normalize_total(raw_total)
        
    except FileNotFoundError:
        logger.error(f"Entity file not found: {path}")
        return entities
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in entity file {path}: {e}")
        return entities
    except Exception as e:
        logger.error(f"Error reading entity file {path}: {e}")
        return entities
        
    return entities


def normalize_date(date_str: str) -> str:
    """Normalize various date formats to DD/MM/YYYY."""
    if not date_str:
        return ''
    
    # Common date patterns
    patterns = [
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # DD/MM/YYYY or MM/DD/YYYY
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD
        r'(\d{1,2})\s+(\w+)\s+(\d{2,4})',       # DD Month YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 3:
                    # Try to parse as DD/MM/YYYY first
                    if len(groups[2]) == 2:
                        year = '20' + groups[2] if int(groups[2]) < 50 else '19' + groups[2]
                    else:
                        year = groups[2]
                    
                    # Assume DD/MM/YYYY format
                    day, month = int(groups[0]), int(groups[1])
                    if day <= 31 and month <= 12:
                        return f"{day:02d}/{month:02d}/{year}"
                    elif month <= 31 and day <= 12:  # Try MM/DD/YYYY
                        return f"{month:02d}/{day:02d}/{year}"
            except (ValueError, IndexError):
                continue
    
    # If no pattern matches, return original
    return date_str


def normalize_total(total_str: str) -> str:
    """Normalize total amount to float string with 2 decimal places."""
    if not total_str:
        return ''
    
    # Remove currency symbols and extract numeric part
    numeric_part = re.sub(r'[^\d.,]', '', total_str)
    
    # Handle different decimal separators
    if ',' in numeric_part and '.' in numeric_part:
        # If both exist, assume last one is decimal separator
        if numeric_part.rfind(',') > numeric_part.rfind('.'):
            numeric_part = numeric_part.replace('.', '').replace(',', '.')
        else:
            numeric_part = numeric_part.replace(',', '')
    elif ',' in numeric_part:
        # Check if comma is decimal separator (European format)
        parts = numeric_part.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            numeric_part = numeric_part.replace(',', '.')
        else:
            numeric_part = numeric_part.replace(',', '')
    
    try:
        total_float = float(numeric_part)
        return f"{total_float:.2f}"
    except ValueError:
        return ''


def assign_bio_labels(box_tokens: List[Dict], entities: Dict) -> List[str]:
    """
    Assigns BIO (Begin-Inside-Outside) NER labels to each token.
    
    Args:
        box_tokens: List of token dictionaries with 'text' key
        entities: Dict with entity values
        
    Returns:
        List of BIO labels for each token
    """
    labels = ['O'] * len(box_tokens)
    
    # Entity types and their corresponding values
    entity_types = {
        'COMPANY': entities.get('company', ''),
        'DATE': entities.get('date', ''),
        'ADDRESS': entities.get('address', ''),
        'TOTAL': entities.get('total', '')
    }
    
    for token_idx, token in enumerate(box_tokens):
        token_text = token['text'].lower().strip()
        
        # Check each entity type
        for entity_type, entity_value in entity_types.items():
            if not entity_value:
                continue
                
            entity_value_lower = entity_value.lower()
            
            # Check if token is part of this entity
            if token_text in entity_value_lower:
                # Find position of token in entity value
                start_pos = entity_value_lower.find(token_text)
                if start_pos != -1:
                    # Check if this is the beginning of the entity
                    is_beginning = (start_pos == 0 or 
                                  entity_value_lower[start_pos-1] in ' \t\n\r')
                    
                    if is_beginning:
                        labels[token_idx] = f'B-{entity_type}'
                    else:
                        labels[token_idx] = f'I-{entity_type}'
                    break
    
    return labels


def get_image_size(img_path: str) -> Tuple[int, int]:
    """
    Returns (width, height) using Pillow without loading full image into memory.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        logger.error(f"Error getting image size for {img_path}: {e}")
        return (0, 0)


def normalize_bbox_for_layoutlm(bbox: List[int], width: int, height: int) -> List[int]:
    """
    Convert pixel coordinates to LayoutLM's 0-1000 scale.
    
    Args:
        bbox: [x_min, y_min, x_max, y_max] in pixels
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Normalized bbox coordinates in 0-1000 range
    """
    if width == 0 or height == 0:
        return [0, 0, 0, 0]
    
    normalized = [
        int((bbox[0] / width) * 1000),
        int((bbox[1] / height) * 1000),
        int((bbox[2] / width) * 1000),
        int((bbox[3] / height) * 1000)
    ]
    
    # Clamp values to 0-1000 range
    return [max(0, min(1000, coord)) for coord in normalized]


def build_processed_sample(receipt_id: str, split: str, base_path: str) -> Optional[Dict]:
    """
    Orchestrates parsing for one receipt (box + entity + image size).
    
    Args:
        receipt_id: Receipt identifier
        split: Dataset split ('train' or 'test')
        base_path: Base path to the dataset
        
    Returns:
        Full dict with {id, split, tokens, bboxes, labels, entity_values} or None
    """
    base_dir = Path(base_path) / split
    
    # Check if all required files exist
    box_file = base_dir / 'box' / f'{receipt_id}.txt'
    entity_file = base_dir / 'entities' / f'{receipt_id}.txt'
    img_file = base_dir / 'img' / f'{receipt_id}.jpg'
    
    if not all(f.exists() for f in [box_file, entity_file, img_file]):
        missing = [f.name for f in [box_file, entity_file, img_file] if not f.exists()]
        logger.warning(f"Missing files for receipt {receipt_id}: {missing}")
        return None
    
    # Parse box file
    box_tokens = parse_box_file(str(box_file))
    if not box_tokens:
        logger.warning(f"No valid tokens found in box file for {receipt_id}")
        return None
    
    # Parse entity file
    entities = parse_entity_file(str(entity_file))
    
    # Get image size
    width, height = get_image_size(str(img_file))
    if width == 0 or height == 0:
        logger.warning(f"Invalid image size for {receipt_id}")
        return None
    
    # Assign BIO labels
    labels = assign_bio_labels(box_tokens, entities)
    
    # Normalize bounding boxes for LayoutLM
    bboxes = []
    for token in box_tokens:
        normalized_bbox = normalize_bbox_for_layoutlm(token['bbox_normalized'], width, height)
        bboxes.append(normalized_bbox)
    
    # Extract tokens text
    tokens = [token['text'] for token in box_tokens]
    
    return {
        'id': receipt_id,
        'split': split,
        'tokens': tokens,
        'bboxes': bboxes,
        'labels': labels,
        'entity_values': entities
    }


if __name__ == "__main__":
    import pprint
    
    # Test on first 3 train samples
    base_path = CFG.data.raw_data_path
    train_path = Path(base_path) / "train"
    
    if train_path.exists():
        # Get first 3 receipt IDs from box files
        box_files = list((train_path / "box").glob("*.txt"))[:3]
        receipt_ids = [f.stem for f in box_files]
        
        print("Testing preprocessing on first 3 train samples:")
        print("=" * 60)
        
        for receipt_id in receipt_ids:
            print(f"\nProcessing receipt: {receipt_id}")
            print("-" * 40)
            
            sample = build_processed_sample(receipt_id, "train", base_path)
            
            if sample:
                # Pretty print the sample
                pprint.pprint(sample, width=120, depth=None)
            else:
                print(f"Failed to process receipt {receipt_id}")
                
    else:
        print(f"Train directory not found: {train_path}")
        print("Please ensure the dataset is properly extracted.")