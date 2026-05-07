"""
Unit tests for preprocessing and fraud detection modules.
Uses synthetic data - no real dataset files required.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data"))

from preprocessing import (
    parse_box_file,
    parse_entity_file,
    assign_bio_labels,
    normalize_bbox_for_layoutlm,
)
from dataloader import build_receipt_fingerprint, ReceiptLedger


# =============================================================================
# Box File Parsing Tests
# =============================================================================

def test_parse_box_file_valid(tmp_path):
    """
    Test parsing of a valid box file with 3 lines.
    Each line format: x1,y1,x2,y2,x3,y3,x4,y4,text
    """
    # Create temp box file with 3 valid lines
    box_file = tmp_path / "test_box.txt"
    box_content = """100,200,300,200,300,400,100,400,Hello
50,60,150,60,150,120,50,120,World
10,20,100,20,100,50,10,50,Test"""
    box_file.write_text(box_content, encoding='utf-8')

    # Parse the file
    tokens = parse_box_file(str(box_file))

    # Assert correct number of tokens parsed
    assert len(tokens) == 3

    # Assert first token parsed correctly
    assert tokens[0]['text'] == "Hello"
    assert tokens[0]['x1'] == 100
    assert tokens[0]['y1'] == 200
    assert tokens[0]['x2'] == 300
    assert tokens[0]['y2'] == 200
    assert tokens[0]['x3'] == 300
    assert tokens[0]['y3'] == 400
    assert tokens[0]['x4'] == 100
    assert tokens[0]['y4'] == 400
    assert tokens[0]['bbox_normalized'] == [100, 200, 300, 400]

    # Assert second token
    assert tokens[1]['text'] == "World"
    assert tokens[1]['x1'] == 50
    assert tokens[1]['y1'] == 60
    assert tokens[1]['bbox_normalized'] == [50, 60, 150, 120]

    # Assert third token
    assert tokens[2]['text'] == "Test"
    assert tokens[2]['x1'] == 10
    assert tokens[2]['y1'] == 20
    assert tokens[2]['bbox_normalized'] == [10, 20, 100, 50]


def test_parse_box_file_malformed(tmp_path):
    """
    Test that malformed lines are skipped while valid lines still parse.
    """
    # Create temp box file with 1 malformed line and 2 valid lines
    box_file = tmp_path / "test_malformed.txt"
    box_content = """100,200,300,200,300,400,100,400,ValidLine1
malformed line without proper format
50,60,150,60,150,120,50,120,ValidLine2"""
    box_file.write_text(box_content, encoding='utf-8')

    # Parse the file
    tokens = parse_box_file(str(box_file))

    # Only 2 valid lines should be parsed
    assert len(tokens) == 2

    # First valid line
    assert tokens[0]['text'] == "ValidLine1"
    assert tokens[0]['x1'] == 100

    # Second valid line (after the malformed one)
    assert tokens[1]['text'] == "ValidLine2"
    assert tokens[1]['x1'] == 50


# =============================================================================
# Entity File Parsing Tests
# =============================================================================

def test_parse_entity_file(tmp_path):
    """
    Test parsing of JSON entity file with all 4 fields.
    """
    # Create temp entity JSON file
    entity_file = tmp_path / "entities.json"
    entity_data = {
        "company": "Acme Corp",
        "date": "15/03/2024",
        "address": "123 Main St, Cityville",
        "total": "150.50"
    }
    entity_file.write_text(json.dumps(entity_data), encoding='utf-8')

    # Parse the file
    entities = parse_entity_file(str(entity_file))

    # Assert all 4 fields parsed correctly
    assert entities['company'] == "Acme Corp"
    assert entities['date'] == "15/03/2024"  # Already normalized format
    assert entities['address'] == "123 Main St, Cityville"
    assert entities['total'] == "150.50"  # Normalized to 2 decimal places


# =============================================================================
# BBox Normalization Tests
# =============================================================================

def test_normalize_bbox_for_layoutlm():
    """
    Test that pixel coordinates are correctly scaled to 0-1000 range and clamped.
    """
    # Test case 1: Normal scaling
    # Image 1000x1000, bbox at 25% of width and height
    bbox = [250, 250, 750, 750]  # x_min, y_min, x_max, y_max
    width, height = 1000, 1000
    normalized = normalize_bbox_for_layoutlm(bbox, width, height)
    assert normalized == [250, 250, 750, 750]  # Should scale directly

    # Test case 2: Different image size
    # Image 500x500, same relative position
    bbox = [125, 125, 375, 375]
    width, height = 500, 500
    normalized = normalize_bbox_for_layoutlm(bbox, width, height)
    assert normalized == [250, 250, 750, 750]  # Should scale to same 0-1000 values

    # Test case 3: Clamping - values should be clamped to 0-1000
    # Image 100x100, bbox extends beyond
    bbox = [50, 50, 150, 150]  # x_max=150 is beyond image width=100
    width, height = 100, 100
    normalized = normalize_bbox_for_layoutlm(bbox, width, height)
    # x_min: 50/100 * 1000 = 500
    # y_min: 50/100 * 1000 = 500
    # x_max: 150/100 * 1000 = 1500 -> clamped to 1000
    # y_max: 150/100 * 1000 = 1500 -> clamped to 1000
    assert normalized == [500, 500, 1000, 1000]

    # Test case 4: Zero width/height returns zeros
    bbox = [10, 20, 50, 80]
    normalized = normalize_bbox_for_layoutlm(bbox, 0, 100)
    assert normalized == [0, 0, 0, 0]

    normalized = normalize_bbox_for_layoutlm(bbox, 100, 0)
    assert normalized == [0, 0, 0, 0]


# =============================================================================
# BIO Label Assignment Tests
# =============================================================================

def test_assign_bio_labels_single_token():
    """
    Test BIO label assignment for a simple case where company is one word.
    """
    # Simple tokens
    box_tokens = [
        {'text': 'Acme'},
        {'text': 'Corp'},
        {'text': 'Receipt'},
    ]

    # Entity dict with single-word company
    entities = {
        'company': 'Acme',
        'date': '',
        'address': '',
        'total': ''
    }

    labels = assign_bio_labels(box_tokens, entities)

    # Acme is the beginning of the company entity
    assert labels[0] == 'B-COMPANY'
    # Corp is not in the company entity (company is just 'Acme')
    assert labels[1] == 'O'
    assert labels[2] == 'O'


def test_assign_bio_labels_multi_token():
    """
    Test that multi-word entities get correct B-/I- sequence.
    """
    # Tokens for "Acme Corporation Inc"
    box_tokens = [
        {'text': 'Acme'},
        {'text': 'Corporation'},
        {'text': 'Inc'},
        {'text': 'Total'},
    ]

    # Multi-word company entity
    entities = {
        'company': 'Acme Corporation',
        'date': '',
        'address': '',
        'total': ''
    }

    labels = assign_bio_labels(box_tokens, entities)

    # Acme is the beginning
    assert labels[0] == 'B-COMPANY'
    # Corporation is inside the entity
    assert labels[1] == 'I-COMPANY'
    # Inc is not part of the company entity string
    assert labels[2] == 'O'
    # Total is outside
    assert labels[3] == 'O'


# =============================================================================
# Fingerprint / Duplicate Detection Tests
# =============================================================================

def test_build_receipt_fingerprint_normalization():
    """
    Test that two entity dicts with different whitespace/punctuation/capitalization
    but same semantic content produce identical fingerprints.
    """
    # First entity dict with extra whitespace, punctuation, and caps
    entity_dict1 = {
        'company': '  ACME, Corp!  ',
        'date': '15/03/2024',
        'total': '$150.50',
        'address': '123 Main St'
    }

    # Second entity dict - clean version
    entity_dict2 = {
        'company': 'acme corp',
        'date': '15-03-2024',
        'total': '150.50',
        'address': '456 Other St'  # Different address doesn't affect fingerprint
    }

    # Generate fingerprints
    fingerprint1 = build_receipt_fingerprint(entity_dict1)
    fingerprint2 = build_receipt_fingerprint(entity_dict2)

    # Should be identical despite formatting differences
    assert fingerprint1 == fingerprint2
    assert len(fingerprint1) == 64  # SHA-256 hex length

    # Different content should produce different fingerprint
    entity_dict3 = {
        'company': 'Different Corp',
        'date': '15/03/2024',
        'total': '$150.50',
        'address': '123 Main St'
    }
    fingerprint3 = build_receipt_fingerprint(entity_dict3)
    assert fingerprint3 != fingerprint1


def test_receipt_ledger_duplicate_detection(tmp_path):
    """
    Test that submitting the same receipt twice is correctly flagged as duplicate.
    """
    # Create ledger with temp path
    ledger_path = tmp_path / "test_ledger.json"
    ledger = ReceiptLedger(str(ledger_path))

    # First submission - new receipt
    entity_dict = {
        'company': 'Test Store',
        'date': '10/05/2024',
        'total': '99.99',
        'address': '123 Test Ave'
    }

    result1 = ledger.check_and_register('receipt_001', entity_dict)

    # Should not be duplicate
    assert result1['is_duplicate'] is False
    assert result1['fingerprint'] is not None
    assert len(result1['fingerprint']) == 64
    assert result1['existing_record'] is None

    # Second submission - same receipt (duplicate)
    result2 = ledger.check_and_register('receipt_002', entity_dict)

    # Should be flagged as duplicate
    assert result2['is_duplicate'] is True
    assert result2['fingerprint'] == result1['fingerprint']
    assert result2['existing_record'] is not None
    assert result2['existing_record']['receipt_id'] == 'receipt_001'

    # Third submission with same semantic content but different formatting
    entity_dict_formatted = {
        'company': '  TEST STORE  ',
        'date': '10-05-2024',
        'total': '$99.99',
        'address': '456 Other St'
    }
    result3 = ledger.check_and_register('receipt_003', entity_dict_formatted)

    # Should still be detected as duplicate (normalization)
    assert result3['is_duplicate'] is True
    assert result3['fingerprint'] == result1['fingerprint']
    assert result3['existing_record']['receipt_id'] == 'receipt_001'

    # Different receipt - not a duplicate
    different_entity = {
        'company': 'Different Store',
        'date': '11/05/2024',
        'total': '199.99',
        'address': '789 Other Ave'
    }
    result4 = ledger.check_and_register('receipt_004', different_entity)

    assert result4['is_duplicate'] is False
    assert result4['fingerprint'] != result1['fingerprint']
