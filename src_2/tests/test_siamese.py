"""
test_siamese.py - Unit tests for Siamese model components.

All tests are self-contained with no real dataset dependencies.
"""

import re
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from ..data.fraud_dataset import (SROIEPairDataset, tamper_company,
                                 tamper_date, tamper_total)
from ..model.evaluation import compute_fraud_metrics, find_best_threshold
from ..model.siamese_model import SiameseSimilarityModel


class TestTamperFunctions:
    """Test tamper functions for fraud generation."""
    
    def test_tamper_date_changes_date(self):
        """Test that tamper_date always changes the date and maintains format."""
        original_date = "25/12/2018"
        changed_dates = []
        
        # Call tamper_date 10 times
        for _ in range(10):
            changed_date = tamper_date(original_date)
            changed_dates.append(changed_date)
            
            # Assert output is never identical to input
            assert changed_date != original_date, f"Date should not be identical: {changed_date}"
            
            # Assert output is always valid DD/MM/YYYY format
            date_pattern = r'^\d{2}/\d{2}/\d{4}$'
            assert re.match(date_pattern, changed_date), f"Invalid date format: {changed_date}"
        
        # Ensure we got different results (not all the same)
        unique_dates = set(changed_dates)
        assert len(unique_dates) > 1, "Should generate different dates"
    
    def test_tamper_total_stays_positive(self):
        """Test that tamper_total never goes negative."""
        original_total = "1.00"
        
        # Call tamper_total 20 times
        for _ in range(20):
            changed_total = tamper_total(original_total)
            
            # Assert output is always >= 0.00
            total_float = float(changed_total)
            assert total_float >= 0.00, f"Total should not be negative: {changed_total}"
            
            # Assert output maintains 2 decimal places
            assert '.' in changed_total, f"Total should have decimal: {changed_total}"
            assert len(changed_total.split('.')[1]) == 2, f"Should have 2 decimals: {changed_total}"
    
    def test_tamper_company_changes_one_char(self):
        """Test that tamper_company changes exactly one character."""
        original_company = "BOOK TA .K SDN BHD"
        changed_company = tamper_company(original_company)
        
        # Assert result differs from original
        assert changed_company != original_company, "Company should be changed"
        
        # Calculate character differences
        original_chars = list(original_company)
        changed_chars = list(changed_company)
        
        # Handle length differences (insert/delete operations)
        if len(original_chars) == len(changed_chars):
            # Count character differences (swap operation)
            diff_count = sum(1 for a, b in zip(original_chars, changed_chars) if a != b)
            assert diff_count == 1, f"Should differ by exactly 1 character, got {diff_count}"
        else:
            # For insert/delete, the length difference should be exactly 1
            length_diff = abs(len(original_chars) - len(changed_company))
            assert length_diff == 1, f"Length should differ by exactly 1, got {length_diff}"


class TestSROIEPairDataset:
    """Test SROIEPairDataset functionality."""
    
    def test_pair_dataset_label_balance(self):
        """Test that dataset maintains approximate label balance."""
        # Create 20 synthetic processed samples
        samples = []
        for i in range(20):
            sample = {
                'id': f'receipt_{i}',
                'tokens': [{'text': f'token_{j}'} for j in range(10)],
                'bboxes': [[j, j, j+10, j+10] for j in range(10)],
                'entities': [
                    {'label': 'COMPANY', 'text': f'Company {i}'},
                    {'label': 'DATE', 'text': f'{i+1}/01/2024'},
                    {'label': 'TOTAL', 'text': f'{i+1}.00'}
                ]
            }
            samples.append(sample)
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.zeros(512, dtype=torch.long),
            'attention_mask': torch.zeros(512, dtype=torch.long),
            'token_type_ids': torch.zeros(512, dtype=torch.long)
        }
        
        # Create dataset with fraud_ratio=0.5
        dataset = SROIEPairDataset(
            samples=samples,
            tokenizer=mock_tokenizer,
            max_length=512,
            fraud_ratio=0.5
        )
        
        # Count labels
        fraud_count = sum(1 for p in dataset.pairs if p['label'] == 1)
        legit_count = sum(1 for p in dataset.pairs if p['label'] == 0)
        total_pairs = len(dataset.pairs)
        
        # Assert balance within 10%
        fraud_ratio = fraud_count / total_pairs
        expected_ratio = 0.5
        tolerance = 0.1
        
        assert abs(fraud_ratio - expected_ratio) <= tolerance, \
            f"Fraud ratio {fraud_ratio:.3f} not within {tolerance} of {expected_ratio}"
        
        # Assert we have both types
        assert fraud_count > 0, "Should have fraud pairs"
        assert legit_count > 0, "Should have legitimate pairs"


class TestSiameseModel:
    """Test SiameseSimilarityModel functionality."""
    
    @patch('src_2.model.siamese_model.LayoutLMModel')
    def test_siamese_model_forward_shape(self, mock_layoutlm):
        """Test that model forward returns correct tensor shapes."""
        # Mock LayoutLM model
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.return_value = Mock()
        mock_model.return_value.last_hidden_state = torch.randn(2, 512, 768)  # [batch, seq, hidden]
        mock_layoutlm.from_pretrained.return_value = mock_model
        
        # Build model
        model = SiameseSimilarityModel(
            model_path="dummy_path",
            dropout=0.1,
            projection_dim=256
        )
        
        # Create random receipt tensor dicts
        batch_size = 2
        seq_len = 512
        receipt_a = {
            'input_ids': torch.randint(0, 30000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'token_type_ids': torch.zeros(batch_size, seq_len, dtype=torch.long),
            'bbox': torch.randint(0, 1000, (batch_size, seq_len, 4))
        }
        receipt_b = {
            'input_ids': torch.randint(0, 30000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'token_type_ids': torch.zeros(batch_size, seq_len, dtype=torch.long),
            'bbox': torch.randint(0, 1000, (batch_size, seq_len, 4))
        }
        labels = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        loss, logits, similarity = model(receipt_a, receipt_b, labels)
        
        # Assert shapes
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert logits.shape == (batch_size, 2), f"Logits shape should be ({batch_size}, 2), got {logits.shape}"
        assert similarity.shape == (batch_size,), f"Similarity shape should be ({batch_size},), got {similarity.shape}"


class TestEvaluation:
    """Test evaluation utilities."""
    
    def test_find_best_threshold(self):
        """Test threshold optimization on synthetic data."""
        # Create synthetic data where fraud has high similarity, legit has low
        np = pytest.importorskip("numpy")
        
        np.random.seed(42)
        fraud_similarities = np.random.normal(0.9, 0.05, 50).clip(0, 1)
        legit_similarities = np.random.normal(0.3, 0.1, 50).clip(0, 1)
        
        similarities = np.concatenate([legit_similarities, fraud_similarities]).tolist()
        labels = [0] * 50 + [1] * 50  # 0=legit, 1=fraud
        
        # Find best threshold
        best_threshold = find_best_threshold(similarities, labels)
        
        # Assert threshold is reasonable (between 0.5 and 0.9)
        assert 0.5 <= best_threshold <= 0.9, \
            f"Threshold {best_threshold:.3f} should be between 0.5 and 0.9"
    
    def test_compute_fraud_metrics_perfect(self):
        """Test metrics calculation with perfect predictions."""
        # Create perfect predictions (all fraud correctly predicted)
        similarities = [0.9, 0.8, 0.95, 0.85]  # All > threshold
        labels = [1, 1, 1, 1]  # All fraud
        threshold = 0.5
        
        metrics = compute_fraud_metrics(similarities, labels, threshold)
        
        # Assert perfect metrics
        assert metrics['precision'] == 1.0, f"Precision should be 1.0, got {metrics['precision']}"
        assert metrics['recall'] == 1.0, f"Recall should be 1.0, got {metrics['recall']}"
        assert metrics['f1'] == 1.0, f"F1 should be 1.0, got {metrics['f1']}"
        assert metrics['accuracy'] == 1.0, f"Accuracy should be 1.0, got {metrics['accuracy']}"
        
        # Check confusion matrix components
        assert metrics['true_positives'] == 4, "Should have 4 true positives"
        assert metrics['false_positives'] == 0, "Should have 0 false positives"
        assert metrics['true_negatives'] == 0, "Should have 0 true negatives"
        assert metrics['false_negatives'] == 0, "Should have 0 false negatives"


class TestReceiptGuardPredictor:
    """Test ReceiptGuardPredictor functionality."""
    
    @patch('src_2.inference.fraud_predictor.ReceiptFieldExtractor')
    @patch('src_2.inference.fraud_predictor.SiameseSimilarityModel')
    @patch('src_2.inference.fraud_predictor.ReceiptLedger')
    @patch('src_2.inference.fraud_predictor.LayoutLMTokenizer')
    def test_receipt_guard_predictor_verdict(self, mock_tokenizer, mock_ledger, 
                                         mock_siamese_model, mock_field_extractor):
        """Test verdict logic with mocked components."""
        # Mock Model 1 to return fixed fields
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract_fields.return_value = {
            'company': 'BOOK TA .K SDN BHD',
            'date': '25/12/2018',
            'address': '123 Main St',
            'total': '9.00'
        }
        mock_field_extractor.load_from_checkpoint.return_value = mock_extractor_instance
        
        # Mock Model 2 to return similarity=0.95
        mock_siamese_instance = Mock()
        mock_siamese_instance.get_similarity.return_value = torch.tensor([0.95])
        mock_siamese_instance.eval.return_value = None
        mock_siamese_model.return_value = mock_siamese_instance
        
        # Mock ledger to return fingerprint match
        mock_ledger_instance = Mock()
        mock_ledger_instance.has_fingerprint.return_value = False  # New receipt
        mock_ledger_instance.receipts = {}  # Empty ledger
        mock_ledger.return_value = mock_ledger_instance
        
        # Mock load_siamese_checkpoint to return threshold
        with patch('src_2.inference.fraud_predictor.load_siamese_checkpoint') as mock_load:
            mock_load.return_value = {
                'model': mock_siamese_instance,
                'similarity_threshold': 0.8
            }
            
            # Import and create predictor
            from ..inference.fraud_predictor import ReceiptGuardPredictor
            
            predictor = ReceiptGuardPredictor(
                model1_checkpoint="dummy_model1.pt",
                model2_checkpoint="dummy_model2.pt",
                model_path="dummy_model_path",
                ledger_path="dummy_ledger.json",
                device="cpu"
            )
            
            # Mock the predict method internals for testing
            with patch.object(predictor, '_compute_similarity', return_value=0.95):
                # Create dummy box file data
                dummy_box_data = {
                    'tokens': [{'text': 'test'}],
                    'bboxes': [[0, 0, 10, 10]]
                }
                
                with patch('builtins.open', mock_open_read(dummy_box_data)):
                    with patch.object(predictor, 'extract_fields', return_value={
                        'company': 'BOOK TA .K SDN BHD',
                        'date': '25/12/2018',
                        'address': '123 Main St',
                        'total': '9.00'
                    }):
                        # Mock ledger to have existing receipt for fingerprint match
                        predictor.ledger.has_fingerprint.return_value = True
                        predictor.ledger.receipts = {
                            'dummy_fingerprint': {
                                'box_file_path': 'dummy_path.json',
                                'extracted_fields': {
                                    'company': 'BOOK TA .K SDN BHD',
                                    'date': '25/12/2018'
                                }
                            }
                        }
                        
                        result = predictor.predict('dummy_box.json')
                        
                        # With high similarity (0.95) > threshold (0.8) and fingerprint match
                        # Should be FRAUD
                        assert result['verdict'] == 'FRAUD', \
                            f"Expected FRAUD, got {result['verdict']}"
                        assert result['confidence'] == 0.95, \
                            f"Expected confidence 0.95, got {result['confidence']}"


def mock_open_read(data):
    """Helper to mock open() for reading."""
    import json
    from unittest.mock import mock_open
    
    return mock_open(read_data=json.dumps(data))
