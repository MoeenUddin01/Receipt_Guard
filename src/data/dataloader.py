import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMTokenizer, LayoutLMTokenizerFast

# Import configuration
from src.config import CFG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PART A — COLLATE & BATCH UTILITIES
# ============================================================================

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader that stacks all tensor fields.
    
    Args:
        batch: List of sample dictionaries from dataset
        
    Returns:
        Dictionary with stacked tensors and receipt_id list
    """
    if not batch:
        return {}
    
    # Extract receipt_ids separately (they're strings, not tensors)
    receipt_ids = [item.get('receipt_id', f'unknown_{i}') for i, item in enumerate(batch)]
    
    # Stack all tensor fields
    collated = {}
    
    # Get all keys from the first item to determine tensor fields
    tensor_keys = [k for k in batch[0].keys() if isinstance(batch[0][k], torch.Tensor)]
    
    for key in tensor_keys:
        try:
            # Stack tensors along batch dimension
            collated[key] = torch.stack([item[key] for item in batch])
        except Exception as e:
            logger.error(f"Failed to stack tensor field '{key}': {e}")
            # Fallback: create a tensor with zeros
            collated[key] = torch.zeros(len(batch), *batch[0][key].shape)
    
    # Add receipt_ids as a separate field
    collated['receipt_ids'] = receipt_ids
    
    return collated


def get_tokenizer(model_path: str) -> Union[LayoutLMTokenizer, LayoutLMTokenizerFast]:
    """
    Loads LayoutLM tokenizer from the specified path.
    
    Args:
        model_path: Path to the LayoutLM model directory
        
    Returns:
        LayoutLM tokenizer instance
    """
    try:
        # Try to load fast tokenizer first
        tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
        logger.info(f"Loaded LayoutLM fast tokenizer from {model_path}")
    except Exception:
        try:
            # Fallback to regular tokenizer
            tokenizer = LayoutLMTokenizer.from_pretrained(model_path)
            logger.info(f"Loaded LayoutLM tokenizer from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {model_path}: {e}")
            # Fallback to HuggingFace hub
            logger.info(f"Falling back to {CFG.model.model_path} from HuggingFace hub")
            tokenizer = LayoutLMTokenizer.from_pretrained(CFG.model.model_path)
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logger.info("Added padding token to tokenizer")
    
    return tokenizer


# ============================================================================
# PART B — DUPLICATION ENGINE
# ============================================================================

def build_receipt_fingerprint(entity_dict: Dict) -> str:
    """
    Creates a canonical fingerprint for duplicate detection.
    
    This function implements the core fraud detection logic by normalizing
    receipt entities and creating a unique hash. Each normalization step
    is critical for catching variations in how the same receipt might be
    submitted multiple times.
    
    Args:
        entity_dict: Dictionary with keys 'company', 'date', 'total', 'address'
        
    Returns:
        SHA-256 hash of the normalized receipt fingerprint
    """
    # Extract entities with fallback to empty strings
    company = entity_dict.get('company', '').strip()
    date = entity_dict.get('date', '').strip()
    total = entity_dict.get('total', '').strip()
    
    # Normalize company name
    # Fraud detection rationale: Same company might be written with different
    # capitalization, punctuation, or extra whitespace
    company_normalized = re.sub(r'[^\w\s]', '', company.lower())  # Remove punctuation, lowercase
    company_normalized = re.sub(r'\s+', ' ', company_normalized)  # Normalize whitespace
    company_normalized = company_normalized.strip()
    
    # Normalize date
    # Fraud detection rationale: Dates might be formatted differently (DD/MM/YYYY,
    # MM/DD/YYYY, with/without separators) but represent the same date
    date_digits = re.sub(r'[^\d]', '', date)  # Extract only digits
    
    if len(date_digits) == 8:
        # Assume DDMMYYYY format and sort to standard format
        day = date_digits[:2]
        month = date_digits[2:4]
        year = date_digits[4:8]
        date_normalized = f"{day}{month}{year}"
    elif len(date_digits) == 6:
        # Assume DDMMYY format
        day = date_digits[:2]
        month = date_digits[2:4]
        year = date_digits[4:6]
        # Assume 2000s for years < 50, 1900s otherwise
        year = '20' + year if int(year) < 50 else '19' + year
        date_normalized = f"{day}{month}{year}"
    else:
        # If we can't parse the date, use original digits
        date_normalized = date_digits
    
    # Normalize total amount
    # Fraud detection rationale: Total might be written with different
    # currency symbols, separators, or precision but represent the same value
    total_clean = re.sub(r'[^\d.,]', '', total)  # Remove currency symbols and text
    
    # Handle different decimal separators
    if ',' in total_clean and '.' in total_clean:
        # If both exist, assume the last one is the decimal separator
        if total_clean.rfind(',') > total_clean.rfind('.'):
            total_clean = total_clean.replace('.', '').replace(',', '.')
        else:
            total_clean = total_clean.replace(',', '')
    elif ',' in total_clean:
        # Check if comma is likely decimal separator (European format)
        parts = total_clean.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            total_clean = total_clean.replace(',', '.')
        else:
            total_clean = total_clean.replace(',', '')
    
    try:
        total_float = float(total_clean)
        total_normalized = f"{total_float:.2f}"  # Always 2 decimal places
    except ValueError:
        # If we can't parse the total, use the cleaned string
        total_normalized = total_clean
    
    # Create the canonical fingerprint string
    # Fraud detection rationale: The combination of normalized company, date,
    # and total creates a unique identifier for the same receipt, regardless
    # of formatting differences or minor OCR errors
    fingerprint_str = f"{company_normalized}|{date_normalized}|{total_normalized}"
    
    # Generate SHA-256 hash for the fingerprint
    # This creates a fixed-length, collision-resistant identifier
    fingerprint_hash = hashlib.sha256(fingerprint_str.encode('utf-8')).hexdigest()
    
    logger.debug(f"Fingerprint created: {fingerprint_str} -> {fingerprint_hash}")
    
    return fingerprint_hash


class ReceiptLedger:
    """
    Ledger for tracking receipt submissions and detecting duplicates.
    
    This class implements the core fraud detection system by maintaining
    a persistent record of all processed receipts and identifying potential
    duplicate submissions through fingerprint matching.
    """
    
    def __init__(self, ledger_path: str = None):
        """
        Initialize the receipt ledger.
        
        Args:
            ledger_path: Path to the ledger JSON file
        """
        if ledger_path is None:
            ledger_path = CFG.paths.ledger_path
        self.ledger_path = Path(ledger_path)
        self.ledger: Dict[str, Dict] = {}
        
        # Create directory if it doesn't exist
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing ledger if file exists
        self._load_ledger()
        
        logger.info(f"ReceiptLedger initialized with {len(self.ledger)} unique receipts")
    
    def _load_ledger(self):
        """Load existing ledger from JSON file."""
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, 'r', encoding='utf-8') as f:
                    self.ledger = json.load(f)
                logger.info(f"Loaded ledger from {self.ledger_path}")
            except Exception as e:
                logger.error(f"Failed to load ledger from {self.ledger_path}: {e}")
                self.ledger = {}
        else:
            logger.info(f"No existing ledger found at {self.ledger_path}, starting fresh")
            self.ledger = {}
    
    def check_and_register(self, receipt_id: str, entity_dict: Dict) -> Dict:
        """
        Check if a receipt is a duplicate and register it if new.
        
        Args:
            receipt_id: Unique identifier for this receipt submission
            entity_dict: Dictionary with receipt entities (company, date, total, address)
            
        Returns:
            Dictionary with duplicate detection results:
            {
                'is_duplicate': bool,
                'fingerprint': str,
                'existing_record': dict | None
            }
        """
        # Generate fingerprint for this receipt
        fingerprint = build_receipt_fingerprint(entity_dict)
        
        # Check if fingerprint exists in ledger
        if fingerprint in self.ledger:
            # Duplicate detected!
            existing_record = self.ledger[fingerprint].copy()
            
            # Increment submission count
            self.ledger[fingerprint]['submission_count'] += 1
            self.ledger[fingerprint]['last_seen'] = datetime.now().isoformat()
            
            logger.warning(f"Duplicate receipt detected: {receipt_id} matches fingerprint {fingerprint}")
            logger.warning(f"Existing receipt ID: {existing_record['receipt_id']}, Submission count: {self.ledger[fingerprint]['submission_count']}")
            
            return {
                'is_duplicate': True,
                'fingerprint': fingerprint,
                'existing_record': existing_record
            }
        else:
            # New receipt, register it
            current_time = datetime.now().isoformat()
            
            # Create new ledger entry
            ledger_entry = {
                'receipt_id': receipt_id,
                'company': entity_dict.get('company', '').strip(),
                'date': entity_dict.get('date', '').strip(),
                'total': entity_dict.get('total', '').strip(),
                'address': entity_dict.get('address', '').strip(),
                'first_seen': current_time,
                'last_seen': current_time,
                'submission_count': 1
            }
            
            # Add to ledger
            self.ledger[fingerprint] = ledger_entry
            
            logger.info(f"New receipt registered: {receipt_id} with fingerprint {fingerprint}")
            
            return {
                'is_duplicate': False,
                'fingerprint': fingerprint,
                'existing_record': None
            }
    
    def save(self):
        """Save the ledger to JSON file with pretty printing."""
        try:
            # Create backup of existing ledger
            if self.ledger_path.exists():
                backup_path = self.ledger_path.with_suffix('.json.backup')
                self.ledger_path.rename(backup_path)
                logger.info(f"Created backup at {backup_path}")
            
            # Save current ledger
            with open(self.ledger_path, 'w', encoding='utf-8') as f:
                json.dump(self.ledger, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Ledger saved to {self.ledger_path}")
            
        except Exception as e:
            logger.error(f"Failed to save ledger to {self.ledger_path}: {e}")
            raise
    
    def get_fraud_report(self) -> List[Dict]:
        """
        Get all records with submission count > 1 (potential fraud).
        
        Returns:
            List of duplicate receipt records sorted by submission count (descending)
        """
        duplicates = [
            {
                'fingerprint': fingerprint,
                **record
            }
            for fingerprint, record in self.ledger.items()
            if record['submission_count'] > 1
        ]
        
        # Sort by submission count (highest first)
        duplicates.sort(key=lambda x: x['submission_count'], reverse=True)
        
        logger.info(f"Found {len(duplicates)} duplicate receipt groups")
        
        return duplicates
    
    def __len__(self) -> int:
        """Return the number of unique receipts in the ledger."""
        return len(self.ledger)
    
    def get_statistics(self) -> Dict:
        """Get ledger statistics."""
        total_submissions = sum(record['submission_count'] for record in self.ledger.values())
        duplicate_groups = len(self.get_fraud_report())
        
        return {
            'unique_receipts': len(self.ledger),
            'total_submissions': total_submissions,
            'duplicate_groups': duplicate_groups,
            'duplicate_rate': duplicate_groups / len(self.ledger) if self.ledger else 0
        }


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("Testing ReceiptLedger deduplication engine...")
    print("=" * 60)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        ledger_path = Path(temp_dir) / "test_ledger.json"
        
        # Create ledger
        ledger = ReceiptLedger(str(ledger_path))
        
        # Test data - two identical receipts with slight variations
        receipt1_entities = {
            'company': 'RESTORAN WAN SHENG',
            'date': '06/05/2018',
            'total': '2.40',
            'address': 'NO.2, JALAN TEMENGGUNG 19/9, SEKSYEN 9, BANDAR MAHKOTA CHERAS, 43200 CHERAS, SELANGOR'
        }
        
        receipt2_entities = {
            'company': 'restoran wan sheng',  # Different case
            'date': '06-05-2018',            # Different separator
            'total': 'RM 2.40',              # With currency symbol
            'address': 'NO.2, JALAN TEMENGGUNG 19/9, SEKSYEN 9, BANDAR MAHKOTA CHERAS, 43200 CHERAS, SELANGOR'
        }
        
        # Register first receipt
        print("Registering first receipt...")
        result1 = ledger.check_and_register("receipt_001", receipt1_entities)
        print(f"Result: {result1}")
        
        # Register second receipt (should be detected as duplicate)
        print("\nRegistering second receipt (duplicate)...")
        result2 = ledger.check_and_register("receipt_002", receipt2_entities)
        print(f"Result: {result2}")
        
        # Assertions
        assert not result1['is_duplicate'], "First receipt should not be marked as duplicate"
        assert result2['is_duplicate'], "Second receipt should be marked as duplicate"
        assert result1['fingerprint'] == result2['fingerprint'], "Fingerprints should match"
        assert result2['existing_record']['receipt_id'] == "receipt_001", "Should reference first receipt"
        
        print("\n✅ All assertions passed!")
        
        # Save and reload ledger
        print("\nSaving ledger...")
        ledger.save()
        
        # Create new ledger instance to test persistence
        print("Reloading ledger from disk...")
        ledger_reloaded = ReceiptLedger(str(ledger_path))
        
        # Check persistence
        assert len(ledger_reloaded) == 1, "Ledger should have 1 unique receipt"
        
        # Test fraud report
        print("\nGenerating fraud report...")
        fraud_report = ledger_reloaded.get_fraud_report()
        print(f"Found {len(fraud_report)} duplicate groups:")
        
        for i, duplicate in enumerate(fraud_report, 1):
            print(f"\nDuplicate Group {i}:")
            print(f"  Fingerprint: {duplicate['fingerprint']}")
            print(f"  Receipt ID: {duplicate['receipt_id']}")
            print(f"  Company: {duplicate['company']}")
            print(f"  Date: {duplicate['date']}")
            print(f"  Total: {duplicate['total']}")
            print(f"  Submission Count: {duplicate['submission_count']}")
            print(f"  First Seen: {duplicate['first_seen']}")
        
        # Test statistics
        print("\nLedger Statistics:")
        stats = ledger_reloaded.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test with a third unique receipt
        print("\nRegistering third unique receipt...")
        receipt3_entities = {
            'company': 'DIFFERENT RESTAURANT',
            'date': '10/06/2018',
            'total': '15.50',
            'address': '123 DIFFERENT STREET, CITY'
        }
        
        result3 = ledger.check_and_register("receipt_003", receipt3_entities)
        print(f"Result: {result3}")
        
        # Final statistics
        print("\nFinal Ledger Statistics:")
        final_stats = ledger_reloaded.get_statistics()
        for key, value in final_stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ ReceiptLedger test completed successfully!")