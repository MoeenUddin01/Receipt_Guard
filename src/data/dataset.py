import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from preprocessing import build_processed_sample

# Import configuration
from src.config import CFG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NER label mapping
LABEL2ID = {
    'O': 0,
    'B-COMPANY': 1,
    'I-COMPANY': 2,
    'B-DATE': 3,
    'I-DATE': 4,
    'B-ADDRESS': 5,
    'I-ADDRESS': 6,
    'B-TOTAL': 7,
    'I-TOTAL': 8
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class ReceiptDataset(Dataset):
    """
    PyTorch Dataset for receipt OCR data with NER labels.
    
    Each item returns:
    - input_ids: Tokenized text IDs
    - bbox: Normalized bounding boxes (0-1000 scale)
    - attention_mask: Attention mask for padding
    - labels: NER label IDs
    """
    
    def __init__(
        self,
        data_path: str,
        split: str,
        tokenizer_name: str = None,
        max_length: int = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset root
            split: Dataset split ('train' or 'test')
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
            cache_dir: Optional cache directory for processed data
        """
        if tokenizer_name is None:
            tokenizer_name = CFG.model.model_path
        if max_length is None:
            max_length = CFG.data.max_length
        self.data_path = Path(data_path)
        self.split = split
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load sample IDs
        self.sample_ids = self._get_sample_ids()
        
        # Load or process data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _get_sample_ids(self) -> List[str]:
        """Get all sample IDs from the box files directory."""
        box_dir = self.data_path / self.split / "box"
        if not box_dir.exists():
            raise FileNotFoundError(f"Box directory not found: {box_dir}")
        
        # Get all .txt files and extract stems as sample IDs
        box_files = list(box_dir.glob("*.txt"))
        sample_ids = [f.stem for f in box_files]
        
        logger.info(f"Found {len(sample_ids)} samples in {self.split} split")
        return sample_ids
    
    def _load_data(self) -> List[Dict]:
        """Load and process data, using cache if available."""
        if self.cache_dir:
            cache_file = self.cache_dir / f"{self.split}_processed.json"
            if cache_file.exists():
                logger.info(f"Loading cached data from {cache_file}")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # Process data
        data = []
        failed_samples = []
        
        for sample_id in self.sample_ids:
            try:
                sample = build_processed_sample(sample_id, self.split, str(self.data_path))
                if sample:
                    data.append(sample)
                else:
                    failed_samples.append(sample_id)
            except Exception as e:
                logger.error(f"Failed to process sample {sample_id}: {e}")
                failed_samples.append(sample_id)
        
        if failed_samples:
            logger.warning(f"Failed to process {len(failed_samples)} samples: {failed_samples[:10]}...")
        
        # Cache processed data if cache directory is specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{self.split}_processed.json"
            logger.info(f"Caching processed data to {cache_file}")
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        return data
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dict containing:
            - input_ids: Tokenized text IDs [max_length]
            - bbox: Normalized bounding boxes [max_length, 4]
            - attention_mask: Attention mask [max_length]
            - labels: NER label IDs [max_length]
        """
        sample = self.data[idx]
        
        # Tokenize text and align with bounding boxes
        encoding = self._tokenize_with_bbox_alignment(
            sample['tokens'], 
            sample['bboxes'], 
            sample['labels']
        )
        
        return encoding
    
    def _tokenize_with_bbox_alignment(
        self, 
        tokens: List[str], 
        bboxes: List[List[int]], 
        labels: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text and align bounding boxes and labels.
        
        Args:
            tokens: List of text tokens
            bboxes: List of bounding boxes [x_min, y_min, x_max, y_max]
            labels: List of NER labels
            
        Returns:
            Dict with tensors: input_ids, bbox, attention_mask, labels
        """
        # Convert labels to IDs
        label_ids = [LABEL2ID[label] for label in labels]
        
        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Align bounding boxes and labels with tokenized words
        word_ids = encoding.word_ids()
        aligned_bboxes = []
        aligned_labels = []
        
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                aligned_bboxes.append([0, 0, 0, 0])
                aligned_labels.append(LABEL2ID['O'])
            elif word_idx != previous_word_idx:
                # First token of a word
                aligned_bboxes.append(bboxes[word_idx])
                aligned_labels.append(label_ids[word_idx])
            else:
                # Subsequent tokens of the same word
                aligned_bboxes.append(bboxes[word_idx])
                aligned_labels.append(label_ids[word_idx])
            
            previous_word_idx = word_idx
        
        # Convert to tensors
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        bbox_tensor = torch.tensor(aligned_bboxes, dtype=torch.long)
        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'bbox': bbox_tensor,
            'attention_mask': attention_mask,
            'labels': labels_tensor
        }
    
    def get_label_names(self) -> List[str]:
        """Get the list of label names."""
        return list(LABEL2ID.keys())
    
    def get_num_labels(self) -> int:
        """Get the number of unique labels."""
        return len(LABEL2ID)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to batch samples.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched tensors
    """
    # Stack all tensors in the batch
    input_ids = torch.stack([item['input_ids'] for item in batch])
    bbox = torch.stack([item['bbox'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'bbox': bbox,
        'attention_mask': attention_mask,
        'labels': labels
    }


if __name__ == "__main__":
    # Test the dataset
    import pprint
    
    print("Testing ReceiptDataset...")
    print("=" * 50)
    
    # Initialize dataset
    try:
        dataset = ReceiptDataset(
            data_path=CFG.data.raw_data_path,
            split="train",
            max_length=CFG.data.max_length  # Use smaller max_length for testing
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of labels: {dataset.get_num_labels()}")
        print(f"Label names: {dataset.get_label_names()}")
        
        # Test first sample
        print("\nTesting first sample:")
        print("-" * 30)
        
        sample = dataset[0]
        
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"BBox shape: {sample['bbox'].shape}")
        print(f"Attention mask shape: {sample['attention_mask'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        
        # Decode some tokens to verify
        tokenizer = dataset.tokenizer
        decoded_tokens = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
        print(f"\nDecoded tokens (first 100 chars): {decoded_tokens[:100]}...")
        
        # Show some labels
        label_names = [ID2LABEL[label_id.item()] for label_id in sample['labels'][:20]]
        print(f"First 20 labels: {label_names}")
        
        print("\n✅ Dataset test successful!")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
