# ReceiptGuard-ML

A PyTorch-based receipt information extraction system using LayoutLM and the SROIE2019 dataset.

## Overview

ReceiptGuard-ML is a machine learning system that extracts key information from scanned receipts using Microsoft's LayoutLM model. The system processes receipt images to identify and extract company names, dates, addresses, and total amounts through named entity recognition (NER).

## Features

- **Receipt Information Extraction**: Extracts company, date, address, and total amount from receipts
- **Fraud Detection**: Fingerprint-based deduplication engine to detect duplicate receipt submissions
- **LayoutLM Integration**: Leverages pre-trained LayoutLM for document understanding with spatial awareness
- **SROIE2019 Dataset**: Built on the benchmark receipt OCR dataset
- **Comprehensive Preprocessing**: Robust data pipeline with bbox normalization and BIO labeling
- **PyTorch Dataset**: Ready-to-use dataset class with proper tensor handling
- **Error Handling**: Graceful handling of malformed data and missing files

## Current Implementation Status

✅ **Completed Modules:**
- Data preprocessing pipeline (`src/data/preprocessing.py`)
- PyTorch dataset wrapper (`src/data/dataset.py`)
- Data loading utilities & deduplication engine (`src/data/dataloader.py`)
- Entity normalization and BIO labeling
- Bounding box normalization for LayoutLM
- Tokenization and bbox alignment
- Fraud detection through fingerprint-based deduplication

🚧 **In Progress:**
- Model architecture implementation
- Training pipeline
- Evaluation metrics

## Installation

### Prerequisites

- Python >= 3.10
- pip (package manager)

### Setup

```bash
# Clone the repository
git clone https://github.com/MoeenUddin01/Receipt_Guard.git
cd Receipt_Guard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision transformers Pillow numpy scikit-learn pandas tqdm matplotlib seaborn tensorboard python-dotenv
```

### Dataset Setup

The SROIE2019 dataset is not included in the repository due to size constraints. Download it separately and place it in `dataset/raw/SROIE2019/`:

```
dataset/raw/SROIE2019/
├── train/
│   ├── img/        # Receipt images (.jpg)
│   ├── box/        # OCR bounding boxes (.txt)
│   └── entities/   # Ground truth labels (.txt)
└── test/
    ├── img/
    ├── box/
    └── entities/
```

## Usage

### Data Preprocessing

```python
from src.data.preprocessing import build_processed_sample

# Process a single receipt
sample = build_processed_sample(
    receipt_id="X51005433494",
    split="train", 
    base_path="dataset/raw/SROIE2019"
)

print(sample)
# Output: {'id': 'X51005433494', 'tokens': [...], 'bboxes': [...], 'labels': [...], 'entity_values': {...}}
```

### PyTorch Dataset

```python
from src.data.dataset import ReceiptDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = ReceiptDataset(
    data_path="dataset/raw/SROIE2019",
    split="train",
    max_length=512
)

# Create data loader
dataloader = DataLoader(
    dataset, 
    batch_size=8, 
    collate_fn=dataset.collate_fn,
    shuffle=True
)

# Iterate over batches
for batch in dataloader:
    input_ids = batch['input_ids']      # [batch_size, max_length]
    bbox = batch['bbox']                # [batch_size, max_length, 4]
    attention_mask = batch['attention_mask']  # [batch_size, max_length]
    labels = batch['labels']            # [batch_size, max_length]
    break
```

### Fraud Detection (Deduplication)

```python
from src.data.dataloader import ReceiptLedger, build_receipt_fingerprint

# Create ledger
ledger = ReceiptLedger("dataset/processed/ledger.json")

# Check and register a receipt
entity_dict = {
    'company': 'RESTORAN WAN SHENG',
    'date': '06/05/2018',
    'total': '2.40',
    'address': 'NO.2, JALAN TEMENGGUNG 19/9, SEKSYEN 9, BANDAR MAHKOTA CHERAS'
}

result = ledger.check_and_register("receipt_001", entity_dict)

if result['is_duplicate']:
    print(f"Duplicate detected! Original receipt: {result['existing_record']['receipt_id']}")
else:
    print("New receipt registered")

# Get fraud report
fraud_report = ledger.get_fraud_report()
print(f"Found {len(fraud_report)} duplicate groups")

# Save ledger
ledger.save()
```

### Testing the Implementation

```bash
# Test preprocessing module
python src/data/preprocessing.py

# Test dataset module
cd src/data
python dataset.py

# Test dataloader and deduplication engine
python src/data/dataloader.py
```

## Project Structure

```
ReceiptGuard-ML/
├── src/
│   ├── data/                   # Data loading and preprocessing
│   │   ├── preprocessing.py    # Complete: OCR parsing, entity extraction, bbox normalization
│   │   ├── dataset.py          # Complete: PyTorch dataset with tokenization
│   │   ├── dataloader.py       # Complete: Collate utilities & deduplication engine
│   ├── model/                  # Model architecture and training
│   │   ├── model.py            # Empty: LayoutLM model wrapper
│   │   ├── train.py            # Empty: Training loop
│   │   └── evaluation.py       # Empty: Evaluation metrics
│   └── pipelines/              # End-to-end pipelines
│       ├── preprocessing_pipeline.py    # Empty: Data preprocessing pipeline
│       ├── model_training_pipeline.py   # Empty: Training pipeline
│       └── evaluation_pipeline.py       # Empty: Evaluation pipeline
├── dataset/                    # Dataset directory (gitignored)
│   ├── raw/                    # Raw SROIE2019 data
│   └── processed/              # Processed data cache
├── dataset.md                  # Dataset documentation
├── pyproject.toml             # Project configuration
├── main.py                    # 📝 Empty: Main entry point
└── README.md                  # This file
```

## Data Format

### Box Files
Each line in box files contains: `x1,y1,x2,y2,x3,y3,x4,y4,text`

### Entity Files
JSON format with keys: `company`, `date`, `address`, `total`

### Processed Output
```python
{
    'id': 'receipt_id',
    'split': 'train|test',
    'tokens': ['token1', 'token2', ...],
    'bboxes': [[x_min, y_min, x_max, y_max], ...],
    'labels': ['B-COMPANY', 'O', 'B-DATE', ...],
    'entity_values': {
        'company': 'Company Name',
        'date': 'DD/MM/YYYY',
        'address': 'Full Address',
        'total': 'XX.XX'
    }
}
```

## NER Labels

The system uses BIO (Begin-Inside-Outside) labeling scheme:

- `O`: Outside any entity
- `B-COMPANY`/`I-COMPANY`: Company name
- `B-DATE`/`I-DATE`: Date
- `B-ADDRESS`/`I-ADDRESS`: Address
- `B-TOTAL`/`I-TOTAL`: Total amount

## Model Architecture

- **Base Model**: LayoutLM (layout-aware language model)
- **Task**: Named Entity Recognition for receipt fields
- **Input Format**: Tokenized text + normalized bounding boxes
- **Output**: NER labels for each token

## Development

### Code Quality

```bash
# Format code (when black is installed)
black src/

# Lint code (when ruff is installed)
ruff check src/
ruff check --fix src/
```

### Testing

```bash
# Test individual modules
python src/data/preprocessing.py
python src/data/dataset.py

# Run tests (when pytest is installed)
pytest
```

## Implementation Details

### Preprocessing Features
- **Robust OCR Parsing**: Handles malformed box files gracefully
- **Entity Normalization**: 
  - Dates normalized to DD/MM/YYYY format
  - Totals normalized to 2 decimal places
- **BIO Labeling**: Intelligent substring matching for multi-token entities
- **BBox Normalization**: Converts pixel coordinates to LayoutLM's 0-1000 scale

### Dataset Features
- **Tokenization Integration**: Uses LayoutLM tokenizer with word-level alignment
- **Caching**: Optional JSON cache for processed data
- **Error Handling**: Skips corrupted samples with detailed logging
- **Flexible Batching**: Custom collate function for proper tensor stacking

### Deduplication Engine Features
- **Fingerprint-Based Detection**: SHA-256 hashing of normalized receipt entities
- **Robust Normalization**: Handles variations in capitalization, punctuation, date formats, and currency symbols
- **Persistent Ledger**: JSON-based storage with automatic backup
- **Submission Tracking**: Counts duplicate submissions with timestamps
- **Fraud Reporting**: Generates reports of all duplicate receipt groups
- **Statistics**: Tracks unique receipts, total submissions, and duplicate rates

## Future Development

- [ ] Complete LayoutLM model wrapper
- [ ] Implement training pipeline with proper loss functions
- [ ] Add evaluation metrics (F1, precision, recall)
- [ ] Create inference pipeline for new receipts
- [x] Add duplicate detection functionality
- [ ] Implement model serialization and loading
- [ ] Add comprehensive test suite
- [ ] Create web interface for receipt processing

## License

MIT License

## Acknowledgments

- [SROIE2019 Dataset](https://rrc.cvc.uab.es/?ch=13) - Scanned Receipt OCR and Information Extraction
- [LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm) by Microsoft Research
- [Transformers](https://github.com/huggingface/transformers) by Hugging Face

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
