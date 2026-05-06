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
- **Data Layer**
  - Data preprocessing pipeline (`src/data/preprocessing.py`)
  - PyTorch dataset wrapper (`src/data/dataset.py`)
  - Data loading utilities & deduplication engine (`src/data/dataloader.py`)
  - Entity normalization and BIO labeling
  - Bounding box normalization for LayoutLM
  - Tokenization and bbox alignment
  - Fraud detection through fingerprint-based deduplication
  
- **Model Layer**
  - LayoutLM-based NER model (`src/model/model.py`)
  - Full fine-tuning with dropout and Xavier-initialized classifier
  - Training loop with mixed precision (`src/model/train.py`)
  - Evaluation metrics with entity-level F1 (`src/model/evaluation.py`)
  - Checkpoint save/load utilities
  - TensorBoard logging and training curve visualization

🚧 **In Progress:**
- End-to-end training pipelines
- Inference pipeline for new receipts

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

### Model Training

```python
from src.model.model import build_model, ModelConfig
from src.model.train import Trainer, get_device
from src.data.dataset import ReceiptDataset, collate_fn
from torch.utils.data import DataLoader

# Configuration
config = ModelConfig(
    model_path="microsoft/layoutlm-base-uncased",
    num_labels=9,
    dropout=0.1,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    max_length=512,
)

# Build model
device = get_device()
model = build_model(config)

# Create data loaders
train_dataset = ReceiptDataset("dataset/raw/SROIE2019", "train", max_length=512)
val_dataset = ReceiptDataset("dataset/raw/SROIE2019", "test", max_length=512)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

# Train
output_dir = "outputs/run_001"
trainer = Trainer(model, train_loader, val_loader, config, device, output_dir)
trainer.train(num_epochs=10)

# Training artifacts:
# - outputs/run_001/best_model.pt
# - outputs/run_001/final_model.pt
# - outputs/run_001/training_log.json
# - outputs/run_001/training_curves.png
# - outputs/run_001/runs/ (TensorBoard logs)
```

### Model Evaluation

```python
from src.model.model import build_model, ModelConfig, load_checkpoint
from src.model.evaluation import run_full_evaluation, plot_confusion_matrix, ID2LABEL
from src.model.train import get_device
from src.data.dataset import ReceiptDataset, collate_fn
from torch.utils.data import DataLoader

# Load model
config = ModelConfig.load("outputs/run_001/config.json")
model = build_model(config)
device = get_device()

# Load checkpoint
load_checkpoint("outputs/run_001/best_model.pt", model)

# Create test loader
test_dataset = ReceiptDataset("dataset/raw/SROIE2019", "test", max_length=512)
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

# Run evaluation
results = run_full_evaluation(model, test_loader, device, ID2LABEL)

print(f"Token Accuracy: {results['token_accuracy']:.4f}")
print(f"Macro F1: {results['ner_metrics']['macro']['f1']:.4f}")
print("Per-entity F1:")
for entity, metrics in results['ner_metrics']['per_entity'].items():
    print(f"  {entity}: {metrics['f1']:.4f}")
```

### Inference on New Receipt

```python
from src.model.model import ReceiptFieldExtractor, load_checkpoint
from src.model.evaluation import extract_entities_from_predictions, ID2LABEL
from transformers import AutoTokenizer
import torch

# Load model
model = ReceiptFieldExtractor(
    model_path="microsoft/layoutlm-base-uncased",
    num_labels=9,
)
load_checkpoint("outputs/run_001/best_model.pt", model)
model.eval()

# Prepare input (from preprocessing)
tokens = ["ACME", "Corp", "Invoice", "Date:", "2024-01-15", "Total:", "$100.50"]
bboxes = [[10, 10, 100, 30], [110, 10, 200, 30], ...]  # normalized 0-1000

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True)

# Get bbox aligned with tokens
# ... (alignment logic from dataset.py)

# Predict
with torch.no_grad():
    loss, logits = model(
        encoding["input_ids"],
        encoding["attention_mask"],
        torch.zeros_like(encoding["input_ids"]),
        bbox_tensor,
    )

# Extract entities
predictions = model.get_predictions(logits, encoding["attention_mask"])
entities = extract_entities_from_predictions(tokens, predictions[0], ID2LABEL)

print(entities)
# Output: {'company': 'ACME Corp', 'date': '2024-01-15', 'address': '', 'total': '$100.50'}
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

# Test model architecture (requires LayoutLM weights)
python src/model/model.py

# Test training utilities
python src/model/train.py

# Test evaluation metrics
python -c "from src.model.evaluation import *; print('Evaluation module OK')"
```

## Project Structure

```
ReceiptGuard-ML/
├── src/
│   ├── data/                   # Data loading and preprocessing
│   │   ├── preprocessing.py    # OCR parsing, entity extraction, bbox normalization
│   │   ├── dataset.py          # PyTorch dataset with tokenization
│   │   └── dataloader.py       # Collate utilities & deduplication engine
│   ├── model/                  # Model architecture and training
│   │   ├── model.py            # LayoutLM-based NER model with classifier head
│   │   ├── train.py            # Training loop with mixed precision
│   │   └── evaluation.py       # NER metrics and fraud detection evaluation
│   └── pipelines/              # End-to-end pipelines
│       ├── preprocessing_pipeline.py
│       ├── model_training_pipeline.py
│       └── evaluation_pipeline.py
├── dataset/                    # Dataset directory (gitignored)
│   ├── raw/                    # Raw SROIE2019 data
│   └── processed/              # Processed data cache
├── dataset.md                  # Dataset documentation
├── pyproject.toml             # Project configuration
├── main.py                    # Main entry point
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

### ReceiptFieldExtractor

- **Base Model**: LayoutLM (layout-aware language model) - full fine-tuning, no frozen layers
- **Classifier Head**: 
  - Dropout layer (`p=0.1` default)
  - Linear layer: `hidden_size → num_labels` (9 classes: O, B/I × COMPANY, DATE, ADDRESS, TOTAL)
  - Xavier uniform weight initialization, zero bias
- **Task**: Named Entity Recognition for receipt fields
- **Input Format**: Tokenized text + normalized bounding boxes (0-1000 scale)
- **Output**: NER labels for each token
- **Loss**: CrossEntropyLoss (ignores padding with index -100)

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

### Model Features
- **Full Fine-Tuning**: All LayoutLM layers trainable (no frozen parameters)
- **Custom Classifier**: Xavier-initialized linear head with dropout regularization
- **Mixed Precision Training**: Automatic AMP when CUDA available (faster, less memory)
- **Gradient Clipping**: Max norm 1.0 for training stability
- **Checkpoint Management**: Best model tracking + final model save
- **TensorBoard Integration**: Live loss/accuracy/LR logging

### Training Features
- **AdamW Optimizer**: Weight decay on non-bias/LayerNorm parameters only
- **Linear Warmup**: Configurable warmup steps followed by linear decay
- **Automatic Device Detection**: CUDA → MPS → CPU fallback
- **Progress Bars**: Live loss display with tqdm
- **Training Curves**: Auto-generated matplotlib plots (loss + accuracy)

### Evaluation Features
- **Entity-Level Metrics**: Span-level precision/recall/F1 per entity type
- **Token-Level Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Visual confusion matrix with seaborn heatmap
- **Per-Receipt Entity Extraction**: Structured output for downstream use
- **Fraud Detection Metrics**: Precision/recall/F1 for duplicate detection

### Deduplication Engine Features
- **Fingerprint-Based Detection**: SHA-256 hashing of normalized receipt entities
- **Robust Normalization**: Handles variations in capitalization, punctuation, date formats, and currency symbols
- **Persistent Ledger**: JSON-based storage with automatic backup
- **Submission Tracking**: Counts duplicate submissions with timestamps
- **Fraud Reporting**: Generates reports of all duplicate receipt groups
- **Statistics**: Tracks unique receipts, total submissions, and duplicate rates

## Future Development

- [x] Complete LayoutLM model wrapper
- [x] Implement training pipeline with proper loss functions
- [x] Add evaluation metrics (F1, precision, recall)
- [x] Implement model serialization and loading
- [x] Add duplicate detection functionality
- [ ] Create inference pipeline for new receipts
- [ ] Add comprehensive test suite
- [ ] Create web interface for receipt processing
- [ ] Export models to ONNX for production deployment

## License

MIT License

## Acknowledgments

- [SROIE2019 Dataset](https://rrc.cvc.uab.es/?ch=13) - Scanned Receipt OCR and Information Extraction
- [LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm) by Microsoft Research
- [Transformers](https://github.com/huggingface/transformers) by Hugging Face

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
