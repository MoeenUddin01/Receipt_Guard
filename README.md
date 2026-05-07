# ReceiptGuard-ML

A PyTorch-based receipt information extraction system using LayoutLM and the SROIE2019 dataset with centralized configuration management.

## Overview

ReceiptGuard-ML is a machine learning system that extracts key information from scanned receipts using Microsoft's LayoutLM model. The system processes receipt images to identify and extract company names, dates, addresses, and total amounts through named entity recognition (NER) with built-in fraud detection capabilities.

## Features

- **Receipt Information Extraction**: Extracts company, date, address, and total amount from receipts
- **Fraud Detection**: Fingerprint-based deduplication engine to detect duplicate receipt submissions
- **LayoutLM Integration**: Leverages pre-trained LayoutLM for document understanding with spatial awareness
- **SROIE2019 Dataset**: Built on the benchmark receipt OCR dataset
- **Centralized Configuration**: YAML-based configuration system with CLI override support
- **CLI Interface**: Complete command-line interface for preprocessing, training, and evaluation
- **Kaggle Support**: Runtime configuration overrides for notebook environments
- **Comprehensive Preprocessing**: Robust data pipeline with bbox normalization and BIO labeling
- **PyTorch Dataset**: Ready-to-use dataset class with proper tensor handling
- **Error Handling**: Graceful handling of malformed data and missing files

## Current Implementation Status

✅ **Completed Modules:**
- **Configuration System**
  - Centralized YAML configuration (`config.yaml`)
  - Global CFG object with attribute access
  - CLI override support for runtime customization
  - Kaggle notebook compatibility with `override_config()`
  
- **CLI Interface**
  - Complete command-line entry point (`main.py`)
  - Subcommands: preprocess, train, evaluate, full
  - Argument validation and help documentation
  - Error handling and progress reporting

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

- **Pipeline Layer**
  - End-to-end preprocessing pipeline (`src/pipelines/preprocessing_pipeline.py`)
  - Model training pipeline (`src/pipelines/model_training_pipeline.py`)
  - Evaluation pipeline (`src/pipelines/evaluation_pipeline.py`)
  - Progress tracking and artifact management

- **Inference Layer**
  - Receipt predictor (`src/inference/predictor.py`)
  - OCR integration with pytesseract
  - Batch inference support
  - Duplicate detection integration

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

### Command Line Interface

The main entry point is `main.py` which provides a complete CLI for all operations:

```bash
# Full pipeline (preprocess → train → evaluate)
python main.py full

# Individual operations
python main.py preprocess --raw_path /path/to/data --processed_path /path/to/output
python main.py train --num_epochs 10 --batch_size 4 --learning_rate 3e-5
python main.py evaluate --checkpoint_path artifacts/checkpoints/best_model.pt

# Get help
python main.py --help
python main.py train --help
```

### Configuration System

All settings are managed through `config.yaml`:

```yaml
# Key configuration sections
project:
  name: "ReceiptGuard-ML"
  version: "1.0.0"

paths:
  artifacts_dir: "artifacts"
  checkpoints_dir: "artifacts/checkpoints"
  evaluation_dir: "artifacts/evaluation"
  logs_dir: "artifacts/logs"
  ledger_path: "artifacts/ledger.json"

training:
  num_epochs: 10
  batch_size: 8
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_ratio: 0.1

model:
  model_path: "dataset/raw/SROIE2019/layoutlm-base-uncased"
  num_labels: 9
  dropout: 0.1

# ... and more
```

#### Runtime Configuration Override

```python
from src.config import CFG, override_config

# Override values at runtime (useful for Kaggle notebooks)
override_config({
    'training.batch_size': 16,
    'training.num_epochs': 15,
    'paths.raw_data_dir': '/kaggle/input/sroie2019'
})

# Access configuration
print(f"Batch size: {CFG.training.batch_size}")
print(f"Model path: {CFG.model.model_path}")
```

#### Path Helper

```python
from src.config import get_path

# Get absolute paths with automatic directory creation
artifacts_path = get_path('paths.artifacts_dir')
checkpoints_path = get_path('paths.checkpoints_dir')

# Paths are absolute and directories are created automatically
print(artifacts_path)  # /home/user/projects/ReceiptGuard-ML/artifacts
```

### Programmatic Usage

#### Data Preprocessing

```python
from src.data.preprocessing import build_processed_sample
from src.config import CFG

# Process a single receipt
sample = build_processed_sample(
    receipt_id="X51005433494",
    split="train", 
    base_path=CFG.paths.raw_data_path
)

print(sample)
# Output: {'id': 'X51005433494', 'tokens': [...], 'bboxes': [...], 'labels': [...], 'entity_values': {...}}
```

#### PyTorch Dataset

```python
from src.data.dataset import ReceiptDataset
from src.config import CFG
from torch.utils.data import DataLoader

# Create dataset using configuration
dataset = ReceiptDataset(
    data_path=CFG.data.raw_data_path,
    split="train",
    max_length=CFG.data.max_length
)

# Create data loader
dataloader = DataLoader(
    dataset, 
    batch_size=CFG.training.batch_size, 
    collate_fn=collate_fn,
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

#### Model Training

```python
from src.model.model import build_model, ModelConfig
from src.model.train import Trainer, get_device
from src.config import CFG
from torch.utils.data import DataLoader

# Configuration from CFG
config = ModelConfig(
    model_path=CFG.model.model_path,
    num_labels=CFG.model.num_labels,
    dropout=CFG.model.dropout,
    learning_rate=CFG.training.learning_rate,
    weight_decay=CFG.training.weight_decay,
    warmup_steps=CFG.training.warmup_steps,
    max_length=CFG.data.max_length,
)

# Build model
device = get_device()
model = build_model(config)

# Train using pipeline
from src.pipelines.model_training_pipeline import run_training_pipeline, TrainingConfig

training_config = TrainingConfig()
summary = run_training_pipeline(training_config)

print(f"Training completed: {summary['status']}")
print(f"Best checkpoint: {summary['best_checkpoint']}")
```

#### Model Evaluation

```python
from src.pipelines.evaluation_pipeline import run_evaluation_pipeline, EvaluationConfig
from src.config import CFG

# Evaluate using pipeline
config = EvaluationConfig(
    checkpoint_path="artifacts/checkpoints/best_model.pt"
)

summary = run_evaluation_pipeline(config)

print(f"Macro F1: {summary['ner_metrics']['macro']['f1']:.4f}")
print(f"Duplicates detected: {summary['fraud_report']['duplicates_found']}")
```

#### Inference on New Receipt

```python
from src.inference.predictor import ReceiptPredictor
from src.config import CFG

# Initialize predictor
predictor = ReceiptPredictor(
    checkpoint_path="artifacts/checkpoints/best_model.pt"
)

# Process receipt image or box file
result = predictor.predict_from_image("receipt.jpg")

print("Extracted entities:")
for entity_type, value in result['entities'].items():
    if value:
        print(f"  {entity_type}: {value}")

# Check for duplicates
if result['is_duplicate']:
    print(f"⚠️  Duplicate detected! Original: {result['duplicate_info']['original_receipt_id']}")
else:
    print("✅ New receipt registered")
```

### Testing the Implementation

```bash
# Test configuration system
python src/config.py

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

# Test CLI help
python main.py --help
```

## Project Structure

```
ReceiptGuard-ML/
├── config.yaml                    # Centralized configuration file
├── src/
│   ├── config.py                  # Configuration management system
│   ├── main.py                    # CLI entry point
│   ├── data/                      # Data loading and preprocessing
│   │   ├── preprocessing.py       # OCR parsing, entity extraction, bbox normalization
│   │   ├── dataset.py             # PyTorch dataset with tokenization
│   │   └── dataloader.py          # Collate utilities & deduplication engine
│   ├── model/                     # Model architecture and training
│   │   ├── model.py               # LayoutLM-based NER model with classifier head
│   │   ├── train.py               # Training loop with mixed precision
│   │   └── evaluation.py          # NER metrics and fraud detection evaluation
│   ├── pipelines/                 # End-to-end pipelines
│   │   ├── preprocessing_pipeline.py
│   │   ├── model_training_pipeline.py
│   │   └── evaluation_pipeline.py
│   └── inference/                 # Inference system
│       └── predictor.py           # Receipt predictor with OCR and deduplication
├── artifacts/                     # Generated artifacts (gitignored)
│   ├── checkpoints/               # Model checkpoints
│   ├── evaluation/                # Evaluation results
│   ├── logs/                      # Training logs
│   └── ledger.json                # Fraud detection ledger
├── dataset/                       # Dataset directory (gitignored)
│   ├── raw/                       # Raw SROIE2019 data
│   └── processed/                 # Processed data cache
├── dataset.md                     # Dataset documentation
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## Configuration Reference

### Configuration Sections

#### Project
- `name`: Project name
- `description`: Project description
- `version`: Project version

#### Paths
- `artifacts_dir`: Base directory for generated artifacts
- `checkpoints_dir`: Model checkpoint storage
- `evaluation_dir`: Evaluation results storage
- `logs_dir`: Training logs storage
- `ledger_path`: Fraud detection ledger file
- `raw_data_dir`: Raw dataset location
- `processed_data_dir`: Processed dataset cache
- `model_dir`: Pretrained model location

#### Data
- `max_length`: Maximum sequence length for tokenization
- `train_split`: Training split name
- `test_split`: Test split name
- `ledger_filename`: Ledger file name

#### Model
- `model_path`: Path to pretrained LayoutLM model
- `num_labels`: Number of NER labels
- `dropout`: Dropout rate
- `hidden_size`: Model hidden size

#### Training
- `num_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay
- `warmup_ratio`: Warmup ratio
- `warmup_steps`: Warmup steps
- `max_grad_norm`: Maximum gradient norm
- `seed`: Random seed
- `output_dir`: Output directory for checkpoints

#### Inference
- `checkpoint_dir`: Default checkpoint directory
- `default_checkpoint`: Default checkpoint filename
- `ledger_path`: Inference ledger path
- `max_length`: Inference max sequence length
- `batch_size`: Inference batch size

#### Kaggle
- `input_path`: Kaggle input directory
- `working_path`: Kaggle working directory
- `receiptguard_path`: Kaggle dataset path
- `sroie2019_path`: Kaggle SROIE2019 path
- `optimized_batch_size`: Optimized batch size for Kaggle
- `optimized_epochs`: Optimized epochs for Kaggle

## CLI Reference

### Global Options
- `--version`: Show version information
- `--help`: Show help message

### Subcommands

#### preprocess
```bash
python main.py preprocess [OPTIONS]
```
Options:
- `--raw_path`: Path to raw dataset (default: from config)
- `--processed_path`: Path for processed output (default: from config)
- `--splits`: Dataset splits to process (default: ["train", "test"])
- `--verify_images`: Verify image files during preprocessing

#### train
```bash
python main.py train [OPTIONS]
```
Options:
- `--model_path`: Path to pretrained LayoutLM model
- `--num_labels`: Number of NER labels
- `--dropout`: Dropout rate
- `--output_dir`: Output directory for checkpoints
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--max_length`: Maximum sequence length
- `--learning_rate`: Learning rate
- `--weight_decay`: Weight decay
- `--warmup_ratio`: Warmup ratio
- `--seed`: Random seed
- `--data_path`: Path to dataset

#### evaluate
```bash
python main.py evaluate [OPTIONS]
```
Options:
- `--checkpoint_path`: Path to model checkpoint (required)
- `--model_path`: Path to LayoutLM model
- `--processed_data_path`: Path to processed data
- `--output_dir`: Output directory for evaluation results
- `--batch_size`: Batch size for evaluation
- `--no_fraud_detection`: Skip fraud detection

#### full
```bash
python main.py full [OPTIONS]
```
Runs complete pipeline: preprocess → train → evaluate
Includes all options from preprocess, train, and evaluate subcommands.

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

### Configuration System Features
- **Centralized Management**: All settings in `config.yaml`
- **Type Safety**: Automatic type conversion and validation
- **Runtime Overrides**: CLI arguments override config values
- **Path Resolution**: Automatic absolute path generation and directory creation
- **Environment Support**: Kaggle notebook compatibility
- **Attribute Access**: `CFG.training.batch_size` syntax
- **Dot Notation**: `CFG.get('training.batch_size')` support

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
- **Gradient Clipping**: Configurable max norm via `CFG.training.max_grad_norm`
- **Checkpoint Management**: Best model tracking + final model save
- **TensorBoard Integration**: Live loss/accuracy/LR logging

### Training Features
- **AdamW Optimizer**: Weight decay on non-bias/LayerNorm parameters only
- **Linear Warmup**: Configurable warmup ratio via `CFG.training.warmup_ratio`
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
- [x] Create centralized configuration system
- [x] Implement CLI interface
- [x] Add inference pipeline for new receipts
- [ ] Add comprehensive test suite
- [ ] Create web interface for receipt processing
- [ ] Export models to ONNX for production deployment
- [ ] Add model quantization for mobile deployment
- [ ] Implement multi-language support

## License

MIT License

## Acknowledgments

- [SROIE2019 Dataset](https://rrc.cvc.uab.es/?ch=13) - Scanned Receipt OCR and Information Extraction
- [LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm) by Microsoft Research
- [Transformers](https://github.com/huggingface/transformers) by Hugging Face

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

1. **Configuration**: Always use `CFG` for configuration values, never hardcode
2. **CLI Integration**: Update CLI arguments when adding new configuration options
3. **Testing**: Add tests for new functionality
4. **Documentation**: Update README and docstrings
5. **Code Style**: Follow existing code style and use black for formatting
