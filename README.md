# ReceiptGuard-ML

A PyTorch-based duplicate receipt detection system using LayoutLM and the SROIE2019 dataset.

## Overview

ReceiptGuard-ML is a machine learning system that extracts key information from scanned receipts and detects duplicate submissions. It uses Microsoft's LayoutLM model to understand both the visual layout and textual content of receipt images.

## Features

- **Receipt Understanding**: Extracts company, date, address, and total amount from receipts
- **Duplicate Detection**: Identifies potential duplicate receipt submissions
- **LayoutLM Integration**: Leverages pre-trained LayoutLM for document understanding
- **SROIE2019 Dataset**: Trained on benchmark receipt dataset
- **Comprehensive Evaluation**: F1, precision, recall metrics with visualization

## Installation

### Prerequisites

- Python >= 3.10
- uv (package manager)

### Setup

```bash
# Clone the repository
git clone https://github.com/MoeenUddin01/Receipt_Guard.git
cd Receipt_Guard

# Create virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"
```

### Dataset Setup

The SROIE2019 dataset is not included in the repository due to size constraints. Download it separately and place it in `dataset/raw/SROIE2019/`:

```
dataset/raw/SROIE2019/
├── train/
│   ├── img/        # Receipt images (.jpg)
│   ├── box/        # OCR bounding boxes (.txt)
│   └── entities/   # Ground truth labels (.txt)
├── test/
│   ├── img/
│   ├── box/
│   └── entities/
└── layoutlm-base-uncased/  # Pre-trained model
```

See `dataset.md` for detailed dataset documentation.

## Usage

### Command Line

```bash
# Run the main application
receiptguard

# Or directly
python main.py
```

### Training

```bash
python -m src.pipelines.model_training_pipeline
```

### Evaluation

```bash
python -m src.pipelines.evaluation_pipeline
```

## Project Structure

```
ReceiptGuard-ML/
├── src/
│   ├── data/                   # Data loading and preprocessing
│   │   ├── dataloader.py
│   │   └── preprocessing.py
│   ├── model/                  # Model architecture and training
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluation.py
│   └── pipelines/               # End-to-end pipelines
│       ├── preprocessing_pipeline.py
│       ├── model_training_pipeline.py
│       └── evaluation_pipeline.py
├── dataset/                    # Dataset directory (gitignored)
├── dataset.md                  # Dataset documentation
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## Model Architecture

- **Base Model**: LayoutLM (layout-aware language model)
- **Task**: Key information extraction from receipts
- **Target Fields**: company, date, address, total

## Development

### Code Quality

```bash
# Format code
black src/

# Lint code
ruff check src/
ruff check --fix src/
```

### Testing

```bash
pytest
```

## License

MIT License

## Acknowledgments

- [SROIE2019 Dataset](https://rrc.cvc.uab.es/?ch=13)
- [LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm) by Microsoft
