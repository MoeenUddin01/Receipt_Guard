# SROIE2019 Dataset

The SROIE2019 (Scanned Receipt OCR and Information Extraction) dataset is a benchmark for receipt understanding and information extraction tasks.

## Dataset Structure

```
dataset/raw/SROIE2019/
├── train/          # Training set (626 samples)
│   ├── img/        # Scanned receipt images (.jpg)
│   ├── box/        # OCR text with bounding boxes (.txt)
│   └── entities/   # Ground truth key-value pairs (.txt)
├── test/           # Test set (347 samples)
│   ├── img/        # Scanned receipt images (.jpg)
│   ├── box/        # OCR text with bounding boxes (.txt)
│   └── entities/   # Ground truth key-value pairs (.txt)
└── layoutlm-base-uncased/  # Pre-trained LayoutLM model
    ├── pytorch_model.bin
    ├── config.json
    ├── vocab.txt
    └── training_args.bin
```

## Data Splits

| Split | Images | Box Files | Entity Files |
|-------|--------|-----------|--------------|
| Train | 626    | 626       | 626          |
| Test  | 347    | 347       | 347          |

## File Formats

### Box Files (`box/*.txt`)
Contain OCR output with 8-point bounding box coordinates:

```
x1,y1,x2,y2,x3,y3,x4,y4,text
```

Example:
```
72,25,326,25,326,64,72,64,TAN WOON YANN
205,372,342,372,342,389,165,389,25/12/2018 8:13:39 PM
```

### Entity Files (`entities/*.txt`)
Contain structured ground truth in JSON format with 4 key fields:

```json
{
    "company": "BOOK TA .K (TAMAN DAYA) SDN BHD",
    "date": "25/12/2018",
    "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR.",
    "total": "9.00"
}
```

## Target Fields

| Field | Description | Example |
|-------|-------------|---------|
| `company` | Store/merchant name | "BOOK TA .K (TAMAN DAYA) SDN BHD" |
| `date` | Transaction date | "25/12/2018" |
| `address` | Store location | "NO.53 55,57 & 59, JALAN SAGU 18..." |
| `total` | Total amount | "9.00" |

## Pre-trained Model

The `layoutlm-base-uncased/` directory contains a pre-trained LayoutLM model:

- **Model**: LayoutLM (Document AI transformer)
- **Type**: Multimodal (text + layout + visual features)
- **Files**:
  - `pytorch_model.bin` (453 MB) - Model weights
  - `vocab.txt` - BERT tokenizer vocabulary
  - `config.json` - Model architecture configuration
  - `training_args.bin` - Training hyperparameters

## Purpose

This dataset is used for:
- **Task**: Key information extraction from scanned receipts
- **Approach**: Combining OCR text, bounding box coordinates, and visual features
- **Model**: LayoutLM for understanding document structure and content
