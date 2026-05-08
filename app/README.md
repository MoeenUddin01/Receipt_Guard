# ReceiptGuard UI

A modern web interface for the ReceiptGuard-ML receipt extraction model.

## Features

- 📤 Upload receipt images (PNG, JPG, JPEG)
- 🔍 Extract key fields: Company, Date, Address, Total
- 📊 Confidence scores for each extraction
- 🖼️ Visual bounding box overlay on receipts
- 📋 Copy extracted data to clipboard

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the backend server:
   ```bash
   python backend/app.py
   ```

3. Open `frontend/index.html` in your browser or serve it with:
   ```bash
   cd frontend && python -m http.server 3000
   ```

## Architecture

- **Backend**: FastAPI server with PyTorch model inference
- **Frontend**: Vanilla HTML/CSS/JS with modern UI
- **Model**: LayoutLM-based NER model for receipt field extraction
