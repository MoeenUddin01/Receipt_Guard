# ReceiptGuard Web Interface

Complete web-based interface for ReceiptGuard fraud detection using both trained models.

## Quick Start

### Prerequisites
- Both models must be trained:
  - Model 1: `artifacts/model1/checkpoints/best_model.pt`
  - Model 2: `artifacts/siamese/best_model.pth`

### Option 1: Easy Start
```bash
# Start backend with automatic setup
python start_backend.py
```

### Option 2: Manual Setup
```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Start API server
python app.py
```

## Features

### Backend API (`http://localhost:8000`)
- **FastAPI** server with automatic model loading
- **Endpoints**:
  - `GET /` - API information
  - `GET /health` - Health check
  - `POST /predict` - Upload and analyze single receipt
  - `POST /predict/image` - Analyze receipt by file path
  - `POST /predict/batch` - Analyze multiple receipts
  - `GET /model/info` - Model information

### Frontend (`http://localhost:8000` in browser)
- **Drag & Drop** receipt image upload
- **Multiple File** selection support
- **Real-time Processing** with loading indicators
- **Visual Results** with verdict visualization:
  - 🚨 FRAUD (Red border)
  - ⚠️ SUSPICIOUS (Orange border)  
  - ✅ LEGITIMATE (Green border)
- **Confidence Scores** with visual progress bars
- **Extracted Information**: Company, Date, Total
- **Batch Analysis** summary for multiple receipts

## API Usage Examples

### Single Receipt Analysis
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@receipt.jpg"
```

### Batch Analysis
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "images=@receipt1.jpg" \
  -F "images=@receipt2.jpg"
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Response Format

### Single Prediction
```json
{
  "company": "BOOK TA .K SDN BHD",
  "date": "25/12/2018", 
  "total": "9.00",
  "verdict": "FRAUD",
  "confidence": 0.94,
  "similarity_score": 0.95,
  "fingerprint": "abc123...",
  "is_new_receipt": false
}
```

### Batch Analysis
```json
{
  "results": [...],
  "total_processed": 10,
  "fraud_count": 2,
  "suspicious_count": 1,
  "legitimate_count": 7
}
```

## Architecture

```
┌─────────────────┐
│   Browser     │
└─────────┬─────┘
          │
    ┌─────┴─────┐
    │  Frontend    │  HTML/CSS/JS
    │  (index.html) │  - Drag & drop upload
    │               │  - Visual results
    │               │  - Batch analysis
    └─────┬─────┘
          │
    ┌─────┴─────┐
    │   Backend     │  FastAPI
    │  (app.py)     │  - Model loading
    │               │  - Inference endpoints
    │               │  - Error handling
    └─────┬─────┘
          │
    ┌─────┴─────┐
    │   Models      │  PyTorch Checkpoints
    │               │  - Model 1: NER (1.3GB)
    │               │  - Model 2: Siamese (51.8MB)
    └───────────────┘
```

## Troubleshooting

### Models Not Found
```
❌ Model 1 not found at artifacts/model1/checkpoints/best_model.pt
❌ Model 2 not found at artifacts/siamese/best_model.pth

💡 To train missing models:
   Model 1: python main.py train
   Model 2: python -m src_2.pipelines.siamese_training_pipeline
```

### Server Won't Start
```
❌ Failed to start application: Model loading error

💡 Check:
   - Model files exist and are not corrupted
   - Python dependencies installed: pip install -r backend/requirements.txt
   - No port conflicts (kill processes on port 8000)
```

### API Connection Issues
```
Cannot connect to backend API

💡 Verify:
   - Backend server is running on localhost:8000
   - No firewall blocking the connection
   - API health check: curl http://localhost:8000/health
```

## Production Deployment

### Environment Setup
```bash
# Set production environment variables
export RECEIPTGUARD_ENV=production
export API_HOST=0.0.0.0
export API_PORT=8000

# Start with production settings
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Security Considerations
- **File Upload Limits**: Maximum 10MB per image, 10 images per batch
- **Temporary Files**: Auto-cleanup after processing
- **Input Validation**: Only image files accepted
- **Error Handling**: No sensitive information leaked in error messages

## Performance

### Expected Response Times
- **Single Prediction**: 2-5 seconds (depends on image size)
- **Batch Processing**: 10-30 seconds for 10 images
- **Model Loading**: 10-15 seconds on startup

### Resource Requirements
- **RAM**: 4GB+ recommended (both models loaded)
- **Storage**: 2GB+ for model checkpoints
- **CPU**: Multi-core recommended for faster processing
