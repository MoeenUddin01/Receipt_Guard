"""
app.py - FastAPI backend for Siamese fraud detection model.

Provides REST API endpoints for receipt upload, processing, and fraud detection
using the trained Siamese model combined with NER model.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

# Add project root to path so src/ and src_2/ are findable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src_2.inference.fraud_predictor import ReceiptGuardPredictor


app = FastAPI(
    title="ReceiptGuard Fraud Detection API",
    description="Backend API for receipt fraud detection using Siamese similarity learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[ReceiptGuardPredictor] = None


def load_predictor():
    """Load the fraud detection predictor on startup."""
    global predictor
    if predictor is None:
        model1_path = "artifacts/best_model.pt"
        model2_path = "artifacts/siamese/best_model.pth"
        
        if not Path(model1_path).exists():
            raise RuntimeError(f"Model 1 not found at {model1_path}")
        if not Path(model2_path).exists():
            raise RuntimeError(f"Model 2 not found at {model2_path}")
        
        try:
            predictor = ReceiptGuardPredictor(
                model1_checkpoint=model1_path,
                model2_checkpoint=model2_path,
                model_path="dataset/raw/SROIE2019/layoutlm-base-uncased",
                ledger_path="artifacts/siamese/ledger.json",
                device="auto"
            )
            print("✅ ReceiptGuardPredictor loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load predictor: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize models on application startup."""
    try:
        load_predictor()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        raise


# Serve frontend static files
_frontend_dir = Path(_project_root) / "frontend"
if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")

@app.get("/")
async def root():
    """Root endpoint - serves frontend or API info."""
    return {
        "message": "ReceiptGuard Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "ui": "/app",
            "health": "/health",
            "predict": "/predict",
            "predict_image": "/predict/image",
            "batch_predict": "/predict/batch"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if predictor is None:
            return {"status": "error", "message": "Models not loaded"}
        
        # Test models are loaded
        test_result = {"status": "healthy", "models_loaded": True}
        return test_result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/predict")
async def predict_receipt(
    image: UploadFile = File(..., description="Receipt image file")
):
    """
    Predict fraud on uploaded receipt image.
    
    Args:
        image: Receipt image file (JPG, PNG, etc.)
        
    Returns:
        JSON response with prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_path = temp_file.name
            
            # Save uploaded image
            with open(temp_path, "wb") as f:
                f.write(await image.read())
            
            # Run prediction
            result = predictor.predict_from_image(temp_path)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return result
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/image")
async def predict_from_image_path(
    image_path: str
):
    """
    Predict fraud on receipt image by file path.
    
    Args:
        image_path: Path to receipt image file
        
    Returns:
        JSON response with prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
    
    try:
        result = predictor.predict_from_image(image_path)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def batch_predict_receipts(
    images: list[UploadFile] = File(...)
):
    """
    Predict fraud on multiple receipt images.
    
    Args:
        images: List of receipt image files
        
    Returns:
        JSON response with batch prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if len(images) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum batch size is 10 images")
    
    try:
        results = []
        temp_files = []
        
        # Process each image
        for i, image in enumerate(images):
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {i+1} is not an image")
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
                temp_files.append(temp_path)
                
                with open(temp_path, "wb") as f:
                    f.write(await image.read())
        
        try:
            # Run batch prediction
            batch_results = predictor.batch_predict(temp_files)
            
            # Clean up temp files
            for temp_path in temp_files:
                if Path(temp_path).exists():
                    os.unlink(temp_path)
            
            return {
                "results": batch_results,
                "total_processed": len(batch_results),
                "fraud_count": sum(1 for r in batch_results if r.get('verdict') == 'FRAUD'),
                "suspicious_count": sum(1 for r in batch_results if r.get('verdict') == 'SUSPICIOUS'),
                "legitimate_count": sum(1 for r in batch_results if r.get('verdict') == 'LEGITIMATE')
            }
            
        except Exception as e:
            # Clean up on error
            for temp_path in temp_files:
                if Path(temp_path).exists():
                    os.unlink(temp_path)
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about loaded models."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "model_1": {
            "loaded": True,
            "checkpoint": "artifacts/best_model.pt"
        },
        "model_2": {
            "loaded": True,
            "checkpoint": "artifacts/siamese/best_model.pth",
            "similarity_threshold": getattr(predictor, 'similarity_threshold', None)
        },
        "device": str(predictor.device) if predictor else None
    }


if __name__ == "__main__":
    print("🚀 Starting ReceiptGuard Fraud Detection API...")
    
    # Check if models exist before starting
    model1_path = "artifacts/best_model.pt"
    model2_path = "artifacts/siamese/best_model.pth"
    
    if not Path(model1_path).exists():
        print(f"❌ Model 1 not found at {model1_path}")
        print("Please train Model 1 first using: python main.py train")
        exit(1)
    
    if not Path(model2_path).exists():
        print(f"❌ Model 2 not found at {model2_path}")
        print("Please train Model 2 first using: python -m src_2.pipelines.siamese_training_pipeline")
        exit(1)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
