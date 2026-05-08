"""
ReceiptGuard API Backend
FastAPI server for receipt field extraction
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import io
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.model import ReceiptFieldExtractor, ModelConfig
from src.data.preprocessing import parse_box_file, parse_entity_file
from transformers import LayoutLMTokenizer
from src.config import CFG

app = FastAPI(title="ReceiptGuard API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
LABEL_LIST = ["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE", "B-ADDRESS", "I-ADDRESS", "B-TOTAL", "I-TOTAL"]


class ExtractedField(BaseModel):
    text: str
    confidence: float
    bbox: List[int]  # [x_min, y_min, x_max, y_max]


class ExtractionResult(BaseModel):
    company: Optional[ExtractedField]
    date: Optional[ExtractedField]
    address: Optional[ExtractedField]
    total: Optional[ExtractedField]
    raw_tokens: List[Dict]
    success: bool
    error: Optional[str] = None


def load_model():
    """Load the trained model checkpoint"""
    global model
    
    checkpoint_path = Path(__file__).parent.parent.parent / "artifacts" / "best_model.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found at {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model config
    model_config = ModelConfig(
        model_path="microsoft/layoutlm-base-uncased",
        num_labels=9,
        dropout=0.1
    )
    
    # Build model
    model = ReceiptFieldExtractor(
        model_path=model_config.model_path,
        num_labels=model_config.num_labels,
        dropout=model_config.dropout
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully (device: {device})")
    return model


def preprocess_image(image: Image.Image) -> tuple:
    """Preprocess image for LayoutLM"""
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Get image dimensions
    width, height = image.size
    
    return image, width, height


def extract_text_with_boxes(image: Image.Image) -> List[Dict]:
    """
    Extract text and bounding boxes from image using OCR
    For now, this is a placeholder - in production you'd use pytesseract or similar
    """
    # TODO: Implement OCR here
    # For demo purposes, return mock data
    return []


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
        print("API will run but predictions will fail")


@app.post("/extract", response_model=ExtractionResult)
async def extract_receipt_fields(file: UploadFile = File(...)):
    """
    Extract fields from a receipt image
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        image, width, height = preprocess_image(image)
        
        # TODO: Implement full inference pipeline
        # For now, return a mock response showing the structure
        
        return ExtractionResult(
            company=ExtractedField(
                text="DEMO COMPANY LTD",
                confidence=0.95,
                bbox=[100, 50, 300, 80]
            ),
            date=ExtractedField(
                text="2024-01-15",
                confidence=0.92,
                bbox=[400, 50, 550, 80]
            ),
            address=ExtractedField(
                text="123 Business Street, City",
                confidence=0.88,
                bbox=[100, 100, 400, 130]
            ),
            total=ExtractedField(
                text="$125.50",
                confidence=0.96,
                bbox=[400, 400, 500, 430]
            ),
            raw_tokens=[],
            success=True
        )
        
    except Exception as e:
        return ExtractionResult(
            company=None,
            date=None,
            address=None,
            total=None,
            raw_tokens=[],
            success=False,
            error=str(e)
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "LayoutLM-base-uncased",
        "num_labels": 9,
        "labels": LABEL_LIST,
        "device": str(device),
        "checkpoint": "best_model.pt"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
