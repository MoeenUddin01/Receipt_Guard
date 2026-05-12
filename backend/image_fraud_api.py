"""
backend/image_fraud_api.py - Dedicated API for ImageFraudDetector.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src_2.inference.image_fraud_detector import ImageFraudDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ReceiptGuard Standalone API",
    description="Dedicated API for image-based receipt fraud detection using Model 2 only.",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directory to serve the UI
_frontend_path = Path(_project_root) / "frontend"
if _frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_path)), name="static")

# Global detector instance
detector: Optional[ImageFraudDetector] = None

def get_detector():
    """Lazy load the detector."""
    global detector
    if detector is None:
        model_path = "artifacts/siamese/2_model/best_model.pth"
        # Fallback if the path is slightly different
        if not Path(model_path).exists():
            model_path = "artifacts/siamese/best_model.pth"
            
        try:
            detector = ImageFraudDetector(
                model2_checkpoint=model_path,
                known_db_path="artifacts/known_receipts.json",
                used_db_path="artifacts/used_receipts.json"
            )
            logger.info("✅ ImageFraudDetector loaded successfully with seeded databases")
        except Exception as e:
            logger.error(f"❌ Failed to load detector: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    return detector

@app.on_event("startup")
async def startup_event():
    """Ensure detector is loaded on startup."""
    try:
        get_detector()
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if detector is not None else "initializing",
        "model_loaded": detector is not None,
        "database_entries": len(detector.database) if detector else 0
    }

@app.post("/check")
async def check_receipt(
    image: UploadFile = File(..., description="Receipt image file")
):
    """
    Check a receipt for fraud using image similarity.
    """
    det = get_detector()
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_path = temp_file.name
            with open(temp_path, "wb") as f:
                f.write(await image.read())
            
            # Run fraud detection
            result = det.check_receipt(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            return result
            
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {
        "app": "ReceiptGuard Standalone API",
        "ui_url": "/static/image_fraud_ui.html",
        "endpoints": ["/check", "/health", "/static/image_fraud_ui.html"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
