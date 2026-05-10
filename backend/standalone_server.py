#!/usr/bin/env python3
"""
standalone_server.py - Minimal FastAPI server for demonstration.

This creates a simple server that can run without complex dependencies
for demonstration purposes.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("Please install with: pip install fastapi uvicorn")
    exit(1)

app = FastAPI(title="ReceiptGuard Demo API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock predictor for demonstration
class MockPredictor:
    def __init__(self):
        self.model1_loaded = Path("artifacts/best_model.pt").exists()
        self.model2_loaded = Path("artifacts/siamese/best_model.pth").exists()
    
    def predict_from_image(self, image_path: str) -> Dict[str, Any]:
        """Mock prediction for demonstration."""
        return {
            "company": "BOOK TA .K SDN BHD",
            "date": "25/12/2018",
            "total": "9.00",
            "verdict": "FRAUD" if self.model2_loaded else "LEGITIMATE",
            "confidence": 0.94 if self.model2_loaded else 0.85,
            "similarity_score": 0.95 if self.model2_loaded else 0.25,
            "fingerprint": "abc123...",
            "is_new_receipt": False,
            "models_loaded": {
                "model1": self.model1_loaded,
                "model2": self.model2_loaded
            }
        }

predictor = MockPredictor()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "ReceiptGuard Fraud Detection API - Demo Mode",
        "version": "1.0.0",
        "models": {
            "model1_loaded": predictor.model1_loaded,
            "model2_loaded": predictor.model2_loaded
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "demo": "/demo"
        }
    }

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "models": {
            "model1_loaded": predictor.model1_loaded,
            "model2_loaded": predictor.model2_loaded
        }
    }

@app.get("/demo")
async def demo():
    """Demo endpoint without file upload."""
    return {
        "message": "Demo prediction result",
        "result": predictor.predict_from_image("demo_receipt.jpg")
    }

@app.post("/predict")
async def predict_demo():
    """Demo prediction endpoint."""
    return JSONResponse(
        content={
            "message": "Demo mode - file upload not implemented in standalone server",
            "suggestion": "Use full backend with dependencies: python start_backend.py",
            "demo_result": predictor.predict_from_image("uploaded_file")
        },
        status_code=200
    )

if __name__ == "__main__":
    print("🚀 Starting ReceiptGuard Demo Server...")
    print("📍 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("🔍 Health: http://localhost:8000/health")
    print("🎭 Demo: http://localhost:8000/demo")
    
    if not predictor.model1_loaded or not predictor.model2_loaded:
        print("⚠️  Warning: Running in demo mode - some models missing")
        print("💡 For full functionality, ensure both models are trained:")
        print("   Model 1: python main.py train")
        print("   Model 2: python -m src_2.pipelines.siamese_training_pipeline")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
