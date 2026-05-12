"""
database_server.py - FastAPI server using the two-database image matching system.

Endpoints:
  POST /predict  - Upload receipt image → DB1/DB2 check → verdict
  GET  /health   - Health check
  GET  /stats    - Database statistics
"""

import os
import sys
import tempfile
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src_2.data.image_database import ImageDatabase

app = FastAPI(title="ReceiptGuard Database Fraud Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_predictor: ImageDatabase = None


def load_predictor():
    global db_predictor
    if db_predictor is None:
        print("Initializing ImageDatabase...")
        db_predictor = ImageDatabase()
        print(f"DB stats: {db_predictor.stats()}")


@app.on_event("startup")
async def startup():
    try:
        load_predictor()
    except Exception as e:
        print(f"Startup error: {e}")
        raise


@app.get("/")
async def root():
    return {
        "message": "ReceiptGuard Database Fraud Detection",
        "endpoints": {"/health": "Health check", "/predict": "Upload image for prediction", "/stats": "Database stats"},
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "predictor_loaded": db_predictor is not None}


@app.get("/stats")
async def stats():
    if db_predictor is None:
        raise HTTPException(503, "Predictor not loaded")
    return db_predictor.stats()


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if db_predictor is None:
        raise HTTPException(503, "Predictor not loaded")

    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await image.read())
            tmp_path = tmp.name

        result = db_predictor.predict(tmp_path)

        os.unlink(tmp_path)

        return {
            "filename": image.filename,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "detail": result["detail"],
            "similarity": result["similarity"],
        }

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


if __name__ == "__main__":
    print("Starting ReceiptGuard Database Fraud Detection API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
