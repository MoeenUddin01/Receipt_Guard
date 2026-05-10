#!/usr/bin/env python3
"""
start_simple.py - Simple backend startup without dependency installation.
"""

import os
import sys
from pathlib import Path


def check_models():
    """Check if both models are available."""
    model1_path = Path("artifacts/best_model.pt")
    model2_path = Path("artifacts/siamese/best_model.pth")
    
    missing_models = []
    
    if not model1_path.exists():
        missing_models.append(f"Model 1: {model1_path}")
    
    if not model2_path.exists():
        missing_models.append(f"Model 2: {model2_path}")
    
    if missing_models:
        print("❌ Missing models:")
        for model in missing_models:
            print(f"   {model}")
        print("\n💡 To train missing models:")
        print("   Model 1: python main.py train")
        print("   Model 2: python -m src_2.pipelines.siamese_training_pipeline")
        return False
    
    print("✅ Both models found:")
    print(f"   Model 1: {model1_path} ({model1_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"   Model 2: {model2_path} ({model2_path.stat().st_size / (1024*1024):.1f} MB)")
    return True


def start_server():
    """Start FastAPI server."""
    print("🚀 Starting ReceiptGuard API server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("\n💡 Usage:")
    print("   1. Open http://localhost:8000 in your browser")
    print("   2. Upload receipt images for fraud detection")
    print("   3. View API docs at http://localhost:8000/docs")
    print("\n⚠️  Press Ctrl+C to stop the server")
    
    try:
        # Set environment variable to avoid virtual env issues
        os.environ['PYTHONPATH'] = '/usr/bin/python3'
        
        # Import and start server
        from backend.app import app
        import uvicorn
        
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return 1
    
    return 0


def main():
    """Main entry point."""
    print("🛡️  ReceiptGuard Fraud Detection Backend")
    print("=" * 50)
    
    # Check models
    if not check_models():
        return 1
    
    # Start server
    return start_server()


if __name__ == "__main__":
    exit(main())
