"""Inference module for ReceiptGuard-ML."""

from .predictor import ReceiptPredictor, predict_receipt

__all__ = ["ReceiptPredictor", "predict_receipt"]
