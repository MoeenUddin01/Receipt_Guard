"""
LayoutLM-based NER model for receipt field extraction.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import LayoutLMModel


@dataclass
class ModelConfig:
    """Configuration for ReceiptFieldExtractor model."""

    model_path: str
    num_labels: int
    dropout: float = 0.1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_length: int = 512

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class ReceiptFieldExtractor(nn.Module):
    """
    LayoutLM-based NER model for extracting receipt fields.
    
    Architecture:
    - LayoutLM encoder (full fine-tuning, no frozen layers)
    - Dropout layer
    - Linear classifier head: hidden_size -> num_labels
    """

    def __init__(self, model_path: str, num_labels: int, dropout: float = 0.1):
        """
        Initialize the receipt field extractor.
        
        Args:
            model_path: Path to pretrained LayoutLM weights
            num_labels: Number of NER labels (including O tag)
            dropout: Dropout probability for classifier
        """
        super().__init__()
        
        # Load LayoutLM encoder (no frozen layers - full fine-tuning)
        self.layoutlm = LayoutLMModel.from_pretrained(model_path)
        
        # Get hidden size from config
        self.hidden_size = self.layoutlm.config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Linear classifier head
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize classifier weights with Xavier uniform + zero bias
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        bbox: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            bbox: Bounding boxes [batch_size, seq_len, 4] (x0, y0, x1, y1)
            labels: Optional label IDs [batch_size, seq_len] (-100 for ignored tokens)
            
        Returns:
            Tuple of (loss, logits) if labels provided, else (None, logits)
        """
        # Run LayoutLM encoder
        outputs = self.layoutlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            bbox=bbox,
        )
        
        # Get sequence output: [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout
        dropped = self.dropout(sequence_output)
        
        # Apply classifier: [batch_size, seq_len, num_labels]
        logits = self.classifier(dropped)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Flatten for loss computation: [batch_size * seq_len, num_labels]
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )
            return loss, logits
        
        return None, logits

    def get_predictions(
        self, logits: Tensor, attention_mask: Tensor
    ) -> list[list[int]]:
        """
        Get predicted label IDs for non-padding tokens.
        
        Args:
            logits: Model output logits [batch_size, seq_len, num_labels]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            List of predicted label ID lists (one per sample, excluding padding)
        """
        # Get predicted IDs: [batch_size, seq_len]
        pred_ids = torch.argmax(logits, dim=-1)
        
        predictions = []
        for i in range(logits.size(0)):
            # Get valid token positions (attention_mask == 1)
            valid_mask = attention_mask[i] == 1
            valid_preds = pred_ids[i][valid_mask].cpu().tolist()
            predictions.append(valid_preds)
        
        return predictions


def build_model(config: ModelConfig) -> ReceiptFieldExtractor:
    """
    Instantiate a ReceiptFieldExtractor from config.
    
    Args:
        config: ModelConfig with model parameters
        
    Returns:
        Initialized ReceiptFieldExtractor model
    """
    model = ReceiptFieldExtractor(
        model_path=config.model_path,
        num_labels=config.num_labels,
        dropout=config.dropout,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model built successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def save_checkpoint(
    model: ReceiptFieldExtractor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: ReceiptFieldExtractor,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, float]:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    
    print(f"Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    
    return epoch, loss


if __name__ == "__main__":
    # Test the model with dummy config and random tensors
    print("=" * 60)
    print("Testing ReceiptFieldExtractor")
    print("=" * 60)
    
    # Use microsoft/layoutlm-base-uncased as dummy path
    # In practice, this should be a local path with downloaded weights
    dummy_config = ModelConfig(
        model_path="microsoft/layoutlm-base-uncased",
        num_labels=12,  # Example: O, B-AMOUNT, I-AMOUNT, B-DATE, I-DATE, etc.
        dropout=0.1,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        max_length=512,
    )
    
    print("\nBuilding model...")
    try:
        model = build_model(dummy_config)
        
        # Create dummy input tensors
        batch_size = 2
        seq_len = 128
        
        print(f"\nCreating dummy inputs (batch_size={batch_size}, seq_len={seq_len})...")
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        # Make some tokens padding (last 10 tokens of second sample)
        attention_mask[1, -10:] = 0
        
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        # Bounding boxes: [batch_size, seq_len, 4] with values 0-1000
        bbox = torch.randint(0, 1000, (batch_size, seq_len, 4))
        
        # Test forward pass without labels
        print("\nForward pass (inference mode)...")
        model.eval()
        with torch.no_grad():
            loss, logits = model(input_ids, attention_mask, token_type_ids, bbox)
        
        print(f"  Loss: {loss}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Expected: [{batch_size}, {seq_len}, {dummy_config.num_labels}]")
        assert logits.shape == (batch_size, seq_len, dummy_config.num_labels)
        
        # Test get_predictions
        print("\nTesting get_predictions...")
        predictions = model.get_predictions(logits, attention_mask)
        print(f"  Number of samples: {len(predictions)}")
        print(f"  Sample 0 predictions: {len(predictions[0])} tokens")
        print(f"  Sample 1 predictions: {len(predictions[1])} tokens (padding excluded)")
        assert len(predictions[0]) == seq_len
        assert len(predictions[1]) == seq_len - 10  # Excluding padding
        
        # Test forward pass with labels
        print("\nForward pass (training mode with labels)...")
        model.train()
        labels = torch.randint(-1, dummy_config.num_labels, (batch_size, seq_len))
        labels[labels == -1] = -100  # Convert -1 to ignore index
        
        loss, logits = model(input_ids, attention_mask, token_type_ids, bbox, labels)
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Loss shape: {loss.shape}")
        assert loss.shape == ()  # Scalar loss
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Note: This requires the LayoutLM model weights to be available.")
        print("You may need to run 'from transformers import LayoutLMTokenizer; LayoutLMTokenizer.from_pretrained(\"microsoft/layoutlm-base-uncased\")' first to download weights.")
        raise
