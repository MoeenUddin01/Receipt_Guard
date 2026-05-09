"""
siamese_model.py - Siamese LayoutLM model for receipt comparison.

This module implements a Siamese network using LayoutLM to compare two receipts
and detect fraud through similarity analysis and binary classification.
Architecture:
  Receipt A → LayoutLM encoder → CLS embedding
                                     ↓
                                cosine similarity → score
                                     ↑
  Receipt B → LayoutLM encoder → CLS embedding
  (shared LayoutLM weights between both branches)
                                     ↓
                            binary classifier head
                                     ↓
                            fraud (1) / legit (0)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMModel, LayoutLMConfig


@dataclass
class SiameseConfig:
    """Configuration for Siamese LayoutLM model."""
    model_path: str = "microsoft/layoutlm-base-uncased"
    dropout: float = 0.1
    projection_dim: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'model_path': self.model_path,
            'dropout': self.dropout,
            'projection_dim': self.projection_dim,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SiameseConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)


class SiameseSimilarityModel(nn.Module):
    """
    Siamese network using LayoutLM for receipt fraud detection.
    
    Uses shared LayoutLM weights for both receipt branches, projects CLS embeddings
    to comparison space, and combines similarity with classification.
    """
    
    def __init__(self, model_path: str, dropout: float = 0.1, projection_dim: int = 256):
        """
        Initialize the Siamese model.
        
        Args:
            model_path: Path or name of pretrained LayoutLM model
            dropout: Dropout rate for regularization
            projection_dim: Dimension for projection head (comparison space)
        """
        super().__init__()
        
        # Load shared LayoutLM model
        self.layoutlm = LayoutLMModel.from_pretrained(model_path)
        hidden_size = self.layoutlm.config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Projection head: Linear(hidden_size → projection_dim) + ReLU
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier head: Linear(projection_dim * 3 → 2)
        # Input is concat of [emb_a, emb_b, |emb_a - emb_b|]
        self.classifier = nn.Linear(projection_dim * 3, 2)
        
        self.projection_dim = projection_dim
    
    def encode(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor, 
        bbox: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a receipt using LayoutLM and project to comparison space.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len]
            bbox: Bounding boxes [batch, seq_len, 4]
            
        Returns:
            Projected embeddings [batch, projection_dim]
        """
        # Forward pass through LayoutLM
        outputs = self.layoutlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            bbox=bbox
        )
        
        # Get CLS token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        
        # Apply dropout and projection
        cls_embedding = self.dropout(cls_embedding)
        projected = self.projection_head(cls_embedding)  # [batch, projection_dim]
        
        return projected
    
    def forward(
        self, 
        receipt_a: dict, 
        receipt_b: dict, 
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass through Siamese network.
        
        Args:
            receipt_a: Dictionary with input_ids, attention_mask, token_type_ids, bbox
            receipt_b: Dictionary with input_ids, attention_mask, token_type_ids, bbox
            labels: Optional labels for computing loss [batch]
            
        Returns:
            Tuple of (loss, logits, similarity_scores)
            - loss: CrossEntropyLoss if labels provided, else None
            - logits: Classification logits [batch, 2]
            - similarity_scores: Cosine similarity scores [batch]
        """
        # Encode both receipts
        emb_a = self.encode(
            receipt_a['input_ids'],
            receipt_a['attention_mask'],
            receipt_a['token_type_ids'],
            receipt_a['bbox']
        )  # [batch, projection_dim]
        
        emb_b = self.encode(
            receipt_b['input_ids'],
            receipt_b['attention_mask'],
            receipt_b['token_type_ids'],
            receipt_b['bbox']
        )  # [batch, projection_dim]
        
        # Compute cosine similarity scores
        similarity_scores = F.cosine_similarity(emb_a, emb_b, dim=1)  # [batch]
        
        # Build classification input: [emb_a, emb_b, |emb_a - emb_b|]
        diff = torch.abs(emb_a - emb_b)  # [batch, projection_dim]
        classifier_input = torch.cat([emb_a, emb_b, diff], dim=1)  # [batch, projection_dim * 3]
        
        # Pass through classifier
        logits = self.classifier(classifier_input)  # [batch, 2]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return loss, logits, similarity_scores
    
    def get_similarity(self, receipt_a: dict, receipt_b: dict) -> torch.Tensor:
        """
        Get cosine similarity scores between two receipts.
        
        Args:
            receipt_a: Dictionary with input_ids, attention_mask, token_type_ids, bbox
            receipt_b: Dictionary with input_ids, attention_mask, token_type_ids, bbox
            
        Returns:
            Cosine similarity scores [batch], clamped to [0, 1] for interpretability
        """
        # Encode both receipts
        emb_a = self.encode(
            receipt_a['input_ids'],
            receipt_a['attention_mask'],
            receipt_a['token_type_ids'],
            receipt_a['bbox']
        )
        
        emb_b = self.encode(
            receipt_b['input_ids'],
            receipt_b['attention_mask'],
            receipt_b['token_type_ids'],
            receipt_b['bbox']
        )
        
        # Compute cosine similarity and clamp to [0, 1]
        similarity = F.cosine_similarity(emb_a, emb_b, dim=1)
        similarity = torch.clamp(similarity, 0.0, 1.0)
        
        return similarity


def build_siamese_model(config: SiameseConfig) -> SiameseSimilarityModel:
    """
    Build Siamese model from configuration.
    
    Args:
        config: SiameseConfig instance
        
    Returns:
        SiameseSimilarityModel instance
    """
    print(f"Building Siamese model with config:")
    print(f"  Model path: {config.model_path}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Projection dim: {config.projection_dim}")
    
    # Create model
    model = SiameseSimilarityModel(
        model_path=config.model_path,
        dropout=config.dropout,
        projection_dim=config.projection_dim
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def save_siamese_checkpoint(
    model: SiameseSimilarityModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    similarity_threshold: float,
    path: str
) -> None:
    """
    Save model checkpoint with all necessary information.
    
    Args:
        model: SiameseSimilarityModel instance
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        similarity_threshold: Similarity threshold for decision making
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'similarity_threshold': similarity_threshold,
        'model_config': {
            'model_path': model.layoutlm.config.name_or_path,
            'projection_dim': model.projection_dim
        }
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_siamese_checkpoint(
    path: str,
    model: Optional[SiameseSimilarityModel] = None,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> dict:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into (if None, creates new model)
        optimizer: Optimizer to load state into (optional)
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    # Create model if not provided
    if model is None:
        model_config = checkpoint['model_config']
        model = SiameseSimilarityModel(
            model_path=model_config['model_path'],
            projection_dim=model_config['projection_dim']
        )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")
    print(f"  Similarity threshold: {checkpoint['similarity_threshold']:.4f}")
    
    return {
        'model': model,
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'similarity_threshold': checkpoint['similarity_threshold'],
        'checkpoint': checkpoint
    }


if __name__ == "__main__":
    """Test the Siamese model with dummy data."""
    
    print("=" * 60)
    print("Testing Siamese Model")
    print("=" * 60)
    
    try:
        # Create dummy config
        config = SiameseConfig(
            model_path="microsoft/layoutlm-base-uncased",
            dropout=0.1,
            projection_dim=256
        )
        
        print("\n1. Testing SiameseConfig:")
        print(f"   Config: {config}")
        
        # Test config save/load
        config_path = "/tmp/test_config.json"
        config.save(config_path)
        loaded_config = SiameseConfig.load(config_path)
        print(f"   Loaded config matches: {config == loaded_config}")
        
        # Build model
        print("\n2. Building model:")
        model = build_siamese_model(config)
        
        # Create dummy receipt data
        batch_size = 2
        seq_len = 512
        dummy_receipt = {
            'input_ids': torch.randint(0, 30000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'token_type_ids': torch.zeros(batch_size, seq_len, dtype=torch.long),
            'bbox': torch.randint(0, 1000, (batch_size, seq_len, 4))
        }
        
        print(f"\n3. Testing forward pass:")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {seq_len}")
        
        # Test forward pass without labels
        loss, logits, similarity_scores = model(dummy_receipt, dummy_receipt)
        
        print(f"   Loss (no labels): {loss}")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Logits: {logits}")
        print(f"   Similarity scores shape: {similarity_scores.shape}")
        print(f"   Similarity scores: {similarity_scores}")
        
        # Test forward pass with labels
        labels = torch.tensor([0, 1])  # legit, fraud
        loss, logits, similarity_scores = model(dummy_receipt, dummy_receipt, labels)
        
        print(f"\n   Loss (with labels): {loss.item():.6f}")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Similarity scores shape: {similarity_scores.shape}")
        
        # Test get_similarity method
        print(f"\n4. Testing get_similarity method:")
        similarity = model.get_similarity(dummy_receipt, dummy_receipt)
        print(f"   Similarity shape: {similarity.shape}")
        print(f"   Similarity values: {similarity}")
        
        # Test checkpoint save/load
        print(f"\n5. Testing checkpoint save/load:")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        checkpoint_path = "/tmp/test_checkpoint.pth"
        save_siamese_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            loss=loss.item(),
            similarity_threshold=0.5,
            path=checkpoint_path
        )
        
        # Load checkpoint
        loaded_info = load_siamese_checkpoint(checkpoint_path)
        print(f"   Successfully loaded checkpoint")
        
        # Test inference with loaded model
        with torch.no_grad():
            loss2, logits2, sim2 = loaded_info['model'](dummy_receipt, dummy_receipt, labels)
            print(f"   Loss from loaded model: {loss2.item():.6f}")
        
        # Cleanup
        import os
        os.remove(config_path)
        os.remove(checkpoint_path)
        
    except Exception as e:
        print(f"\n   ⚠️  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)
