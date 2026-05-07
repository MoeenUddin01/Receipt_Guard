"""Configuration module for ReceiptGuard-ML.

This is a convenience wrapper that imports from src/config.py
to support both direct imports and src.config imports.
"""

from src.config import CFG, override_config, get_path, Config, ConfigDict

__all__ = ['CFG', 'override_config', 'get_path', 'Config', 'ConfigDict']
