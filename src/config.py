"""Configuration module for ReceiptGuard-ML.

Loads config.yaml from project root and provides centralized access to configuration.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigDict:
    """Dictionary that allows attribute access to nested dictionaries."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to regular dictionary."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


class Config:
    """Configuration loader and accessor."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to config.yaml. If None, looks in project root.
        """
        if config_path is None:
            # Get project root (3 levels up from this file: src/config.py -> src -> project_root)
            self.project_root = Path(__file__).parent.parent
            config_path = self.project_root / "config.yaml"
        else:
            config_path = Path(config_path)
            self.project_root = config_path.parent
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def __getattr__(self, name: str) -> Any:
        """Access configuration sections as attributes."""
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'paths.artifacts_dir')."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def resolve_path(self, path_str: Union[str, Path]) -> Path:
        """Resolve a path string to an absolute Path object.
        
        If the path is already absolute, returns it as a Path object.
        If it's relative, resolves it relative to the project root.
        
        Args:
            path_str: Path string or Path object
            
        Returns:
            Absolute Path object
        """
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()
    
    def get_path(self, key: str, create_dir: bool = True) -> Path:
        """Get absolute path from configuration and optionally create directory.
        
        Args:
            key: Configuration key (e.g., 'paths.artifacts_dir')
            create_dir: If True, creates the directory if it doesn't exist
            
        Returns:
            Absolute Path object
        """
        path_str = self.get(key)
        if path_str is None:
            raise KeyError(f"Path key not found: {key}")
        
        path = self.resolve_path(path_str)
        
        # Create directory if requested and it's a directory path
        if create_dir and not key.endswith(('.json', '.pt', '.txt', '.yaml', '.yml')):
            path.mkdir(parents=True, exist_ok=True)
        
        return path
    
    def override(self, overrides: Dict[str, Any]) -> None:
        """Override configuration values at runtime.
        
        Useful for Kaggle notebooks or runtime modifications.
        
        Args:
            overrides: Dictionary of overrides using dot notation keys
                      e.g., {'training.batch_size': 16, 'paths.raw_data_dir': '/kaggle/input/sroie2019'}
        """
        for key, value in overrides.items():
            keys = key.split('.')
            config_section = self._config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config_section:
                    config_section[k] = {}
                config_section = config_section[k]
            
            # Set the final value
            config_section[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config.copy()
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()


# Create global configuration instance
CFG = Config()


def get_path(key: str, create_dir: bool = True) -> Path:
    """Convenience function to get path from global CFG."""
    return CFG.get_path(key, create_dir)


def override_config(overrides: Dict[str, Any]) -> None:
    """Convenience function to override global CFG."""
    CFG.override(overrides)


if __name__ == "__main__":
    """Print full loaded configuration when run as script."""
    import json
    
    print("ReceiptGuard-ML Configuration")
    print("=" * 40)
    print(f"Config file: {CFG.config_path}")
    print(f"Project root: {CFG.project_root}")
    print()
    
    # Pretty print the entire configuration
    print("Full Configuration:")
    print(json.dumps(CFG.to_dict(), indent=2, default=str))
