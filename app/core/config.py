import os
import yaml
from typing import Dict, Any
from pathlib import Path

class Settings:
    """Configuration management class"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Return default configuration if file not found
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration values"""
        return {
            "DATABASE_URL": "sqlite:///./internship_allocation.db",
            "SECRET_KEY": "fallback-secret-key",
            "EMBEDDING_MODEL": "all-mpnet-base-v2",
            "TOP_K_SIMILARITY": 50,
            "TOP_K_OPTIMIZER": 30,
            "RURAL_QUOTA": 0.15,
            "API_HOST": "0.0.0.0",
            "API_PORT": 8000,
            "LOG_LEVEL": "INFO"
        }
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str):
        """Dictionary-like access"""
        return self.config[key]
    
    def __contains__(self, key: str):
        """Check if key exists"""
        return key in self.config

# Global settings instance
settings = Settings()
