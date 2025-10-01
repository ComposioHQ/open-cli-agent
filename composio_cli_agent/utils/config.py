"""Configuration utilities for Composio CLI Agent"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import json

# Configuration constants
DEFAULT_MODEL = "gpt-5"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_USER_ID = "default"
CONFIG_DIR_NAME = ".composio-cli"
CONFIG_FILE_NAME = "config.json"
ENV_FILE_NAME = ".env"


class Config:
    """Configuration manager for the CLI agent"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / CONFIG_DIR_NAME
        self.config_file = self.config_dir / CONFIG_FILE_NAME
        self.env_file = self.config_dir / ENV_FILE_NAME
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(exist_ok=True)
        
        # Load environment variables
        load_dotenv(self.env_file)
        
        # Load configuration
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Return default config
        return {
            "default_user_id": DEFAULT_USER_ID,
            "default_model": DEFAULT_MODEL,
            "default_temperature": DEFAULT_TEMPERATURE,
            "verbose": False,
        }
    
    def _save_config(self) -> None:
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self._config[key] = value
        self._save_config()
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key from environment or config"""
        env_var = f"{service.upper()}_API_KEY"
        return os.getenv(env_var) or self._config.get(f"{service}_api_key")
    
    def set_api_key(self, service: str, api_key: str) -> None:
        """Set API key in config"""
        key = f"{service}_api_key"
        self._config[key] = api_key
        self._save_config()
    
    @property
    def composio_api_key(self) -> Optional[str]:
        """Get Composio API key"""
        return self.get_api_key("composio")
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key"""
        return self.get_api_key("openai")
    
    @property
    def default_user_id(self) -> str:
        """Get default user ID"""
        return self.get("default_user_id", DEFAULT_USER_ID)
    
    @property
    def default_model(self) -> str:
        """Get default model"""
        return self.get("default_model", DEFAULT_MODEL)
    
    @property
    def default_temperature(self) -> float:
        """Get default temperature"""
        return self.get("default_temperature", DEFAULT_TEMPERATURE)
    
    @property
    def verbose(self) -> bool:
        """Get verbose setting"""
        return self.get("verbose", False)


# Global config instance
config = Config()