"""
Configuration Management
Load and validate configuration from environment variables and files
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import yaml
import json
from loguru import logger


@dataclass
class Config:
    """
    Application Configuration
    Loads configuration from environment variables with defaults
    """

    # Application
    APP_NAME: str = "Satellite Terrain Classifier"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))

    # Security
    SECRET_KEY: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "change-me-in-production"))
    ENABLE_AUTH: bool = field(default_factory=lambda: os.getenv("ENABLE_AUTH", "true").lower() == "true")
    API_KEY_HEADER: str = field(default_factory=lambda: os.getenv("API_KEY_HEADER", "X-API-Key"))

    # CORS
    ALLOWED_ORIGINS: list = field(default_factory=lambda: os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8080"
    ).split(","))
    ALLOWED_HOSTS: list = field(default_factory=lambda: os.getenv(
        "ALLOWED_HOSTS",
        "localhost,127.0.0.1"
    ).split(","))

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")))
    RATE_LIMIT_PER_HOUR: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_HOUR", "1000")))
    RATE_LIMIT_PER_DAY: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_DAY", "10000")))

    # File Upload
    MAX_UPLOAD_SIZE: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE", str(100 * 1024 * 1024))))
    MAX_BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "10")))
    ALLOWED_EXTENSIONS: list = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".tif", ".tiff"])

    # Model
    MODEL_PATH: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "models/best_model.h5"))
    INPUT_SHAPE: tuple = field(default_factory=lambda: (224, 224, 3))
    NUM_CLASSES: int = field(default_factory=lambda: int(os.getenv("NUM_CLASSES", "10")))

    # Database (if needed)
    DATABASE_URL: Optional[str] = field(default_factory=lambda: os.getenv("DATABASE_URL"))

    # Redis (for rate limiting, caching)
    REDIS_URL: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_URL"))

    # Logging
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    LOG_FILE: str = field(default_factory=lambda: os.getenv("LOG_FILE", "logs/app.log"))
    LOG_ROTATION: str = field(default_factory=lambda: os.getenv("LOG_ROTATION", "500 MB"))
    LOG_RETENTION: str = field(default_factory=lambda: os.getenv("LOG_RETENTION", "10 days"))

    # API Server
    HOST: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    WORKERS: int = field(default_factory=lambda: int(os.getenv("WORKERS", "1")))
    RELOAD: bool = field(default_factory=lambda: os.getenv("RELOAD", "false").lower() == "true")

    # Training
    BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "32")))
    EPOCHS: int = field(default_factory=lambda: int(os.getenv("EPOCHS", "50")))
    LEARNING_RATE: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "0.001")))

    # GPU
    GPU_MEMORY_FRACTION: float = field(default_factory=lambda: float(os.getenv("GPU_MEMORY_FRACTION", "0.9")))
    ALLOW_GPU_GROWTH: bool = field(default_factory=lambda: os.getenv("ALLOW_GPU_GROWTH", "true").lower() == "true")

    # Timeouts
    REQUEST_TIMEOUT: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "30")))
    MODEL_LOAD_TIMEOUT: int = field(default_factory=lambda: int(os.getenv("MODEL_LOAD_TIMEOUT", "60")))

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate()

    def _validate(self):
        """Validate configuration values"""
        # Validate environment
        valid_environments = ["development", "staging", "production"]
        if self.ENVIRONMENT not in valid_environments:
            raise ValueError(
                f"Invalid ENVIRONMENT: {self.ENVIRONMENT}. "
                f"Must be one of {valid_environments}"
            )

        # Validate secret key in production
        if self.ENVIRONMENT == "production" and self.SECRET_KEY == "change-me-in-production":
            raise ValueError(
                "SECRET_KEY must be changed in production environment"
            )

        # Validate rate limits
        if self.RATE_LIMIT_PER_MINUTE < 1:
            raise ValueError("RATE_LIMIT_PER_MINUTE must be at least 1")

        if self.RATE_LIMIT_PER_HOUR < self.RATE_LIMIT_PER_MINUTE:
            raise ValueError(
                "RATE_LIMIT_PER_HOUR must be >= RATE_LIMIT_PER_MINUTE"
            )

        if self.RATE_LIMIT_PER_DAY < self.RATE_LIMIT_PER_HOUR:
            raise ValueError(
                "RATE_LIMIT_PER_DAY must be >= RATE_LIMIT_PER_HOUR"
            )

        # Validate file size
        if self.MAX_UPLOAD_SIZE < 1:
            raise ValueError("MAX_UPLOAD_SIZE must be at least 1 byte")

        if self.MAX_UPLOAD_SIZE > 1024 * 1024 * 1024:  # 1 GB
            logger.warning(
                f"MAX_UPLOAD_SIZE is very large: {self.MAX_UPLOAD_SIZE} bytes"
            )

        # Validate training parameters
        if self.BATCH_SIZE < 1:
            raise ValueError("BATCH_SIZE must be at least 1")

        if self.EPOCHS < 1:
            raise ValueError("EPOCHS must be at least 1")

        if self.LEARNING_RATE <= 0 or self.LEARNING_RATE > 1:
            raise ValueError("LEARNING_RATE must be between 0 and 1")

        # Validate GPU settings
        if self.GPU_MEMORY_FRACTION <= 0 or self.GPU_MEMORY_FRACTION > 1:
            raise ValueError("GPU_MEMORY_FRACTION must be between 0 and 1")

        logger.debug("Configuration validated successfully")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

    def save(self, path: Path, format: str = "yaml"):
        """
        Save configuration to file

        Args:
            path: Path to save configuration
            format: Format (yaml or json)
        """
        config_dict = self.to_dict()

        # Redact sensitive information
        sensitive_keys = ['SECRET_KEY', 'DATABASE_URL', 'REDIS_URL']
        for key in sensitive_keys:
            if key in config_dict and config_dict[key]:
                config_dict[key] = "***REDACTED***"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "yaml":
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif format == "json":
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Configuration saved to {path}")


def load_config(config_file: Optional[Path] = None) -> Config:
    """
    Load configuration from file and environment variables

    Args:
        config_file: Optional path to configuration file

    Returns:
        Config object
    """
    config_dict = {}

    # Load from file if provided
    if config_file and Path(config_file).exists():
        with open(config_file) as f:
            if config_file.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_file.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_file.suffix}")

        logger.info(f"Loaded configuration from {config_file}")

    # Environment variables override file configuration
    # (Config class already loads from environment in defaults)
    config = Config(**config_dict)

    return config


def validate_config(config: Config) -> bool:
    """
    Validate configuration

    Args:
        config: Config object to validate

    Returns:
        True if valid

    Raises:
        ValueError: If configuration invalid
    """
    config._validate()
    return True


def get_config() -> Config:
    """
    Get configuration (singleton pattern)

    Returns:
        Config object
    """
    if not hasattr(get_config, "_instance"):
        get_config._instance = load_config()
    return get_config._instance
