"""
Comprehensive Input Validation
Validates all inputs for security, safety, and robustness
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import magic  # python-magic for file type detection
from PIL import Image
import numpy as np
from loguru import logger


class ValidationError(Exception):
    """Custom validation error"""
    pass


class InputValidator:
    """
    Comprehensive input validation
    Validates all user inputs to prevent injection attacks and ensure data integrity
    """

    @staticmethod
    def validate_string(
        value: str,
        min_length: int = 1,
        max_length: int = 1000,
        pattern: Optional[str] = None,
        allowed_chars: Optional[str] = None
    ) -> str:
        """
        Validate string input

        Args:
            value: String to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            pattern: Optional regex pattern to match
            allowed_chars: Optional string of allowed characters

        Returns:
            Validated string

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value)}")

        if len(value) < min_length:
            raise ValidationError(f"String too short (min: {min_length})")

        if len(value) > max_length:
            raise ValidationError(f"String too long (max: {max_length})")

        if pattern and not re.match(pattern, value):
            raise ValidationError(f"String does not match pattern: {pattern}")

        if allowed_chars:
            invalid_chars = set(value) - set(allowed_chars)
            if invalid_chars:
                raise ValidationError(f"Invalid characters: {invalid_chars}")

        return value

    @staticmethod
    def validate_integer(
        value: Union[int, str],
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> int:
        """
        Validate integer input

        Args:
            value: Integer to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated integer

        Raises:
            ValidationError: If validation fails
        """
        try:
            int_value = int(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid integer: {value}") from e

        if min_value is not None and int_value < min_value:
            raise ValidationError(f"Value {int_value} below minimum {min_value}")

        if max_value is not None and int_value > max_value:
            raise ValidationError(f"Value {int_value} above maximum {max_value}")

        return int_value

    @staticmethod
    def validate_float(
        value: Union[float, str],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False
    ) -> float:
        """
        Validate float input

        Args:
            value: Float to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_nan: Allow NaN values
            allow_inf: Allow infinite values

        Returns:
            Validated float

        Raises:
            ValidationError: If validation fails
        """
        try:
            float_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid float: {value}") from e

        if not allow_nan and np.isnan(float_value):
            raise ValidationError("NaN values not allowed")

        if not allow_inf and np.isinf(float_value):
            raise ValidationError("Infinite values not allowed")

        if min_value is not None and float_value < min_value:
            raise ValidationError(f"Value {float_value} below minimum {min_value}")

        if max_value is not None and float_value > max_value:
            raise ValidationError(f"Value {float_value} above maximum {max_value}")

        return float_value

    @staticmethod
    def validate_choice(value: Any, allowed_values: List[Any]) -> Any:
        """
        Validate that value is in allowed list

        Args:
            value: Value to validate
            allowed_values: List of allowed values

        Returns:
            Validated value

        Raises:
            ValidationError: If value not in allowed list
        """
        if value not in allowed_values:
            raise ValidationError(
                f"Invalid value: {value}. Must be one of {allowed_values}"
            )
        return value

    @staticmethod
    def validate_list(
        value: List[Any],
        min_length: int = 0,
        max_length: Optional[int] = None,
        item_type: Optional[type] = None
    ) -> List[Any]:
        """
        Validate list input

        Args:
            value: List to validate
            min_length: Minimum number of items
            max_length: Maximum number of items
            item_type: Required type for all items

        Returns:
            Validated list

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, list):
            raise ValidationError(f"Expected list, got {type(value)}")

        if len(value) < min_length:
            raise ValidationError(f"List too short (min: {min_length})")

        if max_length is not None and len(value) > max_length:
            raise ValidationError(f"List too long (max: {max_length})")

        if item_type is not None:
            for i, item in enumerate(value):
                if not isinstance(item, item_type):
                    raise ValidationError(
                        f"Item {i} has wrong type: {type(item)} (expected {item_type})"
                    )

        return value


class FileValidator:
    """
    File validation for security and safety
    Prevents path traversal, validates file types, checks sizes
    """

    # Maximum file sizes (in bytes)
    MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100 MB
    MAX_MODEL_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB

    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    ALLOWED_MODEL_EXTENSIONS = {'.h5', '.hdf5', '.pt', '.pth', '.onnx', '.pb'}

    @staticmethod
    def validate_file_exists(path: Union[str, Path]) -> Path:
        """
        Validate that file exists

        Args:
            path: File path to validate

        Returns:
            Validated Path object

        Raises:
            ValidationError: If file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise ValidationError(f"File not found: {path}")
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        return path

    @staticmethod
    def validate_file_size(
        path: Union[str, Path],
        max_size: int
    ) -> Path:
        """
        Validate file size

        Args:
            path: File path to validate
            max_size: Maximum allowed size in bytes

        Returns:
            Validated Path object

        Raises:
            ValidationError: If file too large
        """
        path = Path(path)
        size = path.stat().st_size
        if size > max_size:
            raise ValidationError(
                f"File too large: {size} bytes (max: {max_size})"
            )
        return path

    @staticmethod
    def validate_file_extension(
        path: Union[str, Path],
        allowed_extensions: set
    ) -> Path:
        """
        Validate file extension

        Args:
            path: File path to validate
            allowed_extensions: Set of allowed extensions (with dots)

        Returns:
            Validated Path object

        Raises:
            ValidationError: If extension not allowed
        """
        path = Path(path)
        ext = path.suffix.lower()
        if ext not in allowed_extensions:
            raise ValidationError(
                f"Invalid file extension: {ext}. Allowed: {allowed_extensions}"
            )
        return path

    @staticmethod
    def validate_image_file(path: Union[str, Path]) -> Path:
        """
        Comprehensive image file validation

        Args:
            path: Image file path

        Returns:
            Validated Path object

        Raises:
            ValidationError: If validation fails
        """
        path = Path(path)

        # Check existence
        FileValidator.validate_file_exists(path)

        # Check size
        FileValidator.validate_file_size(path, FileValidator.MAX_IMAGE_SIZE)

        # Check extension
        FileValidator.validate_file_extension(
            path,
            FileValidator.ALLOWED_IMAGE_EXTENSIONS
        )

        # Validate actual image content
        try:
            with Image.open(path) as img:
                img.verify()  # Verify it's a valid image
        except Exception as e:
            raise ValidationError(f"Invalid image file: {e}") from e

        return path

    @staticmethod
    def validate_model_file(path: Union[str, Path]) -> Path:
        """
        Validate model file

        Args:
            path: Model file path

        Returns:
            Validated Path object

        Raises:
            ValidationError: If validation fails
        """
        path = Path(path)

        # Check existence
        FileValidator.validate_file_exists(path)

        # Check size
        FileValidator.validate_file_size(path, FileValidator.MAX_MODEL_SIZE)

        # Check extension
        FileValidator.validate_file_extension(
            path,
            FileValidator.ALLOWED_MODEL_EXTENSIONS
        )

        return path


class PathValidator:
    """
    Path validation to prevent directory traversal attacks
    """

    @staticmethod
    def validate_path(
        path: Union[str, Path],
        base_dir: Optional[Union[str, Path]] = None,
        must_exist: bool = False,
        allow_creation: bool = True
    ) -> Path:
        """
        Validate and sanitize path

        Args:
            path: Path to validate
            base_dir: Optional base directory to restrict path to
            must_exist: Whether path must already exist
            allow_creation: Whether to allow creating the path

        Returns:
            Validated absolute Path object

        Raises:
            ValidationError: If validation fails
        """
        # Convert to Path
        path = Path(path)

        # Resolve to absolute path
        try:
            path = path.resolve()
        except Exception as e:
            raise ValidationError(f"Invalid path: {e}") from e

        # Check for path traversal
        if base_dir:
            base_dir = Path(base_dir).resolve()
            try:
                path.relative_to(base_dir)
            except ValueError:
                raise ValidationError(
                    f"Path {path} is outside base directory {base_dir}"
                )

        # Check existence
        if must_exist and not path.exists():
            raise ValidationError(f"Path does not exist: {path}")

        # Check if we can create parent directories
        if not path.exists() and allow_creation:
            parent = path.parent
            if not parent.exists():
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValidationError(
                        f"Cannot create directory {parent}: {e}"
                    ) from e

        return path

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove directory components
        filename = os.path.basename(filename)

        # Remove dangerous characters
        filename = re.sub(r'[^\w\s\-\.]', '', filename)

        # Remove leading dots
        filename = filename.lstrip('.')

        # Ensure not empty
        if not filename:
            filename = 'unnamed_file'

        return filename


class ImageValidator:
    """
    Validate image data and dimensions
    """

    @staticmethod
    def validate_image_array(
        image: np.ndarray,
        min_height: int = 32,
        min_width: int = 32,
        max_height: int = 10000,
        max_width: int = 10000,
        allowed_channels: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Validate numpy image array

        Args:
            image: Image array to validate
            min_height: Minimum allowed height
            min_width: Minimum allowed width
            max_height: Maximum allowed height
            max_width: Maximum allowed width
            allowed_channels: List of allowed channel counts

        Returns:
            Validated image array

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(image, np.ndarray):
            raise ValidationError(f"Expected numpy array, got {type(image)}")

        if image.ndim not in [2, 3]:
            raise ValidationError(
                f"Image must be 2D or 3D, got {image.ndim}D"
            )

        height, width = image.shape[:2]

        if height < min_height or width < min_width:
            raise ValidationError(
                f"Image too small: {height}x{width} "
                f"(min: {min_height}x{min_width})"
            )

        if height > max_height or width > max_width:
            raise ValidationError(
                f"Image too large: {height}x{width} "
                f"(max: {max_height}x{max_width})"
            )

        if image.ndim == 3:
            channels = image.shape[2]
            if allowed_channels and channels not in allowed_channels:
                raise ValidationError(
                    f"Invalid channel count: {channels} "
                    f"(allowed: {allowed_channels})"
                )

        # Check for NaN or Inf
        if np.isnan(image).any():
            raise ValidationError("Image contains NaN values")

        if np.isinf(image).any():
            raise ValidationError("Image contains infinite values")

        return image

    @staticmethod
    def validate_image_values(
        image: np.ndarray,
        expected_range: tuple = (0, 255)
    ) -> np.ndarray:
        """
        Validate image value range

        Args:
            image: Image array
            expected_range: Expected min/max values

        Returns:
            Validated image array

        Raises:
            ValidationError: If values outside expected range
        """
        min_val, max_val = expected_range

        if image.min() < min_val or image.max() > max_val:
            raise ValidationError(
                f"Image values outside expected range {expected_range}. "
                f"Got: [{image.min()}, {image.max()}]"
            )

        return image


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary

    Args:
        config: Configuration to validate

    Returns:
        Validated configuration

    Raises:
        ValidationError: If validation fails
    """
    required_keys = ['model_path', 'data_path']

    for key in required_keys:
        if key not in config:
            raise ValidationError(f"Missing required config key: {key}")

    # Validate specific fields
    if 'epochs' in config:
        config['epochs'] = InputValidator.validate_integer(
            config['epochs'],
            min_value=1,
            max_value=10000
        )

    if 'batch_size' in config:
        config['batch_size'] = InputValidator.validate_integer(
            config['batch_size'],
            min_value=1,
            max_value=1024
        )

    if 'learning_rate' in config:
        config['learning_rate'] = InputValidator.validate_float(
            config['learning_rate'],
            min_value=1e-10,
            max_value=1.0
        )

    return config
