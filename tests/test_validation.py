"""
Unit Tests for Input Validation
Tests all validation functions for security and correctness
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from security.validation import (
    InputValidator,
    FileValidator,
    PathValidator,
    ImageValidator,
    ValidationError,
    validate_config
)


class TestInputValidator:
    """Test InputValidator class"""

    def test_validate_string_success(self):
        """Test valid string validation"""
        result = InputValidator.validate_string("test", min_length=1, max_length=10)
        assert result == "test"

    def test_validate_string_too_short(self):
        """Test string too short"""
        with pytest.raises(ValidationError, match="too short"):
            InputValidator.validate_string("x", min_length=5)

    def test_validate_string_too_long(self):
        """Test string too long"""
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_string("x" * 100, max_length=10)

    def test_validate_string_pattern(self):
        """Test string pattern matching"""
        # Valid pattern
        result = InputValidator.validate_string("test123", pattern=r'^[a-z0-9]+$')
        assert result == "test123"

        # Invalid pattern
        with pytest.raises(ValidationError, match="does not match pattern"):
            InputValidator.validate_string("test@123", pattern=r'^[a-z0-9]+$')

    def test_validate_integer_success(self):
        """Test valid integer validation"""
        assert InputValidator.validate_integer(42) == 42
        assert InputValidator.validate_integer("42") == 42

    def test_validate_integer_invalid(self):
        """Test invalid integer"""
        with pytest.raises(ValidationError, match="Invalid integer"):
            InputValidator.validate_integer("abc")

    def test_validate_integer_range(self):
        """Test integer range validation"""
        # Within range
        assert InputValidator.validate_integer(50, min_value=0, max_value=100) == 50

        # Below minimum
        with pytest.raises(ValidationError, match="below minimum"):
            InputValidator.validate_integer(-5, min_value=0)

        # Above maximum
        with pytest.raises(ValidationError, match="above maximum"):
            InputValidator.validate_integer(150, max_value=100)

    def test_validate_float_success(self):
        """Test valid float validation"""
        assert InputValidator.validate_float(3.14) == 3.14
        assert InputValidator.validate_float("3.14") == 3.14

    def test_validate_float_nan(self):
        """Test NaN handling"""
        # NaN not allowed by default
        with pytest.raises(ValidationError, match="NaN"):
            InputValidator.validate_float(float('nan'))

        # NaN allowed
        result = InputValidator.validate_float(float('nan'), allow_nan=True)
        assert np.isnan(result)

    def test_validate_float_inf(self):
        """Test infinity handling"""
        # Infinity not allowed by default
        with pytest.raises(ValidationError, match="Infinite"):
            InputValidator.validate_float(float('inf'))

        # Infinity allowed
        result = InputValidator.validate_float(float('inf'), allow_inf=True)
        assert np.isinf(result)

    def test_validate_choice(self):
        """Test choice validation"""
        allowed = ['swin', 'vit', 'resnet']

        # Valid choice
        assert InputValidator.validate_choice('swin', allowed) == 'swin'

        # Invalid choice
        with pytest.raises(ValidationError, match="Invalid value"):
            InputValidator.validate_choice('invalid', allowed)

    def test_validate_list(self):
        """Test list validation"""
        # Valid list
        result = InputValidator.validate_list([1, 2, 3], min_length=1, max_length=5)
        assert result == [1, 2, 3]

        # Too short
        with pytest.raises(ValidationError, match="too short"):
            InputValidator.validate_list([], min_length=1)

        # Too long
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_list([1] * 10, max_length=5)

        # Wrong item type
        with pytest.raises(ValidationError, match="wrong type"):
            InputValidator.validate_list([1, "2", 3], item_type=int)


class TestFileValidator:
    """Test FileValidator class"""

    def test_validate_file_exists(self):
        """Test file existence validation"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            # Valid file
            result = FileValidator.validate_file_exists(temp_path)
            assert result.exists()

            # Non-existent file
            with pytest.raises(ValidationError, match="not found"):
                FileValidator.validate_file_exists("/nonexistent/file.txt")
        finally:
            os.unlink(temp_path)

    def test_validate_file_size(self):
        """Test file size validation"""
        # Create temporary file with known size
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 1000)  # 1000 bytes
            temp_path = f.name

        try:
            # Within size limit
            result = FileValidator.validate_file_size(temp_path, 2000)
            assert result.exists()

            # Exceeds size limit
            with pytest.raises(ValidationError, match="too large"):
                FileValidator.validate_file_size(temp_path, 500)
        finally:
            os.unlink(temp_path)

    def test_validate_file_extension(self):
        """Test file extension validation"""
        # Valid extension
        path = Path("/tmp/test.jpg")
        result = FileValidator.validate_file_extension(
            path,
            {'.jpg', '.png'}
        )
        assert result.suffix == '.jpg'

        # Invalid extension
        with pytest.raises(ValidationError, match="Invalid file extension"):
            FileValidator.validate_file_extension(
                Path("/tmp/test.exe"),
                {'.jpg', '.png'}
            )


class TestPathValidator:
    """Test PathValidator class"""

    def test_validate_path_basic(self):
        """Test basic path validation"""
        # Valid path
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            result = PathValidator.validate_path(path)
            assert result.is_absolute()

    def test_validate_path_traversal(self):
        """Test path traversal prevention"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            # Valid path within base
            safe_path = base_dir / "subdir" / "file.txt"
            result = PathValidator.validate_path(safe_path, base_dir=base_dir)
            assert str(result).startswith(str(base_dir))

            # Path traversal attempt
            with pytest.raises(ValidationError, match="outside base directory"):
                PathValidator.validate_path(
                    base_dir / ".." / ".." / "etc" / "passwd",
                    base_dir=base_dir
                )

    def test_sanitize_filename(self):
        """Test filename sanitization"""
        # Dangerous filename
        result = PathValidator.sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
        assert result == "etcpasswd"

        # Special characters
        result = PathValidator.sanitize_filename("test<>:file.txt")
        assert result == "testfile.txt"

        # Empty result
        result = PathValidator.sanitize_filename("...")
        assert result == "unnamed_file"


class TestImageValidator:
    """Test ImageValidator class"""

    def test_validate_image_array_2d(self):
        """Test 2D image validation"""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = ImageValidator.validate_image_array(img)
        assert result.shape == (100, 100)

    def test_validate_image_array_3d(self):
        """Test 3D image validation"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = ImageValidator.validate_image_array(
            img,
            allowed_channels=[3]
        )
        assert result.shape == (100, 100, 3)

    def test_validate_image_array_wrong_channels(self):
        """Test invalid channel count"""
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValidationError, match="Invalid channel count"):
            ImageValidator.validate_image_array(
                img,
                allowed_channels=[1, 3]
            )

    def test_validate_image_array_too_small(self):
        """Test image too small"""
        img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        with pytest.raises(ValidationError, match="too small"):
            ImageValidator.validate_image_array(
                img,
                min_height=32,
                min_width=32
            )

    def test_validate_image_array_too_large(self):
        """Test image too large"""
        img = np.random.randint(0, 255, (20000, 20000), dtype=np.uint8)
        with pytest.raises(ValidationError, match="too large"):
            ImageValidator.validate_image_array(
                img,
                max_height=10000,
                max_width=10000
            )

    def test_validate_image_array_nan(self):
        """Test NaN detection"""
        img = np.array([[1.0, 2.0], [np.nan, 4.0]])
        with pytest.raises(ValidationError, match="NaN"):
            ImageValidator.validate_image_array(img)

    def test_validate_image_array_inf(self):
        """Test infinity detection"""
        img = np.array([[1.0, 2.0], [np.inf, 4.0]])
        with pytest.raises(ValidationError, match="infinite"):
            ImageValidator.validate_image_array(img)

    def test_validate_image_values(self):
        """Test image value range validation"""
        # Valid range
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = ImageValidator.validate_image_values(img, expected_range=(0, 255))
        assert result is not None

        # Out of range
        img = np.array([[0, 100], [300, 400]])
        with pytest.raises(ValidationError, match="outside expected range"):
            ImageValidator.validate_image_values(img, expected_range=(0, 255))


class TestConfigValidation:
    """Test configuration validation"""

    def test_validate_config_success(self):
        """Test valid configuration"""
        config = {
            'model_path': '/tmp/model.h5',
            'data_path': '/tmp/data',
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        result = validate_config(config)
        assert result['epochs'] == 100
        assert result['batch_size'] == 32
        assert result['learning_rate'] == 0.001

    def test_validate_config_missing_key(self):
        """Test missing required key"""
        config = {'model_path': '/tmp/model.h5'}
        with pytest.raises(ValidationError, match="Missing required"):
            validate_config(config)

    def test_validate_config_invalid_epochs(self):
        """Test invalid epochs value"""
        config = {
            'model_path': '/tmp/model.h5',
            'data_path': '/tmp/data',
            'epochs': -5
        }
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_validate_config_invalid_learning_rate(self):
        """Test invalid learning rate"""
        config = {
            'model_path': '/tmp/model.h5',
            'data_path': '/tmp/data',
            'learning_rate': 10.0  # Too high
        }
        with pytest.raises(ValidationError):
            validate_config(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
