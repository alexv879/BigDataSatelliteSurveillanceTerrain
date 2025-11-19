"""
Unit Tests for Input Sanitization
Tests sanitization functions for security
"""
import pytest
from pathlib import Path
import tempfile

from security.sanitization import (
    sanitize_string,
    sanitize_filename,
    sanitize_path,
    sanitize_input,
    sanitize_dict,
    remove_control_characters,
    sanitize_url
)


class TestSanitizeString:
    """Test string sanitization"""

    def test_basic_sanitization(self):
        """Test basic string sanitization"""
        result = sanitize_string("hello world")
        assert result == "hello world"

    def test_html_escape(self):
        """Test HTML tag removal"""
        result = sanitize_string("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_sql_injection_prevention(self):
        """Test SQL injection attempt removal"""
        result = sanitize_string("test'; DROP TABLE users; --")
        assert "DROP" not in result
        assert "TABLE" not in result

    def test_null_byte_removal(self):
        """Test null byte removal"""
        result = sanitize_string("test\x00malicious")
        assert "\x00" not in result

    def test_max_length_truncation(self):
        """Test string truncation"""
        long_string = "x" * 2000
        result = sanitize_string(long_string, max_length=100)
        assert len(result) == 100


class TestSanitizeFilename:
    """Test filename sanitization"""

    def test_basic_filename(self):
        """Test basic filename sanitization"""
        result = sanitize_filename("document.pdf")
        assert result == "document.pdf"

    def test_path_traversal_prevention(self):
        """Test path traversal attempt"""
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
        assert "\\" not in result

    def test_dangerous_characters(self):
        """Test removal of dangerous characters"""
        result = sanitize_filename("file<>:|?.txt")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result

    def test_leading_dots(self):
        """Test leading dot removal"""
        result = sanitize_filename("...hidden_file.txt")
        assert not result.startswith(".")

    def test_empty_filename(self):
        """Test handling of empty filename"""
        result = sanitize_filename("...")
        assert result == "unnamed_file"

    def test_multiple_spaces(self):
        """Test multiple space normalization"""
        result = sanitize_filename("file   with    spaces.txt")
        assert "   " not in result


class TestSanitizePath:
    """Test path sanitization"""

    def test_valid_path(self):
        """Test valid path sanitization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            result = sanitize_path("subdir/file.txt", base_dir=base)
            assert str(result).startswith(str(base))

    def test_path_traversal_prevention(self):
        """Test path traversal prevention"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            with pytest.raises(ValueError, match="escape"):
                sanitize_path("../../../etc/passwd", base_dir=base)

    def test_absolute_path_normalization(self):
        """Test absolute path handling"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            result = sanitize_path("file.txt", base_dir=base)
            assert result.is_absolute()


class TestSanitizeInput:
    """Test general input sanitization"""

    def test_sanitize_string_input(self):
        """Test string input sanitization"""
        result = sanitize_input("<script>alert('xss')</script>")
        assert "<script>" not in result

    def test_sanitize_list_input(self):
        """Test list input sanitization"""
        input_list = ["<b>test</b>", "normal", "'; DROP TABLE;"]
        result = sanitize_input(input_list)
        assert "<b>" not in result[0]
        assert result[1] == "normal"
        assert "DROP" not in result[2]

    def test_sanitize_dict_input(self):
        """Test dictionary input sanitization"""
        input_dict = {
            "<key>": "<value>",
            "normal": "text"
        }
        result = sanitize_input(input_dict)
        assert all("<" not in str(v) for v in result.values())

    def test_sanitize_dict_with_allowed_keys(self):
        """Test dictionary sanitization with allowed keys"""
        data = {
            "name": "John<script>",
            "email": "test@example.com",
            "malicious": "'; DROP TABLE;"
        }
        result = sanitize_dict(data, allowed_keys=["name", "email"])
        assert "malicious" not in result
        assert "name" in result
        assert "email" in result
        assert "<script>" not in result["name"]


class TestRemoveControlCharacters:
    """Test control character removal"""

    def test_remove_control_chars(self):
        """Test control character removal"""
        text = "Hello\x00\x01\x02World"
        result = remove_control_characters(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert result == "HelloWorld"

    def test_keep_allowed_chars(self):
        """Test keeping allowed control characters"""
        text = "Hello\nWorld\tTest\r"
        result = remove_control_characters(text)
        assert "\n" in result
        assert "\t" in result
        assert "\r" in result


class TestSanitizeURL:
    """Test URL sanitization"""

    def test_valid_http_url(self):
        """Test valid HTTP URL"""
        result = sanitize_url("http://example.com/image.jpg")
        assert result == "http://example.com/image.jpg"

    def test_valid_https_url(self):
        """Test valid HTTPS URL"""
        result = sanitize_url("https://example.com/image.jpg")
        assert result == "https://example.com/image.jpg"

    def test_dangerous_protocol_file(self):
        """Test file:// protocol rejection"""
        with pytest.raises(ValueError, match="Unsafe protocol"):
            sanitize_url("file:///etc/passwd")

    def test_dangerous_protocol_ftp(self):
        """Test ftp:// protocol rejection"""
        with pytest.raises(ValueError, match="Unsafe protocol"):
            sanitize_url("ftp://internal.server/file")

    def test_invalid_protocol(self):
        """Test invalid protocol rejection"""
        with pytest.raises(ValueError, match="must use http"):
            sanitize_url("javascript:alert('xss')")

    def test_localhost_rejection(self):
        """Test localhost URL rejection"""
        with pytest.raises(ValueError, match="Cannot access internal"):
            sanitize_url("http://localhost:8080/admin")

    def test_internal_ip_rejection(self):
        """Test internal IP rejection"""
        internal_ips = [
            "http://127.0.0.1/admin",
            "http://192.168.1.1/router",
            "http://10.0.0.1/internal",
            "http://172.16.0.1/network"
        ]
        for url in internal_ips:
            with pytest.raises(ValueError, match="Cannot access internal"):
                sanitize_url(url)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
