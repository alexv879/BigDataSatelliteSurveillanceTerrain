"""
Input Sanitization
Sanitizes user inputs to prevent injection attacks
"""
import re
import html
from pathlib import Path
from typing import Any, Dict, List, Union


def sanitize_string(
    value: str,
    max_length: int = 1000,
    remove_html: bool = True,
    remove_sql: bool = True
) -> str:
    """
    Sanitize string input

    Args:
        value: String to sanitize
        max_length: Maximum length
        remove_html: Remove HTML tags
        remove_sql: Remove SQL injection attempts

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return str(value)

    # Truncate
    value = value[:max_length]

    # Remove null bytes
    value = value.replace('\x00', '')

    # HTML escape
    if remove_html:
        value = html.escape(value)

    # Remove SQL injection attempts
    if remove_sql:
        # Remove common SQL keywords
        sql_keywords = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE',
            'ALTER', 'EXEC', 'EXECUTE', 'UNION', 'SCRIPT', '--', ';--'
        ]
        for keyword in sql_keywords:
            value = re.sub(
                f'\\b{keyword}\\b',
                '',
                value,
                flags=re.IGNORECASE
            )

    return value.strip()


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename to prevent directory traversal and injection

    Args:
        filename: Original filename
        max_length: Maximum filename length

    Returns:
        Sanitized filename
    """
    # Get base name only (remove directory components)
    filename = Path(filename).name

    # Remove dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    # Replace multiple dots with single dot
    filename = re.sub(r'\.+', '.', filename)

    # Replace multiple spaces/hyphens with single
    filename = re.sub(r'\s+', ' ', filename)
    filename = re.sub(r'\-+', '-', filename)

    # Truncate
    filename = filename[:max_length]

    # Ensure not empty
    if not filename:
        filename = 'unnamed_file'

    return filename


def sanitize_path(
    path: Union[str, Path],
    base_dir: Union[str, Path],
    allow_creation: bool = False
) -> Path:
    """
    Sanitize file path to prevent directory traversal

    Args:
        path: Path to sanitize
        base_dir: Base directory to restrict to
        allow_creation: Allow creating the path

    Returns:
        Sanitized absolute Path

    Raises:
        ValueError: If path escapes base directory
    """
    # Convert to Path objects
    path = Path(path)
    base_dir = Path(base_dir).resolve()

    # Resolve to absolute path
    try:
        resolved = (base_dir / path).resolve()
    except Exception as e:
        raise ValueError(f"Invalid path: {e}")

    # Ensure path is within base directory
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        raise ValueError(
            f"Path {path} attempts to escape base directory {base_dir}"
        )

    # Check if we should create parent directories
    if allow_creation and not resolved.parent.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)

    return resolved


def sanitize_input(value: Any) -> Any:
    """
    General purpose input sanitization

    Args:
        value: Value to sanitize

    Returns:
        Sanitized value
    """
    if isinstance(value, str):
        return sanitize_string(value)
    elif isinstance(value, list):
        return [sanitize_input(item) for item in value]
    elif isinstance(value, dict):
        return {
            sanitize_input(k): sanitize_input(v)
            for k, v in value.items()
        }
    else:
        return value


def sanitize_dict(data: Dict[str, Any], allowed_keys: List[str]) -> Dict[str, Any]:
    """
    Sanitize dictionary by filtering allowed keys

    Args:
        data: Dictionary to sanitize
        allowed_keys: List of allowed keys

    Returns:
        Sanitized dictionary with only allowed keys
    """
    return {
        k: sanitize_input(v)
        for k, v in data.items()
        if k in allowed_keys
    }


def remove_control_characters(value: str) -> str:
    """
    Remove control characters from string

    Args:
        value: String to clean

    Returns:
        String without control characters
    """
    # Remove all control characters except newline, tab, carriage return
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)


def sanitize_url(url: str) -> str:
    """
    Sanitize URL to prevent SSRF and injection

    Args:
        url: URL to sanitize

    Returns:
        Sanitized URL

    Raises:
        ValueError: If URL is unsafe
    """
    # Remove whitespace
    url = url.strip()

    # Check for dangerous protocols
    dangerous_protocols = ['file://', 'ftp://', 'gopher://', 'dict://', 'jar://']
    for protocol in dangerous_protocols:
        if url.lower().startswith(protocol):
            raise ValueError(f"Unsafe protocol: {protocol}")

    # Only allow http and https
    if not (url.startswith('http://') or url.startswith('https://')):
        raise ValueError("URL must use http:// or https://")

    # Check for localhost/internal IPs
    internal_patterns = [
        r'localhost',
        r'127\.0\.0\.1',
        r'0\.0\.0\.0',
        r'192\.168\.',
        r'10\.',
        r'172\.(1[6-9]|2[0-9]|3[01])\.'
    ]

    for pattern in internal_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            raise ValueError("Cannot access internal/localhost URLs")

    return url
