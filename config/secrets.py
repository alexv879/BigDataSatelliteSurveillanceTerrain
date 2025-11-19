"""
Secrets Management
Secure handling of API keys, passwords, and other secrets
"""
import os
import json
import secrets
from pathlib import Path
from typing import Dict, Optional
from cryptography.fernet import Fernet
from loguru import logger


class SecretsManager:
    """
    Secure secrets management
    Encrypts and stores sensitive credentials
    """

    def __init__(self, secrets_file: Optional[Path] = None, encryption_key: Optional[bytes] = None):
        """
        Initialize secrets manager

        Args:
            secrets_file: Path to encrypted secrets file
            encryption_key: Encryption key (generated if not provided)
        """
        self.secrets_file = secrets_file or Path("secrets.enc")

        # Get or generate encryption key
        if encryption_key:
            self.encryption_key = encryption_key
        elif os.getenv("SECRETS_KEY"):
            self.encryption_key = os.getenv("SECRETS_KEY").encode()
        else:
            # Generate new key (should be stored securely in production)
            self.encryption_key = Fernet.generate_key()
            logger.warning(
                "Generated new encryption key. Store this securely: "
                f"{self.encryption_key.decode()}"
            )

        self.fernet = Fernet(self.encryption_key)
        self._secrets: Dict[str, str] = {}

        # Load existing secrets
        if self.secrets_file.exists():
            self.load()

    def set(self, key: str, value: str):
        """
        Set a secret

        Args:
            key: Secret identifier
            value: Secret value
        """
        self._secrets[key] = value
        logger.debug(f"Secret set: {key}")

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret

        Args:
            key: Secret identifier
            default: Default value if secret not found

        Returns:
            Secret value or default
        """
        # First check environment variables
        env_value = os.getenv(key)
        if env_value:
            return env_value

        # Then check stored secrets
        return self._secrets.get(key, default)

    def delete(self, key: str):
        """
        Delete a secret

        Args:
            key: Secret identifier
        """
        if key in self._secrets:
            del self._secrets[key]
            logger.debug(f"Secret deleted: {key}")

    def save(self):
        """Save secrets to encrypted file"""
        try:
            # Convert secrets to JSON
            secrets_json = json.dumps(self._secrets)

            # Encrypt
            encrypted_data = self.fernet.encrypt(secrets_json.encode())

            # Save to file
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)

            # Set restrictive permissions (Unix only)
            try:
                os.chmod(self.secrets_file, 0o600)
            except Exception:
                pass  # Windows doesn't support chmod

            logger.info(f"Secrets saved to {self.secrets_file}")

        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise

    def load(self):
        """Load secrets from encrypted file"""
        try:
            # Read encrypted data
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()

            # Decrypt
            decrypted_data = self.fernet.decrypt(encrypted_data)

            # Parse JSON
            self._secrets = json.loads(decrypted_data.decode())

            logger.info(f"Loaded {len(self._secrets)} secrets from {self.secrets_file}")

        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            raise

    def generate_api_key(self) -> str:
        """
        Generate a secure API key

        Returns:
            Generated API key
        """
        return secrets.token_urlsafe(32)

    def generate_secret_key(self, length: int = 32) -> str:
        """
        Generate a secure secret key

        Args:
            length: Key length in bytes

        Returns:
            Generated secret key
        """
        return secrets.token_hex(length)

    def rotate_key(self, new_key: bytes):
        """
        Rotate encryption key

        Args:
            new_key: New encryption key
        """
        # Save with old key first
        self.save()

        # Update to new key
        old_fernet = self.fernet
        self.encryption_key = new_key
        self.fernet = Fernet(new_key)

        # Save with new key
        self.save()

        logger.info("Encryption key rotated successfully")

    def list_keys(self) -> list:
        """
        List all secret keys (not values)

        Returns:
            List of secret identifiers
        """
        return list(self._secrets.keys())

    def __len__(self) -> int:
        """Get number of secrets"""
        return len(self._secrets)

    def __contains__(self, key: str) -> bool:
        """Check if secret exists"""
        return key in self._secrets


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """
    Get global secrets manager instance

    Returns:
        SecretsManager instance
    """
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get a secret

    Args:
        key: Secret identifier
        default: Default value

    Returns:
        Secret value or default
    """
    return get_secrets_manager().get(key, default)
