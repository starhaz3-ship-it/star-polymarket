"""
Encryption utilities for Star Polymarket.
"""

import os
import base64
import getpass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive an encryption key from a password."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt_key(private_key: str, password: str) -> tuple[str, str]:
    """Encrypt a private key with a password.

    Returns:
        Tuple of (encrypted_key, salt) both as base64 strings
    """
    salt = os.urandom(16)
    key = derive_key(password, salt)
    f = Fernet(key)
    encrypted = f.encrypt(private_key.encode())
    return base64.urlsafe_b64encode(encrypted).decode(), base64.urlsafe_b64encode(salt).decode()


def decrypt_key(encrypted_key: str, salt: str, password: str) -> str:
    """Decrypt a private key with a password."""
    salt_bytes = base64.urlsafe_b64decode(salt.encode())
    key = derive_key(password, salt_bytes)
    f = Fernet(key)
    encrypted_bytes = base64.urlsafe_b64decode(encrypted_key.encode())
    return f.decrypt(encrypted_bytes).decode()


def encrypt_env_file(env_path: str = ".env"):
    """Encrypt the private key in an .env file."""
    from dotenv import dotenv_values

    config = dotenv_values(env_path)
    private_key = config.get("POLYMARKET_PRIVATE_KEY")

    if not private_key:
        print("No POLYMARKET_PRIVATE_KEY found in .env")
        return

    if private_key.startswith("ENC:"):
        print("Key is already encrypted")
        return

    password = getpass.getpass("Enter encryption password: ")
    password_confirm = getpass.getpass("Confirm password: ")

    if password != password_confirm:
        print("Passwords do not match")
        return

    encrypted, salt = encrypt_key(private_key, password)

    # Rewrite the .env file with encrypted key
    with open(env_path, "w") as f:
        f.write(f"POLYMARKET_PRIVATE_KEY=ENC:{encrypted}\n")
        f.write(f"POLYMARKET_KEY_SALT={salt}\n")
        if config.get("POLYMARKET_WALLET_ADDRESS"):
            f.write(f"POLYMARKET_WALLET_ADDRESS={config['POLYMARKET_WALLET_ADDRESS']}\n")

    print("Private key encrypted successfully")


if __name__ == "__main__":
    encrypt_env_file()
