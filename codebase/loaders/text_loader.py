"""
Text Loader Module

Handles plain text file loading with encoding detection.
Supports: .txt, .md, .markdown
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from charset_normalizer import from_bytes
    HAS_CHARSET_NORMALIZER = True
except ImportError:
    HAS_CHARSET_NORMALIZER = False

# Supported extensions for this loader
SUPPORTED_EXTENSIONS = ['.txt', '.md', '.markdown']


def _detect_encoding(file_path: str) -> str:
    """
    Detect file encoding using charset_normalizer.
    Falls back to utf-8 if detection fails or library not available.
    """
    if not HAS_CHARSET_NORMALIZER:
        return 'utf-8'

    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()

        result = from_bytes(raw_data).best()
        if result:
            return result.encoding
    except Exception:
        pass

    return 'utf-8'


def load_text(path: str) -> str:
    """
    Load text content from a plain text file.

    Uses charset_normalizer for encoding detection to handle
    various text encodings gracefully.

    Args:
        path: Path to the text file

    Returns:
        Text content as string
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found: {path}")

    encoding = _detect_encoding(path)

    try:
        with open(path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback: try with utf-8 and error handling
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python text_loader.py <text_file_path>")
        sys.exit(1)

    text_path = sys.argv[1]
    text = load_text(text_path)
    print(f"Loaded {len(text)} characters from {text_path}")
    print(text[:500] + "..." if len(text) > 500 else text)
