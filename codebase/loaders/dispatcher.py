"""
Loader Dispatcher Module

Provides a unified interface for loading various file formats.
Automatically dispatches to the appropriate loader based on file extension.
"""

import os
from typing import Callable, Optional

from .pdf_loader import load_pdf, SUPPORTED_EXTENSIONS as PDF_EXTENSIONS
from .image_loader import load_image, SUPPORTED_EXTENSIONS as IMAGE_EXTENSIONS
from .text_loader import load_text, SUPPORTED_EXTENSIONS as TEXT_EXTENSIONS
from .document_loader import load_document, SUPPORTED_EXTENSIONS as DOC_EXTENSIONS


# Build extension to loader mapping
SUPPORTED_EXTENSIONS = {}

# PDF files
for ext in PDF_EXTENSIONS:
    SUPPORTED_EXTENSIONS[ext] = load_pdf

# Image files
for ext in IMAGE_EXTENSIONS:
    SUPPORTED_EXTENSIONS[ext] = load_image

# Text files
for ext in TEXT_EXTENSIONS:
    SUPPORTED_EXTENSIONS[ext] = load_text

# Document files
for ext in DOC_EXTENSIONS:
    SUPPORTED_EXTENSIONS[ext] = load_document


def get_loader(path: str) -> Optional[Callable]:
    """
    Get the appropriate loader function for a file.

    Args:
        path: Path to the file

    Returns:
        Loader function, or None if format not supported
    """
    ext = os.path.splitext(path)[1].lower()
    return SUPPORTED_EXTENSIONS.get(ext)


def load_file(path: str, **kwargs) -> str:
    """
    Load content from a file using the appropriate loader.

    This is a convenience function that automatically detects
    the file type and uses the correct loader.

    Args:
        path: Path to the file
        **kwargs: Additional arguments passed to the loader

    Returns:
        Extracted text content as string

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    loader = SUPPORTED_EXTENSIONS.get(ext)

    if loader is None:
        supported = ', '.join(sorted(SUPPORTED_EXTENSIONS.keys()))
        raise ValueError(
            f"Unsupported file format: {ext}\n"
            f"Supported formats: {supported}"
        )

    return loader(path, **kwargs)


def is_supported(path: str) -> bool:
    """
    Check if a file format is supported.

    Args:
        path: Path to the file

    Returns:
        True if format is supported, False otherwise
    """
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def get_supported_extensions() -> list:
    """
    Get list of all supported file extensions.

    Returns:
        List of supported extensions (e.g., ['.pdf', '.docx', ...])
    """
    return sorted(SUPPORTED_EXTENSIONS.keys())


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dispatcher.py <file_path>")
        print(f"\nSupported formats: {', '.join(get_supported_extensions())}")
        sys.exit(1)

    file_path = sys.argv[1]

    if not is_supported(file_path):
        print(f"Error: Unsupported file format")
        print(f"Supported formats: {', '.join(get_supported_extensions())}")
        sys.exit(1)

    text = load_file(file_path)
    print(f"Loaded {len(text)} characters from {file_path}")
    print(text[:500] + "..." if len(text) > 500 else text)
