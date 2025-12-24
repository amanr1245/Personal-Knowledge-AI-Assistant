"""
Loaders Package

Provides unified file loading for various formats:
- PDF files (.pdf)
- Image files (.jpg, .jpeg, .png, .gif, .webp, .bmp)
- Text files (.txt, .md, .markdown)
- Document files (.docx, .odt, .html, .htm)

Usage:
    from loaders import load_file, get_loader, SUPPORTED_EXTENSIONS

    # Load any supported file
    text = load_file("document.pdf")

    # Get specific loader
    loader = get_loader("image.png")
    text = loader("image.png")

    # Check supported formats
    print(SUPPORTED_EXTENSIONS)
"""

from .dispatcher import (
    load_file,
    get_loader,
    is_supported,
    get_supported_extensions,
    SUPPORTED_EXTENSIONS
)

from .pdf_loader import load_pdf
from .image_loader import load_image
from .text_loader import load_text
from .document_loader import load_document

__all__ = [
    # Main interface
    'load_file',
    'get_loader',
    'is_supported',
    'get_supported_extensions',
    'SUPPORTED_EXTENSIONS',
    # Individual loaders
    'load_pdf',
    'load_image',
    'load_text',
    'load_document',
]
