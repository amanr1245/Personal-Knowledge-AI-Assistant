"""
Document Loader Module

Handles document file loading using pypandoc for conversion.
Supports: .docx, .odt, .html, .htm

For .docx and .odt files, also extracts embedded images from the
ZIP archive and processes them with shared Vision API from ocr_processor.
"""

import os
import sys
import zipfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_processor import _extract_text_with_vision, _image_to_base64
import cache_manager

try:
    import pypandoc
    HAS_PYPANDOC = True
except ImportError:
    HAS_PYPANDOC = False

# Supported extensions for this loader
SUPPORTED_EXTENSIONS = ['.docx', '.odt', '.html', '.htm']

# Image extensions to extract from documents
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')


def _extract_images_from_zip(path: str) -> list:
    """
    Extract embedded images from DOCX or ODT files.

    These formats are ZIP archives containing media files.

    Args:
        path: Path to the document file

    Returns:
        List of (image_name, image_bytes) tuples
    """
    images = []

    try:
        with zipfile.ZipFile(path, 'r') as zf:
            for name in zf.namelist():
                # DOCX uses word/media/, ODT uses Pictures/
                if ('media/' in name or 'Pictures/' in name) and name.lower().endswith(IMAGE_EXTENSIONS):
                    try:
                        image_bytes = zf.read(name)
                        images.append((name, image_bytes))
                    except Exception as e:
                        print(f"Error extracting {name}: {e}")
    except zipfile.BadZipFile:
        print(f"Warning: {path} is not a valid ZIP archive")
    except Exception as e:
        print(f"Error reading ZIP archive {path}: {e}")

    return images


def _process_image_with_vision(image_bytes: bytes, image_name: str) -> str:
    """
    Process an image with shared GPT-4 Vision function from ocr_processor.

    Args:
        image_bytes: Raw image bytes
        image_name: Name of the image file (for logging)

    Returns:
        Extracted text/description from the image
    """
    # Check cache first
    cached_text = cache_manager.get_cached_ocr(image_bytes)
    if cached_text is not None:
        return cached_text

    # Convert to base64 and use shared Vision function
    image_base64 = _image_to_base64(image_bytes)
    prompt = "Extract any text from this image. If it's a diagram or figure, describe its content and any labels. Be concise."

    text = _extract_text_with_vision(image_base64, prompt)

    # Cache the result
    cache_manager.cache_ocr(image_bytes, text)

    return text


def load_document(path: str, extract_images: bool = True) -> str:
    """
    Load text content from a document file.

    Uses pypandoc to convert documents to plain text.
    For DOCX and ODT files, also extracts and processes embedded images.

    Args:
        path: Path to the document file
        extract_images: Whether to extract and process embedded images (default True)

    Returns:
        Extracted text content as string
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document file not found: {path}")

    if not HAS_PYPANDOC:
        raise ImportError(
            "pypandoc is required for document loading. "
            "Install with: pip install pypandoc\n"
            "Also install pandoc: https://pandoc.org/installing.html"
        )

    ext = os.path.splitext(path)[1].lower()

    # Convert document to plain text using pypandoc
    try:
        text = pypandoc.convert_file(path, 'plain')
    except Exception as e:
        print(f"Error converting {path} with pypandoc: {e}")
        text = ""

    # Extract images from DOCX/ODT files
    if extract_images and ext in ['.docx', '.odt']:
        images = _extract_images_from_zip(path)

        if images:
            print(f"Found {len(images)} embedded images in {path}")
            image_texts = []

            for image_name, image_bytes in images:
                image_text = _process_image_with_vision(image_bytes, image_name)
                if image_text:
                    image_texts.append(f"[embedded image: {image_text}]")

            if image_texts:
                text = text + "\n\n" + "\n\n".join(image_texts)

    return text.strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <document_path> [extract_images]")
        print("  extract_images: true (default) or false")
        sys.exit(1)

    doc_path = sys.argv[1]
    extract_images = sys.argv[2].lower() != 'false' if len(sys.argv) > 2 else True

    text = load_document(doc_path, extract_images)
    print(f"Extracted {len(text)} characters from {doc_path}")
    print(text[:500] + "..." if len(text) > 500 else text)
