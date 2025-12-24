"""
Image Loader Module

Handles image file loading using shared Vision API from ocr_processor.
Supports: .jpg, .jpeg, .png, .gif, .webp, .bmp
"""

import os
import sys
import base64

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_processor import _extract_text_with_vision, _image_to_base64
import cache_manager

# Supported extensions for this loader
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']


def load_image(path: str) -> str:
    """
    Load and extract text/description from an image file.

    Uses shared GPT-4 Vision function from ocr_processor.
    Results are cached to avoid redundant API calls.

    Args:
        path: Path to the image file

    Returns:
        Description and extracted text from the image
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    # Read image bytes
    with open(path, 'rb') as f:
        image_bytes = f.read()

    # Check cache first
    cached_text = cache_manager.get_cached_ocr(image_bytes)
    if cached_text is not None:
        print(f"(cached) {path}")
        return cached_text

    # Convert to base64 and process with shared Vision function
    image_base64 = _image_to_base64(image_bytes)
    prompt = "Describe this image and extract any visible text. Be thorough but concise."

    text = _extract_text_with_vision(image_base64, prompt)

    # Cache the result
    if text:
        cache_manager.cache_ocr(image_bytes, text)

    return text if text else f"[Unable to process image: {path}]"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_loader.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    text = load_image(image_path)
    print(f"Extracted content from {image_path}:")
    print(text)
