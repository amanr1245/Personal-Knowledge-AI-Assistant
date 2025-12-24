"""PDF loader - delegates to ocr_processor for text extraction."""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_processor import (
    process_pdf_with_ocr_combined,
    OcrStrategy
)


def load_pdf(
    path: str,
    use_ocr: bool = True,
    strategy: OcrStrategy = OcrStrategy.AUTO,
    threshold: int = 50
) -> str:
    """
    Load text content from a PDF file.

    Args:
        path: Path to PDF file
        use_ocr: Whether to use OCR for scanned pages (default True)
        strategy: OCR strategy (AUTO, STRICT, RELAXED)
        threshold: Word count threshold for AUTO strategy

    Returns:
        Extracted text content
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: {path}")

    if use_ocr:
        return process_pdf_with_ocr_combined(path, strategy, threshold)
    else:
        # Fast path: PyMuPDF text extraction only (no Vision API)
        import fitz
        with fitz.open(path) as pdf:
            return " ".join(page.get_text("text") for page in pdf)


# Supported extensions for this loader
SUPPORTED_EXTENSIONS = ['.pdf']


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <pdf_path> [use_ocr] [strategy] [threshold]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    use_ocr = sys.argv[2].lower() != 'false' if len(sys.argv) > 2 else True
    strategy_str = sys.argv[3].lower() if len(sys.argv) > 3 else "auto"
    strategy_map = {"auto": OcrStrategy.AUTO, "strict": OcrStrategy.STRICT, "relaxed": OcrStrategy.RELAXED}
    strategy = strategy_map.get(strategy_str, OcrStrategy.AUTO)
    threshold = int(sys.argv[4]) if len(sys.argv) > 4 else 50

    text = load_pdf(pdf_path, use_ocr, strategy, threshold)
    print(f"Extracted {len(text)} characters")
    print(text[:500] + "..." if len(text) > 500 else text)
