"""
Unified PDF Text Extractor Module

Provides a single source of truth for PDF text extraction using PyMuPDF (fitz).
Replaces all pdfplumber usage with a consistent PyMuPDF-based interface.

Functions:
- extract_text_from_page: Extract text from a single page
- extract_text_from_pdf: Extract all text from a PDF
- extract_pages_text: Extract text as a list (one string per page)
- get_page_count: Get number of pages in a PDF
- get_page_word_count: Count words on a page
"""

import fitz  # PyMuPDF
from typing import List


def extract_text_from_page(page: fitz.Page) -> str:
    """
    Extract text content from a single PDF page.

    Args:
        page: A PyMuPDF Page object

    Returns:
        Extracted text with whitespace stripped
    """
    text = page.get_text("text")
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text content from a PDF file.

    Opens the PDF, extracts text from all pages, and returns
    the combined text as a single string.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Combined text from all pages, joined with spaces

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        fitz.FileDataError: If the file is not a valid PDF
    """
    doc = None
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for page in doc:
            page_text = extract_text_from_page(page)
            if page_text:
                texts.append(page_text)
        return " ".join(texts)
    finally:
        if doc:
            doc.close()


def extract_pages_text(pdf_path: str) -> List[str]:
    """
    Extract text from a PDF as a list of strings, one per page.

    Useful for page-by-page processing where you need to maintain
    page boundaries.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of strings, where each string is the text from one page

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        fitz.FileDataError: If the file is not a valid PDF
    """
    doc = None
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            page_text = extract_text_from_page(page)
            pages_text.append(page_text)
        return pages_text
    finally:
        if doc:
            doc.close()


def get_page_count(pdf_path: str) -> int:
    """
    Get the number of pages in a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Number of pages in the PDF

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        fitz.FileDataError: If the file is not a valid PDF
    """
    doc = None
    try:
        doc = fitz.open(pdf_path)
        return len(doc)
    finally:
        if doc:
            doc.close()


def get_page_word_count(page: fitz.Page) -> int:
    """
    Count the number of words on a PDF page.

    Used for OCR strategy detection - pages with low word counts
    may need OCR processing.

    Args:
        page: A PyMuPDF Page object

    Returns:
        Number of words on the page
    """
    text = extract_text_from_page(page)
    if not text:
        return 0
    return len(text.split())


def get_pdf_word_counts(pdf_path: str) -> List[int]:
    """
    Get word counts for each page in a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of word counts, one per page
    """
    doc = None
    try:
        doc = fitz.open(pdf_path)
        word_counts = []
        for page in doc:
            word_counts.append(get_page_word_count(page))
        return word_counts
    finally:
        if doc:
            doc.close()


# CLI test interface
if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        # Try to use PDF_PATH from environment
        from dotenv import load_dotenv
        load_dotenv()
        pdf_path = os.getenv("PDF_PATH")
        if not pdf_path:
            print("Usage: python pdf_extractor.py <pdf_path>")
            print("Or set PDF_PATH in .env file")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"PDF File: {pdf_path}")
    print("=" * 60)

    # Page count
    page_count = get_page_count(pdf_path)
    print(f"Page Count: {page_count}")

    # Word counts per page
    word_counts = get_pdf_word_counts(pdf_path)
    print(f"\nWord Counts per Page:")
    for i, count in enumerate(word_counts):
        print(f"  Page {i + 1}: {count} words")
    print(f"  Total: {sum(word_counts)} words")

    # First 500 chars of extracted text
    full_text = extract_text_from_pdf(pdf_path)
    print(f"\nFirst 500 characters of extracted text:")
    print("-" * 40)
    print(full_text[:500])
    if len(full_text) > 500:
        print("...")
    print("-" * 40)
    print(f"\nTotal characters: {len(full_text)}")
