"""
OCR Processor Module for RAG Project

Uses PyMuPDF (fitz) for PDF handling and OpenAI GPT-4 Vision for image-to-text.
Supports parallel processing for faster ingestion.

Strategies:
- AUTO: Chooses STRICT or RELAXED based on word count per page
- STRICT: Renders full page at 150 DPI, extracts text via Vision API
- RELAXED: Uses PyMuPDF text + extracts embedded images via Vision API
"""

import os
import base64
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openai import OpenAI
import cache_manager
from retry_manager import with_retry
from pdf_extractor import extract_text_from_page
from processors import VisionProcessor, VisionRequest

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Parallel processing settings
PARALLEL_ENABLED = os.getenv("OCR_PARALLEL", "true").lower() == "true"
PARALLEL_WORKERS = int(os.getenv("OCR_PARALLEL_WORKERS", "8"))


class OcrStrategy(Enum):
    """OCR processing strategy for PDF pages."""
    AUTO = "auto"
    STRICT = "strict"
    RELAXED = "relaxed"


def _image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _extract_text_with_vision(image_base64: str, prompt: str) -> str:
    """
    Send an image to OpenAI Vision API and extract text.
    """
    try:
        response = with_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Vision API error: {e}")
        return ""


def _render_page_to_image(page: fitz.Page, dpi: int = 150) -> bytes:
    """Render a PDF page to PNG image bytes at specified DPI."""
    pixmap = page.get_pixmap(dpi=dpi)
    return pixmap.tobytes("png")


def _extract_embedded_images(page: fitz.Page) -> List[Tuple[bytes, int]]:
    """
    Extract embedded images from a PDF page.
    Returns list of (image_bytes, image_index) tuples.
    """
    images = []
    image_list = page.get_images(full=True)

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        try:
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            images.append((image_bytes, img_index))
        except Exception as e:
            print(f"Error extracting image {img_index}: {e}")
            continue

    return images


def resolve_ocr_strategy(page_text: str, threshold: int = 50) -> OcrStrategy:
    """Determine OCR strategy based on word count."""
    word_count = len(page_text.split())
    if word_count < threshold:
        return OcrStrategy.STRICT
    return OcrStrategy.RELAXED


def _process_page_strict(pdf_doc: fitz.Document, page_num: int) -> str:
    """
    STRICT mode: Render full page at 150 DPI and extract text via Vision API.
    Results are cached by page image hash.
    """
    page = pdf_doc[page_num]
    image_bytes = _render_page_to_image(page, dpi=150)

    # Check cache first
    cached_text = cache_manager.get_cached_ocr(image_bytes)
    if cached_text is not None:
        print(f"(cached)", end=" ")
        return cached_text

    image_base64 = _image_to_base64(image_bytes)
    prompt = "Extract all text from this document page exactly as written. Preserve formatting and structure."
    text = _extract_text_with_vision(image_base64, prompt)

    # Cache the result
    cache_manager.cache_ocr(image_bytes, text)

    return text


def _process_page_relaxed(pdf_doc: fitz.Document, page_num: int) -> str:
    """
    RELAXED mode: Keep PyMuPDF text, extract embedded images via Vision API.
    Image text is inserted as [image: extracted text here].
    """
    # Extract text using PyMuPDF
    page = pdf_doc[page_num]
    page_text = extract_text_from_page(page)

    # Extract embedded images using PyMuPDF
    embedded_images = _extract_embedded_images(page)

    if not embedded_images:
        return page_text

    # Process each embedded image with Vision API (with caching)
    image_texts = []
    for image_bytes, img_index in embedded_images:
        # Check cache first
        cached_text = cache_manager.get_cached_ocr(image_bytes)
        if cached_text is not None:
            if cached_text:
                image_texts.append(f"[image: {cached_text}]")
            continue

        image_base64 = _image_to_base64(image_bytes)
        prompt = (
            "Extract any text from this image. If it's a diagram or figure, "
            "describe its content and any labels. Be concise."
        )
        image_text = _extract_text_with_vision(image_base64, prompt)

        # Cache the result
        cache_manager.cache_ocr(image_bytes, image_text)

        if image_text:
            image_texts.append(f"[image: {image_text}]")

    # Combine text with image descriptions
    if image_texts:
        return page_text + "\n\n" + "\n\n".join(image_texts)
    return page_text


@dataclass
class PageProcessingTask:
    """Represents a page to be processed."""
    page_num: int
    strategy: OcrStrategy
    image_bytes: Optional[bytes] = None
    base_text: str = ""
    embedded_images: List[Tuple[bytes, int]] = None

    def __post_init__(self):
        if self.embedded_images is None:
            self.embedded_images = []


def _prepare_pages_for_parallel(
    pdf_doc: fitz.Document,
    strategy: OcrStrategy,
    threshold: int
) -> Tuple[List[PageProcessingTask], List[VisionRequest]]:
    """
    Prepare all pages for parallel processing.

    Front-loads all data extraction so Vision API calls can be parallelized.

    Returns:
        (page_tasks, vision_requests) tuple
    """
    num_pages = len(pdf_doc)
    page_tasks = []
    vision_requests = []
    request_index = 0

    print(f"Preparing {num_pages} pages for parallel processing...")

    for page_num in range(num_pages):
        page = pdf_doc[page_num]

        # Determine strategy for this page
        if strategy == OcrStrategy.AUTO:
            initial_text = extract_text_from_page(page)
            page_strategy = resolve_ocr_strategy(initial_text, threshold)
        else:
            page_strategy = strategy
            initial_text = extract_text_from_page(page) if strategy == OcrStrategy.RELAXED else ""

        task = PageProcessingTask(
            page_num=page_num,
            strategy=page_strategy,
            base_text=initial_text
        )

        if page_strategy == OcrStrategy.STRICT:
            # Render page to image
            image_bytes = _render_page_to_image(page, dpi=150)
            task.image_bytes = image_bytes

            # Check cache - only add to requests if not cached
            cached = cache_manager.get_cached_ocr(image_bytes)
            if cached is None:
                vision_requests.append(VisionRequest(
                    image_bytes=image_bytes,
                    prompt="Extract all text from this document page exactly as written. Preserve formatting and structure.",
                    index=request_index,
                    image_id=f"page_{page_num}"
                ))
                request_index += 1

        else:  # RELAXED
            # Extract embedded images
            embedded = _extract_embedded_images(page)
            task.embedded_images = embedded

            # Add uncached images to requests
            for img_bytes, img_idx in embedded:
                cached = cache_manager.get_cached_ocr(img_bytes)
                if cached is None:
                    vision_requests.append(VisionRequest(
                        image_bytes=img_bytes,
                        prompt="Extract any text from this image. If it's a diagram or figure, describe its content and any labels. Be concise.",
                        index=request_index,
                        image_id=f"page_{page_num}_img_{img_idx}"
                    ))
                    request_index += 1

        page_tasks.append(task)

    return page_tasks, vision_requests


def _assemble_results(
    page_tasks: List[PageProcessingTask],
    vision_results: dict
) -> List[str]:
    """
    Assemble final text for each page using vision results.

    Args:
        page_tasks: List of page processing tasks
        vision_results: Dict mapping image_bytes hash to extracted text

    Returns:
        List of page texts
    """
    extracted_pages = []

    for task in page_tasks:
        if task.strategy == OcrStrategy.STRICT:
            # Get result from cache or vision results
            cached = cache_manager.get_cached_ocr(task.image_bytes)
            if cached is not None:
                page_text = cached
            else:
                # Look up in vision results
                img_hash = hash(task.image_bytes)
                page_text = vision_results.get(img_hash, "")

        else:  # RELAXED
            page_text = task.base_text

            # Add image descriptions
            image_texts = []
            for img_bytes, img_idx in task.embedded_images:
                cached = cache_manager.get_cached_ocr(img_bytes)
                if cached is not None:
                    if cached:
                        image_texts.append(f"[image: {cached}]")
                else:
                    img_hash = hash(img_bytes)
                    img_text = vision_results.get(img_hash, "")
                    if img_text:
                        image_texts.append(f"[image: {img_text}]")

            if image_texts:
                page_text = page_text + "\n\n" + "\n\n".join(image_texts)

        extracted_pages.append(page_text)

    return extracted_pages


def process_pdf_with_ocr_parallel(
    pdf_path: str,
    strategy: OcrStrategy = OcrStrategy.AUTO,
    threshold: int = 50,
    max_workers: int = None
) -> List[str]:
    """
    Process a PDF file with parallel OCR capabilities.

    Front-loads all page preparation, then processes Vision API calls
    in parallel for significant speedup on multi-page documents.

    Args:
        pdf_path: Path to the PDF file
        strategy: OCR strategy (AUTO, STRICT, or RELAXED)
        threshold: Word count threshold for AUTO strategy (default 50)
        max_workers: Max parallel workers (default from env or 8)

    Returns:
        List of extracted text strings, one per page
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pdf_doc = fitz.open(pdf_path)
    num_pages = len(pdf_doc)

    print(f"Processing PDF (parallel): {pdf_path}")
    print(f"Total pages: {num_pages}, Strategy: {strategy.value}")

    # Phase 1: Prepare all pages (front-load data extraction)
    page_tasks, vision_requests = _prepare_pages_for_parallel(pdf_doc, strategy, threshold)

    pdf_doc.close()

    # Phase 2: Process vision requests in parallel
    vision_results = {}

    if vision_requests:
        print(f"\nProcessing {len(vision_requests)} vision requests in parallel...")
        processor = VisionProcessor(max_workers=max_workers or PARALLEL_WORKERS)
        results = processor.process_batch(vision_requests)

        # Map results back by image hash
        for req, (idx, text) in zip(vision_requests, results):
            img_hash = hash(req.image_bytes)
            vision_results[img_hash] = text or ""

            # Cache the result
            if text:
                cache_manager.cache_ocr(req.image_bytes, text)
    else:
        print("All pages cached - no vision requests needed")

    # Phase 3: Assemble final results
    print("\nAssembling results...")
    extracted_pages = _assemble_results(page_tasks, vision_results)

    print(f"Processing complete. Extracted {len(extracted_pages)} pages.")
    return extracted_pages


def process_pdf_with_ocr(
    pdf_path: str,
    strategy: OcrStrategy = OcrStrategy.AUTO,
    threshold: int = 50,
    parallel: bool = None
) -> List[str]:
    """
    Process a PDF file with OCR capabilities.

    Args:
        pdf_path: Path to the PDF file
        strategy: OCR strategy (AUTO, STRICT, or RELAXED)
        threshold: Word count threshold for AUTO strategy (default 50)
        parallel: Use parallel processing (default from env OCR_PARALLEL)

    Returns:
        List of extracted text strings, one per page
    """
    # Determine if parallel processing should be used
    use_parallel = parallel if parallel is not None else PARALLEL_ENABLED

    if use_parallel:
        return process_pdf_with_ocr_parallel(pdf_path, strategy, threshold)

    # Sequential processing (original implementation)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pdf_doc = fitz.open(pdf_path)
    num_pages = len(pdf_doc)

    print(f"Processing PDF (sequential): {pdf_path}")
    print(f"Total pages: {num_pages}, Strategy: {strategy.value}")

    extracted_pages = []

    for page_num in range(num_pages):
        print(f"Processing page {page_num + 1}/{num_pages}...", end=" ")

        if strategy == OcrStrategy.STRICT:
            page_text = _process_page_strict(pdf_doc, page_num)
            mode_used = "STRICT"

        elif strategy == OcrStrategy.RELAXED:
            page_text = _process_page_relaxed(pdf_doc, page_num)
            mode_used = "RELAXED"

        else:  # AUTO strategy
            # Get text using PyMuPDF to check word count
            page = pdf_doc[page_num]
            initial_text = extract_text_from_page(page)

            resolved_strategy = resolve_ocr_strategy(initial_text, threshold)

            if resolved_strategy == OcrStrategy.STRICT:
                page_text = _process_page_strict(pdf_doc, page_num)
                mode_used = "STRICT"
            else:
                page_text = _process_page_relaxed(pdf_doc, page_num)
                mode_used = "RELAXED"

        extracted_pages.append(page_text)
        print(f"[{mode_used}] {len(page_text)} chars")

    pdf_doc.close()
    print(f"Processing complete.")
    return extracted_pages


def process_pdf_with_ocr_combined(
    pdf_path: str,
    strategy: OcrStrategy = OcrStrategy.AUTO,
    threshold: int = 50
) -> str:
    """Process a PDF and return all text combined into a single string."""
    pages = process_pdf_with_ocr(pdf_path, strategy, threshold)
    return "\n\n".join(pages)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_processor.py <pdf_path> [strategy] [threshold]")
        print("  strategy: auto (default), strict, relaxed")
        print("  threshold: word count threshold for auto (default 50)")
        sys.exit(1)

    pdf_path = sys.argv[1]
    strategy_str = sys.argv[2].lower() if len(sys.argv) > 2 else "auto"
    strategy_map = {"auto": OcrStrategy.AUTO, "strict": OcrStrategy.STRICT, "relaxed": OcrStrategy.RELAXED}
    strategy = strategy_map.get(strategy_str, OcrStrategy.AUTO)
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    pages = process_pdf_with_ocr(pdf_path, strategy, threshold)

    print("\n" + "=" * 60)
    for i, page_text in enumerate(pages):
        print(f"\n--- Page {i + 1} ---")
        print(page_text[:500] + "..." if len(page_text) > 500 else page_text)
