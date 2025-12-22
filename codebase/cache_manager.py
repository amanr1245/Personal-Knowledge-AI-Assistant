"""
Cache Manager for RAG System

Provides disk-based caching to reduce API costs by caching:
- Embeddings (by chunk text hash)
- Chunking results (by document text hash)
- OCR results (by page image hash)

Uses diskcache for persistent key-value storage at ~/.rag_cache/
"""

import os
import hashlib
from pathlib import Path
from typing import Any, Optional
import diskcache

# Cache directory
CACHE_DIR = Path.home() / ".rag_cache"

# Global cache instance
_cache: Optional[diskcache.Cache] = None

# Global flag to disable caching
_cache_enabled = True


def init_cache(cache_dir: str = None) -> diskcache.Cache:
    """
    Initialize the disk cache.

    Args:
        cache_dir: Optional custom cache directory

    Returns:
        diskcache.Cache instance
    """
    global _cache
    if _cache is None:
        dir_path = Path(cache_dir) if cache_dir else CACHE_DIR
        dir_path.mkdir(parents=True, exist_ok=True)
        _cache = diskcache.Cache(str(dir_path))
    return _cache


def get_cache() -> diskcache.Cache:
    """Get the cache instance, initializing if needed."""
    if _cache is None:
        init_cache()
    return _cache


def set_cache_enabled(enabled: bool):
    """Enable or disable caching globally."""
    global _cache_enabled
    _cache_enabled = enabled


def is_cache_enabled() -> bool:
    """Check if caching is enabled."""
    return _cache_enabled


def cache_key(operation: str, content: str | bytes) -> str:
    """
    Generate a cache key from operation type and content.

    Args:
        operation: Type of operation (e.g., 'embedding', 'chunk', 'ocr')
        content: Content to hash (string or bytes)

    Returns:
        Cache key string in format "operation:hash"
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
    content_hash = hashlib.sha256(content).hexdigest()[:16]
    return f"{operation}:{content_hash}"


def get(key: str) -> Optional[Any]:
    """
    Get a value from cache.

    Args:
        key: Cache key

    Returns:
        Cached value or None if not found or caching disabled
    """
    if not _cache_enabled:
        return None

    cache = get_cache()
    return cache.get(key)


def set(key: str, value: Any, expire: int = None) -> bool:
    """
    Set a value in cache.

    Args:
        key: Cache key
        value: Value to cache
        expire: Optional expiration time in seconds

    Returns:
        True if successful, False if caching disabled
    """
    if not _cache_enabled:
        return False

    cache = get_cache()
    cache.set(key, value, expire=expire)
    return True


def delete(key: str) -> bool:
    """
    Delete a key from cache.

    Args:
        key: Cache key

    Returns:
        True if deleted, False otherwise
    """
    cache = get_cache()
    return cache.delete(key)


def clear() -> int:
    """
    Clear all cached data.

    Returns:
        Number of items cleared
    """
    cache = get_cache()
    count = len(cache)
    cache.clear()
    return count


def stats() -> dict:
    """
    Get cache statistics.

    Returns:
        Dict with cache stats (size, count, directory)
    """
    cache = get_cache()
    return {
        "directory": str(CACHE_DIR),
        "count": len(cache),
        "size_bytes": cache.volume(),
        "enabled": _cache_enabled
    }


# Convenience functions for specific operations

def cache_embedding(text: str, embedding: list) -> bool:
    """Cache an embedding result."""
    key = cache_key("embedding", text)
    return set(key, embedding)


def get_cached_embedding(text: str) -> Optional[list]:
    """Get a cached embedding."""
    key = cache_key("embedding", text)
    return get(key)


def cache_chunks(document_text: str, chunks: list) -> bool:
    """Cache chunking results."""
    key = cache_key("chunks", document_text)
    return set(key, chunks)


def get_cached_chunks(document_text: str) -> Optional[list]:
    """Get cached chunks."""
    key = cache_key("chunks", document_text)
    return get(key)


def cache_ocr(image_bytes: bytes, text: str) -> bool:
    """Cache OCR result."""
    key = cache_key("ocr", image_bytes)
    return set(key, text)


def get_cached_ocr(image_bytes: bytes) -> Optional[str]:
    """Get cached OCR result."""
    key = cache_key("ocr", image_bytes)
    return get(key)


def cache_semantic_chunks(document_text: str, chunks: list) -> bool:
    """Cache semantic chunking results (with headers)."""
    key = cache_key("semantic_chunks", document_text)
    return set(key, chunks)


def get_cached_semantic_chunks(document_text: str) -> Optional[list]:
    """Get cached semantic chunks."""
    key = cache_key("semantic_chunks", document_text)
    return get(key)


# CLI for cache management
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cache_manager.py <command>")
        print("Commands:")
        print("  stats  - Show cache statistics")
        print("  clear  - Clear all cached data")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "stats":
        s = stats()
        print(f"Cache Directory: {s['directory']}")
        print(f"Items Cached: {s['count']}")
        print(f"Size: {s['size_bytes'] / 1024:.2f} KB")
        print(f"Enabled: {s['enabled']}")

    elif command == "clear":
        count = clear()
        print(f"Cleared {count} cached items")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
