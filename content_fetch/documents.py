"""Document conversion helpers."""

import logging
import os
from typing import Optional as _Optional

logger = logging.getLogger(__name__)

_MAX_DOC_SIZE = 50 * 1024 * 1024  # 50 MB
_MAX_DOC_OUTPUT = 30000  # chars


def convert_document(file_path: str, max_chars: int = _MAX_DOC_OUTPUT) -> _Optional[str]:
    """Convert a document (PDF, DOCX, PPTX, XLSX, HTML) to markdown text.

    Returns markdown string or None on failure. Pass max_chars=0 for unlimited.
    """
    try:
        from markitdown import MarkItDown
    except ImportError:
        logger.warning("[shared] markitdown not installed")
        return None

    if not os.path.isfile(file_path):
        logger.warning("[shared] convert_document: file not found: %s", file_path)
        return None

    size = os.path.getsize(file_path)
    if size > _MAX_DOC_SIZE:
        logger.warning("[shared] convert_document: file too large (%d bytes)", size)
        return None

    converter = MarkItDown()
    result = converter.convert(file_path)  # raises on failure — caller surfaces it
    text = result.text_content or ""
    if not text:
        return None
    return text if max_chars == 0 else text[:max_chars]


