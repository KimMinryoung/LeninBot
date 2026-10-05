"""Fetch, download, and document-conversion runtime tools."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import re
import time
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse
from tool_gateway.results import ToolFailure, ToolRejection, ToolResult

logger = logging.getLogger(__name__)

FETCH_TOOLS = [
    {
        "name": "fetch_url",
        "description": (
            "Fetch and extract body text from a URL. Use when the user shares a link and asks about "
            "its content, or to verify a claim against its source. Returns a character slice of cleaned "
            "body text (default offset 0, max_chars 10,000). Use offset from the next hint to paginate "
            "long pages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch content from."},
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "Max characters of body text to return (1,000-50,000, default 10,000). "
                        "Raise for long primary sources (reports, transcripts, statistical releases) "
                        "you intend to cite precisely."
                    ),
                    "default": 10000,
                },
                "offset": {
                    "type": "integer",
                    "description": "0-indexed character offset into the extracted body text. Default 0.",
                    "default": 0,
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "download_file",
        "description": "Download URL -> data/downloads/. Returns local path. For PDFs/docs to feed convert_document. Max 100 MB.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute URL of the file to download (http/https)."},
                "filename": {"type": "string", "description": "Optional; auto from URL."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "download_image",
        "description": (
            "Download an image from a URL and save it locally. "
            "Returns the local file path, which can be passed to generate_image's reference_image parameter. "
            "Use this when you need a reference photo for image generation. Max 20 MB."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Image URL to download."},
                "filename": {"type": "string", "description": "Optional filename (without extension). Auto-generated if omitted."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "convert_document",
        "description": "Local PDF/DOCX/PPTX/XLSX/HTML -> markdown saved to data/converted/. Returns path + head preview; use read_file to paginate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path."},
            },
            "required": ["file_path"],
        },
    },
]


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def _exec_fetch_url(
    url: str,
    max_chars: int = 10000,
    offset: int = 0,
    char_offset: int | None = None,
    **kwargs,
) -> str:
    """Fetch and extract main body text from a URL."""
    try:
        max_chars = max(1000, min(int(max_chars), 50000))
    except (TypeError, ValueError):
        max_chars = 10000
    if char_offset is None:
        char_offset = (
            kwargs.get("offset_chars")
            or kwargs.get("char_start")
            or kwargs.get("start_char")
        )
    try:
        start = int(char_offset if char_offset is not None else offset)
        if start < 0:
            raise ValueError()
    except (TypeError, ValueError):
        raise ToolRejection("offset must be a non-negative integer")
    # Fetch one extra character as a sentinel so exact-boundary pages do not
    # falsely report another page. The returned slice still respects max_chars.
    fetch_limit = start + max_chars + 1
    try:
        from content_fetch.urls import diagnose_url_fetch_failure, fetch_url_content_async
        from provenance.runtime import _wrap_external

        content = await fetch_url_content_async(url, max_chars=fetch_limit)
        if not content or not content.strip():
            return ToolFailure("Failed to extract content from this URL. No usable body was obtained; "
                               "this is not evidence about the page's claims.\n" + diagnose_url_fetch_failure(url),
                               {"empty": True, "extracted_chars": 0, "failure_type": "empty_extraction"})
        if start >= len(content):
            raise ToolRejection(
                f"[fetch_url] url={url}\n"
                f"chars {start}:{start} of at least {len(content)}\n"
                f"Error: offset is beyond the fetched content. Last available offset is {max(len(content) - 1, 0)}."
            )
        end = min(len(content), start + max_chars)
        body = content[start:end]
        more = len(content) > end
        known_chars = end if more else len(content)
        next_hint = f"\nnext: fetch_url(url='{url}', offset={end}, max_chars={max_chars})" if more else ""
        header = (
            f"[fetch_url] url={url}\n"
            f"chars {start}:{end} of {'at least ' if more else ''}{known_chars} "
            f"truncated={more}{next_hint}\n\n"
        )
        from datetime import datetime, timezone
        header += (f"source_kind=extracted_page_text; observed_at={datetime.now(timezone.utc).isoformat()}; "
                   "publication_date=unknown; event_date=unknown. Extraction success alone "
                   "does not establish relevance, freshness or factual accuracy.\n\n")
        return ToolResult(header + _wrap_external(body, f"url:{url}"),
                          {**(getattr(content, "result_metadata", None) or {}),
                           "extracted_chars": len(content), "returned_chars": len(body), "empty": False})
    except ToolRejection:
        raise
    except Exception as exc:
        logger.error("fetch_url error: %s", exc)
        try:
            from content_fetch.urls import diagnose_url_fetch_failure

            diagnosis = diagnose_url_fetch_failure(url, [str(exc)])
            return ToolFailure(f"URL fetch failed: {exc}\n{diagnosis}", {"failure_type": type(exc).__name__})
        except Exception:
            return ToolFailure(f"URL fetch failed: {exc}", {"failure_type": type(exc).__name__})


def _download_bounded(url: str, filename: str, *, image: bool) -> str:
    from content_fetch.url_security import safe_requests_get

    limit_mb = 20 if image else 100
    max_size = limit_mb * 1024 * 1024
    out_dir = Path(_project_root()) / "data" / ("reference_images" if image else "downloads")
    temporary = None
    try:
        with safe_requests_get(url, timeout=60 if image else 120,
                               headers={"User-Agent": "Mozilla/5.0"}, stream=True) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
            if image and not content_type.startswith("image/"):
                return ToolFailure(f"❌ Not an image (Content-Type: {content_type})",
                                   {"failure_type": "invalid_content_type"})
            if int(resp.headers.get("Content-Length", "0") or 0) > max_size:
                return ToolFailure(f"❌ File exceeds {limit_mb} MB limit", {"failure_type": "size_limit"})
            base = filename or os.path.basename(unquote(urlparse(url).path)) or time.strftime("%Y%m%d_%H%M%S")
            safe_name = re.sub(r"[^a-zA-Z0-9가-힣._-]+", "-", base).strip(".-")[:120] or "download"
            if image:
                safe_name = Path(safe_name).stem + (mimetypes.guess_extension(content_type) or ".jpg")
            elif "." not in safe_name:
                safe_name += mimetypes.guess_extension(content_type) or ""
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / safe_name
            written = 0
            with tempfile.NamedTemporaryFile(dir=out_dir, prefix=".download-", suffix=".part", delete=False) as file:
                temporary = Path(file.name)
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_size:
                        return ToolFailure(f"❌ File exceeded {limit_mb} MB limit during download",
                                           {"failure_type": "size_limit"})
                    file.write(chunk)
            if not written:
                return ToolFailure("❌ Download produced empty content", {"failure_type": "empty_download"})
            os.replace(temporary, path)
            return f"✅ Downloaded: {path} ({written / 1024 / 1024:.2f} MB, {content_type or 'unknown'})"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def _exec_download_file(url: str, filename: str = "") -> str:
    try:
        return await asyncio.to_thread(_download_bounded, url, filename, image=False)
    except Exception as exc:
        return ToolFailure(f"❌ Download failed: {exc}", {"failure_type": type(exc).__name__})


async def _exec_download_image(url: str, filename: str = "") -> str:
    try:
        return await asyncio.to_thread(_download_bounded, url, filename, image=True)
    except Exception as exc:
        return ToolFailure(f"❌ Download failed: {exc}", {"failure_type": type(exc).__name__})


async def _exec_convert_document(file_path: str, preview_lines: int = 60) -> str:
    """Convert a document to markdown, save it, and return path plus preview."""
    try:
        from content_fetch.documents import convert_document
        from provenance.runtime import _wrap_external

        if not os.path.isfile(file_path):
            return ToolFailure(f"❌ File not found: {file_path}", {"failure_type": "file_missing"})

        try:
            text = await asyncio.to_thread(convert_document, file_path, 0)
        except Exception as conv_err:
            logger.error("convert_document inner error: %s", conv_err)
            return ToolFailure(f"❌ Conversion failed: {conv_err}", {"path": "document", "failure_type": type(conv_err).__name__})
        if not text or not text.strip():
            return ToolFailure("❌ Conversion produced empty content.", {"empty": True, "extracted_chars": 0, "failure_type": "empty_extraction"})

        out_dir = Path(_project_root()) / "data" / "converted"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{Path(file_path).stem}.md"
        out_path.write_text(text, encoding="utf-8")

        lines = text.splitlines()
        total_lines = len(lines)
        total_chars = len(text)
        preview = "\n".join(lines[:preview_lines])
        wrapped_preview = _wrap_external(preview, f"document:{file_path}")

        return ToolResult(
            f"✅ Converted -> {out_path}\n"
            f"   {total_lines} lines, {total_chars} chars\n"
            f"   Use read_file with offset/limit to paginate.\n\n"
            f"── preview (first {min(preview_lines, total_lines)} lines) ──\n"
            f"{wrapped_preview}",
            {"path": "document", "extracted_chars": total_chars, "empty": False},
        )
    except Exception as exc:
        logger.error("convert_document error: %s", exc)
        return ToolFailure(f"❌ Document conversion failed: {exc}", {"path": "document", "failure_type": type(exc).__name__})


FETCH_TOOL_HANDLERS = {
    "fetch_url": _exec_fetch_url,
    "download_file": _exec_download_file,
    "download_image": _exec_download_image,
    "convert_document": _exec_convert_document,
}
