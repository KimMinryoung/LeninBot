"""Fetch, download, and document-conversion runtime tools."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

FETCH_TOOLS = [
    {
        "name": "fetch_url",
        "description": "Fetch and extract body text from a URL. Use when the user shares a link and asks about its content. Returns up to 10,000 chars of cleaned body text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch content from."},
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
            "Use this when you need a reference photo for image generation."
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


async def _exec_fetch_url(url: str) -> str:
    """Fetch and extract main body text from a URL."""
    try:
        from content_fetch.urls import diagnose_url_fetch_failure, fetch_url_content_async
        from provenance.runtime import _wrap_external

        content = await fetch_url_content_async(url)
        if not content:
            return "Failed to extract content from this URL.\n" + diagnose_url_fetch_failure(url)
        return _wrap_external(content, f"url:{url}")
    except Exception as exc:
        logger.error("fetch_url error: %s", exc)
        try:
            from content_fetch.urls import diagnose_url_fetch_failure

            diagnosis = diagnose_url_fetch_failure(url, [str(exc)])
            return f"URL fetch failed: {exc}\n{diagnosis}"
        except Exception:
            return f"URL fetch failed: {exc}"


async def _exec_download_file(url: str, filename: str = "") -> str:
    """Download an arbitrary file from a URL and save under data/downloads/."""
    out_dir = Path(_project_root()) / "data" / "downloads"
    max_size = 100 * 1024 * 1024

    def _download() -> str:
        import requests as _req

        with _req.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"}, stream=True) as resp:
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
            content_length = int(resp.headers.get("Content-Length", "0") or 0)
            if content_length and content_length > max_size:
                return f"❌ File too large ({content_length / 1024 / 1024:.1f} MB > 100 MB limit)"

            if filename:
                safe_name = re.sub(r"[^a-zA-Z0-9가-힣._-]+", "-", filename).strip("-")[:120]
            else:
                url_path = unquote(urlparse(url).path)
                base = os.path.basename(url_path) or time.strftime("%Y%m%d_%H%M%S")
                safe_name = re.sub(r"[^a-zA-Z0-9가-힣._-]+", "-", base).strip("-")[:120]
                if "." not in safe_name:
                    safe_name += mimetypes.guess_extension(content_type) or ""

            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / safe_name

            written = 0
            with open(path, "wb") as file:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_size:
                        file.close()
                        path.unlink(missing_ok=True)
                        return "❌ File exceeded 100 MB limit during download"
                    file.write(chunk)

            size_mb = written / 1024 / 1024
            return f"✅ Downloaded: {path} ({size_mb:.2f} MB, {content_type or 'unknown'})"

    try:
        return await asyncio.to_thread(_download)
    except Exception as exc:
        logger.error("download_file error: %s", exc)
        return f"❌ Download failed: {exc}"


async def _exec_download_image(url: str, filename: str = "") -> str:
    """Download an image from a URL and save locally."""
    out_dir = Path(_project_root()) / "data" / "reference_images"

    def _download() -> str:
        import requests as _req

        resp = _req.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return f"❌ Not an image (Content-Type: {content_type})"

        ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".jpg"
        if filename:
            safe_name = re.sub(r"[^a-zA-Z0-9가-힣_-]+", "-", filename).strip("-")[:60]
        else:
            safe_name = time.strftime("%Y%m%d_%H%M%S")
        final_name = f"{safe_name}{ext}"

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / final_name
        path.write_bytes(resp.content)

        size_kb = len(resp.content) / 1024
        return f"✅ Downloaded: {path} ({size_kb:.0f} KB, {content_type})"

    try:
        return await asyncio.to_thread(_download)
    except Exception as exc:
        logger.error("download_image error: %s", exc)
        return f"❌ Download failed: {exc}"


async def _exec_convert_document(file_path: str, preview_lines: int = 60) -> str:
    """Convert a document to markdown, save it, and return path plus preview."""
    try:
        from content_fetch.documents import convert_document
        from provenance.runtime import _wrap_external

        if not os.path.isfile(file_path):
            return f"❌ File not found: {file_path}"

        try:
            text = await asyncio.to_thread(convert_document, file_path, 0)
        except Exception as conv_err:
            logger.error("convert_document inner error: %s", conv_err)
            return f"❌ Conversion failed: {conv_err}"
        if not text:
            return "❌ Conversion produced empty content."

        out_dir = Path(_project_root()) / "data" / "converted"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{Path(file_path).stem}.md"
        out_path.write_text(text, encoding="utf-8")

        lines = text.splitlines()
        total_lines = len(lines)
        total_chars = len(text)
        preview = "\n".join(lines[:preview_lines])
        wrapped_preview = _wrap_external(preview, f"document:{file_path}")

        return (
            f"✅ Converted -> {out_path}\n"
            f"   {total_lines} lines, {total_chars} chars\n"
            f"   Use read_file with offset/limit to paginate.\n\n"
            f"── preview (first {min(preview_lines, total_lines)} lines) ──\n"
            f"{wrapped_preview}"
        )
    except Exception as exc:
        logger.error("convert_document error: %s", exc)
        return f"❌ Document conversion failed: {exc}"


FETCH_TOOL_HANDLERS = {
    "fetch_url": _exec_fetch_url,
    "download_file": _exec_download_file,
    "download_image": _exec_download_image,
    "convert_document": _exec_convert_document,
}
