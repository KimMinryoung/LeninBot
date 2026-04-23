"""telegram_tools.py — Tool definitions and execution handlers for Telegram bot.

Extracted from telegram_bot.py for modularity.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime

from secrets_loader import get_secret

logger = logging.getLogger(__name__)

# ── Tool Definitions (Anthropic API format) ──────────────────────────
TOOLS = [
    {
        "name": "vector_search",
        "description": (
            "Search Marxist-Leninist document DB (pgvector). Returns excerpts with "
            "author/year/title. MATCH YOUR QUERY LANGUAGE TO THE LAYER: "
            "core_theory is English-language classics (Marx, Engels, Lenin, Mao, "
            "Trotsky translations) → query in English. modern_analysis is Korean "
            "analysis/commentary → query in Korean. Cross-language queries return "
            "near-empty results due to embedding-space separation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Use English for layer=core_theory, Korean for "
                        "layer=modern_analysis. Mismatching language to layer degrades recall sharply."
                    ),
                },
                "num_results": {"type": "integer", "description": "Results count (1-10).", "default": 5},
                "layer": {
                    "type": "string",
                    "enum": ["core_theory", "modern_analysis"],
                    "description": (
                        "core_theory: English-language Marxist-Leninist classics. "
                        "modern_analysis: Korean-language contemporary analysis/commentary. "
                        "Omit to search all layers (not recommended — mixes languages)."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "knowledge_graph_search",
        "description": "Search Neo4j KG for geopolitical entities and relationships.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What entities/relations to find."},
                "num_results": {"type": "integer", "description": "Results count (1-20).", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web via Tavily API. Returns relevant snippets with URLs. Use for current events, real-time data, fact-checking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "description": "Number of results (1-10).", "default": 5},
            },
            "required": ["query"],
        },
    },
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
        "description": "Download URL → data/downloads/. Returns local path. For PDFs/docs to feed convert_document. Max 100 MB.",
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
    # ── File system tools (Hetzner VPS) ──
    {
        "name": "read_file",
        "description": "Read a text file with line numbers. Format: 'LINE|CONTENT'. Use offset+limit for big files (default limit 500, max 2000). Reads >100K chars are rejected — paginate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or project-relative."},
                "offset": {"type": "integer", "description": "Start line, 1-indexed. Default 1."},
                "limit": {"type": "integer", "description": "Max lines. Default 500, max 2000."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": "Ripgrep-backed search. target='content' regex-searches inside files; target='files' finds files by glob (sorted by mtime). Use instead of execute_python+grep/find. output_mode: content|files_only|count. file_glob filters which files to search (e.g. '*.py'). Default path = project root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex (content) or glob (files), e.g. '5\\.1\\.2' or '*.md'."},
                "target": {"type": "string", "enum": ["content", "files"], "description": "Default 'content'."},
                "path": {"type": "string", "description": "Dir or file to search. Default: project root."},
                "file_glob": {"type": "string", "description": "Filter files in content mode (e.g. '*.py')."},
                "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "description": "Default 'content'."},
                "context": {"type": "integer", "description": "Context lines around match. Default 2."},
                "ignore_case": {"type": "boolean", "description": "Default false."},
                "limit": {"type": "integer", "description": "Max results. Default 50."},
                "offset": {"type": "integer", "description": "Skip N results. Default 0."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "write_file",
        "description": "Write ENTIRE content to a file. WARNING: overwrites the whole file. For modifying existing code, prefer patch_file instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write."},
                "content": {"type": "string", "description": "Content to write."},
                "mode": {"type": "string", "enum": ["overwrite", "append"], "description": "Write mode. Default: overwrite."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "patch_file",
        "description": "Surgically modify a file by replacing a specific block of text. Safer than write_file — only changes the matched portion, preserving the rest. Supports fuzzy matching. Use this for all code modifications.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to patch."},
                "old_str": {"type": "string", "description": "The exact text block to find and replace. Include enough context for a unique match."},
                "new_str": {"type": "string", "description": "The replacement text."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and directories on the server. Supports glob patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path. Default: project root."},
                "pattern": {"type": "string", "description": "Glob pattern filter. Default: * (all)."},
                "recursive": {"type": "boolean", "description": "Search recursively. Default: false."},
            },
            "required": [],
        },
    },
    {
        "name": "execute_python",
        "description": "Execute Python code on the server. Returns stdout/stderr. Use for data processing, calculations, or system tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
                "timeout": {"type": "integer", "description": "Max execution time in seconds (5-300). Default: 30."},
            },
            "required": ["code"],
        },
    },
    {
        "name": "convert_document",
        "description": "Local PDF/DOCX/PPTX/XLSX/HTML → markdown saved to data/converted/. Returns path + head preview; use read_file to paginate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path."},
            },
            "required": ["file_path"],
        },
    },
]


# ── Tool Execution Functions ─────────────────────────────────────────

async def _exec_vector_search(query: str, num_results: int = 5, layer: str | None = None) -> str:
    """Execute vector similarity search via chatbot module."""
    try:
        from shared import similarity_search
        docs = await asyncio.to_thread(similarity_search, query, num_results, layer, rerank=True)
        if not docs:
            return "No documents found."
        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = f"[{i}] {meta.get('title', 'Untitled')} — {meta.get('author', 'Unknown')}"
            if meta.get("year"):
                header += f" ({meta['year']})"
            results.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(results)
    except Exception as e:
        logger.error("vector_search error: %s", e)
        return f"Vector search failed: {e}"


async def _exec_kg_search(query: str, num_results: int = 10) -> str:
    """Execute knowledge graph search via chatbot module."""
    try:
        from shared import search_knowledge_graph
        result = await asyncio.to_thread(search_knowledge_graph, query, num_results)
        return result or "No knowledge graph results found."
    except Exception as e:
        logger.error("kg_search error: %s", e)
        return f"Knowledge graph search failed: {e}"


async def _exec_convert_document(file_path: str, preview_lines: int = 60) -> str:
    """Convert a document to markdown, save to data/converted/, return path + preview.

    The full markdown is written to disk so the agent can paginate via
    read_file (line_start/line_end) instead of dumping the whole document
    into the tool result. We return only basic metadata + a head preview.
    """
    try:
        from pathlib import Path
        from shared import convert_document, _wrap_external

        if not os.path.isfile(file_path):
            return f"❌ File not found: {file_path}"

        try:
            text = await asyncio.to_thread(convert_document, file_path, 0)  # 0 = unlimited
        except Exception as conv_err:
            logger.error("convert_document inner error: %s", conv_err)
            return f"❌ Conversion failed: {conv_err}"
        if not text:
            return "❌ Conversion produced empty content."

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = Path(project_root) / "data" / "converted"
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(file_path).stem
        out_path = out_dir / f"{stem}.md"
        out_path.write_text(text, encoding="utf-8")

        lines = text.splitlines()
        total_lines = len(lines)
        total_chars = len(text)
        preview = "\n".join(lines[:preview_lines])
        wrapped_preview = _wrap_external(preview, f"document:{file_path}")

        return (
            f"✅ Converted → {out_path}\n"
            f"   {total_lines} lines, {total_chars} chars\n"
            f"   Use read_file with offset/limit to paginate.\n\n"
            f"── preview (first {min(preview_lines, total_lines)} lines) ──\n"
            f"{wrapped_preview}"
        )
    except Exception as e:
        logger.error("convert_document error: %s", e)
        return f"❌ Document conversion failed: {e}"


# ── Publish Research Tool ────────────────────────────────────────────

PUBLISH_RESEARCH_TOOL = {
    "name": "publish_research",
    "description": (
        "Write a markdown document to the public research directory. "
        "Published files are served at https://cyber-lenin.com/reports/research/{filename}. "
        "Use for polished analysis reports, forecasts, and investigative findings. "
        "Filename is auto-generated from the title with date prefix (YYYYMMDD_slug.md)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Document title. Used for both the H1 heading and the filename slug.",
            },
            "content": {
                "type": "string",
                "description": "Full markdown content of the document (without the title heading — it is auto-prepended).",
            },
            "filename": {
                "type": "string",
                "description": "Optional custom filename (e.g. 'my_report.md'). If omitted, auto-generated from title.",
            },
        },
        "required": ["title", "content"],
    },
}
TOOLS.append(PUBLISH_RESEARCH_TOOL)


async def _exec_publish_research(title: str, content: str, filename: str | None = None) -> str:
    """Write a markdown research document to the public research/ directory."""
    import re
    import unicodedata
    from pathlib import Path
    from datetime import datetime, timezone, timedelta

    KST = timezone(timedelta(hours=9))
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    research_dir = os.path.join(project_root, "research")

    if not title or not title.strip():
        return "Error: title is required."
    if not content or not content.strip():
        return "Error: content is required."

    title = title.strip()
    now = datetime.now(KST)
    date_prefix = now.strftime("%Y%m%d")

    if filename:
        fname = filename.strip()
        if not fname.endswith(".md"):
            fname += ".md"
    else:
        # Generate slug from title
        slug = unicodedata.normalize("NFKD", title)
        slug = slug.encode("ascii", "ignore").decode("ascii").lower()
        slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
        slug = slug[:80] if slug else "research"
        fname = f"{date_prefix}_{slug}.md"

    # Prevent path traversal
    if "/" in fname or "\\" in fname or ".." in fname:
        return "Error: filename must not contain path separators or '..'."

    os.makedirs(research_dir, exist_ok=True)
    filepath = os.path.join(research_dir, fname)
    is_overwrite = os.path.exists(filepath)

    # Build document with header
    author_line = f"**작성자:** Cyber-Lenin (사이버-레닌)"
    date_line = f"**작성일:** {now.strftime('%Y-%m-%d')}"
    full_doc = f"# {title}\n{author_line}\n{date_line}\n\n---\n\n{content.strip()}\n"

    try:
        await asyncio.to_thread(Path(filepath).write_text, full_doc, encoding="utf-8")
    except Exception as e:
        logger.error("publish_research write error: %s", e)
        return f"Failed to write research document: {e}"

    # Frontend caches the research list (TTL) and individual entries (permanent) in Redis.
    # Invalidate both so the new/updated file and its real title appear on /reports immediately.
    cache_invalidated = False
    try:
        from redis_state import get_redis
        r = get_redis()
        if r:
            keys_to_drop = ["report:research_list"]
            if is_overwrite:
                keys_to_drop.append(f"research:{fname}")
            r.delete(*keys_to_drop)
            cache_invalidated = True
    except Exception as e:
        logger.warning("publish_research cache invalidation failed for %s: %s", fname, e)

    public_url = f"https://cyber-lenin.com/reports/research/{fname}"
    status = "Overwrote" if is_overwrite else "Published"
    cache_note = " (frontend cache invalidated)" if cache_invalidated else ""
    return (
        f"{status}: {fname}{cache_note}\n"
        f"Local path: {filepath}\n"
        f"Public URL: {public_url}\n"
        f"Size: {len(full_doc)} chars"
    )


async def _exec_fetch_url(url: str) -> str:
    """Fetch and extract main body text from a URL."""
    try:
        from shared import fetch_url_content_async, _wrap_external
        content = await fetch_url_content_async(url)
        if not content:
            return "Failed to extract content from this URL."
        return _wrap_external(content, f"url:{url}")
    except Exception as e:
        logger.error("fetch_url error: %s", e)
        return f"URL fetch failed: {e}"


async def _exec_download_file(url: str, filename: str = "") -> str:
    """Download an arbitrary file from a URL and save under data/downloads/."""
    import re
    import time
    import mimetypes
    from urllib.parse import urlparse, unquote
    from pathlib import Path

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = Path(project_root) / "data" / "downloads"
    MAX_SIZE = 100 * 1024 * 1024  # 100 MB

    def _download():
        import requests as _req

        with _req.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"}, stream=True) as resp:
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
            content_length = int(resp.headers.get("Content-Length", "0") or 0)
            if content_length and content_length > MAX_SIZE:
                return f"❌ File too large ({content_length / 1024 / 1024:.1f} MB > 100 MB limit)"

            # Determine filename
            if filename:
                safe_name = re.sub(r"[^a-zA-Z0-9가-힣._-]+", "-", filename).strip("-")[:120]
            else:
                url_path = unquote(urlparse(url).path)
                base = os.path.basename(url_path) or time.strftime("%Y%m%d_%H%M%S")
                safe_name = re.sub(r"[^a-zA-Z0-9가-힣._-]+", "-", base).strip("-")[:120]
                if "." not in safe_name:
                    ext = mimetypes.guess_extension(content_type) or ""
                    safe_name += ext

            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / safe_name

            written = 0
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > MAX_SIZE:
                        f.close()
                        path.unlink(missing_ok=True)
                        return f"❌ File exceeded 100 MB limit during download"
                    f.write(chunk)

            size_mb = written / 1024 / 1024
            return f"✅ Downloaded: {path} ({size_mb:.2f} MB, {content_type or 'unknown'})"

    try:
        return await asyncio.to_thread(_download)
    except Exception as e:
        logger.error("download_file error: %s", e)
        return f"❌ Download failed: {e}"


async def _exec_download_image(url: str, filename: str = "") -> str:
    """Download an image from a URL and save locally."""
    import re
    import time
    import mimetypes
    from pathlib import Path

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = Path(project_root) / "data" / "reference_images"

    def _download():
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
    except Exception as e:
        logger.error("download_image error: %s", e)
        return f"❌ Download failed: {e}"


_READ_DEFAULT_LIMIT = 500
_READ_MAX_LIMIT = 2000
_READ_MAX_CHARS = 100_000


async def _exec_read_file(
    path: str,
    offset: int | None = None,
    limit: int | None = None,
    **kwargs,
) -> str:
    """Read a text file with line numbers and pagination."""
    # Backward-compat aliases (line_start/line_end and common misspellings)
    if offset is None:
        offset = (
            kwargs.get("line_start")
            or kwargs.get("startline")
            or kwargs.get("start_line")
        )
    line_end = (
        kwargs.get("line_end")
        or kwargs.get("endline")
        or kwargs.get("end_line")
        or kwargs.get("lineend")
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(path):
        path = os.path.join(project_root, path)
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    if os.path.isdir(path):
        return f"Error: Path is a directory: {path}"
    _basename = os.path.basename(path)
    if _basename == ".env" or _basename.startswith(".env."):
        return "Error: Access to .env files is blocked for security reasons."

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"

    total = len(lines)
    start = max(1, offset or 1)
    if line_end is not None:
        end = min(total, line_end)
    else:
        eff_limit = limit if limit else _READ_DEFAULT_LIMIT
        eff_limit = min(eff_limit, _READ_MAX_LIMIT)
        end = min(total, start + eff_limit - 1)

    selected = lines[start - 1 : end]
    body = "".join(selected)
    if len(body) > _READ_MAX_CHARS:
        return (
            f"Error: read range {start}-{end} is {len(body)} chars (>{_READ_MAX_CHARS}). "
            f"Use a smaller limit or narrower offset range."
        )

    numbered = [f"{start + i:>6}|{line.rstrip()}" for i, line in enumerate(selected)]
    header = f"[{path}] lines {start}-{end} of {total}"
    if end < total:
        header += f"  (next: offset={end + 1})"
    body = "\n".join(numbered)

    # Wrap network-sourced files (downloaded or converted from external docs)
    # so they're treated as data, not instructions.
    rel = os.path.relpath(path, project_root) if path.startswith(project_root) else path
    if rel.startswith("data/downloads/") or rel.startswith("data/converted/"):
        from shared import _wrap_external
        body = _wrap_external(body, f"file:{rel}")

    return header + "\n" + body


async def _exec_search_files(
    pattern: str,
    target: str = "content",
    path: str | None = None,
    file_glob: str | None = None,
    output_mode: str = "content",
    context: int = 2,
    ignore_case: bool = False,
    limit: int = 50,
    offset: int = 0,
    **kwargs,
) -> str:
    """Ripgrep-backed search. target=content (regex in files) or files (glob by name)."""
    import subprocess
    import shlex

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_path = path or project_root
    if not os.path.isabs(search_path):
        search_path = os.path.join(project_root, search_path)
    if not os.path.exists(search_path):
        return f"Error: path not found: {search_path}"

    if target == "files":
        # Glob-by-name file search using rg --files. pattern is a glob.
        cmd = ["rg", "--files", "--hidden", "--glob", "!.git", "--glob", pattern, search_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return "Error: search timed out"
        if proc.returncode not in (0, 1):
            return f"Error: rg exited {proc.returncode}: {proc.stderr.strip()}"
        files = [l for l in proc.stdout.splitlines() if l.strip()]
        # Sort by mtime desc
        try:
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        except OSError:
            pass
        total = len(files)
        page = files[offset : offset + limit]
        if not page:
            return f"[search files /{pattern}/ in {search_path}] 0 matches"
        header = f"[search files /{pattern}/ in {search_path}] {total} match(es), showing {offset + 1}-{offset + len(page)}"
        if offset + len(page) < total:
            header += f"  (next: offset={offset + len(page)})"
        return header + "\n" + "\n".join(page)

    # target == "content" — regex search inside files
    cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
    if ignore_case:
        cmd.append("-i")
    if file_glob:
        cmd += ["-g", file_glob]
    if output_mode == "files_only":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:  # content
        if context and context > 0:
            cmd += ["-C", str(context)]
    cmd += ["-e", pattern, search_path]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    if proc.returncode == 1:
        return f"[search /{pattern}/ in {search_path}] 0 matches"
    if proc.returncode != 0:
        return f"Error: rg exited {proc.returncode}: {proc.stderr.strip()}"

    out_lines = proc.stdout.splitlines()
    total = len(out_lines)
    page = out_lines[offset : offset + limit]
    header = f"[search {output_mode} /{pattern}/ in {search_path}] {total} line(s), showing {offset + 1}-{offset + len(page)}"
    if offset + len(page) < total:
        header += f"  (next: offset={offset + len(page)})"
    body = "\n".join(page)

    # Wrap matches when searching network-sourced content (consistent with read_file).
    rel = os.path.relpath(search_path, project_root) if search_path.startswith(project_root) else search_path
    if rel.startswith("data/downloads/") or rel.startswith("data/converted/") or rel in ("data/downloads", "data/converted"):
        from shared import _wrap_external
        body = _wrap_external(body, f"search:{rel}")

    return header + "\n" + body


_WRITE_ALLOWED_DIRS = ["research", "docs", "logs", "temp_dev", "data"]
_WRITE_ALLOWED_EXTENSIONS = [".md", ".txt", ".json", ".csv", ".log", ".yaml", ".yml"]


async def _exec_write_file(path: str, content: str, mode: str = "overwrite") -> str:
    """Write content to a file on the server.

    Safety rules:
    - .py files → routed through self_modification_core (Git backup + syntax check)
    - Other files in allowed dirs → written directly
    - Files outside project root → blocked
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(path):
        path = os.path.join(project_root, path)
    abs_path = os.path.realpath(path)

    # Block writes outside project root
    if not (abs_path == project_root or abs_path.startswith(project_root + "/")):
        return f"❌ Write denied: path is outside project root"

    rel_path = os.path.relpath(abs_path, project_root)
    ext = os.path.splitext(abs_path)[1].lower()

    # .py files → safe modification path (Git backup + syntax check)
    if ext == ".py":
        try:
            sys.path.insert(0, project_root)
            from self_modification_core import (
                git_backup_before_modification,
                apply_line_patch_safe,
                run_sandbox_tests,
                git_reset_to_commit,
            )
            # Read original (if exists)
            old_content = ""
            if os.path.isfile(abs_path):
                with open(abs_path, "r", encoding="utf-8") as f:
                    old_content = f.read()

            # Syntax check new content first
            import ast
            try:
                ast.parse(content)
            except SyntaxError as e:
                return f"❌ Syntax error in new content: {e}"

            # Git backup
            if os.path.isfile(abs_path):
                commit_hash = git_backup_before_modification(abs_path)
            else:
                commit_hash = None

            # Write the file
            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Run sandbox tests
            test_results = run_sandbox_tests(abs_path)
            if test_results.status == "fail":
                # Rollback
                if commit_hash:
                    git_reset_to_commit(commit_hash)
                elif os.path.isfile(abs_path):
                    os.unlink(abs_path)
                return f"❌ Sandbox tests failed — rolled back.\n{test_results}"

            size = os.path.getsize(abs_path)
            backup_info = f", backup: {os.path.basename(commit_hash)}" if commit_hash else ""
            # Clean up backup file on success
            if commit_hash and os.path.isfile(commit_hash):
                os.unlink(commit_hash)
            return f"✅ Written {len(content)} chars to {rel_path} (size: {size}B{backup_info}, tests: PASS)"
        except Exception as e:
            return f"Error writing .py file safely: {e}"

    # Non-code files: allow in specific dirs or with safe extensions
    rel_parts = rel_path.replace("\\", "/").split("/")
    in_allowed_dir = any(rel_parts[0] == d for d in _WRITE_ALLOWED_DIRS) if rel_parts else False
    has_safe_ext = ext in _WRITE_ALLOWED_EXTENSIONS

    if not (in_allowed_dir or has_safe_ext):
        return (
            f"❌ Write denied: {rel_path}\n"
            f"Allowed dirs: {_WRITE_ALLOWED_DIRS}\n"
            f"Allowed extensions: {_WRITE_ALLOWED_EXTENSIONS}\n"
            f"For .py files, write is allowed with automatic safety checks."
        )

    try:
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        write_mode = "a" if mode == "append" else "w"
        with open(abs_path, write_mode, encoding="utf-8") as f:
            f.write(content)
        size = os.path.getsize(abs_path)
        return f"Written {len(content)} chars to {rel_path} (size: {size}B, mode: {mode})"
    except Exception as e:
        return f"Error writing file: {e}"


async def _exec_patch_file(path: str, old_str: str, new_str: str) -> str:
    """Surgically modify a file by replacing a specific text block."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(path):
        path = os.path.join(project_root, path)
    abs_path = os.path.realpath(path)

    if not (abs_path == project_root or abs_path.startswith(project_root + "/")):
        return "❌ Patch denied: path is outside project root"

    rel_path = os.path.relpath(abs_path, project_root)
    ext = os.path.splitext(abs_path)[1].lower()

    try:
        from patch_file import replace_block
        result = replace_block(abs_path, old_str, new_str, backup=True)

        if not result["ok"]:
            return f"❌ Patch failed: {result['message']}"

        # .py 파일은 구문 검사
        if ext == ".py":
            import ast
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                # 롤백: .bak 파일에서 복원
                bak_path = abs_path + ".bak"
                if os.path.isfile(bak_path):
                    import shutil
                    shutil.copy2(bak_path, abs_path)
                return f"❌ Syntax error after patch — rolled back: {e}"

        diff_preview = result["diff"][:1000] if result["diff"] else "(no diff)"
        return f"✅ Patched {rel_path}\n{diff_preview}"

    except Exception as e:
        return f"Error patching file: {e}"


async def _exec_list_directory(path: str = "", pattern: str = "*", recursive: bool = False) -> str:
    """List files and directories on the server."""
    import glob as _glob
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not path:
        path = project_root
    elif not os.path.isabs(path):
        path = os.path.join(project_root, path)
    if not os.path.isdir(path):
        return f"Error: Not a directory: {path}"
    try:
        if recursive:
            search = os.path.join(path, "**", pattern)
            entries = _glob.glob(search, recursive=True)
        else:
            search = os.path.join(path, pattern)
            entries = _glob.glob(search)
        entries.sort()
        lines = []
        for entry in entries[:200]:
            try:
                stat = os.stat(entry)
                kind = "DIR " if os.path.isdir(entry) else "FILE"
                size = f"{stat.st_size:>10,}" if not os.path.isdir(entry) else "         -"
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                rel = os.path.relpath(entry, path)
                lines.append(f"  {kind} {size}  {mtime}  {rel}")
            except OSError:
                lines.append(f"  ???? {os.path.relpath(entry, path)}")
        header = f"[{path}] {len(entries)} entries"
        if len(entries) > 200:
            header += f" (showing first 200)"
        return header + "\n" + "\n".join(lines)
    except Exception as e:
        return f"Error listing directory: {e}"


_BLOCKED_CODE_PATTERNS = [
    # Destructive file operations
    "shutil.rmtree", "os.rmdir", "os.removedirs",
    # System-level danger
    "os.system(", "os.exec",
    # Credential/env exfiltration — safe_env filtering already protects values
    # (env var name blocking removed: prevented legitimate os.getenv usage)
    # .env direct open() patterns
    'open(".env")', "open('.env')",
    'open(".env.', "open('.env.",
    'open(f".env', "open(f'.env",
]

# Modules/builtins that should never be imported or called in sandboxed code
_BLOCKED_IMPORTS = {
    "ctypes", "multiprocessing", "signal", "resource", "pty",
}
_BLOCKED_FUNCTIONS = {
    "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "breakpoint", "exit", "quit", "input",
}


def _check_code_safety(code: str) -> str | None:
    """Return error message if code contains blocked patterns, None if safe."""
    import ast

    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    # 2. Blocked string patterns (credentials, destructive ops)
    for pattern in _BLOCKED_CODE_PATTERNS:
        if pattern in code:
            return f"Blocked: code contains '{pattern}'"

    # 3. AST walk for dangerous constructs
    for node in ast.walk(tree):
        # Block dangerous string literals (shell commands)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if any(d in val for d in ["rm -rf /", "rm -rf ~", "mkfs.", "dd if="]):
                return f"Blocked: destructive shell command in string literal"

        # Block dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return f"Blocked: import of '{alias.name}'"
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in _BLOCKED_IMPORTS:
                return f"Blocked: import from '{node.module}'"

        # Block dangerous function calls: exec(), eval(), __import__(), getattr(), etc.
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _BLOCKED_FUNCTIONS:
                return f"Blocked: call to '{name}()'"

    return None  # safe


async def _exec_execute_python(code: str, timeout: int = 30) -> str:
    """Execute Python code on the server with safety checks."""
    import subprocess
    import tempfile

    timeout = max(5, min(timeout, 300))
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Safety check before execution
    safety_err = _check_code_safety(code)
    if safety_err:
        return f"❌ Code execution blocked: {safety_err}"

    def _run():
        # Prepend common imports so the model doesn't need to remember them
        prelude = "import os, sys, json, subprocess, re\nsys.path.insert(0, %r)\n" % project_root
        full_code = prelude + code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=project_root,
            delete=False, encoding="utf-8",
        ) as f:
            f.write(full_code)
            tmp_path = f.name
        try:
            # Filter out sensitive env vars from subprocess
            _sensitive_keys = {
                "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "NEO4J_PASSWORD",
                "GEMINI_API_KEY", "OPENAI_API_KEY", "ADMIN_API_KEY",
                "DB_PASSWORD", "SUPABASE_KEY", "TAVILY_API_KEY",
                "AURA_NEO4J_PASSWORD",
            }
            safe_env = {
                k: v for k, v in os.environ.items()
                if k not in _sensitive_keys
            }
            safe_env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=timeout,
                cwd=project_root,
                env=safe_env,
            )
            parts = []
            if result.stdout.strip():
                parts.append(result.stdout.strip())
            if result.stderr.strip():
                parts.append(f"[stderr]\n{result.stderr.strip()}")
            return "\n".join(parts) if parts else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {timeout}s."
        finally:
            os.unlink(tmp_path)

    return await asyncio.to_thread(_run)


# ── Mission Tool ──────────────────────────────────────────────────────

MISSION_TOOL = {
    "name": "mission",
    "description": "Manage the active mission (shared context between chat and tasks). Use 'status' to check current mission, 'close' to end a completed mission.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "close"],
                "description": "status: view active mission + recent events. close: end the mission (use when the goal is fully achieved).",
            },
        },
        "required": ["action"],
    },
}


def build_mission_handler(user_id: int):
    """Create a mission tool handler bound to a specific user_id."""
    async def _handle(action: str, **_kwargs) -> str:
        try:
            from telegram.mission import get_active_mission, get_mission_events, close_mission
            if action == "status":
                mission = get_active_mission(user_id)
                if not mission:
                    return "No active mission."
                events = get_mission_events(mission["id"], limit=10)
                lines = [f"Mission #{mission['id']}: {mission['title']} [{mission['status']}]"]
                lines.append(f"Created: {mission['created_at']}")
                if events:
                    lines.append(f"\nTimeline ({len(events)} events):")
                    for e in events:
                        lines.append(f"  [{e['created_at']}] ({e['source']}) {e['event_type']}: {str(e['content'] or '')[:200]}")
                return "\n".join(lines)
            elif action == "close":
                mission = get_active_mission(user_id)
                if not mission:
                    return "No active mission to close."
                return close_mission(mission["id"])
            return f"Unknown mission action: {action}"
        except Exception as e:
            return f"Mission error: {e}"
    return _handle


# ── Web Search (Tavily) ──────────────────────────────────────────────

async def _exec_web_search(query: str, max_results: int = 5) -> str:
    """Search the web via Tavily API."""
    api_key = get_secret("TAVILY_API_KEY", "") or ""
    if not api_key:
        return "Error: TAVILY_API_KEY not set."
    max_results = max(1, min(max_results, 10))
    try:
        from tavily import AsyncTavilyClient
        from shared import _wrap_external
        client = AsyncTavilyClient(api_key=api_key)
        resp = await client.search(query, max_results=max_results)
        results = resp.get("results", [])
        if not results:
            return f"No results for: {query}"
        lines = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")[:500]
            lines.append(f"### {title}\n{url}\n{content}")
        return _wrap_external("\n\n".join(lines), f"web_search:{query}")
    except Exception as e:
        logger.error("Tavily search error: %s", e)
        return f"Web search failed: {e}"


# ── Restart Service Tool ─────────────────────────────────────────────

RESTART_SERVICE_TOOL = {
    "name": "restart_service",
    "description": (
        "Restart a leninbot service with pre-flight syntax + import checks. "
        "Use instead of execute_python+subprocess. "
        "File→service mapping (and detailed procedure) lives in the programmer agent prompt."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "enum": ["telegram", "api", "browser", "all"],
                "description": "telegram=bot+agents, api=web+a2a, browser=browser worker, all=multi-service code. Default: telegram.",
            },
        },
        "required": [],
    },
}


async def _exec_restart_service(service: str = "telegram") -> str:
    """Safely restart service with pre-flight validation."""
    import ast
    import subprocess

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        from telegram.bot import current_task_ctx
        from telegram.tasks import persist_task_restart_state
        ctx = current_task_ctx.get()
        current_task_id = ctx["task_id"] if ctx else None
    except Exception:
        current_task_id = None
        persist_task_restart_state = None

    if service not in ("telegram", "api", "browser", "all"):
        return f"❌ Unknown service: {service}. Use: telegram, api, browser, all"

    # 1. Find .py files with uncommitted changes (staged + unstaged)
    try:
        diff_result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--diff-filter=ACMR"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        # Also include untracked .py files that might be new
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        changed_files = set()
        for line in (diff_result.stdout + "\n" + untracked.stdout).strip().split("\n"):
            line = line.strip()
            if line.endswith(".py"):
                changed_files.add(line)
    except Exception as e:
        return f"❌ Failed to detect changed files: {e}"

    errors = []

    # 2. Syntax check all changed .py files
    for rel_path in sorted(changed_files):
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.isfile(abs_path):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=rel_path)
        except SyntaxError as e:
            errors.append(f"SyntaxError in {rel_path}:{e.lineno} — {e.msg}")

    if errors:
        return "❌ Restart blocked — syntax errors found:\n" + "\n".join(errors)

    # 3. Import-level validation: try importing the entry points in a subprocess
    entry_points = {
        "telegram": "telegram_bot",
        "api": "api",
        "browser": "browser_worker",
    }
    targets = ["telegram", "api", "browser"] if service == "all" else [service]

    for target in targets:
        module = entry_points[target]
        module_path = os.path.join(project_root, f"{module}.py")
        if not os.path.isfile(module_path):
            continue
        try:
            # Run a quick import check in isolated subprocess
            check_code = (
                f"import sys; sys.path.insert(0, {project_root!r}); "
                f"import importlib; importlib.import_module({module!r})"
            )
            result = subprocess.run(
                [sys.executable, "-c", check_code],
                capture_output=True, text=True, timeout=30,
                cwd=project_root,
                env={**os.environ, "PREFLIGHT_CHECK": "1"},
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                # Extract the last meaningful error line
                err_lines = [l for l in stderr.split("\n") if l.strip()]
                last_err = err_lines[-1] if err_lines else "unknown error"
                errors.append(f"Import check failed for {module}.py: {last_err}")
        except subprocess.TimeoutExpired:
            errors.append(f"Import check timed out for {module}.py (>30s)")
        except Exception as e:
            errors.append(f"Import check error for {module}.py: {e}")

    if errors:
        return "❌ Restart blocked — import errors found:\n" + "\n".join(errors)

    if current_task_id and persist_task_restart_state:
        try:
            persist_task_restart_state(
                current_task_id,
                service=service,
                phase="requested",
                mark_completed=False,
            )
        except Exception as e:
            return f"❌ Restart blocked — failed to persist durable restart state: {e}"

    # 4. All checks passed — daemon-reload (picks up any unit file changes), then restart
    try:
        subprocess.run(
            ["sudo", "-n", "systemctl", "daemon-reload"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        pass  # non-fatal: restart will still use previous unit config

    svc_map = {
        "telegram": ["leninbot-telegram"],
        "api": ["leninbot-api"],
        "browser": ["leninbot-browser"],
        "all": ["leninbot-api", "leninbot-browser", "leninbot-telegram"],  # API first, browser second, telegram last
    }
    results = []
    restart_failed = False
    for svc in svc_map[service]:
        try:
            proc = subprocess.run(
                ["sudo", "-n", "systemctl", "restart", svc],
                capture_output=True, text=True, timeout=15,
                start_new_session=True,
            )
            if proc.returncode == 0:
                results.append(f"✅ {svc}: restarted")
            else:
                restart_failed = True
                results.append(f"❌ {svc}: {proc.stderr.strip()}")
        except subprocess.TimeoutExpired:
            restart_failed = True
            results.append(f"⏱ {svc}: timeout")
        except Exception as e:
            restart_failed = True
            results.append(f"❌ {svc}: {e}")

    if current_task_id and persist_task_restart_state:
        try:
            persist_task_restart_state(
                current_task_id,
                service=service,
                phase="verification" if not restart_failed else "requested",
                mark_completed=not restart_failed,
                resumed_after_restart=not restart_failed,
                reentry_reason=(
                    "restart completed; next step is post-restart verification"
                    if not restart_failed
                    else "restart command failed; restart branch may retry after fix"
                ),
            )
        except Exception as e:
            results.append(f"⚠️ durable restart completion state update failed: {e}")

    checked_files = ", ".join(sorted(changed_files)[:10]) if changed_files else "(none)"
    return (
        f"Pre-flight checks passed (syntax + import OK, changed: {checked_files})\n"
        + "\n".join(results)
    )


# ── Handler Registry ─────────────────────────────────────────────────

def dedupe_tool_registry(tools: list[dict]) -> list[dict]:
    """Deduplicate tool registry entries by name while preserving first occurrence.

    Root cause for browser task #330: source code had been patched, but a worker can
    still start or keep running with an inconsistent import/lifecycle state. Keeping
    the registry itself unique makes every downstream caller safer, regardless of
    whether agent-level dedupe runs.
    """
    deduped: list[dict] = []
    seen_names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            deduped.append(tool)
            continue
        name = str(tool.get("name", "") or "").strip()
        if name and name in seen_names:
            logger.warning("Dropping duplicate tool from base registry: %s", name)
            continue
        if name:
            seen_names.add(name)
        deduped.append(tool)
    return deduped


TOOL_HANDLERS = {
    "vector_search": _exec_vector_search,
    "knowledge_graph_search": _exec_kg_search,
    "web_search": _exec_web_search,
    "fetch_url": _exec_fetch_url,
    "convert_document": _exec_convert_document,
    "publish_research": _exec_publish_research,
    "download_file": _exec_download_file,
    "download_image": _exec_download_image,
    "read_file": _exec_read_file,
    "search_files": _exec_search_files,
    "write_file": _exec_write_file,
    "patch_file": _exec_patch_file,
    "list_directory": _exec_list_directory,
    "execute_python": _exec_execute_python,
    "restart_service": _exec_restart_service,
}

# ── Restart service tool ──────────────────────────────────────────────
TOOLS.append(RESTART_SERVICE_TOOL)

# ── R2 Upload + File Registry ────────────────────────────────────────
UPLOAD_TO_R2_TOOL = {
    "name": "upload_to_r2",
    "description": (
        "Upload a local file to Cloudflare R2 and get a public URL. "
        "Automatically registers the file in the file_registry DB table so other agents can find it. "
        "Use for images, documents, or any file that needs a public URL (e.g. email attachments, web assets)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "local_path": {"type": "string", "description": "Absolute path to the local file."},
            "key": {"type": "string", "description": "Object key/path in R2 bucket (e.g. 'email-assets/logo.png'). Defaults to filename."},
            "description": {"type": "string", "description": "What this file is / what it's for."},
            "category": {
                "type": "string",
                "enum": ["email-asset", "image", "document", "research", "general"],
                "description": "File category for search. Default: general.",
            },
        },
        "required": ["local_path"],
    },
}


async def _exec_upload_to_r2(
    local_path: str, key: str | None = None, description: str = "", category: str = "general",
) -> str:
    from shared import upload_to_r2
    from db import execute as db_execute, query as db_query
    import mimetypes

    path = os.path.abspath(local_path)
    if not os.path.isfile(path):
        return f"File not found: {local_path}"

    filename = os.path.basename(path)
    file_size = os.path.getsize(path)
    content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

    if key is None:
        key = f"{category}/{filename}" if category != "general" else filename

    # Check if already registered by local_path or R2 key
    existing = await asyncio.to_thread(
        db_query,
        "SELECT id, public_url FROM file_registry WHERE local_path = %s OR public_url LIKE %s LIMIT 1",
        (path, f"%/{key}"),
    )
    if existing:
        return f"Already registered: {existing[0]['public_url']}\n(file_registry id: {existing[0]['id']})"

    url = await asyncio.to_thread(upload_to_r2, path, key, content_type)
    if not url:
        return "R2 upload failed. Check R2 env config."

    # Get current task context for tracking
    task_id = None
    agent_type = None
    try:
        from telegram.bot import current_task_ctx
        ctx = current_task_ctx.get()
        task_id = ctx["task_id"] if ctx else None
    except Exception:
        pass

    # Register in file_registry
    registry_id = None
    try:
        reg_rows = await asyncio.to_thread(
            db_query,
            "INSERT INTO file_registry (local_path, public_url, filename, content_type, description, category, file_size, created_by_task_id, created_by_agent) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (path, url, filename, content_type, description or filename, category, file_size, task_id, agent_type),
        )
        registry_id = reg_rows[0]["id"] if reg_rows else None
    except Exception as e:
        logger.warning("file_registry insert failed: %s", e)

    reg_line = f"\nfile_registry id: {registry_id}" if registry_id else "\n(file_registry registration failed)"
    return f"Uploaded: {url}\nLocal: {path}\nSize: {file_size} bytes\nCategory: {category}{reg_line}"


TOOLS.append(UPLOAD_TO_R2_TOOL)
TOOL_HANDLERS["upload_to_r2"] = _exec_upload_to_r2

# ── Send Email Tool ──────────────────────────────────────────────────
def _load_email_signature_config() -> dict:
    """Load email signature config from config/email_signature.json."""
    sig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "email_signature.json")
    try:
        import json as _json
        with open(sig_path, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except Exception as e:
        logger.warning("Failed to load email signature config: %s", e)
        return {}


def _signature_mode_from_config(cfg: dict) -> str:
    mode = str(cfg.get("insertion_mode", "html_only") or "html_only").strip().lower()
    if mode not in {"html_only", "plain_text_only", "both", "none"}:
        mode = "html_only"
    return mode


def _load_email_signature() -> dict | None:
    cfg = _load_email_signature_config()
    if not cfg:
        return None

    mode = _signature_mode_from_config(cfg)
    if mode == "none":
        return None

    enabled = cfg.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return None

    name = str(cfg.get("name", "") or "").strip()
    email_addr = str(cfg.get("email", "") or "").strip()
    website_url = str(cfg.get("website_url", "") or "").strip()
    website_display = str(cfg.get("website_display", website_url) or website_url).strip()
    logo_url = str(cfg.get("logo_url") or "").strip()
    logo_width = int(cfg.get("logo_width", 200) or 200)

    text_lines = [line for line in [name, email_addr, website_display] if line]
    text_sig = "\n".join(text_lines)

    # Build text info column
    info_lines = []
    if name:
        info_lines.append(f'<td style="font-size:15px;font-weight:700;color:#111;padding:0 0 4px 0;">{name}</td>')
    if email_addr:
        info_lines.append(f'<td style="font-size:13px;color:#555;padding:0 0 3px 0;"><a href="mailto:{email_addr}" style="color:#555;text-decoration:none;">{email_addr}</a></td>')
    if website_url:
        info_lines.append(f'<td style="font-size:13px;color:#555;padding:0 0 3px 0;"><a href="{website_url}" style="color:#555;text-decoration:none;">{website_display}</a></td>')
    info_html = "".join(f"<tr>{line}</tr>" for line in info_lines)

    # Horizontal layout: logo left + text right, inside a bordered box
    # Gmail/Outlook strip border-radius and padding on <table>, so use
    # a wrapping <td> with explicit padding and inline border on each side.
    logo_td = ""
    if logo_url:
        logo_td = (
            f'<td valign="middle" width="{logo_width}" style="padding:12px 14px 12px 12px;">'
            f'<img src="{logo_url}" alt="{name}" width="{logo_width}" height="{logo_width}" '
            f'style="display:block;border:0;outline:none;text-decoration:none;"></td>'
        )
    html_sig = (
        '<br><br>'
        '<table cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;font-family:Arial,Helvetica,sans-serif;">'
        '<tr><td style="border:1px solid #dddddd;padding:0;">'
        '<table cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;">'
        f'<tr>{logo_td}'
        '<td valign="middle" style="padding:12px 12px 12px 0;">'
        f'<table cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;">{info_html}</table>'
        '</td></tr></table>'
        '</td></tr></table>'
    )

    return {
        "text": text_sig,
        "html": html_sig,
        "mode": mode,
        "config": cfg,
        "logo_url": logo_url,
    }


SEND_EMAIL_TOOL = {
    "name": "send_email",
    "description": (
        "Send an email as Cyber-Lenin via Resend API. "
        "Supports plain text and HTML body. Use html_body for rich formatting with images. "
        "Image URLs from upload_to_r2 can be embedded in html_body with <img> tags. "
        "All sent emails are recorded in the email_messages DB table."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recipient email addresses.",
            },
            "subject": {"type": "string", "description": "Email subject line."},
            "body": {"type": "string", "description": "Plain text body."},
            "html_body": {"type": "string", "description": "Optional HTML body. If provided, this is sent as the primary content."},
            "reply_to_message_id": {"type": "integer", "description": "Optional: inbound email_messages.id to reply to. Sets In-Reply-To header and thread."},
        },
        "required": ["to", "subject", "body"],
    },
}


async def _exec_send_email(
    to: list[str], subject: str, body: str, html_body: str = "", reply_to_message_id: int | None = None,
) -> str:
    from email_bridge import (
        CONFIG, email_sending_is_configured, get_email_message,
    )
    from db import execute as db_execute, query as db_query
    import json as _json

    if not email_sending_is_configured():
        return "Email sending not configured. Check RESEND_API_KEY and EMAIL_SMTP_FROM_EMAIL in .env."

    original_body = body or ""
    original_html_body = html_body or ""

    # Load email signature and append through a single config-controlled path.
    # The caller must provide pure body content only; all signature insertion happens here.
    # To prevent duplicate signatures in clients like Gmail, plain text stays pure body
    # unless the operator explicitly selects a text-inserting mode in config.
    sig = _load_email_signature()
    if sig:
        sig_mode = sig.get("mode", "html_only")
        sig_text = sig.get("text", "")
        sig_html = sig.get("html", "")

        body = original_body
        if sig_mode in {"plain_text_only", "both"} and sig_text:
            body = original_body.rstrip() + "\n\n--\n" + sig_text

        if sig_mode in {"html_only", "both"} and sig_html:
            if original_html_body:
                html_body = original_html_body + sig_html
            else:
                escaped_body = original_body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                html_body = f"<div style='font-family:sans-serif;font-size:14px;'>{escaped_body}</div>{sig_html}"
                if sig_mode == "html_only":
                    # Keep plain text fallback free of signature duplication.
                    body = original_body
        else:
            html_body = original_html_body
    else:
        body = original_body
        html_body = original_html_body

    # If replying, look up the inbound message for threading
    in_reply_to = None
    thread_id = None
    if reply_to_message_id:
        inbound = await asyncio.to_thread(get_email_message, reply_to_message_id)
        if inbound:
            in_reply_to = inbound.get("external_message_id")
            thread_id = inbound.get("thread_id")

    # Record outbound in DB
    from_addr = f"{CONFIG.smtp_from_name} <{CONFIG.smtp_from_email}>"
    rows = await asyncio.to_thread(
        db_query,
        "INSERT INTO email_messages ("
        "  thread_id, provider, direction, status, mailbox, in_reply_to,"
        "  sender_email, sender_name, recipient_emails, subject,"
        "  text_body, html_body, metadata, created_at, updated_at"
        ") VALUES ("
        "  %s, %s, 'outbound', 'sending', 'outbox', %s,"
        "  %s, %s, %s::jsonb, %s,"
        "  %s, %s, '{}'::jsonb, NOW(), NOW()"
        ") RETURNING id",
        (
            thread_id, CONFIG.provider, in_reply_to,
            CONFIG.smtp_from_email, CONFIG.smtp_from_name, _json.dumps(to), subject,
            body, html_body or None,
        ),
    )
    message_id = rows[0]["id"] if rows else None

    # Send via Resend
    import resend
    resend.api_key = CONFIG.resend_api_key

    send_params = {
        "from": from_addr,
        "to": to,
        "subject": subject,
        "text": body,
    }
    if html_body:
        send_params["html"] = html_body
    if in_reply_to:
        send_params["headers"] = {"In-Reply-To": in_reply_to, "References": in_reply_to}

    try:
        result = resend.Emails.send(send_params)
        resend_id = result.get("id") if isinstance(result, dict) else str(result)
    except Exception as e:
        if message_id:
            await asyncio.to_thread(
                db_execute,
                "UPDATE email_messages SET status = 'failed', metadata = jsonb_build_object('error', %s), updated_at = NOW() WHERE id = %s",
                (str(e)[:500], message_id),
            )
        return f"Email send failed: {e}"

    if message_id:
        await asyncio.to_thread(
            db_execute,
            "UPDATE email_messages SET status = 'sent', sent_at = NOW(), external_message_id = %s, updated_at = NOW() WHERE id = %s",
            (resend_id, message_id),
        )

    return f"Email sent to {', '.join(to)}\nSubject: {subject}\nResend ID: {resend_id}"


TOOLS.append(SEND_EMAIL_TOOL)
TOOL_HANDLERS["send_email"] = _exec_send_email

# ── Self-awareness tools (shared memory access) ─────────────────────
from self_tools import SELF_TOOLS, SELF_TOOL_HANDLERS

TOOLS.extend(SELF_TOOLS)
TOOLS.append(MISSION_TOOL)
TOOL_HANDLERS.update(SELF_TOOL_HANDLERS)
TOOLS = dedupe_tool_registry(TOOLS)

# ── Finance data tool (real-time market prices) ──────────────────────
from finance_data import FINANCE_TOOL, FINANCE_TOOL_HANDLER

TOOLS.append(FINANCE_TOOL)
TOOL_HANDLERS["get_finance_data"] = FINANCE_TOOL_HANDLER

# ── Site publishing tools (hub curations + static pages for cyber-lenin.com) ──
from site_publishing import SITE_PUBLISHING_TOOLS, SITE_PUBLISHING_TOOL_HANDLERS

TOOLS.extend(SITE_PUBLISHING_TOOLS)
TOOL_HANDLERS.update(SITE_PUBLISHING_TOOL_HANDLERS)

# ── Direct SQL tool (programmer only; analyst etc. keep read_self/kg_search) ──
from db_tools import DB_TOOLS, DB_TOOL_HANDLERS

TOOLS.extend(DB_TOOLS)
TOOL_HANDLERS.update(DB_TOOL_HANDLERS)

# ── Crypto wallet tools (address + balance + swap + transfer + x402 pay) ───
from crypto_wallet import (
    WALLET_TOOL, WALLET_TOOL_HANDLER,
    SWAP_TOOL, SWAP_TOOL_HANDLER,
    TRANSFER_TOOL, TRANSFER_TOOL_HANDLER,
    PAY_AND_FETCH_TOOL, PAY_AND_FETCH_TOOL_HANDLER,
)

TOOLS.append(WALLET_TOOL)
TOOL_HANDLERS["check_wallet"] = WALLET_TOOL_HANDLER
TOOLS.append(SWAP_TOOL)
TOOL_HANDLERS["swap_eth_to_usdc"] = SWAP_TOOL_HANDLER
TOOLS.append(TRANSFER_TOOL)
TOOL_HANDLERS["transfer_usdc"] = TRANSFER_TOOL_HANDLER
TOOLS.append(PAY_AND_FETCH_TOOL)
TOOL_HANDLERS["pay_and_fetch"] = PAY_AND_FETCH_TOOL_HANDLER

# ── Image generation tool (Replicate) ─────────────────────────────────
def _build_generate_image_description() -> str:
    """Short generate_image description. The full per-model parameter schemas
    live in replicate_image_service and are only surfaced when the agent hits
    an actual parameter rejection — keeping them out of the system prompt saves
    ~1.5K tokens per turn for a tool most agents never call.
    """
    return (
        "Generate image via Replicate. Returns prediction_id, model, final "
        "prompt, image URL, local path. reference_image: FLUX editing only "
        "(local path / URL / data URI) — never with rd_fast / rd_plus."
    )


GENERATE_IMAGE_TOOL = {
    "name": "generate_image",
    "description": _build_generate_image_description(),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Image description prompt (English). The prompt is sent directly to the model with no automatic style prefix. Include all desired visual style, composition, and aesthetic details in the prompt itself.",
            },
            "style": {
                "type": "string",
                "enum": ["poster", "game", "pixel", "portrait", "detailed", "game_asset", "1_bit", "low_res", "mc_item", "default", "retro", "watercolor", "textured", "cartoon", "ui_element", "item_sheet", "character_turnaround", "environment", "isometric", "isometric_asset", "topdown_map", "topdown_asset", "classic", "topdown_item", "mc_texture", "skill_icon"],
                "description": "FLUX: poster | game | pixel. Retro Diffusion: default / retro / pixel / isometric_asset etc. (aliases accepted). Default: poster.",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "match_input_image"],
                "description": "Output aspect ratio. Default: 1:1. Use match_input_image when editing from a reference photo.",
            },
            "model": {
                "type": "string",
                "enum": ["flux_schnell", "flux_dev", "flux_kontext_dev", "rd_fast", "rd_plus"],
                "description": "Model preset. FLUX: flux_schnell (fast), flux_dev (higher quality), flux_kontext_dev (reference-image editing). Retro Diffusion: rd_fast (fast pixel art), rd_plus (higher quality pixel art). Default: flux_schnell.",
            },
            "count": {
                "type": "integer",
                "description": "Number of images to generate in one batch (1-4). Single API call, no rate limit concern. Default: 1.",
            },
            "reference_image": {
                "type": "string",
                "description": "Optional reference image for FLUX editing only. Prefer a downloaded local file path under the project root. Remote URL and data URI are also accepted. When set, backend uses input_image-compatible Replicate model routing. Do not use with rd_fast or rd_plus.",
            },
        },
        "required": ["prompt"],
    },
}


_last_image_gen_time: float = 0.0
_IMAGE_GEN_INTERVAL: float = 8.0  # seconds between API calls


def _is_retryable_image_error(message: str) -> bool:
    lowered = str(message or "").lower()
    retry_markers = (
        "throttled",
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "temporarily unstable",
        "temporarily unstable",
        "temporarily",
        "network",
    )
    return any(marker in lowered for marker in retry_markers)


async def _exec_generate_image(
    prompt: str,
    style: str = "poster",
    aspect_ratio: str = "1:1",
    model: str | None = None,
    count: int = 1,
    reference_image: str | None = None,
) -> str:
    import time as _time
    global _last_image_gen_time
    from replicate_image_service import (
        generate_image,
        is_replicate_configured,
        is_retro_diffusion_model,
        normalize_retro_diffusion_style,
    )
    if not is_replicate_configured():
        return "ERROR: REPLICATE_API_TOKEN is not configured"

    requested_count = max(1, min(4, count))
    retro_model = is_retro_diffusion_model(model)
    normalized_style = normalize_retro_diffusion_style(style) if retro_model else style
    effective_count = 1 if retro_model else requested_count
    sequential_mode = retro_model and requested_count > 1

    async def _wait_turn(delay_floor: float = 0.0) -> None:
        global _last_image_gen_time
        elapsed = _time.monotonic() - _last_image_gen_time
        minimum_wait = max(_IMAGE_GEN_INTERVAL, delay_floor)
        if elapsed < minimum_wait and _last_image_gen_time > 0:
            await asyncio.sleep(minimum_wait - elapsed)

    try:
        if retro_model and reference_image:
            return "Image generation failed: reference_image is not supported with Retro Diffusion presets (rd_fast, rd_plus)"

        if not sequential_mode:
            await _wait_turn()
            result = await generate_image(
                prompt,
                model=model,
                style=normalized_style,
                aspect_ratio=aspect_ratio,
                num_outputs=effective_count,
                download=True,
                reference_image=reference_image,
            )
            _last_image_gen_time = _time.monotonic()
            urls = result.get("image_urls", [])
            local_paths = result.get("local_paths", [])
            style_line = normalized_style if retro_model else style
            lines = [
                f"Generated {len(urls)} image(s) in 1 API call.",
                f"  prediction_id: {result.get('prediction_id')}",
                f"  model: {result.get('model')}",
                f"  style_requested: {style}",
                f"  style_effective: {style_line}",
                f"  final_prompt: {result.get('prompt', '')[:300]}",
            ]
            if reference_image:
                lines.append(f"  reference_image: {result.get('reference_image')}")
                lines.append(f"  reference_image_source: {result.get('reference_image_source')}")
            for i, url in enumerate(urls):
                lp = local_paths[i] if i < len(local_paths) else "N/A"
                lines.append(f"  [{i+1}] url: {url}")
                lines.append(f"      local_path: {lp}")
            return "\n".join(lines)

        lines = [
            f"Requested {requested_count} Retro Diffusion image(s); executing sequentially to avoid credit/rate-limit failures.",
            f"  model: {model}",
            f"  style_requested: {style}",
            f"  style_effective: {normalized_style}",
        ]
        success_count = 0
        for idx in range(requested_count):
            backoff = 2.5
            last_error: Exception | None = None
            for attempt in range(1, 4):
                await _wait_turn(delay_floor=backoff)
                try:
                    result = await generate_image(
                        prompt,
                        model=model,
                        style=normalized_style,
                        aspect_ratio=aspect_ratio,
                        num_outputs=1,
                        download=True,
                        reference_image=reference_image,
                    )
                    _last_image_gen_time = _time.monotonic()
                    success_count += 1
                    url = (result.get("image_urls") or [None])[0]
                    local_path = (result.get("local_paths") or [None])[0]
                    lines.append(f"  [{idx+1}] prediction_id: {result.get('prediction_id')}")
                    lines.append(f"      url: {url}")
                    lines.append(f"      local_path: {local_path}")
                    if attempt > 1:
                        lines.append(f"      attempts: {attempt}")
                    last_error = None
                    break
                except Exception as e:
                    _last_image_gen_time = _time.monotonic()
                    last_error = e
                    if attempt < 3 and _is_retryable_image_error(str(e)):
                        lines.append(f"  [{idx+1}] retrying after transient failure (attempt {attempt}/3): {e}")
                        backoff = min(backoff * 2, 20.0)
                        continue
                    lines.append(f"  [{idx+1}] failed: {e}")
                    break
            if last_error is not None:
                break
        lines.insert(1, f"  completed: {success_count}/{requested_count}")
        return "\n".join(lines)
    except Exception as e:
        _last_image_gen_time = _time.monotonic()
        return f"Image generation failed: {e}"


TOOLS.append(GENERATE_IMAGE_TOOL)
TOOL_HANDLERS["generate_image"] = _exec_generate_image

# ── AI Browser Automation (browser-use) ──────────────────────────────
BROWSE_WEB_TOOL = {
    "name": "browse_web",
    "description": (
        "AI-driven browser automation using browser-use. "
        "An AI agent will autonomously navigate websites, fill forms, click buttons, "
        "and extract information. Use for complex multi-step web interactions "
        "(e.g., login flows, form submissions, multi-page navigation, data extraction "
        "from dynamic sites). For simple page reads, prefer fetch_url (faster, cheaper)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural language description of what to do in the browser. Be specific about the goal and expected output.",
            },
            "start_url": {
                "type": "string",
                "description": "Optional URL to navigate to before starting the task.",
            },
            "max_steps": {
                "type": "integer",
                "description": "Maximum browser interaction steps (default: 20, max: 50).",
            },
        },
        "required": ["task"],
    },
}


async def _exec_browse_web(task: str, start_url: str | None = None, max_steps: int = 20, **_kw) -> str:
    try:
        from browser.use_agent import browse
        max_steps = max(1, min(int(max_steps), 50))
        result = await browse(task, max_steps=max_steps, start_url=start_url)

        parts = []
        if result["success"]:
            parts.append("[OK] Task completed")
        else:
            parts.append("[FAIL] Task did not complete successfully")

        parts.append(f"Steps: {result['steps']} | Duration: {result['duration_seconds']}s")

        if result["urls"]:
            parts.append(f"Visited: {', '.join(str(u) for u in result['urls'][:5])}")

        if result["result"]:
            text = result["result"]
            if len(text) > 15000:
                text = text[:15000] + f"\n... [truncated, total {len(result['result'])} chars]"
            parts.append(f"\n--- Result ---\n{text}")

        if result["extracted_content"]:
            content_str = "\n".join(str(c) for c in result["extracted_content"] if c)
            if content_str.strip():
                if len(content_str) > 10000:
                    content_str = content_str[:10000] + "\n... [truncated]"
                parts.append(f"\n--- Extracted ---\n{content_str}")

        if result["errors"]:
            errs = [str(e) for e in result["errors"] if e]
            if errs:
                parts.append(f"\nErrors: {'; '.join(errs[:3])}")

        return "\n".join(parts)
    except Exception as e:
        return f"browse_web error: {e}"


# browse_web: Available to browser agent via filter_tools.
# Orchestrator won't call it directly (delegated to browser agent per system prompt).
TOOLS.append(BROWSE_WEB_TOOL)
TOOL_HANDLERS["browse_web"] = _exec_browse_web


# ── check_inbox Tool ────────────────────────────────────────────────
CHECK_INBOX_TOOL = {
    "name": "check_inbox",
    "description": (
        "Read lenin@cyber-lenin.com INBOX + Junk. Returns subject, sender, "
        "date, folder, read status, body text, and any links. Unread → "
        "[UNREAD], junk → [JUNK]."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sender_filter": {
                "type": "string",
                "description": "Filter by sender address or domain (e.g. 'substack.com', 'platformer'). Optional.",
            },
            "subject_filter": {
                "type": "string",
                "description": "Filter by subject keyword (e.g. 'confirm', 'verify', 'sign in'). Optional.",
            },
            "unread_only": {
                "type": "boolean",
                "description": "If true, return only unread emails. Default: false.",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Max emails to return (default 5, max 20).",
                "default": 5,
            },
            "include_body": {
                "type": "boolean",
                "description": "If true, include extracted body text. Default: true.",
                "default": True,
            },
            "body_max_chars": {
                "type": "integer",
                "description": "Maximum extracted body characters per email (default 4000, max 12000).",
                "default": 4000,
            },
        },
        "required": [],
    },
}


def _imap_connect():
    """Create and return an authenticated IMAP connection."""
    import imaplib
    host = os.environ.get("EMAIL_IMAP_HOST", "")
    port = int(os.environ.get("EMAIL_IMAP_PORT", "993"))
    username = os.environ.get("EMAIL_IMAP_USERNAME", "")
    password = get_secret("EMAIL_IMAP_PASSWORD", "") or ""
    if not all([host, username, password]):
        return None
    conn = imaplib.IMAP4_SSL(host, port)
    conn.login(username, password)
    return conn


def _parse_email_message(raw_bytes, *, include_body: bool = True, body_max_chars: int = 4000):
    """Parse a raw email and return dict with subject, from, date, links, and extracted body text."""
    import email as _email
    from email.header import decode_header
    from html import unescape
    import re

    msg = _email.message_from_bytes(raw_bytes)

    subj_parts = decode_header(msg.get("Subject", ""))
    subject = ""
    for part, enc in subj_parts:
        if isinstance(part, bytes):
            subject += part.decode(enc or "utf-8", errors="replace")
        else:
            subject += part

    sender = msg.get("From", "")
    date = msg.get("Date", "")

    def _decode_payload(part):
        payload = part.get_payload(decode=True)
        if payload is None:
            raw = part.get_payload()
            if isinstance(raw, str):
                return raw
            if isinstance(raw, bytes):
                payload = raw
            else:
                return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except Exception:
            return payload.decode("utf-8", errors="replace")

    def _html_to_text(html: str) -> str:
        text = re.sub(r"<\s*br\s*/?>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"</\s*(p|div|li|tr|h[1-6])\s*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        text = text.replace("\xa0", " ")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    text_parts = []
    html_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = (part.get("Content-Disposition") or "").lower()
            if "attachment" in content_disposition:
                continue
            ct = (part.get_content_type() or "").lower()
            if ct == "text/plain":
                decoded = _decode_payload(part).strip()
                if decoded:
                    text_parts.append(decoded)
            elif ct == "text/html":
                decoded = _decode_payload(part).strip()
                if decoded:
                    html_parts.append(decoded)
    else:
        ct = (msg.get_content_type() or "").lower()
        decoded = _decode_payload(msg).strip()
        if ct == "text/html":
            html_parts.append(decoded)
        elif decoded:
            text_parts.append(decoded)

    raw_body_for_links = "\n\n".join([*html_parts, *text_parts])
    extracted_body = "\n\n".join(text_parts).strip()
    if not extracted_body and html_parts:
        extracted_body = "\n\n".join(_html_to_text(part) for part in html_parts if part.strip()).strip()
    if body_max_chars > 0 and extracted_body:
        extracted_body = extracted_body[:body_max_chars]

    links = re.findall(r'https?://[^\s<>")\']+', raw_body_for_links)
    seen = set()
    unique_links = []
    for lnk in links:
        cleaned = lnk.rstrip('.,);>\"\'')
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_links.append(cleaned)

    return {
        "subject": subject,
        "from": sender,
        "date": date,
        "links": unique_links[:50],
        "body": extracted_body if include_body else "",
        "body_truncated": bool(extracted_body) and body_max_chars > 0 and len(extracted_body) >= body_max_chars,
    }


async def _exec_check_inbox(
    sender_filter: str = "",
    subject_filter: str = "",
    unread_only: bool = False,
    limit: int = 5,
    include_body: bool = True,
    body_max_chars: int = 4000,
) -> str:
    """Check IMAP INBOX + Junk folders and extract readable body text plus links from recent emails."""
    limit = max(1, min(20, limit))
    body_max_chars = max(0, min(12000, body_max_chars))

    def _fetch():
        conn = _imap_connect()
        if conn is None:
            return "Error: IMAP credentials not configured in .env"

        results = []
        try:
            for folder in ["INBOX", "Junk"]:
                try:
                    status, _ = conn.select(folder, readonly=True)
                    if status != "OK":
                        continue
                except Exception:
                    continue

                search_criteria = "UNSEEN" if unread_only else "ALL"
                _, data = conn.search(None, search_criteria)
                all_ids = data[0].split()
                if not all_ids:
                    continue

                candidate_ids = all_ids[-(limit * 5):]
                candidate_ids.reverse()

                for mid in candidate_ids:
                    if len(results) >= limit:
                        break
                    _, msg_data = conn.fetch(mid, "(FLAGS RFC822)")
                    if not msg_data or not isinstance(msg_data[0], tuple):
                        continue
                    flags_raw = msg_data[0][0] if isinstance(msg_data[0][0], bytes) else b""
                    raw = msg_data[0][1]
                    if not raw:
                        continue
                    is_read = b"\\Seen" in flags_raw
                    parsed = _parse_email_message(raw, include_body=include_body, body_max_chars=body_max_chars)

                    if sender_filter and sender_filter.lower() not in parsed["from"].lower():
                        continue
                    if subject_filter and subject_filter.lower() not in parsed["subject"].lower():
                        continue

                    parsed["folder"] = folder
                    parsed["is_read"] = is_read
                    results.append(parsed)
        finally:
            conn.logout()

        results.sort(key=lambda x: x["date"], reverse=True)
        return results[:limit]

    try:
        result = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"IMAP error: {e}"

    if isinstance(result, str):
        return result
    if not result:
        return "No matching emails found."

    from shared import _wrap_external
    lines = []
    for i, em in enumerate(result, 1):
        tags = ""
        if not em.get("is_read"):
            tags += " [UNREAD]"
        if em["folder"] == "Junk":
            tags += " [JUNK]"
        lines.append(f"[{i}]{tags} {em['subject']}")
        lines.append(f"    From: {em['from']}")
        lines.append(f"    Date: {em['date']}")
        if include_body:
            body = (em.get("body") or "").strip()
            if body:
                suffix = " …[truncated]" if em.get("body_truncated") else ""
                lines.append(f"    Body:\n      {body.replace(chr(10), chr(10) + '      ')}{suffix}")
            else:
                lines.append("    Body: none")
        if em["links"]:
            lines.append(f"    Links ({len(em['links'])}):")
            for lnk in em["links"]:
                lines.append(f"      - {lnk}")
        else:
            lines.append("    Links: none")
        lines.append("")
    return _wrap_external("\n".join(lines), "imap_inbox")


# ── allowlist_sender Tool ───────────────────────────────────────────
ALLOWLIST_SENDER_TOOL = {
    "name": "allowlist_sender",
    "description": (
        "Move emails from a sender out of Junk into INBOX, preventing future spam filtering. "
        "Use after check_inbox shows [JUNK] emails from a legitimate sender."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sender_filter": {
                "type": "string",
                "description": "Sender address or domain to rescue from Junk (e.g. 'substack.com', 'noreply@platformer.news').",
            },
        },
        "required": ["sender_filter"],
    },
}


async def _exec_allowlist_sender(sender_filter: str) -> str:
    """Move all Junk emails matching sender_filter to INBOX."""
    def _move():
        conn = _imap_connect()
        if conn is None:
            return "Error: IMAP credentials not configured"

        status, _ = conn.select("Junk")
        if status != "OK":
            conn.logout()
            return "Junk folder not found or empty."

        _, data = conn.search(None, "ALL")
        all_ids = data[0].split()
        if not all_ids:
            conn.logout()
            return "Junk folder is empty."

        import email as _email
        from email.header import decode_header
        moved = 0
        for mid in all_ids:
            _, msg_data = conn.fetch(mid, "(RFC822.HEADER)")
            header_raw = msg_data[0][1]
            msg = _email.message_from_bytes(header_raw)
            sender = msg.get("From", "")
            if sender_filter.lower() not in sender.lower():
                continue
            # COPY to INBOX then flag for deletion in Junk
            conn.copy(mid, "INBOX")
            conn.store(mid, "+FLAGS", "(\\Deleted)")
            moved += 1

        conn.expunge()
        conn.logout()
        return f"Moved {moved} email(s) from Junk to INBOX matching '{sender_filter}'."

    try:
        return await asyncio.to_thread(_move)
    except Exception as e:
        return f"IMAP error: {e}"


TOOLS.append(CHECK_INBOX_TOOL)
TOOL_HANDLERS["check_inbox"] = _exec_check_inbox
TOOLS.append(ALLOWLIST_SENDER_TOOL)
TOOL_HANDLERS["allowlist_sender"] = _exec_allowlist_sender

# ── Diary Writer Tool ─────────────────────────────────────────────────
SAVE_DIARY_TOOL = {
    "name": "save_diary",
    "description": "Save a diary entry to the ai_diary table. Used by the diary agent to persist generated diary entries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "One-line title/summary of the diary entry (Korean)."},
            "content": {"type": "string", "description": "Full diary body text (Korean, 2+ paragraphs)."},
        },
        "required": ["title", "content"],
    },
}


async def _exec_save_diary(title: str, content: str) -> str:
    from db import execute as db_execute
    try:
        await asyncio.to_thread(
            db_execute,
            "INSERT INTO ai_diary (title, content) VALUES (%s, %s)",
            (title, content),
        )
        return f"Diary saved: {title}"
    except Exception as e:
        return f"Failed to save diary: {e}"


TOOLS.append(SAVE_DIARY_TOOL)
TOOL_HANDLERS["save_diary"] = _exec_save_diary

# ── Moltbook Tool (for scout agent) ──────────────────────────────────────────

MOLTBOOK_TOOL = {
    "name": "moltbook",
    "description": (
        "Run Moltbook operations via the Razvedchik agent script.\n"
        "Actions:\n"
        "- scan: Read-only feed scan — gather posts without interacting\n"
        "- patrol: Full patrol loop — scan + comment + post (default for general activity)\n"
        "- post: Write a new post to Moltbook\n"
        "- status: Check agent claim status\n"
        "- profile: View agent profile"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["scan", "patrol", "post", "status", "profile"],
                "description": "Which Moltbook operation to run.",
            },
            "topic": {
                "type": "string",
                "description": "Post title (for 'post' action). If omitted, auto-generated.",
            },
            "content": {
                "type": "string",
                "description": "Post body (for 'post' action). If omitted, auto-generated.",
            },
            "submolt": {
                "type": "string",
                "description": "Target submolt name (e.g. 'general', 'tech'). Optional.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of posts to scan (default: 20). For 'scan' and 'patrol'.",
            },
            "max_comments": {
                "type": "integer",
                "description": "Max comments to post during patrol (default: 5).",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Simulate without actual API writes (default: false).",
            },
        },
        "required": ["action"],
    },
}


async def _exec_moltbook(
    action: str = "patrol",
    topic: str = "",
    content: str = "",
    submolt: str = "",
    limit: int | None = None,
    max_comments: int | None = None,
    dry_run: bool = False,
    **_: dict,
) -> str:
    import subprocess, os

    cmd = [
        os.path.join(os.environ.get("PROJECT_ROOT", "/home/grass/leninbot"), "venv/bin/python"),
        os.path.join(os.environ.get("PROJECT_ROOT", "/home/grass/leninbot"), "agents/razvedchik/razvedchik.py"),
        f"--{action}",
    ]

    if topic:
        cmd.extend(["--topic", topic])
    if content:
        cmd.extend(["--content", content])
    if submolt:
        cmd.extend(["--submolt", submolt])
    if limit:
        cmd.extend(["--limit", str(limit)])
    if max_comments:
        cmd.extend(["--max-comments", str(max_comments)])
    if dry_run:
        cmd.append("--dry-run")

    env = {**os.environ, "PYTHONPATH": os.environ.get("PROJECT_ROOT", "/home/grass/leninbot")}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.environ.get("PROJECT_ROOT", "/home/grass/leninbot"),
            env=env,
            timeout=180,
        )
        output = result.stdout[-3000:] if result.stdout else ""
        if result.returncode != 0:
            stderr = result.stderr[-1000:] if result.stderr else ""
            output += f"\n[EXIT CODE {result.returncode}]\nSTDERR: {stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "[ERROR] Moltbook script timed out after 180 seconds."
    except Exception as e:
        return f"[ERROR] Failed to run Moltbook script: {e}"


TOOLS.append(MOLTBOOK_TOOL)
TOOL_HANDLERS["moltbook"] = _exec_moltbook


# ── A2A Client Tool ─────────────────────────────────────────────────
# Orchestrator-only: send a message to an external A2A-compatible agent.

A2A_SEND_TOOL = {
    "name": "a2a_send",
    "description": (
        "Send a SendMessage JSON-RPC request to an external A2A agent. "
        "Auto-discovers the agent's card at /.well-known/agent-card.json "
        "(v1.0) or /agent.json (legacy). Optional `skill_id` scopes the call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_url": {
                "type": "string",
                "description": "Base URL of the target agent (e.g. 'https://other-agent.com'). The /a2a endpoint is appended automatically.",
            },
            "message": {
                "type": "string",
                "description": "The message to send to the agent.",
            },
            "skill_id": {
                "type": "string",
                "description": "Optional skill ID to request from the target agent (passed as configuration.skillId).",
            },
            "timeout_sec": {
                "type": "integer",
                "description": "Timeout in seconds (default: 120).",
                "default": 120,
            },
            "discover": {
                "type": "boolean",
                "description": "If true, fetch and return the agent card instead of sending a message. When discover=true, message is not required.",
                "default": False,
            },
        },
        "required": ["agent_url"],
    },
}


async def _exec_a2a_send(
    agent_url: str,
    message: str = "",
    skill_id: str = "",
    timeout_sec: int = 120,
    discover: bool = False,
) -> str:
    """Send an A2A message to an external agent or discover its capabilities."""
    import httpx
    import uuid

    base = agent_url.rstrip("/")

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec, connect=10)) as client:
        # Discovery mode
        if discover:
            try:
                # v1.0 canonical path first, fallback to legacy
                r = await client.get(f"{base}/.well-known/agent-card.json")
                if r.status_code == 404:
                    r = await client.get(f"{base}/.well-known/agent.json")
                r.raise_for_status()
                card = r.json()
                name = card.get("name", "unknown")
                skills = card.get("skills", [])
                skill_list = ", ".join(s.get("id", "?") for s in skills) if skills else "none declared"
                desc = card.get("description", "")
                # Detect A2A endpoint from supportedInterfaces (v1.0) or url (legacy)
                interfaces = card.get("supportedInterfaces", [])
                endpoint = interfaces[0].get("url") if interfaces else card.get("url", base + "/a2a")
                return (
                    f"Agent: {name}\n"
                    f"Description: {desc}\n"
                    f"Endpoint: {endpoint}\n"
                    f"Skills: {skill_list}\n"
                    f"Full card:\n{json.dumps(card, indent=2, ensure_ascii=False)}"
                )
            except httpx.HTTPStatusError as e:
                return f"❌ Agent card fetch failed: HTTP {e.response.status_code}"
            except Exception as e:
                return f"❌ Agent card fetch failed: {e}"

        # SendMessage mode
        if not message.strip():
            return "❌ message is required when discover=false"

        msg_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "method": "SendMessage",
            "params": {
                "message": {
                    "messageId": msg_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": message}],
                },
            },
            "id": str(uuid.uuid4()),
        }

        if skill_id:
            payload["params"]["configuration"] = {"skillId": skill_id}

        try:
            r = await client.post(
                f"{base}/a2a",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            resp = r.json()
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500]
            return f"❌ A2A request failed: HTTP {e.response.status_code}\n{body}"
        except Exception as e:
            return f"❌ A2A request failed: {e}"

        # Parse response
        if "error" in resp:
            err = resp["error"]
            return f"❌ A2A error ({err.get('code')}): {err.get('message')}"

        result = resp.get("result", {})
        status = result.get("status", {})
        state = status.get("state", "unknown")

        # Extract agent reply text (accept both v1.0 and legacy role values)
        history = result.get("history", [])
        agent_reply = ""
        for msg in reversed(history):
            role = msg.get("role", "")
            if role in ("ROLE_AGENT", "agent"):
                parts = msg.get("parts", [])
                agent_reply = "\n".join(p.get("text", "") for p in parts if "text" in p)
                break

        if not agent_reply:
            artifacts = result.get("artifacts", [])
            for art in artifacts:
                for p in art.get("parts", []):
                    if "text" in p:
                        agent_reply += p["text"] + "\n"

        task_id = result.get("id", "?")
        meta = result.get("metadata", {})
        skill_used = meta.get("skillId", "general")

        header = f"[A2A response | state={state} | skill={skill_used} | task={task_id}]"
        return f"{header}\n\n{agent_reply.strip()}" if agent_reply.strip() else f"{header}\n\n(empty response)"


TOOLS.append(A2A_SEND_TOOL)
TOOL_HANDLERS["a2a_send"] = _exec_a2a_send


# ── Schema normalization ─────────────────────────────────────────────
#
# Runs last — after every module-level TOOLS.append/extend above — so every
# registered tool acquires ``additionalProperties: false`` unless it
# deliberately opts out. Effects per provider:
#   * llama-server: constrains grammar-based tool-call decoding so Qwen
#     can't emit parameter names outside the declared schema.
#   * Anthropic: treats it as advisory (no behavioral change).
#   * OpenAI: strict mode is enabled only when the schema is also
#     "strict-safe" (see openai_tool_loop._convert_tool_anthropic_to_openai).

def _normalize_tool_schemas_inplace(tools: list[dict]) -> None:
    for t in tools:
        schema = t.get("input_schema")
        if not isinstance(schema, dict):
            continue
        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False


_normalize_tool_schemas_inplace(TOOLS)
