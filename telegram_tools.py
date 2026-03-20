"""telegram_tools.py — Tool definitions and execution handlers for Telegram bot.

Extracted from telegram_bot.py for modularity.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Tool Definitions (Anthropic API format) ──────────────────────────
TOOLS = [
    {
        "name": "vector_search",
        "description": "Search Marxist-Leninist document DB (pgvector). Returns excerpts with author/year/title.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (Korean or English)."},
                "num_results": {"type": "integer", "description": "Results count (1-10).", "default": 5},
                "layer": {
                    "type": "string",
                    "enum": ["core_theory", "modern_analysis"],
                    "description": "Filter: core_theory (classical) or modern_analysis. Omit for all.",
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
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5,
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
    # ── File system tools (Hetzner VPS) ──
    {
        "name": "read_file",
        "description": "Read a file on the server. Returns content with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (absolute or relative to project root)."},
                "line_start": {"type": "integer", "description": "Start line (1-based). Omit for beginning."},
                "line_end": {"type": "integer", "description": "End line (inclusive). Omit for end."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file on the server. Creates parent directories if needed.",
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
]


# ── Tool Execution Functions ─────────────────────────────────────────

async def _exec_vector_search(query: str, num_results: int = 5, layer: str | None = None) -> str:
    """Execute vector similarity search via chatbot module."""
    try:
        from chatbot import similarity_search
        docs = await asyncio.to_thread(similarity_search, query, num_results, layer)
        if not docs:
            return "No documents found."
        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = f"[{i}] {meta.get('title', 'Untitled')} — {meta.get('author', 'Unknown')}"
            if meta.get("year"):
                header += f" ({meta['year']})"
            results.append(f"{header}\n{doc.page_content[:500]}")
        return "\n\n".join(results)
    except Exception as e:
        logger.error("vector_search error: %s", e)
        return f"Vector search failed: {e}"


async def _exec_kg_search(query: str, num_results: int = 10) -> str:
    """Execute knowledge graph search via chatbot module."""
    try:
        from chatbot import search_knowledge_graph
        result = await asyncio.to_thread(search_knowledge_graph, query, num_results)
        return result or "No knowledge graph results found."
    except Exception as e:
        logger.error("kg_search error: %s", e)
        return f"Knowledge graph search failed: {e}"


async def _exec_fetch_url(url: str) -> str:
    """Fetch and extract main body text from a URL."""
    try:
        from shared import fetch_url_content
        content = await asyncio.to_thread(fetch_url_content, url)
        return content or "Failed to extract content from this URL."
    except Exception as e:
        logger.error("fetch_url error: %s", e)
        return f"URL fetch failed: {e}"


async def _exec_read_file(path: str, line_start: int | None = None, line_end: int | None = None) -> str:
    """Read a file on the server."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(path):
        path = os.path.join(project_root, path)
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    if os.path.isdir(path):
        return f"Error: Path is a directory: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        start = max(1, line_start or 1)
        end = min(total, line_end or total)
        selected = lines[start - 1 : end]
        numbered = [f"{start + i:>6}\t{line.rstrip()}" for i, line in enumerate(selected)]
        return f"[{path}] lines {start}-{end} of {total}\n" + "\n".join(numbered)
    except Exception as e:
        return f"Error reading file: {e}"


_WRITE_ALLOWED_DIRS = ["research", "docs", "logs", "temp_dev", "data"]
_WRITE_ALLOWED_EXTENSIONS = [".md", ".txt", ".json", ".csv", ".log", ".yaml", ".yml"]


async def _exec_write_file(path: str, content: str, mode: str = "overwrite") -> str:
    """Write content to a file on the server.

    Safety rules:
    - .py files → routed through self_modification_core (Git backup + syntax check)
    - Other files in allowed dirs → written directly
    - Files outside project root → blocked
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
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


async def _exec_list_directory(path: str = "", pattern: str = "*", recursive: bool = False) -> str:
    """List files and directories on the server."""
    import glob as _glob
    project_root = os.path.dirname(os.path.abspath(__file__))
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
    # Credential/env exfiltration
    "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "NEO4J_PASSWORD",
    "GEMINI_API_KEY", "OPENAI_API_KEY",
]


def _check_code_safety(code: str) -> str | None:
    """Return error message if code contains blocked patterns, None if safe."""
    import ast

    # 1. Syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    # 2. Blocked string patterns (credentials, destructive ops)
    for pattern in _BLOCKED_CODE_PATTERNS:
        if pattern in code:
            return f"Blocked: code contains '{pattern}'"

    # 3. AST walk for dangerous constructs
    tree = ast.parse(code)
    for node in ast.walk(tree):
        # Block rm -rf style via subprocess
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if any(d in val for d in ["rm -rf /", "rm -rf ~", "mkfs.", "dd if="]):
                return f"Blocked: destructive shell command in string literal"

    return None  # safe


async def _exec_execute_python(code: str, timeout: int = 30) -> str:
    """Execute Python code on the server with safety checks."""
    import subprocess
    import tempfile

    timeout = max(5, min(timeout, 300))
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Safety check before execution
    safety_err = _check_code_safety(code)
    if safety_err:
        return f"❌ Code execution blocked: {safety_err}"

    def _run():
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=project_root,
            delete=False, encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=timeout,
                cwd=project_root,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
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


# ── Handler Registry ─────────────────────────────────────────────────
TOOL_HANDLERS = {
    "vector_search": _exec_vector_search,
    "knowledge_graph_search": _exec_kg_search,
    "fetch_url": _exec_fetch_url,
    "read_file": _exec_read_file,
    "write_file": _exec_write_file,
    "list_directory": _exec_list_directory,
    "execute_python": _exec_execute_python,
}

# ── Self-awareness tools (shared memory access) ─────────────────────
from self_tools import SELF_TOOLS, SELF_TOOL_HANDLERS

TOOLS.extend(SELF_TOOLS)
TOOL_HANDLERS.update(SELF_TOOL_HANDLERS)
