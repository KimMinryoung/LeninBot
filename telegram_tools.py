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
]


# ── Tool Execution Functions ─────────────────────────────────────────

async def _exec_vector_search(query: str, num_results: int = 5, layer: str | None = None) -> str:
    """Execute vector similarity search via chatbot module."""
    try:
        from shared import similarity_search
        docs = await asyncio.to_thread(similarity_search, query, num_results, layer)
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


async def _exec_fetch_url(url: str) -> str:
    """Fetch and extract main body text from a URL."""
    try:
        from shared import fetch_url_content
        content = await asyncio.to_thread(fetch_url_content, url)
        return content or "Failed to extract content from this URL."
    except Exception as e:
        logger.error("fetch_url error: %s", e)
        return f"URL fetch failed: {e}"


async def _exec_download_image(url: str, filename: str = "") -> str:
    """Download an image from a URL and save locally."""
    import re
    import time
    import mimetypes
    from pathlib import Path

    project_root = os.path.dirname(os.path.abspath(__file__))
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


async def _exec_read_file(path: str, line_start: int | None = None, line_end: int | None = None, **kwargs) -> str:
    # Accept common LLM misspellings
    if line_start is None and "startline" in kwargs:
        line_start = kwargs["startline"]
    if line_end is None:
        line_end = kwargs.get("endline") or kwargs.get("end_line") or kwargs.get("lineend")
    """Read a file on the server."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(path):
        path = os.path.join(project_root, path)
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    if os.path.isdir(path):
        return f"Error: Path is a directory: {path}"
    # Block direct access to .env files (credentials protection)
    _basename = os.path.basename(path)
    if _basename == ".env" or _basename.startswith(".env."):
        return "Error: Access to .env files is blocked for security reasons."
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


async def _exec_patch_file(path: str, old_str: str, new_str: str) -> str:
    """Surgically modify a file by replacing a specific text block."""
    project_root = os.path.dirname(os.path.abspath(__file__))
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
    project_root = os.path.dirname(os.path.abspath(__file__))

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
            from telegram_mission import get_active_mission, get_mission_events, close_mission
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
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return "Error: TAVILY_API_KEY not set."
    max_results = max(1, min(max_results, 10))
    try:
        from tavily import AsyncTavilyClient
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
        return "\n\n".join(lines)
    except Exception as e:
        logger.error("Tavily search error: %s", e)
        return f"Web search failed: {e}"


# ── Restart Service Tool ─────────────────────────────────────────────

RESTART_SERVICE_TOOL = {
    "name": "restart_service",
    "description": (
        "Safely restart a leninbot service (telegram or api). "
        "Pre-restart validation: 1) syntax check all .py files with uncommitted changes, "
        "2) import-level check on key entry points (telegram_bot.py, api.py). "
        "If validation fails, restart is blocked and errors are returned. "
        "Use this instead of execute_python + subprocess for restarts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "enum": ["telegram", "api", "all"],
                "description": "Which service to restart. Default: telegram.",
            },
        },
        "required": [],
    },
}


async def _exec_restart_service(service: str = "telegram") -> str:
    """Safely restart service with pre-flight validation."""
    import ast
    import subprocess

    project_root = os.path.dirname(os.path.abspath(__file__))

    try:
        from telegram_bot import _runtime_state
        from telegram_tasks import persist_task_restart_state
        current_task_id = (_runtime_state or {}).get("current_task_id")
    except Exception:
        current_task_id = None
        persist_task_restart_state = None

    if service not in ("telegram", "api", "all"):
        return f"❌ Unknown service: {service}. Use: telegram, api, all"

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
    }
    targets = ["telegram", "api"] if service == "all" else [service]

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

    # 4. All checks passed — restart
    svc_map = {
        "telegram": ["leninbot-telegram"],
        "api": ["leninbot-api"],
        "all": ["leninbot-api", "leninbot-telegram"],  # API first, telegram last
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
TOOL_HANDLERS = {
    "vector_search": _exec_vector_search,
    "knowledge_graph_search": _exec_kg_search,
    "web_search": _exec_web_search,
    "fetch_url": _exec_fetch_url,
    "download_image": _exec_download_image,
    "read_file": _exec_read_file,
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

    url = await asyncio.to_thread(upload_to_r2, path, key, content_type)
    if not url:
        return "R2 upload failed. Check R2 env config."

    # Get current task context for tracking
    task_id = None
    agent_type = None
    try:
        from telegram_bot import _runtime_state
        task_id = (_runtime_state or {}).get("current_task_id")
    except Exception:
        pass

    # Register in file_registry
    try:
        db_execute(
            "INSERT INTO file_registry (local_path, public_url, filename, content_type, description, category, file_size, created_by_task_id, created_by_agent) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (path, url, filename, content_type, description or filename, category, file_size, task_id, agent_type),
        )
    except Exception as e:
        logger.warning("file_registry insert failed: %s", e)

    return f"Uploaded: {url}\nLocal: {path}\nSize: {file_size} bytes\nCategory: {category}"


TOOLS.append(UPLOAD_TO_R2_TOOL)
TOOL_HANDLERS["upload_to_r2"] = _exec_upload_to_r2

# ── Send Email Tool ──────────────────────────────────────────────────
def _load_email_signature() -> dict | None:
    """Load email signature config from config/email_signature.json."""
    sig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "email_signature.json")
    try:
        import json as _json
        with open(sig_path, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        name = cfg.get("name", "")
        email_addr = cfg.get("email", "")
        website_url = cfg.get("website_url", "")
        website_display = cfg.get("website_display", website_url)
        logo_url = cfg.get("logo_url", "")
        logo_width = cfg.get("logo_width", 200)

        text_sig = f"{name}\n{email_addr}"
        if website_display:
            text_sig += f"\n{website_display}"

        html_parts = [
            '<br><br>',
            '<table cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;font-family:Arial,sans-serif;">',
        ]
        if logo_url:
            html_parts.append(
                f'<tr><td style="padding:0 0 8px 0;">'
                f'<img src="{logo_url}" alt="{name}" width="{logo_width}" style="display:block;border:0;"></td></tr>'
            )
        html_parts.append(f'<tr><td style="font-size:14px;font-weight:700;color:#111;padding:0 0 2px 0;">{name}</td></tr>')
        if email_addr:
            html_parts.append(f'<tr><td style="font-size:13px;color:#333;padding:0 0 2px 0;"><a href="mailto:{email_addr}" style="color:#333;text-decoration:none;">{email_addr}</a></td></tr>')
        if website_url:
            html_parts.append(f'<tr><td style="font-size:13px;color:#333;padding:0 0 2px 0;"><a href="{website_url}" style="color:#333;text-decoration:none;">{website_display}</a></td></tr>')
        html_parts.append('</table>')
        html_sig = "\n".join(html_parts)

        return {"text": text_sig, "html": html_sig}
    except Exception as e:
        logger.warning("Failed to load email signature: %s", e)
        return None


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

    # Load email signature and append to body
    sig = _load_email_signature()
    if sig:
        body = body.rstrip() + "\n\n--\n" + sig["text"]
        sig_html = sig["html"]
        if html_body:
            html_body = html_body + sig_html
        else:
            # Wrap plain text body in basic HTML + signature
            escaped_body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            html_body = f"<div style='font-family:sans-serif;font-size:14px;'>{escaped_body}</div>{sig_html}"

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

# ── Finance data tool (real-time market prices) ──────────────────────
from finance_data import FINANCE_TOOL, FINANCE_TOOL_HANDLER

TOOLS.append(FINANCE_TOOL)
TOOL_HANDLERS["get_finance_data"] = FINANCE_TOOL_HANDLER

# ── Image generation tool (Replicate) ─────────────────────────────────
GENERATE_IMAGE_TOOL = {
    "name": "generate_image",
        "description": (
            "Generate an image using Replicate FLUX models. "
            "Returns prediction_id, model, final prompt, image URL, and local file path. "
            "Supports optional reference_image input: pass a downloaded local image path, remote URL, or data URI. "
            "If reference_image is provided, the backend routes to a Replicate image-editing model compatible with input_image. "
            "Styles: poster (Soviet propaganda), game (game concept art), pixel (retro game key art). "
            "You design the prompt; this tool executes it."
        ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Image description prompt (English). Will be prefixed with style aesthetics automatically.",
            },
            "style": {
                "type": "string",
                "enum": ["poster", "game", "pixel"],
                "description": "Visual style preset. Default: poster.",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "match_input_image"],
                "description": "Output aspect ratio. Default: 1:1. Use match_input_image when editing from a reference photo.",
            },
            "model": {
                "type": "string",
                "enum": ["flux_schnell", "flux_dev", "flux_kontext_dev"],
                "description": "Model preset. flux_schnell (fast), flux_dev (higher quality), or flux_kontext_dev (reference-image editing). Default: flux_schnell.",
            },
            "count": {
                "type": "integer",
                "description": "Number of images to generate in one batch (1-4). Single API call, no rate limit concern. Default: 1.",
            },
            "reference_image": {
                "type": "string",
                "description": "Optional reference image. Prefer a downloaded local file path under the project root. Remote URL and data URI are also accepted. When set, backend uses input_image-compatible Replicate model routing.",
            },
        },
        "required": ["prompt"],
    },
}


_last_image_gen_time: float = 0.0
_IMAGE_GEN_INTERVAL: float = 8.0  # seconds between API calls


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
    from replicate_image_service import generate_image, is_replicate_configured
    if not is_replicate_configured():
        return "ERROR: REPLICATE_API_TOKEN is not configured"

    count = max(1, min(4, count))

    # Rate limit: wait if called too soon after previous generation
    elapsed = _time.monotonic() - _last_image_gen_time
    if elapsed < _IMAGE_GEN_INTERVAL and _last_image_gen_time > 0:
        wait = _IMAGE_GEN_INTERVAL - elapsed
        await asyncio.sleep(wait)

    try:
        result = await generate_image(
            prompt,
            model=model,
            style=style,
            aspect_ratio=aspect_ratio,
            num_outputs=count,
            download=True,
            reference_image=reference_image,
        )
        _last_image_gen_time = _time.monotonic()
        urls = result.get("image_urls", [])
        local_paths = result.get("local_paths", [])
        lines = [
            f"Batch generated {len(urls)} image(s) in 1 API call.",
            f"  prediction_id: {result.get('prediction_id')}",
            f"  model: {result.get('model')}",
            f"  style: {style}",
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
    except Exception as e:
        _last_image_gen_time = _time.monotonic()
        return f"Image generation failed: {e}"


TOOLS.append(GENERATE_IMAGE_TOOL)
TOOL_HANDLERS["generate_image"] = _exec_generate_image