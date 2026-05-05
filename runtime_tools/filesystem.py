"""Filesystem and local Python execution runtime tools."""

from __future__ import annotations

import asyncio
import datetime as _datetime
import glob
import os
import shutil
import subprocess
import sys
import tempfile

FILESYSTEM_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a text file with line numbers. Format: 'LINE|CONTENT'. Use offset+limit for line pagination (default limit 500, max 2000). For converted PDFs or markdown with awkward line breaks, use char_offset+char_limit for character pagination. Reads >100K chars are rejected — paginate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or project-relative."},
                "offset": {"type": "integer", "description": "Start line, 1-indexed. Default 1."},
                "limit": {"type": "integer", "description": "Max lines. Default 500, max 2000."},
                "char_offset": {"type": "integer", "description": "Optional 0-indexed character offset. Use for converted PDFs/markdown when line pagination is not useful."},
                "char_limit": {"type": "integer", "description": "Optional max characters for char_offset mode. Default 20000, max 100000."},
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
]

_READ_DEFAULT_LIMIT = 500
_READ_MAX_LIMIT = 2000
_READ_MAX_CHARS = 100_000
_BLOCKED_CONTEXT_FILES = {
    os.path.normpath("dev_docs/project_state.md"),
}

_WRITE_ALLOWED_DIRS = ["research", "docs", "logs", "temp_dev", "data"]
_WRITE_ALLOWED_EXTENSIONS = [".md", ".txt", ".json", ".csv", ".log", ".yaml", ".yml"]

_BLOCKED_CODE_PATTERNS = [
    "shutil.rmtree", "os.rmdir", "os.removedirs",
    "os.system(", "os.exec",
    'open(".env")', "open('.env')",
    'open(".env.', "open('.env.",
    'open(f".env', "open(f'.env",
]
_BLOCKED_IMPORTS = {
    "ctypes", "multiprocessing", "signal", "resource", "pty",
}
_BLOCKED_FUNCTIONS = {
    "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "breakpoint", "exit", "quit", "input",
}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _normalize_project_path(path: str, project_root: str | None = None) -> tuple[str, str]:
    project_root = project_root or _project_root()
    abs_path = path if os.path.isabs(path) else os.path.join(project_root, path)
    abs_path = os.path.normpath(abs_path)
    rel_path = os.path.relpath(abs_path, project_root) if abs_path.startswith(project_root) else abs_path
    return abs_path, os.path.normpath(rel_path)


def _is_blocked_context_file(path: str, project_root: str | None = None) -> bool:
    _, rel_path = _normalize_project_path(path, project_root)
    return rel_path in _BLOCKED_CONTEXT_FILES


async def _exec_read_file(
    path: str,
    offset: int | None = None,
    limit: int | None = None,
    char_offset: int | None = None,
    char_limit: int | None = None,
    **kwargs,
) -> str:
    """Read a text file with line numbers and pagination."""
    if offset is None:
        offset = (
            kwargs.get("line_start")
            or kwargs.get("startline")
            or kwargs.get("start_line")
        )
    if char_offset is None:
        char_offset = (
            kwargs.get("offset_chars")
            or kwargs.get("char_start")
            or kwargs.get("start_char")
        )
    if char_limit is None:
        char_limit = (
            kwargs.get("max_chars")
            or kwargs.get("limit_chars")
            or kwargs.get("chars_limit")
        )
    line_end = (
        kwargs.get("line_end")
        or kwargs.get("endline")
        or kwargs.get("end_line")
        or kwargs.get("lineend")
    )

    project_root = _project_root()
    path, rel_path = _normalize_project_path(path, project_root)
    if rel_path in _BLOCKED_CONTEXT_FILES:
        return (
            "Error: dev_docs/project_state.md is blocked from agent context. "
            "It is a stale human-maintained snapshot; use live tools, database state, "
            "or targeted source files instead."
        )
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    if os.path.isdir(path):
        return f"Error: Path is a directory: {path}"
    basename = os.path.basename(path)
    if basename == ".env" or basename.startswith(".env."):
        return "Error: Access to .env files is blocked for security reasons."

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            text = file.read()
    except Exception as exc:
        return f"Error reading file: {exc}"

    lines = text.splitlines(keepends=True)
    total = len(lines)
    rel = os.path.relpath(path, project_root) if path.startswith(project_root) else path
    is_external_file = rel.startswith("data/downloads/") or rel.startswith("data/converted/")

    def _externalize(body: str) -> str:
        if is_external_file:
            from provenance.runtime import _wrap_external

            return _wrap_external(body, f"file:{rel}")
        return body

    if char_offset is not None:
        try:
            start_char = max(0, int(char_offset or 0))
        except (TypeError, ValueError):
            start_char = 0
        try:
            eff_char_limit = int(char_limit) if char_limit else 20_000
        except (TypeError, ValueError):
            eff_char_limit = 20_000
        eff_char_limit = max(1, min(eff_char_limit, _READ_MAX_CHARS))
        end_char = min(len(text), start_char + eff_char_limit)
        if start_char >= len(text):
            return (
                f"[{path}] chars {start_char}-{start_char} of {len(text)}\n"
                f"Error: char_offset is beyond end of file. Last valid char_offset is {max(len(text) - 1, 0)}."
            )
        body = text[start_char:end_char]
        if len(body) > _READ_MAX_CHARS:
            return (
                f"Error: read range chars {start_char}-{end_char} is {len(body)} chars "
                f"(>{_READ_MAX_CHARS}). Use a smaller char_limit."
            )
        header = f"[{path}] chars {start_char}-{end_char} of {len(text)}"
        if end_char < len(text):
            header += f"  (next: char_offset={end_char})"
        return header + "\n" + _externalize(body)

    try:
        start = max(1, int(offset or 1))
    except (TypeError, ValueError):
        start = 1
    if start > total:
        return (
            f"[{path}] lines {start}-{start} of {total}\n"
            f"Error: offset is a 1-indexed line number and is beyond end of file. "
            f"Last valid line offset is {max(total, 1)}. "
            f"For character-based pagination, call read_file with char_offset and char_limit explicitly."
        )
    if line_end is not None:
        end = min(total, int(line_end))
    else:
        eff_limit = limit if limit else _READ_DEFAULT_LIMIT
        eff_limit = min(int(eff_limit), _READ_MAX_LIMIT)
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

    return header + "\n" + _externalize(body)


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
    """Ripgrep-backed search. target=content or files."""
    project_root = _project_root()
    search_path = path or project_root
    search_path, rel_search_path = _normalize_project_path(search_path, project_root)
    if rel_search_path in _BLOCKED_CONTEXT_FILES:
        return (
            "Error: dev_docs/project_state.md is blocked from agent context. "
            "Use live tools, database state, or targeted source files instead."
        )
    if not os.path.exists(search_path):
        return f"Error: path not found: {search_path}"

    if target == "files":
        cmd = [
            "rg", "--files", "--hidden",
            "--glob", "!.git",
            "--glob", "!dev_docs/project_state.md",
            "--glob", pattern,
            search_path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return "Error: search timed out"
        if proc.returncode not in (0, 1):
            return f"Error: rg exited {proc.returncode}: {proc.stderr.strip()}"
        files = [
            line for line in proc.stdout.splitlines()
            if line.strip() and not _is_blocked_context_file(line, project_root)
        ]
        try:
            files.sort(key=lambda item: os.path.getmtime(item), reverse=True)
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

    cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
    if ignore_case:
        cmd.append("-i")
    if file_glob:
        cmd += ["-g", file_glob]
    cmd += ["-g", "!dev_docs/project_state.md"]
    if output_mode == "files_only":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:
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

    rel = os.path.relpath(search_path, project_root) if search_path.startswith(project_root) else search_path
    if rel.startswith("data/downloads/") or rel.startswith("data/converted/") or rel in ("data/downloads", "data/converted"):
        from provenance.runtime import _wrap_external

        body = _wrap_external(body, f"search:{rel}")

    return header + "\n" + body


async def _exec_write_file(path: str, content: str, mode: str = "overwrite") -> str:
    """Write content to a file on the server."""
    project_root = _project_root()
    if not os.path.isabs(path):
        path = os.path.join(project_root, path)
    abs_path = os.path.realpath(path)

    if not (abs_path == project_root or abs_path.startswith(project_root + "/")):
        return "❌ Write denied: path is outside project root"

    rel_path = os.path.relpath(abs_path, project_root)
    ext = os.path.splitext(abs_path)[1].lower()

    if ext == ".py":
        try:
            sys.path.insert(0, project_root)
            from self_modification_core import (
                git_backup_before_modification,
                git_reset_to_commit,
                run_sandbox_tests,
            )

            import ast

            try:
                ast.parse(content)
            except SyntaxError as exc:
                return f"❌ Syntax error in new content: {exc}"

            if os.path.isfile(abs_path):
                commit_hash = git_backup_before_modification(abs_path)
            else:
                commit_hash = None

            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as file:
                file.write(content)

            test_results = run_sandbox_tests(abs_path)
            if test_results.status == "fail":
                if commit_hash:
                    git_reset_to_commit(commit_hash)
                elif os.path.isfile(abs_path):
                    os.unlink(abs_path)
                return f"❌ Sandbox tests failed — rolled back.\n{test_results}"

            size = os.path.getsize(abs_path)
            backup_info = f", backup: {os.path.basename(commit_hash)}" if commit_hash else ""
            if commit_hash and os.path.isfile(commit_hash):
                os.unlink(commit_hash)
            return f"✅ Written {len(content)} chars to {rel_path} (size: {size}B{backup_info}, tests: PASS)"
        except Exception as exc:
            return f"Error writing .py file safely: {exc}"

    rel_parts = rel_path.replace("\\", "/").split("/")
    in_allowed_dir = any(rel_parts[0] == directory for directory in _WRITE_ALLOWED_DIRS) if rel_parts else False
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
        with open(abs_path, write_mode, encoding="utf-8") as file:
            file.write(content)
        size = os.path.getsize(abs_path)
        return f"Written {len(content)} chars to {rel_path} (size: {size}B, mode: {mode})"
    except Exception as exc:
        return f"Error writing file: {exc}"


async def _exec_patch_file(path: str, old_str: str, new_str: str) -> str:
    """Surgically modify a file by replacing a specific text block."""
    project_root = _project_root()
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

        if ext == ".py":
            import ast

            try:
                with open(abs_path, "r", encoding="utf-8") as file:
                    ast.parse(file.read())
            except SyntaxError as exc:
                bak_path = abs_path + ".bak"
                if os.path.isfile(bak_path):
                    shutil.copy2(bak_path, abs_path)
                return f"❌ Syntax error after patch — rolled back: {exc}"

        diff_preview = result["diff"][:1000] if result["diff"] else "(no diff)"
        return f"✅ Patched {rel_path}\n{diff_preview}"

    except Exception as exc:
        return f"Error patching file: {exc}"


async def _exec_list_directory(path: str = "", pattern: str = "*", recursive: bool = False) -> str:
    """List files and directories on the server."""
    project_root = _project_root()
    if not path:
        path = project_root
    elif not os.path.isabs(path):
        path = os.path.join(project_root, path)
    if not os.path.isdir(path):
        return f"Error: Not a directory: {path}"
    try:
        if recursive:
            search = os.path.join(path, "**", pattern)
            entries = glob.glob(search, recursive=True)
        else:
            search = os.path.join(path, pattern)
            entries = glob.glob(search)
        entries.sort()
        lines = []
        for entry in entries[:200]:
            try:
                stat = os.stat(entry)
                kind = "DIR " if os.path.isdir(entry) else "FILE"
                size = f"{stat.st_size:>10,}" if not os.path.isdir(entry) else "         -"
                mtime = _datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                rel = os.path.relpath(entry, path)
                lines.append(f"  {kind} {size}  {mtime}  {rel}")
            except OSError:
                lines.append(f"  ???? {os.path.relpath(entry, path)}")
        header = f"[{path}] {len(entries)} entries"
        if len(entries) > 200:
            header += " (showing first 200)"
        return header + "\n" + "\n".join(lines)
    except Exception as exc:
        return f"Error listing directory: {exc}"


def _check_code_safety(code: str) -> str | None:
    """Return an error message if code contains blocked patterns."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Syntax error: {exc}"

    for pattern in _BLOCKED_CODE_PATTERNS:
        if pattern in code:
            return f"Blocked: code contains '{pattern}'"

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if any(danger in val for danger in ["rm -rf /", "rm -rf ~", "mkfs.", "dd if="]):
                return "Blocked: destructive shell command in string literal"

        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return f"Blocked: import of '{alias.name}'"
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in _BLOCKED_IMPORTS:
                return f"Blocked: import from '{node.module}'"

        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _BLOCKED_FUNCTIONS:
                return f"Blocked: call to '{name}()'"

    return None


async def _exec_execute_python(code: str, timeout: int = 30) -> str:
    """Execute Python code on the server with safety checks."""
    timeout = max(5, min(timeout, 300))
    project_root = _project_root()

    safety_err = _check_code_safety(code)
    if safety_err:
        return f"❌ Code execution blocked: {safety_err}"

    def _run() -> str:
        prelude = "import os, sys, json, subprocess, re\nsys.path.insert(0, %r)\n" % project_root
        full_code = prelude + code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=project_root,
            delete=False, encoding="utf-8",
        ) as file:
            file.write(full_code)
            tmp_path = file.name
        try:
            sensitive_keys = {
                "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "NEO4J_PASSWORD",
                "GEMINI_API_KEY", "OPENAI_API_KEY", "ADMIN_API_KEY",
                "DB_PASSWORD", "SUPABASE_KEY", "TAVILY_API_KEY",
                "AURA_NEO4J_PASSWORD",
            }
            safe_env = {
                key: value for key, value in os.environ.items()
                if key not in sensitive_keys
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


FILESYSTEM_TOOL_HANDLERS = {
    "read_file": _exec_read_file,
    "search_files": _exec_search_files,
    "write_file": _exec_write_file,
    "patch_file": _exec_patch_file,
    "list_directory": _exec_list_directory,
    "execute_python": _exec_execute_python,
}
