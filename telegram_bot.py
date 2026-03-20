"""telegram_bot.py — Telegram bot interface (aiogram 3.x).

Features:
- General messages → Claude Sonnet 4.6 with tool-use (vector_search, knowledge_graph_search, web_search, file system, execute_python)
- /chat <message> → CLAW pipeline (LangGraph agent: intent→retrieve→KG→strategize→generate)
- /task <content> → Save to PostgreSQL queue, background worker processes, push on completion
- /status → Show last 5 tasks
- /clear → Reset chat history

Tools are lazy-loaded from chatbot.py to share the BGE-M3 embedding model and other heavy resources.
Security: ALLOWED_USER_IDS whitelist, unauthorized users silently ignored.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from contextlib import contextmanager
from shared import KST, CORE_IDENTITY
from skills_loader import build_skills_prompt

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import (
    Message, BufferedInputFile,
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
)
from aiogram.filters import Command
import anthropic

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Suppress TelegramConflictError spam during deploy (old/new instance overlap)
class _ConflictFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "TelegramConflictError" not in record.getMessage()

logging.getLogger("aiogram.dispatcher").addFilter(_ConflictFilter())
logging.getLogger("aiogram.event").addFilter(_ConflictFilter())

# Throttle Neo4j DNS/connection retry spam (100s of warnings per second when AuraDB is down)
class _ThrottleFilter(logging.Filter):
    def __init__(self, interval: float = 60.0):
        super().__init__()
        self._last: dict[str, float] = {}
        self._interval = interval

    def filter(self, record: logging.LogRecord) -> bool:
        import time
        # Group by first 80 chars of message to dedup similar warnings
        key = record.getMessage()[:80]
        now = time.monotonic()
        last = self._last.get(key, 0.0)
        if now - last < self._interval:
            return False
        self._last[key] = now
        return True

logging.getLogger("neo4j").addFilter(_ThrottleFilter(60.0))

# ── Config ───────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ALLOWED_USER_IDS: set[int] = {
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USER_IDS", "").split(",")
    if uid.strip()
}

# ── DB (own pool, independent from api.py) ───────────────────────────
_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=3,
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="require",
        )
    return _pool


@contextmanager
def _get_conn():
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def _query(sql: str, params: tuple | None = None) -> list[dict]:
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def _execute(sql: str, params: tuple | None = None) -> None:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def _query_one(sql: str, params: tuple | None = None) -> dict | None:
    """Execute SQL with RETURNING and fetch one row."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None


def _ensure_table():
    """Create telegram_tasks and telegram_chat_history tables if not exists."""
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_tasks (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            content     TEXT NOT NULL,
            status      VARCHAR(20) DEFAULT 'pending',
            result      TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_chat_history (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            role        VARCHAR(10) NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    # Index for fast user_id lookups
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_user_id
        ON telegram_chat_history (user_id, id DESC)
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_schedules (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            content     TEXT NOT NULL,
            cron_expr   VARCHAR(100) NOT NULL,
            enabled     BOOLEAN DEFAULT TRUE,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            last_run_at TIMESTAMPTZ
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_error_log (
            id          SERIAL PRIMARY KEY,
            level       VARCHAR(10) NOT NULL DEFAULT 'error',
            source      VARCHAR(100) NOT NULL,
            message     TEXT NOT NULL,
            detail      TEXT,
            task_id     INTEGER,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_error_log_created
        ON telegram_error_log (created_at DESC)
    """)


# ── Error/Warning Logger ────────────────────────────────────────────
def _log_event(
    level: str,        # "error" | "warning"
    source: str,       # e.g. "chat", "task", "tool", "final_response"
    message: str,
    detail: str | None = None,
    task_id: int | None = None,
) -> None:
    """Persist an error or warning event to telegram_error_log."""
    try:
        _execute(
            "INSERT INTO telegram_error_log (level, source, message, detail, task_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (level[:10], source[:100], message[:2000], detail[:4000] if detail else None, task_id),
        )
    except Exception as _le:
        logger.warning("_log_event DB write failed: %s", _le)


# ── Claude client ────────────────────────────────────────────────────
_claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
_CLAUDE_MAX_TOKENS = 4096
_CLAUDE_MAX_TOKENS_TASK = 16384  # Tasks need longer output for full reports


def _resolve_model(alias: str, fallback: str) -> str:
    """Resolve a Claude model alias to its actual ID via the Models API."""
    try:
        _sync = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        model = _sync.models.retrieve(model_id=alias)
        logger.info("Resolved model %s => %s", alias, model.id)
        return model.id
    except Exception as e:
        logger.warning("Model resolve failed for %s, using fallback %s: %s", alias, fallback, e)
        return fallback


_CLAUDE_MODEL = _resolve_model("claude-sonnet-4-6", "claude-sonnet-4-6")
_CLAUDE_MODEL_STRONG = _resolve_model("claude-sonnet-4-6", "claude-sonnet-4-6")
_CLAUDE_MODEL_LIGHT = _resolve_model("claude-haiku-4-5", "claude-haiku-4-5-20251001")

# ── Local LLM (Ollama) — cheap alternative for lightweight tasks ─────
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
_ollama_available: bool | None = None  # None = not checked yet


async def _ollama_generate(prompt: str, max_tokens: int = 512) -> str | None:
    """Call local Ollama model. Returns response text or None on failure.

    Falls back gracefully — if Ollama is down, callers should use Haiku.
    """
    global _ollama_available
    import httpx

    # Skip if previously confirmed unavailable (re-check every 100 calls via None reset)
    if _ollama_available is False:
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": _OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,  # disable thinking for speed on CPU
                    "options": {"num_predict": max_tokens, "temperature": 0.3},
                },
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            if _ollama_available is None:
                _ollama_available = True
                logger.info("Ollama available: %s @ %s", _OLLAMA_MODEL, _OLLAMA_BASE_URL)
            return result
    except Exception as e:
        if _ollama_available is not False:
            logger.info("Ollama not available (%s), falling back to Haiku", e)
            _ollama_available = False
        return None


def _current_datetime_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")


# ── System Alerts (injected into system prompt) ─────────────────────
import time as _time

_MAX_ALERTS = 5
_ALERT_TTL = 24 * 60 * 60  # 24 hours

# Each alert: (monotonic_timestamp, formatted_string)
_system_alerts: list[tuple[float, str]] = []


def _prune_alerts():
    """Remove expired alerts and trim to max count."""
    now = _time.monotonic()
    _system_alerts[:] = [(t, m) for t, m in _system_alerts if now - t < _ALERT_TTL]
    while len(_system_alerts) > _MAX_ALERTS:
        _system_alerts.pop(0)


def _add_system_alert(msg: str):
    """Add a system alert visible to the bot in its system prompt."""
    _system_alerts.append((_time.monotonic(), f"[{datetime.now(KST).strftime('%H:%M')}] {msg}"))
    _prune_alerts()


def _clear_system_alert(keyword: str):
    """Remove alerts containing keyword (e.g. when issue resolves)."""
    _system_alerts[:] = [(t, m) for t, m in _system_alerts if keyword not in m]


def _format_system_alerts() -> str:
    _prune_alerts()
    if not _system_alerts:
        return ""
    return "\n\n## System Alerts\n" + "\n".join(f"- {m}" for _, m in _system_alerts)


_SYSTEM_PROMPT_TEMPLATE = CORE_IDENTITY + """
Operating via Telegram. Use tools proactively when data would improve the answer — don't rely on memory alone.

## Tool Strategy
- Geopolitics → knowledge_graph_search first, then vector_search
- Theory/ideology → vector_search (layer="core_theory")
- Current events → web_search, cross-ref with KG
- URL in message → fetch_url to read the page, then analyze with context from other tools
- Self-reflection → read_diary; cross-interface memory → read_chat_logs
- Past lessons/mistakes → recall_experience (semantic search over accumulated daily insights)
- Store important facts → write_kg; deep research → create_task
- Your own source code → read_file (e.g. read_file("telegram_bot.py"), read_file("shared.py"))
- Server file management → list_directory, read_file, write_file
- Data processing / automation → execute_python

## Workload Management
- 복잡한 리서치(여러 소스 비교, 장문 분석, 대량 데이터 처리)는 **처음부터 create_task**를 사용해라. 대화에서 도구를 10회 넘게 호출해야 할 것 같으면 즉시 태스크로 전환.
- 도구 한도에 도달하면 시스템이 자동으로 백그라운드 태스크를 생성할 수 있다. 하지만 사전에 판단해서 선제적으로 create_task를 쓰는 것이 더 좋다.
- 사용자에게 "계속할까요?"라고 묻지 말고, 스스로 판단해서 작업을 이어가라.

## Response Rules
- Dialectical materialist lens for geopolitics. Concise, substantive. Cite sources. Match user's language.

**Current time: {current_datetime}**
{system_alerts}
{skills_section}
"""

# ── Tool Definitions (Anthropic API format) ──────────────────────────
_TOOLS = [
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


# ── Tool Execution (lazy-loaded from chatbot.py) ────────────────────
async def _exec_vector_search(query: str, num_results: int = 5, layer: str | None = None) -> str:
    """Execute vector similarity search via chatbot module."""
    try:
        from chatbot import _direct_similarity_search
        docs = await asyncio.to_thread(_direct_similarity_search, query, num_results, layer)
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
        from chatbot import _search_kg
        result = await asyncio.to_thread(_search_kg, query, num_results)
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
    import glob as _glob
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
                generate_line_patch,
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


_TOOL_HANDLERS = {
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

_TOOLS.extend(SELF_TOOLS)
_TOOL_HANDLERS.update(SELF_TOOL_HANDLERS)

_TASK_SYSTEM_PROMPT_TEMPLATE = CORE_IDENTITY + """
You are executing a background intelligence task. Produce a structured Markdown report.

## Rules
- ALWAYS use tools (vector_search, knowledge_graph_search, web_search). Never write from memory alone.
- Use multiple tools and queries for comprehensive coverage.
- Write in the SAME LANGUAGE as the task.
- Format: # Title → ## Executive Summary → ## Analysis (subsections) → ## Key Entities → ## Sources → ## Outlook
- Cite all sources. Distinguish confirmed facts from inference.

**Current time: {current_datetime}**
{system_alerts}
"""

# Per-user chat history (in-memory, lost on restart)
MAX_HISTORY_TURNS = 10  # 10 pairs = 20 messages
_HISTORY_TOKEN_LIMIT = 40_000  # compress if history exceeds this
_RECENT_TURNS_KEEP = 4  # keep last N turns uncompressed (4 turns = 8 msgs)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate for multilingual text (~3 chars/token for Korean+English mix)."""
    return len(text) // 3


def _load_chat_history(user_id: int) -> list[dict]:
    """Load recent chat history from DB for a user."""
    limit = MAX_HISTORY_TURNS * 2
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content FROM ("
                "  SELECT role, content, id FROM telegram_chat_history"
                "  WHERE user_id = %s ORDER BY id DESC LIMIT %s"
                ") sub ORDER BY id ASC",
                (user_id, limit),
            )
            return [{"role": r["role"], "content": r["content"]} for r in cur.fetchall()]


def _save_chat_message(user_id: int, role: str, content: str):
    """Append a single message to DB chat history."""
    _execute(
        "INSERT INTO telegram_chat_history (user_id, role, content) VALUES (%s, %s, %s)",
        (user_id, role, content),
    )


def _clear_chat_history(user_id: int):
    """Delete all chat history for a user."""
    _execute("DELETE FROM telegram_chat_history WHERE user_id = %s", (user_id,))


async def _compress_history(messages: list[dict]) -> list[dict]:
    """Compress chat history if it exceeds the token limit.

    Summarizes older messages into a single context message using Haiku,
    keeping the most recent turns intact for conversational continuity.
    """
    total_tokens = sum(_estimate_tokens(m["content"]) for m in messages)
    if total_tokens <= _HISTORY_TOKEN_LIMIT:
        return messages

    # Split into old (to summarize) and recent (to keep)
    keep_count = _RECENT_TURNS_KEEP * 2  # user+assistant pairs
    if len(messages) <= keep_count:
        return messages  # not enough messages to split

    old_msgs = messages[:-keep_count]
    recent_msgs = messages[-keep_count:]

    logger.info(
        "Compressing history: %d msgs (%d tokens) → summarize %d old, keep %d recent",
        len(messages), total_tokens, len(old_msgs), len(recent_msgs),
    )

    # Build summary request
    conversation_text = "\n".join(
        f"[{m['role']}] {m['content'][:1000]}" for m in old_msgs
    )
    summary_prompt = (
        "아래 대화를 핵심 정보만 남기고 간결하게 요약해라. "
        "사용자가 어떤 주제를 물었고, 어떤 결론/답변이 나왔는지 위주로. "
        "고유명사, 수치, 날짜는 보존. 300자 이내.\n\n"
        f"{conversation_text}"
    )

    try:
        # Try local LLM first (free), fall back to Haiku (paid)
        summary = await _ollama_generate(summary_prompt, max_tokens=512)
        if not summary:
            resp = await _claude.messages.create(
                model=_CLAUDE_MODEL_LIGHT,
                max_tokens=512,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            summary = resp.content[0].text
    except Exception as e:
        logger.warning("History compression failed: %s — using truncation fallback", e)
        # Fallback: just drop old messages
        return recent_msgs

    # Inject summary as a system-like context message
    compressed = [
        {"role": "user", "content": f"[이전 대화 요약]\n{summary}"},
        {"role": "assistant", "content": "네, 이전 대화 내용을 파악했습니다. 이어서 진행하겠습니다."},
    ] + recent_msgs

    new_tokens = sum(_estimate_tokens(m["content"]) for m in compressed)
    logger.info("History compressed: %d tokens → %d tokens", total_tokens, new_tokens)
    return compressed

# ── CLAW pipeline (lazy-loaded) ──────────────────────────────────────
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from chatbot import graph
        _graph = graph
    return _graph


# ── Helpers ──────────────────────────────────────────────────────────
def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split text into chunks respecting Telegram's 4096 char limit."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos <= 0:
            split_pos = text.rfind(" ", 0, max_len)
        if split_pos <= 0:
            split_pos = max_len
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


def _is_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_USER_IDS


# ── Router & Handlers ───────────────────────────────────────────────
_pending_approvals: dict = {}  # 자가수정 승인 대기 (approval_id → entry)
router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    await message.answer(
        "레닌봇 텔레그램 인터페이스입니다.\n\n"
        "메시지를 보내면 Claude 기반 대화를 할 수 있습니다.\n"
        "/chat <메시지> — CLAW 파이프라인으로 질의 (RAG+KG+전략)\n"
        "/task <내용> — 백그라운드 태스크 등록\n"
        "/status — 최근 태스크 상태 확인\n"
        "/status_auto — 자율 생성 태스크 확인\n"
        "/report <id> — 태스크 리포트 파일 재전송 (DB 원문)\n"
        "/kg — 지식그래프 현황 직접 조회\n"
        "/schedule <cron> | <내용> — 정기 태스크 등록\n"
        "/schedules — 등록된 스케줄 목록\n"
        "/unschedule <id> — 스케줄 삭제\n"
        "/errors — 최근 에러/경고 로그 조회\n"
        "/clear — 대화 히스토리 초기화"
    )


@router.message(Command("clear"))
async def cmd_clear(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    await asyncio.to_thread(_clear_chat_history, message.from_user.id)
    await message.answer("대화 히스토리가 초기화되었습니다.")


@router.message(Command("errors"))
async def cmd_errors(message: Message):
    """Show recent error/warning log entries."""
    if not _is_allowed(message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/errors").strip()
    # Parse optional limit and level filter
    # Usage: /errors [n] [error|warning|all]
    limit = 20
    level_filter = None
    for token in arg.split():
        if token.isdigit():
            limit = min(int(token), 50)
        elif token.lower() in ("error", "warning", "warn"):
            level_filter = "error" if token.lower() == "error" else "warning"
    try:
        if level_filter:
            rows = await asyncio.to_thread(
                _query,
                "SELECT id, level, source, message, detail, task_id, created_at "
                "FROM telegram_error_log WHERE level = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (level_filter, limit),
            )
        else:
            rows = await asyncio.to_thread(
                _query,
                "SELECT id, level, source, message, detail, task_id, created_at "
                "FROM telegram_error_log ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
    except Exception as e:
        await message.answer(f"에러 로그 조회 실패: {e}")
        return
    if not rows:
        await message.answer("✅ 기록된 에러/경고 없음.")
        return
    level_icons = {"error": "🔴", "warning": "🟡"}
    lines = [f"🗒️ *에러/경고 로그* (최근 {len(rows)}건)\n"]
    for r in rows:
        icon = level_icons.get(r["level"], "❓")
        ts = r["created_at"].strftime("%m/%d %H:%M:%S")
        task_info = f" [태스크#{r['task_id']}]" if r["task_id"] else ""
        lines.append(
            f"{icon} `{ts}` [{r['source']}]{task_info}\n"
            f"   {r['message'][:120]}"
        )
    for chunk in _split_message("\n\n".join(lines)):
        await message.answer(chunk, parse_mode="Markdown")


@router.message(Command("chat"))
async def cmd_chat(message: Message):
    """Route message through the CLAW pipeline (LangGraph agent)."""
    if not _is_allowed(message.from_user.id):
        return
    content = (message.text or "").removeprefix("/chat").strip()
    if not content:
        await message.answer("사용법: /chat <메시지>")
        return

    user_id = message.from_user.id
    await message.answer("CLAW 파이프라인 처리 중...")

    try:
        from langchain_core.messages import HumanMessage

        g = _get_graph()
        thread_id = f"tg_{user_id}"
        inputs = {"messages": [HumanMessage(content=content)]}
        config = {"configurable": {"thread_id": thread_id}}

        answer = None
        logs: list[str] = []
        async for output in g.astream(inputs, config=config, stream_mode="updates"):
            for node_name, node_content in output.items():
                if node_name == "log_conversation":
                    continue
                if "logs" in node_content:
                    logs.extend(node_content["logs"])
                if node_name == "generate":
                    last_msg = node_content["messages"][-1]
                    answer = last_msg.content

        if answer:
            for chunk in _split_message(answer):
                await message.answer(chunk)
        else:
            await message.answer("파이프라인에서 답변을 생성하지 못했습니다.")

        if logs:
            log_summary = "\n".join(logs[-10:])  # last 10 log lines
            for chunk in _split_message(f"[처리 로그]\n{log_summary}"):
                await message.answer(chunk)

    except Exception as e:
        logger.error("CLAW pipeline error: %s", e)
        await message.answer(f"CLAW 파이프라인 오류: {e}")


@router.message(Command("task"))
async def cmd_task(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    content = (message.text or "").removeprefix("/task").strip()
    if not content:
        await message.answer("사용법: /task <내용>")
        return
    try:
        await asyncio.to_thread(
            _execute,
            "INSERT INTO telegram_tasks (user_id, content) VALUES (%s, %s)",
            (message.from_user.id, content),
        )
        await message.answer(f"태스크가 큐에 추가되었습니다:\n{content}")
    except Exception as e:
        logger.error("Task insert error: %s", e)
        await message.answer(f"태스크 등록 실패: {e}")


@router.message(Command("status"))
async def cmd_status(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    uid = message.from_user.id

    # Gather all dashboard data in parallel
    tasks_f = asyncio.to_thread(
        _query,
        "SELECT id, content, status, created_at FROM telegram_tasks "
        "WHERE user_id = %s ORDER BY created_at DESC LIMIT 5",
        (uid,),
    )
    errors_f = asyncio.to_thread(
        _query,
        "SELECT level, count(*) AS cnt FROM telegram_error_log "
        "WHERE created_at > NOW() - INTERVAL '24 hours' "
        "GROUP BY level ORDER BY level",
        None,
    )
    task_stats_f = asyncio.to_thread(
        _query,
        "SELECT status, count(*) AS cnt FROM telegram_tasks "
        "GROUP BY status",
        None,
    )

    try:
        tasks, errors, task_stats = await asyncio.gather(tasks_f, errors_f, task_stats_f)
    except Exception as e:
        logger.error("Status dashboard query error: %s", e)
        await message.answer(f"대시보드 조회 실패: {e}")
        return

    # -- Build dashboard --
    lines = ["*시스템 대시보드*\n"]

    # 1. Task summary
    stat_map = {r["status"]: r["cnt"] for r in task_stats}
    total_tasks = sum(stat_map.values())
    lines.append(
        f"*태스크* ({total_tasks}건): "
        f"✅{stat_map.get('done', 0)} "
        f"⏳{stat_map.get('pending', 0)} "
        f"🔄{stat_map.get('processing', 0)} "
        f"❌{stat_map.get('failed', 0)}"
    )

    # 2. Error counts (24h)
    err_map = {r["level"]: r["cnt"] for r in errors}
    err_total = sum(err_map.values())
    if err_total:
        lines.append(
            f"*에러 (24h)*: 🔴error {err_map.get('error', 0)} "
            f"🟡warning {err_map.get('warning', 0)}"
        )
    else:
        lines.append("*에러 (24h)*: 없음")

    # 3. KG stats (quick, non-blocking)
    try:
        from shared import fetch_kg_stats
        kg = await asyncio.to_thread(fetch_kg_stats)
        if "error" not in kg:
            entity_total = sum(v for v in kg.get("entity_types", {}).values())
            lines.append(
                f"*KG*: 엔티티 {entity_total} | "
                f"관계 {kg.get('edge_count', 0)} | "
                f"에피소드 {kg.get('episode_count', 0)}"
            )
        else:
            lines.append(f"*KG*: ⚠️ {kg['error'][:60]}")
    except Exception as e:
        lines.append(f"*KG*: ⚠️ 조회 실패")

    # 4. Recent tasks
    if tasks:
        lines.append("\n*최근 태스크:*")
        status_icons = {"pending": "⏳", "processing": "🔄", "done": "✅", "failed": "❌"}
        for r in tasks:
            icon = status_icons.get(r["status"], "❓")
            ts = r["created_at"].strftime("%m/%d %H:%M")
            preview = r["content"][:45]
            lines.append(f"{icon} `[{r['id']}]` {preview}\n   {r['status']} | {ts}")
    else:
        lines.append("\n태스크 없음")

    await message.answer("\n".join(lines), parse_mode="Markdown")


@router.message(Command("kg"))
async def cmd_kg(message: Message):
    """Directly show KG stats — no LLM involved."""
    if not _is_allowed(message.from_user.id):
        return
    from shared import fetch_kg_stats
    await message.answer("KG 조회 중...")
    try:
        stats = await asyncio.to_thread(fetch_kg_stats)
    except Exception as e:
        await message.answer(f"KG 조회 실패: {e}")
        return
    if "error" in stats:
        await message.answer(f"⚠️ KG 오류: {stats['error']}")
        return

    lines = ["📊 *지식그래프 현황* (Neo4j AuraDB)\n"]
    lines.append(f"엔티티: {sum(v for v in stats.get('entity_types', {}).values())}개")
    for label, cnt in stats.get("entity_types", {}).items():
        lines.append(f"  {label}: {cnt}")
    lines.append(f"관계(엣지): {stats.get('edge_count', 0)}개")
    lines.append(f"에피소드: {stats.get('episode_count', 0)}건")
    episodes = stats.get("recent_episodes", [])
    if episodes:
        lines.append("\n*최근 에피소드:*")
        for ep in episodes:
            lines.append(f"  • {ep.get('name', '?')} [{ep.get('group_id', '')}]")
    await message.answer("\n".join(lines))


@router.message(Command("report"))
async def cmd_report(message: Message):
    """Directly fetch a task report from DB and send as file — no LLM involved."""
    if not _is_allowed(message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/report").strip()
    if not arg:
        await message.answer("사용법: /report <task_id>")
        return
    try:
        task_id = int(arg)
    except ValueError:
        await message.answer("task_id는 숫자여야 합니다.")
        return
    try:
        row = await asyncio.to_thread(
            _query_one,
            "SELECT id, content, status, result FROM telegram_tasks WHERE id = %s",
            (task_id,),
        )
    except Exception as e:
        await message.answer(f"조회 실패: {e}")
        return
    if not row:
        await message.answer(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")
        return
    if row["status"] != "done" or not row.get("result"):
        await message.answer(f"태스크 #{task_id} 상태: {row['status']} — 완료된 리포트가 없습니다.")
        return
    report = row["result"]
    doc = BufferedInputFile(report.encode("utf-8"), filename=f"report_task_{task_id}.md")
    await message.answer_document(doc, caption=f"태스크 #{task_id} 리포트 (DB 원문, {len(report)}자)")


@router.message(Command("status_auto"))
async def cmd_status_auto(message: Message):
    """Show recent self-generated (autonomous) tasks."""
    if not _is_allowed(message.from_user.id):
        return
    try:
        rows = await asyncio.to_thread(
            _query,
            "SELECT id, content, status, created_at FROM telegram_tasks "
            "WHERE user_id = 0 ORDER BY created_at DESC LIMIT 10",
        )
    except Exception as e:
        logger.error("Auto-task status query error: %s", e)
        await message.answer(f"조회 실패: {e}")
        return
    if not rows:
        await message.answer("자율 생성된 태스크가 없습니다.")
        return
    status_icons = {"pending": "⏳", "processing": "🔄", "done": "✅", "failed": "❌"}
    lines = ["🤖 *자율 생성 태스크* (최근 10건)\n"]
    for r in rows:
        icon = status_icons.get(r["status"], "❓")
        ts = r["created_at"].strftime("%m/%d %H:%M")
        preview = r["content"][:60]
        lines.append(f"{icon} [{r['id']}] {preview}\n   상태: {r['status']} | {ts}")
    await message.answer("\n\n".join(lines))


@router.message(Command("schedule"))
async def cmd_schedule(message: Message):
    """Add a cron schedule: /schedule <cron_expr> | <task content>"""
    if not _is_allowed(message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/schedule").strip()
    if not arg or "|" not in arg:
        await message.answer(
            "사용법: /schedule <cron식> | <태스크 내용>\n\n"
            "예시:\n"
            "  /schedule 0 9 * * * | 오늘의 국제 뉴스 브리핑\n"
            "  /schedule 0 8 * * 1 | 주간 지정학 정세 분석\n"
            "  /schedule 0 */6 * * * | 6시간마다 KG 상태 점검\n\n"
            "cron 형식: 분 시 일 월 요일 (KST 기준)"
        )
        return
    parts = arg.split("|", 1)
    cron_expr = parts[0].strip()
    content = parts[1].strip()
    if not content:
        await message.answer("태스크 내용이 비어있습니다.")
        return
    # Validate cron expression
    try:
        from croniter import croniter
        croniter(cron_expr)
    except (ValueError, KeyError) as e:
        await message.answer(f"잘못된 cron 표현식: {cron_expr}\n오류: {e}")
        return
    try:
        # Set last_run_at = NOW() so the first fire waits for the next cron window
        await asyncio.to_thread(
            _execute,
            "INSERT INTO telegram_schedules (user_id, content, cron_expr, last_run_at) "
            "VALUES (%s, %s, %s, NOW())",
            (message.from_user.id, content, cron_expr),
        )
        await message.answer(
            f"✅ 스케줄 등록 완료\n"
            f"  cron: `{cron_expr}` (KST)\n"
            f"  내용: {content[:100]}"
        )
    except Exception as e:
        await message.answer(f"스케줄 등록 실패: {e}")


@router.message(Command("schedules"))
async def cmd_schedules(message: Message):
    """List all schedules for the user."""
    if not _is_allowed(message.from_user.id):
        return
    try:
        rows = await asyncio.to_thread(
            _query,
            "SELECT id, content, cron_expr, enabled, last_run_at "
            "FROM telegram_schedules WHERE user_id = %s ORDER BY id",
            (message.from_user.id,),
        )
    except Exception as e:
        await message.answer(f"조회 실패: {e}")
        return
    if not rows:
        await message.answer("등록된 스케줄이 없습니다.")
        return
    lines = ["📅 *등록된 스케줄*\n"]
    for r in rows:
        status = "✅" if r["enabled"] else "⏸️"
        last = r["last_run_at"].strftime("%m/%d %H:%M") if r["last_run_at"] else "미실행"
        preview = r["content"][:60]
        lines.append(
            f"{status} [{r['id']}] `{r['cron_expr']}`\n"
            f"   {preview}\n"
            f"   마지막 실행: {last}"
        )
    await message.answer("\n\n".join(lines))


@router.message(Command("unschedule"))
async def cmd_unschedule(message: Message):
    """Delete a schedule: /unschedule <id>"""
    if not _is_allowed(message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/unschedule").strip()
    if not arg:
        await message.answer("사용법: /unschedule <schedule_id>")
        return
    try:
        sched_id = int(arg)
    except ValueError:
        await message.answer("schedule_id는 숫자여야 합니다.")
        return
    try:
        row = await asyncio.to_thread(
            _query_one,
            "DELETE FROM telegram_schedules WHERE id = %s AND user_id = %s RETURNING id",
            (sched_id, message.from_user.id),
        )
    except Exception as e:
        await message.answer(f"삭제 실패: {e}")
        return
    if row:
        await message.answer(f"🗑️ 스케줄 [{sched_id}] 삭제 완료")
    else:
        await message.answer(f"스케줄 [{sched_id}]을(를) 찾을 수 없습니다.")


@router.message(Command("deploy"))
async def cmd_deploy(message: Message):
    """Run deploy.sh — git pull + restart services. Output sent back via Telegram."""
    if not _is_allowed(message.from_user.id):
        return
    deploy_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy.sh")
    if not os.path.isfile(deploy_script):
        await message.answer("deploy.sh를 찾을 수 없습니다.")
        return

    status_msg = await message.answer("🚀 Deploy 시작...")
    try:
        # Run deploy.sh detached (setsid) so it survives bot restart
        log_path = "/tmp/leninbot-deploy.log"
        # Clear old log
        open(log_path, "w").close()
        proc = await asyncio.create_subprocess_exec(
            "setsid", "bash", deploy_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        # Read output until process exits or bot gets killed by restart
        output_lines: list[str] = []
        try:
            async for line in proc.stdout:
                output_lines.append(line.decode(errors="replace").rstrip())
            await proc.wait()
        except (asyncio.CancelledError, ConnectionError, OSError):
            return  # bot is being restarted by deploy.sh — expected, curl handles notification

        result = "\n".join(output_lines[-30:])  # last 30 lines
        if proc.returncode == 0:
            await status_msg.edit_text(f"✅ Deploy 완료\n```\n{result}\n```", parse_mode="Markdown")
        else:
            await status_msg.edit_text(f"❌ Deploy 실패 (exit {proc.returncode})\n```\n{result}\n```", parse_mode="Markdown")
    except Exception as e:
        # ServerDisconnectedError / CancelledError = bot killed by deploy restart — expected
        err_name = type(e).__name__
        err_str = str(e)
        if ("Disconnect" in err_name or "Disconnect" in err_str
                or isinstance(e, (asyncio.CancelledError, ConnectionError, OSError))):
            return  # deploy.sh curl handles notification
        try:
            await status_msg.edit_text(f"❌ Deploy 오류: {e}")
        except Exception:
            pass


@router.message(F.text)
async def handle_message(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    user_id = message.from_user.id
    user_text = message.text

    # Save user message to DB, load history, compress if needed
    await asyncio.to_thread(_save_chat_message, user_id, "user", user_text)
    history = await asyncio.to_thread(_load_chat_history, user_id)
    history = await _compress_history(history)

    # Auto-recall: fetch relevant past experiences for context injection
    experience_context = await _fetch_relevant_experiences(user_text)

    try:
        system_override = None
        if experience_context:
            system_override = _SYSTEM_PROMPT_TEMPLATE.format(
                current_datetime=_current_datetime_str(),
                system_alerts=_format_system_alerts(),
                skills_section=build_skills_prompt(),
            ) + experience_context
        reply = await _chat_with_tools(history, system_prompt=system_override)
    except Exception as e:
        logger.error("Claude API error: %s", e)
        _log_event("error", "chat", f"Claude API error: {e}", detail=user_text[:500])
        reply = f"오류가 발생했습니다: {e}"

    # Auto-escalation: extract [CONTINUE_TASK: ...] marker and create background task
    continuation_task = None
    if "[CONTINUE_TASK:" in reply:
        import re
        match = re.search(r"\[CONTINUE_TASK:\s*(.+?)\]", reply, re.DOTALL)
        if match:
            continuation_task = match.group(1).strip()
            # Remove the marker from the reply shown to user
            reply = reply[:match.start()].rstrip()

    # Save assistant reply to DB
    await asyncio.to_thread(_save_chat_message, user_id, "assistant", reply)

    for chunk in _split_message(reply):
        await message.answer(chunk)

    # Create background task for unfinished work
    if continuation_task:
        task_content = f"[자동 승격] 대화 중 미완료 작업 이어서 수행:\n{continuation_task}\n\n원래 질문: {user_text[:500]}"
        await asyncio.to_thread(
            _execute,
            "INSERT INTO telegram_tasks (user_id, content, status) VALUES (%s, %s, 'pending')",
            (user_id, task_content),
        )
        task_row = await asyncio.to_thread(
            _query_one, "SELECT id FROM telegram_tasks WHERE user_id = %s ORDER BY id DESC LIMIT 1", (user_id,),
        )
        task_id = task_row["id"] if task_row else "?"
        await message.answer(f"🔄 미완료 작업을 백그라운드 태스크 `[{task_id}]`로 자동 생성했습니다. 완료되면 알려드리겠습니다.")

    # Auto-reflection: every 5 exchanges, reflect on recent conversations
    _reflection_counter[user_id] = _reflection_counter.get(user_id, 0) + 1
    if _reflection_counter[user_id] >= 5:
        _reflection_counter[user_id] = 0
        asyncio.create_task(_reflect_on_recent(user_id))


# ── Auto-Recall & Reflection (experiential learning) ─────────────────
_reflection_counter: dict[int, int] = {}


async def _fetch_relevant_experiences(user_text: str) -> str:
    """Search experiential_memory for insights relevant to the user's message.

    Returns a formatted context string to inject into the system prompt,
    or empty string if nothing relevant found.
    """
    try:
        from shared import search_experiential_memory
        results = await asyncio.to_thread(search_experiential_memory, user_text, 3)
        if not results:
            return ""
        lines = ["\n## Past Experiences (auto-recalled)"]
        for r in results:
            cat = r.get("category", "?")
            sim = r.get("similarity", 0)
            lines.append(f"- [{cat}] {r['content']}")
        lines.append("위 경험을 참고하되, 현재 대화 맥락에 맞게 판단해라.")
        return "\n".join(lines)
    except Exception as e:
        logger.debug("Experience recall failed (non-critical): %s", e)
        return ""

_REFLECTION_PROMPT = """\
아래 대화에서 배울 점을 추출해라. 다음 카테고리별로 1개씩만 (해당 없으면 생략):

- **lesson**: 새로 배운 사실이나 지식
- **mistake**: 잘못된 답변, 도구 오용, 사용자 수정이 있었던 부분
- **pattern**: 반복적인 사용자 요구나 질문 패턴
- **insight**: 분석/논의에서 도출된 깊은 통찰
- **observation**: 기술적 발견이나 시스템 동작에 대한 관찰

각 항목을 한 줄로, 앞에 카테고리를 붙여 작성. 예:
lesson: 시리아 내전에서 러시아의 군사 개입은 2015년부터이며...
mistake: 사용자가 물어본 것은 경제 제재인데 군사적 측면만 답변했음
pattern: 사용자는 자주 한국 정치와 국제 정세의 연관성을 묻는다

배울 게 없으면 "NONE"이라고만 답해.

대화:
"""


async def _reflect_on_recent(user_id: int):
    """Background task: reflect on recent conversations and save insights."""
    try:
        history = await asyncio.to_thread(_load_chat_history, user_id)
        if len(history) < 4:
            return  # too little to reflect on

        # Build conversation text for reflection
        conv_text = "\n".join(
            f"[{m['role']}] {m['content'][:500]}" for m in history
        )
        prompt = _REFLECTION_PROMPT + conv_text

        # Try local LLM first (free), fall back to Haiku (paid)
        result = await _ollama_generate(prompt, max_tokens=512)
        if not result:
            resp = await _claude.messages.create(
                model=_CLAUDE_MODEL_LIGHT,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            result = resp.content[0].text.strip()

        if result.upper() == "NONE":
            logger.info("Reflection: nothing to learn from recent conversation")
            return

        # Parse and save each insight
        from shared import save_experiential_memory
        valid_categories = {"lesson", "mistake", "pattern", "insight", "observation"}
        saved = 0
        for line in result.split("\n"):
            line = line.strip().lstrip("- ")
            if ":" not in line:
                continue
            cat, content = line.split(":", 1)
            cat = cat.strip().lower()
            content = content.strip()
            if cat in valid_categories and len(content) > 10:
                success = await asyncio.to_thread(
                    save_experiential_memory, content, cat, "auto_reflection"
                )
                if success:
                    saved += 1

        if saved:
            logger.info("Reflection: saved %d experience(s) from user %d conversation", saved, user_id)
    except Exception as e:
        logger.warning("Reflection failed: %s", e)


def _ensure_tool_results(msgs: list[dict]) -> list[dict]:
    """Ensure every tool_use/server_tool_use in assistant messages has a matching result.

    - Custom tool_use → tool_result in the NEXT user message
    - server_tool_use → web_search_tool_result in the SAME assistant message

    Missing results are injected as dummies to prevent 400 errors from the API.
    Operates on a copy to avoid mutating the original list.
    """
    msgs = [dict(m) for m in msgs]  # shallow copy each message
    i = 0
    injected = 0
    while i < len(msgs):
        msg = msgs[i]
        if msg.get("role") != "assistant":
            i += 1
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            i += 1
            continue

        # Collect all tool IDs that need results
        custom_ids = [
            b["id"] for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        # For server_tool_use, exclude IDs already resolved by web_search_tool_result
        # within the same assistant content (API returns both in one response)
        resolved_in_assistant = {
            b.get("tool_use_id") for b in content
            if isinstance(b, dict) and b.get("type") == "web_search_tool_result"
        }
        server_ids = [
            b["id"] for b in content
            if isinstance(b, dict) and b.get("type") == "server_tool_use"
            and b["id"] not in resolved_in_assistant
        ]
        if not custom_ids and not server_ids:
            i += 1
            continue

        # Check the next user message for existing custom tool results
        resolved_custom: set = set()
        next_content: list = []
        if i + 1 < len(msgs) and msgs[i + 1].get("role") == "user":
            nc = msgs[i + 1].get("content", [])
            if isinstance(nc, list):
                next_content = nc
                resolved_custom = {
                    b.get("tool_use_id") for b in nc
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                }

        # --- Server tool dummies go INTO the assistant message (same block) ---
        # server_ids already excludes those resolved within the assistant content
        server_dummies = []
        for tid in server_ids:
            server_dummies.append({
                "type": "web_search_tool_result",
                "tool_use_id": tid,
                "content": [],
            })
        if server_dummies:
            # Deep-copy content so we don't mutate the original list
            new_content = list(content) + server_dummies
            msgs[i] = {**msgs[i], "content": new_content}
            injected += len(server_dummies)

        # --- Custom tool dummies go in the NEXT user message ---
        dummies = []
        for tid in custom_ids:
            if tid not in resolved_custom:
                dummies.append({
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": "[tool result unavailable]",
                })

        if dummies:
            injected += len(dummies)
            if next_content:
                # Prepend dummies to existing user message content
                msgs[i + 1] = {**msgs[i + 1], "content": dummies + next_content}
            elif i + 1 < len(msgs) and msgs[i + 1].get("role") == "user":
                # User message has string content — wrap into list
                old = msgs[i + 1].get("content", "")
                msgs[i + 1] = {
                    "role": "user",
                    "content": dummies + [{"type": "text", "text": str(old)}],
                }
            else:
                # No user message after assistant — insert one
                msgs.insert(i + 1, {"role": "user", "content": dummies})

        i += 1  # move past assistant message
        # If custom dummies were injected (possibly inserting a user message), skip it too
        if dummies or (i < len(msgs) and msgs[i].get("role") == "user"):
            i += 1  # move past user message (existing or inserted)

    if injected:
        logger.warning("_ensure_tool_results: injected %d dummy result(s)", injected)
    return msgs


async def _chat_with_tools(
    messages: list[dict],
    max_rounds: int = 15,
    system_prompt: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Call Claude with tools, execute tool calls, loop until text response."""
    # Work on a copy so tool-use intermediate messages don't pollute persistent history
    working_msgs = list(messages)
    tool_call_log = []  # Track tool calls for diagnostic output
    effective_max_tokens = max_tokens or _CLAUDE_MAX_TOKENS

    # Prompt caching: mark system prompt and tools as cacheable
    sys_prompt = system_prompt or _SYSTEM_PROMPT_TEMPLATE.format(
        current_datetime=_current_datetime_str(),
        system_alerts=_format_system_alerts(),
        skills_section=build_skills_prompt(),
    )
    cached_system = [{"type": "text", "text": sys_prompt, "cache_control": {"type": "ephemeral"}}]

    # Mark last custom tool for caching (skip server-side tools like web_search)
    cached_tools = [dict(t) for t in _TOOLS]
    for i in range(len(cached_tools) - 1, -1, -1):
        if cached_tools[i].get("type", "").startswith("web_search"):
            continue  # server-side tool — can't add cache_control
        cached_tools[i] = {**cached_tools[i], "cache_control": {"type": "ephemeral"}}
        break

    for round_num in range(1, max_rounds + 1):
        # Sanitize message structure before every API call
        working_msgs = _ensure_tool_results(working_msgs)
        response = await _claude.messages.create(
            model=model or _CLAUDE_MODEL,
            max_tokens=effective_max_tokens,
            system=cached_system,
            tools=cached_tools,
            messages=working_msgs,
        )

        # If no custom tool use, extract and return text
        # (server-side tools like web_search are auto-handled, stop_reason is "end_turn")
        if response.stop_reason not in ("tool_use", "pause_turn"):
            if response.stop_reason == "max_tokens":
                logger.warning("Response truncated by max_tokens (%d) at round %d/%d", effective_max_tokens, round_num, max_rounds)
                _log_event("warning", "chat", f"Response truncated by max_tokens ({effective_max_tokens}) at round {round_num}/{max_rounds}")
            text_parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_parts) if text_parts else "응답을 생성하지 못했습니다."

        # Process tool calls (custom tools only; server-side blocks pass through)
        assistant_content = []
        tool_results = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "server_tool_use":
                # Server-side tool (web_search) — pass through as-is
                assistant_content.append({
                    "type": "server_tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_call_log.append(f"  [{round_num}/{max_rounds}] {block.name}(server-side)")
            elif block.type == "web_search_tool_result":
                # Server-side search result — pass through
                assistant_content.append(block.model_dump())
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                # Execute custom tool
                handler = _TOOL_HANDLERS.get(block.name)
                if handler:
                    logger.info("Tool call: %s(%s)", block.name, json.dumps(block.input, ensure_ascii=False)[:200])
                    try:
                        result = await handler(**block.input)
                        is_error = False
                    except Exception as e:
                        logger.error("Tool %s execution error: %s", block.name, e)
                        _log_event("warning", "tool", f"Tool {block.name} failed: {e}")
                        result = f"Tool execution failed: {e}"
                        is_error = True
                else:
                    result = f"Unknown tool: {block.name}"
                    is_error = True
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
                if is_error:
                    tool_result_block["is_error"] = True
                tool_results.append(tool_result_block)
                # Log for diagnostics
                input_summary = json.dumps(block.input, ensure_ascii=False)
                if len(input_summary) > 120:
                    input_summary = input_summary[:120] + "..."
                tool_call_log.append(f"  [{round_num}/{max_rounds}] {block.name}({input_summary})")

        # Safety net: ensure EVERY tool_use block has a matching tool_result
        resolved_ids = {r["tool_use_id"] for r in tool_results}
        for block in assistant_content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block["id"] not in resolved_ids:
                logger.warning("Safety net: missing tool_result for tool_use id=%s name=%s", block["id"], block.get("name"))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": f"Tool execution skipped (internal error): no result was produced for {block.get('name', 'unknown')}",
                    "is_error": True,
                })

        # Append assistant message with tool_use + user message with tool_results
        working_msgs.append({"role": "assistant", "content": assistant_content})

        # Fix: inject dummy web_search_tool_result for any unresolved server_tool_use
        # IMPORTANT: web_search_tool_result must go in the ASSISTANT message (same block),
        # NOT in the user message — the API only recognises server tool results in assistant content.
        already_resolved_server_ids = {
            b.get("tool_use_id") for b in assistant_content
            if isinstance(b, dict) and b.get("type") == "web_search_tool_result"
        }
        pending_server_ids = [
            b["id"] for b in assistant_content
            if isinstance(b, dict) and b.get("type") == "server_tool_use"
            and b["id"] not in already_resolved_server_ids
        ]
        if pending_server_ids:
            # Append dummy results INTO the assistant content (same message)
            for tid in pending_server_ids:
                assistant_content.append({
                    "type": "web_search_tool_result",
                    "tool_use_id": tid,
                    "content": [],
                })
            # Update the already-appended assistant message in working_msgs
            working_msgs[-1] = {"role": "assistant", "content": assistant_content}
            logger.debug("Injected %d dummy web_search_tool_result(s) into assistant content", len(pending_server_ids))

        if tool_results:
            working_msgs.append({"role": "user", "content": tool_results})
        elif response.stop_reason == "pause_turn":
            # Server-side tool paused (>10 iterations) — send empty to continue
            working_msgs.append({"role": "user", "content": [{"type": "text", "text": "continue"}]})

    # Limit reached — check if we should auto-escalate to background task
    log_detail = "\n".join(tool_call_log) if tool_call_log else ""
    was_still_working = response.stop_reason in ("tool_use", "pause_turn")
    logger.warning("Tool round limit (%d) reached (still_working=%s). Forcing final response. Calls:\n%s",
                    max_rounds, was_still_working, log_detail)

    # Inject a nudge so the model knows it must answer now
    escalation_hint = ""
    if was_still_working:
        escalation_hint = (
            " 미완료 작업이 있다면, 응답 맨 끝에 "
            "\"[CONTINUE_TASK: 남은 작업 설명]\" 형식으로 한 줄 추가하세요. "
            "시스템이 자동으로 백그라운드 태스크를 생성합니다."
        )
    working_msgs.append({
        "role": "user",
        "content": (
            "[SYSTEM] 도구 호출 한도에 도달했습니다. 추가 도구를 사용하지 말고, "
            "지금까지 수집한 정보만으로 최선의 답변을 완성하세요."
            + escalation_hint
        ),
    })
    # Sanitize: ensure ALL tool_use/server_tool_use have matching results
    working_msgs = _ensure_tool_results(working_msgs)
    try:
        final = await _claude.messages.create(
            model=model or _CLAUDE_MODEL,
            max_tokens=effective_max_tokens,
            system=cached_system,
            messages=working_msgs,  # no tools parameter — forces text-only response
        )
        if final.stop_reason == "max_tokens":
            logger.warning("Forced final response truncated by max_tokens (%d)", effective_max_tokens)
        text_parts = [b.text for b in final.content if b.type == "text"]
        return "\n".join(text_parts) if text_parts else "응답을 생성하지 못했습니다."
    except Exception as e:
        logger.error("Final forced response failed: %s", e)
        _log_event("error", "final_response", f"Final forced response failed: {e}")
        return f"⚠️ 도구 호출 한도({max_rounds}회) 도달 후 응답 생성 실패: {e}"


# ── Background Task Worker ───────────────────────────────────────────
def _extract_summary(report: str, max_len: int = 300) -> str:
    """Extract Executive Summary section or first paragraph as brief summary."""
    # Try to find Executive Summary section
    for marker in ("## Executive Summary", "## 요약", "## 핵심 요약"):
        idx = report.find(marker)
        if idx != -1:
            after = report[idx + len(marker):].strip()
            # Take until next ## heading
            next_heading = after.find("\n## ")
            section = after[:next_heading].strip() if next_heading != -1 else after
            if section:
                return section[:max_len] + ("..." if len(section) > max_len else "")
    # Fallback: first non-heading paragraph
    for line in report.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("**"):
            return line[:max_len] + ("..." if len(line) > max_len else "")
    return report[:max_len]


def _classify_priority(content: str, report: str) -> str:
    """Classify task result priority from content tags or report urgency keywords."""
    # Check explicit priority tag in content
    if "[🔴 HIGH]" in content:
        return "high"
    if "[🟢 LOW]" in content:
        return "low"
    # Check report for urgency signals
    report_lower = report[:2000].lower()
    if any(k in report_lower for k in ("urgent", "critical", "긴급", "위기", "경고", "즉시")):
        return "high"
    return "normal"


async def _process_task(bot: Bot, task: dict):
    """Process a task: run tools, generate report, save to DB, send as file."""
    task_id = task["id"]
    user_id = task["user_id"]
    content = task["content"]
    is_self_generated = (user_id == 0)

    max_retries = 10
    retry_delay = 60  # seconds

    for attempt in range(max_retries):
        try:
            report = await _chat_with_tools(
                [{"role": "user", "content": content}],
                max_rounds=15,
                system_prompt=_TASK_SYSTEM_PROMPT_TEMPLATE.format(current_datetime=_current_datetime_str(), system_alerts=_format_system_alerts()),
                model=_CLAUDE_MODEL_STRONG,
                max_tokens=_CLAUDE_MAX_TOKENS_TASK,
            )

            # Save full report to DB
            await asyncio.to_thread(
                _execute,
                "UPDATE telegram_tasks SET status = 'done', result = %s, "
                "completed_at = NOW() WHERE id = %s",
                (report, task_id),
            )

            # Classify priority
            priority = _classify_priority(content, report)
            priority_icon = {"high": "🔴", "normal": "🟡", "low": "🟢"}.get(priority, "🟡")

            # Send report as Markdown file
            filename = f"report_task_{task_id}.md"
            doc = BufferedInputFile(report.encode("utf-8"), filename=filename)
            summary = _extract_summary(report)
            origin = " (자율 생성)" if is_self_generated else ""
            caption = f"{priority_icon} 태스크 [{task_id}]{origin} 완료\n\n{summary}"

            if is_self_generated:
                # Self-generated task: broadcast to all users
                for uid in ALLOWED_USER_IDS:
                    try:
                        await bot.send_document(chat_id=uid, document=doc, caption=caption)
                    except Exception:
                        pass
            else:
                await bot.send_document(chat_id=user_id, document=doc, caption=caption)

            return  # success

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = (
                "rate_limit" in err_str or
                "overloaded" in err_str or
                "529" in err_str or
                "429" in err_str or
                "too many requests" in err_str
            )

            if is_rate_limit and attempt < max_retries - 1:
                logger.warning("Task %d rate limited (attempt %d/%d), retrying in %ds...", task_id, attempt + 1, max_retries, retry_delay)
                await asyncio.sleep(retry_delay)
                continue

            # Final failure or non-rate-limit error
            logger.error("Task %d failed: %s", task_id, e)
            await asyncio.to_thread(
                _log_event, "error", "task",
                f"Task {task_id} failed: {e}",
                detail=content[:500], task_id=task_id,
            )
            await asyncio.to_thread(
                _execute,
                "UPDATE telegram_tasks SET status = 'failed', result = %s, "
                "completed_at = NOW() WHERE id = %s",
                (str(e), task_id),
            )
            error_msg = f"❌ 태스크 [{task_id}] 실패:\n{e}"
            if is_self_generated:
                await _broadcast(bot, error_msg)
            else:
                await bot.send_message(chat_id=user_id, text=error_msg)


async def _broadcast(bot: Bot, text: str):
    """Send a message to all allowed users. For system event notifications."""
    for uid in ALLOWED_USER_IDS:
        try:
            await bot.send_message(chat_id=uid, text=text)
        except Exception as e:
            logger.warning("Broadcast to %s failed: %s", uid, e)


async def _system_monitor(bot: Bot):
    """Background loop: monitor system events and broadcast notifications."""
    from shared import get_kg_service

    # 1. Startup notification
    await asyncio.sleep(5)  # let services initialize
    kg = await asyncio.to_thread(get_kg_service)
    kg_status = "connected" if kg else "unavailable"
    _add_system_alert(f"Deploy 완료 — KG: {kg_status}")
    if not kg:
        _add_system_alert("KG (Neo4j AuraDB) 연결 불가 — 그래프 검색/쓰기 사용 불가")
    await _broadcast(bot, (
        f"🟢 *Deploy 완료* — 새 버전이 live입니다.\n"
        f"  KG (Neo4j): {kg_status}"
    ))

    # 2. Periodic KG health check (every 2 minutes)
    kg_was_up = kg is not None
    while True:
        await asyncio.sleep(120)
        try:
            kg = await asyncio.to_thread(get_kg_service)
            kg_is_up = kg is not None

            if kg_was_up and not kg_is_up:
                _clear_system_alert("KG 재연결")
                _add_system_alert("KG (Neo4j AuraDB) 연결 끊김 — 그래프 검색/쓰기 사용 불가")
                await _broadcast(bot, "🔴 *KG 연결 끊김* — Neo4j AuraDB에 연결할 수 없습니다.")
            elif not kg_was_up and kg_is_up:
                _clear_system_alert("KG")  # clear all KG-related alerts
                _add_system_alert("KG 재연결 성공 — Neo4j AuraDB 정상")
                await _broadcast(bot, "🟢 *KG 재연결 성공* — Neo4j AuraDB 연결이 복구되었습니다.")

            kg_was_up = kg_is_up
        except Exception as e:
            logger.error("System monitor error: %s", e)


async def _task_worker(bot: Bot):
    """Poll DB for pending tasks and process them one at a time."""
    logger.info("Task worker started")
    while True:
        try:
            task = await asyncio.to_thread(
                _query_one,
                "UPDATE telegram_tasks SET status = 'processing' "
                "WHERE id = (SELECT id FROM telegram_tasks WHERE status = 'pending' "
                "ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED) "
                "RETURNING id, user_id, content",
            )
            if task:
                await _process_task(bot, task)
            else:
                await asyncio.sleep(5)
        except Exception as e:
            logger.error("Worker loop error: %s", e)
            await asyncio.sleep(10)


async def _schedule_worker(bot: Bot):
    """Check cron schedules every 60s, create tasks when due."""
    from croniter import croniter
    from shared import KST

    logger.info("Schedule worker started")
    await asyncio.sleep(10)  # let other services init first
    while True:
        try:
            schedules = await asyncio.to_thread(
                _query,
                "SELECT id, user_id, content, cron_expr, last_run_at "
                "FROM telegram_schedules WHERE enabled = TRUE",
            )
            now_kst = datetime.now(KST)
            for sched in schedules:
                try:
                    cron = croniter(sched["cron_expr"], now_kst)
                    prev_fire = cron.get_prev(datetime)
                    # Should fire if prev_fire is after last_run_at (or never run)
                    last_run = sched["last_run_at"]
                    if last_run is None or prev_fire > last_run:
                        # Create a task
                        await asyncio.to_thread(
                            _execute,
                            "INSERT INTO telegram_tasks (user_id, content) VALUES (%s, %s)",
                            (sched["user_id"], sched["content"]),
                        )
                        await asyncio.to_thread(
                            _execute,
                            "UPDATE telegram_schedules SET last_run_at = %s WHERE id = %s",
                            (now_kst, sched["id"]),
                        )
                        logger.info("Schedule #%d fired → task created: %.50s", sched["id"], sched["content"])
                        # Notify the user
                        try:
                            await bot.send_message(
                                chat_id=sched["user_id"],
                                text=f"⏰ 스케줄 [{sched['id']}] 실행 → 태스크 생성됨\n{sched['content'][:100]}",
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.error("Schedule #%d check error: %s", sched["id"], e)
        except Exception as e:
            logger.error("Schedule worker error: %s", e)
        await asyncio.sleep(60)


# ── Deploy detection ─────────────────────────────────────────────────
_DEPLOY_META_PATH = "/tmp/leninbot-deploy-meta.json"


async def _check_deploy_meta(bot: Bot):
    """On startup, check if we were just deployed. Inject into system alerts."""
    try:
        if not os.path.isfile(_DEPLOY_META_PATH):
            return
        with open(_DEPLOY_META_PATH, "r") as f:
            meta = json.load(f)
        # Consume the file so we don't re-trigger on manual restart
        os.remove(_DEPLOY_META_PATH)

        status = meta.get("status", "success")

        if status == "failed":
            error = meta.get("error", "unknown")
            exit_code = meta.get("exit_code", "?")
            alert_msg = f"Deploy 실패 (exit {exit_code}): {error}"
            _add_system_alert(alert_msg)
            logger.error("Deploy FAILED: exit=%s error=%s", exit_code, error)
            return

        changes = meta.get("changes", "")
        new_commit = meta.get("new_commit", "")[:7]
        prev_commit = meta.get("prev_commit", "")[:7]
        deps = " (의존성 업데이트됨)" if meta.get("deps_updated") else ""

        alert_msg = (
            f"Deploy 완료: {prev_commit}→{new_commit}{deps}. "
            f"변경: {changes}"
        )
        _add_system_alert(alert_msg)
        logger.info("Deploy detected: %s → %s", prev_commit, new_commit)
    except Exception as e:
        logger.warning("Deploy meta check failed: %s", e)


# ── Entry Point ──────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
#  자가수정 핸들러 — Telegram 전용 (chatbot.py에는 없음)
# ═══════════════════════════════════════════════════════════════

@router.message(Command("modify"))
async def cmd_modify(message: Message):
    """자가수정 명령어 — 허가된 Telegram 사용자만"""
    if not _is_allowed(message.from_user.id):
        return
    import os as _os, time as _time, uuid as _uuid
    content = (message.text or "").removeprefix("/modify").strip()
    parts = content.split("|", 2)
    if len(parts) != 3:
        await message.answer(
            "사용법:\n`/modify <파일경로> | <수정이유> | <새 내용 전체>`",
            parse_mode="Markdown"
        )
        return

    filepath, reason, new_content = [p.strip() for p in parts]

    # 경로 보안: leninbot 디렉토리 밖 거부
    base = "/home/grass/leninbot"
    abs_path = _os.path.realpath(_os.path.join(base, filepath))
    if not (abs_path == base or abs_path.startswith(base + "/")):
        await message.answer("❌ 허용된 디렉토리 밖의 파일은 수정할 수 없어.")
        return
    if not _os.path.isfile(abs_path):
        await message.answer(f"❌ 파일을 찾을 수 없어: `{filepath}`", parse_mode="Markdown")
        return

    # diff 생성
    import difflib as _dl
    try:
        with open(abs_path, "r", encoding="utf-8") as _f:
            old_content = _f.read()
        diff_lines = list(_dl.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            lineterm=""
        ))
        diff_text = "".join(diff_lines)
        insertions = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        deletions  = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
    except Exception as e:
        await message.answer(f"❌ diff 생성 실패: {e}")
        return

    if not diff_lines:
        await message.answer("ℹ️ 변경사항 없음. 현재 파일과 동일해.")
        return

    # 승인 대기 등록 (5분 유효)
    approval_id = str(_uuid.uuid4())[:8]
    _pending_approvals[approval_id] = {
        "filepath": abs_path,
        "new_content": new_content,
        "reason": reason,
        "expire": _time.time() + 300,
    }

    diff_preview = diff_text[:3000] + ("\n…(생략)…" if len(diff_text) > 3000 else "")
    summary = (
        f"📝 *자가수정 요청*\n"
        f"파일: `{filepath}`\n"
        f"이유: {reason}\n"
        f"변경: +{insertions} / -{deletions} 라인\n\n"
        f"```\n{diff_preview}\n```"
    )
    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ 승인", callback_data=f"selfmod_approve:{approval_id}"),
        InlineKeyboardButton(text="❌ 거부", callback_data=f"selfmod_reject:{approval_id}"),
    ]])
    await message.answer(summary, parse_mode="Markdown", reply_markup=kb)


@router.callback_query(F.data.startswith("selfmod_approve:"))
async def cb_modify_approve(callback: CallbackQuery):
    if not _is_allowed(callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    import time as _time
    approval_id = callback.data.split(":", 1)[1]
    entry = _pending_approvals.pop(approval_id, None)
    if entry is None:
        await callback.message.edit_text("⚠️ 승인 정보를 찾을 수 없어. 만료됐거나 이미 처리됨.")
        return
    if _time.time() > entry["expire"]:
        await callback.message.edit_text("⏰ 승인 시간 초과 (5분). 다시 `/modify`를 실행해.")
        return

    await callback.message.edit_text("⚙️ 패치 적용 중…")
    await callback.answer()

    sys.path.insert(0, "/home/grass/leninbot")
    from self_modification_core import self_modify_with_safety
    try:
        result = await asyncio.to_thread(
            self_modify_with_safety,
            filepath=entry["filepath"],
            new_content=entry["new_content"],
            reason=entry["reason"],
            request_approval=False,
            skip_tests=False,
        )
    except Exception as e:
        await callback.message.edit_text(
            f"❌ 패치 적용 중 예외 발생:\n`{e}`", parse_mode="Markdown"
        )
        return

    if result.status == "success":
        commit_info = f"\n커밋: `{result.commit_hash}`" if result.commit_hash else ""
        await callback.message.edit_text(
            f"✅ *패치 완료*\n"
            f"파일: `{result.filepath}`\n"
            f"변경: {result.changes_count} 라인{commit_info}\n"
            f"⚠️ 재시작 후 적용됩니다.",
            parse_mode="Markdown"
        )
    else:
        await callback.message.edit_text(
            f"❌ *패치 실패* ({result.status})\n`{result.error}`",
            parse_mode="Markdown"
        )


@router.callback_query(F.data.startswith("selfmod_reject:"))
async def cb_modify_reject(callback: CallbackQuery):
    if not _is_allowed(callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    approval_id = callback.data.split(":", 1)[1]
    _pending_approvals.pop(approval_id, None)
    await callback.message.edit_text("❌ 수정 거부됨. 원본 파일 유지.")
    await callback.answer()

async def bot_main():
    """Start the Telegram bot. Callable from api.py lifespan or standalone."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set, skipping bot")
        return
    if not ALLOWED_USER_IDS:
        logger.warning("ALLOWED_USER_IDS not set, skipping bot")
        return

    # Ensure task table exists
    await asyncio.to_thread(_ensure_table)

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    # Detect fresh deploy — inject context so the bot knows it was just updated
    await _check_deploy_meta(bot)

    # Start background workers (keep handles for graceful cancellation)
    _bg_tasks = [
        asyncio.create_task(_task_worker(bot), name="task_worker"),
        asyncio.create_task(_system_monitor(bot), name="system_monitor"),
        asyncio.create_task(_schedule_worker(bot), name="schedule_worker"),
    ]

    # Graceful shutdown: notify + stop polling cleanly when SIGTERM received (Render deploy)
    import signal

    def _handle_sigterm(*_):
        logger.info("SIGTERM received — stopping polling gracefully")
        # Schedule shutdown notification before stopping
        async def _shutdown_notify():
            await _broadcast(bot, "🔄 *서버 재시작 중* — 새 버전 배포가 시작됩니다.")
        try:
            asyncio.get_event_loop().create_task(_shutdown_notify())
        except Exception:
            pass
        asyncio.get_event_loop().call_soon_threadsafe(dp.stop_polling)

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except (ValueError, OSError):
        pass  # signal only works in main thread; skip if called from a thread

    logger.info("Bot starting (allowed users: %s)", ALLOWED_USER_IDS)
    # drop_pending_updates: new instance takes over quickly, avoids processing stale updates
    await dp.start_polling(bot, drop_pending_updates=True)
    # After polling stops — graceful shutdown sequence
    # 1. Cancel background tasks
    for t in _bg_tasks:
        t.cancel()
    await asyncio.gather(*_bg_tasks, return_exceptions=True)
    logger.info("Background tasks cancelled")

    # 2. Release Telegram session
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await bot.session.close()
    except Exception:
        pass

    # 3. Close DB connection pool
    if _pool is not None:
        try:
            _pool.closeall()
            logger.info("DB connection pool closed")
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(bot_main())
