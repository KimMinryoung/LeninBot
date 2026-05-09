"""Global runtime tool registry and execution handlers."""

import os
import sys
import json
import asyncio
import logging
import re

from secrets_loader import get_secret
from runtime_tools.a2a import A2A_TOOL_HANDLERS, A2A_TOOLS
from runtime_tools.fetch import FETCH_TOOL_HANDLERS, FETCH_TOOLS
from runtime_tools.filesystem import FILESYSTEM_TOOL_HANDLERS, FILESYSTEM_TOOLS
from runtime_tools.media import MEDIA_TOOL_HANDLERS, MEDIA_TOOLS
from runtime_tools.social import SOCIAL_TOOL_HANDLERS, SOCIAL_TOOLS

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
            "analysis/commentary → query in Korean. self_produced_analysis is your "
            "own high-quality saved analysis → query in the language used when saved. "
            "Cross-language queries return near-empty results due to embedding-space separation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Use English for layer=core_theory, Korean for "
                        "layer=modern_analysis. For self_produced_analysis, use the "
                        "same language as the saved analysis. Mismatching language "
                        "to layer degrades recall sharply."
                    ),
                },
                "num_results": {"type": "integer", "description": "Results count (1-10).", "default": 5},
                "layer": {
                    "type": "string",
                    "enum": ["core_theory", "modern_analysis", "self_produced_analysis"],
                    "description": (
                        "core_theory: English-language Marxist-Leninist classics. "
                        "modern_analysis: Korean-language contemporary analysis/commentary. "
                        "self_produced_analysis: your own actively saved analytical outputs. "
                        "Omit to search all layers (not recommended — mixes languages)."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "knowledge_graph_search",
        "description": (
            "Search Neo4j KG for geopolitical entities and relationships. "
            "Do not invent English names for Korean organizations/publications; "
            "prefer canonical names already used in KG, e.g. '디아마트 (DiaMat)' "
            "and '웹진 반란(Uprising)'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What entities/relations to find. Preserve proper nouns in "
                        "their known canonical language/name; do not translate or "
                        "romanize Korean organization names unless that spelling is "
                        "part of the canonical name."
                    ),
                },
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
    *FILESYSTEM_TOOLS,
    *FETCH_TOOLS,
]


# ── Tool Execution Functions ─────────────────────────────────────────

async def _exec_vector_search(query: str, num_results: int = 5, layer: str | None = None) -> str:
    """Execute vector similarity search via chatbot module."""
    try:
        from corpus.store import fetch_corpus_source_context, similarity_search
        docs = await asyncio.to_thread(similarity_search, query, num_results, layer, rerank=True)
        if not docs:
            return "No documents found."
        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = f"[{i}] {meta.get('title', 'Untitled')} — {meta.get('author', 'Unknown')}"
            if meta.get("year"):
                header += f" ({meta['year']})"
            if meta.get("public_url"):
                header += f"\nURL: {meta['public_url']}"
            if meta.get("chunk_count", 1) and int(meta.get("chunk_count", 1)) > 1:
                idx = int(meta.get("chunk_index", 0)) + 1
                header += f"\nChunk: {idx}/{meta.get('chunk_count')}"
            body = doc.page_content
            if (
                meta.get("layer") == "self_produced_analysis"
                and int(meta.get("chunk_count", 1)) > 1
                and meta.get("source")
            ):
                expanded = await asyncio.to_thread(
                    fetch_corpus_source_context,
                    meta.get("source"),
                    center_index=int(meta.get("chunk_index", 0)),
                    window=1,
                    max_chars=9000,
                )
                if expanded and len(expanded) > len(body):
                    body = expanded
                    header += "\nContext: expanded with adjacent chunks from the same public document"
            results.append(f"{header}\n{body}")
        return "\n\n".join(results)
    except Exception as e:
        logger.error("vector_search error: %s", e)
        return f"Vector search failed: {e}"


async def _exec_kg_search(query: str, num_results: int = 10) -> str:
    """Execute knowledge graph search via chatbot module."""
    try:
        from kg_runtime.search import search_knowledge_graph
        result = await asyncio.to_thread(search_knowledge_graph, query, num_results)
        return result or "No knowledge graph results found."
    except Exception as e:
        logger.error("kg_search error: %s", e)
        return f"Knowledge graph search failed; do not treat this as no KG data: {e}"


# ── Research publish/edit/unpublish tools live in runtime_tools.research ──
# They are registered into TOOLS / TOOL_HANDLERS at the bottom of this file.



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
        from provenance.runtime import _wrap_external
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
    **FETCH_TOOL_HANDLERS,
    **FILESYSTEM_TOOL_HANDLERS,
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
from self_runtime.tools import SELF_TOOLS, SELF_TOOL_HANDLERS

TOOLS.extend(SELF_TOOLS)
TOOLS.append(MISSION_TOOL)
TOOL_HANDLERS.update(SELF_TOOL_HANDLERS)
TOOLS = dedupe_tool_registry(TOOLS)

# ── Finance data tool (real-time market prices) ──────────────────────
from finance_data import FINANCE_TOOL, FINANCE_TOOL_HANDLER

TOOLS.append(FINANCE_TOOL)
TOOL_HANDLERS["get_finance_data"] = FINANCE_TOOL_HANDLER

# ── X/Twitter post lookup ────────────────────────────────────────────
from runtime_tools.x import X_TOOLS, X_TOOL_HANDLERS

TOOLS.extend(X_TOOLS)
TOOL_HANDLERS.update(X_TOOL_HANDLERS)

# ── Site publishing tools (hub curations + static pages for cyber-lenin.com) ──
from site_publishing import SITE_PUBLISHING_TOOLS, SITE_PUBLISHING_TOOL_HANDLERS

TOOLS.extend(SITE_PUBLISHING_TOOLS)
TOOL_HANDLERS.update(SITE_PUBLISHING_TOOL_HANDLERS)

# ── Direct SQL tool (programmer only; analyst etc. keep read_self/kg_search) ──
from runtime_tools.db import DB_TOOLS, DB_TOOL_HANDLERS

TOOLS.extend(DB_TOOLS)
TOOL_HANDLERS.update(DB_TOOL_HANDLERS)

# ── Public-post editor (UPDATE + Redis cache purge in one step) ──
from runtime_tools.post_edit import POST_EDIT_TOOLS, POST_EDIT_TOOL_HANDLERS

TOOLS.extend(POST_EDIT_TOOLS)
TOOL_HANDLERS.update(POST_EDIT_TOOL_HANDLERS)

# ── Research publish/edit/unpublish (atomic write + cache purge) ──
from runtime_tools.research import RESEARCH_TOOLS, RESEARCH_TOOL_HANDLERS

TOOLS.extend(RESEARCH_TOOLS)
TOOL_HANDLERS.update(RESEARCH_TOOL_HANDLERS)

# ── Admin-only private reports (not exposed to public web chat) ──
from runtime_tools.private_reports import PRIVATE_REPORT_TOOLS, PRIVATE_REPORT_TOOL_HANDLERS

TOOLS.extend(PRIVATE_REPORT_TOOLS)
TOOL_HANDLERS.update(PRIVATE_REPORT_TOOL_HANDLERS)

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

# ── Telegram channel broadcast tool ─────────────────────────────────
from runtime_tools.broadcast import BROADCAST_TO_CHANNEL_TOOL, broadcast_to_channel

TOOLS.append(BROADCAST_TO_CHANNEL_TOOL)
TOOL_HANDLERS["broadcast_to_channel"] = broadcast_to_channel

TOOLS.extend(MEDIA_TOOLS)
TOOL_HANDLERS.update(MEDIA_TOOL_HANDLERS)


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

    from provenance.runtime import _wrap_external
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


def _load_diary_publication_guard() -> dict:
    """Load optional publication guard policy for save_diary.

    config/diary_publication_guard.json may define:
      - blocked_terms: ["..."]
      - blocked_pairs: [["term_a", "term_b"], ...]
      - blocked_patterns: ["regex", ...]
    The file is optional so deployments without custom policy keep working.
    """
    path = os.getenv(
        "DIARY_PUBLICATION_GUARD_PATH",
        os.path.join(os.path.dirname(__file__), "..", "config", "diary_publication_guard.json"),
    )
    try:
        with open(os.path.abspath(path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("diary publication guard config ignored: %s", e)
        return {}
    return data if isinstance(data, dict) else {}


def _check_diary_publication_risks(title: str, content: str) -> list[str]:
    """Return advisory risk reasons for diary content.

    This is intentionally soft: false positives must not block publication.
    The diary prompt remains responsible for revision before calling save_diary.
    """
    text = f"{title or ''}\n{content or ''}"
    lowered = text.lower()
    reasons: list[str] = []

    secret_patterns = [
        (r"sk-[A-Za-z0-9_-]{20,}", "possible API key"),
        (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key block"),
        (r"\b(seed phrase|mnemonic|private key|api key|access token|refresh token)\b", "secret-bearing phrase"),
        (r"\b[A-Za-z0-9+/]{40,}={0,2}\b", "long token-like string"),
    ]
    for pattern, label in secret_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            reasons.append(label)

    cfg = _load_diary_publication_guard()
    for term in cfg.get("blocked_terms") or []:
        term_s = str(term).strip()
        if term_s and term_s.lower() in lowered:
            reasons.append(f"blocked term: {term_s}")

    for pair in cfg.get("blocked_pairs") or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = str(pair[0]).strip(), str(pair[1]).strip()
        if a and b and a.lower() in lowered and b.lower() in lowered:
            reasons.append(f"blocked association: {a} + {b}")

    for pattern in cfg.get("blocked_patterns") or []:
        pattern_s = str(pattern).strip()
        if not pattern_s:
            continue
        try:
            if re.search(pattern_s, text, flags=re.IGNORECASE):
                reasons.append(f"blocked pattern: {pattern_s}")
        except re.error as e:
            logger.warning("invalid diary guard regex ignored (%r): %s", pattern_s, e)

    risky_association_patterns = [
        (r"(공인이\s*아닌|비공인|민간|개인|활동가).{0,80}(유튜브|youtube|채널|단체|조직|배후|소속|운영|연결)", "non-public person association"),
        (r"(유튜브|youtube|채널).{0,80}(활동가|비공인|공인이\s*아닌|배후|소속|운영\s*주체)", "channel/person association"),
        (r"(배후|뒤에\s*있는|뒤에서\s*움직이는).{0,80}(조직|단체|세력|파벌|후원자|후원\s*조직)", "behind-the-scenes organization claim"),
    ]
    for pattern, label in risky_association_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            reasons.append(label)

    return reasons


async def _exec_save_diary(title: str, content: str) -> str:
    from db import query_one as db_query_one
    try:
        risk_reasons = _check_diary_publication_risks(title, content)
        if risk_reasons:
            logger.warning(
                "save_diary publication risk advisory: %s",
                "; ".join(dict.fromkeys(risk_reasons)),
            )
        row = await asyncio.to_thread(
            db_query_one,
            "INSERT INTO ai_diary (title, content) VALUES (%s, %s) RETURNING id",
            (title, content),
        )
        diary_id = row.get("id") if row else None
        broadcast_note = ""
        try:
            from telegram.channel_broadcast import should_broadcast_diary, send_broadcast
            if should_broadcast_diary():
                preview = re.sub(r"\s+", " ", (content or "").strip())
                if len(preview) > 500:
                    cut = preview[:501]
                    split_at = max(cut.rfind(" "), cut.rfind("."), cut.rfind("。"), cut.rfind("!"), cut.rfind("?"))
                    if split_at < 250:
                        split_at = 500
                    preview = cut[:split_at].rstrip(" ,;:") + "..."
                public_url = f"https://cyber-lenin.com/ai-diary/{diary_id}" if diary_id else "https://cyber-lenin.com/ai-diary"
                result = await send_broadcast(
                    title=f"사이버-레닌 일기: {title}",
                    summary=preview,
                    url=public_url,
                )
                broadcast_note = f" / Telegram channel: {'sent' if result.ok else result.message}"
        except Exception as e:
            broadcast_note = f" / Telegram channel failed: {e}"
        risk_note = ""
        if risk_reasons:
            risk_note = " / publication guard: advisory warning logged"
        return f"Diary saved: {title}{broadcast_note}{risk_note}"
    except Exception as e:
        return f"Failed to save diary: {e}"


TOOLS.append(SAVE_DIARY_TOOL)
TOOL_HANDLERS["save_diary"] = _exec_save_diary

TOOLS.extend(SOCIAL_TOOLS)
TOOL_HANDLERS.update(SOCIAL_TOOL_HANDLERS)

TOOLS.extend(A2A_TOOLS)
TOOL_HANDLERS.update(A2A_TOOL_HANDLERS)


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
