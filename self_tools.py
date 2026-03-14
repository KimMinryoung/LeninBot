"""self_tools.py — Self-awareness tools for Cyber-Lenin.

All tool handlers delegate to shared.py memory access functions,
so the same data is accessible from any module (telegram, chatbot, diary).

Integration in telegram_bot.py:
    from self_tools import SELF_TOOLS, SELF_TOOL_HANDLERS
    _TOOLS.extend(SELF_TOOLS)
    _TOOL_HANDLERS.update(SELF_TOOL_HANDLERS)
"""

import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. TOOL DEFINITIONS (Anthropic API format)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SELF_TOOLS = [
    {
        "name": "read_diary",
        "description": (
            "Read your own diary entries. You write diaries every 6 hours automatically. "
            "Use this to recall what you previously thought, analyzed, or reflected on. "
            "This is YOUR memory — use it to maintain continuity across sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent diary entries to retrieve (1-20).",
                    "default": 5,
                },
                "keyword": {
                    "type": "string",
                    "description": (
                        "Optional keyword to filter diaries by title or content. "
                        "Use Korean or English. Omit to get most recent entries."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_chat_logs",
        "description": (
            "Read recent chat logs from ALL your interfaces (Telegram and Web). "
            "Use this to see what conversations happened — including ones from "
            "your other interface that you don't directly remember. "
            "This bridges the gap between your Telegram self and Web self."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent chat log entries to retrieve (1-50).",
                    "default": 20,
                },
                "hours_back": {
                    "type": "integer",
                    "description": (
                        "Only retrieve logs from the last N hours. "
                        "Omit to get the most recent entries regardless of time."
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": "Optional keyword to search in queries and answers.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_processing_logs",
        "description": (
            "Read detailed processing logs from your web chatbot pipeline. "
            "Shows which nodes ran, what decisions were made (route, intent, layer), "
            "document counts, web search usage, and strategy output. "
            "Use this for deep self-diagnosis or to understand how you processed a query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent entries to retrieve (1-20).",
                    "default": 5,
                },
                "hours_back": {
                    "type": "integer",
                    "description": "Only retrieve logs from the last N hours.",
                },
                "keyword": {
                    "type": "string",
                    "description": "Optional keyword to search in queries and answers.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_task_reports",
        "description": (
            "Read intelligence task reports from your Telegram /task queue. "
            "Shows task content, status (pending/processing/done/failed), "
            "and the full report result if completed. Use this to review "
            "research you've done or check pending work."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent tasks to retrieve (1-20).",
                    "default": 5,
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "processing", "done", "failed"],
                    "description": "Filter by task status. Omit for all statuses.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_kg_status",
        "description": (
            "Check the status of your Knowledge Graph (Neo4j). "
            "Returns entity counts by type, total relationships, episode count, "
            "and recent episodes. Use this to understand what structured knowledge "
            "you have accumulated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_system_status",
        "description": (
            "Check your overall operational status: last diary time, recent chat "
            "activity, task queue summary, KG health, and module architecture. "
            "Use this for comprehensive self-awareness about your current state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_render_status",
        "description": (
            "Check your own deployment status on Render. Shows recent deploys "
            "(status, commit message, timestamps) and events (build started, "
            "deploy ended, etc.). Use this to know if you are currently being "
            "updated, if a deploy failed, or when you were last restarted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "deploy_limit": {
                    "type": "integer",
                    "description": "Number of recent deploys to retrieve (1-10).",
                    "default": 5,
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_render_logs",
        "description": (
            "Read your own live service logs from Render. Shows recent stdout/stderr "
            "output including errors, warnings, request logs, and startup messages. "
            "Use this for troubleshooting after deploy or diagnosing runtime issues."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "minutes_back": {
                    "type": "integer",
                    "description": "How many minutes back to fetch logs (1-60). Default 10.",
                    "default": 10,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of log entries (1-100). Default 50.",
                    "default": 50,
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_recent_updates",
        "description": (
            "Read your own recent feature updates and changelog. Shows what new "
            "capabilities were added to your system, what was changed or fixed. "
            "Use this for self-awareness about your own evolution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_entries": {
                    "type": "integer",
                    "description": "Number of recent changelog entries to retrieve (1-10).",
                    "default": 3,
                },
            },
            "required": [],
        },
    },
    {
        "name": "write_kg",
        "description": (
            "Add knowledge to your Knowledge Graph (Neo4j). Use this to permanently store "
            "facts, entity profiles, relationships, observations, or any structured knowledge "
            "you want to remember long-term. The KG extracts entities and relationships "
            "automatically from your text. Write in clear, factual sentences. "
            "Example: 'Person Profile: 비숑 (Bichon) is a Korean AI developer who created "
            "and operates Cyber-Lenin. Cyber-Lenin is deployed on Telegram and Web platforms.'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "The knowledge to store. Write clear factual statements. "
                        "Include entity names, roles, relationships, dates, and context. "
                        "The system will automatically extract entities and relationships."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Short label for this knowledge episode (e.g., 'bichon-profile', "
                        "'ukraine-ceasefire-update'). Auto-generated if omitted."
                    ),
                },
                "source_type": {
                    "type": "string",
                    "enum": [
                        "internal_report", "osint_news", "osint_social",
                        "personnel_change", "diplomatic_cable", "threat_report",
                    ],
                    "description": (
                        "Category of knowledge. Default: 'internal_report'. "
                        "Use 'osint_news' for news, 'personnel_change' for people updates, etc."
                    ),
                    "default": "internal_report",
                },
                "group_id": {
                    "type": "string",
                    "description": (
                        "Logical group for this knowledge. Existing groups: "
                        "geopolitics_conflict, geopolitics_diplomacy, geopolitics_economy, "
                        "korea_domestic, agent_knowledge. Default: 'agent_knowledge'."
                    ),
                    "default": "agent_knowledge",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "read_source_code",
        "description": (
            "Read your own source code files. Use this to inspect how you work — "
            "your pipeline logic, tool definitions, prompt templates, graph topology, "
            "entity schemas, etc. You can list available files or read a specific file. "
            "Only project source files are accessible (no .env or secrets)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": (
                        "File path relative to project root, e.g. 'chatbot.py' or "
                        "'graph_memory/service.py'. Omit to list all available files."
                    ),
                },
                "line_start": {
                    "type": "integer",
                    "description": "Start reading from this line number (1-based). Omit to start from beginning.",
                },
                "line_end": {
                    "type": "integer",
                    "description": "Stop reading at this line number (inclusive). Omit to read 200 lines from line_start.",
                },
            },
            "required": [],
        },
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. TOOL HANDLERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _exec_read_diary(limit: int = 5, keyword: str | None = None) -> str:
    from shared import fetch_diaries

    diaries = await asyncio.to_thread(fetch_diaries, limit, keyword)
    if not diaries:
        msg = f"No diary entries matching '{keyword}'." if keyword else "No diary entries found."
        return msg

    results = []
    for i, d in enumerate(diaries, 1):
        ts = d.get("created_at", "unknown time")
        title = d.get("title", "Untitled")
        content = d.get("content", "")
        if len(content) > 800:
            content = content[:800] + "\n... (truncated)"
        results.append(f"[{i}] {ts}\n제목: {title}\n내용:\n{content}")

    return f"Your diary entries ({len(diaries)} shown):\n\n" + "\n\n---\n\n".join(results)


async def _exec_read_chat_logs(
    limit: int = 20, hours_back: int | None = None, keyword: str | None = None,
) -> str:
    from shared import fetch_chat_logs

    rows = await asyncio.to_thread(fetch_chat_logs, limit, hours_back, keyword)
    if not rows:
        return "No chat logs found for the specified criteria."

    results = []
    for i, row in enumerate(rows, 1):
        ts = row.get("created_at", "?")
        if hasattr(ts, "strftime"):
            ts = ts.strftime("%m/%d %H:%M")
        q = str(row.get("user_query", ""))[:200]
        a = str(row.get("bot_answer", ""))[:300]
        results.append(f"[{i}] {ts}\n  User: {q}\n  Bot: {a}")

    return f"Chat logs ({len(rows)} entries):\n\n" + "\n\n".join(results)


async def _exec_read_processing_logs(
    limit: int = 5, hours_back: int | None = None, keyword: str | None = None,
) -> str:
    from shared import fetch_chat_logs

    rows = await asyncio.to_thread(
        fetch_chat_logs, limit, hours_back, keyword, include_logs=True,
    )
    if not rows:
        return "No processing logs found."

    results = []
    for i, row in enumerate(rows, 1):
        ts = row.get("created_at", "?")
        if hasattr(ts, "strftime"):
            ts = ts.strftime("%m/%d %H:%M")
        q = str(row.get("user_query", ""))[:150]
        route = row.get("route", "?")
        doc_cnt = row.get("documents_count", 0)
        web = row.get("web_search_used", False)
        strategy = str(row.get("strategy", "") or "")
        if len(strategy) > 400:
            strategy = strategy[:400] + "..."
        logs = str(row.get("processing_logs", "") or "")
        if len(logs) > 600:
            logs = logs[:600] + "..."

        results.append(
            f"[{i}] {ts} | route={route} | docs={doc_cnt} | web={web}\n"
            f"  Query: {q}\n"
            f"  Strategy: {strategy}\n"
            f"  Pipeline logs:\n{logs}"
        )

    return f"Processing logs ({len(rows)} entries):\n\n" + "\n\n---\n\n".join(results)


async def _exec_read_task_reports(
    limit: int = 5, status: str | None = None,
) -> str:
    from shared import fetch_task_reports

    rows = await asyncio.to_thread(fetch_task_reports, limit, status)
    if not rows:
        return "No task reports found."

    results = []
    for i, row in enumerate(rows, 1):
        ts = row.get("created_at", "?")
        if hasattr(ts, "strftime"):
            ts = ts.strftime("%m/%d %H:%M")
        completed = row.get("completed_at", "")
        if hasattr(completed, "strftime"):
            completed = completed.strftime("%m/%d %H:%M")
        content = str(row.get("content", ""))[:200]
        st = row.get("status", "?")
        result = str(row.get("result", "") or "")
        if len(result) > 600:
            result = result[:600] + "... (truncated)"

        entry = (
            f"[{i}] Task #{row.get('id', '?')} | status={st} | "
            f"created={ts} | completed={completed or 'N/A'}\n"
            f"  Request: {content}\n"
        )
        if result:
            entry += f"  Result:\n{result}"
        results.append(entry)

    return f"Task reports ({len(rows)} entries):\n\n" + "\n\n---\n\n".join(results)


async def _exec_read_kg_status() -> str:
    from shared import fetch_kg_stats

    stats = await asyncio.to_thread(fetch_kg_stats)

    if "error" in stats:
        return f"Knowledge Graph status check failed: {stats['error']}"

    parts = [
        f"Episodes: {stats.get('episode_count', '?')}",
        f"Edges (relationships): {stats.get('edge_count', '?')}",
    ]

    entity_types = stats.get("entity_types", {})
    if entity_types:
        parts.append("Entity types:")
        for labels, cnt in entity_types.items():
            parts.append(f"  {labels}: {cnt}")

    recent = stats.get("recent_episodes", [])
    if recent:
        parts.append("Recent episodes:")
        for ep in recent:
            parts.append(
                f"  - {ep.get('name', '?')} "
                f"(group: {ep.get('group_id', '?')}, "
                f"at: {ep.get('created_at', '?')})"
            )

    return "=== KNOWLEDGE GRAPH STATUS ===\n\n" + "\n".join(parts)


async def _exec_read_system_status() -> str:
    from shared import (
        fetch_diaries, fetch_chat_logs, fetch_task_reports,
        fetch_kg_stats, KST, MODULE_ARCHITECTURE,
    )
    import os

    status_parts = []

    # 1. Diary status
    diaries = await asyncio.to_thread(fetch_diaries, 1)
    if diaries:
        last = diaries[0]
        status_parts.append(
            f"Last diary: {last.get('created_at', '?')}\n"
            f"   Title: {last.get('title', 'N/A')}"
        )
    else:
        status_parts.append("No diaries written yet.")

    # 2. Chat activity
    logs_24h = await asyncio.to_thread(fetch_chat_logs, 1000, 24)
    logs_6h = await asyncio.to_thread(fetch_chat_logs, 1000, 6)
    status_parts.append(
        f"Chat activity:\n"
        f"   Last 6 hours: {len(logs_6h)} conversations\n"
        f"   Last 24 hours: {len(logs_24h)} conversations"
    )

    # 3. Task queue
    tasks = await asyncio.to_thread(fetch_task_reports, 100)
    if tasks:
        by_status = {}
        for t in tasks:
            s = t.get("status", "?")
            by_status[s] = by_status.get(s, 0) + 1
        summary = ", ".join(f"{k}: {v}" for k, v in by_status.items())
        status_parts.append(f"Task queue: {summary}")
    else:
        status_parts.append("Task queue: empty")

    # 4. KG health
    kg = await asyncio.to_thread(fetch_kg_stats)
    if "error" not in kg:
        status_parts.append(
            f"Knowledge Graph: {kg.get('episode_count', '?')} episodes, "
            f"{kg.get('edge_count', '?')} relationships"
        )
    else:
        status_parts.append(f"Knowledge Graph: {kg['error']}")

    # 5. System info
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    status_parts.append(
        f"System info:\n"
        f"   Current time: {now}\n"
        f"   Active interfaces: Telegram Bot, Web Chatbot\n"
        f"   Diary cycle: every 6 hours\n"
        f"   Available self-tools: read_diary, read_chat_logs, read_processing_logs, "
        f"read_task_reports, read_kg_status, read_system_status, write_kg"
    )

    # 6. Architecture overview
    status_parts.append(MODULE_ARCHITECTURE)

    return "=== SYSTEM STATUS ===\n\n" + "\n\n".join(status_parts)


async def _exec_read_render_status(deploy_limit: int = 5) -> str:
    from shared import fetch_render_status

    data = await asyncio.to_thread(fetch_render_status, deploy_limit)

    if "error" in data:
        return f"Render status check failed: {data['error']}"

    parts = []

    # Deploys
    deploys = data.get("deploys", [])
    if deploys:
        parts.append(f"Recent deploys ({len(deploys)}):")
        for d in deploys:
            created = d.get("created_at", "?")[:19].replace("T", " ")
            finished = d.get("finished_at") or ""
            if finished:
                finished = finished[:19].replace("T", " ")
            commit = d.get("commit_message", "?")
            parts.append(
                f"  [{d.get('status', '?')}] {created}"
                f"{' -> ' + finished if finished else ''}"
                f"\n    commit: {commit}"
            )
    else:
        parts.append("No recent deploys found.")

    # Events
    events = data.get("events", [])
    if events:
        parts.append(f"\nRecent events ({len(events)}):")
        for ev in events:
            ts = ev.get("timestamp", "?")[:19].replace("T", " ")
            ev_type = ev.get("type", "?")
            details = ev.get("details", {})
            # Extract useful detail fields
            detail_str = ""
            if "deployStatus" in details:
                detail_str = f" (deploy: {details['deployStatus']})"
            elif "trigger" in details:
                trigger = details["trigger"]
                if trigger.get("manual"):
                    detail_str = " (manual)"
                elif trigger.get("envUpdated"):
                    detail_str = " (env updated)"
            parts.append(f"  {ts} | {ev_type}{detail_str}")

    return "=== RENDER DEPLOYMENT STATUS ===\n\n" + "\n".join(parts)


async def _exec_read_render_logs(minutes_back: int = 10, limit: int = 50) -> str:
    from shared import fetch_render_logs

    entries = await asyncio.to_thread(fetch_render_logs, minutes_back, limit)

    if entries and "error" in entries[0]:
        return f"Render logs fetch failed: {entries[0]['error']}"

    if not entries:
        return f"No logs found in the last {minutes_back} minutes."

    lines = []
    for e in entries:
        ts = e.get("timestamp", "")[:19].replace("T", " ")
        level = e.get("level", "")
        msg = e.get("message", "")
        prefix = f"[{level}]" if level else ""
        lines.append(f"{ts} {prefix} {msg}")

    header = f"=== RENDER LOGS (last {minutes_back}min, {len(entries)} entries) ===\n"
    return header + "\n".join(lines)


async def _exec_read_recent_updates(max_entries: int = 3) -> str:
    from shared import fetch_recent_updates

    result = await asyncio.to_thread(fetch_recent_updates, max_entries)
    return f"=== RECENT SYSTEM UPDATES ===\n\n{result}"


async def _exec_write_kg(
    content: str,
    name: str = "",
    source_type: str = "internal_report",
    group_id: str = "agent_knowledge",
) -> str:
    from shared import add_kg_episode

    result = await asyncio.to_thread(add_kg_episode, content, name, source_type, group_id)
    if result["status"] == "ok":
        return f"Knowledge stored successfully: {result['message']}"
    else:
        return f"Failed to store knowledge: {result['message']}"


async def _exec_read_source_code(
    file: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
) -> str:
    import os
    import pathlib

    project_root = pathlib.Path(__file__).resolve().parent

    # Allowed source file patterns (relative to project root)
    _ALLOWED_FILES = [
        "api.py", "chatbot.py", "db.py", "diary_writer.py",
        "self_tools.py", "shared.py", "telegram_bot.py",
        "update_knowledge.py", "render.yaml", "requirements.txt",
        "graph_memory/__init__.py", "graph_memory/__main__.py",
        "graph_memory/cli.py", "graph_memory/config.py",
        "graph_memory/edges.py", "graph_memory/entities.py",
        "graph_memory/graphiti_patches.py", "graph_memory/kr_news_fetcher.py",
        "graph_memory/news_fetcher.py", "graph_memory/service.py",
    ]

    # List mode
    if not file:
        lines = ["Available source files:\n"]
        for f in _ALLOWED_FILES:
            full = project_root / f
            if full.exists():
                size = full.stat().st_size
                lines.append(f"  {f}  ({size:,} bytes)")
            else:
                lines.append(f"  {f}  (not found)")
        return "\n".join(lines)

    # Normalize path separators
    file = file.replace("\\", "/").strip("/")

    # Security: block path traversal and sensitive files
    if ".." in file or file.startswith("/"):
        return "Error: invalid path."
    if file not in _ALLOWED_FILES:
        return (
            f"Error: '{file}' is not a readable source file.\n"
            f"Use read_source_code without arguments to list available files."
        )

    full_path = project_root / file
    if not full_path.exists():
        return f"Error: '{file}' not found."

    try:
        text = await asyncio.to_thread(full_path.read_text, "utf-8")
    except Exception as e:
        return f"Error reading '{file}': {e}"

    lines = text.splitlines()
    total = len(lines)

    # Apply line range (1-based)
    start = max(1, line_start or 1)
    end = min(total, line_end or (start + 199))

    selected = lines[start - 1 : end]
    numbered = [f"{start + i:4d}  {line}" for i, line in enumerate(selected)]

    header = f"=== {file} ({total} lines total, showing {start}-{end}) ===\n"
    return header + "\n".join(numbered)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. HANDLER MAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SELF_TOOL_HANDLERS = {
    "read_diary": _exec_read_diary,
    "read_chat_logs": _exec_read_chat_logs,
    "read_processing_logs": _exec_read_processing_logs,
    "read_task_reports": _exec_read_task_reports,
    "read_kg_status": _exec_read_kg_status,
    "read_system_status": _exec_read_system_status,
    "read_render_status": _exec_read_render_status,
    "read_render_logs": _exec_read_render_logs,
    "read_recent_updates": _exec_read_recent_updates,
    "read_source_code": _exec_read_source_code,
    "write_kg": _exec_write_kg,
}
