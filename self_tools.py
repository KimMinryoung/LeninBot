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
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))


def _to_kst(ts) -> str:
    """Convert a timestamp (datetime or ISO string) to KST formatted string."""
    if ts is None:
        return "?"
    if isinstance(ts, str):
        if not ts or ts == "?":
            return ts
        # ISO string from APIs (e.g. "2026-03-14T03:00:57Z")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.astimezone(_KST).strftime("%m/%d %H:%M KST")
        except (ValueError, TypeError):
            return ts[:19].replace("T", " ")
    if hasattr(ts, "astimezone"):
        return ts.astimezone(_KST).strftime("%m/%d %H:%M KST")
    if hasattr(ts, "strftime"):
        return ts.strftime("%m/%d %H:%M")
    return str(ts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. TOOL DEFINITIONS (Anthropic API format)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SELF_TOOLS = [
    {
        "name": "read_self",
        "description": (
            "Read internal data. source: diary (6h entries), chat_logs (telegram/web), "
            "processing_logs (pipeline), task_reports (queue), kg_status (graph stats), "
            "system_status (overview), server_logs (journald), recent_updates (changelog)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["diary", "chat_logs", "processing_logs", "task_reports",
                             "kg_status", "system_status", "server_logs", "recent_updates"],
                },
                "limit": {"type": "integer", "description": "Results count."},
                "keyword": {"type": "string", "description": "Filter keyword."},
                "hours_back": {"type": "integer", "description": "Only last N hours."},
                "service": {"type": "string", "enum": ["telegram", "api", "nginx"], "description": "For server_logs."},
                "grep": {"type": "string", "description": "For server_logs: filter text."},
                "status": {"type": "string", "enum": ["pending", "processing", "done", "failed"], "description": "For task_reports."},
                "chat_source": {"type": "string", "enum": ["telegram", "web"], "description": "For chat_logs. Default: web."},
            },
            "required": ["source"],
        },
    },
    {
        "name": "write_kg",
        "description": "Store facts/entities/relationships in KG permanently. Write clear factual sentences.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Factual statements to store."},
                "name": {"type": "string", "description": "Episode label (auto-generated if omitted)."},
                "source_type": {
                    "type": "string",
                    "enum": ["internal_report", "osint_news", "osint_social", "personnel_change", "diplomatic_cable", "threat_report"],
                    "default": "internal_report",
                },
                "group_id": {
                    "type": "string",
                    "description": "Group: geopolitics_conflict/diplomacy/economy, korea_domestic, agent_knowledge.",
                    "default": "agent_knowledge",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "create_task",
        "description": "Create async background task (Sonnet, 50 rounds, $1 budget). Use for: deep research, multi-step coding/patching, file edits requiring multiple tool calls, or any task where tool-call limits could interrupt progress. Prefer this over direct multi-step execution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to research, analyze, or implement. Be specific: include file paths, requirements, constraints, and expected outcome."},
                "priority": {"type": "string", "enum": ["high", "normal", "low"], "default": "normal"},
                "parent_task_id": {"type": "integer", "description": "Parent task ID for task chaining (optional). Child reads mission timeline for context."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall_experience",
        "description": "Search your experiential memory (past lessons, mistakes, insights, patterns). Stored daily from all conversations and tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What experience to recall (semantic search)."},
                "limit": {"type": "integer", "description": "Max results (1-10).", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "kg_admin",
        "description": "KG admin ops. action: query (Cypher), delete_episode, merge_entities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["query", "delete_episode", "merge_entities"]},
                "query": {"type": "string", "description": "Cypher query (action=query)."},
                "write": {"type": "boolean", "description": "Allow writes (action=query). Default false."},
                "episode_name": {"type": "string", "description": "Episode name (action=delete_episode)."},
                "source_name": {"type": "string", "description": "Entity to merge FROM (action=merge_entities)."},
                "target_name": {"type": "string", "description": "Entity to merge INTO (action=merge_entities)."},
            },
            "required": ["action"],
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
        ts = _to_kst(d.get("created_at"))
        title = d.get("title", "Untitled")
        content = d.get("content", "")
        if len(content) > 800:
            content = content[:800] + "\n... (truncated)"
        results.append(f"[{i}] {ts}\n제목: {title}\n내용:\n{content}")

    return f"Your diary entries ({len(diaries)} shown):\n\n" + "\n\n---\n\n".join(results)


async def _exec_read_chat_logs(
    limit: int = 20, hours_back: int | None = None, keyword: str | None = None,
    source: str = "web",
) -> str:
    from shared import fetch_chat_logs

    rows = await asyncio.to_thread(fetch_chat_logs, limit, hours_back, keyword, source=source)
    if not rows:
        return "No chat logs found for the specified criteria."

    results = []
    for i, row in enumerate(rows, 1):
        ts = _to_kst(row.get("created_at"))
        role = str(row.get("role", "") or "").lower()
        content = str(row.get("content", "") or "")
        if role in ("user", "assistant") and content:
            label = "User" if role == "user" else "Bot"
            text = content[:300]
            results.append(f"[{i}] {ts}\n  {label}: {text}")
            continue

        q = str(row.get("user_query", "") or "")[:200]
        a = str(row.get("bot_answer", "") or "")[:300]
        lines = [f"[{i}] {ts}"]
        if q:
            lines.append(f"  User: {q}")
        if a:
            lines.append(f"  Bot: {a}")
        if not q and not a and content:
            lines.append(f"  Msg: {content[:300]}")
        results.append("\n".join(lines))

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
        ts = _to_kst(row.get("created_at"))
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
        ts = _to_kst(row.get("created_at"))
        completed = _to_kst(row.get("completed_at")) if row.get("completed_at") else ""
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
        parts.append(f"\nRecent episodes ({len(recent)} most recent):")
        for ep in recent:
            source = ep.get("source", "")
            source_tag = f" [{source}]" if source else ""
            parts.append(
                f"\n  📌 [{_to_kst(ep.get('created_at'))}] "
                f"{ep.get('name', '?')}"
                f"{source_tag} "
                f"(group: {ep.get('group_id', '?')})"
            )
            # Show extracted entities
            entities = ep.get("entities", [])
            if entities:
                ent_strs = [
                    f"{e['name']} ({','.join(l for l in e.get('labels', []) if l != 'Entity')})"
                    if any(l != 'Entity' for l in e.get('labels', []))
                    else e['name']
                    for e in entities[:8]
                ]
                more = f" +{len(entities) - 8} more" if len(entities) > 8 else ""
                parts.append(f"    Entities: {', '.join(ent_strs)}{more}")
            # Show extracted facts
            facts = ep.get("facts", [])
            if facts:
                for f in facts[:5]:
                    parts.append(f"    → {f.get('fact', '?')}")
                if len(facts) > 5:
                    parts.append(f"    ... +{len(facts) - 5} more facts")

    return "=== KNOWLEDGE GRAPH STATUS ===\n\n" + "\n".join(parts)


async def _exec_read_system_status() -> str:
    from shared import (
        fetch_diaries, fetch_chat_logs, fetch_task_reports,
        fetch_kg_stats, KST, MODULE_ARCHITECTURE,
    )

    status_parts = []

    # 1. Diary status
    diaries = await asyncio.to_thread(fetch_diaries, 1)
    if diaries:
        last = diaries[0]
        status_parts.append(f"Last diary: {_to_kst(last.get('created_at'))} — {last.get('title', 'N/A')}")
    else:
        status_parts.append("No diaries written yet.")

    # 2. Chat activity (use small limits — we only need counts)
    logs_24h = await asyncio.to_thread(fetch_chat_logs, 100, 24)
    logs_6h = await asyncio.to_thread(fetch_chat_logs, 100, 6)
    status_parts.append(f"Chats: {len(logs_6h)} (6h), {len(logs_24h)} (24h)")

    # 3. Task queue
    tasks = await asyncio.to_thread(fetch_task_reports, 20)
    if tasks:
        by_status = {}
        for t in tasks:
            s = t.get("status", "?")
            by_status[s] = by_status.get(s, 0) + 1
        summary = ", ".join(f"{k}: {v}" for k, v in by_status.items())
        status_parts.append(f"Tasks: {summary}")
    else:
        status_parts.append("Tasks: none")

    # 4. KG health
    kg = await asyncio.to_thread(fetch_kg_stats)
    if "error" not in kg:
        status_parts.append(f"KG: {kg.get('episode_count', '?')} episodes, {kg.get('edge_count', '?')} edges")
    else:
        status_parts.append(f"KG: {kg['error']}")

    # 5. Time + architecture
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    status_parts.append(f"Time: {now}")
    status_parts.append(MODULE_ARCHITECTURE)

    return "=== SYSTEM STATUS ===\n" + "\n".join(status_parts)


async def _exec_read_server_logs(
    service: str = "telegram", minutes_back: int = 10, limit: int = 50, grep: str = "",
) -> str:
    """Read journald logs for a systemd service."""
    import subprocess

    service_map = {
        "telegram": "leninbot-telegram",
        "api": "leninbot-api",
        "nginx": "nginx",
    }
    unit = service_map.get(service, "leninbot-telegram")
    minutes_back = max(1, min(60, minutes_back))
    limit = max(1, min(200, limit))

    cmd = ["journalctl", "-u", unit, f"--since={minutes_back} min ago", "--no-pager", "-n", str(limit)]
    if grep:
        cmd.extend(["--grep", grep])

    try:
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=10,
        )
        output = result.stdout.strip() if result.stdout else "(no output)"
        if result.returncode != 0 and result.stderr:
            output += f"\n[stderr] {result.stderr.strip()}"
        return f"=== SERVER LOGS ({unit}, last {minutes_back}min) ===\n{output}"
    except subprocess.TimeoutExpired:
        return f"Log fetch timed out for {unit}."
    except Exception as e:
        return f"Log fetch failed: {e}"


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
        # Audit log for KG writes
        ts = datetime.now(_KST).strftime("%Y-%m-%d %H:%M KST")
        logger.info(
            "[KG AUDIT] wrote episode | name=%s | group=%s | source=%s | time=%s | content_len=%d",
            name or "(auto)", group_id, source_type, ts, len(content),
        )
        return f"Knowledge stored successfully: {result['message']}"
    else:
        return f"Failed to store knowledge: {result['message']}"


async def _exec_create_task(
    content: str,
    priority: str = "normal",
    parent_task_id: int | None = None,
) -> str:
    from shared import create_task_in_db

    result = await asyncio.to_thread(
        create_task_in_db, content, 0, priority, parent_task_id=parent_task_id,
    )
    if result["status"] == "ok":
        depth_info = f", depth={result.get('depth', 0)}" if parent_task_id else ""
        return f"Task #{result['task_id']} created (priority: {priority}{depth_info}). It will be processed in the background."
    else:
        return f"Failed to create task: {result['error']}"


    # read_source_code removed — replaced by read_file tool in telegram_bot.py


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. HANDLER MAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _exec_recall_experience(query: str, limit: int = 5) -> str:
    from shared import search_experiential_memory

    limit = max(1, min(10, limit))
    rows = await asyncio.to_thread(search_experiential_memory, query, limit)
    if not rows:
        return "No relevant experiential memories found."
    lines = []
    for r in rows:
        sim = f"{r.get('similarity', 0):.0%}"
        cat = r.get("category", "?")
        src = r.get("source_type", "?")
        ts = str(r.get("created_at", ""))[:10]
        lines.append(f"[{cat}|{src}|{ts}|sim={sim}] {r['content']}")
    return f"Found {len(rows)} experience(s):\n" + "\n\n".join(lines)


async def _exec_kg_query(query: str, write: bool = False) -> str:
    from shared import kg_cypher
    result = await asyncio.to_thread(kg_cypher, query, write)
    if "error" in result:
        return f"KG query failed: {result['error']}"
    rows = result.get("rows", [])
    count = result.get("count", 0)
    if not rows:
        return f"Query returned 0 rows. (write={write})"
    import json
    formatted = json.dumps(rows[:50], ensure_ascii=False, indent=2, default=str)
    suffix = f"\n... (+{count-50} more rows)" if count > 50 else ""
    return f"KG query result ({count} rows):\n{formatted}{suffix}"


async def _exec_kg_delete_episode(episode_name: str) -> str:
    from shared import kg_delete_episode
    result = await asyncio.to_thread(kg_delete_episode, episode_name)
    if "error" in result:
        return f"Delete failed: {result['error']}"
    if result.get("not_found"):
        return f"Episode not found: '{episode_name}'"
    return (
        f"✅ Episode deleted: '{episode_name}'\n"
        f"  Deleted episodes: {result.get('deleted_episode', 0)}\n"
        f"  Deleted orphaned entities: {result.get('deleted_entities', 0)}"
    )


async def _exec_kg_merge_entities(source_name: str, target_name: str) -> str:
    from shared import kg_merge_entities
    result = await asyncio.to_thread(kg_merge_entities, source_name, target_name)
    if "error" in result:
        return f"Merge failed: {result['error']}"
    return (
        f"✅ Entity merged: '{result['deleted_source']}' → '{result['merged_into']}'\n"
        f"  Outgoing relations transferred: {result.get('transferred_outgoing', 0)}\n"
        f"  Incoming relations transferred: {result.get('transferred_incoming', 0)}\n"
        f"  MENTIONS transferred: {result.get('transferred_mentions', 0)}"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. TASK-CONTEXT TOOL DEFINITIONS (injected only during task execution)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK_CONTEXT_TOOLS = [
    {
        "name": "save_finding",
        "description": "Save intermediate findings to the active mission timeline. Visible to both chat and future tasks. Use to preserve important progress, decisions, and discoveries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Text to save (findings, partial results, notes)."},
                "event_type": {
                    "type": "string",
                    "enum": ["finding", "decision"],
                    "default": "finding",
                    "description": "Type of event: finding (discovery/result) or decision (strategic choice).",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "request_continuation",
        "description": "Create a child task to continue unfinished work. Records progress to the mission timeline and spawns a new task. Use when budget/round limit is approaching.",
        "input_schema": {
            "type": "object",
            "properties": {
                "progress_summary": {"type": "string", "description": "What has been accomplished so far."},
                "next_steps": {"type": "string", "description": "What the child task should do next. Be specific."},
            },
            "required": ["progress_summary", "next_steps"],
        },
    },
]


def build_task_context_tools(task_id: int, user_id: int, depth: int = 0):
    """Build task-context tool handlers with task_id bound via closure.

    Returns (tools_list, handlers_dict) ready to merge into the tool loop.
    """

    async def _exec_save_finding(content: str, event_type: str = "finding") -> str:
        try:
            from telegram_mission import get_active_mission, add_mission_event
            mission = await asyncio.to_thread(get_active_mission, user_id)
            if not mission:
                return "No active mission — finding not saved."
            truncated = content[:2000]
            await asyncio.to_thread(
                add_mission_event, mission["id"], f"task#{task_id}", event_type, truncated
            )
            return f"Saved {event_type} to mission #{mission['id']} ({len(truncated)} chars)."
        except Exception as e:
            logger.error("save_finding error (task %d): %s", task_id, e)
            return f"Failed to save finding: {e}"

    async def _exec_request_continuation(progress_summary: str, next_steps: str) -> str:
        from shared import create_task_in_db

        # 1. Record progress to mission timeline
        try:
            from telegram_mission import get_active_mission, add_mission_event
            mission = await asyncio.to_thread(get_active_mission, user_id)
            if mission:
                event_content = f"Progress: {progress_summary[:500]}\nNext: {next_steps[:500]}"
                await asyncio.to_thread(
                    add_mission_event, mission["id"], f"task#{task_id}", "decision", event_content
                )
        except Exception:
            pass

        # 2. Create child task (no scratchpad — child reads mission timeline)
        result = await asyncio.to_thread(
            create_task_in_db,
            next_steps,
            user_id=user_id,
            parent_task_id=task_id,
        )

        if result["status"] == "ok":
            child_id = result["task_id"]
            child_depth = result.get("depth", depth + 1)
            return (
                f"Child task #{child_id} created (depth={child_depth}, parent=#{task_id}). "
                f"Progress saved to mission. You can now finish your current response."
            )
        else:
            return f"Failed to create continuation task: {result['error']}"

    handlers = {
        "save_finding": _exec_save_finding,
        "request_continuation": _exec_request_continuation,
    }
    return list(TASK_CONTEXT_TOOLS), handlers


async def _exec_read_self(
    source: str, limit: int | None = None, keyword: str | None = None,
    hours_back: int | None = None, service: str = "telegram",
    grep: str = "", status: str | None = None, chat_source: str = "web",
) -> str:
    """Dispatcher for all read_self sources."""
    if source == "diary":
        return await _exec_read_diary(limit=limit or 5, keyword=keyword)
    if source == "chat_logs":
        return await _exec_read_chat_logs(limit=limit or 20, hours_back=hours_back, keyword=keyword, source=chat_source)
    if source == "processing_logs":
        return await _exec_read_processing_logs(limit=limit or 5, hours_back=hours_back, keyword=keyword)
    if source == "task_reports":
        return await _exec_read_task_reports(limit=limit or 5, status=status)
    if source == "kg_status":
        return await _exec_read_kg_status()
    if source == "system_status":
        return await _exec_read_system_status()
    if source == "server_logs":
        return await _exec_read_server_logs(service=service, minutes_back=(hours_back or 1) * 60, limit=limit or 50, grep=grep)
    if source == "recent_updates":
        return await _exec_read_recent_updates(max_entries=limit or 3)
    return f"Unknown source: {source}"


async def _exec_kg_admin(
    action: str, query: str = "", write: bool = False,
    episode_name: str = "", source_name: str = "", target_name: str = "",
) -> str:
    """Dispatcher for KG admin operations."""
    if action == "query":
        if not query:
            return "Error: 'query' parameter required for action=query."
        return await _exec_kg_query(query=query, write=write)
    if action == "delete_episode":
        if not episode_name:
            return "Error: 'episode_name' parameter required for action=delete_episode."
        return await _exec_kg_delete_episode(episode_name=episode_name)
    if action == "merge_entities":
        if not source_name or not target_name:
            return "Error: 'source_name' and 'target_name' required for action=merge_entities."
        return await _exec_kg_merge_entities(source_name=source_name, target_name=target_name)
    return f"Unknown action: {action}"


SELF_TOOL_HANDLERS = {
    "read_self": _exec_read_self,
    "recall_experience": _exec_recall_experience,
    "write_kg": _exec_write_kg,
    "create_task": _exec_create_task,
    "kg_admin": _exec_kg_admin,
}
