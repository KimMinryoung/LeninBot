"""Server sync — push/pull data to central PostgreSQL and KG via shared.py + db.py."""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)


async def sync_push(data_type: str, content: str, metadata: dict) -> str:
    """Push data to the central server."""

    if data_type == "kg_episode":
        from shared import add_kg_episode
        name = metadata.get("name", "")
        group_id = metadata.get("group_id", "local_agent")
        source_type = metadata.get("source_type", "local_crawl")
        result = await asyncio.to_thread(
            add_kg_episode, content, name, source_type, group_id
        )
        return f"KG episode push: {json.dumps(result, ensure_ascii=False)}"

    elif data_type == "report":
        from shared import create_task_in_db
        priority = metadata.get("priority", "normal")
        result = await asyncio.to_thread(
            create_task_in_db, content, 0, priority
        )
        if result.get("task_id"):
            # Mark it as completed immediately with the content as result
            from db import execute as db_execute
            await asyncio.to_thread(
                db_execute,
                "UPDATE telegram_tasks SET status = 'done', result = %s, completed_at = NOW() WHERE id = %s",
                (content, result["task_id"]),
            )
            return f"Report saved as task #{result['task_id']} (completed)"
        return f"Report push result: {json.dumps(result, ensure_ascii=False)}"

    return f"Unknown data_type: {data_type}"


async def sync_pull(data_type: str, params: dict) -> str:
    """Pull data from the central server."""

    if data_type == "diaries":
        from shared import fetch_diaries
        limit = params.get("limit", 5)
        keyword = params.get("keyword")
        rows = await asyncio.to_thread(fetch_diaries, limit, keyword)
        if not rows:
            return "No diary entries found."
        lines = []
        for r in rows:
            date = str(r.get("created_at", ""))[:19]
            title = r.get("title", "Untitled")
            content = r.get("content", "")[:500]
            lines.append(f"[{date}] {title}\n{content}\n")
        return f"{len(rows)} diary entries:\n\n" + "\n".join(lines)

    elif data_type == "chat_logs":
        from shared import fetch_chat_logs
        limit = params.get("limit", 20)
        hours_back = params.get("hours_back")
        keyword = params.get("keyword")
        rows = await asyncio.to_thread(fetch_chat_logs, limit, hours_back, keyword)
        if not rows:
            return "No chat logs found."
        lines = []
        for r in rows:
            date = str(r.get("created_at", ""))[:19]
            q = r.get("user_query", "")[:200]
            a = r.get("bot_answer", "")[:300]
            lines.append(f"[{date}] Q: {q}\nA: {a}\n")
        return f"{len(rows)} chat log(s):\n\n" + "\n".join(lines)

    elif data_type == "task_reports":
        from shared import fetch_task_reports
        limit = params.get("limit", 10)
        status = params.get("status")
        rows = await asyncio.to_thread(fetch_task_reports, limit, status)
        if not rows:
            return "No task reports found."
        lines = []
        for r in rows:
            date = str(r.get("created_at", ""))[:19]
            st = r.get("status", "?")
            content = r.get("content", "")[:100]
            result_text = (r.get("result") or "")[:300]
            lines.append(f"  #{r.get('id', '?')} [{st}] {content}")
            if result_text:
                lines.append(f"    Result: {result_text}")
        return f"{len(rows)} task report(s):\n" + "\n".join(lines)

    elif data_type == "kg_stats":
        from shared import fetch_kg_stats
        stats = await asyncio.to_thread(fetch_kg_stats)
        if not stats or "error" in stats:
            return f"KG stats error: {stats}"
        return json.dumps(stats, ensure_ascii=False, indent=2, default=str)

    elif data_type == "experience":
        query_text = params.get("query", "")
        k = params.get("k", 5)
        if not query_text:
            return "Error: 'query' param required for experience search."
        from shared import search_experiential_memory
        rows = await asyncio.to_thread(search_experiential_memory, query_text, k)
        if not rows:
            return "No relevant experiential memories found."
        lines = []
        for r in rows:
            sim = f"{r.get('similarity', 0):.0%}"
            cat = r.get("category", "?")
            ts = str(r.get("created_at", ""))[:10]
            lines.append(f"[{cat}|{ts}|sim={sim}] {r['content']}")
        return f"{len(rows)} experience(s):\n\n" + "\n\n".join(lines)

    return f"Unknown data_type: {data_type}"
