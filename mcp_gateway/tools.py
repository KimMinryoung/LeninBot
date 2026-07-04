"""Read-only MCP Gateway tools and adapters."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable

from mcp_gateway.policy import allowed_tool_names
from tool_gateway.selection import build_toolset

ROOT = Path(__file__).resolve().parents[1]
DEV_DOCS_DIR = ROOT / "dev_docs"

ToolHandler = Callable[..., Awaitable[str]]


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


GATEWAY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "gateway_status",
        "description": "Return MCP Gateway profile and exposed tool count.",
        "input_schema": _schema({}),
    },
    {
        "name": "list_mcp_tools",
        "description": "List tools exposed by this MCP Gateway profile.",
        "input_schema": _schema({}),
    },
    {
        "name": "search_dev_docs",
        "description": "Search current developer documentation under dev_docs/ without reading legacy handoff notes.",
        "input_schema": _schema(
            {
                "query": {"type": "string", "description": "Case-insensitive text to search for."},
                "limit": {"type": "integer", "description": "Maximum matches to return. Default 20, max 50."},
            },
            ["query"],
        ),
    },
    {
        "name": "get_project_runtime_summary",
        "description": "Return a concise runtime summary from dev_docs/project_state.md.",
        "input_schema": _schema({}),
    },
    {
        "name": "list_recent_tasks",
        "description": "List recent telegram_tasks rows with status, agent, priority, timestamps, and short result preview.",
        "input_schema": _schema(
            {
                "limit": {"type": "integer", "description": "Maximum rows. Default 10, max 50."},
                "status": {"type": "string", "description": "Optional exact task status filter."},
                "agent_type": {"type": "string", "description": "Optional exact agent_type filter."},
            }
        ),
    },
    {
        "name": "get_task_status",
        "description": "Read one telegram_tasks row by id, including content/result previews and metadata.",
        "input_schema": _schema(
            {
                "task_id": {"type": "integer", "description": "telegram_tasks.id"},
                "include_result": {
                    "type": "boolean",
                    "description": "Include full result text up to max_chars. Default false.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum field chars. Default 4000, max 20000.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset for the selected field. Default 0.",
                },
                "field": {
                    "type": "string",
                    "description": "Long text field to paginate: content, result, or tool_log. Default result when include_result=true, otherwise content.",
                },
            },
            ["task_id"],
        ),
    },
    {
        "name": "list_recent_task_reports",
        "description": "List recent completed non-programmer task reports without full markdown bodies.",
        "input_schema": _schema(
            {
                "limit": {"type": "integer", "description": "Maximum reports. Default 10, max 50."},
                "agent_type": {"type": "string", "description": "Optional exact agent_type filter."},
            }
        ),
    },
    {
        "name": "corpus_metadata_audit",
        "description": "Summarize lenin_corpus metadata quality for a layer and optional author/title filters.",
        "input_schema": _schema(
            {
                "layer": {
                    "type": "string",
                    "enum": ["core_theory", "modern_analysis", "self_produced_analysis"],
                    "description": "Corpus layer to audit.",
                },
                "author": {"type": "string", "description": "Optional exact metadata author filter."},
                "title": {"type": "string", "description": "Optional title substring filter."},
                "limit": {"type": "integer", "description": "Maximum grouped rows. Default 20, max 100."},
            },
            ["layer"],
        ),
    },
    {
        "name": "kg_integrity_check",
        "description": "Run the read-only KG integrity checker, optionally with a KG search smoke query.",
        "input_schema": _schema(
            {
                "smoke_query": {
                    "type": "string",
                    "description": "Optional KG search query to test end-to-end search health.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Subprocess timeout. Default 120, max 600.",
                },
            }
        ),
    },
    {
        "name": "readonly_query_db",
        "description": (
            "Operator profile only. Run one read-only SQL diagnostic through scripts/query-db "
            "(SELECT/WITH/SHOW/EXPLAIN only, read-only transaction)."
        ),
        "input_schema": _schema(
            {
                "sql": {"type": "string", "description": "Single read-only SQL statement."},
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Subprocess timeout. Default 30, max 120.",
                },
            },
            ["sql"],
        ),
    },
    {
        "name": "bounded_query_db",
        "description": (
            "Operator profile only. Run one SQL statement through the existing runtime_tools.db query_db guard. "
            "SELECT/WITH/SHOW/EXPLAIN return rows; INSERT/UPDATE/DELETE return affected row count; "
            "DROP/TRUNCATE are blocked; UPDATE/DELETE affecting >=10 rows are rolled back."
        ),
        "input_schema": _schema(
            {
                "sql": {
                    "type": "string",
                    "description": "A single SQL statement. Use %s placeholders with params for dynamic values.",
                },
                "params": {
                    "type": "array",
                    "items": {},
                    "description": "Optional positional parameters for %s placeholders.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Cap on SELECT rows in the response. Default 100, max 1000.",
                },
            },
            ["sql"],
        ),
    },
    {
        "name": "kg_maintenance_run",
        "description": (
            "Operator profile only. Run a bounded KG maintenance script. Mutating actions require "
            "execute=true and confirm='APPLY_KG_MAINTENANCE'; the wrapper runs a KG backup before mutation."
        ),
        "input_schema": _schema(
            {
                "action": {
                    "type": "string",
                    "enum": [
                        "backup",
                        "duplicate_candidates",
                        "merge_exact_name_dupes",
                        "cleanup_orphans",
                        "classify_untyped",
                        "full_cleanup",
                    ],
                    "description": "Maintenance action. Mutating actions default to dry-run unless execute=true.",
                },
                "execute": {
                    "type": "boolean",
                    "description": "Apply changes for mutating actions. Default false.",
                },
                "confirm": {
                    "type": "string",
                    "description": "Required as APPLY_KG_MAINTENANCE when execute=true.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Subprocess timeout. Default 300, max 1800.",
                },
            },
            ["action"],
        ),
    },
]


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=_json_default)


def _clip(text: str | None, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _slice_text(text: str | None, max_chars: int, offset: int = 0) -> tuple[str, int, int, bool]:
    text = text or ""
    try:
        start = max(0, int(offset or 0))
    except (TypeError, ValueError):
        start = 0
    if start >= len(text):
        return "", start, start, False
    end = min(len(text), start + max_chars)
    return text[start:end], start, end, end < len(text)


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(int(value), maximum))
    except (TypeError, ValueError):
        return default


async def gateway_status(*, profile: str = "inspect", **_: Any) -> str:
    names = allowed_tool_names(profile)
    return _json_text({
        "status": "ok",
        "profile": profile,
        "tool_count": len(names),
        "write_tools_exposed": False,
    })


async def list_mcp_tools(*, profile: str = "inspect", **_: Any) -> str:
    catalog = build_tool_catalog(profile)
    return _json_text([
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
        }
        for tool in catalog
    ])


async def search_dev_docs(query: str, limit: int = 20, **_: Any) -> str:
    needle = (query or "").strip().lower()
    if not needle:
        return "Error: query is required."
    limit = _bounded_int(limit, 20, 1, 50)
    matches: list[dict[str, Any]] = []
    for path in sorted(DEV_DOCS_DIR.glob("*.md")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(lines, 1):
            if needle not in line.lower():
                continue
            matches.append({
                "path": str(path.relative_to(ROOT)),
                "line": lineno,
                "text": line.strip()[:500],
            })
            if len(matches) >= limit:
                return _json_text(matches)
    return _json_text(matches)


async def get_project_runtime_summary(**_: Any) -> str:
    path = DEV_DOCS_DIR / "project_state.md"
    text = path.read_text(encoding="utf-8")
    sections = []
    keep = False
    for line in text.splitlines():
        if line.startswith("## Runtime Map") or line.startswith("## Service Units") or line.startswith("## Main Data Stores"):
            keep = True
        elif line.startswith("## ") and keep:
            keep = False
        if keep:
            sections.append(line)
    return "\n".join(sections).strip()


async def list_recent_tasks(limit: int = 10, status: str = "", agent_type: str = "", **_: Any) -> str:
    from db import query

    limit = _bounded_int(limit, 10, 1, 50)
    clauses = []
    params: list[Any] = []
    if status:
        clauses.append("status = %s")
        params.append(status)
    if agent_type:
        clauses.append("agent_type = %s")
        params.append(agent_type)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"""
        SELECT id, status, priority, agent_type, mission_id, parent_task_id,
               created_at, updated_at, available_at,
               left(content, 300) AS content_preview,
               left(result, 300) AS result_preview
        FROM telegram_tasks
        {where}
        ORDER BY id DESC
        LIMIT %s
    """
    params.append(limit)
    rows = await asyncio.to_thread(query, sql, tuple(params))
    return _json_text(rows)


async def get_task_status(
    task_id: int,
    include_result: bool = False,
    max_chars: int = 4000,
    offset: int = 0,
    field: str = "",
    **_: Any,
) -> str:
    from db import query

    max_chars = _bounded_int(max_chars, 4000, 500, 20000)
    offset = _bounded_int(offset, 0, 0, 10_000_000)
    selected_field = (field or ("result" if include_result else "content")).strip()
    if selected_field not in {"content", "result", "tool_log"}:
        selected_field = "result" if include_result else "content"
    rows = await asyncio.to_thread(
        query,
        """
        SELECT id, user_id, status, priority, agent_type, mission_id, parent_task_id,
               depth, plan_id, plan_role, created_at, updated_at, available_at,
               metadata, content, result, tool_log
        FROM telegram_tasks
        WHERE id = %s
        LIMIT 1
        """,
        (int(task_id),),
    )
    if not rows:
        return f"Task not found: {task_id}"
    row = rows[0]
    page, start, end, truncated = _slice_text(row.get(selected_field), max_chars, offset)
    total = len(row.get(selected_field) or "")

    row["content_preview"] = _clip(row.get("content"), 500)
    row["tool_log_preview"] = _clip(row.get("tool_log"), 500)
    row.pop("content", None)
    row.pop("tool_log", None)
    if include_result:
        row["result_preview"] = _clip(row.get("result"), 500)
    row.pop("result", None)

    row["selected_field"] = selected_field
    row["selected_field_chars"] = total
    row["returned_chars"] = [start, end]
    row["truncated"] = truncated
    if truncated:
        row["next"] = {
            "tool": "get_task_status",
            "task_id": int(task_id),
            "include_result": include_result,
            "field": selected_field,
            "offset": end,
            "max_chars": max_chars,
        }
    row[selected_field] = page
    return _json_text(row)


async def list_recent_task_reports(limit: int = 10, agent_type: str = "", **_: Any) -> str:
    from db import query

    limit = _bounded_int(limit, 10, 1, 50)
    clauses = ["status = 'done'", "result IS NOT NULL", "result != ''", "COALESCE(agent_type, '') != 'programmer'"]
    params: list[Any] = []
    if agent_type:
        clauses.append("agent_type = %s")
        params.append(agent_type)
    params.append(limit)
    rows = await asyncio.to_thread(
        query,
        f"""
        SELECT id, agent_type, mission_id, created_at, updated_at,
               left(content, 240) AS content_preview,
               left(result, 500) AS result_preview
        FROM telegram_tasks
        WHERE {' AND '.join(clauses)}
        ORDER BY id DESC
        LIMIT %s
        """,
        tuple(params),
    )
    return _json_text(rows)


async def corpus_metadata_audit(
    layer: str,
    author: str = "",
    title: str = "",
    limit: int = 20,
    **_: Any,
) -> str:
    from db import query

    limit = _bounded_int(limit, 20, 1, 100)
    clauses = ["metadata->>'layer' = %s"]
    params: list[Any] = [layer]
    if author:
        clauses.append("metadata->>'author' = %s")
        params.append(author)
    if title:
        clauses.append("metadata->>'title' ILIKE %s")
        params.append(f"%{title}%")
    where = " AND ".join(clauses)
    summary = await asyncio.to_thread(
        query,
        f"""
        SELECT
          COUNT(*) AS chunks,
          COUNT(*) FILTER (WHERE COALESCE(metadata->>'title', '') = '') AS missing_title,
          COUNT(*) FILTER (WHERE COALESCE(metadata->>'author', '') = '') AS missing_author,
          COUNT(*) FILTER (WHERE COALESCE(metadata->>'year', '') = '') AS missing_year,
          COUNT(*) FILTER (WHERE COALESCE(metadata->>'chunk_size', '') = '') AS missing_chunk_size,
          COUNT(*) FILTER (
            WHERE COALESCE(metadata->>'source_url', metadata->>'public_url', '') = ''
          ) AS missing_url
        FROM lenin_corpus
        WHERE {where}
        """,
        tuple(params),
    )
    grouped = await asyncio.to_thread(
        query,
        f"""
        SELECT
          COALESCE(metadata->>'author', 'Unknown') AS author,
          COALESCE(metadata->>'title', 'Untitled') AS title,
          COALESCE(metadata->>'year', '') AS year,
          COALESCE(metadata->>'language', '') AS language,
          COALESCE(metadata->>'chunk_size', '') AS chunk_size,
          COUNT(*) AS chunks
        FROM lenin_corpus
        WHERE {where}
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY chunks DESC, author, title
        LIMIT %s
        """,
        tuple([*params, limit]),
    )
    return _json_text({"summary": summary[0] if summary else {}, "groups": grouped})


def _run_project_command(command: list[str], timeout_seconds: int) -> str:
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    output = (proc.stdout or "").strip()
    error = (proc.stderr or "").strip()
    body = "\n".join(part for part in [output, error] if part)
    if len(body) > 30000:
        body = body[:30000] + "\n...[truncated at 30000 chars]"
    if proc.returncode == 0:
        return body or "(no output)"
    return f"Error: command exited {proc.returncode}\n{body}"


async def kg_integrity_check(smoke_query: str = "", timeout_seconds: int = 120, **_: Any) -> str:
    timeout_seconds = _bounded_int(timeout_seconds, 120, 10, 600)
    command = [str(ROOT / "venv" / "bin" / "python"), "scripts/check_kg_integrity.py"]
    if smoke_query:
        command.extend(["--smoke-query", smoke_query])
    try:
        return await asyncio.to_thread(_run_project_command, command, timeout_seconds)
    except subprocess.TimeoutExpired:
        return f"Error: KG integrity check timed out after {timeout_seconds}s"


def _kg_maintenance_command(action: str, execute: bool) -> list[str] | None:
    py = str(ROOT / "venv" / "bin" / "python")
    skill_scripts = ROOT / "skills" / "kg-maintenance" / "scripts"
    if action == "backup":
        return [py, str(skill_scripts / "backup_kg.py")]
    if action == "duplicate_candidates":
        return [py, str(skill_scripts / "dedup_entities.py")]
    if action == "merge_exact_name_dupes":
        cmd = [py, str(skill_scripts / "merge_exact_name_dupes.py")]
        if execute:
            cmd.append("--execute")
        return cmd
    if action == "cleanup_orphans":
        cmd = [py, str(skill_scripts / "cleanup_orphans.py")]
        if execute:
            cmd.append("--execute")
        return cmd
    if action == "classify_untyped":
        cmd = [py, "scripts/classify_untyped_entities.py"]
        if not execute:
            cmd.append("--dry-run")
        return cmd
    if action == "full_cleanup":
        cmd = [py, str(skill_scripts / "run_cleanup.py")]
        if execute:
            cmd.append("--execute")
        return cmd
    return None


async def kg_maintenance_run(
    action: str,
    execute: bool = False,
    confirm: str = "",
    timeout_seconds: int = 300,
    **_: Any,
) -> str:
    timeout_seconds = _bounded_int(timeout_seconds, 300, 10, 1800)
    action = (action or "").strip()
    command = _kg_maintenance_command(action, bool(execute))
    if command is None:
        return "Error: unknown KG maintenance action."

    mutating = action in {"merge_exact_name_dupes", "cleanup_orphans", "classify_untyped", "full_cleanup"}
    if execute and mutating and confirm != "APPLY_KG_MAINTENANCE":
        return "Error: execute=true requires confirm='APPLY_KG_MAINTENANCE'."

    try:
        sections: list[str] = []
        if execute and mutating and action != "full_cleanup":
            backup_cmd = _kg_maintenance_command("backup", False)
            assert backup_cmd is not None
            sections.append("## Pre-mutation KG backup\n" + await asyncio.to_thread(
                _run_project_command,
                backup_cmd,
                min(timeout_seconds, 600),
            ))
        sections.append(f"## KG maintenance: {action} ({'execute' if execute else 'dry-run'})\n" + await asyncio.to_thread(
            _run_project_command,
            command,
            timeout_seconds,
        ))
        return "\n\n".join(sections)
    except subprocess.TimeoutExpired:
        return f"Error: KG maintenance action '{action}' timed out after {timeout_seconds}s"


async def bounded_query_db(
    sql: str,
    params: list | None = None,
    max_rows: int = 100,
    **_: Any,
) -> str:
    from runtime_tools.db import DB_TOOL_HANDLERS

    handler = DB_TOOL_HANDLERS["query_db"]
    return await handler(sql=sql, params=params, max_rows=max_rows)


async def readonly_query_db(sql: str, timeout_seconds: int = 30, **_: Any) -> str:
    timeout_seconds = _bounded_int(timeout_seconds, 30, 1, 120)
    if not (sql or "").strip():
        return "Error: sql is required."

    def _run() -> str:
        proc = subprocess.run(
            [str(ROOT / "scripts" / "query-db"), sql],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = (proc.stdout or "").strip()
        error = (proc.stderr or "").strip()
        if proc.returncode == 0:
            return output or "(no output)"
        return f"Error: scripts/query-db exited {proc.returncode}\n{error or output}"

    try:
        return await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        return f"Error: query timed out after {timeout_seconds}s"


GATEWAY_HANDLERS: dict[str, ToolHandler] = {
    "gateway_status": gateway_status,
    "list_mcp_tools": list_mcp_tools,
    "search_dev_docs": search_dev_docs,
    "get_project_runtime_summary": get_project_runtime_summary,
    "list_recent_tasks": list_recent_tasks,
    "get_task_status": get_task_status,
    "list_recent_task_reports": list_recent_task_reports,
    "corpus_metadata_audit": corpus_metadata_audit,
    "kg_integrity_check": kg_integrity_check,
    "readonly_query_db": readonly_query_db,
    "bounded_query_db": bounded_query_db,
    "kg_maintenance_run": kg_maintenance_run,
}


def _mcp_tool(tool: dict[str, Any]) -> dict[str, Any]:
    converted = dict(tool)
    converted["inputSchema"] = converted.pop("input_schema", {"type": "object", "properties": {}})
    return converted


def build_tool_catalog(profile: str = "inspect") -> list[dict[str, Any]]:
    from runtime_tools.registry import TOOL_HANDLERS, TOOLS

    allowed = allowed_tool_names(profile)
    runtime_tools, _runtime_handlers = build_toolset(TOOLS, TOOL_HANDLERS, allowed)
    runtime = [_mcp_tool(tool) for tool in runtime_tools]
    gateway = [_mcp_tool(tool) for tool in GATEWAY_TOOLS if str(tool.get("name") or "") in allowed]
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for tool in [*gateway, *runtime]:
        name = str(tool.get("name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        merged.append(tool)
    return merged


def build_handlers(profile: str = "inspect") -> dict[str, ToolHandler]:
    from runtime_tools.registry import TOOL_HANDLERS

    allowed = allowed_tool_names(profile)
    handlers: dict[str, ToolHandler] = build_toolset([], TOOL_HANDLERS, allowed)[1]
    handlers.update({
        name: handler
        for name, handler in GATEWAY_HANDLERS.items()
        if name in allowed
    })
    return handlers

