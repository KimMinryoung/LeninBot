"""Operator-only, bounded, read-only audit diagnostics. Never rewrite audit history."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from tool_gateway.results import ToolRejection

LIMITATIONS = (
    "Rows without execution_kind remain unknown; historical tests cannot be reliably separated. "
    "Audit delivery is best effort and spool replay may duplicate rows. Repeated calls alone do not prove a retry."
)


def _operator(profile):
    from mcp_gateway.policy import normalize_profile
    if normalize_profile(profile) != "operator":
        raise ToolRejection("This tool requires the MCP operator profile")


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ToolRejection(f"{name} must be an integer between {low} and {high}")
    return value


from ops.readonly_query import query as _query


async def tool_usage_report(days=7, tool_name=None, interface=None, agent_name=None, *, profile="inspect"):
    _operator(profile)
    _integer(days, "days", 1, 90)
    end = datetime.now(timezone.utc)
    start, previous = end - timedelta(days=days), end - timedelta(days=2 * days)
    rows = await asyncio.to_thread(_query, """
        WITH calls AS (
            SELECT tool_name, interface, agent_name, latency_ms, result_status,
                   CASE WHEN result_metadata->>'execution_kind' IN ('runtime','test')
                        THEN result_metadata->>'execution_kind' ELSE 'unknown' END AS execution_kind,
                   CASE WHEN ts >= %s THEN 'current' ELSE 'previous' END AS period
              FROM tool_audit_log
             WHERE ts >= %s AND ts < %s
               AND (%s IS NULL OR tool_name = %s)
               AND (%s IS NULL OR interface = %s)
               AND (%s IS NULL OR agent_name = %s)
        ), totals AS (
            SELECT tool_name, interface, agent_name, execution_kind, period,
                   count(*) AS calls,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS latency_p50_ms,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS latency_p95_ms
              FROM calls GROUP BY tool_name, interface, agent_name, execution_kind, period
        ), statuses AS (
            SELECT tool_name, interface, agent_name, execution_kind, period,
                   jsonb_object_agg(status, n) AS result_statuses
              FROM (SELECT tool_name, interface, agent_name, execution_kind, period,
                           COALESCE(result_status, 'unknown') AS status, count(*) AS n
                      FROM calls GROUP BY tool_name, interface, agent_name, execution_kind, period, result_status) s
             GROUP BY tool_name, interface, agent_name, execution_kind, period
        ) SELECT t.*, s.result_statuses FROM totals t JOIN statuses s
            ON t.tool_name = s.tool_name AND t.interface IS NOT DISTINCT FROM s.interface
           AND t.agent_name IS NOT DISTINCT FROM s.agent_name
           AND t.execution_kind = s.execution_kind AND t.period = s.period
           ORDER BY t.execution_kind, t.tool_name, t.interface, t.agent_name, t.period
        """, (start, previous, end, tool_name, tool_name, interface, interface, agent_name, agent_name))
    groups = {}
    for row in rows:
        key = tuple(row.get(k) for k in ("tool_name", "interface", "agent_name", "execution_kind"))
        group = groups.setdefault(key, dict(zip(("tool_name", "interface", "agent_name", "execution_kind"), key)))
        group[row["period"]] = {k: row[k] for k in ("calls", "latency_p50_ms", "latency_p95_ms", "result_statuses")}
    partitions = {"runtime": [], "test": [], "unknown": []}
    for group in groups.values():
        for period in ("current", "previous"):
            group.setdefault(period, {"calls": 0, "latency_p50_ms": None, "latency_p95_ms": None, "result_statuses": {}})
        current, prior = group["current"]["calls"], group["previous"]["calls"]
        group["calls_delta"] = current - prior
        group["calls_change_pct"] = round((current - prior) * 100 / prior, 2) if prior else None
        group["latency_delta_ms"] = {
            key: group["current"][key] - group["previous"][key]
            if group["current"][key] is not None and group["previous"][key] is not None else None
            for key in ("latency_p50_ms", "latency_p95_ms")}
        current_status, previous_status = group["current"]["result_statuses"], group["previous"]["result_statuses"]
        group["result_status_delta"] = {key: current_status.get(key, 0) - previous_status.get(key, 0)
                                        for key in sorted(current_status.keys() | previous_status.keys())}
        partitions[group["execution_kind"]].append(group)
    return json.dumps({"start": start, "end": end, "previous_start": previous,
                       "groups": partitions, "limitations": LIMITATIONS}, default=str, ensure_ascii=False)


async def trace_tool_run(request_id=None, task_id=None, limit=200, offset=0, *, profile="inspect"):
    _operator(profile)
    if (request_id is None) == (task_id is None):
        raise ToolRejection("Supply exactly one of request_id or task_id")
    if request_id is not None and (not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 200):
        raise ToolRejection("request_id must be a nonempty identifier of at most 200 characters")
    if task_id is not None:
        _integer(task_id, "task_id", 1, 2**63 - 1)
    _integer(limit, "limit", 1, 200)
    _integer(offset, "offset", 0, 100000)
    # UNION deduplicates IDs and terminates even for malformed cyclic parent links.
    # Arguments/error text are deliberately not selected. Only a bounded redacted
    # argument fingerprint is used internally to identify possible repeat calls.
    rows = await asyncio.to_thread(_query, """
        WITH RECURSIVE runs(request_id) AS (
            SELECT %s::text WHERE %s IS NOT NULL
            UNION
            SELECT request_id FROM tool_audit_log WHERE %s IS NOT NULL AND task_id = %s
            UNION
            SELECT a.request_id FROM tool_audit_log a JOIN runs r ON a.parent_request_id = r.request_id
             WHERE a.request_id IS NOT NULL
        ), calls AS (
            SELECT a.id, a.ts, a.tool_name, a.interface, a.agent_name, a.request_id,
                   a.parent_request_id, a.task_id, a.result_status, a.decision, a.latency_ms,
                   COALESCE(a.result_metadata->>'execution_kind', 'unknown') AS execution_kind,
                   row_number() OVER (PARTITION BY a.request_id, a.tool_name, md5(COALESCE(a.args_summary,''))
                                      ORDER BY a.ts, a.id) AS same_call_ordinal
              FROM tool_audit_log a
             WHERE a.request_id IN (SELECT request_id FROM runs)
                OR (%s IS NOT NULL AND a.task_id = %s)
        ) SELECT * FROM calls ORDER BY ts, id LIMIT %s OFFSET %s
        """, (request_id, request_id, str(task_id) if task_id is not None else None,
              str(task_id) if task_id is not None else None, task_id, str(task_id), limit + 1, offset))
    more = len(rows) > limit
    rows = rows[:limit]
    task_ids = sorted({int(r['task_id']) for r in rows if str(r.get('task_id') or '').isdigit()})
    if task_id is not None and task_id not in task_ids:
        task_ids.append(task_id)
    tasks = await asyncio.to_thread(_query, """
        SELECT id, parent_task_id, status, agent_type, created_at, completed_at
          FROM telegram_tasks WHERE id = ANY(%s) ORDER BY id
        """, (task_ids,)) if task_ids else []
    for row in rows:
        row['retry_evidence'] = ('replay_suppressed' if row['result_status'] in {'deduplicated', 'deduplicated_durable'}
                                 else 'possible_repeat' if row['same_call_ordinal'] > 1 else 'none')
    return json.dumps({"calls": rows, "tasks": tasks, "next_offset": offset + limit if more else None,
                       "task_status_scope": "current persisted state of tasks represented on this page; missing is unknown",
                       "limitations": LIMITATIONS + " Parent IDs are retained; descendants are included. Audit order does not imply serial execution."},
                      default=str, ensure_ascii=False)


AUDIT_TOOLS = [
    {"name": "tool_usage_report", "description": "Operator-only read-only tool usage, statuses, p50/p95 and previous-period comparison. Tests and unclassified history are separate.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {
         "days": {"type": "integer", "minimum": 1, "maximum": 90, "default": 7},
         "tool_name": {"type": "string"}, "interface": {"type": "string"}, "agent_name": {"type": "string"}}}},
    {"name": "trace_tool_run", "description": "Operator-only audit sequence and descendant runs by request_id or task_id. At most 200 calls per page; excludes arguments. Repeats are hints, not proven retries.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {
         "request_id": {"type": "string", "minLength": 1, "maxLength": 200},
         "task_id": {"type": "integer", "minimum": 1},
         "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 200},
         "offset": {"type": "integer", "minimum": 0, "maximum": 100000, "default": 0}},
         "oneOf": [{"required": ["request_id"], "not": {"required": ["task_id"]}},
                   {"required": ["task_id"], "not": {"required": ["request_id"]}}]}}
]
AUDIT_HANDLERS = {"tool_usage_report": tool_usage_report, "trace_tool_run": trace_tool_run}
