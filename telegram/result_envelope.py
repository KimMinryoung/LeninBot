"""Structured result envelope for delegated agent tasks.

A delegated agent's report used to be free text only. The orchestrator, the
synthesis task and DAG-dependent subtasks had to read status, evidence and
gaps out of prose. ``report_result`` lets the agent state them as fields
before its final answer:

* ``status``: done | partial | blocked | failed (the agent's own judgment of
  goal completion, distinct from the runtime's execution status and from the
  independent verifier's verdict)
* ``answer``: the substantive result in a few sentences
* ``findings``: claims, each with the sources that support it
* ``open_questions``, ``blockers``, ``artifacts``

The envelope is stored in ``telegram_tasks.metadata['result_envelope']`` (JSONB,
no migration) and rendered next to the free-text report wherever task outcomes
are consumed: the orchestrator callback, subtask synthesis and dependency
results. It is labelled as the agent's claims. When an agent ends without
calling the tool, consumers see the free-text report exactly as before.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

STATUSES = ("done", "partial", "blocked", "failed")
MAX_ANSWER = 4000
MAX_FINDINGS = 20
MAX_CLAIM = 600
MAX_SOURCES = 5
MAX_SOURCE = 300
MAX_LIST = 10
MAX_ITEM = 500
METADATA_KEY = "result_envelope"

TOOL = {
    "name": "report_result",
    "description": (
        "Record the structured outcome of this task, once, right before your final answer "
        "(call again only to correct it; the last call wins). status is your own judgment of "
        "goal completion against the task and its success criteria: done = every requirement "
        "met; partial = some met, list what is missing in open_questions; blocked = an external "
        "obstacle stopped you (permission, CAPTCHA, missing input), name it in blockers; "
        "failed = it could not be done. Each finding is one claim with the sources that "
        "support it: URL, document slug/id, tool name with its key argument, or task id. A "
        "claim without a source you actually read must say so in the claim. This does not "
        "replace your final answer; still write the full report after calling it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": list(STATUSES)},
            "answer": {
                "type": "string",
                "description": "The substantive result in a few sentences: what the requester needs to know.",
            },
            "findings": {
                "type": "array",
                "maxItems": MAX_FINDINGS,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": MAX_SOURCES,
                            "description": "Where you verified it: URL, slug/id, tool(key arg), task id.",
                        },
                    },
                    "required": ["claim"],
                },
            },
            "open_questions": {
                "type": "array", "items": {"type": "string"}, "maxItems": MAX_LIST,
                "description": "Requirements not met or questions left unresolved.",
            },
            "blockers": {
                "type": "array", "items": {"type": "string"}, "maxItems": MAX_LIST,
                "description": "External obstacles. Required for blocked and failed.",
            },
            "artifacts": {
                "type": "array", "items": {"type": "string"}, "maxItems": MAX_LIST,
                "description": "Things you created or changed: URL, document slug/id, post id.",
            },
        },
        "required": ["status", "answer"],
    },
}

GUIDANCE = (
    "\n\nResult reporting: before your final answer, call report_result once with your "
    "status (done/partial/blocked/failed), a short answer, sourced findings, open questions, "
    "blockers and created artifacts. Then write the full report as usual."
)


def _clip(value, limit: int, *, collapse: bool = False) -> str:
    text = str(value or "").strip()
    if collapse:
        text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _str_list(values, limit_items: int, limit_chars: int) -> list[str]:
    if not isinstance(values, list):
        return []
    out = [_clip(v, limit_chars, collapse=True) for v in values if str(v or "").strip()]
    return out[:limit_items]


def normalize(status, answer, findings=None, open_questions=None, blockers=None, artifacts=None) -> dict:
    """Validate and bound tool arguments. Raises ValueError on a contract breach."""
    status = str(status or "").strip().lower()
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    answer = _clip(answer, MAX_ANSWER)
    if not answer:
        raise ValueError("answer must not be empty")
    clean_findings = []
    for item in findings or []:
        if isinstance(item, str):
            item = {"claim": item}
        if not isinstance(item, dict) or not str(item.get("claim") or "").strip():
            continue
        clean_findings.append({
            "claim": _clip(item["claim"], MAX_CLAIM),
            "sources": _str_list(item.get("sources"), MAX_SOURCES, MAX_SOURCE),
        })
    envelope = {
        "status": status,
        "answer": answer,
        "findings": clean_findings[:MAX_FINDINGS],
        "open_questions": _str_list(open_questions, MAX_LIST, MAX_ITEM),
        "blockers": _str_list(blockers, MAX_LIST, MAX_ITEM),
        "artifacts": _str_list(artifacts, MAX_LIST, MAX_ITEM),
    }
    if status in ("blocked", "failed") and not envelope["blockers"]:
        raise ValueError(f"status={status} requires at least one entry in blockers")
    if status == "partial" and not envelope["open_questions"]:
        raise ValueError("status=partial requires open_questions naming what is missing")
    return envelope


def save(task_id: int, envelope: dict) -> None:
    from db import execute
    execute(
        "UPDATE telegram_tasks SET metadata = COALESCE(metadata, '{}'::jsonb) "
        "|| jsonb_build_object(%s::text, %s::jsonb) WHERE id = %s",
        (METADATA_KEY, json.dumps(envelope, ensure_ascii=False), task_id),
    )


def make_handler(task_id: int):
    async def _exec_report_result(status, answer, findings=None, open_questions=None,
                                  blockers=None, artifacts=None) -> str:
        import asyncio
        from tool_gateway.results import ToolFailure, ToolRejection

        try:
            envelope = normalize(status, answer, findings, open_questions, blockers, artifacts)
        except ValueError as exc:
            raise ToolRejection(f"report_result rejected: {exc}. Correct the fields and call again.")
        try:
            await asyncio.to_thread(save, task_id, envelope)
        except Exception as exc:
            logger.error("report_result save failed (task %d): %s", task_id, exc)
            return ToolFailure(f"Result envelope not saved: {exc}. Put the same content in your final report.")
        return (
            f"Result envelope recorded for task #{task_id} (status={envelope['status']}, "
            f"{len(envelope['findings'])} finding(s)). Now write your full final report."
        )
    return _exec_report_result


def load(row_or_metadata) -> dict | None:
    """Envelope from a task row (with ``metadata``) or a metadata dict/JSON string."""
    metadata = row_or_metadata
    if isinstance(row_or_metadata, dict) and "metadata" in row_or_metadata:
        metadata = row_or_metadata.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, ValueError):
            return None
    if not isinstance(metadata, dict):
        return None
    envelope = metadata.get(METADATA_KEY)
    if not isinstance(envelope, dict) or envelope.get("status") not in STATUSES:
        return None
    return envelope


def render(envelope: dict | None, *, execution_status: str | None = None,
           verification_status: str | None = None) -> str:
    """Plain-text block for prompts. Empty string when there is no envelope."""
    if not envelope:
        return ""
    lines = [
        "[Agent result envelope — the agent's own structured claims, not independent verification]",
        f"agent-reported status: {envelope['status']}",
    ]
    if execution_status == "done" and envelope["status"] != "done":
        lines.append(
            f"note: execution ended normally but the agent itself reports {envelope['status']}; "
            "treat the goal as not fully met."
        )
    if verification_status == "failed" and envelope["status"] == "done":
        lines.append("note: the agent reports done but independent verification failed.")
    lines.append(f"answer: {envelope['answer']}")
    if envelope.get("findings"):
        lines.append("findings:")
        for i, finding in enumerate(envelope["findings"], 1):
            sources = "; ".join(finding.get("sources") or []) or "no source given"
            lines.append(f"  {i}. {finding['claim']} [sources: {sources}]")
    for key, label in (("open_questions", "open questions"), ("blockers", "blockers"),
                       ("artifacts", "artifacts")):
        if envelope.get(key):
            lines.append(f"{label}:")
            lines.extend(f"  - {item}" for item in envelope[key])
    return "\n".join(lines)


def attach_rendered(rows: list[dict]) -> list[dict]:
    """Add ``result_envelope_text`` to task rows for llm.prompt_context previews."""
    for row in rows or []:
        text = render(load(row), execution_status=row.get("status"),
                      verification_status=row.get("verification_status"))
        if text:
            row["result_envelope_text"] = text
    return rows
