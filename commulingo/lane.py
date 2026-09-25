"""Shared plumbing for the CommuLingo curator lanes.

The people maintainer (and its new/enrich wrappers), the glossary lane and the
gap worker run the curator the same way: one maintainer config, a per-lane
approved-edit counter, narrow write tools whose structured errors become
retryable tool failures, the typed no-edit terminal, and one stage runner.
They live here so the lane scripts import a library instead of each other.

This module does not import commulingo.people or the tool
registry. That module reads the lane name from COMMULINGO_SUGGESTED_BY once, at
import time, so importing this one never fixes the lane name early.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from bot_config import resolve_agent_tool_loop
from db import query_one as db_query_one
from commulingo.research_memory import ResearchMemory
from tool_gateway.results import ToolRejection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "commulingo_maintainer.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    defaults = {
        "enabled": True,
        "mode": "auto",
        "new_person_every": 8,
        "recent_days": 30,
        # Cards with basic gaps (empty bio/epithet/moment, no career, role,
        # citizenship, event link, or section) age back in on this much
        # shorter cooldown so one-step-per-run enrichment can actually finish
        # a card; the long recent_days cooldown only throttles complete cards,
        # where a forced re-pick would just accrete filler edits.
        "incomplete_recent_days": 2,
        "new_person_cooldown_runs": 6,
        # A card the curator could not enrich steps aside for this many enrich
        # runs. Only an applied edit writes a revision, so the DB cooldown above
        # never fires for a failed run: 미하일 코즐롭스키 had no findable sources,
        # and the hourly lane re-picked him three hours straight, burning nine
        # 16-round attempts (most of that day's spend) before one finally landed.
        "enrich_failure_cooldown_runs": 6,
        # Parallel-lane switch: when false, the dedicated new-person lane
        # (COMMULINGO_SUGGESTED_BY=commulingo-maintainer-new) no-ops so all
        # maintenance effort concentrates on enriching existing cards.
        "new_lane_enabled": True,
        # Same switch for the glossary lane (commulingo_terms_maintainer.py).
        "term_lane_enabled": True,
        # Pause this semantic category without narrowing all other enrichment.
        "enrich_non_soviet_revolutionaries": True,
        # How many enrich edits one `--mode enrich` invocation (one batch slot)
        # lands. The batch gave enrichment one slot against the gap worker's
        # one create, and the gap worker's ~25 cards a day then absorbed every
        # slot (2026-08-29..09-01: 76 of 76 enriched cards were under 30 days
        # old). More slots per batch is the lever; the daily cap still binds.
        "enrich_runs_per_batch": 2,
        # Extra enrich runs when the person gap queue is empty, i.e. the batch
        # slot the gap worker would have used goes to existing cards instead.
        "enrich_extra_runs_when_gap_empty": 1,
        # Stop starting further runs once this much wall-clock has gone, so a
        # multi-run invocation stays inside the unit's TimeoutStartSec and the
        # 20-minute batch spacing: the unit allows 900s, a retry-heavy run has
        # taken 10 minutes, so no run starts past the seventh minute.
        "enrich_batch_time_budget_sec": 420,
        # Every Nth enrich run (by the lane's applied-edit count) puts cards no
        # curator lane has touched for stale_priority_days at the head of the
        # queue, ahead of the standard-field gaps that newly created cards
        # arrive with. Without it the gap worker's fresh cards, which land
        # missing a `moment`, outrank every older card forever. 0 disables.
        "stale_priority_every": 2,
        "stale_priority_days": 30,
        # New-person discovery is independently focusable.
        "new_person_focus": "all",
        # Era groups the absence roster covers. Empty derives it from the focus
        # via ROSTER_GROUPS_BY_FOCUS; an explicit list overrides that, which is
        # the lever for widening the roster back out if duplicates climb.
        "roster_groups": [],
    }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return defaults
    if not isinstance(raw, dict):
        raise ValueError("commulingo maintainer config must be an object")
    cfg = {**defaults, **raw}
    if cfg["mode"] not in {"auto", "enrich", "new"}:
        raise ValueError("mode must be auto, enrich, or new")
    cfg["new_person_every"] = max(0, int(cfg["new_person_every"]))
    cfg["recent_days"] = max(1, int(cfg["recent_days"]))
    cfg["incomplete_recent_days"] = max(1, int(cfg["incomplete_recent_days"]))
    cfg["new_person_cooldown_runs"] = max(0, int(cfg["new_person_cooldown_runs"]))
    cfg["enrich_failure_cooldown_runs"] = max(0, int(cfg["enrich_failure_cooldown_runs"]))
    cfg["new_lane_enabled"] = bool(cfg["new_lane_enabled"])
    cfg["term_lane_enabled"] = bool(cfg["term_lane_enabled"])
    cfg["enrich_non_soviet_revolutionaries"] = bool(cfg["enrich_non_soviet_revolutionaries"])
    cfg["enrich_runs_per_batch"] = max(1, int(cfg["enrich_runs_per_batch"]))
    cfg["enrich_extra_runs_when_gap_empty"] = max(0, int(cfg["enrich_extra_runs_when_gap_empty"]))
    cfg["enrich_batch_time_budget_sec"] = max(0, int(cfg["enrich_batch_time_budget_sec"]))
    cfg["stale_priority_every"] = max(0, int(cfg["stale_priority_every"]))
    cfg["stale_priority_days"] = max(1, int(cfg["stale_priority_days"]))
    if cfg["new_person_focus"] not in {"all", "soviet_institutions", "old_regime", "china"}:
        raise ValueError("new_person_focus must be all, soviet_institutions, old_regime, or china")
    if not isinstance(cfg["roster_groups"], list):
        raise ValueError("roster_groups must be a list of group ids")
    cfg["roster_groups"] = [str(g) for g in cfg["roster_groups"]]
    return cfg


def completed_run_count() -> int:
    row = db_query_one(
        """SELECT COUNT(*)::int AS n
             FROM commulingo_agent_suggestions
            WHERE suggested_by = %(s)s
              AND status = 'approved'""",
        # Read at call time so the count follows the lane even if this module
        # was imported before a wrapper set the variable.
        {"s": os.environ["COMMULINGO_SUGGESTED_BY"]},
    )
    return int((row or {}).get("n") or 0)


COMMULINGO_NO_EDIT_TOOL = {
    "name": "commulingo_no_edit",
    "description": (
        "Finish the enrich run by declaring that the commissioned step honestly needs no "
        "write — the card already satisfies it, or no supportable content exists (e.g. no "
        "listed event genuinely fits this person). Only call this after completing the "
        "commissioned reads and research; it is a judged conclusion, not a shortcut."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["complete", "not_applicable", "sources_unavailable"]},
            "sources": {"type": "array", "items": {"type": "string"}, "description": "References inspected, including unsuccessful evidence checks."},
            "reason": {
                "type": "string",
                "description": "One or two sentences: what was checked and why no write is warranted.",
            },
        },
        "required": ["reason", "status", "sources"],
    },
}


def build_no_edit_handler(box: dict):
    async def _no_edit(reason: str = "", status: str = "sources_unavailable", sources: list | None = None) -> str:
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("reason is required")
        if status not in {"complete", "not_applicable", "sources_unavailable"}:
            raise ValueError("invalid completion status")
        box.update(reason=reason, status=status, sources=sources or [])
        return json.dumps({"ok": True}, ensure_ascii=False)
    return _no_edit


NARROW_WRITE_TOOLS = frozenset({
    "commulingo_person_create", "commulingo_person_update",
    "commulingo_section_save", "commulingo_event_link", "commulingo_term_create",
    "commulingo_event_update", "commulingo_event_section_save",
})


def build_retrying_write_handler(handler):
    """Turn structured write errors into tool failures so the model can retry."""
    async def _validated_write(**kwargs) -> str:
        result = await handler(**kwargs)
        if not str(result).startswith("Error:"):
            return result
        raw = str(result).removeprefix("Error:").strip()
        try:
            payload = json.loads(raw)
            error = payload.get("error") or {}
            code = str(error.get("code") or "unknown")
            retryable = bool(error.get("retryable", True))
            message = str(error.get("message") or raw)
        except (json.JSONDecodeError, AttributeError):
            code, retryable, message = "legacy_error", True, raw
        raise ToolRejection(
            f"commulingo_write[{code}; retryable={str(retryable).lower()}]: {message}"
        )
    return _validated_write


async def _call_curator_stage(
    *, task: str, spec, tools: list, handlers: dict,
    policy, stage: str, expect_edit: bool,
    finalization_tools: list[str], terminal_tools: list[str],
    candidate_box: dict | None = None, no_edit_box: dict | None = None,
    research_key: str | None = None, baseline: dict | None = None, run_budget=None,
) -> tuple[str, dict, dict | None]:
    import asyncio
    from commulingo.run import RunBudget, RunFailure
    from tool_gateway.results import ToolFailure
    binding = resolve_agent_tool_loop(spec, policy)
    memory = ResearchMemory(research_key or f"{spec.name}:{stage}:{task}")
    memory.baseline = baseline
    run = run_budget or RunBudget(policy, memory.path, stage, research_key or stage)
    run.stage, run.target = stage, research_key or stage
    outcomes = []
    terminal_recorded = False

    def capture(handler, name):
        async def captured(**kwargs):
            nonlocal terminal_recorded
            if terminal_recorded:
                raise ToolRejection("the commissioned terminal action already succeeded; stop this run")
            result = await handler(**kwargs)
            if not isinstance(result, ToolFailure) and not str(result).startswith("Error:"):
                terminal_recorded = name in terminal_tools
                if name in NARROW_WRITE_TOOLS:
                    outcomes.append({"tool": name, "result": str(result),
                        "target": kwargs.get("person_id") or kwargs.get("event_id") or kwargs.get("term_id")})
                    run.record("submitted", writes=outcomes, metrics=memory.metrics)
            return result
        return captured

    handlers = {name: capture(handler, name) if name in NARROW_WRITE_TOOLS or name in terminal_tools else handler
                for name, handler in handlers.items()}
    # A fresh attempt is distinct from output continuation and shares the job envelope.
    last_result, last_error = "", None
    for attempt in range(2):
        tracker = {}
        seconds, rounds, cost = run.remaining()
        continuations = min(policy.max_output_continuations, max(0, rounds - 1))
        try:
            last_result = await asyncio.wait_for(memory.chat(binding.chat,
                [{"role": "user", "content": task + ("\nRepair the saved draft using the existing evidence; finish with the commissioned tool." if attempt else "")}],
                client=binding.client, model=binding.model, tools=tools, tool_handlers=handlers,
                system_prompt=spec.render_prompt(provider=binding.render_provider)
                    + ("\nDISCOVERY: finish only with commulingo_candidate_select." if not expect_edit else ""),
                max_rounds=max(1, rounds - continuations), max_tokens=policy.max_output_tokens,
                max_input_tokens=policy.max_input_tokens, recover_input_via_tools=True,
                continue_on_length=continuations > 0,
                max_length_continuations=continuations,
                budget_usd=cost, budget_tracker=tracker, agent_name=spec.name,
                finalization_tools=finalization_tools, terminal_tools=terminal_tools,
                **binding.reasoning), timeout=seconds)
            tracker['pipeline_call_complete'] = True
        except Exception as exc:
            last_error = exc
        finally:
            run.account(tracker)
            run.record("running", metrics=memory.metrics, writes=outcomes)
        writes = [item for item in outcomes if item["tool"] in terminal_tools]
        if writes:
            status = "pending_review" if writes[-1]["result"].startswith("OK — pending:") else "applied"
            return writes[-1]["result"], run.record(status, metrics=memory.metrics, writes=outcomes), None
        if no_edit_box and no_edit_box.get("reason"):
            status = no_edit_box.get("status", "complete")
            return last_result, run.record(status, metrics=memory.metrics, writes=outcomes), None
        if not expect_edit and (candidate_box or {}).get("candidate"):
            return last_result, run.record("selected", metrics=memory.metrics), candidate_box["candidate"]
        # A handler exception after a side effect must never replay the task.
        if outcomes or last_error is not None:
            break
    summary = run.record("error" if last_error else "exhausted", metrics=memory.metrics, writes=outcomes,
                         error=str(last_error or "no terminal result")[:500])
    raise RunFailure(f"{stage} ended without its terminal result: {last_error or last_result[:500]}", summary)
