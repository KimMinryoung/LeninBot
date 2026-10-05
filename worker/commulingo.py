"""CommuLingo agent sessions as worker task kinds (frontend
dev_docs/commulingo-agent-pipeline.md, decision 2026-10-05 option 1).

The frontend owns the enrichment queue, planning, validation and publication.
It hands one stage of one job to the worker: `commulingo_editor` (research and
draft, one author session), `commulingo_review` (independent review) or
`commulingo_discover` (which requested entries are missing). The
session code is the pipeline's Editor/Review, unchanged; this module only
replaces the queue store they used with the task's own input and output:

- input: the job row and its artifacts as the frontend stored them;
- output: the stage Result (value, nextStage, status, delaySeconds), the
  checkpoint artifacts the session saved (editor_checkpoint, fetch_failures)
  so the frontend can store them and pass them back on the next attempt, and
  review notes the frontend writes itself (the worker never writes CommuLingo).

Fetched pages stay in leninbot's source cache (commulingo_pipeline_sources,
fetch_cache, job_sources), which is research infrastructure.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

KINDS = {"commulingo_editor": ("research", "draft"), "commulingo_review": ("review",), "commulingo_discover": ("discover",)}
JOB_KEYS = ("id", "kind", "action", "target", "topic", "stage", "payload", "attempts", "baseline", "reason", "priority")


def validate_request(request: dict) -> dict:
    kind = request.get("kind")
    payload = request.get("input")
    if not isinstance(payload, dict) or not isinstance(payload.get("job"), dict) or not isinstance(payload.get("artifacts"), list):
        raise ValueError("input must be {job, artifacts}")
    job = payload["job"]
    if job.get("stage") not in KINDS[kind] or job.get("kind") not in ("person", "term"):
        raise ValueError(f"{kind} handles stages {KINDS[kind]} of person/term jobs")
    if not all(isinstance(a, dict) and isinstance(a.get("stage"), str) and isinstance(a.get("value"), dict) for a in payload["artifacts"]):
        raise ValueError("artifacts must be [{stage, value}]")
    budget = request.get("budgetUsd", 0.2)
    if not isinstance(budget, (int, float)) or not 0 < budget <= 0.6:
        raise ValueError("budgetUsd must be > 0 and <= 0.6")
    settings = payload.get("settings") or {}
    if not isinstance(settings, dict) or set(settings) - {"term_event_overlap_allow"}:
        raise ValueError("settings may carry term_event_overlap_allow only")
    return {"kind": kind, "budgetUsd": float(budget),
            "input": {"job": {k: job.get(k) for k in JOB_KEYS}, "settings": settings, "artifacts": [
                {"stage": a["stage"], "value": a["value"]} for a in payload["artifacts"]]}}


class TaskStore:
    """The queue-store surface Editor/Review use, bound to one worker task."""

    def __init__(self):
        from commulingo.pipeline.store import Store
        self._sources = Store()
        self.captured: dict[str, dict] = {}

    # Source cache: leninbot's research infrastructure.
    def job_sources(self, job_id):
        return self._sources.job_sources(job_id)

    def save_source(self, page):
        return self._sources.save_source(page)

    def cache_source(self, tool, args, source_id):
        return self._sources.cache_source(tool, args, source_id)

    def cached_source(self, tool, args):
        return self._sources.cached_source(tool, args)

    def link_source(self, job_id, source_id):
        return self._sources.link_source(job_id, source_id)

    def sources(self, ids):
        return self._sources.sources(ids)

    # Session checkpoints go back to the frontend as artifacts; the latest wins.
    def save_editor_checkpoint(self, job, value):
        self.captured["editor_checkpoint"] = value

    def save_fetch_failures(self, job, value):
        self.captured["fetch_failures"] = value


async def run(task_id: int, request: dict) -> dict:
    from commulingo.pipeline.engine import Usage
    from commulingo.pipeline.editor import Editor
    from commulingo.pipeline.workflow import Review

    job = dict(request["input"]["job"])
    job["payload"] = {**(job.get("payload") or {}), **(request["input"].get("settings") or {})}
    artifacts = request["input"]["artifacts"]
    store = TaskStore()
    usage = Usage()
    notes: list[dict] = []
    if request["kind"] == "commulingo_editor":
        stage = Editor(store)
    elif request["kind"] == "commulingo_discover":
        from commulingo.pipeline.stages import Discover
        stage = Discover()
    else:
        class WorkerReview(Review):
            async def leave_note(self, job, artifacts, decision, reason):
                notes.append({"decision": decision, "reason": reason or ""})
        stage = WorkerReview(store)
    result = error = None
    try:
        result = await stage(job, artifacts, usage, request["budgetUsd"])
    except Exception as exc:  # noqa: BLE001 - the checkpoint and cost still go back to the requester
        logger.warning("worker task %s (%s job %s) failed: %s", task_id, request["kind"], job.get("id"), exc)
        error = str(exc)
    tracker = usage.tracker
    report = {
        "artifacts": [{"stage": name, "value": value} for name, value in store.captured.items()],
        "notes": notes,
        "usage": {"costUsd": round(float(tracker.get("total_cost", 0) or 0), 6), "modelCalls": tracker.get("model_calls", 0),
                  **({"providerFallback": tracker["provider_fallback"]} if tracker.get("provider_fallback") else {})},
        "metrics": {k: tracker[k] for k in ("rejections", "workflow", "preflight_no_model", "partial_submissions",
                                            "citation_checks", "citation_rejections") if k in tracker},
        # A provider/process failure leaves usage unknown: the requester keeps the reservation.
        "costComplete": bool(usage.complete or not usage.started),
    }
    if error is not None:
        return {"error": error, **report}
    return {"stage": {"value": result.value, "nextStage": result.next_stage, "status": result.status,
                      "delaySeconds": result.delay_seconds}, **report}
