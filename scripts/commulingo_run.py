"""One budget and durable outcome across all attempts of a curator job."""
import json
import sqlite3
import time
import uuid
import re
from pathlib import Path


class RunFailure(RuntimeError):
    def __init__(self, message, summary):
        super().__init__(message)
        self.summary = summary


def submitted_edit(writes, query_one):
    """Verify this run's explicit submission ID, never a lane-wide counter."""
    for write in reversed(writes):
        match = re.search(r"Logged as edit #(\d+)", write["result"])
        if match:
            row = query_one("SELECT id, target_type, target_id, action, status, confidence, created_at "
                "FROM commulingo_agent_suggestions WHERE id = %(id)s", {"id": int(match[1])})
            if not row or (write.get("target") and row["target_id"] != write["target"]):
                raise RuntimeError("submission receipt does not match the commissioned target")
            return row
    raise RuntimeError("successful write returned no submission receipt")


def finish_record(path, run_id, status):
    """Persist the final review/commit outcome after the LLM phase returns."""
    if not run_id:
        return
    with sqlite3.connect(path) as db:
        row = db.execute("SELECT summary FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row:
            summary = json.loads(row[0])
            summary["status"] = status
            db.execute("UPDATE runs SET status=?,summary=? WHERE run_id=?",
                       (status, json.dumps(summary, ensure_ascii=False), run_id))


class RunBudget:
    def __init__(self, policy, path, stage, target, *, seconds=480):
        from tool_gateway.security import get_caller
        self.run_id = getattr(get_caller(), "request_id", None) or uuid.uuid4().hex
        self.policy, self.path = policy, path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.stage, self.target = stage, target
        self.started = time.monotonic()
        self.seconds = seconds
        self.cost = 0.0
        self.rounds = 0
        self.attempts = 0
        self.extra = {}
        self.reservation = None
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, ts REAL, stage TEXT, target TEXT, status TEXT, summary TEXT)")
        self.record("running")

    def remaining(self):
        seconds = self.seconds - (time.monotonic() - self.started)
        rounds = self.policy.max_rounds - self.rounds
        cost = self.policy.budget_usd - self.cost if self.policy.budget_usd > 0 else 0
        if seconds <= 0 or rounds <= 0 or (self.policy.budget_usd > 0 and cost <= 0):
            raise RunFailure("job budget, round or time limit exhausted", self.record("exhausted"))
        if not self.reservation:
            from commulingo_pipeline.config import legacy_reserve
            from commulingo_pipeline.store import BudgetUnavailable
            try:
                self.reservation = legacy_reserve(cost or 0.35, 'review' if self.stage=='review' else self.stage)
            except BudgetUnavailable as exc:
                raise RunFailure(str(exc),self.record('budget_deferred')) from exc
        return seconds, rounds, cost

    def account(self, tracker):
        if self.reservation:
            store, token = self.reservation
            # A failed call with unknown usage retains its conservative reservation.
            if tracker.get('pipeline_call_complete'):
                store.settle(token,tracker['total_cost'])
            self.reservation = None
        self.cost += float(tracker.get("total_cost") or 0)
        self.rounds += int(tracker.get("rounds_used") or 0)
        self.attempts += 1

    def record(self, status, **extra):
        self.extra.update(extra)
        summary = {"run_id": self.run_id, "status": status,
            "total_cost": self.cost, "cost_usd": self.cost, "rounds_used": self.rounds,
            "attempts": self.attempts, "elapsed_seconds": time.monotonic() - self.started, **self.extra}
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
                (self.run_id, time.time(), self.stage, self.target, status, json.dumps(summary, ensure_ascii=False)))
        return summary
