"""Recorded KG healthcheck results, read by the MCP gateway instead of hourly Telegram alerts.

The hourly `scripts/check_kg_integrity.py --record` run writes the latest result and
keeps each failing component open with the time it started. Only a component that
stays failing for ESCALATE_AFTER is sent to Telegram, once per continuous failure.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ops.paths import PROJECT_ROOT

STATE_DIR = Path(os.getenv("KG_HEALTH_STATE_DIR", PROJECT_ROOT / "output" / "kg_health"))
ESCALATE_AFTER = timedelta(days=3)


def _state_path() -> Path:
    return STATE_DIR / "state.json"


def failure_key(reason: str) -> str:
    # "sync.documents: 1 document(s) failed …" and "sync.documents: unresolved …" are
    # one incident; numbers after the colon change every run.
    return reason.split(":", 1)[0].strip()


def load() -> dict | None:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def record(health: dict, *, now: datetime | None = None) -> list[dict]:
    """Store this run and return open failures that just crossed ESCALATE_AFTER."""
    now = now or datetime.now(timezone.utc)
    stamp = now.isoformat(timespec="seconds")
    prev = (load() or {}).get("open", {})
    current: dict[str, list[str]] = {}
    for reason in health["failures"]:
        current.setdefault(failure_key(reason), []).append(reason)
    opened, events, escalate = {}, [], []
    for key, reasons in current.items():
        item = dict(prev.get(key) or {"since": stamp})
        item.update(last_seen=stamp, reasons=reasons)
        if key not in prev:
            events.append({"at": stamp, "event": "opened", "key": key, "reasons": reasons})
        if not item.get("escalated_at") and now - datetime.fromisoformat(item["since"]) >= ESCALATE_AFTER:
            item["escalated_at"] = stamp
            escalate.append({"key": key, **item})
        opened[key] = item
    for key, item in prev.items():
        if key not in current:
            events.append({"at": stamp, "event": "resolved", "key": key, "since": item["since"]})
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _state_path().with_suffix(".tmp")
    tmp.write_text(json.dumps({"checked_at": stamp, "ok": health["ok"], "failures": health["failures"],
                               "open": opened}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_state_path())
    if events:
        with (STATE_DIR / "history.jsonl").open("a", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return escalate


def format_escalation(items: list[dict]) -> str:
    lines = [f"KG healthcheck: {len(items)} failure(s) unresolved for {ESCALATE_AFTER.days}+ days"
             " (sent once; details in MCP pipeline_status kg_sync)"]
    for item in items:
        lines.append(f"- {item['key']} since {item['since'][:16]}Z")
        lines.extend(f"  {reason[:400]}" for reason in item["reasons"][:3])
    return "\n".join(lines)[:3500]


def summary(*, now: datetime | None = None) -> dict:
    """Latest recorded result for the MCP gateway; never reruns the check."""
    state = load()
    if state is None:
        return {"recorded": False, "note": "no recorded healthcheck (timer not yet run with --record)"}
    now = now or datetime.now(timezone.utc)
    age = (now - datetime.fromisoformat(state["checked_at"])).total_seconds() / 3600
    open_ = {key: {**item, "open_hours": round((now - datetime.fromisoformat(item["since"])).total_seconds() / 3600, 1)}
             for key, item in state.get("open", {}).items()}
    return {"recorded": True, "checked_at": state["checked_at"], "age_hours": round(age, 2),
            "stale": age > 2, "ok": state["ok"], "open_failures": open_}
