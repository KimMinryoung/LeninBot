"""Shared, fail-closed paid web budget. No queries, URLs or credentials stored.

SQLite BEGIN IMMEDIATE serializes reservations across local service processes.
Only provider requests enter this ledger; caches and free retrieval do not.
"""
from __future__ import annotations

import json
import os
from contextvars import ContextVar
import logging
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path

from security_gateway.context import get_caller

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/web_research.json"
STORE_PATH = Path(os.environ.get("WEB_RESEARCH_USAGE_DB", "/var/lib/leninbot-web-gateway/usage.sqlite3"))
usage_identity: ContextVar[dict | None] = ContextVar("web_usage_identity", default=None)
logger = logging.getLogger(__name__)


class PaidWebBudgetError(RuntimeError):
    """No paid request may be made; existing evidence/free tools remain usable."""


def _micros(value) -> int:
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError("expected a finite non-negative amount")
    return int((number * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


def policy() -> dict:
    # Read on each reservation so a budget edit applies to running services.
    config = json.loads(CONFIG_PATH.read_text())
    return {key: _micros(config[key]) for key in
            ("daily_budget_usd", "tavily_credit_usd", "brave_search_usd")}


def _service() -> str:
    try:
        for part in Path("/proc/self/cgroup").read_text().replace("\n", "/").split("/"):
            if part.endswith(".service"):
                return part
    except OSError:
        pass
    return "cli:" + Path(sys.argv[0]).name


def _identity() -> dict:
    if usage_identity.get() is not None:
        return usage_identity.get()
    caller = get_caller()
    return {"service": _service(), "interface": caller.interface,
            "agent": caller.agent_name or "", "task_id": caller.task_id or "",
            "scope_type": caller.scope_type or "", "scope_id": caller.scope_id or "",
            "request_id": caller.request_id or ""}


@contextmanager
def _db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=5)
    try:
        with db:
            db.execute("""CREATE TABLE IF NOT EXISTS web_usage (
                id TEXT PRIMARY KEY, ts TEXT NOT NULL, day TEXT NOT NULL,
                service TEXT, interface TEXT, agent TEXT, task_id TEXT,
                scope_type TEXT, scope_id TEXT, request_id TEXT,
                provider TEXT, operation TEXT, depth TEXT,
                status TEXT, reserved_micros INTEGER, accounted_micros INTEGER,
                credit_micros INTEGER, credit_price_micros INTEGER)""")
            db.execute("CREATE INDEX IF NOT EXISTS web_usage_day ON web_usage(day)")
            yield db
    finally:
        db.close()


@dataclass
class Charge:
    path: Path
    token: str
    reserved: int
    credit_price: int
    done: bool = False

    def complete(self, credits=None):
        """Use reported credits, including zero; absent usage retains the estimate."""
        credit_micros = None
        if credits is not None:
            try:
                credit_micros = _micros(credits)
            except (ValueError, InvalidOperation):
                pass
        amount = self.reserved if credit_micros is None else (
            credit_micros * self.credit_price + 999_999) // 1_000_000
        status = "reported" if credit_micros is not None else "estimated"
        self._finish(status, amount, credit_micros)

    def _finish(self, status, amount, credits=None):
        try:
            with _db(self.path) as db:
                db.execute("UPDATE web_usage SET status=?, accounted_micros=?, credit_micros=? WHERE id=?",
                           (status, amount, credits, self.token))
            self.done = True
            logger.info("paid_web settled id=%s status=%s accounted_micros=%s", self.token, status, amount)
        except (OSError, sqlite3.Error):
            # The durable pre-call reservation still counts. Never lose a good
            # provider result (and induce a retry) because settlement failed.
            logger.exception("paid_web settlement unavailable id=%s; reservation retained", self.token)


def reserve(provider: str, operation: str, depth: str) -> Charge:
    try:
        config = policy()
        if provider == "tavily" and operation in {"search", "extract"}:
            credit_price = config["tavily_credit_usd"]
            # Single-URL extract reserves a whole credit conservatively (the
            # provider documents batches of five); actual usage settles it.
            amount = credit_price * (2 if depth == "advanced" else 1)
        elif provider == "brave" and operation == "search":
            credit_price = 0
            amount = config["brave_search_usd"]
        else:
            raise ValueError("unknown paid web operation")
        if amount <= 0:
            raise ValueError("paid provider price must be positive")
        identity = _identity()
        now = datetime.now(timezone.utc)
        day, token = now.date().isoformat(), uuid.uuid4().hex
        path = STORE_PATH
        with _db(path) as db:
            db.execute("BEGIN IMMEDIATE")
            spent = db.execute("SELECT COALESCE(SUM(accounted_micros),0) FROM web_usage WHERE day=?", (day,)).fetchone()[0]
            if spent + amount > config["daily_budget_usd"]:
                logger.warning("paid_web budget_denied provider=%s operation=%s caller=%s", provider, operation, json.dumps(identity))
                raise PaidWebBudgetError(
                    "Paid web daily budget exhausted (UTC). Stop paid searches/extraction; "
                    "reuse saved evidence or free sources and report any unresolved facts. "
                    "Do not retry with another query or provider.")
            db.execute("INSERT INTO web_usage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                token, now.isoformat(), day, *identity.values(), provider, operation, depth,
                "reserved", amount, amount, None, credit_price,
            ))
        logger.info("paid_web reserved id=%s provider=%s operation=%s micros=%d caller=%s",
                    token, provider, operation, amount, json.dumps(identity))
        return Charge(path, token, amount, credit_price)
    except PaidWebBudgetError:
        raise
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, InvalidOperation) as exc:
        logger.warning("paid_web budget unavailable: %s", type(exc).__name__)
        raise PaidWebBudgetError(
            "Paid web budget unavailable; no paid request was sent. "
            "Use saved evidence/free sources and report unresolved facts.") from exc


@contextmanager
def paid_request(provider: str, operation: str, depth: str):
    charge = reserve(provider, operation, depth)
    try:
        yield charge
    finally:
        if not charge.done:
            # Includes timeout, cancellation, malformed response and process
            # interruption. A hard crash leaves status=reserved, also counted.
            charge._finish("outcome_unknown", charge.reserved)


def report(since: str, *, by: str = "service") -> list[dict]:
    """Read-only aggregate; never initializes an absent ledger."""
    groups = {"service": "service, interface, agent", "task": "service, task_id, scope_type, scope_id, request_id"}
    columns = groups[by]
    if not STORE_PATH.exists():
        return []
    db = sqlite3.connect(STORE_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(f"""SELECT day, {columns}, provider, operation, depth,
            COUNT(*) AS requests, SUM(accounted_micros)/1000000.0 AS accounted_usd,
            SUM(CASE WHEN status IN ('reserved','outcome_unknown') THEN 1 ELSE 0 END) AS unknown_requests,
            SUM(CASE WHEN status='estimated' THEN 1 ELSE 0 END) AS estimated_requests,
            SUM(credit_micros)/1000000.0 AS reported_credits
            FROM web_usage WHERE day>=? GROUP BY day, {columns}, provider, operation, depth
            ORDER BY day DESC, accounted_usd DESC""", (since,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        db.close()


def import_legacy_usage():
    """Carry pre-gateway reservations forward once per row, without resetting the day.

    Deployment drains old consumers first. Repeated startup is idempotent; any
    old unconfirmed reservation remains conservative rather than being refunded.
    """
    legacy = ROOT / "data/web_research_usage.sqlite3"
    if not legacy.exists() or legacy.resolve() == STORE_PATH.resolve():
        return
    source = sqlite3.connect(legacy.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = source.execute("SELECT * FROM web_usage").fetchall()
        with _db(STORE_PATH) as db:
            db.executemany("INSERT OR IGNORE INTO web_usage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    finally:
        source.close()
