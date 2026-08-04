"""llm/gateway.py — single seam for every LLM API call.

Every managed LLM call in the project flows through this module's two calls:

  check_llm_call(...)   before the provider request — policy gate
  record_llm_call(...)  after it — spend/usage audit

Integration points (the seam is these three, not N call sites):

  1. agent_loop.LoopState.add_cost — every tool-loop round on every provider
     (both protocol adapters already funnel their cost events here)
  2. run_tool_loop entry — one policy check per agent turn
  3. llm.call_registry.generate_sync — every registered one-shot call

Design borrows from LiteLLM's proxy data model (spend logs with model/tokens/
cost/tags, hierarchical budgets, hard vs soft limits) but stays in-process:
putting a proxy hop in front of streaming + prompt-cache breakpoints + thinking
replay would risk regressions in exactly the provider mechanics the adapters
exist to protect. Audit follows security_gateway/audit.py: a structured JSON
line on the ``llm_gateway.audit`` logger (journald), plus a best-effort row in
the ``llm_audit_log`` Postgres table from one background worker thread. Neither
sink may ever raise into, or block, an LLM call.

Policy (config/llm_gateway.json, hot-reloaded on mtime):

  {
    "enforce": false,                  // false = shadow: log would-deny, allow
    "block_all": false,                // kill switch
    "blocked_providers": [],           // e.g. ["openai"]
    "blocked_models": [],              // exact model IDs
    "daily_budget_usd": null,          // total across providers, UTC day
    "daily_budget_per_provider": {}    // e.g. {"claude": 20.0}
  }

Budget checks read today's SUM(cost_usd) from llm_audit_log (cached 60s).
A DB failure fails OPEN: availability of the bot outranks enforcement.

Known gaps (calls that do not pass this seam): graphiti-core's internal
OpenAI client (KG extraction), browser-use's internal calls, razvedchik's
own cloud client, and the Codex CLI delegation. Documented in
dev_docs/llm_gateway.md; migrate them by pointing their call paths here.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

logger = logging.getLogger("llm_gateway.audit")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "llm_gateway.json"

_ERROR_EXCERPT_CAP = 500

_DEFAULTS = {
    "enforce": False,
    "block_all": False,
    "blocked_providers": [],
    "blocked_models": [],
    "daily_budget_usd": None,
    "daily_budget_per_provider": {},
}


class LLMGatewayDenied(RuntimeError):
    """An LLM call was refused by gateway policy (enforce mode only)."""


# ── Config (hot reload, mirrors call_registry) ────────────────────────

_config_lock = threading.Lock()
_config_cache: dict | None = None
_config_mtime: float | None = None


def load_policy() -> dict:
    global _config_cache, _config_mtime
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        return dict(_DEFAULTS)
    with _config_lock:
        if _config_cache is not None and _config_mtime == mtime:
            return _config_cache
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON must be an object")
        except Exception as e:
            logger.error("[llm-gateway] config unreadable (%s); keeping previous", e)
            return _config_cache or dict(_DEFAULTS)
        _config_cache = {**_DEFAULTS, **data}
        _config_mtime = mtime
        return _config_cache


# ── Provider inference / cost estimation ─────────────────────────────

_PROVIDER_PREFIXES = (
    ("claude", "claude"),
    ("gpt", "openai"),
    ("deepseek", "deepseek"),
    ("kimi", "kimi"),
    ("qwen", "local"),
    ("qwopus", "local"),
    ("gemini", "gemini"),
)


def infer_provider(model: str | None) -> str | None:
    """Best-effort provider name from a model ID, for audit labeling."""
    m = (model or "").strip().lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if m.startswith(prefix):
            return provider
    return None


def estimate_cost_usd(
    model: str | None, *, tokens_in: int = 0, tokens_out: int = 0,
    cache_read: int = 0, cache_create: int = 0,
) -> float | None:
    """Best-effort cost from the canonical pricing tables.

    Anthropic-protocol semantics: tokens_in excludes cache tokens, which are
    billed separately. OpenAI-compatible semantics: tokens_in INCLUDES the
    cached tokens (prompt_tokens), so cached are subtracted before pricing.
    Unknown models return None rather than a fabricated number — the loop
    adapters already computed their own exact cost and pass it in, so this
    is only used for one-shot calls.
    """
    if not model:
        return None
    from llm.provider_registry import OPENAI_COMPATIBLE_PRICING, anthropic_pricing_table

    def _lookup(table: dict) -> dict | None:
        if model in table:
            return table[model]
        # Dated/pinned variants, e.g. claude-haiku-4-5-20251001.
        for base, price in table.items():
            if model.startswith(base + "-") or model.startswith(base + "."):
                return price
        return None

    p = _lookup(anthropic_pricing_table())
    if p is not None:
        return (
            tokens_in * p["input"]
            + tokens_out * p["output"]
            + cache_create * p.get("cache_creation", p["input"])
            + cache_read * p.get("cache_read", 0.0)
        )
    p = _lookup(OPENAI_COMPATIBLE_PRICING)
    if p is not None:
        non_cached = max(0, tokens_in - cache_read)
        return (
            non_cached * p["input"]
            + cache_read * p.get("cached_input", p["input"])
            + tokens_out * p["output"]
        )
    return None


# ── Daily spend (for budget policy; cached, fail-open) ───────────────

_SPEND_CACHE_TTL_SEC = 60.0
_spend_lock = threading.Lock()
_spend_cache: dict[str, float] | None = None
_spend_cache_at = 0.0


def _today_spend() -> dict[str, float] | None:
    """Today's (UTC) cost by provider from llm_audit_log; None if unknown."""
    global _spend_cache, _spend_cache_at
    now = time.monotonic()
    with _spend_lock:
        if _spend_cache is not None and now - _spend_cache_at < _SPEND_CACHE_TTL_SEC:
            return _spend_cache
    try:
        from db import query

        rows = query(
            """SELECT COALESCE(provider, '?') AS provider,
                      COALESCE(SUM(cost_usd), 0)::float AS spend
                 FROM llm_audit_log
                WHERE ts >= date_trunc('day', now() AT TIME ZONE 'utc')
                  AND status IN ('ok', 'error')
                GROUP BY 1""",
        )
        spend = {str(r["provider"]): float(r["spend"]) for r in rows or []}
    except Exception as e:
        logger.warning("[llm-gateway] daily spend unavailable (fail-open): %s", e)
        return None
    with _spend_lock:
        _spend_cache, _spend_cache_at = spend, now
    return spend


def _invalidate_spend_cache() -> None:
    global _spend_cache
    with _spend_lock:
        _spend_cache = None


# ── Policy gate ──────────────────────────────────────────────────────

def check_llm_call(
    *, surface: str, caller: str | None, model: str | None,
    provider: str | None = None,
) -> None:
    """Gate one LLM call. Raises LLMGatewayDenied only in enforce mode.

    In shadow mode (enforce=false) a would-deny is logged and recorded but
    the call proceeds. This function itself must never raise anything else.
    """
    try:
        policy = load_policy()
        provider = provider or infer_provider(model)
        reason = None

        if policy["block_all"]:
            reason = "block_all kill switch"
        elif provider and provider in (policy["blocked_providers"] or []):
            reason = f"provider {provider} blocked"
        elif model and model in (policy["blocked_models"] or []):
            reason = f"model {model} blocked"
        else:
            total_cap = policy["daily_budget_usd"]
            provider_caps = policy["daily_budget_per_provider"] or {}
            if total_cap is not None or provider_caps:
                spend = _today_spend()
                if spend is not None:
                    total = sum(spend.values())
                    if total_cap is not None and total >= float(total_cap):
                        reason = f"daily budget ${float(total_cap):.2f} exhausted (${total:.2f})"
                    elif provider and provider in provider_caps and (
                        spend.get(provider, 0.0) >= float(provider_caps[provider])
                    ):
                        reason = (
                            f"daily {provider} budget ${float(provider_caps[provider]):.2f} "
                            f"exhausted (${spend.get(provider, 0.0):.2f})"
                        )

        if reason is None:
            return
        enforce = bool(policy["enforce"])
        _emit(
            {
                "surface": surface, "caller": caller, "provider": provider,
                "model": model, "label": None, "tokens_in": None,
                "tokens_out": None, "cache_read": None, "cache_create": None,
                "cost_usd": None, "latency_ms": None,
                "status": "denied" if enforce else "would_deny",
                "error_excerpt": reason,
            },
            warn=True,
        )
        if enforce:
            raise LLMGatewayDenied(f"LLM call denied: {reason}")
    except LLMGatewayDenied:
        raise
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[llm-gateway] check failed (allowing call): %s", e)


# ── Audit recording ──────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS llm_audit_log (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    surface       TEXT NOT NULL,
    caller        TEXT,
    provider      TEXT,
    model         TEXT,
    label         TEXT,
    tokens_in     INTEGER,
    tokens_out    INTEGER,
    cache_read    INTEGER,
    cache_create  INTEGER,
    cost_usd      NUMERIC(14, 8),
    latency_ms    INTEGER,
    status        TEXT NOT NULL,
    error_excerpt TEXT
);
"""
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS llm_audit_log_ts_idx ON llm_audit_log (ts DESC)",
    "CREATE INDEX IF NOT EXISTS llm_audit_log_caller_ts_idx ON llm_audit_log (caller, ts DESC)",
    "CREATE INDEX IF NOT EXISTS llm_audit_log_provider_ts_idx ON llm_audit_log (provider, ts DESC)",
]

_INSERT = """
INSERT INTO llm_audit_log
    (surface, caller, provider, model, label, tokens_in, tokens_out,
     cache_read, cache_create, cost_usd, latency_ms, status, error_excerpt)
VALUES (%(surface)s, %(caller)s, %(provider)s, %(model)s, %(label)s,
        %(tokens_in)s, %(tokens_out)s, %(cache_read)s, %(cache_create)s,
        %(cost_usd)s, %(latency_ms)s, %(status)s, %(error_excerpt)s)
"""


def ensure_llm_audit_log_table() -> None:
    """Create the llm_audit_log table and indexes. Applied via schema_migrations."""
    from db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
            for stmt in _INDEXES:
                cur.execute(stmt)
        conn.commit()


_DB_QUEUE: "queue.Queue[dict]" = queue.Queue(maxsize=2000)
_worker_started = False
_worker_lock = threading.Lock()
_insert_failed_before = False


def _drain_one(row: dict) -> None:
    global _insert_failed_before
    try:
        from db import get_conn

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_INSERT, row)
            conn.commit()
        _insert_failed_before = False
    except Exception as e:
        # First failure at WARNING, repeats at DEBUG: an ad-hoc process under
        # the read-only DB guard would otherwise warn on every single call.
        log = logger.debug if _insert_failed_before else logger.warning
        log("llm audit DB insert failed (dropped): %s", e)
        _insert_failed_before = True


def _worker_loop() -> None:
    while True:
        row = _DB_QUEUE.get()
        try:
            _drain_one(row)
        finally:
            _DB_QUEUE.task_done()


def _ensure_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_worker_loop, name="llm-audit-writer", daemon=True)
        t.start()
        _worker_started = True


def _emit(row: dict, *, warn: bool = False) -> None:
    """Send one audit row to both sinks. Never raises."""
    try:
        (logger.warning if warn else logger.info)(
            "llm_call %s",
            json.dumps(
                {k: v for k, v in row.items() if v is not None},
                ensure_ascii=False, default=str,
            ),
        )
        if os.getenv("LENINBOT_LLM_AUDIT_DB", "1") == "0":
            return
        _ensure_worker()
        try:
            _DB_QUEUE.put_nowait(row)
        except queue.Full:
            logger.warning("llm audit queue full; dropped row for %s", row.get("caller"))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("llm audit emit failed (ignored): %s", e)


def record_llm_call(
    *, surface: str, caller: str | None, model: str | None,
    provider: str | None = None, label: str | None = None,
    tokens_in: int = 0, tokens_out: int = 0,
    cache_read: int = 0, cache_create: int = 0,
    cost_usd: float | None = None, latency_ms: int | None = None,
    status: str = "ok", error_excerpt: str | None = None,
) -> None:
    """Record one completed (or failed) LLM call. Never raises."""
    try:
        if error_excerpt and len(error_excerpt) > _ERROR_EXCERPT_CAP:
            error_excerpt = error_excerpt[:_ERROR_EXCERPT_CAP] + "…"
        if cost_usd is None:
            cost_usd = estimate_cost_usd(
                model, tokens_in=tokens_in, tokens_out=tokens_out,
                cache_read=cache_read, cache_create=cache_create,
            )
        _emit(
            {
                "surface": surface,
                "caller": caller,
                "provider": provider or infer_provider(model),
                "model": model,
                "label": label,
                "tokens_in": int(tokens_in or 0),
                "tokens_out": int(tokens_out or 0),
                "cache_read": int(cache_read or 0),
                "cache_create": int(cache_create or 0),
                "cost_usd": round(cost_usd, 8) if cost_usd is not None else None,
                "latency_ms": latency_ms,
                "status": status,
                "error_excerpt": error_excerpt,
            },
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("record_llm_call failed (ignored): %s", e)
