"""llm/gateway.py — single seam for every LLM API call.

Every managed LLM call in the project flows through this module's two calls:

  check_llm_call(...)   before the provider request — policy gate
  record_llm_call(...)  after it — spend/usage audit

Primary integration points:

  1. agent_loop.LoopState.add_cost — every tool-loop round on every provider
     (both protocol adapters already funnel their cost events here)
  2. run_tool_loop entry — one policy check per agent turn
  3. llm.call_registry.generate_sync — every registered one-shot call
  4. audited third-party wrappers — Graphiti Gemini, browser-use, vision

Design borrows from LiteLLM's proxy data model (spend logs with model/tokens/
cost/tags, hierarchical budgets, hard vs soft limits) but stays in-process:
putting a proxy hop in front of streaming + prompt-cache breakpoints + thinking
replay would risk regressions in exactly the provider mechanics the adapters
exist to protect. Audit follows security_gateway/audit.py: a structured JSON
line on the ``llm_gateway.audit`` logger (journald), plus a best-effort row in
the ``llm_audit_log`` Postgres table from one background worker thread. Neither
sink may ever raise into, or block, an LLM call.

Policy (tracked defaults + ignored local overrides, hot-reloaded on mtime):

  {
    "enforce": true,                   // local false = shadow: log would-deny, allow
    "block_all": false,                // kill switch
    "blocked_providers": [],           // e.g. ["openai"]
    "blocked_models": [],              // exact model IDs
    "daily_budget_usd": null,          // total across providers, UTC day
    "daily_budget_per_provider": {}    // e.g. {"claude": 20.0}
  }

Budget checks read today's SUM(cost_usd) from llm_audit_log (cached 60s).
A DB failure fails OPEN: availability of the bot outranks enforcement.

This module is the observation/policy half; the enforcement half is the
key-injection passthrough proxy (llm_proxy/, switched on via "proxy_base").
Coverage and the enforcement runbook: dev_docs/llm_gateway.md.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

logger = logging.getLogger("llm_gateway.audit")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "llm_gateway.defaults.json"
LOCAL_CONFIG_PATH = CONFIG_DIR / "llm_gateway.local.json"

_ERROR_EXCERPT_CAP = 500

_DEFAULTS = {
    "enforce": True,
    "block_all": False,
    "blocked_providers": [],
    "blocked_models": [],
    "daily_budget_usd": None,
    "daily_budget_per_provider": {},
    # When set (e.g. "http://127.0.0.1:8110"), managed clients route through
    # the key-injection passthrough proxy (llm_proxy/, the enforcement half
    # of the gateway) instead of calling providers directly. Applied at
    # client construction — a service restart picks it up.
    "proxy_base": None,
}


class LLMGatewayDenied(RuntimeError):
    """An LLM call was refused by gateway policy (enforce mode only)."""


# ── Config (hot reload, mirrors call_registry) ────────────────────────

_config_lock = threading.Lock()
_config_cache: dict | None = None
_config_signature: tuple | None = None


def _path_signature(path: Path) -> tuple[bool, int | None, int | None]:
    try:
        stat = path.stat()
        return True, stat.st_mtime_ns, stat.st_size
    except OSError:
        return False, None, None


def _read_policy_file(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")
    return data


def load_policy() -> dict:
    """Load code defaults, tracked defaults, then host-local overrides."""
    global _config_cache, _config_signature
    signature = (
        _path_signature(DEFAULT_CONFIG_PATH),
        _path_signature(LOCAL_CONFIG_PATH),
    )
    with _config_lock:
        if _config_cache is not None and _config_signature == signature:
            return _config_cache

        merged = dict(_DEFAULTS)
        try:
            if DEFAULT_CONFIG_PATH.exists():
                merged.update(_read_policy_file(DEFAULT_CONFIG_PATH))
        except Exception as e:
            logger.error(
                "[llm-gateway] defaults config unreadable at %s (%s); keeping previous",
                DEFAULT_CONFIG_PATH,
                e,
            )
            return _config_cache or dict(_DEFAULTS)

        try:
            if LOCAL_CONFIG_PATH.exists():
                merged.update(_read_policy_file(LOCAL_CONFIG_PATH))
        except Exception as e:
            logger.error(
                "[llm-gateway] local config unreadable at %s (%s); keeping previous",
                LOCAL_CONFIG_PATH,
                e,
            )
            return _config_cache or merged

        _config_cache = merged
        _config_signature = signature
        return _config_cache


# ── Proxy routing (enforcement half — see llm_proxy/app.py) ──────────

# Placeholder API key clients present when the real key lives only in the
# proxy. Never a secret; the proxy strips and replaces it.
PROXY_PLACEHOLDER_KEY = "via-llm-proxy"


def proxy_base() -> str | None:
    """The llm_proxy base URL, or None when clients call providers directly."""
    base = load_policy().get("proxy_base")
    return str(base).rstrip("/") if base else None


def provider_endpoint(
    proxy_path: str, direct_base: str | None, api_key: str | None,
) -> tuple[str | None, str]:
    """Resolve (base_url, api_key) for one provider client.

    proxy_path is the path prefix on the proxy that maps to the provider's
    upstream host (e.g. "deepseek/anthropic" → api.deepseek.com/anthropic).
    Precedence: explicit env override in direct_base stays authoritative at
    the call site; this only swaps in the proxy when configured. With the
    proxy on, a missing real key is fine — the placeholder satisfies SDK
    constructors and the proxy injects the real one.
    """
    base = proxy_base()
    if base:
        return f"{base}/{proxy_path}", (api_key or PROXY_PLACEHOLDER_KEY)
    return direct_base, (api_key or "")


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
    token_semantics: str | None = None,
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
    from llm.provider_registry import (
        GEMINI_PRICING,
        OPENAI_COMPATIBLE_PRICING,
        anthropic_pricing_table,
    )

    def _lookup(table: dict) -> dict | None:
        if model in table:
            return table[model]
        # Dated/pinned variants, e.g. claude-haiku-4-5-20251001.
        for base, price in table.items():
            if model.startswith(base + "-") or model.startswith(base + "."):
                return price
        return None

    def _anthropic_cost(p: dict) -> float:
        return (
            tokens_in * p["input"]
            + tokens_out * p["output"]
            + cache_create * p.get("cache_creation", p["input"])
            + cache_read * p.get("cache_read", 0.0)
        )

    def _prompt_total_cost(p: dict) -> float:
        non_cached = max(0, tokens_in - cache_read)
        return (
            non_cached * p["input"]
            + cache_read * p.get("cached_input", p["input"])
            + tokens_out * p["output"]
        )

    # Protocol is authoritative for overlapping models such as DeepSeek and
    # Kimi.  Auto mode remains for legacy callers and prefers native Claude,
    # then Gemini, then OpenAI-compatible pricing.
    if token_semantics == "anthropic":
        p = _lookup(anthropic_pricing_table())
        return _anthropic_cost(p) if p is not None else None
    if token_semantics == "gemini":
        p = _lookup(GEMINI_PRICING)
        return _prompt_total_cost(p) if p is not None else None
    if token_semantics == "openai":
        p = _lookup(OPENAI_COMPATIBLE_PRICING)
        return _prompt_total_cost(p) if p is not None else None

    p = _lookup(anthropic_pricing_table())
    if p is not None:
        return _anthropic_cost(p)
    p = _lookup(GEMINI_PRICING)
    if p is not None:
        return _prompt_total_cost(p)
    p = _lookup(OPENAI_COMPATIBLE_PRICING)
    if p is not None:
        return _prompt_total_cost(p)
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

def evaluate_policy(
    *, provider: str | None, model: str | None,
) -> tuple[str | None, bool]:
    """Pure policy decision: (deny_reason_or_None, enforce). Never raises.

    The single source of policy truth. Called from BOTH evaluation points —
    the in-process seam (check_llm_call: early fail, caller-tagged shadow
    rows) and the authoritative one, llm_proxy (which a client cannot
    bypass once the provider keys live only there).
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
        return reason, bool(policy["enforce"])
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[llm-gateway] policy evaluation failed (allowing): %s", e)
        return None, False


def check_llm_call(
    *, surface: str, caller: str | None, model: str | None,
    provider: str | None = None,
) -> None:
    """Gate one LLM call. Raises LLMGatewayDenied only in enforce mode.

    In shadow mode (enforce=false) a would-deny is logged and recorded but
    the call proceeds. This function itself must never raise anything else.
    """
    provider = provider or infer_provider(model)
    reason, enforce = evaluate_policy(provider=provider, model=model)
    if reason is None:
        return
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
        # 짧게 살다 죽는 프로세스를 위한 보험. 아래 flush_audit 설명 참조.
        atexit.register(_flush_at_exit)


# 종료 시 감사 큐를 비우는 데 기다릴 최대 시간(초).
FLUSH_TIMEOUT_SECONDS = float(os.getenv("LENINBOT_LLM_AUDIT_FLUSH_SECONDS", "5"))


def flush_audit(timeout: float = FLUSH_TIMEOUT_SECONDS) -> bool:
    """큐에 남은 감사 행이 DB에 들어갈 때까지 기다린다.

    기록 워커는 daemon 스레드다. 오래 사는 서비스에서는 문제가 없지만, 배치
    스크립트처럼 한 번 돌고 끝나는 프로세스는 워커가 큐를 비우기 전에 죽어서
    행이 통째로 사라진다. 프록시 쪽 행(surface=proxy)은 남으니 지출 자체는
    보존되지만, '누가 썼는지'(caller)를 들고 있는 것은 이쪽 행이라 CLI 실행만
    비용 귀속이 빠진 채로 남았다.

    큐 자신의 조건변수를 쓰므로 Queue.join()의 타임아웃 있는 판본과 같다.
    큐가 시간 안에 비면 True.
    """
    if not _worker_started:
        return True
    deadline = time.monotonic() + max(0.0, timeout)
    with _DB_QUEUE.all_tasks_done:
        while _DB_QUEUE.unfinished_tasks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "llm audit flush timed out; %d row(s) dropped",
                    _DB_QUEUE.unfinished_tasks,
                )
                return False
            _DB_QUEUE.all_tasks_done.wait(remaining)
    return True


def _flush_at_exit() -> None:
    # atexit에서는 절대 예외를 올리지 않는다 — 종료 경로를 깨는 것보다 감사
    # 행 몇 개를 잃는 편이 낫다.
    try:
        flush_audit()
    except Exception:
        pass


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
    token_semantics: str | None = None, estimate_cost: bool = True,
) -> None:
    """Record one completed (or failed) LLM call. Never raises."""
    try:
        if error_excerpt and len(error_excerpt) > _ERROR_EXCERPT_CAP:
            error_excerpt = error_excerpt[:_ERROR_EXCERPT_CAP] + "…"
        if cost_usd is None and estimate_cost:
            cost_usd = estimate_cost_usd(
                model, tokens_in=tokens_in, tokens_out=tokens_out,
                cache_read=cache_read, cache_create=cache_create,
                token_semantics=token_semantics,
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
