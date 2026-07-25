"""gateway.py — authorize a tool call against the unified policy.

``authorize(ctx, tool_name)`` is the decision function called at the
``execute_tool`` seam. It is pure apart from the Redis-backed sliding-window
rate counter. Taxonomy errors, authorization errors, and an unavailable rate
store for capped side effects fail closed.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from security_gateway import policy
from security_gateway.context import CallerContext

logger = logging.getLogger(__name__)

ALLOW = "allow"
DENY = "deny"
SHADOW_DENY = "shadow_deny"

_ATOMIC_SLIDING_WINDOW_LUA = r"""
local key = KEYS[1]
local cutoff = tonumber(ARGV[1])
local now = tonumber(ARGV[2])
local member = ARGV[3]
local ttl = tonumber(ARGV[4])
local max_calls = tonumber(ARGV[5])
local enforce = tonumber(ARGV[6])

redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)
local count = tonumber(redis.call('ZCARD', key))
local over = 0
if count >= max_calls then
    over = 1
end
if over == 1 and enforce == 1 then
    return {0, count, over}
end
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, ttl)
return {1, count + 1, over}
"""


@dataclass(frozen=True)
class Decision:
    """Outcome of an authorization check."""

    allowed: bool
    label: str
    risk_class: str
    reason: str
    mode: str
    rule: str

    @property
    def denied(self) -> bool:
        return not self.allowed


@dataclass(frozen=True)
class RateWindowResult:
    available: bool
    over: bool = False
    consumed: bool = False
    count: int = 0
    error: str = ""


def _who(ctx: CallerContext) -> str:
    return ctx.agent_name or ctx.user_id or ("owner" if ctx.is_owner else "anon")


def _rl_key(ctx: CallerContext, rclass: str) -> str:
    return f"gw:rl:{ctx.interface}:{_who(ctx)}:{rclass}"


def _window_consume_atomic(
    key: str,
    window_seconds: int,
    max_calls: int,
    now: float,
    *,
    enforce: bool,
) -> RateWindowResult:
    """Atomically evict, count, decide, and consume one sliding-window slot."""
    try:
        from redis_state import get_redis

        r = get_redis()
        if r is None:
            return RateWindowResult(False, error="Redis is unavailable")
        raw = r.eval(
            _ATOMIC_SLIDING_WINDOW_LUA,
            1,
            key,
            now - window_seconds,
            now,
            f"{now:.6f}:{uuid.uuid4().hex}",
            window_seconds + 5,
            max_calls,
            1 if enforce else 0,
        )
        consumed, count, over = [int(value) for value in raw]
        return RateWindowResult(True, bool(over), bool(consumed), count)
    except Exception as exc:
        logger.error("rate-limit store unavailable for %s: %s", key, exc)
        return RateWindowResult(False, error=str(exc))


def _authorize(
    ctx: CallerContext,
    tool_name: str,
    now: float,
    *,
    consume_rate_limit: bool,
) -> Decision:
    rclass = policy.risk_class(tool_name)
    mode = policy.enforce_mode()

    if rclass == policy.UNCATEGORIZED:
        return Decision(
            False,
            DENY,
            rclass,
            "uncategorized tool is denied until its risk class is registered",
            mode,
            "taxonomy",
        )

    public_allowed = {
        "webchat": policy.WEBCHAT_ALLOWED_RISK_CLASSES,
        "a2a": policy.A2A_ALLOWED_RISK_CLASSES,
    }.get(ctx.interface)
    if public_allowed is not None and rclass not in public_allowed:
        return Decision(
            False,
            DENY,
            rclass,
            f"{ctx.interface} is not permitted to call '{rclass}' tools",
            mode,
            "interface",
        )

    if rclass in policy.owner_required_classes() and not ctx.is_owner:
        if mode == policy.ENFORCE:
            return Decision(
                False,
                DENY,
                rclass,
                f"'{rclass}' is owner-only",
                mode,
                "owner",
            )
        return Decision(
            True,
            SHADOW_DENY,
            rclass,
            f"'{rclass}' is owner-only (shadow)",
            mode,
            "owner",
        )

    # Owner-gating for individual tools whose whole risk class is not owner-only
    # (the CommuLingo dictionary writers). Same shadow/enforce behaviour as the
    # class rule above, reported under its own rule name so the audit rows can
    # be told apart.
    if tool_name in policy.owner_required_tools() and not ctx.is_owner:
        if mode == policy.ENFORCE:
            return Decision(
                False,
                DENY,
                rclass,
                f"'{tool_name}' is owner-only",
                mode,
                "owner_tool",
            )
        return Decision(
            True,
            SHADOW_DENY,
            rclass,
            f"'{tool_name}' is owner-only (shadow)",
            mode,
            "owner_tool",
        )

    # Per-tool caller allow-list. The owner test above does not separate the
    # roleplay bot from the curation lanes, because that bot runs on a private
    # allow-listed channel and declares is_owner. Always enforced, like the
    # interface rule and for the same reason: it mirrors a profile pre-filter,
    # so a stale profile must not be able to talk past it.
    allowed_callers = policy.caller_allowlist_for(tool_name)
    if allowed_callers is not None:
        caller = ctx.agent_name or ctx.interface
        if caller not in allowed_callers:
            return Decision(
                False,
                DENY,
                rclass,
                f"'{caller}' is not permitted to call '{tool_name}'",
                mode,
                "caller",
            )

    rl = policy.rate_limit_for(rclass) if consume_rate_limit else None
    if rl and rl.get("max_calls"):
        window = int(rl.get("window_seconds", 3600))
        max_calls = int(rl["max_calls"])
        key = _rl_key(ctx, rclass)
        window_result = _window_consume_atomic(
            key,
            window,
            max_calls,
            now,
            enforce=mode == policy.ENFORCE,
        )
        if not window_result.available:
            return Decision(
                False,
                DENY,
                rclass,
                f"rate-limit store unavailable for '{rclass}' (fail-closed)",
                mode,
                "rate_store",
            )
        if window_result.over and mode == policy.ENFORCE:
            return Decision(
                False,
                DENY,
                rclass,
                f"rate limit exceeded ({max_calls}/{window}s for '{rclass}')",
                mode,
                "rate",
            )
        if window_result.over:
            return Decision(
                True,
                SHADOW_DENY,
                rclass,
                f"rate limit exceeded ({max_calls}/{window}s for '{rclass}', shadow)",
                mode,
                "rate",
            )

    return Decision(True, ALLOW, rclass, "", mode, "none")


def authorize(
    ctx: CallerContext,
    tool_name: str,
    args: dict | None = None,
    *,
    consume_rate_limit: bool = True,
) -> Decision:
    """Authorize a tool call. Fails closed on any internal gateway error."""
    now = time.time()
    try:
        return _authorize(
            ctx,
            tool_name,
            now,
            consume_rate_limit=consume_rate_limit,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("gateway.authorize failed closed for %s: %s", tool_name, exc)
        try:
            rclass = policy.risk_class(tool_name)
        except Exception:
            rclass = policy.UNCATEGORIZED
        try:
            mode = policy.enforce_mode()
        except Exception:
            mode = policy.ENFORCE
        return Decision(
            False,
            DENY,
            rclass,
            f"gateway error (fail-closed): {exc}",
            mode,
            "error",
        )
