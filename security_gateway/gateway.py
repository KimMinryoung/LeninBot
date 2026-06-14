"""gateway.py — authorize a tool call against the unified policy.

``authorize(ctx, tool_name)`` is the decision function called at the
``execute_tool`` seam. It is pure apart from the Redis-backed sliding-window
rate counter, and degrades open if Redis is unavailable. Any internal error
fails open (the tool runs) and is logged — a broken gateway must never take
down the agent.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from security_gateway import policy
from security_gateway.context import CallerContext

logger = logging.getLogger(__name__)

# Decision labels (also used as the audit ``decision`` column value).
ALLOW = "allow"
DENY = "deny"
SHADOW_DENY = "shadow_deny"


@dataclass(frozen=True)
class Decision:
    """Outcome of an authorization check."""

    allowed: bool          # whether execute_tool should run the handler
    label: str             # allow | deny | shadow_deny  (audited)
    risk_class: str
    reason: str            # empty for plain allow
    mode: str              # shadow | enforce  (posture at decision time)
    rule: str              # which rule fired: none | interface | owner | rate | error

    @property
    def denied(self) -> bool:
        return not self.allowed


def _who(ctx: CallerContext) -> str:
    return ctx.agent_name or ctx.user_id or ("owner" if ctx.is_owner else "anon")


def _rl_key(ctx: CallerContext, rclass: str) -> str:
    return f"gw:rl:{ctx.interface}:{_who(ctx)}:{rclass}"


def _window_count(key: str, window_seconds: int, now: float) -> int:
    """Evict expired entries and return the count in the sliding window.

    Returns 0 (degrade open) if Redis is unavailable.
    """
    try:
        from redis_state import get_redis

        r = get_redis()
        if r is None:
            return 0
        r.zremrangebyscore(key, 0, now - window_seconds)
        return int(r.zcard(key) or 0)
    except Exception:
        return 0


def _window_consume(key: str, window_seconds: int, now: float) -> None:
    """Record one call in the sliding window. Best-effort; never raises."""
    try:
        from redis_state import get_redis

        r = get_redis()
        if r is None:
            return
        r.zadd(key, {f"{now:.6f}:{uuid.uuid4().hex}": now})
        r.expire(key, window_seconds + 5)
    except Exception:
        pass


def _authorize(ctx: CallerContext, tool_name: str, now: float) -> Decision:
    rclass = policy.risk_class(tool_name)
    mode = policy.enforce_mode()

    # Unknown risk class (e.g. a dynamic tool not in the taxonomy): never block.
    if rclass == policy.UNCATEGORIZED:
        return Decision(True, ALLOW, rclass, "uncategorized tool — not gated", mode, "none")

    # 1. Interface restriction — public web chat may only reach read-ish classes.
    #    Always enforced (the tool list is already pre-filtered to the same set).
    if ctx.interface == "webchat" and rclass not in policy.WEBCHAT_ALLOWED_RISK_CLASSES:
        return Decision(
            False, DENY, rclass,
            f"webchat is not permitted to call '{rclass}' tools", mode, "interface",
        )

    # 2. Owner-gating — high-impact classes restricted to the owner.
    #    Shadow by default; enforced only when mode == enforce.
    if rclass in policy.owner_required_classes() and not ctx.is_owner:
        if mode == policy.ENFORCE:
            return Decision(False, DENY, rclass, f"'{rclass}' is owner-only", mode, "owner")
        return Decision(True, SHADOW_DENY, rclass, f"'{rclass}' is owner-only (shadow)", mode, "owner")

    # 3. Rate limit — per (caller, risk class) sliding window.
    #    Shadow by default; enforced only when mode == enforce.
    rl = policy.rate_limit_for(rclass)
    if rl and rl.get("max_calls"):
        window = int(rl.get("window_seconds", 3600))
        max_calls = int(rl["max_calls"])
        key = _rl_key(ctx, rclass)
        over = _window_count(key, window, now) >= max_calls
        if over and mode == policy.ENFORCE:
            # Denied call does not consume a slot.
            return Decision(
                False, DENY, rclass,
                f"rate limit exceeded ({max_calls}/{window}s for '{rclass}')",
                mode, "rate",
            )
        _window_consume(key, window, now)
        if over:
            return Decision(
                True, SHADOW_DENY, rclass,
                f"rate limit exceeded ({max_calls}/{window}s for '{rclass}', shadow)",
                mode, "rate",
            )

    return Decision(True, ALLOW, rclass, "", mode, "none")


def authorize(ctx: CallerContext, tool_name: str, args: dict | None = None) -> Decision:
    """Authorize a tool call. Fails open (allow) on any internal gateway error."""
    now = time.time()
    try:
        return _authorize(ctx, tool_name, now)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("gateway.authorize failed open for %s: %s", tool_name, e)
        return Decision(
            True, ALLOW, policy.risk_class(tool_name),
            f"gateway error (fail-open): {e}", policy.enforce_mode(), "error",
        )
