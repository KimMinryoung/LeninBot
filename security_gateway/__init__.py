"""security_gateway — unified tool-call policy, authorization, and audit logging.

Every tool call from every interface funnels through ``tool_loop_common.execute_tool``.
This package is the control plane mounted at that seam:

* ``policy``  — single source of truth for tool risk classes and per-caller rules.
* ``context`` — ``CallerContext`` carried via a contextvar across the async tool loop.
* ``gateway`` — ``authorize()`` turns (context, tool, args) into an allow/deny Decision.
* ``audit``   — append-only security audit log (Postgres + structured journal log).

Design invariant: the gateway is **fail-open on internal error**. A broken gateway
logs a warning and lets the tool run — it must never take down the agent.
"""

from security_gateway.context import (
    CallerContext,
    caller_scope,
    get_caller,
    set_caller,
    reset_caller,
)
from security_gateway.gateway import Decision, authorize
from security_gateway.audit import audit, ensure_tool_audit_log_table
from security_gateway import policy

__all__ = [
    "CallerContext",
    "caller_scope",
    "get_caller",
    "set_caller",
    "reset_caller",
    "Decision",
    "authorize",
    "audit",
    "ensure_tool_audit_log_table",
    "policy",
]
