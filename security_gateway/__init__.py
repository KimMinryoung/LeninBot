"""security_gateway — unified tool-call policy, authorization, and audit logging.

Every tool call from every interface funnels through ``tool_gateway.dispatcher.execute_tool``.
This package is the control plane mounted at that seam:

* ``policy``  — single source of truth for tool risk classes and per-caller rules.
* ``context`` — ``CallerContext`` carried via a contextvar across the async tool loop.
* ``gateway`` — ``authorize()`` turns (context, tool, args) into an allow/deny Decision.
* ``audit``   — append-only security audit log (Postgres + structured journal log).

Design invariant: authorization and dispatcher pre-check errors are **fail-closed**.
Audit sink failures remain non-fatal, and Redis-backed rate limits retain their
documented degrade-open behavior.
"""

from security_gateway.context import (
    CallerContext,
    caller_scope,
    get_caller,
    new_request_id,
    new_run_context,
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
    "new_request_id",
    "new_run_context",
    "set_caller",
    "reset_caller",
    "Decision",
    "authorize",
    "audit",
    "ensure_tool_audit_log_table",
    "policy",
]
