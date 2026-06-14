"""CallerContext — who is invoking a tool, carried across the async tool loop.

The interface boundary (Telegram handler, web chat, agent runner, autonomous
worker) builds a ``CallerContext`` and hands it to ``chat_with_tools``, which
installs it in a ``contextvars.ContextVar`` for the duration of the run. The
gateway seam inside ``execute_tool`` reads it back via :func:`get_caller`.

ContextVars are snapshotted into the child tasks created by ``asyncio.gather``,
so the existing parallel-batch path in ``execute_tools_batch`` sees the caller
without any extra threading.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, replace

# Known interfaces. "unknown" is the fail-open default for call sites that have
# not yet been annotated — those calls are still authorized and audited.
INTERFACE_TELEGRAM = "telegram"
INTERFACE_WEBCHAT = "webchat"
INTERFACE_AGENT = "agent"
INTERFACE_AUTONOMOUS = "autonomous"
INTERFACE_SYSTEM = "system"
INTERFACE_UNKNOWN = "unknown"

KNOWN_INTERFACES = frozenset({
    INTERFACE_TELEGRAM,
    INTERFACE_WEBCHAT,
    INTERFACE_AGENT,
    INTERFACE_AUTONOMOUS,
    INTERFACE_SYSTEM,
    INTERFACE_UNKNOWN,
})


@dataclass(frozen=True)
class CallerContext:
    """Identity of the actor making a tool call."""

    interface: str = INTERFACE_UNKNOWN
    agent_name: str | None = None
    user_id: str | None = None
    is_owner: bool = False
    task_id: str | None = None
    session_id: str | None = None

    def with_agent(self, agent_name: str | None) -> "CallerContext":
        """Return a copy scoped to a delegated agent (interface stays the same)."""
        if not agent_name:
            return self
        return replace(self, agent_name=agent_name)

    def label(self) -> str:
        """Short human/log label, e.g. ``telegram:owner`` or ``agent:scout``."""
        who = self.agent_name or self.user_id or ("owner" if self.is_owner else "anon")
        return f"{self.interface}:{who}"


# Default caller when nothing has been installed. Fail-open interface, not owner.
_DEFAULT = CallerContext(interface=INTERFACE_UNKNOWN, is_owner=False)

current_caller: contextvars.ContextVar[CallerContext] = contextvars.ContextVar(
    "current_caller", default=_DEFAULT,
)


def get_caller() -> CallerContext:
    """Return the active CallerContext, or the fail-open default if none set."""
    try:
        return current_caller.get()
    except LookupError:  # pragma: no cover - default makes this unreachable
        return _DEFAULT


def set_caller(ctx: CallerContext) -> contextvars.Token:
    """Install ``ctx`` as the active caller. Returns a token for :func:`reset_caller`."""
    return current_caller.set(ctx)


def reset_caller(token: contextvars.Token) -> None:
    """Restore the caller to what it was before the matching :func:`set_caller`."""
    try:
        current_caller.reset(token)
    except (ValueError, LookupError):
        pass


@contextlib.contextmanager
def caller_scope(ctx: CallerContext):
    """Install ``ctx`` for the duration of the ``with`` block, then restore.

    Safe to wrap an ``await`` of a tool loop: the contextvar stays set across
    the loop (and is snapshotted into its ``asyncio.gather`` children), and is
    restored on exit even if the loop raises. Correctly nests — a delegated
    agent's scope restores the orchestrator's caller when it exits.
    """
    token = set_caller(ctx)
    try:
        yield ctx
    finally:
        reset_caller(token)
