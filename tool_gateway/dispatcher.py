"""Runtime tool dispatch facade.

The provider loops import dispatch helpers from here so the runtime has a
single named control plane for batching, execution, security authorization, and
audit. The implementation delegates to ``tool_loop_common`` for now; that keeps
behavior stable while moving callers to the gateway boundary.
"""

from __future__ import annotations

from tool_loop_common import (
    PARALLEL_SAFE_TOOLS,
    compact_tool_definitions,
    execute_tool,
    execute_tools_batch,
)

__all__ = [
    "PARALLEL_SAFE_TOOLS",
    "compact_tool_definitions",
    "execute_tool",
    "execute_tools_batch",
]
