"""Client of the CommuLingo admin MCP that the frontend owns.

CommuLingo is a separate service: leninbot reads and edits its data only through
this MCP (frontend dev_docs/commulingo-admin-mcp.md), never through the
frontend container or its tables. Stateless Streamable HTTP: one JSON-RPC
``tools/call`` per POST, authenticated with the ``leninbot`` client token.
"""
from __future__ import annotations

import http.client
import itertools
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from secrets_loader import get_secret

DEFAULT_URL = "http://127.0.0.1:3100/mcp"
TOKEN_FILE = Path.home() / ".config" / "commulingo-mcp" / "leninbot.token"
# The MCP port is closed for a few seconds while a frontend deploy swaps the
# container; a refused connection never reached the server, so retrying is
# safe even for writes.
REFUSED_RETRIES = (1, 2, 4, 8)
# A connection reset while the new container is still starting is retried too,
# except for writes that carry no idempotency key.
NOT_REPEATABLE = frozenset({"editorial_store", "people_upsert"})
_ids = itertools.count(1)


class CommuLingoUnavailable(RuntimeError):
    """The MCP could not be reached or answered outside the protocol."""


class CommuLingoToolError(ValueError):
    """The tool ran and rejected the request (validation, conflict, not found)."""

    def __init__(self, payload: dict):
        self.status = payload.get("status")
        self.code = payload.get("code")
        self.current_revision = payload.get("currentRevision")
        self.payload = payload
        # Same "<code or status>: <message>" shape the old private RPC raised,
        # which callers match on (e.g. 'revision_conflict' in str(exc)).
        super().__init__(f"{self.code or self.status}: {payload.get('error')}")


def _token() -> str:
    token = get_secret("COMMULINGO_MCP_TOKEN")
    if not token and TOKEN_FILE.is_file():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise CommuLingoUnavailable(f"CommuLingo MCP token missing (COMMULINGO_MCP_TOKEN or {TOKEN_FILE})")
    return token


def _refused(exc: urllib.error.URLError) -> bool:
    return isinstance(getattr(exc, "reason", None), ConnectionRefusedError)


def _reset(exc: Exception) -> bool:
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, (ConnectionResetError, http.client.RemoteDisconnected))


def call_tool(name: str, arguments: dict | None = None, *, timeout: float = 120) -> dict:
    """Call one MCP tool and return its structured result."""
    body = json.dumps({"jsonrpc": "2.0", "id": next(_ids), "method": "tools/call",
                       "params": {"name": name, "arguments": arguments or {}}}, ensure_ascii=False).encode()
    request = urllib.request.Request(
        os.environ.get("COMMULINGO_MCP_URL", DEFAULT_URL), data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                 "Authorization": f"Bearer {_token()}"})
    for delay in (*REFUSED_RETRIES, None):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                message = json.loads(response.read())
            break
        except urllib.error.HTTPError as exc:
            raise CommuLingoUnavailable(f"CommuLingo MCP HTTP {exc.code}; no fallback write was attempted") from exc
        except urllib.error.URLError as exc:
            if delay is None or not (_refused(exc) or (_reset(exc) and name not in NOT_REPEATABLE)):
                raise CommuLingoUnavailable(f"CommuLingo MCP unreachable ({exc.reason}); no fallback write was attempted") from exc
            time.sleep(delay)
        except (ConnectionResetError, http.client.RemoteDisconnected) as exc:
            if delay is None or name in NOT_REPEATABLE:
                raise CommuLingoUnavailable(f"CommuLingo MCP connection reset ({exc}); no fallback write was attempted") from exc
            time.sleep(delay)
        except (ValueError, OSError) as exc:
            raise CommuLingoUnavailable(f"CommuLingo MCP answered outside the protocol ({type(exc).__name__}); "
                                        "no fallback write was attempted") from exc
    if "error" in message:
        raise CommuLingoUnavailable(f"CommuLingo MCP {name}: {message['error'].get('message')}")
    result = message.get("result") or {}
    if result.get("isError"):
        raise CommuLingoToolError(result.get("structuredContent") or {"error": "tool error", "status": 500})
    return result.get("structuredContent") or {}


def read_entry(target: str, entry_id: str):
    """Editorial state of a person (also for person_section) or term; None when absent."""
    tool, key = ("term_get", "term") if target == "term" else ("person_get", "person")
    try:
        return call_tool(tool, {"id": entry_id})[key]
    except CommuLingoToolError as exc:
        if exc.status == 404:
            return None
        raise
