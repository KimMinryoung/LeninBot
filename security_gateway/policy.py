"""policy.py — single source of truth for tool-call security policy.

Consolidates the rules that were previously scattered across
``scripts/smoke_tool_allowlists.py`` (risk classes, web-forbidden classes),
``runtime_tools/allowlists.py`` (orchestrator allow-list), and the implicit
owner/admin gating in ``web_chat.py`` / ``telegram/bot.py``.

The values below are safe baked-in defaults. An optional
``config/security_policy.json`` overlay can tune rate limits and the
owner-required set without a code change. Enforcement posture is read from the
mutable runtime config (``gateway_enforce_mode``), so a deployment can run new
rules in shadow first and flip to enforce later.
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# ── Risk taxonomy ─────────────────────────────────────────────────────
# Maps every registered tool to a coarse risk class. Moved here from
# scripts/smoke_tool_allowlists.py, which now imports it from this module.
TOOL_RISK_CLASS: dict[str, str] = {
    # Coordination / routing
    "delegate": "delegate",
    "multi_delegate": "delegate",
    "run_agent": "delegate",
    "route_task": "delegate",
    "mission": "state",
    "save_finding": "state",
    "add_research_note": "state",
    "read_research_notes": "read",
    "read_document": "read",
    "revise_plan": "state",
    "set_project_state": "state",
    "list_agent_tools": "read",
    # Read / search / fetch
    "knowledge_graph_search": "read",
    "vector_search": "read",
    "web_search": "fetch",
    "fetch_url": "fetch",
    "fetch_x_post": "fetch",
    "convert_document": "fetch",
    "download_file": "fetch",
    "download_image": "fetch",
    "get_finance_data": "read",
    "read_self": "read",
    "recall_experience": "read",
    "query_db": "read",
    "check_inbox": "read",
    "check_wallet": "wallet_read",
    # Writes / publication
    "save_self_analysis": "write",
    "write_kg": "write",
    "write_kg_structured": "write",
    "save_diary": "write",
    "research_document": "publish",
    "edit_content": "publish",
    "publish_hub_curation": "publish",
    "publish_static_page": "publish",
    "publish_static_page_translation": "publish",
    "publish_comic": "publish",
    "broadcast_to_channel": "send",
    "send_email": "send",
    "a2a_send": "send",
    "allowlist_sender": "send",
    # Files / code / browser / media
    "read_file": "file_read",
    "list_directory": "file_read",
    "search_files": "file_read",
    "write_file": "file_write",
    "patch_file": "file_write",
    "execute_python": "execute",
    "restart_service": "execute",
    "browse_web": "browser",
    "generate_image": "media",
    "upload_to_r2": "publish",
    # Wallet / payment
    "pay_and_fetch": "pay",
    "swap_eth_to_usdc": "pay",
    "transfer_usdc": "pay",
    # External platform integrations
    "mersoom": "send",
    "moltbook": "send",
    "kg_admin": "admin",
}

UNCATEGORIZED = "uncategorized"


def risk_class(tool_name: str) -> str:
    """Return the risk class for a tool, or ``"uncategorized"`` if unknown."""
    return TOOL_RISK_CLASS.get(tool_name, UNCATEGORIZED)


# ── Per-interface access rules ────────────────────────────────────────
# Public web chat may only reach read-ish classes. This mirrors the existing
# WEB_ALLOWED/WEB_FORBIDDEN sets and is **always enforced** because the tool
# list is already pre-filtered to the same set — enforcing here changes nothing
# observable, it only adds defense-in-depth + an audit trail.
WEBCHAT_ALLOWED_RISK_CLASSES = frozenset({"read", "fetch", "wallet_read"})

# Interfaces with no class restriction (the full orchestrator / agent surface).
# Per-tool allow-listing for these already happens upstream (orchestrator and
# per-agent allow-lists); the gateway adds owner-gating + rate limits + audit.
UNRESTRICTED_INTERFACES = frozenset({"telegram", "agent", "autonomous", "system"})

# ── Owner-gated classes (NEW — shadow by default) ─────────────────────
# Risk classes that should only run for the owner. Enforced only when the
# gateway is in "enforce" mode; in "shadow" mode a non-owner call is allowed
# but recorded as ``shadow_deny`` so we can see what enforcement would block.
OWNER_REQUIRED_RISK_CLASSES = frozenset({"pay", "send", "execute", "admin"})

# ── Rate limits (NEW — shadow by default) ─────────────────────────────
# Per (caller, risk_class) sliding-window caps. window_seconds + max_calls.
# Absent class => unlimited.
#
# Only outbound/irreversible side-effect classes are capped: pay (funds leave),
# send (messages go out), publish (content goes public). execute and admin are
# intentionally NOT capped — their risk depends on the payload, not the call
# count, and legitimate bulk runs are common; throttling by count would just
# break normal work without adding safety.
DEFAULT_RATE_LIMITS: dict[str, dict[str, int]] = {
    "pay": {"window_seconds": 3600, "max_calls": 3},
    "send": {"window_seconds": 3600, "max_calls": 20},
    "publish": {"window_seconds": 3600, "max_calls": 20},
}

# ── Config overlay ────────────────────────────────────────────────────
_OVERLAY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "security_policy.json",
)
_overlay_cache: dict | None = None
_overlay_mtime: float = -1.0


def _load_overlay() -> dict:
    """Load config/security_policy.json if present, mtime-cached. Never raises."""
    global _overlay_cache, _overlay_mtime
    try:
        mtime = os.path.getmtime(_OVERLAY_PATH)
    except OSError:
        _overlay_cache = {}
        _overlay_mtime = -1.0
        return _overlay_cache
    if _overlay_cache is not None and mtime == _overlay_mtime:
        return _overlay_cache
    try:
        with open(_OVERLAY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _overlay_cache = data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("security_policy.json load failed, using defaults: %s", e)
        _overlay_cache = {}
    _overlay_mtime = mtime
    return _overlay_cache


def owner_required_classes() -> frozenset[str]:
    overlay = _load_overlay().get("owner_required_risk_classes")
    if isinstance(overlay, list):
        return frozenset(str(x) for x in overlay)
    return OWNER_REQUIRED_RISK_CLASSES


def rate_limits() -> dict[str, dict[str, int]]:
    overlay = _load_overlay().get("rate_limits")
    if isinstance(overlay, dict):
        merged = dict(DEFAULT_RATE_LIMITS)
        for cls, spec in overlay.items():
            if isinstance(spec, dict):
                merged[str(cls)] = {
                    "window_seconds": int(spec.get("window_seconds", 3600)),
                    "max_calls": int(spec.get("max_calls", 0)),
                }
        return merged
    return dict(DEFAULT_RATE_LIMITS)


def rate_limit_for(rclass: str) -> dict[str, int] | None:
    return rate_limits().get(rclass)


# ── Enforcement posture ───────────────────────────────────────────────
ENFORCE = "enforce"
SHADOW = "shadow"
_mode_cache: tuple[float, str] | None = None
_MODE_TTL_SECONDS = 30.0


def enforce_mode(monotonic=time.monotonic) -> str:
    """Return the current enforcement mode ("shadow" | "enforce").

    Read from the mutable runtime config key ``gateway_enforce_mode`` with a
    short TTL cache so operators can flip shadow→enforce without restarting
    (the value is re-read at most once per ``_MODE_TTL_SECONDS``).
    """
    global _mode_cache
    now = monotonic()
    if _mode_cache is not None and (now - _mode_cache[0]) < _MODE_TTL_SECONDS:
        return _mode_cache[1]
    mode = SHADOW
    try:
        import bot_config

        raw = str(bot_config.get_gateway_enforce_mode()).strip().lower()
        mode = ENFORCE if raw == ENFORCE else SHADOW
    except Exception:
        mode = SHADOW
    _mode_cache = (now, mode)
    return mode


def reset_caches() -> None:
    """Drop cached overlay/mode state. Used by tests and the CLI."""
    global _overlay_cache, _overlay_mtime, _mode_cache
    _overlay_cache = None
    _overlay_mtime = -1.0
    _mode_cache = None


def describe() -> dict:
    """Return a JSON-serializable snapshot of the active policy (for the CLI)."""
    by_class: dict[str, list[str]] = {}
    for tool, cls in sorted(TOOL_RISK_CLASS.items()):
        by_class.setdefault(cls, []).append(tool)
    return {
        "enforce_mode": enforce_mode(),
        "tools_by_risk_class": by_class,
        "webchat_allowed_risk_classes": sorted(WEBCHAT_ALLOWED_RISK_CLASSES),
        "unrestricted_interfaces": sorted(UNRESTRICTED_INTERFACES),
        "owner_required_risk_classes": sorted(owner_required_classes()),
        "rate_limits": rate_limits(),
        "overlay_path": _OVERLAY_PATH if _load_overlay() else None,
    }
