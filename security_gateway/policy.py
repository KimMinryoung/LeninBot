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
    "research_deep_dive": "delegate",  # autonomous tick: bounded read-only analyst sub-call
    "route_task": "delegate",
    "mission": "state",
    "save_finding": "state",
    "add_research_note": "state",
    "send_message": "state",
    "read_user_chat": "read",
    "read_messages": "read",
    "read_research_notes": "read",
    "read_document": "read",
    "revise_plan": "state",
    "set_project_state": "state",
    "list_agent_tools": "read",
    # Read / search / fetch
    "knowledge_graph_search": "read",
    "vector_search": "read",
    "web_search": "fetch",
    "research_web": "fetch",  # writer: light-agent web research delegation
    "fetch_url": "fetch",
    "wiki_search": "fetch",
    "wiki_get": "fetch",
    "fetch_x_post": "fetch",
    "convert_document": "fetch",
    "download_file": "fetch",
    "download_image": "fetch",
    "get_finance_data": "read",
    "read_self": "read",
    "recall_experience": "read",
    "search_manuscript": "read",
    "read_manuscript": "read",
    "read_document": "read",
    "search_documents": "read",
    "query_db": "read",
    "commulingo_people": "read",
    # Runner-local typed discovery terminal; validates through read-only duplicate lookup.
    "commulingo_candidate_select": "read",
    "check_inbox": "read",
    "check_wallet": "wallet_read",
    # Writes / publication
    "save_self_analysis": "write",
    "append_to_manuscript": "write",
    "replace_in_manuscript": "write",
    "save_document": "write",
    "write_kg": "write",
    "write_kg_structured": "write",
    "save_diary": "write",
    # write, not publish: every edit is transactional, revision-snapshotted and
    # reversible (unlike broadcast-style publish tools), so the publish rate
    # cap only throttled legitimate bulk curation (2026-07-11 incident: 11
    # section writes bounced mid-task).
    "commulingo_person_create": "write",
    "commulingo_person_update": "write",
    "commulingo_section_save": "write",
    "commulingo_event_link": "write",
    "commulingo_office_row_save": "write",
    "commulingo_term_create": "write",
    "research_document": "publish",
    "edit_content": "publish",
    "edit_public_post": "publish",
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
# Public webchat and A2A may only reach read-ish classes. These rules mirror
# their profile pre-filters and are always enforced as defense-in-depth.
PUBLIC_READONLY_ALLOWED_RISK_CLASSES = frozenset({"read", "fetch", "wallet_read"})
WEBCHAT_ALLOWED_RISK_CLASSES = PUBLIC_READONLY_ALLOWED_RISK_CLASSES
A2A_ALLOWED_RISK_CLASSES = PUBLIC_READONLY_ALLOWED_RISK_CLASSES

# Interfaces with no class restriction (the full orchestrator / agent surface).
# Per-tool allow-listing for these already happens upstream (orchestrator and
# per-agent allow-lists); the gateway adds owner-gating + rate limits + audit.
UNRESTRICTED_INTERFACES = frozenset({"telegram", "agent", "autonomous", "system"})

# ── Owner-gated classes (NEW — shadow by default) ─────────────────────
# Risk classes that should only run for the owner. Enforced only when the
# gateway is in "enforce" mode; in "shadow" mode a non-owner call is allowed
# but recorded as ``shadow_deny`` so we can see what enforcement would block.
OWNER_REQUIRED_RISK_CLASSES = frozenset({"pay", "send", "execute", "admin"})

# ── Owner-gated individual tools ──────────────────────────────────────
# Some tools need owner-gating without their whole risk class being owner-only.
# The CommuLingo narrow writers are 'write', a class that legitimately runs on
# several surfaces, but they edit the public dictionaries at /commulingo and
# only the curator lanes and the owner should reach them. Until now nothing at
# this layer said so: containment lived entirely in the per-surface tool lists,
# so one over-broad profile entry would have been enough to open them up. This
# mirrors the webchat/A2A class rules, which likewise duplicate a profile
# pre-filter on purpose.
#
# The curator lanes run as the owner (interface 'agent', is_owner true), so the
# same owner test covers them without naming the agent.
OWNER_REQUIRED_TOOLS = frozenset({
    "commulingo_person_create",
    "commulingo_person_update",
    "commulingo_section_save",
    "commulingo_event_link",
    "commulingo_office_row_save",
    "commulingo_term_create",
})

# ── Per-tool caller allow-lists ───────────────────────────────────────
# The owner flag alone does not contain the CommuLingo writers. The roleplay
# bot runs on an allow-listed private channel and so declares is_owner, which is
# correct for what it is but would let the personas edit the dictionaries. This
# names the callers instead. Matched against agent_name, falling back to the
# interface when no agent name is set, which is how the Telegram orchestrator
# appears.
#
# The list is what has actually written, taken from tool_audit_log over the
# whole history of the table: the curation lanes, the analyst on research
# follow-ups, and the orchestrator itself. Nothing else has ever written.
COMMULINGO_WRITE_CALLERS = frozenset({"commulingo_curator", "analyst", "telegram"})

TOOL_CALLER_ALLOWLIST: dict[str, frozenset[str]] = {
    tool: COMMULINGO_WRITE_CALLERS for tool in OWNER_REQUIRED_TOOLS
}

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


def owner_required_tools() -> frozenset[str]:
    overlay = _load_overlay().get("owner_required_tools")
    if isinstance(overlay, list):
        return frozenset(str(x) for x in overlay)
    return OWNER_REQUIRED_TOOLS


def caller_allowlist_for(tool_name: str) -> frozenset[str] | None:
    """Callers permitted to run ``tool_name``, or None when unrestricted."""
    overlay = _load_overlay().get("tool_caller_allowlist")
    if isinstance(overlay, dict):
        entry = overlay.get(tool_name)
        if isinstance(entry, list):
            return frozenset(str(x) for x in entry)
        if tool_name in overlay:
            return None
    return TOOL_CALLER_ALLOWLIST.get(tool_name)


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
        "a2a_allowed_risk_classes": sorted(A2A_ALLOWED_RISK_CLASSES),
        "unrestricted_interfaces": sorted(UNRESTRICTED_INTERFACES),
        "owner_required_risk_classes": sorted(owner_required_classes()),
        "owner_required_tools": sorted(owner_required_tools()),
        "tool_caller_allowlist": {
            tool: sorted(caller_allowlist_for(tool) or ())
            for tool in sorted(TOOL_CALLER_ALLOWLIST)
        },
        "rate_limits": rate_limits(),
        "overlay_path": _OVERLAY_PATH if _load_overlay() else None,
    }
