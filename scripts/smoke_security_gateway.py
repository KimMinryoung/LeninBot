#!/usr/bin/env python3
"""Smoke checks for the tool security gateway (policy + authorize + audit).

Pure in-process checks — no DB or Redis required (rate-limit checks degrade
open when Redis is absent, which we assert). Run:

    venv/bin/python scripts/smoke_security_gateway.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def _force_mode(mode: str):
    """Pin enforce_mode() without touching config.json."""
    from security_gateway import policy

    policy.reset_caches()
    policy._mode_cache = (1e18, mode)  # far-future TTL so it sticks for the test


def main() -> int:
    from security_gateway import CallerContext, authorize, policy
    from security_gateway.audit import redact_args

    print("== registry: every tool has a risk class ==")
    from runtime_tools.registry import TOOLS

    uncategorized = sorted(
        t["name"] for t in TOOLS
        if t.get("name") and policy.risk_class(t["name"]) == policy.UNCATEGORIZED
    )
    check("no uncategorized tools in registry", not uncategorized, str(uncategorized))

    print("== webchat interface restriction (always enforced) ==")
    wc = CallerContext(interface="webchat", is_owner=False)
    for tool in ("send_email", "execute_python", "transfer_usdc", "write_file"):
        d = authorize(wc, tool)
        check(f"webchat denies {tool}", d.denied and d.label == "deny", f"{d.label}/{d.rule}")
    for tool in ("vector_search", "fetch_url", "read_self", "check_wallet"):
        d = authorize(wc, tool)
        check(f"webchat allows {tool}", d.allowed and d.label == "allow", f"{d.label}/{d.reason}")

    print("== owner-gating: shadow vs enforce ==")
    # Unique user ids per run so rate-limit keys never collide / accumulate.
    uid = uuid.uuid4().hex[:8]
    nonowner_agent = CallerContext(interface="agent", agent_name=f"scout-{uid}", is_owner=False)
    owner = CallerContext(interface="telegram", user_id=f"owner-{uid}", is_owner=True)

    _force_mode(policy.SHADOW)
    d = authorize(nonowner_agent, "send_email")
    check("shadow: non-owner send_email allowed but shadow_deny",
          d.allowed and d.label == "shadow_deny" and d.rule == "owner", f"{d.label}/{d.rule}")

    _force_mode(policy.ENFORCE)
    d = authorize(nonowner_agent, "send_email")
    check("enforce: non-owner send_email denied",
          d.denied and d.label == "deny" and d.rule == "owner", f"{d.label}/{d.rule}")
    d = authorize(owner, "send_email")
    check("enforce: owner send_email allowed", d.allowed and d.label == "allow", f"{d.label}/{d.reason}")

    print("== rate limiting (enforce) ==")
    from redis_state import redis_available

    _force_mode(policy.ENFORCE)
    # 'pay' is limited to 5/hour. Use a fresh caller so the window starts empty.
    pay_ctx = CallerContext(interface="telegram", user_id=f"payer-{uuid.uuid4().hex[:8]}", is_owner=True)
    labels = [authorize(pay_ctx, "transfer_usdc").label for _ in range(7)]
    if redis_available():
        first5_ok = all(x == "allow" for x in labels[:5])
        rest_denied = all(x == "deny" for x in labels[5:])
        check("rate limit: first 5 pay allowed, rest denied", first5_ok and rest_denied, str(labels))
    else:
        check("rate limit fails open without Redis", all(x == "allow" for x in labels), str(labels))

    print("== uncategorized tool is never blocked ==")
    _force_mode(policy.ENFORCE)
    d = authorize(wc, "some_unknown_future_tool")
    check("unknown tool allowed (fail-open on taxonomy)", d.allowed, f"{d.label}/{d.rule}")

    print("== redaction drops secret-looking args ==")
    summary = redact_args({
        "query": "lenin",
        "api_key": "sk-secret-123",
        "password": "hunter2",
        "private_key": "0xdead",
        "url": "https://example.com",
    })
    check("api_key redacted", "sk-secret-123" not in summary, summary)
    check("password redacted", "hunter2" not in summary, summary)
    check("private_key redacted", "0xdead" not in summary, summary)
    check("non-secret kept", "lenin" in summary, summary)

    print("== gateway fails open on internal error ==")
    # Passing a context missing attributes shouldn't raise; authorize catches all.
    class Broken:
        interface = "telegram"
        # deliberately missing is_owner etc. to trip an AttributeError inside
        agent_name = None
        user_id = None
    d = authorize(Broken(), "send_email")  # type: ignore[arg-type]
    check("authorize never raises (fails open)", d.allowed, f"{d.label}/{d.rule}")

    policy.reset_caches()
    print(f"\n== RESULT: {_PASS} passed, {_FAIL} failed ==")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
