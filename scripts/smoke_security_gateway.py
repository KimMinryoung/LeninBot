#!/usr/bin/env python3
"""Smoke checks for the tool security gateway (policy + authorize + audit).

Pure in-process checks — no DB or Redis required (rate-limit checks degrade
open when Redis is absent, which we assert). Run:

    venv/bin/python scripts/smoke_security_gateway.py
"""

from __future__ import annotations

import sys
import uuid
import asyncio
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
    from security_gateway.audit import _IMMUTABILITY_DDL, redact_args

    print("== registry: every tool has a risk class ==")
    from runtime_tools.registry import TOOLS

    uncategorized = sorted(
        t["name"] for t in TOOLS
        if t.get("name") and policy.risk_class(t["name"]) == policy.UNCATEGORIZED
    )
    check("no uncategorized tools in registry", not uncategorized, str(uncategorized))
    for dynamic_tool, expected_class in {
        "send_message": "state",
        "read_user_chat": "read",
        "read_messages": "read",
        "edit_public_post": "publish",
    }.items():
        check(
            f"dynamic tool classified: {dynamic_tool}",
            policy.risk_class(dynamic_tool) == expected_class,
            policy.risk_class(dynamic_tool),
        )

    print("== webchat interface restriction (always enforced) ==")
    wc = CallerContext(interface="webchat", is_owner=False)
    for tool in ("send_email", "execute_python", "transfer_usdc", "write_file"):
        d = authorize(wc, tool)
        check(f"webchat denies {tool}", d.denied and d.label == "deny", f"{d.label}/{d.rule}")
    for tool in ("vector_search", "fetch_url", "read_self", "check_wallet"):
        d = authorize(wc, tool)
        check(f"webchat allows {tool}", d.allowed and d.label == "allow", f"{d.label}/{d.reason}")

    print("== public A2A interface restriction (always enforced) ==")
    a2a = CallerContext(interface="a2a", is_owner=False)
    for tool in ("write_kg_structured", "send_email", "research_document", "transfer_usdc"):
        d = authorize(a2a, tool)
        check(f"a2a denies {tool}", d.denied and d.label == "deny", f"{d.label}/{d.rule}")
    for tool in ("knowledge_graph_search", "vector_search", "web_search", "fetch_url"):
        d = authorize(a2a, tool)
        check(f"a2a allows {tool}", d.allowed and d.label == "allow", f"{d.label}/{d.reason}")

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
    # Read the configured 'pay' cap so the test tracks policy, not a literal.
    cap = policy.rate_limit_for("pay")["max_calls"]
    pay_ctx = CallerContext(interface="telegram", user_id=f"payer-{uuid.uuid4().hex[:8]}", is_owner=True)
    labels = [authorize(pay_ctx, "transfer_usdc").label for _ in range(cap + 2)]
    if redis_available():
        first_ok = all(x == "allow" for x in labels[:cap])
        rest_denied = all(x == "deny" for x in labels[cap:])
        check(f"rate limit: first {cap} pay allowed, rest denied", first_ok and rest_denied, str(labels))
    else:
        check("rate limit fails closed without Redis", all(x == "deny" for x in labels), str(labels))

    # execute and admin are intentionally uncapped.
    for uncapped in ("execute", "admin"):
        check(f"{uncapped} has no rate limit", policy.rate_limit_for(uncapped) is None,
              str(policy.rate_limit_for(uncapped)))

    print("== uncategorized tool fails closed ==")
    _force_mode(policy.ENFORCE)
    d = authorize(wc, "some_unknown_future_tool")
    check(
        "unknown tool denied until classified",
        d.denied and d.rule == "taxonomy",
        f"{d.label}/{d.rule}",
    )

    print("== atomic rate-store boundary ==")
    import security_gateway.gateway as gateway_module

    original_window = gateway_module._window_consume_atomic
    window_calls = []
    try:
        def fake_window(*args, **kwargs):
            window_calls.append((args, kwargs))
            return gateway_module.RateWindowResult(True, over=False, consumed=True, count=1)

        gateway_module._window_consume_atomic = fake_window
        probe_ctx = CallerContext(
            interface="telegram", user_id=f"probe-{uuid.uuid4().hex[:8]}", is_owner=True
        )
        preflight = authorize(
            probe_ctx, "transfer_usdc", consume_rate_limit=False
        )
        check(
            "preflight authorization does not consume a slot",
            preflight.allowed and not window_calls,
            str(window_calls),
        )
        authorize(probe_ctx, "transfer_usdc")
        check("rate decision uses one atomic store call", len(window_calls) == 1, str(window_calls))

        gateway_module._window_consume_atomic = lambda *_a, **_kw: gateway_module.RateWindowResult(
            False, error="redis down"
        )
        unavailable = authorize(probe_ctx, "transfer_usdc")
        check(
            "capped side effect fails closed on rate-store outage",
            unavailable.denied and unavailable.rule == "rate_store",
            f"{unavailable.label}/{unavailable.rule}",
        )
    finally:
        gateway_module._window_consume_atomic = original_window

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

    print("== audit table immutability DDL ==")
    check("blocks update/delete", "BEFORE UPDATE OR DELETE ON tool_audit_log" in _IMMUTABILITY_DDL)
    check("blocks truncate", "BEFORE TRUNCATE ON tool_audit_log" in _IMMUTABILITY_DDL)
    check("requires explicit admin approval setting", "leninbot.audit_log_mutation_approved" in _IMMUTABILITY_DDL)

    print("== gateway fails closed on internal error ==")
    # Passing a context missing attributes shouldn't raise; authorize denies.
    class Broken:
        interface = "telegram"
        # deliberately missing is_owner etc. to trip an AttributeError inside
        agent_name = None
        user_id = None
    d = authorize(Broken(), "send_email")  # type: ignore[arg-type]
    check(
        "authorize never raises and denies",
        d.denied and d.label == "deny" and d.rule == "error",
        f"{d.label}/{d.rule}",
    )
    original_risk_class = policy.risk_class
    original_enforce_mode = policy.enforce_mode
    try:
        policy.risk_class = lambda _name: (_ for _ in ()).throw(RuntimeError("risk down"))
        policy.enforce_mode = lambda: (_ for _ in ()).throw(RuntimeError("mode down"))
        d = authorize(wc, "send_email")
        check(
            "policy helper failure still returns deny",
            d.denied and d.rule == "error" and d.mode == policy.ENFORCE,
            f"{d.label}/{d.rule}/{d.mode}",
        )
    finally:
        policy.risk_class = original_risk_class
        policy.enforce_mode = original_enforce_mode

    print("== dispatcher fail-closed and run-local idempotency ==")

    async def _dispatcher_checks() -> None:
        import tool_gateway.security as gateway_adapter
        from tool_gateway.dispatcher import execute_tool
        from tool_gateway.security import CallerContext, caller_scope

        original_authorize = gateway_adapter.authorize
        calls = {"write": 0, "read": 0, "blocked": 0}

        def write_handler(content: str) -> str:
            calls["write"] += 1
            return f"saved:{content}"

        def read_handler(query: str) -> str:
            calls["read"] += 1
            return f"read:{query}:{calls['read']}"

        def blocked_handler() -> str:
            calls["blocked"] += 1
            return "must-not-run"

        with caller_scope(CallerContext(interface="system", is_owner=True)):
            cache: dict[str, tuple[str, bool]] = {}
            first = await execute_tool(
                "save_diary",
                {"content": "same"},
                {"save_diary": write_handler},
                idempotency_cache=cache,
            )
            second = await execute_tool(
                "save_diary",
                {"content": "same"},
                {"save_diary": write_handler},
                idempotency_cache=cache,
            )
            check(
                "duplicate write executes once",
                calls["write"] == 1 and first == second == ("saved:same", False),
                str((calls, first, second)),
            )

            await execute_tool(
                "vector_search",
                {"query": "same"},
                {"vector_search": read_handler},
                idempotency_cache=cache,
            )
            await execute_tool(
                "vector_search",
                {"query": "same"},
                {"vector_search": read_handler},
                idempotency_cache=cache,
            )
            check("duplicate reads still execute", calls["read"] == 2, str(calls))

            def broken_authorize(*_args, **_kwargs):
                raise RuntimeError("policy unavailable")

            gateway_adapter.authorize = broken_authorize
            blocked = await execute_tool(
                "save_diary",
                {},
                {"save_diary": blocked_handler},
            )
            check(
                "dispatcher blocks pre-check exception",
                blocked[1] and calls["blocked"] == 0 and "pre-check failed" in blocked[0],
                str((calls, blocked)),
            )
        gateway_adapter.authorize = original_authorize

        print("== dispatcher schema validation and durable idempotency ==")
        schema_calls = {"count": 0}

        def schema_handler(content: str, mode: str = "append") -> str:
            schema_calls["count"] += 1
            return f"{mode}:{content}"

        schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["append"], "default": "append"},
            },
            "required": ["content"],
        }
        with caller_scope(CallerContext(interface="system", is_owner=True)):
            rejected = await execute_tool(
                "save_diary",
                {"content": "x", "recpient": "typo"},
                {"save_diary": schema_handler},
                tool_schema=schema,
            )
            check(
                "unknown schema argument is rejected, not dropped",
                rejected[1] and schema_calls["count"] == 0 and "arguments rejected" in rejected[0],
                str((rejected, schema_calls)),
            )
            cache = {}
            first = await execute_tool(
                "save_diary",
                {"content": "normalized"},
                {"save_diary": schema_handler},
                tool_schema=schema,
                idempotency_cache=cache,
            )
            second = await execute_tool(
                "save_diary",
                {"content": "normalized", "mode": "append"},
                {"save_diary": schema_handler},
                tool_schema=schema,
                idempotency_cache=cache,
            )
            check(
                "schema defaults normalize the idempotency key",
                first == second == ("append:normalized", False) and schema_calls["count"] == 1,
                str((first, second, schema_calls)),
            )

            invalid_amount = await execute_tool(
                "transfer_usdc",
                {"to_address": "0x" + "1" * 40, "amount_usdc": -1},
                {"transfer_usdc": lambda to_address, amount_usdc: "must-not-run"},
                tool_schema={
                    "type": "object",
                    "properties": {
                        "to_address": {"type": "string"},
                        "amount_usdc": {"type": "number"},
                    },
                    "required": ["to_address", "amount_usdc"],
                },
            )
            check("non-positive payment amount rejected", invalid_amount[1], str(invalid_amount))

            nonfinite = await execute_tool(
                "transfer_usdc",
                {"to_address": "0x" + "1" * 40, "amount_usdc": 1, "meta": {"x": float("nan")}},
                {"transfer_usdc": lambda **kwargs: "must-not-run"},
                tool_schema={
                    "type": "object",
                    "properties": {
                        "to_address": {"type": "string"},
                        "amount_usdc": {"type": "number"},
                        "meta": {"type": "object"},
                    },
                    "required": ["to_address", "amount_usdc"],
                },
            )
            check("nested non-finite JSON number rejected", nonfinite[1], str(nonfinite))

        import security_gateway.idempotency as idem

        originals = (idem.lookup, idem.reserve, idem.mark_succeeded, idem.mark_outcome_unknown)
        records = {}
        durable_calls = {"ok": 0, "boom": 0}
        try:
            def fake_lookup(scope, name, args):
                return records.get((scope, name, str(sorted(args.items()))))

            def fake_reserve(scope, name, args, *, tool_call_id):
                key = (scope, name, str(sorted(args.items())))
                record = idem.IdempotencyRecord(
                    key_hash=str(key), scope=scope, tool_name=name, status="running",
                    reservation_token="token", tool_call_id=tool_call_id, acquired=True,
                )
                records[key] = record
                return record

            def fake_success(record, result):
                for existing_key, existing in list(records.items()):
                    if existing is record:
                        records[existing_key] = idem.IdempotencyRecord(
                            key_hash=record.key_hash, scope=record.scope, tool_name=record.tool_name,
                            status="succeeded", reservation_token=record.reservation_token,
                            result_text=result, tool_call_id=record.tool_call_id,
                        )

            def fake_unknown(record, result):
                for existing_key, existing in list(records.items()):
                    if existing is record:
                        records[existing_key] = idem.IdempotencyRecord(
                            key_hash=record.key_hash, scope=record.scope, tool_name=record.tool_name,
                            status="outcome_unknown", reservation_token=record.reservation_token,
                            result_text=result, is_error=True, tool_call_id=record.tool_call_id,
                        )

            idem.lookup = fake_lookup
            idem.reserve = fake_reserve
            idem.mark_succeeded = fake_success
            idem.mark_outcome_unknown = fake_unknown

            def durable_ok(content: str) -> str:
                durable_calls["ok"] += 1
                return f"saved:{content}"

            with caller_scope(CallerContext(
                interface="agent", agent_name="smoke", is_owner=True, task_id="durable-1"
            )):
                one = await execute_tool(
                    "save_diary", {"content": "durable"}, {"save_diary": durable_ok},
                    tool_call_id="call-1",
                )
                two = await execute_tool(
                    "save_diary", {"content": "durable"}, {"save_diary": durable_ok},
                    tool_call_id="call-2",
                )
                check(
                    "durable scoped retry reuses prior success",
                    one == two == ("saved:durable", False) and durable_calls["ok"] == 1,
                    str((one, two, durable_calls, records)),
                )

                def durable_boom(content: str) -> str:
                    durable_calls["boom"] += 1
                    raise TimeoutError("response lost")

                failed = await execute_tool(
                    "save_diary", {"content": "boom"}, {"save_diary": durable_boom},
                    tool_call_id="call-3",
                )
                retried = await execute_tool(
                    "save_diary", {"content": "boom"}, {"save_diary": durable_boom},
                    tool_call_id="call-4",
                )
                check(
                    "ambiguous handler failure is not retried",
                    failed[1] and retried[1] and durable_calls["boom"] == 1
                    and "outcome_unknown" in retried[0],
                    str((failed, retried, durable_calls)),
                )
        finally:
            idem.lookup, idem.reserve, idem.mark_succeeded, idem.mark_outcome_unknown = originals

    asyncio.run(_dispatcher_checks())

    policy.reset_caches()
    print(f"\n== RESULT: {_PASS} passed, {_FAIL} failed ==")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
