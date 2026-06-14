#!/usr/bin/env python3
"""Operator CLI for the tool security gateway.

Subcommands:
  policy   — print the consolidated policy (risk classes, rules, enforce mode)
  check    — dry-run an authorization decision for a (interface, agent, tool)
  audit    — query the tool_audit_log table

Examples:
  venv/bin/python scripts/security_gateway.py policy
  venv/bin/python scripts/security_gateway.py check --interface webchat --tool send_email
  venv/bin/python scripts/security_gateway.py check --interface telegram --tool transfer_usdc --owner
  venv/bin/python scripts/security_gateway.py audit --since 24h --decision deny --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_since(value: str) -> str:
    """Turn '24h' / '90m' / '7d' into a Postgres interval string. Defaults to hours."""
    v = str(value).strip().lower()
    if v.endswith("d"):
        return f"{int(v[:-1])} days"
    if v.endswith("m"):
        return f"{int(v[:-1])} minutes"
    if v.endswith("h"):
        return f"{int(v[:-1])} hours"
    return f"{int(v)} hours"


def cmd_policy(_args) -> int:
    from security_gateway import policy

    policy.reset_caches()
    print(json.dumps(policy.describe(), indent=2, ensure_ascii=False))
    return 0


def cmd_check(args) -> int:
    from security_gateway import CallerContext, authorize, policy

    policy.reset_caches()
    ctx = CallerContext(
        interface=args.interface,
        agent_name=args.agent,
        user_id=args.user,
        is_owner=bool(args.owner),
    )
    decision = authorize(ctx, args.tool)
    print(json.dumps({
        "caller": ctx.label(),
        "interface": ctx.interface,
        "is_owner": ctx.is_owner,
        "tool": args.tool,
        "risk_class": decision.risk_class,
        "mode": decision.mode,
        "decision": decision.label,
        "allowed": decision.allowed,
        "rule": decision.rule,
        "reason": decision.reason,
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_audit(args) -> int:
    from db import query

    where = ["ts > now() - %s::interval"]
    params: list = [_parse_since(args.since)]
    if args.tool:
        where.append("tool_name = %s")
        params.append(args.tool)
    if args.interface:
        where.append("interface = %s")
        params.append(args.interface)
    if args.decision:
        where.append("decision = %s")
        params.append(args.decision)
    sql = (
        "SELECT ts, interface, agent_name, user_id, is_owner, task_id, tool_name, "
        "risk_class, decision, enforced, deny_reason, result_status, latency_ms, "
        "args_summary, error_excerpt "
        f"FROM tool_audit_log WHERE {' AND '.join(where)} "
        "ORDER BY ts DESC LIMIT %s"
    )
    params.append(int(args.limit))
    rows = query(sql, tuple(params))
    print(f"# {len(rows)} row(s)")
    for r in rows:
        print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Tool security gateway operator CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("policy", help="print the consolidated policy")

    c = sub.add_parser("check", help="dry-run an authorization decision")
    c.add_argument("--interface", required=True, help="telegram|webchat|agent|autonomous|a2a|...")
    c.add_argument("--agent", default=None)
    c.add_argument("--user", default=None)
    c.add_argument("--owner", action="store_true", help="treat caller as the owner")
    c.add_argument("--tool", required=True)

    a = sub.add_parser("audit", help="query the tool_audit_log table")
    a.add_argument("--since", default="24h", help="e.g. 90m, 24h, 7d")
    a.add_argument("--tool", default=None)
    a.add_argument("--interface", default=None)
    a.add_argument("--decision", default=None, help="allow|deny|shadow_deny")
    a.add_argument("--limit", default=100)

    args = p.parse_args()
    if args.cmd == "policy":
        return cmd_policy(args)
    if args.cmd == "check":
        return cmd_check(args)
    if args.cmd == "audit":
        return cmd_audit(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
