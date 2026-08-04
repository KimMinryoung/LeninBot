#!/usr/bin/env python3
"""Operate the LLM gateway: policy status, spend reports, recent calls.

  venv/bin/python scripts/llm_gateway_cli.py status
  venv/bin/python scripts/llm_gateway_cli.py spend --days 7
  venv/bin/python scripts/llm_gateway_cli.py tail -n 30
  venv/bin/python scripts/llm_gateway_cli.py set enforce true
  venv/bin/python scripts/llm_gateway_cli.py set daily_budget_usd 25

DB-backed commands read llm_audit_log and need DB credentials (read-only is
enough; see dev_docs/secret_management.md for ad-hoc access).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.gateway import CONFIG_PATH, _DEFAULTS, load_policy  # noqa: E402

_SETTABLE = set(_DEFAULTS)


def cmd_status(_args) -> int:
    policy = load_policy()
    print("policy:", json.dumps(policy, ensure_ascii=False, indent=2))
    try:
        from db import query

        rows = query(
            """SELECT COALESCE(provider, '?') AS provider,
                      COUNT(*) AS calls,
                      COALESCE(SUM(cost_usd), 0)::float AS spend
                 FROM llm_audit_log
                WHERE ts >= date_trunc('day', now() AT TIME ZONE 'utc')
                GROUP BY 1 ORDER BY spend DESC""",
        )
    except Exception as e:
        print(f"(today's spend unavailable: {e})")
        return 0
    total = sum(r["spend"] for r in rows or [])
    print(f"\ntoday (UTC): ${total:.4f} total")
    for r in rows or []:
        print(f"  {r['provider']:<10} {r['calls']:>5} calls  ${r['spend']:.4f}")
    return 0


def cmd_spend(args) -> int:
    from db import query

    rows = query(
        """SELECT date_trunc('day', ts AT TIME ZONE 'utc')::date AS day,
                  COALESCE(provider, '?') AS provider,
                  COALESCE(caller, '?') AS caller,
                  COUNT(*) AS calls,
                  COALESCE(SUM(cost_usd), 0)::float AS spend
             FROM llm_audit_log
            WHERE ts >= now() - make_interval(days => %(days)s)
            GROUP BY 1, 2, 3 ORDER BY 1 DESC, spend DESC""",
        {"days": int(args.days)},
    )
    day = None
    for r in rows or []:
        if r["day"] != day:
            day = r["day"]
            print(f"\n{day}")
        print(f"  {r['provider']:<10} {r['caller']:<28} {r['calls']:>5} calls  ${r['spend']:.4f}")
    if not rows:
        print("(no rows)")
    return 0


def cmd_tail(args) -> int:
    from db import query

    rows = query(
        """SELECT ts, surface, caller, provider, model, tokens_in, tokens_out,
                  cache_read, cost_usd::float AS cost_usd, latency_ms, status, error_excerpt
             FROM llm_audit_log ORDER BY id DESC LIMIT %(n)s""",
        {"n": int(args.n)},
    )
    for r in reversed(rows or []):
        cost = f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "$?"
        line = (
            f"{r['ts']:%m-%d %H:%M:%S} [{r['surface']}] {r['caller'] or '?'} "
            f"{r['provider'] or '?'}/{r['model'] or '?'} "
            f"in={r['tokens_in'] or 0} out={r['tokens_out'] or 0} "
            f"cached={r['cache_read'] or 0} {cost} {r['status']}"
        )
        if r["error_excerpt"]:
            line += f" — {r['error_excerpt'][:120]}"
        print(line)
    if not rows:
        print("(no rows)")
    return 0


def cmd_set(args) -> int:
    if args.key not in _SETTABLE:
        print(f"unknown key {args.key!r}; settable: {', '.join(sorted(_SETTABLE))}")
        return 1
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value  # bare strings allowed
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    raw[args.key] = value
    merged = {**_DEFAULTS, **raw}
    CONFIG_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{args.key} = {json.dumps(value, ensure_ascii=False)} (hot-reloaded on next call)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p_spend = sub.add_parser("spend")
    p_spend.add_argument("--days", type=int, default=7)
    p_tail = sub.add_parser("tail")
    p_tail.add_argument("-n", type=int, default=20)
    p_set = sub.add_parser("set")
    p_set.add_argument("key")
    p_set.add_argument("value")
    args = parser.parse_args()
    return {"status": cmd_status, "spend": cmd_spend, "tail": cmd_tail, "set": cmd_set}[
        args.cmd
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
