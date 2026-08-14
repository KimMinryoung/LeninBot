#!/usr/bin/env python3
"""Operate the LLM gateway: policy status, spend reports, recent calls.

  venv/bin/python scripts/llm_gateway_cli.py status
  venv/bin/python scripts/llm_gateway_cli.py spend --days 7
  venv/bin/python scripts/llm_gateway_cli.py tail -n 30
  venv/bin/python scripts/llm_gateway_cli.py set enforce true
  venv/bin/python scripts/llm_gateway_cli.py unset enforce
  venv/bin/python scripts/llm_gateway_cli.py set daily_budget_usd 25

DB-backed commands read llm_audit_log and need DB credentials (read-only is
enough; see dev_docs/secret_management.md for ad-hoc access).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.gateway import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    LOCAL_CONFIG_PATH,
    _DEFAULTS,
    load_policy,
)

_SETTABLE = set(_DEFAULTS)


def cmd_status(_args) -> int:
    policy = load_policy()
    print(f"defaults: {DEFAULT_CONFIG_PATH}")
    print(f"local overrides: {LOCAL_CONFIG_PATH}")
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
    denials = query(
        """SELECT status, COUNT(*) AS n
             FROM llm_audit_log
            WHERE ts >= date_trunc('day', now() AT TIME ZONE 'utc')
              AND status IN ('denied', 'would_deny')
            GROUP BY 1""",
    )
    for r in denials or []:
        print(f"  !! {r['status']}: {r['n']} call(s) today")
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
    day_total = 0.0
    total = 0.0

    def _close_day():
        if day is not None:
            print(f"  {'':<10} {'= day total':<28} {'':>11}  ${day_total:.4f}")

    for r in rows or []:
        if r["day"] != day:
            _close_day()
            day = r["day"]
            day_total = 0.0
            print(f"\n{day}")
        day_total += r["spend"]
        total += r["spend"]
        print(f"  {r['provider']:<10} {r['caller']:<28} {r['calls']:>5} calls  ${r['spend']:.4f}")
    _close_day()
    if not rows:
        print("(no rows)")
    else:
        print(f"\n{args.days}-day total: ${total:.4f}")
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


def _read_local_overrides() -> dict:
    if not LOCAL_CONFIG_PATH.exists():
        return {}
    data = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("local config top-level JSON must be an object")
    return data


def _write_local_overrides(data: dict) -> None:
    LOCAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{LOCAL_CONFIG_PATH.name}.",
        suffix=".tmp",
        dir=str(LOCAL_CONFIG_PATH.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o664)
        os.replace(tmp_path, LOCAL_CONFIG_PATH)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def cmd_set(args) -> int:
    if args.key not in _SETTABLE:
        print(f"unknown key {args.key!r}; settable: {', '.join(sorted(_SETTABLE))}")
        return 1
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value  # bare strings allowed
    raw = _read_local_overrides()
    raw[args.key] = value
    _write_local_overrides(raw)
    print(f"{args.key} = {json.dumps(value, ensure_ascii=False)} (hot-reloaded on next call)")
    return 0


def cmd_unset(args) -> int:
    if args.key not in _SETTABLE:
        print(f"unknown key {args.key!r}; settable: {', '.join(sorted(_SETTABLE))}")
        return 1
    raw = _read_local_overrides()
    existed = args.key in raw
    raw.pop(args.key, None)
    _write_local_overrides(raw)
    state = "removed" if existed else "already inherited"
    print(f"{args.key}: local override {state}; tracked default applies on next call")
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
    p_unset = sub.add_parser("unset")
    p_unset.add_argument("key")
    args = parser.parse_args()
    return {
        "status": cmd_status,
        "spend": cmd_spend,
        "tail": cmd_tail,
        "set": cmd_set,
        "unset": cmd_unset,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
