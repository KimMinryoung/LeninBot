#!/usr/bin/env python3
"""commulingo_suggestions.py — review queue for staged CommuLingo people edits.

Used when config/commulingo_people.json has direct_apply=false, so agent
edits land as pending rows in commulingo_agent_suggestions instead of being
applied. Approving reuses the exact apply path the direct mode uses
(runtime_tools.commulingo_people.apply_edit), so behavior is identical.

Usage (from repo root; needs DB_PASSWORD — see scripts/run_writer_tests.sh
for the credential-loading pattern, or run inside a service credential env):

  venv/bin/python scripts/commulingo_suggestions.py list [--status pending|all]
  venv/bin/python scripts/commulingo_suggestions.py show <id>
  venv/bin/python scripts/commulingo_suggestions.py approve <id> [--note "..."]
  venv/bin/python scripts/commulingo_suggestions.py reject <id> --note "..."
"""

import argparse
import getpass
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from psycopg2.extras import RealDictCursor

from db import get_conn
from runtime_tools.commulingo_people import apply_edit, _validate, _dumps


def _reviewer() -> str:
    return getpass.getuser() or "operator"


def cmd_list(status: str) -> int:
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """SELECT id, target_type, target_id, action, status, confidence,
                      suggested_by, reviewer, created_at, reviewed_at
               FROM commulingo_agent_suggestions
               WHERE (%s = 'all' OR status = %s)
               ORDER BY id""",
            (status, status),
        )
        rows = cur.fetchall()
    if not rows:
        print(f"no suggestions (status={status})")
        return 0
    for r in rows:
        print(
            f"#{r['id']:<4} {r['status']:<10} {r['action']:<6} {r['target_type']:<10} "
            f"{r['target_id']:<24} conf={r['confidence'] or '-':<6} "
            f"created={r['created_at']:%Y-%m-%d %H:%M}"
        )
    return 0


def _fetch(cur, sid: int) -> dict | None:
    cur.execute("SELECT * FROM commulingo_agent_suggestions WHERE id = %s", (sid,))
    return cur.fetchone()


def cmd_show(sid: int) -> int:
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        row = _fetch(cur, sid)
    if not row:
        print(f"suggestion #{sid} not found")
        return 1
    print(_dumps(dict(row)))
    return 0


def cmd_review(sid: int, approve: bool, note: str) -> int:
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        row = _fetch(cur, sid)
        if not row:
            print(f"suggestion #{sid} not found")
            return 1
        if row["status"] != "pending":
            print(f"suggestion #{sid} is already {row['status']} (reviewer: {row['reviewer']})")
            return 1
        if approve:
            patch = row["patch_json"] or {}
            error = _validate(cur, row["target_type"], row["action"], row["target_id"], patch)
            if error:
                print(f"cannot apply: {error}")
                print("reject it, or fix the data and retry.")
                return 1
            summary = apply_edit(
                cur, row["target_type"], row["action"], row["target_id"], patch,
                changed_by=f"agent-suggestion:{sid}",
            )
            print(f"applied: {summary}")
        cur.execute(
            """UPDATE commulingo_agent_suggestions
               SET status = %s, reviewer = %s, review_note = %s, reviewed_at = NOW()
               WHERE id = %s""",
            ("approved" if approve else "rejected", _reviewer(), note, sid),
        )
    print(f"suggestion #{sid} {'approved' if approve else 'rejected'}")
    if approve:
        print("live on cyber-lenin.com/commulingo/people within ~1 minute (server cache TTL).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("--status", default="pending")
    p_show = sub.add_parser("show")
    p_show.add_argument("id", type=int)
    p_approve = sub.add_parser("approve")
    p_approve.add_argument("id", type=int)
    p_approve.add_argument("--note", default="")
    p_reject = sub.add_parser("reject")
    p_reject.add_argument("id", type=int)
    p_reject.add_argument("--note", default="")
    args = parser.parse_args()

    if args.cmd == "list":
        return cmd_list(args.status)
    if args.cmd == "show":
        return cmd_show(args.id)
    if args.cmd == "approve":
        return cmd_review(args.id, approve=True, note=args.note)
    return cmd_review(args.id, approve=False, note=args.note or "rejected")


if __name__ == "__main__":
    sys.exit(main())
