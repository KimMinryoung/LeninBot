#!/usr/bin/env python3
"""commulingo_suggestions.py — review queue for staged CommuLingo edits.

Used when config/commulingo_people.json has direct_apply=false, so agent
edits land as pending rows in commulingo_agent_suggestions instead of being
applied. Listing and review go through the CommuLingo admin MCP
(commulingo/mcp_client.py); approval applies the stored patch with the same
editorial service the direct mode uses, so behavior is identical.

Usage (from repo root; the MCP token is ~/.config/commulingo-mcp/leninbot.token
or COMMULINGO_MCP_TOKEN):

  venv/bin/python scripts/commulingo_suggestions.py list [--status pending|approved|rejected|all]
  venv/bin/python scripts/commulingo_suggestions.py show <id>
  venv/bin/python scripts/commulingo_suggestions.py approve <id> --note "..."
  venv/bin/python scripts/commulingo_suggestions.py reject <id> --note "..."
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from commulingo.mcp_client import call_tool
from commulingo.person_service import call_person_service


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _suggestions(**arguments) -> list[dict]:
    rows, offset = [], 0
    while True:
        page = call_tool("suggestions_list", {**arguments, "limit": 100, "offset": offset})
        rows += page["items"]
        offset += len(page["items"])
        if not page["items"] or offset >= page["total"]:
            return rows


def cmd_list(status: str) -> int:
    rows = sorted(_suggestions(status=status), key=lambda r: int(r["id"]))
    if not rows:
        print(f"no suggestions (status={status})")
        return 0
    for r in rows:
        print(
            f"#{r['id']:<4} {r['status']:<10} {r['action']:<6} {r['target_type']:<10} "
            f"{r['target_id']:<24} conf={r['confidence'] or '-':<6} "
            f"created={str(r['created_at'])[:16].replace('T', ' ')}"
        )
    return 0


def _fetch(sid: int) -> dict | None:
    items = call_tool("suggestions_list", {"id": str(sid), "includePatch": True, "limit": 1})["items"]
    return items[0] if items else None


def cmd_show(sid: int) -> int:
    row = _fetch(sid)
    if not row:
        print(f"suggestion #{sid} not found")
        return 1
    print(_dumps(row))
    return 0


def cmd_review(sid: int, approve: bool, note: str) -> int:
    row = _fetch(sid)
    if not row:
        print(f"suggestion #{sid} not found")
        return 1
    if row["status"] != "pending":
        print(f"suggestion #{sid} is already {row['status']} (reviewer: {row['reviewer']})")
        return 1
    try:
        result = call_person_service({"command": "review", "target": row["target_type"], "suggestionId": str(sid),
                                      "approve": approve, "note": note, "changedBy": f"agent-suggestion:{sid}"})
    except ValueError as exc:
        print(f"cannot review: {exc}")
        return 1
    print(_dumps(result))
    print(f"suggestion #{sid} {result.get('status')}")
    if result.get("status") == "approved":
        print("live on cyber-lenin.com/commulingo within ~1 minute (server cache TTL).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("--status", default="pending", choices=["pending", "approved", "rejected", "all"])
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
