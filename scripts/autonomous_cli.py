#!/usr/bin/env python3
"""scripts/autonomous_cli.py — Manage autonomous projects.

Usage:
  python scripts/autonomous_cli.py create --title "..." --topic "..." --goal "..."
  python scripts/autonomous_cli.py list
  python scripts/autonomous_cli.py show <project_id>
  python scripts/autonomous_cli.py events <project_id> [--limit N]
  python scripts/autonomous_cli.py edit <project_id> [--title T] [--topic X] [--goal G | --goal-file F]
  python scripts/autonomous_cli.py advise <project_id> [--message "…" | --file path]
  python scripts/autonomous_cli.py advisories <project_id>
  python scripts/autonomous_cli.py pause <project_id>
  python scripts/autonomous_cli.py resume <project_id>
  python scripts/autonomous_cli.py archive <project_id>
  python scripts/autonomous_cli.py tick               # fire one tick right now (for testing)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# Allow running as `python scripts/autonomous_cli.py` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import execute as db_execute, query as db_query, query_one as db_query_one  # noqa: E402
from autonomous_project import (  # noqa: E402
    _ensure_tables,
    _log_event,
    run_tick,
    STATE_RESEARCHING,
    STATE_PAUSED,
    STATE_ARCHIVED,
)


def _cmd_create(args: argparse.Namespace) -> int:
    _ensure_tables()
    row = db_query_one(
        """
        INSERT INTO autonomous_projects(title, topic, goal, state)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (args.title, args.topic, args.goal, STATE_RESEARCHING),
    )
    pid = row["id"]
    _log_event(pid, "project_created", f"title={args.title}", {"topic": args.topic})
    print(f"Created project #{pid}: {args.title}")
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    _ensure_tables()
    rows = db_query(
        """
        SELECT id, title, state, turn_count, last_run_at, created_at
          FROM autonomous_projects
         ORDER BY state, id
        """,
    )
    if not rows:
        print("(no projects)")
        return 0
    for r in rows:
        last = r["last_run_at"].strftime("%Y-%m-%d %H:%M") if r["last_run_at"] else "never"
        print(f"#{r['id']}  [{r['state']:<11}]  turns={r['turn_count']:<4}  last={last}  {r['title']}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    _ensure_tables()
    row = db_query_one(
        "SELECT * FROM autonomous_projects WHERE id = %s", (args.project_id,)
    )
    if not row:
        print(f"Project #{args.project_id} not found")
        return 1
    print(f"#{row['id']}  {row['title']}")
    print(f"  state: {row['state']}   turns: {row['turn_count']}   last_run: {row['last_run_at']}")
    print(f"  topic: {row['topic']}")
    print(f"  goal:\n    {row['goal']}")
    print(f"  plan:\n{json.dumps(row['plan'], indent=2, ensure_ascii=False)}")
    notes = row["research_notes"] or []
    print(f"  research_notes: {len(notes)} entries")
    for n in notes[-5:]:
        src = ", ".join((n.get("sources") or [])[:3])
        print(f"    - turn {n.get('turn')}: {(n.get('text') or '')[:300]}" + (f"  [{src}]" if src else ""))
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    _ensure_tables()
    rows = db_query(
        """
        SELECT id, event_type, content, meta, created_at
          FROM autonomous_project_events
         WHERE project_id = %s
         ORDER BY created_at DESC
         LIMIT %s
        """,
        (args.project_id, args.limit),
    )
    if not rows:
        print("(no events)")
        return 0
    for r in rows:
        ts = r["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {r['event_type']}: {(r['content'] or '')[:300]}")
    return 0


def _set_state(project_id: int, target: str, reason: str) -> int:
    _ensure_tables()
    cur = db_query_one("SELECT state FROM autonomous_projects WHERE id = %s", (project_id,))
    if not cur:
        print(f"Project #{project_id} not found")
        return 1
    db_execute(
        "UPDATE autonomous_projects SET state = %s, updated_at = NOW() WHERE id = %s",
        (target, project_id),
    )
    _log_event(project_id, "state_transition", reason, {"from": cur["state"], "to": target, "via": "cli"})
    print(f"Project #{project_id}: {cur['state']} → {target}")
    return 0


def _cmd_pause(args: argparse.Namespace) -> int:
    return _set_state(args.project_id, STATE_PAUSED, "paused via CLI")


def _cmd_resume(args: argparse.Namespace) -> int:
    return _set_state(args.project_id, STATE_RESEARCHING, "resumed via CLI")


def _cmd_archive(args: argparse.Namespace) -> int:
    return _set_state(args.project_id, STATE_ARCHIVED, "archived via CLI")


def _cmd_edit(args: argparse.Namespace) -> int:
    """Update title / topic / goal. Each is optional; unspecified stays put.

    Goal is often multi-line; pass via --goal-file PATH or stdin (-) to avoid
    shell quoting pain. Logs a `project_edited` event with before/after so the
    audit trail captures the redirection.
    """
    _ensure_tables()
    cur = db_query_one("SELECT * FROM autonomous_projects WHERE id = %s", (args.project_id,))
    if not cur:
        print(f"Project #{args.project_id} not found")
        return 1

    updates: dict[str, str] = {}
    before: dict[str, str] = {}

    if args.title is not None:
        updates["title"] = args.title
        before["title"] = cur["title"]
    if args.topic is not None:
        updates["topic"] = args.topic
        before["topic"] = cur["topic"]

    new_goal = None
    if args.goal_file:
        if args.goal_file == "-":
            new_goal = sys.stdin.read()
        else:
            with open(args.goal_file, "r", encoding="utf-8") as f:
                new_goal = f.read()
    elif args.goal is not None:
        new_goal = args.goal
    if new_goal is not None:
        new_goal = new_goal.strip()
        if not new_goal:
            print("error: goal cannot be empty")
            return 1
        updates["goal"] = new_goal
        before["goal"] = cur["goal"]

    if not updates:
        print("error: specify at least one of --title / --topic / --goal / --goal-file")
        return 1

    set_clause = ", ".join(f"{k} = %s" for k in updates) + ", updated_at = NOW()"
    params = list(updates.values()) + [args.project_id]
    db_execute(f"UPDATE autonomous_projects SET {set_clause} WHERE id = %s", params)

    _log_event(
        args.project_id, "project_edited",
        f"fields changed: {', '.join(updates.keys())}",
        {"before": before, "after": updates, "via": "cli"},
    )
    print(f"Project #{args.project_id} updated: {', '.join(updates.keys())}")
    print("(next tick will read the new values automatically)")
    return 0


def _cmd_advise(args: argparse.Namespace) -> int:
    """Leave an operator advisory for the next tick to read.

    The advisory appears in the agent's prompt as an <operator-advice> block above
    the plan, and is marked consumed after the tick succeeds. Multiple advisories
    queue up in time order if left before the next tick fires.
    """
    _ensure_tables()
    cur = db_query_one("SELECT state FROM autonomous_projects WHERE id = %s", (args.project_id,))
    if not cur:
        print(f"Project #{args.project_id} not found")
        return 1

    if args.file:
        if args.file == "-":
            content = sys.stdin.read()
        else:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()
    elif args.message is not None:
        content = args.message
    else:
        print("error: specify a --message or --file")
        return 1

    content = content.strip()
    if not content:
        print("error: advisory cannot be empty")
        return 1

    row = db_query_one(
        "INSERT INTO autonomous_project_advisories(project_id, content) VALUES (%s, %s) RETURNING id, created_at",
        (args.project_id, content),
    )
    _log_event(args.project_id, "advisory_created",
               f"advisory #{row['id']} created ({len(content)} chars)",
               {"advisory_id": row["id"]})
    print(f"Advisory #{row['id']} queued for project #{args.project_id}")
    print(f"  created_at: {row['created_at']}")
    print(f"  length: {len(content)} chars")
    print(f"  (agent will read this at the next tick; state = '{cur['state']}')")
    return 0


def _cmd_advisories(args: argparse.Namespace) -> int:
    """List advisories for a project — pending (unread) + recent consumed."""
    _ensure_tables()
    rows = db_query(
        """SELECT id, content, created_at, consumed_at FROM autonomous_project_advisories
           WHERE project_id = %s ORDER BY created_at DESC LIMIT %s""",
        (args.project_id, args.limit),
    )
    if not rows:
        print("(no advisories)")
        return 0
    pending = [r for r in rows if r["consumed_at"] is None]
    consumed = [r for r in rows if r["consumed_at"] is not None]
    if pending:
        print(f"=== PENDING ({len(pending)}) — will be shown to next tick ===")
        for r in pending:
            ts = r["created_at"].strftime("%Y-%m-%d %H:%M")
            print(f"#{r['id']}  {ts}")
            print("  " + r["content"][:500].replace("\n", "\n  "))
            if len(r["content"]) > 500:
                print(f"  ... ({len(r['content']) - 500} chars more)")
            print()
    if consumed:
        print(f"=== CONSUMED ({len(consumed)}) — already read by an earlier tick ===")
        for r in consumed:
            ts = r["created_at"].strftime("%Y-%m-%d %H:%M")
            ts_c = r["consumed_at"].strftime("%Y-%m-%d %H:%M")
            print(f"#{r['id']}  created={ts}  consumed={ts_c}")
            print("  " + r["content"][:300].replace("\n", "\n  "))
            if len(r["content"]) > 300:
                print("  ...")
            print()
    return 0


def _cmd_tick(_args: argparse.Namespace) -> int:
    result = run_tick()
    if not result:
        return 0
    print(f"\n=== Tick result for project #{result['project_id']} ===")
    print(f"cost: ${result['cost_usd']:.4f}   rounds: {result['rounds_used']}")
    print(f"\n--- result text ---\n{result['result_text']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Manage Cyber-Lenin autonomous projects.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("create", help="Create a new project")
    pc.add_argument("--title", required=True)
    pc.add_argument("--topic", required=True, help="Short description of the research subject")
    pc.add_argument("--goal", required=True, help="This project's directive: what we want to accomplish + success criteria")
    pc.set_defaults(func=_cmd_create)

    pl = sub.add_parser("list", help="List all projects")
    pl.set_defaults(func=_cmd_list)

    ps = sub.add_parser("show", help="Show full details of one project")
    ps.add_argument("project_id", type=int)
    ps.set_defaults(func=_cmd_show)

    pe = sub.add_parser("events", help="Show event log for a project")
    pe.add_argument("project_id", type=int)
    pe.add_argument("--limit", type=int, default=30)
    pe.set_defaults(func=_cmd_events)

    pp = sub.add_parser("pause", help="Pause a project")
    pp.add_argument("project_id", type=int)
    pp.set_defaults(func=_cmd_pause)

    pr = sub.add_parser("resume", help="Resume a paused project")
    pr.add_argument("project_id", type=int)
    pr.set_defaults(func=_cmd_resume)

    pa = sub.add_parser("archive", help="Archive a project")
    pa.add_argument("project_id", type=int)
    pa.set_defaults(func=_cmd_archive)

    ped = sub.add_parser("edit", help="Change a project's title / topic / goal mid-flight")
    ped.add_argument("project_id", type=int)
    ped.add_argument("--title")
    ped.add_argument("--topic")
    ped.add_argument("--goal", help="Inline goal string. For multi-line use --goal-file instead.")
    ped.add_argument("--goal-file", help="Path to a file containing the new goal, or '-' for stdin.")
    ped.set_defaults(func=_cmd_edit)

    pad = sub.add_parser("advise", help="Leave an operator advisory for the next tick")
    pad.add_argument("project_id", type=int)
    pad.add_argument("--message", "-m", help="Advisory text (inline). Use --file for multi-line.")
    pad.add_argument("--file", "-f", help="Path to file containing the advisory, or '-' for stdin.")
    pad.set_defaults(func=_cmd_advise)

    pav = sub.add_parser("advisories", help="List pending + recent consumed advisories")
    pav.add_argument("project_id", type=int)
    pav.add_argument("--limit", type=int, default=20)
    pav.set_defaults(func=_cmd_advisories)

    pt = sub.add_parser("tick", help="Run one tick now (manual test)")
    pt.set_defaults(func=_cmd_tick)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
