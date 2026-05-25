#!/usr/bin/env python3
"""scripts/autonomous_cli.py — Manage autonomous projects.

Usage:
  python scripts/autonomous_cli.py create --title "..." --topic "..." --goal "..."
  python scripts/autonomous_cli.py status [--json]
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
import subprocess
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



def _systemctl_state(unit: str) -> str:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", unit],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception as exc:
        return f"unavailable ({exc.__class__.__name__})"
    value = (result.stdout or result.stderr or "").strip()
    return value or f"unknown ({result.returncode})"


def _systemctl_show(unit: str, *properties: str) -> dict[str, str]:
    try:
        cmd = ["systemctl", "show", unit, "--no-pager"]
        for prop in properties:
            cmd.append(f"--property={prop}")
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception as exc:
        return {"error": f"unavailable ({exc.__class__.__name__})"}
    data: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value.strip()
    if not data and result.returncode != 0:
        data["error"] = (result.stderr or "").strip() or f"unknown ({result.returncode})"
    return data


def _cmd_status(args: argparse.Namespace) -> int:
    """Show scheduler/runtime status without requiring DB credentials."""
    from bot_config import _config, _load_config, get_current_model_selection

    cfg = _load_config()
    previous_config = dict(_config)
    try:
        _config.clear()
        _config.update(cfg)
        model = get_current_model_selection("autonomous")
    finally:
        _config.clear()
        _config.update(previous_config)

    active = bool(cfg.get("autonomous_active", True))
    status = {
        "autonomous_active": active,
        "provider": model.get("provider"),
        "model_tier": model.get("tier"),
        "model_id": model.get("model_id"),
        "effective_state": (
            "enabled; next timer/manual tick can advance a due project"
            if active
            else "paused by config; timer wakes will skip run_tick"
        ),
    }

    if not getattr(args, "no_systemd", False):
        timer_state = _systemctl_state("leninbot-autonomous.timer")
        service_state = _systemctl_state("leninbot-autonomous.service")
        timer_info = _systemctl_show(
            "leninbot-autonomous.timer",
            "NextElapseUSecRealtime",
            "LastTriggerUSec",
        )
        service_info = _systemctl_show(
            "leninbot-autonomous.service",
            "Result",
            "ExecMainStatus",
            "InactiveEnterTimestamp",
        )
        status["timer"] = {"state": timer_state}
        if timer_info.get("error"):
            status["timer"]["detail"] = timer_info["error"]
        else:
            status["timer"]["next"] = timer_info.get("NextElapseUSecRealtime") or "n/a"
            status["timer"]["last"] = timer_info.get("LastTriggerUSec") or "n/a"
        status["service"] = {"state": service_state}
        if service_info.get("error"):
            status["service"]["detail"] = service_info["error"]
        else:
            status["service"]["result"] = service_info.get("Result") or "n/a"
            status["service"]["exit_status"] = service_info.get("ExecMainStatus") or "n/a"
            status["service"]["last_exit"] = service_info.get("InactiveEnterTimestamp") or "n/a"

    if getattr(args, "json", False):
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    print(f"autonomous_active: {str(active).lower()}")
    print(f"provider: {status['provider']}")
    print(f"model: {status['model_tier']} ({status['model_id']})")
    print(f"effective_state: {status['effective_state']}")

    timer = status.get("timer")
    if timer:
        print(f"timer: {timer['state']}")
        if timer.get("detail"):
            print(f"timer_detail: {timer['detail']}")
        else:
            print(f"timer_next: {timer.get('next') or 'n/a'}")
            print(f"timer_last: {timer.get('last') or 'n/a'}")
    service = status.get("service")
    if service:
        print(f"service: {service['state']}")
        if service.get("detail"):
            print(f"service_detail: {service['detail']}")
        else:
            print(f"service_result: {service.get('result') or 'n/a'} (exit={service.get('exit_status') or 'n/a'})")
            print(f"service_last_exit: {service.get('last_exit') or 'n/a'}")
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    _ensure_tables()
    rows = db_query(
        """
        SELECT p.id, p.title, p.state, p.turn_count, p.last_run_at, p.created_at,
               COALESCE(a.pending_advisories, 0) AS pending_advisories,
               e.event_type AS last_event_type,
               e.created_at AS last_event_at
          FROM autonomous_projects p
          LEFT JOIN LATERAL (
              SELECT COUNT(*)::int AS pending_advisories
                FROM autonomous_project_advisories adv
               WHERE adv.project_id = p.id
                 AND adv.consumed_at IS NULL
          ) a ON TRUE
          LEFT JOIN LATERAL (
              SELECT event_type, created_at
                FROM autonomous_project_events ev
               WHERE ev.project_id = p.id
               ORDER BY ev.created_at DESC, ev.id DESC
               LIMIT 1
          ) e ON TRUE
         ORDER BY CASE p.state WHEN 'researching' THEN 0 WHEN 'planning' THEN 0
                               WHEN 'paused' THEN 1 ELSE 2 END,
                  p.id
        """,
    )
    if not rows:
        print("(no projects)")
        return 0
    for r in rows:
        last = r["last_run_at"].strftime("%Y-%m-%d %H:%M") if r["last_run_at"] else "never"
        bits = [f"turns={r['turn_count']}", f"last={last}"]
        pending = int(r.get("pending_advisories") or 0)
        if pending:
            bits.append(f"advice={pending}")
        if r.get("last_event_type"):
            event_at = r["last_event_at"].strftime("%Y-%m-%d %H:%M") if hasattr(r.get("last_event_at"), "strftime") else str(r.get("last_event_at") or "?")
            bits.append(f"event={r['last_event_type']}@{event_at}")
        print(f"#{r['id']}  [{r['state']:<11}]  {'  '.join(bits)}  {r['title']}")
    return 0


def _coerce_note_sources(sources) -> list[str]:
    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except Exception:
            sources = [sources]
    if sources is None:
        return []
    if not isinstance(sources, list):
        sources = [sources]
    return [str(source) for source in sources]


def _recent_project_notes(project_id: int, *, legacy_notes: list, limit: int = 5) -> tuple[list[dict], int, str]:
    try:
        count_row = db_query_one(
            "SELECT COUNT(*) AS count FROM autonomous_project_notes WHERE project_id = %s",
            (project_id,),
        )
        rows = db_query(
            """
            SELECT turn, text, sources, created_at
              FROM autonomous_project_notes
             WHERE project_id = %s
             ORDER BY created_at DESC, id DESC
             LIMIT %s
            """,
            (project_id, limit),
        )
        return (
            list(reversed([dict(row) for row in rows])),
            int((count_row or {}).get("count") or 0),
            "autonomous_project_notes",
        )
    except Exception:
        notes = legacy_notes or []
        return notes[-limit:], len(notes), "legacy JSONB"


def _latest_project_event(project_id: int, event_type: str) -> dict | None:
    try:
        rows = db_query(
            """
            SELECT content, meta, created_at
              FROM autonomous_project_events
             WHERE project_id = %s
               AND event_type = %s
             ORDER BY created_at DESC, id DESC
             LIMIT 1
            """,
            (project_id, event_type),
        )
    except Exception:
        return None
    return dict(rows[0]) if rows else None


def _recent_project_advisories(project_id: int, limit: int = 10) -> list[dict]:
    try:
        rows = db_query(
            """
            SELECT id, content, created_at, consumed_at
              FROM autonomous_project_advisories
             WHERE project_id = %s
             ORDER BY created_at DESC, id DESC
             LIMIT %s
            """,
            (project_id, limit),
        )
    except Exception:
        return []
    return [dict(row) for row in rows]


def _format_tick_log_header(row: dict) -> str:
    meta = row.get("meta") or {}
    bits = []
    if meta.get("turn") is not None:
        bits.append(f"turn={meta.get('turn')}")
    if meta.get("rounds_used") is not None:
        bits.append(f"rounds={meta.get('rounds_used')}")
    if meta.get("tool_calls") is not None:
        bits.append(f"tools={meta.get('tool_calls')}")
    if meta.get("cost_usd") is not None:
        bits.append(f"cost=${float(meta.get('cost_usd') or 0):.3f}")
    if bits:
        return ", ".join(bits)
    created = row.get("created_at")
    return created.strftime("%Y-%m-%d %H:%M") if hasattr(created, "strftime") else str(created or "?")


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
    advisories = _recent_project_advisories(row["id"])
    if advisories:
        pending = [a for a in advisories if a.get("consumed_at") is None]
        consumed = [a for a in advisories if a.get("consumed_at") is not None]
        print(f"  operator_advisories: pending={len(pending)} recent_consumed={len(consumed)}")
        for a in pending:
            created = a.get("created_at")
            when = created.strftime("%Y-%m-%d %H:%M") if hasattr(created, "strftime") else str(created or "?")
            print(f"    - PENDING #{a['id']} @ {when}: {(a.get('content') or '')[:300]}")
        for a in consumed[:3]:
            created = a.get("created_at")
            when = created.strftime("%Y-%m-%d %H:%M") if hasattr(created, "strftime") else str(created or "?")
            print(f"    - consumed #{a['id']} @ {when}: {(a.get('content') or '')[:180]}")
    notes, note_total, note_source = _recent_project_notes(
        row["id"], legacy_notes=row["research_notes"] or [], limit=5
    )
    print(f"  research_notes: {note_total} entries (source={note_source})")
    for n in notes:
        src = ", ".join(_coerce_note_sources(n.get("sources"))[:3])
        ts = n.get("created_at")
        when = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else (str(ts) if ts else "?")
        print(
            f"    - turn {n.get('turn')} @ {when}: {(n.get('text') or '')[:300]}"
            + (f"  [{src}]" if src else "")
        )
    tick_error = _latest_project_event(row["id"], "tick_error")
    if tick_error:
        content = str(tick_error.get("content") or "")
        created = tick_error.get("created_at")
        when = created.strftime("%Y-%m-%d %H:%M") if hasattr(created, "strftime") else str(created or "?")
        print(f"  last_tick_error: {when}")
        print("    " + content[:800].replace("\n", "\n    ") + ("…" if len(content) > 800 else ""))
    no_action = _latest_project_event(row["id"], "tick_no_durable_action")
    if no_action:
        content = str(no_action.get("content") or "")
        created = no_action.get("created_at")
        when = created.strftime("%Y-%m-%d %H:%M") if hasattr(created, "strftime") else str(created or "?")
        print(f"  last_tick_no_durable_action: {when}")
        print("    " + content[:800].replace("\n", "\n    ") + ("…" if len(content) > 800 else ""))
    tick_log = _latest_project_event(row["id"], "tick_tool_log")
    if tick_log:
        content = str(tick_log.get("content") or "")
        print(f"  last_tick_tool_log: {_format_tick_log_header(tick_log)}")
        print("    " + content[:1000].replace("\n", "\n    ") + ("…" if len(content) > 1000 else ""))
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    _ensure_tables()
    rows = db_query(
        """
        SELECT id, event_type, content, meta, created_at
          FROM autonomous_project_events
         WHERE project_id = %s
         ORDER BY created_at DESC, id DESC
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

    pst = sub.add_parser("status", help="Show autonomous runtime status without DB access")
    pst.add_argument("--no-systemd", action="store_true", help="Only read config; do not call systemctl")
    pst.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    pst.set_defaults(func=_cmd_status)

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
