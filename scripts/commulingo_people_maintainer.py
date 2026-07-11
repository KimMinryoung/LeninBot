#!/usr/bin/env python3
"""Run one bounded CommuLingo people-dictionary maintenance cycle.

The script deterministically selects one sparse existing person (or periodically asks for
one missing person), then gives only that task to the dedicated DeepSeek V4 Pro curator.
The curator has four tools and `commulingo_edit` is terminal, enforcing one write per run.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Distinguish unattended writes in revisions/suggestion provenance. This must be set before
# runtime_tools.registry imports runtime_tools.commulingo_people.
os.environ.setdefault("COMMULINGO_SUGGESTED_BY", "commulingo-maintainer")

from agents import get_agent
from bot_config import _deepseek_anthropic_client, _resolve_deepseek_model
from claude_loop import chat_with_tools
from db import query as db_query, query_one as db_query_one
from runtime_tools.registry import TOOLS, TOOL_HANDLERS
from tool_gateway.security import CallerContext, caller_scope

logger = logging.getLogger("commulingo_people_maintainer")

CONFIG_PATH = PROJECT_ROOT / "config" / "commulingo_maintainer.json"
LOCK_PATH = Path("/tmp/leninbot-commulingo-maintainer.lock")


def load_config(path: Path = CONFIG_PATH) -> dict:
    defaults = {
        "enabled": True,
        "mode": "auto",
        "new_person_every": 8,
        "recent_days": 30,
        "max_tokens": 8192,
    }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return defaults
    if not isinstance(raw, dict):
        raise ValueError("commulingo maintainer config must be an object")
    cfg = {**defaults, **raw}
    if cfg["mode"] not in {"auto", "enrich", "new"}:
        raise ValueError("mode must be auto, enrich, or new")
    cfg["new_person_every"] = max(0, int(cfg["new_person_every"]))
    cfg["recent_days"] = max(1, int(cfg["recent_days"]))
    cfg["max_tokens"] = max(2048, int(cfg["max_tokens"]))
    return cfg


def completed_run_count() -> int:
    row = db_query_one(
        """SELECT COUNT(*)::int AS n
             FROM commulingo_agent_suggestions
            WHERE suggested_by = 'commulingo-maintainer'
              AND status = 'approved'"""
    )
    return int((row or {}).get("n") or 0)


def choose_mode(config: dict, requested: str | None = None) -> str:
    mode = requested or config["mode"]
    if mode != "auto":
        return mode
    every = int(config["new_person_every"])
    if every > 0 and (completed_run_count() + 1) % every == 0:
        return "new"
    return "enrich"


def select_sparse_person(recent_days: int, forced_id: str = "") -> dict | None:
    params = {"recent_days": recent_days, "forced_id": forced_id.strip()}
    rows = db_query(
        """SELECT p.id, p.group_id, p.name_ko, p.name_en,
                  LENGTH(COALESCE(p.bio_ko, '')) AS bio_chars,
                  COUNT(DISTINCT c.id)::int AS career_count,
                  COUNT(DISTINCT s.id)::int AS section_count,
                  CASE WHEN COALESCE(p.moment_ko, '') = '' THEN 0 ELSE 1 END AS has_moment,
                  CASE WHEN r.person_id IS NULL THEN 0 ELSE 1 END AS has_role
             FROM commulingo_people p
             LEFT JOIN commulingo_person_career_entries c ON c.person_id = p.id
             LEFT JOIN commulingo_person_sections s ON s.person_id = p.id
             LEFT JOIN commulingo_person_roles r ON r.person_id = p.id
            WHERE (%(forced_id)s = '' OR p.id = %(forced_id)s)
              AND (%(forced_id)s <> '' OR NOT EXISTS (
                    SELECT 1 FROM commulingo_people_revisions rev
                     WHERE (rev.entity_id = p.id OR rev.entity_id LIKE p.id || '/%%')
                       AND rev.changed_by = 'commulingo-maintainer'
                       AND rev.created_at >= NOW() - (%(recent_days)s * INTERVAL '1 day')
                  ))
            GROUP BY p.id, p.group_id, p.name_ko, p.name_en, p.bio_ko, p.moment_ko, r.person_id
            ORDER BY
                  COUNT(DISTINCT s.id) ASC,
                  CASE WHEN COUNT(DISTINCT c.id) <= 1 THEN 0 ELSE 1 END ASC,
                  CASE WHEN r.person_id IS NULL THEN 0 ELSE 1 END ASC,
                  LENGTH(COALESCE(p.bio_ko, '')) ASC,
                  p.sort_order ASC
            LIMIT 1""",
        params,
    )
    return rows[0] if rows else None


def build_task(mode: str, candidate: dict | None) -> str:
    if mode == "new":
        return """MODE: NEW PERSON

Identify one historically important person missing from CommuLingo whose inclusion would
materially improve coverage of revolutionary or Soviet history. Inspect list_groups,
list_categories and list_offices, then search_people under the proposed name and aliases to
prove there is no duplicate. Start by opening the Russian Wikipedia article when available. One opened source is enough for routine card facts; use a second only for disputed or consequential claims. Create one
complete bilingual person card with a correct group and one primary role. Make exactly one
`commulingo_edit(target_type='person', action='create', ...)` call and stop. Do not create a
section or office row in this run."""
    if not candidate:
        raise RuntimeError("no eligible sparse person found")
    return f"""MODE: ENRICH EXISTING PERSON

Target exactly this person and no one else:
- id: {candidate['id']}
- Korean name: {candidate['name_ko']}
- English name: {candidate['name_en']}
- group: {candidate['group_id']}
- current Korean bio length: {candidate['bio_chars']} characters
- career rows: {candidate['career_count']}
- detail sections: {candidate['section_count']}
- has moment: {bool(candidate['has_moment'])}
- has primary role: {bool(candidate['has_role'])}

Call get_person and get_sections first. Find the single most valuable missing topic. Prefer
creating one substantial bilingual `person_section` (one topic, roughly 350-700 Korean
characters plus equivalent English) when no section covers it. If the real defect is a bad
classification or a very thin career, update the person instead, preserving every wholesale
field exactly. Start with Russian Wikipedia when available. One opened source is enough for routine card facts; use a second only for disputed or consequential claims. Make one commulingo_edit call and stop."""


def latest_maintainer_edit() -> dict | None:
    return db_query_one(
        """SELECT id, target_type, target_id, action, status, confidence, created_at
             FROM commulingo_agent_suggestions
            WHERE suggested_by = 'commulingo-maintainer'
            ORDER BY id DESC LIMIT 1"""
    )


async def run_once(*, mode: str, candidate_id: str, config: dict) -> dict:
    if not config["enabled"]:
        return {"status": "disabled"}

    from runtime_tools.commulingo_people import direct_apply_enabled
    if not direct_apply_enabled():
        raise RuntimeError("config/commulingo_people.json direct_apply must be true")

    chosen_mode = choose_mode(config, mode if mode != "auto" else None)
    candidate = select_sparse_person(config["recent_days"], candidate_id) if chosen_mode == "enrich" else None
    task = build_task(chosen_mode, candidate)
    before = completed_run_count()

    spec = get_agent("commulingo_curator")
    tools, handlers = spec.filter_tools(TOOLS, TOOL_HANDLERS)
    expected = set(spec.tools)
    available = {str(t.get("name") or "") for t in tools} & set(handlers)
    if expected != available:
        raise RuntimeError(f"curator toolset incomplete: missing={sorted(expected - available)}")

    model = _resolve_deepseek_model(spec.model or "deepseek_pro")
    tracker: dict = {}
    ctx = CallerContext(interface="agent", agent_name=spec.name, is_owner=True)
    with caller_scope(ctx):
        result = await chat_with_tools(
            [{"role": "user", "content": task}],
            client=_deepseek_anthropic_client,
            model=model,
            tools=tools,
            tool_handlers=handlers,
            system_prompt=spec.render_prompt(provider="deepseek"),
            max_rounds=spec.max_rounds,
            max_tokens=config["max_tokens"],
            budget_usd=spec.budget_usd,
            budget_tracker=tracker,
            agent_name=spec.name,
            finalization_tools=spec.finalization_tools,
            terminal_tools=spec.terminal_tools,
            thinking={"type": "enabled"},
            output_config={"effort": "high"},
        )

    after = completed_run_count()
    if after != before + 1:
        raise RuntimeError(f"expected exactly one applied edit, count changed {before} -> {after}; result={result[:500]}")
    edit = latest_maintainer_edit()
    if not edit or edit.get("status") != "approved":
        raise RuntimeError("applied edit was not recorded as approved")
    return {
        "status": "applied",
        "mode": chosen_mode,
        "candidate": candidate and candidate["id"],
        "model": model,
        "cost_usd": round(float(tracker.get("total_cost") or 0.0), 4),
        "rounds": int(tracker.get("rounds_used") or 0),
        "edit": edit,
        "result": result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one direct CommuLingo maintenance edit.")
    parser.add_argument("--mode", choices=["auto", "enrich", "new"], default="auto")
    parser.add_argument("--candidate", default="", help="Force an existing person id (enrich mode only).")
    parser.add_argument("--print-candidate", action="store_true", help="Print the selected candidate without calling the model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = load_config()
    lock_file = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.info("another maintainer run is active; exiting")
        return 0

    if args.print_candidate:
        print(json.dumps(select_sparse_person(config["recent_days"], args.candidate), ensure_ascii=False, default=str, indent=2))
        return 0
    result = asyncio.run(run_once(mode=args.mode, candidate_id=args.candidate, config=config))
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
