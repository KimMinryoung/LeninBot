#!/usr/bin/env python3
"""Register one missing CommuLingo glossary term per run.

Each run picks source material the site already publishes (a research report,
the history-event texts, or a batch of person bios, rotating by run count),
shows the curator that material together with every already-registered term
alias, and asks it to find ONE concept term genuinely used in the material that
the glossary does not cover yet, research it, and register it via
commulingo_edit(target_type='term'). When the material contains no unregistered
concept the run ends with NO_CANDIDATE and no write, which is a success.
"""

from __future__ import annotations

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

SUGGESTED_BY = "commulingo-maintainer-terms"
os.environ["COMMULINGO_SUGGESTED_BY"] = SUGGESTED_BY

from scripts import commulingo_people_maintainer as maintainer  # noqa: E402
from agents import get_agent  # noqa: E402
from bot_config import _deepseek_anthropic_client, _resolve_deepseek_model  # noqa: E402
from claude_loop import chat_with_tools  # noqa: E402
from db import query as db_query, query_one as db_query_one  # noqa: E402
from runtime_tools.registry import TOOLS, TOOL_HANDLERS  # noqa: E402
from tool_gateway.security import CallerContext, caller_scope  # noqa: E402

logger = logging.getLogger("commulingo_terms_maintainer")

LOCK_PATH = Path(f"/tmp/leninbot-{SUGGESTED_BY}.lock")
MATERIAL_CHARS = 7000


def completed_run_count() -> int:
    row = db_query_one(
        """SELECT COUNT(*)::int AS n
             FROM commulingo_agent_suggestions
            WHERE suggested_by = %(s)s AND status = 'approved'""",
        {"s": SUGGESTED_BY},
    )
    return int((row or {}).get("n") or 0)


def latest_lane_edit() -> dict | None:
    return db_query_one(
        """SELECT id, target_type, target_id, action, status, confidence, created_at
             FROM commulingo_agent_suggestions
            WHERE suggested_by = %(s)s
            ORDER BY id DESC LIMIT 1""",
        {"s": SUGGESTED_BY},
    )


def registered_aliases() -> list[str]:
    rows = db_query(
        """SELECT t.term_ko AS a FROM commulingo_terms t
           UNION SELECT t.term_en FROM commulingo_terms t
           UNION SELECT a.alias FROM commulingo_term_aliases a"""
    )
    return sorted({str(r["a"]).strip() for r in rows if r.get("a")})


def pick_material(run_index: int) -> tuple[str, str]:
    """(label, text) rotating over reports / history events / person bios."""
    kind = run_index % 3
    if kind == 0:
        rows = db_query(
            """SELECT slug, title, markdown FROM research_documents
                WHERE status = 'public' ORDER BY random() LIMIT 1"""
        )
        if rows:
            row = rows[0]
            return (
                f"research report '{row['title']}' (cyber-lenin.com/reports/research/{row['slug']})",
                str(row["markdown"] or "")[:MATERIAL_CHARS],
            )
    if kind == 1:
        rows = db_query(
            """SELECT title_ko, summary_ko, outcome_ko, timeline FROM commulingo_history_events
                ORDER BY random() LIMIT 6"""
        )
        def _ko(value):
            if isinstance(value, dict):
                return str(value.get("ko") or value.get("en") or "")
            return str(value or "")

        parts = []
        for row in rows:
            timeline = row.get("timeline") or []
            timeline_text = " ".join(
                f"{_ko(item.get('title'))} {_ko(item.get('body'))}"
                for item in timeline if isinstance(item, dict)
            )
            parts.append(f"[{row['title_ko']}] {row['summary_ko']} {row['outcome_ko']} {timeline_text}")
        if parts:
            return "history-event pages (titles, summaries, timelines)", "\n".join(parts)[:MATERIAL_CHARS]
    rows = db_query(
        """SELECT name_ko, bio_ko, moment_ko FROM commulingo_people
            WHERE COALESCE(bio_ko, '') <> '' ORDER BY random() LIMIT 30"""
    )
    parts = [f"[{row['name_ko']}] {row['bio_ko']} {row.get('moment_ko') or ''}" for row in rows]
    return "person-dictionary bios", "\n".join(parts)[:MATERIAL_CHARS]


def build_term_task(material_label: str, material: str, existing: list[str]) -> str:
    return f"""Glossary curation run. Source material below comes from {material_label}.

Find CONCEPT TERMS actually used in this material that the glossary (/commulingo/terms)
does not cover yet, pick the single most valuable one, research it, and register it.

What qualifies as a glossary term: a concept, institution-type, policy, doctrine,
social category, or period-specific vocabulary item (like 네프맨, 굴라크, 일국사회주의).
NOT a person (people dictionary), NOT a single historical event (events dictionary),
NOT an institution that already has an office page (Comintern, Gosplan, state security
organs), and NOT an ordinary everyday word.

ALREADY REGISTERED (never re-register these or their variants):
{', '.join(existing)}

Workflow:
1. Read the material, list candidate terms it actually uses, and drop everything in the
   registered list. Confirm absence with commulingo_people(action='list_terms').
2. Pick ONE term. Wikipedia-first research (wiki_search/wiki_get; Russian article for
   Soviet-era vocabulary). One solid article is enough; record source URLs in `sources`.
3. Register exactly one term: commulingo_edit(target_type='term', action='create',
   target_id='<kebab-slug>') with term {{ko,en}}, original (native-script form),
   period, definition {{ko,en}} (2-3 sentences, the card paragraph), aliases {{ko,en}}
   including the EXACT spelling the material uses plus common variants, and related
   people/events ids when clearly applicable (verify ids via search_people/list_events).
4. If every concept in the material is already registered, reply with the single line
   NO_CANDIDATE and STOP — do not call commulingo_edit, do not stretch the definition
   of a term to justify a write.

One run, at most one write."""


EDITORIAL_POLICY = """

EDITORIAL POLICY (MANDATORY):
- Definitions are historically grounded and factually complex; criticism is allowed when
  relevant and sourced, but polemical anti-Soviet framing is not the voice of this glossary.
- Lead with what the concept was and how it functioned; repression statistics belong in the
  definition only when they are the historical point of the term itself.
"""


async def run_once() -> dict:
    config = maintainer.load_config()
    if not config.get("enabled") or not config.get("term_lane_enabled"):
        return {"status": "skipped", "reason": "term_lane_enabled=false"}

    from runtime_tools.commulingo_people import direct_apply_enabled
    from tool_gateway.inference import resolve_agent_inference_policy, resolve_inference_extra

    if not direct_apply_enabled():
        raise RuntimeError("config/commulingo_people.json direct_apply must be true")

    before = completed_run_count()
    spec = get_agent("commulingo_curator")
    policy = resolve_agent_inference_policy(spec)
    tools, handlers = spec.filter_tools(TOOLS, TOOL_HANDLERS)
    handlers = dict(handlers)
    handlers["commulingo_edit"] = maintainer.build_validating_edit_handler(handlers["commulingo_edit"])
    model = _resolve_deepseek_model(spec.model or "deepseek_pro")
    reasoning = resolve_inference_extra(policy, "deepseek")

    label, material = pick_material(before)
    task = build_term_task(label, material, registered_aliases()) + EDITORIAL_POLICY

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
            max_rounds=policy.max_rounds,
            max_tokens=policy.max_output_tokens,
            max_input_tokens=policy.max_input_tokens,
            recover_input_via_tools=True,
            continue_on_length=policy.max_output_continuations > 0,
            max_length_continuations=policy.max_output_continuations,
            budget_usd=policy.budget_usd,
            budget_tracker=tracker,
            agent_name=spec.name,
            finalization_tools=[],
            terminal_tools=spec.terminal_tools,
            thinking=reasoning.get("thinking"),
            output_config=reasoning.get("output_config"),
        )

    after = completed_run_count()
    summary = {
        "material": label,
        "model": model,
        "cost_usd": round(float(tracker.get("total_cost") or 0.0), 4),
        "rounds": int(tracker.get("rounds_used") or 0),
    }
    if after == before:
        if "NO_CANDIDATE" in str(result):
            return {"status": "skipped", "reason": "no unregistered term in material", **summary}
        raise RuntimeError(f"term run ended with neither an edit nor NO_CANDIDATE: {str(result)[:500]}")
    if after != before + 1:
        raise RuntimeError(f"expected exactly one applied edit, count changed {before} -> {after}")
    edit = latest_lane_edit()
    if not edit or edit.get("status") != "approved" or edit.get("target_type") != "term":
        raise RuntimeError(f"applied edit was not an approved term edit: {edit}")
    return {"status": "applied", "edit": edit, "result": result, **summary}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    lock_file = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.info("another terms run is active; exiting")
        return 0
    result = asyncio.run(run_once())
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
