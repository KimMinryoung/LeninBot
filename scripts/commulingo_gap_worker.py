#!/usr/bin/env python3
"""Resolve one queued curation gap per run.

The event lane files what its sections needed and the site does not have
(commulingo_curation_gaps, frontend migration 125). This drains that queue: it
claims the highest-priority pending row, commissions the card that closes it,
and writes back what was produced.

This is the half of event-led curation the existing lanes could not do. Their
selection is self-directed by design — the people lane picks the sparsest card,
the glossary lane picks an unregistered concept out of whatever material came
up in the rotation — which fills the dictionaries evenly and leaves the event
pages surrounded by names that do not resolve. A gap worker inverts that: the
narrative decides what gets written next.

Kinds:
  person, with target_id  -> enrich that existing card (people maintainer, --candidate)
  person, no target_id    -> create the card, skipping the discovery stage: the
                             event lane already did the deciding, so paying a
                             model to re-decide would undo the point
  term                    -> register the concept, briefed from the event text
  doc                     -> left queued and reported. A reference document is a
                             translation-pipeline spec, not a card, and it needs
                             a human to choose an edition and check the licence.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _worker_tag() -> str:
    """This copy's provenance suffix, read from argv before the imports below.

    Every worker used to write under one `commulingo-gap-worker`, and the stage
    decides whether its commissioned write landed by counting rows with that
    value. With one worker that is exact. With four it is not: A's count moves
    when B writes, so A's own successful write is reported as an unexpected edit
    and the run raises after the card is already saved. On 2026-08-09 all seven
    counter-revolution cards were written that way and every one of them closed
    with an error note.

    argparse runs in main(), which is far too late — COMMULINGO_SUGGESTED_BY has
    to be set before runtime_tools.commulingo_people is imported a few lines
    down. Hence reading argv by hand here. Same fix as commulingo_people_parallel,
    which gives each of its lanes its own suggested_by for the same reason.
    """
    argv = sys.argv[1:]
    values = {"--kind": "any", "--worker": "0"}
    for index, arg in enumerate(argv):
        for flag in values:
            if arg == flag and index + 1 < len(argv):
                values[flag] = argv[index + 1]
            elif arg.startswith(f"{flag}="):
                values[flag] = arg.split("=", 1)[1]
    return f"{values['--kind'] or 'any'}-{values['--worker'] or '0'}"


SUGGESTED_BY = f"commulingo-gap-worker-{_worker_tag()}"
os.environ["COMMULINGO_SUGGESTED_BY"] = SUGGESTED_BY

from scripts import commulingo_people_maintainer as maintainer  # noqa: E402
from scripts import commulingo_gap_event_links as event_links  # noqa: E402
from agents import get_agent  # noqa: E402
from bot_config import _resolve_deepseek_model  # noqa: E402
from db import query as db_query, query_one as db_query_one, get_conn  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402
from runtime_tools.registry import TOOLS, TOOL_HANDLERS  # noqa: E402
from tool_gateway.security import caller_scope, new_run_context  # noqa: E402

logger = logging.getLogger("commulingo_gap_worker")

LOCK_PATH = Path(f"/tmp/leninbot-{SUGGESTED_BY}.lock")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def completed_run_count() -> int:
    row = db_query_one(
        """SELECT COUNT(*)::int AS n
             FROM commulingo_agent_suggestions
            WHERE suggested_by = %(s)s AND status = 'approved'""",
        {"s": SUGGESTED_BY},
    )
    return int((row or {}).get("n") or 0)


# The maintainer's own counter is hardcoded to the 'commulingo-maintainer' lane,
# and _call_curator_stage uses it to decide whether the commissioned write landed.
# Reused unchanged from here it compares this lane's count against that lane's and
# reads the difference as an unexplained edit. Rebinding it is how
# commulingo_people_parallel.py solves the same problem for its two lanes.
maintainer.completed_run_count = completed_run_count
maintainer.latest_maintainer_edit = lambda: db_query_one(
    """SELECT id, target_type, target_id, action, status, confidence, created_at
         FROM commulingo_agent_suggestions
        WHERE suggested_by = %(s)s
        ORDER BY id DESC LIMIT 1""",
    {"s": SUGGESTED_BY},
)
maintainer.LOCK_PATH = LOCK_PATH


def claim_gap(kind: str = "") -> dict | None:
    """Take the highest-priority pending gap and mark it claimed, atomically.

    SKIP LOCKED rather than a plain UPDATE ... LIMIT 1 so several workers can
    drain the queue at once without two of them writing the same card.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """UPDATE commulingo_curation_gaps
                      SET status = 'claimed', claimed_by = %(by)s, claimed_at = NOW(),
                          updated_at = NOW()
                    WHERE id = (
                        SELECT id FROM commulingo_curation_gaps
                         WHERE (status = 'pending'
                                -- A worker that died mid-run leaves its row claimed
                                -- forever. Nothing else would ever pick it up, so a
                                -- claim older than any single run is treated as
                                -- abandoned and re-offered.
                                OR (status = 'claimed'
                                    AND claimed_at < NOW() - INTERVAL '40 minutes'))
                           AND (%(kind)s = '' OR kind = %(kind)s)
                           -- Only the kinds this worker can actually finish. The
                           -- 2026-08-09 backfill left 18 kind='event' rows that the
                           -- gap_report enum no longer admits; they were commissioned
                           -- as glossary terms (which the term task itself forbids
                           -- for episodes) and produced_entry() knows no 'event'
                           -- lookup, so a created entry never closed its row — each
                           -- burned a second full run before skipping (gaps 1101 and
                           -- 1107, 2026-08-11). They stay pending for human triage,
                           -- like 'doc'.
                           AND kind IN ('person', 'term')
                         ORDER BY priority DESC, id
                         FOR UPDATE SKIP LOCKED
                         LIMIT 1)
                RETURNING *""",
                {"by": SUGGESTED_BY, "kind": kind},
            )
            row = cur.fetchone()
            return dict(row) if row else None


def produced_entry(gap: dict) -> str:
    """The id of the entry this gap asked for, if it now exists.

    Closing a gap used to be decided by whether the suggestion count moved,
    which is a proxy for "my write landed" and was a wrong one while every
    worker shared a suggested_by: another worker's write moved the same counter,
    so a run whose own write never happened still closed its row as done. 36
    gaps were retired that way on 2026-08-09 with nothing written for them, and
    nothing noticed because the queue said they were finished.

    This asks the dictionaries instead. It is the same lookup the gap tool uses
    to refuse a gap for an entry that already exists, so a gap closes as done
    only when the thing it asked for is actually there.
    """
    if gap["target_id"]:
        return gap["target_id"]  # a deepen request; the card existed all along
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            from runtime_tools.commulingo_people import _already_covered
            return _already_covered(
                cur, gap["kind"], {"ko": gap["label_ko"], "en": gap["label_en"]}, ""
            )



async def link_back(gap: dict, person_id: str) -> bool:
    """Connect a new card to the event that commissioned it.

    The write tool that creates a card is a finalization tool, so the run that
    made it ends there and cannot go on to call commulingo_event_link. Left
    alone, that meant 213 of the cards an event page had asked for never
    appeared on it: the first man killed at Chernobyl was missing from the
    Chernobyl event while eighteen of his colleagues were on it.

    The link needs a relation line and a note in both languages, which the gap
    row does not carry, so this is one small model call outside the run rather
    than a plain INSERT. It is best-effort: a card with no link is worth more
    than a failed run, and the backfill script sweeps up whatever this misses.
    """
    if not gap.get("event_id") or gap.get("kind") != "person":
        return False
    try:
        rows = event_links.pending_rows_for(person_id, gap["event_id"])
        if not rows:
            return False   # the curator linked it itself, or the event is gone
        entries = await event_links.describe(rows, _resolve_deepseek_model("deepseek_flash"))
        problem = event_links.acceptable(entries[0])
        if problem:
            logger.warning("event link for %s not written: %s", person_id, problem)
            return False
        event_links.write_link(rows[0], entries[0])
        return True
    except Exception as exc:
        logger.warning("event link for %s failed: %s", person_id, exc)
        return False


def close_gap(gap_id: int, status: str, resolved_id: str, resolution: str) -> None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """UPDATE commulingo_curation_gaps
                      SET status = %(status)s, resolved_id = %(rid)s,
                          resolution = %(note)s, updated_at = NOW()
                    WHERE id = %(id)s""",
                {"status": status, "rid": resolved_id, "note": resolution[:500], "id": gap_id},
            )


def slugify(label_en: str, label_ko: str) -> str:
    # Fold diacritics first (Joaquín → joaquin); otherwise every accented letter
    # becomes a hyphen and the public URL reads joaqu-n-maur-n.
    import unicodedata
    folded = unicodedata.normalize("NFKD", label_en or "").encode("ascii", "ignore").decode()
    base = _SLUG_STRIP.sub("-", folded.strip().lower()).strip("-")
    if not base:
        # No Latin form to work from. The id is internal, so a stable transliteration
        # is not worth a model round; the curator renames it if it matters.
        base = "gap-" + _SLUG_STRIP.sub("-", str(abs(hash(label_ko)))[:8])
    return base[:60]


def build_person_task(gap: dict) -> str:
    return f"""MODE: NEW PERSON CREATION, commissioned by a history-event page.

Create exactly this person and no one else:
- suggested id: {slugify(gap['label_en'], gap['label_ko'])}
- Korean name: {gap['label_ko']}
- English name: {gap['label_en']}

WHY THE SITE NEEDS THIS CARD (written by the curator of `{gap['event_id']}`, whose
section leans on this person):
{gap['reason']}

That brief tells you what the card has to carry. The reader arrives from the event page,
so the parts of this life that touch `{gap['event_id']}` are the parts that matter most;
do not reduce the card to them, but do not omit them either.

First run search_people on both names and on plausible transliteration variants. If the
person already has a card, do NOT create a second one: finish with a single
commulingo_no_edit call naming that card's id. Otherwise research and create one complete
bilingual card in a single commulingo_person_create call.
""" + maintainer.CARD_STYLE_GUIDANCE


def build_term_task(gap: dict) -> str:
    # The glossary lane hard-injects this list and forbids re-registering any of
    # it; the gap worker used to carry only a sentence of prose about events, and
    # the model overrode it. Seven event titles were registered as glossary terms
    # on 2026-08-09 that way, each one a second page saying what its event page
    # already said. Same list, same instruction, so the two lanes refuse the same
    # thing.
    from runtime_tools.commulingo_people import registered_event_labels
    events = registered_event_labels()
    return f"""MODE: GLOSSARY TERM REGISTRATION, commissioned by a history-event page.

Register exactly this concept and no other:
- suggested id: {slugify(gap['label_en'], gap['label_ko'])}
- Korean term: {gap['label_ko']}
- English term: {gap['label_en']}

WHY THE SITE NEEDS THIS ENTRY (written by the curator of `{gap['event_id']}`, whose
section uses it):
{gap['reason']}

ALREADY IN THE EVENTS DICTIONARY (these are events, not glossary terms — never register
one of these, a renaming of it, or the same title with a date bracket added):
{', '.join(events)}

If what was commissioned is one of those, finish with commulingo_no_edit naming the event.
A concept the event narrates (관동군, 그단스크 협정) is a term and belongs here; the
episode itself (「폴란드 연대노조와 계엄 (1980~1981)」) does not, and neither does a
compound headword that welds two of them together.

First confirm it is not already registered:
commulingo_people(action='list_terms', q='<the term>'), matched against ids, both names,
the native-script form and every alias. If it is already there, finish with a single
commulingo_no_edit call naming that entry's id. If what was requested is really a person or
a single historical event rather than a concept, finish with commulingo_no_edit saying so.

Otherwise research it (Wikipedia first, Russian article for Soviet-era vocabulary, then at
least one source outside Wikipedia) and make exactly one commulingo_term_create call with
term {{ko,en}}, original (native-script form), definition {{ko,en}} (2-3 sentences; depth
goes in body), aliases {{ko,en}} including the exact spelling the event text uses, period
{{ko,en}}, startYear/endYear when the label names a year, and category, one of: theory,
economy, party-state, factions, repression, nationalities, culture, international, korea,
contemporary. Keep citations top-level.
"""


async def run_once(kind: str = "") -> dict:
    config = maintainer.load_config()
    if not config.get("enabled"):
        return {"status": "skipped", "reason": "maintainer disabled"}

    from runtime_tools.commulingo_people import direct_apply_enabled
    from tool_gateway.inference import resolve_agent_inference_policy

    if not direct_apply_enabled():
        raise RuntimeError("config/commulingo_people.json direct_apply must be true")

    gap = claim_gap(kind)
    if not gap:
        return {"status": "skipped", "reason": f"no pending {kind or 'person/term'} gap"}

    # The queue runs hundreds deep, so hours pass between a gap being filed and
    # a worker reaching it, and another worker may have written the entry in
    # between. Five of the twenty-two glossary gaps dismissed on 2026-08-09 were
    # that: the check at filing time found nothing, and the run that eventually
    # picked the row spent a full research cycle rediscovering an entry that by
    # then existed. One query here costs nothing and closes the row instead.
    if not gap["target_id"]:
        already = produced_entry(gap)
        if already:
            close_gap(gap["id"], "skipped", already,
                      f"이미 등재됨({already}) — 큐에 넣은 뒤 다른 실행이 만든 것")
            return {"status": "skipped", "gap": gap["id"], "kind": gap["kind"],
                    "label": gap["label_ko"], "reason": f"already covered by {already}",
                    "cost_usd": 0.0, "rounds": 0}

    if gap["kind"] == "person" and gap["target_id"]:
        # An existing but thin card: the people lane's enrich path already knows
        # how to deepen one, so hand it the id rather than duplicating that logic.
        # Wrapped, because an unwrapped raise here left the row 'claimed' and
        # invisible to every other worker until the 40-minute reclaim — seven of
        # them were sitting like that on 2026-08-09 with their sections already
        # written.
        before_enrich = completed_run_count()
        try:
            result = await maintainer.run_once(
                mode="enrich", candidate_id=gap["target_id"], config=config
            )
        except Exception as exc:
            landed = completed_run_count() != before_enrich
            close_gap(
                gap["id"], "done" if landed else "pending", gap["target_id"],
                f"enrich raised after {'a' if landed else 'no'} write: {exc}",
            )
            return {"status": "applied" if landed else "error", "gap": gap["id"],
                    "kind": "person-enrich", "target": gap["target_id"],
                    "error": str(exc)[:300]}
        close_gap(
            gap["id"],
            "done" if result.get("status") == "applied" else "pending",
            gap["target_id"],
            f"enrich run: {result.get('status')}",
        )
        return {"status": result.get("status"), "gap": gap["id"], "kind": "person-enrich",
                "target": gap["target_id"], "cost_usd": result.get("cost_usd"),
                "rounds": result.get("rounds")}

    spec = get_agent("commulingo_curator")
    policy = resolve_agent_inference_policy(spec)
    tools, handlers = spec.filter_tools(TOOLS, TOOL_HANDLERS)
    handlers = dict(handlers)
    for name in maintainer.NARROW_WRITE_TOOLS & handlers.keys():
        handlers[name] = maintainer.build_retrying_write_handler(handlers[name])

    write_name = "commulingo_person_create" if gap["kind"] == "person" else "commulingo_term_create"
    stage_tools = [t for t in tools if t.get("name") not in maintainer.NARROW_WRITE_TOOLS
                   or t.get("name") == write_name]
    stage_handlers = {name: handler for name, handler in handlers.items()
                      if name not in maintainer.NARROW_WRITE_TOOLS or name == write_name}
    # "The entry already exists" and "this is not a glossary term" are correct,
    # complete outcomes, but the stage treats a run with no write as a failure and
    # retries it three times. commulingo_no_edit is its typed way of saying the
    # commissioned write should not happen, so wire it rather than leaving the
    # verdict in free text the stage cannot see.
    no_edit_box: dict = {}
    stage_tools = [*stage_tools, maintainer.COMMULINGO_NO_EDIT_TOOL]
    stage_handlers = {
        **stage_handlers,
        "commulingo_no_edit": maintainer.build_no_edit_handler(no_edit_box),
    }
    task = build_person_task(gap) if gap["kind"] == "person" else build_term_task(gap)

    before = completed_run_count()
    ctx = new_run_context(
        interface="autonomous", agent_name=spec.name, is_owner=True,
        scope_type="maintenance_job", scope_id=f"commulingo_gap:{gap['id']}",
    )
    try:
        with caller_scope(ctx):
            result, tracker, _ = await maintainer._call_curator_stage(
                task=task, spec=spec,
                tools=stage_tools, handlers=stage_handlers, policy=policy,
                stage=f"gap-{gap['kind']}", expect_edit=True, before_count=before,
                finalization_tools=[write_name, "commulingo_no_edit"],
                terminal_tools=[write_name, "commulingo_no_edit"],
                no_edit_box=no_edit_box,
            )
    except Exception as exc:
        # The stage raises after exhausting its own retries. Whether the row is
        # finished is decided by looking for the entry, not by the counter.
        made = produced_entry(gap)
        if made:
            close_gap(gap["id"], "done", made, f"written, then the stage raised: {exc}")
            raise
        close_gap(gap["id"], "pending", "", f"requeued after stage error: {exc}")
        return {"status": "error", "gap": gap["id"], "kind": gap["kind"],
                "label": gap["label_ko"], "error": str(exc)[:300]}

    text = str(result or "")
    summary = {
        "gap": gap["id"], "kind": gap["kind"], "label": gap["label_ko"],
        "event": gap["event_id"], "cost_usd": round(float(tracker.get("total_cost") or 0), 4),
        "rounds": int(tracker.get("rounds_used") or 0),
    }
    made = produced_entry(gap)
    if made:
        close_gap(gap["id"], "done", made, "created")
        summary["linked"] = await link_back(gap, made)
        return {"status": "applied", "created": made, **summary}
    reason = str(no_edit_box.get("reason") or "")
    if reason:
        close_gap(gap["id"], "skipped", "", reason)
        return {"status": "skipped", "reason": reason[:200], **summary}
    # Nothing written and no verdict given: put it back for the next worker
    # rather than burning the row on one barren run.
    close_gap(gap["id"], "pending", "", "no edit; requeued")
    return {"status": "no_edit", "result": text[:400], **summary}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="", choices=["", "person", "term"],
                        help="Restrict to one gap kind; default takes the highest priority of either.")
    parser.add_argument("--worker", type=int, default=0,
                        help="This copy's index. Several workers of the same kind drain the queue "
                             "together; claim_gap's FOR UPDATE SKIP LOCKED is what makes that safe, "
                             "and this only keeps one index from overlapping itself.")
    parser.add_argument("--queue", action="store_true", help="Print the pending queue and exit.")
    args = parser.parse_args()

    if args.queue:
        rows = db_query(
            """SELECT id, kind, label_ko, event_id, priority, status
                 FROM commulingo_curation_gaps
                WHERE status IN ('pending', 'claimed')
                ORDER BY priority DESC, id"""
        )
        print(json.dumps(rows, ensure_ascii=False, default=str, indent=2))
        return 0

    # One lock per (kind, worker index), so as many workers as the queue warrants
    # run side by side. claim_gap already makes that safe on the row level
    # (FOR UPDATE SKIP LOCKED); this only stops one index overlapping itself.
    lock_path = Path(f"{LOCK_PATH}.{args.kind or 'any'}.{args.worker}")
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.info("another gap worker is active; exiting")
        return 0
    result = asyncio.run(run_once(args.kind))
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
