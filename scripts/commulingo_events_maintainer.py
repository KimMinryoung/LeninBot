#!/usr/bin/env python3
"""Write one section of one CommuLingo history-event body per run.

The event pages were the thinnest thing on the site: on 2026-08-08, 37 of the
39 events had a summary, a timeline and no body at all, so a reader who arrived
at /commulingo/events/great-terror got a list of dates where an encyclopedia
article should be. The two events that did have a body (spanish-civil-war,
sino-soviet-split) were written by hand and are what this lane is aimed at.

A body is built one `## ` section per run rather than in one pass, for the same
reason a person's detail sections are: a run that had to restate the whole body
would spend most of its output reproducing text it is not changing, and would
regress that text whenever it drifted. Selection is deterministic — the event
with the least body, oldest first — so consecutive runs walk the whole
dictionary instead of deepening whichever event the model finds most
interesting.

The run also files what its section needed and the site lacks (see
commulingo_gap_report); the people and glossary lanes drain that queue. That is
the point of running events first: the dictionaries get filled with what the
narrative actually leans on rather than with whatever was sparsest.
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

def _lane_tag() -> str:
    """This copy's partition index, read from argv before the imports below.

    The three lanes shared one suggested_by, and the run's own success check
    counts rows with that value, so lane 0's count moved whenever lane 1 wrote
    and the run raised after its section was already saved. That happened 226
    times on 2026-08-09. No section was lost — the write commits before the
    check — but every one of those runs exited without reporting what it did.

    COMMULINGO_SUGGESTED_BY has to be set before runtime_tools is imported a few
    lines down, which is well before argparse runs.
    """
    argv = sys.argv[1:]
    lane = "0"
    for index, arg in enumerate(argv):
        if arg == "--lane" and index + 1 < len(argv):
            lane = argv[index + 1]
        elif arg.startswith("--lane="):
            lane = arg.split("=", 1)[1]
    return lane


SUGGESTED_BY = f"commulingo-maintainer-events-{_lane_tag()}"
os.environ["COMMULINGO_SUGGESTED_BY"] = SUGGESTED_BY

from scripts import commulingo_people_maintainer as maintainer  # noqa: E402
from agents import get_agent  # noqa: E402
from bot_config import resolve_agent_tool_loop  # noqa: E402
from db import query as db_query, query_one as db_query_one  # noqa: E402
from runtime_tools.commulingo_people import (  # noqa: E402
    EVENT_SECTION_TARGET, event_section_headings,
)
from runtime_tools.registry import TOOLS, TOOL_HANDLERS  # noqa: E402
from tool_gateway.security import caller_scope, new_run_context  # noqa: E402

logger = logging.getLogger("commulingo_events_maintainer")

LOCK_PATH = Path(f"/tmp/leninbot-{SUGGESTED_BY}.lock")
WRITE_TOOL = "commulingo_event_section_save"
# How many sections an event is built out to before the lane treats it as done
# and moves on. Sino-soviet-split and spanish-civil-war, the two hand-written
# bodies, have 7 sections each; past roughly a dozen a page wants splitting
# rather than deepening, which is also where EVENT_BODY_CEILING sits.
TARGET_SECTIONS = 9
ATTEMPTS = 2


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


def select_event(forced_id: str = "", lane: int = 0, lanes: int = 1, skeleton: bool = False) -> dict | None:
    """The event that most needs its next section.

    Ordered by how much body it already has, then by period, so an event with no
    body at all is always picked before one that has any: the first pass across
    the dictionary gives every event an opening section before any event gets a
    second one.

    `lane`/`lanes` partition the dictionary so several copies of this script can
    run at once. The partition is on sort_order, which never moves, rather than
    on position in the ordering above, which changes every time a body grows.
    That matters for correctness, not just tidiness: a body section is a
    read-modify-write of one markdown column, so two lanes that drifted onto the
    same event would silently drop one of the two sections.
    """
    if forced_id:
        row = db_query_one(
            "SELECT * FROM commulingo_history_events WHERE id = %(id)s", {"id": forced_id}
        )
        return dict(row) if row else None
    if skeleton:
        # A row with no summary has never been written at all, and its section
        # count is beside the point.
        row = db_query_one(
            """SELECT * FROM commulingo_history_events
                WHERE COALESCE(summary_ko, '') = '' ORDER BY sort_order LIMIT 1"""
        )
        return dict(row) if row else None
    rows = db_query(
        """SELECT * FROM commulingo_history_events
            WHERE %(lanes)s = 1 OR (sort_order %% %(lanes)s) = %(lane)s
            ORDER BY length(body_ko) ASC, sort_order ASC""",
        {"lane": lane, "lanes": lanes},
    )
    for row in rows:
        if len(event_section_headings(row["body_ko"] or "")) < TARGET_SECTIONS:
            return dict(row)
    return None


def event_people(event_id: str) -> list[str]:
    rows = db_query(
        """SELECT p.name_ko, p.years_label, ep.relation_kind, ep.relation_ko
             FROM commulingo_history_event_people ep
             JOIN commulingo_people p ON p.id = ep.person_id
            WHERE ep.event_id = %(id)s
            ORDER BY ep.sort_order, ep.person_id
            LIMIT 60""",
        {"id": event_id},
    )
    return [
        f"{row['name_ko']} ({row['years_label'] or '?'}, {row['relation_ko'] or row['relation_kind']})"
        for row in rows
    ]


def open_gaps(event_id: str) -> list[str]:
    """Gaps already queued for this event, so a run does not re-file them."""
    rows = db_query(
        """SELECT kind, label_ko FROM commulingo_curation_gaps
            WHERE event_id = %(id)s AND status IN ('pending', 'claimed')
            ORDER BY priority DESC, id LIMIT 40""",
        {"id": event_id},
    )
    return [f"{row['label_ko']} ({row['kind']})" for row in rows]


def writes_since(since_id: int) -> dict:
    """{'body': n, 'links': n, 'other': n} for this lane's writes after `since_id`.

    A body write is the section or card the run was commissioned for; links are
    the person-to-event rows it may add alongside. Splitting them is what lets
    the run demand exactly one of the former while allowing several of the
    latter.
    """
    rows = db_query(
        """SELECT target_type, count(*)::int AS n
             FROM commulingo_agent_suggestions
            WHERE suggested_by = %(s)s AND status = 'approved' AND id > %(since)s
            GROUP BY target_type""",
        {"s": SUGGESTED_BY, "since": since_id},
    )
    tally = {"body": 0, "links": 0, "other": 0}
    for row in rows:
        if row["target_type"] in ("history_event_section", "history_event"):
            tally["body"] += row["n"]
        elif row["target_type"] == "history_event_person":
            tally["links"] += row["n"]
        else:
            tally["other"] += row["n"]
    return tally


def _ko(value) -> str:
    if isinstance(value, dict):
        return str(value.get("ko") or value.get("en") or "")
    return str(value or "")


def section_assignment(event: dict) -> dict:
    """Which part of the story this run writes, decided here rather than by the model.

    The first two runs of this lane spent 32 paid rounds between them researching
    without ever writing: asked to "choose one part of this event's story", the
    curator kept searching for a better part instead of committing to one. Topic
    choice is the expensive open-ended step and it does not need a model — the
    event's own timeline is already the outline. So the lane walks it: an opening
    section that puts the reader in the situation, one section per couple of
    timeline entries, and a closing section on what the event left unsettled.

    The model still decides the heading, the argument and what to research; it
    just no longer decides what the run is about.
    """
    done = len(event_section_headings(event["body_ko"] or ""))
    timeline = [item for item in (event.get("timeline") or []) if isinstance(item, dict)]
    # One opening and one closing section bracket the narrative ones.
    narrative_slots = max(1, TARGET_SECTIONS - 2)
    per_slot = max(1, -(-len(timeline) // narrative_slots))  # ceil

    if done == 0:
        first = timeline[0].get("date", "") if timeline else event["period_label"]
        return {
            "kind": "opening",
            "brief": (
                f"the OPENING section: what the situation was on the eve of {first}, so "
                f"that a reader who knows nothing about the period can follow everything "
                f"that comes after. The conditions, the actors and the pressures that made "
                f"this event possible — not a summary of the event itself."
            ),
            "entries": [],
        }

    index = done - 1
    chunk = timeline[index * per_slot:(index + 1) * per_slot]
    if not chunk:
        return {
            "kind": "closing",
            "brief": (
                "the CLOSING section: what the event settled, what it did not, and what "
                "historians still disagree about. The card's `outcome` field states the "
                "consequences in a sentence; this section is where the argument goes."
            ),
            "entries": [],
        }
    return {
        "kind": "narrative",
        "brief": (
            "the section covering this stretch of the timeline. Do not retell the entries "
            "below — they are the outline. Write what was actually going on across them: "
            "why the decisions were taken, who was arguing for what, what it cost."
        ),
        "entries": chunk,
    }


SKELETON_TOOL = "commulingo_event_update"


def build_skeleton_task(event: dict) -> str:
    """The task for an event row that has a title and a period and nothing else.

    A new event is registered with only the three things a person had to decide:
    its id, when it ran, and what to call it. Everything a reader meets first —
    the question the page asks, the summary, what it left behind, the timeline —
    is written here, once, before any body section exists. The nightly lane then
    deepens it section by section.
    """
    return f"""MODE: NEW EVENT SKELETON. The event `{event['id']}` exists as a row with a title and a
period and nothing else. Write the parts a reader meets before the body.

  title:  {event['title_ko']} / {event['title_en']}
  period: {event['period_label']}

Research first. Use web_search and the dictionaries; two solid non-Wikipedia sources before
you write. Then call `{SKELETON_TOOL}` EXACTLY ONCE with all of:

  question — the one question this page answers, as a question, in both languages. Not a
      label: 「압도적으로 우세한 붉은 군대는 왜 작은 이웃 앞에서 105일을 흘렸는가?」 is the
      shape. It is what a reader who knows nothing would want settled.
  summary — what happened, in one paragraph. Open with what the event is and when, then the
      cause, then the course. A reader who stops here should be able to say what happened.
  outcome — what it left behind: the settlement, the cost, what it made possible or
      foreclosed, and where the consequence shows up later on this site.
  timeline — eight to twelve dated entries, each with a short title and a sentence of body,
      in both languages. Dates as the existing events write them: 1939.08.23, or 1921.03 when
      the day is not the point. First entry is the precondition, last is the consequence.
  sources — the works you actually used. Author and title; no bare URLs.

WRITING RULES

  Both languages carry the same content. English is not a gloss of Korean word order.
  No em dash anywhere. Korean is 한다체 written prose.
  Write 그루지야, never 조지아. Write 조선민주주의인민공화국 or 조선, never 북한.
  Korea before 1948 is 조선 (대한제국 for 1897–1910), 조선인, 조선어. 한국 is the
  southern republic of 1948 and nothing earlier; 한국전쟁 is the one fixed exception.
  Name people as their dictionary card names them.
  Do not invent numbers. A figure you cannot source does not go in.

Nothing is saved except through the tool call. A reply containing the draft is a failed run.
"""


def build_task(event: dict, brief: str = "") -> str:
    """`brief` commissions one specific section instead of the next one in the walk.

    The topic walker is right for a lane grinding through the catalogue, and
    wrong when a person has read the page and knows what is missing. The
    Kronstadt card asserted that the 1921 sailors were the men of 1917, which is
    the claim Trotsky made and Getzler's muster rolls contradict; the section
    that has to exist is the dispute itself, and no walk over the timeline would
    have produced it.
    """
    headings = event_section_headings(event["body_ko"] or "")
    timeline = event.get("timeline") or []
    timeline_text = "\n".join(
        f"  {item.get('date', '?')} — {_ko(item.get('title'))}: {_ko(item.get('body'))}"
        for item in timeline if isinstance(item, dict)
    ) or "  (none)"
    people = event_people(event["id"])
    queued = open_gaps(event["id"])

    assignment = section_assignment(event)
    if brief:
        assignment = {"kind": "commissioned", "brief": brief, "entries": []}
    if headings:
        body_state = (
            "This event's body already has these sections, in order:\n"
            + "\n".join(f"  {i + 1}. {h}" for i, h in enumerate(headings))
            + "\n\nAppend the next one at the end (omit `after`). It must not repeat any "
              "of the above."
        )
    else:
        body_state = "This event has NO body yet. Yours is its first section."

    assigned = f"\nYOUR ASSIGNMENT THIS RUN\n  Write {assignment['brief']}\n"
    if assignment["entries"]:
        assigned += "\n  The timeline entries this section spans:\n" + "\n".join(
            f"    {item.get('date', '?')} — {_ko(item.get('title'))}"
            for item in assignment["entries"]
        ) + "\n"
    assigned += (
        "\n  The topic is settled. Do not look for a better one, and do not widen it to "
        "cover the whole event. Research THIS and write it.\n"
    )

    return f"""History-event curation run. Target: `{event['id']}`.

THE EVENT AS IT STANDS
  Title:    {event['title_ko']} / {event['title_en']}
  Period:   {event['period_label']}
  Question: {event['question_ko']}
  Summary:  {event['summary_ko']}
  Outcome:  {event['outcome_ko']}

  Timeline entries (the page's outline — do NOT simply retell these):
{timeline_text}

  People already linked to this event (spell any of these EXACTLY as written here,
  which is how their dictionary cards spell them):
  {', '.join(people) if people else '(none)'}

{body_state}
{assigned}
WHAT TO DO
1. Research the assigned topic. Wikipedia first for the routine facts, then at least TWO
   sources outside Wikipedia. A section that could have been written from the Wikipedia
   article alone is not worth the run. Four to six research calls is the budget; stop
   when you have the two outside sources, not when you run out of leads.
2. Call `commulingo_gap_report` ONCE with the people, terms and documents your section
   leans on that this site does not have or covers too thinly.
   {"Already queued for this event, do not re-file: " + ", ".join(queued) if queued else
    "Nothing is queued for this event yet."}
3. Call `{WRITE_TOOL}` with action='create', a heading that tells the reader what the part
   is about, and a body of {EVENT_SECTION_TARGET[0]}-{EVENT_SECTION_TARGET[1]} Korean
   characters with the English twin carrying the same claims. Send ONLY this section,
   never the whole body.

   The section text goes in the tool call and nowhere else. Do not write the draft into
   your reply first: a reply holding the draft saves nothing and the run is wasted.

One run, one event, one write."""


async def run_once(forced_id: str = "", lane: int = 0, lanes: int = 1, skeleton: bool = False, brief: str = "") -> dict:
    config = maintainer.load_config()
    if not config.get("enabled"):
        return {"status": "skipped", "reason": "maintainer disabled"}

    from runtime_tools.commulingo_people import direct_apply_enabled
    from tool_gateway.inference import resolve_agent_inference_policy

    if not direct_apply_enabled():
        raise RuntimeError("config/commulingo_people.json direct_apply must be true")

    if not forced_id and not skeleton and select_event(skeleton=True) is not None:
        # A hand-seeded skeleton (empty summary) outranks section work: the
        # section walker's outline IS the timeline, which a skeleton does not
        # have yet, and the frontend hides summaryless events, so until the
        # card is written the page the seeder asked for does not exist. This
        # also means seeding a row is all a human has to do — no manual
        # --skeleton run needed anymore.
        skeleton = True
    event = select_event(forced_id, lane, lanes, skeleton=skeleton)
    if not event:
        return {
            "status": "skipped",
            "reason": f"every event in this lane already has {TARGET_SECTIONS}+ body sections",
        }

    before = completed_run_count()
    # Highest suggestion id this lane had written before the run, so the tally
    # below counts only what this run added.
    before_max_id = int(((db_query_one(
        "SELECT COALESCE(MAX(id), 0) AS id FROM commulingo_agent_suggestions WHERE suggested_by = %(s)s",
        {"s": SUGGESTED_BY},
    ) or {}).get("id")) or 0)
    spec = get_agent("commulingo_event_curator")
    policy = resolve_agent_inference_policy(spec)
    tools, handlers = spec.filter_tools(TOOLS, TOOL_HANDLERS)
    for name in maintainer.NARROW_WRITE_TOOLS:
        if name in handlers:
            handlers[name] = maintainer.build_retrying_write_handler(handlers[name])
    binding = resolve_agent_tool_loop(spec, policy)

    task = build_skeleton_task(event) if skeleton else build_task(event, brief)
    ctx = new_run_context(
        interface="autonomous", agent_name=spec.name, is_owner=True,
        scope_type="maintenance_job", scope_id=f"commulingo_events:{SUGGESTED_BY}",
    )
    total_cost = 0.0
    total_rounds = 0
    result = ""
    for attempt in range(1, max(1, ATTEMPTS) + 1):
        tracker: dict = {}
        # Each attempt is a fresh conversation, so the retry cannot say "use what
        # you already found" — that research is gone. What it can do is cut the
        # searching that made the first attempt run out of room to write.
        retry_note = "" if attempt == 1 else (
            "\n\nRETRY: the previous attempt ended without saving anything, which is the "
            "one way this run can fail. Almost always the cause is writing the draft into "
            "the reply instead of into the tool call. This time: at most four research "
            "calls, then write. The section text belongs in the `body` argument of "
            f"`{WRITE_TOOL}` and nowhere else."
        )
        with caller_scope(ctx):
            result = await binding.chat(
                [{"role": "user", "content": task + retry_note}],
                client=binding.client,
                model=binding.model,
                tools=tools,
                tool_handlers=handlers,
                system_prompt=spec.render_prompt(provider=binding.render_provider),
                max_rounds=policy.max_rounds,
                max_tokens=policy.max_output_tokens,
                max_input_tokens=policy.max_input_tokens,
                recover_input_via_tools=True,
                continue_on_length=policy.max_output_continuations > 0,
                max_length_continuations=policy.max_output_continuations,
                budget_usd=policy.budget_usd,
                budget_tracker=tracker,
                agent_name=spec.name,
                finalization_tools=spec.finalization_tools,
                terminal_tools=spec.terminal_tools,
                **binding.reasoning,
            )
        total_cost += float(tracker.get("total_cost") or 0.0)
        total_rounds += int(tracker.get("rounds_used") or 0)
        # The run is done when the BODY is written, not when anything is. Since
        # the curator can also link people, a plain "did the count move" test
        # would call it finished on a run that linked three names and never
        # wrote the section it was commissioned for.
        if writes_since(before_max_id).get("body", 0):
            break
        logger.warning(
            "event attempt %d/%d ended without a body write: %s",
            attempt, ATTEMPTS, str(result)[:300],
        )

    after = completed_run_count()
    fresh = db_query_one(
        "SELECT body_ko, body_en FROM commulingo_history_events WHERE id = %(id)s",
        {"id": event["id"]},
    ) or {}
    summary = {
        "event": event["id"],
        "model": binding.model,
        "cost_usd": round(total_cost, 4),
        "rounds": total_rounds,
        "attempts": attempt,
        "sections_before": len(event_section_headings(event["body_ko"] or "")),
        "sections_after": len(event_section_headings(fresh.get("body_ko") or "")),
        "body_ko_chars": len(fresh.get("body_ko") or ""),
        "gaps_open": len(open_gaps(event["id"])),
    }
    if after == before:
        # Nothing written and nothing inconsistent: report the barren run and exit
        # clean, so the next tick is the retry.
        return {"status": "no_edit", "result": str(result)[:500], **summary}

    # A run writes one piece of the event and may also link the people that
    # piece is about, so "exactly one edit" is no longer the shape to check for.
    # What has to stay true is that the run produced one body/card write; the
    # person links are supporting work and are counted, not forbidden.
    counts = writes_since(before_max_id)
    summary["links_made"] = counts.get("links", 0)
    if counts.get("body", 0) != 1:
        raise RuntimeError(
            f"expected one body or card write, got {counts.get('body', 0)} (all writes: {counts})"
        )
    if counts.get("other"):
        raise RuntimeError(f"run wrote {counts['other']} edit(s) of a type it should not touch")
    edit = latest_lane_edit()
    if not edit or edit.get("status") != "approved":
        raise RuntimeError(f"applied edit was not approved: {edit}")
    return {"status": "applied", "edit": edit, **summary}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default="", help="Force one event id instead of selecting.")
    parser.add_argument("--lane", type=int, default=0, help="This copy's partition index.")
    parser.add_argument("--lanes", type=int, default=1, help="How many copies share the dictionary.")
    parser.add_argument("--brief", default="",
                        help="Commission one specific section instead of the next in the walk.")
    parser.add_argument("--skeleton", action="store_true",
                        help="Write question/summary/outcome/timeline for an event that has none.")
    parser.add_argument("--print-candidate", action="store_true",
                        help="Print the event that would be selected, without calling the model.")
    args = parser.parse_args()

    if args.print_candidate:
        event = select_event(args.event, args.lane, args.lanes, skeleton=args.skeleton)
        print(json.dumps(
            {
                "event": (event or {}).get("id"),
                "sections": event_section_headings((event or {}).get("body_ko") or ""),
            },
            ensure_ascii=False, indent=2,
        ))
        return 0

    # Per-partition lock: a lane must not overlap itself, but the other lanes are
    # working on other events and are meant to run at the same time.
    lock_file = Path(f"{LOCK_PATH}.{args.lane}of{args.lanes}").open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.info("another events run is active; exiting")
        return 0
    result = asyncio.run(run_once(args.event, args.lane, args.lanes, skeleton=args.skeleton, brief=args.brief))
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
