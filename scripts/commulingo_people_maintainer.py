#!/usr/bin/env python3
"""Run one bounded CommuLingo people-dictionary maintenance cycle.

The script deterministically selects one sparse existing person (or periodically asks for
one missing person), then gives only that task to the dedicated DeepSeek V4 Pro curator.
Each stage exposes only its read tools and the narrow terminal write tools it can use.
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

# Distinguish unattended writes in revisions/suggestion provenance. This must be set before
# runtime_tools.registry imports runtime_tools.commulingo_people.
os.environ.setdefault("COMMULINGO_SUGGESTED_BY", "commulingo-maintainer")

from agents import get_agent
from bot_config import _deepseek_anthropic_client, _resolve_deepseek_model
from claude_loop import chat_with_tools
from db import query as db_query, query_one as db_query_one
from runtime_tools.commulingo_people import (
    DENSE_SENTENCE_CHARS, FIELD_LIMITS, _dedup_key, sentence_budget,
)
from runtime_tools.registry import TOOLS, TOOL_HANDLERS
from tool_gateway.security import CallerContext, caller_scope

logger = logging.getLogger("commulingo_people_maintainer")

CONFIG_PATH = PROJECT_ROOT / "config" / "commulingo_maintainer.json"
LOCK_PATH = Path("/tmp/leninbot-commulingo-maintainer.lock")
STATE_PATH = PROJECT_ROOT / "data" / "commulingo_maintainer_state.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    defaults = {
        "enabled": True,
        "mode": "auto",
        "new_person_every": 8,
        "recent_days": 30,
        # Cards with basic gaps (empty bio/epithet/moment, no career, role,
        # citizenship, event link, or section) age back in on this much
        # shorter cooldown so one-step-per-run enrichment can actually finish
        # a card; the long recent_days cooldown only throttles complete cards,
        # where a forced re-pick would just accrete filler edits.
        "incomplete_recent_days": 2,
        "new_person_cooldown_runs": 6,
        # A card the curator could not enrich steps aside for this many enrich
        # runs. Only an applied edit writes a revision, so the DB cooldown above
        # never fires for a failed run: 미하일 코즐롭스키 had no findable sources,
        # and the hourly lane re-picked him three hours straight, burning nine
        # 16-round attempts (most of that day's spend) before one finally landed.
        "enrich_failure_cooldown_runs": 6,
        # Parallel-lane switch: when false, the dedicated new-person lane
        # (COMMULINGO_SUGGESTED_BY=commulingo-maintainer-new) no-ops so all
        # maintenance effort concentrates on enriching existing cards.
        "new_lane_enabled": True,
        # Same switch for the glossary lane (commulingo_terms_maintainer.py).
        "term_lane_enabled": True,
        # Pause this semantic category without narrowing all other enrichment.
        "enrich_non_soviet_revolutionaries": True,
        # New-person discovery is independently focusable.
        "new_person_focus": "all",
        # Era groups the absence roster covers. Empty derives it from the focus
        # via ROSTER_GROUPS_BY_FOCUS; an explicit list overrides that, which is
        # the lever for widening the roster back out if duplicates climb.
        "roster_groups": [],
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
    cfg["incomplete_recent_days"] = max(1, int(cfg["incomplete_recent_days"]))
    cfg["new_person_cooldown_runs"] = max(0, int(cfg["new_person_cooldown_runs"]))
    cfg["enrich_failure_cooldown_runs"] = max(0, int(cfg["enrich_failure_cooldown_runs"]))
    cfg["new_lane_enabled"] = bool(cfg["new_lane_enabled"])
    cfg["term_lane_enabled"] = bool(cfg["term_lane_enabled"])
    cfg["enrich_non_soviet_revolutionaries"] = bool(cfg["enrich_non_soviet_revolutionaries"])
    if cfg["new_person_focus"] not in {"all", "soviet_institutions", "old_regime"}:
        raise ValueError("new_person_focus must be all, soviet_institutions, or old_regime")
    if not isinstance(cfg["roster_groups"], list):
        raise ValueError("roster_groups must be a list of group ids")
    cfg["roster_groups"] = [str(g) for g in cfg["roster_groups"]]
    return cfg


def _clean_rejected(raw) -> list:
    """Keep only well-formed rejection entries, newest last, bounded."""
    if not isinstance(raw, list):
        return []
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "").strip()
        if label:
            out.append({"label": label, "existing_id": str(entry.get("existing_id") or "")})
    return out[-REJECTED_MEMORY:]


def _clean_failed(raw) -> list:
    """Keep only well-formed enrich-failure cooldown entries."""
    if not isinstance(raw, list):
        return []
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        person_id = str(entry.get("id") or "").strip()
        try:
            runs_left = int(entry.get("runs_left", 0))
        except (TypeError, ValueError):
            continue
        if person_id and runs_left > 0:
            out.append({"id": person_id, "runs_left": runs_left})
    return out[-FAILED_MEMORY:]


def load_state(path: Path = STATE_PATH) -> dict:
    # This function is the state whitelist: a key that is not rebuilt here is
    # dropped on the next run, so anything meant to survive must be listed.
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"new_cooldown_remaining": 0, "rejected_candidates": [], "failed_candidates": []}
    return {
        "new_cooldown_remaining": max(0, int(raw.get("new_cooldown_remaining", 0))),
        "rejected_candidates": _clean_rejected(raw.get("rejected_candidates")),
        "failed_candidates": _clean_failed(raw.get("failed_candidates")),
    }


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def completed_run_count() -> int:
    row = db_query_one(
        """SELECT COUNT(*)::int AS n
             FROM commulingo_agent_suggestions
            WHERE suggested_by = 'commulingo-maintainer'
              AND status = 'approved'"""
    )
    return int((row or {}).get("n") or 0)


def choose_mode(config: dict, requested: str | None = None, state: dict | None = None) -> str:
    mode = requested or config["mode"]
    if mode != "auto":
        return mode
    if int((state or {}).get("new_cooldown_remaining", 0)) > 0:
        return "enrich"
    every = int(config["new_person_every"])
    if every > 0 and (completed_run_count() + 1) % every == 0:
        return "new"
    return "enrich"


# Length is PRESCRIBED as SENTENCES and VERIFIED in characters. Prescribing a character
# band produced the stilted cards: a writer cannot land natural Korean inside a
# 30-character window without padding or truncating to hit the number, so the character
# constants below are backstops, never a target to write toward.
#
# Forbidding the count outright was the overcorrection. The curator could not tell a
# 4-character overflow from a whole surplus sentence, so it shaved and resubmitted at the
# same length: on 2026-07-31 one card went 427 -> 401 -> 384 -> 384 against the 380
# ceiling, five paid rounds to place one card, and person_create sat at 57% rejected for
# the day. The prompt now says to count before calling and scales the remedy to the size
# of the overflow — cut a sentence when it is a sentence, cut a clause when it is a clause.
#
# The sentence count is DERIVED from the ceiling (sentence_budget), not chosen next to
# it. Prescribing 4-5 sentences under a ceiling that pays for 4 made every major
# old-regime card arrive 15-20% over — 17 of 19 curator rejections in the 2026-07-29
# window were a fifth bio sentence (bio.ko 414-480 against 380, bio.en 950-1000 against
# 900), each one a paid round spent to learn the arithmetic. The retry then landed at
# 342-376 characters, so the four-sentence card was the outcome either way; the fifth
# sentence only ever bought the rejection.
#
# `moment` is the pull-quote on the person LIST card, so its budget is rendered lines.
# Measured in the real card at 1280/768/390px: 44-85 chars -> 2 lines, 86-127 -> 3,
# 128-170 -> 4. It had no hard ceiling at all, which is how 308-character moments got in.
MAJOR_PROMINENCE = 4
MINOR_PROMINENCE_MAX = 1
# Hard ceilings: refuse past here, do not write toward here. Sourced from the
# same FIELD_LIMITS table that generates the tool schemas and save checks —
# never restate these numbers locally.
BIO_HARD_CEILING = FIELD_LIMITS["bio"][0]
MOMENT_HARD_CEILING = FIELD_LIMITS["moment"][0]
BIO_SENTENCE_BUDGET = sentence_budget("bio")
MAJOR_BIO_SENTENCES = (
    f"{BIO_SENTENCE_BUDGET - 1}-{BIO_SENTENCE_BUDGET} sentences "
    f"(a {BIO_SENTENCE_BUDGET + 1}th only when every sentence stays short: at "
    f"~{DENSE_SENTENCE_CHARS[0]} Korean characters a dense sentence, "
    f"{BIO_SENTENCE_BUDGET} is what the {BIO_HARD_CEILING}-character ceiling pays for)"
)
STANDARD_BIO_SENTENCES = "2-4 sentences"
MINOR_BIO_SENTENCES = "1-2 sentences"
MOMENT_SENTENCES = "one sentence, two at most"
# Stub detection for candidate selection: a bio shorter than this for the tier reads as
# unfinished and moves the card up the enrich queue. Never shown to the curator.
MAJOR_BIO_STUB = 100
STANDARD_BIO_STUB = 80
MINOR_BIO_STUB = 50
# Detail sections a person card may carry. Today 683 of 700 people sit at 5 or
# fewer; the thirteen above that are the 2026-07 runaway, not a standard to keep.
MAX_SECTIONS = 12

# Nationality flag codes the frontend has vendored SVGs for (data/commulingo/flag-icons.js).
# The curator must pick citizenship_code / nationalOrigin code from this set or the card shows no flag.
NATIONALITY_CODES = (
    "soviet, russia, ukraine, georgia, armenia, azerbaijan, belarus, kazakhstan, "
    "latvia, lithuania, estonia, uzbekistan, moldova, turkmenistan, tajikistan, "
    "kyrgyzstan, poland, finland, germany, east-germany, austria, hungary, czechia, "
    "romania, bulgaria, yugoslavia, france, italy, spain, uk, netherlands, usa, "
    "cuba, argentina, chile, china, japan, india, turkey, vietnam, north-korea, south-korea, "
    "albania, angola, burkina-faso, congo, ghana, guinea-bissau, indonesia, "
    "mozambique, peru, trinidad, portugal, brazil, el-salvador, grenada, guyana, "
    "nicaragua, south-africa, tanzania, ireland, slovakia, czechoslovakia, korea, martinique"
)


def person_tier(candidate: dict) -> dict:
    """Derive the bio-length band for a candidate from its prominence signals."""
    prominence = int(candidate.get("event_count") or 0) + int(candidate.get("office_count") or 0)
    is_major = prominence >= MAJOR_PROMINENCE
    is_minor = prominence <= MINOR_PROMINENCE_MAX
    tier = "major" if is_major else "minor" if is_minor else "standard"
    if is_major:
        sentences, stub = MAJOR_BIO_SENTENCES, MAJOR_BIO_STUB
    elif is_minor:
        sentences, stub = MINOR_BIO_SENTENCES, MINOR_BIO_STUB
    else:
        sentences, stub = STANDARD_BIO_SENTENCES, STANDARD_BIO_STUB
    return {
        "tier": tier,
        "is_major": is_major,
        "is_minor": is_minor,
        "prominence": prominence,
        "bio_sentences": sentences,
        "bio_stub": stub,
    }


def select_sparse_person(
    recent_days: int,
    forced_id: str = "",
    incomplete_recent_days: int | None = None,
    enrich_non_soviet_revolutionaries: bool = True,
    exclude_ids: list[str] | None = None,
) -> dict | None:
    params = {
        "recent_days": recent_days,
        "incomplete_days": incomplete_recent_days if incomplete_recent_days is not None else recent_days,
        "forced_id": forced_id.strip(),
        "exclude_ids": list(exclude_ids or []),
        "major_prom": MAJOR_PROMINENCE,
        "minor_max": MINOR_PROMINENCE_MAX,
        "major_stub": MAJOR_BIO_STUB,
        "std_stub": STANDARD_BIO_STUB,
        "minor_stub": MINOR_BIO_STUB,
        "max_sections": MAX_SECTIONS,
        "enrich_non_soviet_revolutionaries": enrich_non_soviet_revolutionaries,
    }
    rows = db_query(
        """SELECT p.id, p.group_id, p.name_ko, p.name_en,
                  LENGTH(COALESCE(p.bio_ko, '')) AS bio_chars,
                  CASE WHEN COALESCE(p.epithet_ko, '') = '' THEN 0 ELSE 1 END AS has_epithet,
                  COUNT(DISTINCT c.id)::int AS career_count,
                  COUNT(DISTINCT s.id)::int AS section_count,
                  COUNT(DISTINCT ep.event_id)::int AS event_count,
                  COUNT(DISTINCT o.id)::int AS office_count,
                  p.citizenship_code AS citizenship_code,
                  p.origin_code AS origin_code,
                  CASE WHEN COALESCE(p.moment_ko, '') = '' THEN 0 ELSE 1 END AS has_moment,
                  CASE WHEN r.person_id IS NULL THEN 0 ELSE 1 END AS has_role
             FROM commulingo_people p
             LEFT JOIN commulingo_person_career_entries c ON c.person_id = p.id
             LEFT JOIN commulingo_person_sections s ON s.person_id = p.id
             LEFT JOIN commulingo_person_roles r ON r.person_id = p.id
             LEFT JOIN commulingo_history_event_people ep ON ep.person_id = p.id
             LEFT JOIN commulingo_office_rows o ON o.person_id = p.id
            WHERE (%(forced_id)s = '' OR p.id = %(forced_id)s)
              AND NOT (p.id = ANY(%(exclude_ids)s))
              AND (
                    %(forced_id)s <> ''
                    OR %(enrich_non_soviet_revolutionaries)s
                    OR NOT EXISTS (
                        SELECT 1
                          FROM commulingo_person_roles excluded_role
                         WHERE excluded_role.person_id = p.id
                           AND excluded_role.category_id = 'non-soviet-revolutionary'
                    )
                  )
            GROUP BY p.id, p.group_id, p.name_ko, p.name_en, p.bio_ko, p.epithet_ko, p.moment_ko,
                     p.citizenship_code, p.origin_code, r.person_id
           HAVING %(forced_id)s <> ''
               OR (COALESCE((
                    SELECT MAX(rev.created_at) FROM commulingo_people_revisions rev
                     WHERE (rev.entity_id = p.id OR rev.entity_id LIKE p.id || '/%%')
                       AND rev.changed_by LIKE 'commulingo-maintainer%%'
                  ), TIMESTAMP '-infinity') < NOW() - (
                    CASE WHEN COALESCE(p.bio_ko, '') = '' OR COALESCE(p.epithet_ko, '') = ''
                              OR COALESCE(p.moment_ko, '') = ''
                              OR COALESCE(p.citizenship_code, '') = ''
                              OR r.person_id IS NULL
                              OR COUNT(DISTINCT c.id) = 0
                              OR COUNT(DISTINCT ep.event_id) = 0
                              OR COUNT(DISTINCT s.id) = 0
                         THEN %(incomplete_days)s ELSE %(recent_days)s END * INTERVAL '1 day')
                -- A card with every field filled falls through the enrich checklist to
                -- its last step, "add one more section", which had no ceiling. When the
                -- nationality step could not be satisfied in 2026-07 because the flag
                -- code list carried no entry for their countries, the card stayed
                -- "incomplete" on the 2-day cooldown and kept coming back: 카브랄 took 19
                -- sections on 7/14 and 14 more on 7/15, and twelve other people ran to
                -- 19-36. Once a card is complete AND at MAX_SECTIONS there is nothing
                -- left to commission, so it leaves the queue instead of growing.
                -- The event step is deliberately excluded from this test. It is the
                -- other step a card can be unable to satisfy: the events dictionary
                -- holds 24 Soviet events, so 카브랄, 루뭄바, 응크루마 and the rest of the
                -- non-Soviet revolutionaries have nothing honest to link to, and the
                -- checklist says never to force a weak connection. Requiring an event
                -- link here would keep exactly the runaway cards in the queue with
                -- every remaining step refusing to fire.
                AND NOT (
                        COUNT(DISTINCT s.id) >= %(max_sections)s
                    AND COALESCE(p.bio_ko, '') <> '' AND COALESCE(p.epithet_ko, '') <> ''
                    AND COALESCE(p.moment_ko, '') <> ''
                    AND COALESCE(p.citizenship_code, '') <> ''
                    AND r.person_id IS NOT NULL
                    AND COUNT(DISTINCT c.id) > 0
                ))
            ORDER BY
                  CASE WHEN COALESCE(p.bio_ko, '') = '' OR COALESCE(p.epithet_ko, '') = ''
                         OR COUNT(DISTINCT c.id) = 0 OR r.person_id IS NULL THEN 0 ELSE 1 END ASC,
                  CASE WHEN COALESCE(p.citizenship_code, '') = '' THEN 0 ELSE 1 END ASC,
                  -- Stub bios and missing moments move up the queue. A bio that merely
                  -- runs long does NOT: the shorter standard applies to what the curator
                  -- writes from now on, and the cards already written stay as they are
                  -- rather than becoming a trimming campaign.
                  CASE WHEN LENGTH(COALESCE(p.bio_ko, '')) <
                             CASE WHEN COUNT(DISTINCT ep.event_id) + COUNT(DISTINCT o.id) >= %(major_prom)s
                                       THEN %(major_stub)s
                                  WHEN COUNT(DISTINCT ep.event_id) + COUNT(DISTINCT o.id) <= %(minor_max)s
                                       THEN %(minor_stub)s
                                  ELSE %(std_stub)s END
                         OR COALESCE(p.moment_ko, '') = '' THEN 0 ELSE 1 END ASC,
                  CASE WHEN COUNT(DISTINCT ep.event_id) = 0 THEN 0 ELSE 1 END ASC,
                  COUNT(DISTINCT s.id) ASC,
                  CASE WHEN COUNT(DISTINCT c.id) <= 1 THEN 0 ELSE 1 END ASC,
                  CASE WHEN r.person_id IS NULL THEN 0 ELSE 1 END ASC,
                  LENGTH(COALESCE(p.bio_ko, '')) ASC,
                  p.sort_order ASC
            LIMIT 1""",
        params,
    )
    return rows[0] if rows else None


CARD_STYLE_GUIDANCE = (
    "LENGTH AND STYLE (keep every card consistent):\n"
    "- `bio` and `moment` are the paragraph and the pull-quote on the person LIST card, not the "
    "detail page. Anything that needs more room belongs in a person_section, which the detail "
    "page renders in full.\n"
    "- COUNT THE CHARACTERS OF EVERY CAPPED FIELD BEFORE YOU CALL THE TOOL, and fix an "
    "over-length draft in the same turn you wrote it. The ceilings are stated with each field in "
    "the tool schema. A rejected write costs a full paid round and teaches you nothing you could "
    "not have checked yourself, so the count is part of drafting, not something the save "
    "discovers for you.\n"
    f"- Compose bio to a SENTENCE COUNT, then check it against the ceiling. A MAJOR figure (head "
    f"of state, party leader, or someone central to many events) gets {MAJOR_BIO_SENTENCES}; a "
    f"STANDARD figure {STANDARD_BIO_SENTENCES}; a MINOR/obscure figure {MINOR_BIO_SENTENCES}. "
    f"These are CEILINGS AND RANGES, NOT QUOTAS. Write the sentences the subject actually "
    f"warrants and stop there: a major figure with thin documented material gets a short bio, and "
    f"filling to the limit with restatement is padding. Inflating an obscure functionary's bio is "
    f"a defect. The enrich task states the tier for the specific person. {BIO_HARD_CEILING} "
    f"Korean characters is a hard ceiling the save rejects, not a target. How you fix an "
    f"over-length draft depends on how far over it is. Over by roughly a sentence "
    f"(~{DENSE_SENTENCE_CHARS[0]} Korean characters) or more: it is one sentence too many, so "
    f"drop a whole sentence or move the material into a person_section — trimming clauses out of "
    f"{BIO_SENTENCE_BUDGET + 1} dense sentences does not recover that much, it just produces a "
    f"cramped card that overflows anyway. Over by less: it is one clause too long, so delete the "
    f"weakest clause outright. Do NOT reword at the same length and resubmit — rephrasing a "
    f"384-character bio into a different 384 characters spends a round and changes nothing. Give "
    f"the English bio comparable substance (its {FIELD_LIMITS['bio'][1]}-character ceiling binds "
    f"at the same sentence count and is counted the same way). Never leave a one-line stub.\n"
    "- Every new person requires both citizenship and nationalOrigin. nationalOrigin may equal "
    "citizenship; never omit it because the two match or because a distinct background is not "
    "documented. Apply the editorial defaults below instead of storing a blank.\n"
    "- The bio states who the person essentially was and why they matter — their core "
    "significance and defining tension. It is NOT a chronological list of posts, dates, and "
    "ministries: the detailed career timeline already lists positions year by year, so do not "
    "duplicate that in the bio. Capture the essence, not the resume.\n"
    f"- `moment` is a pull-quote, not a paragraph: {MOMENT_SENTENCES} (with an English "
    f"equivalent), capturing ONE defining scene, line, or turn. One sentence is the norm and the "
    f"second is for when the scene genuinely needs its turn; if it still needs more to explain "
    f"itself, the scene is wrong — pick a sharper one rather than adding sentences. "
    f"{MOMENT_HARD_CEILING} Korean characters is a hard ceiling the save rejects, and a quotation "
    f"longer than that is excerpted to its sharpest clause with '…' or traded for a shorter one. "
    f"Check the length of a quote before building the moment around it.\n"
    "- The epithet stays a short phrase — one clause. A characterization that needs a second "
    "clause after a dash belongs in the bio. If any field runs long, tighten it rather than pad "
    "it.\n"
    "- nationalOrigin editorial policy for people born in territory now within Ukraine under the "
    "Russian Empire or USSR: `ukraine` requires documented Ukrainian self-identification, "
    "Ukrainian parentage/family, or a substantive tie to Ukrainian national culture or autonomy; "
    "documented Polish background uses `poland`; otherwise use `russia`. Birthplace and work in "
    "the Ukrainian SSR alone never suffice. Jewish ancestry alone does not create a separate "
    "nationalOrigin category in this dictionary."
)


def build_task(mode: str, candidate: dict | None) -> str:
    if mode == "new":
        return """MODE: NEW PERSON

Identify one historically important person missing from CommuLingo whose inclusion would
materially improve coverage of revolutionary or Soviet history. Inspect list_groups,
list_categories and list_offices, then search_people under the proposed name and aliases to
prove there is no duplicate. Research with the free wiki_search/wiki_get tools first (Russian
Wikipedia when available), then open at least one source outside Wikipedia — an archive or
document collection, marxists.org, a journal or university page, or a published reference work —
before writing. Wikipedia alone is acceptable only for a minor figure whose card is routine dates
and posts. Never cite cyber-lenin.com or any page on this site. Create one
complete bilingual person card with a correct group and one primary role, including a bio and a
one-line `moment` that follow the style rules below, via one `commulingo_person_create` call.
Then stop. Event links can be added by a later enrichment run. Do not create a section or office
row in this run.

""" + CARD_STYLE_GUIDANCE
    if not candidate:
        raise RuntimeError("no eligible sparse person found")
    tier = person_tier(candidate)
    sentences = tier["bio_sentences"]
    # A bio that merely runs longer than today's standard is NOT a reason to rewrite it:
    # the shorter standard governs what gets written from here on, and the cards already
    # in the dictionary stay as they are. Only a stub or a resume-style bio is a defect.
    rewrite_trigger = (
        "reads as a list of posts and dates rather than the person's core significance, or is "
        "so thin it reads as unfinished"
    )
    if tier["is_major"]:
        tier_line = (
            f"- prominence tier: MAJOR (linked events + offices = {tier['prominence']}). "
            f"Korean bio: {sentences} — a ceiling, not a quota."
        )
        bio_step = (
            f"3. BIO DEPTH/STYLE: else if the Korean bio {rewrite_trigger}, rewrite it in both "
            f"languages in essence-first style as one person update. This is a MAJOR figure, so it "
            f"may carry {sentences}: use the room only for significance, defining tensions, and "
            f"historical weight the sources actually support. Stop when the subject is covered — "
            f"restating the career timeline to reach six sentences is padding. Keep the facts."
        )
    elif tier["is_minor"]:
        tier_line = (
            f"- prominence tier: MINOR (linked events + offices = {tier['prominence']}). "
            f"Korean bio {sentences} — keep it short."
        )
        bio_step = (
            f"3. BIO SIZE/STYLE: else if the Korean bio {rewrite_trigger}, rewrite it in both "
            f"languages in essence-first style as one person update. This is a MINOR figure: "
            f"{sentences} is the whole budget, so trim to the essentials. Keep the facts."
        )
    else:
        tier_line = (
            f"- prominence tier: standard. Korean bio {sentences} as the material warrants."
        )
        bio_step = (
            f"3. BIO SIZE/STYLE: else if the Korean bio {rewrite_trigger}, rewrite it in both "
            f"languages in essence-first style as one person update — {sentences}, keep the facts, "
            f"just refocus."
        )
    return f"""MODE: ENRICH EXISTING PERSON

Target exactly this person and no one else:
- id: {candidate['id']}
- Korean name: {candidate['name_ko']}
- English name: {candidate['name_en']}
- group: {candidate['group_id']}
- current Korean bio length: {candidate['bio_chars']} characters
- has epithet: {bool(candidate['has_epithet'])}
- career rows: {candidate['career_count']}
- detail sections: {candidate['section_count']}
- linked historical events: {candidate['event_count']}
- has moment: {bool(candidate['has_moment'])}
- has primary role: {bool(candidate['has_role'])}
- citizenship flag code: {candidate.get('citizenship_code') or '(unset)'}
- national/ethnic background flag code: {candidate.get('origin_code') or '(unset)'}
{tier_line}

Call get_person and get_sections first, then make exactly one available narrow write, choosing the
first step below that applies.

CONTENT PRESERVATION: before writing anything, compare your draft against what the card
already holds. Default to building on the existing content — keep accurate, in-style facts and
prose, and fold them into any rewrite rather than regenerating a field from scratch. You MAY
remove or replace existing material, but only on a judged reason (factually wrong, contradicted
by sources, duplicated elsewhere on the card, or clearly violating the style rules) — never as
an accidental side effect of a rewrite. If the existing content already satisfies a step, that
step does not apply; move to the next one.
Korean Soviet-history prose uses `그루지야`, not `조지아`; modern citizenship labels may still
use `조지아`.
1. BASIC COMPLETENESS: if bio or epithet is empty, career has no rows, or the primary role is
   missing, one `commulingo_person_update` that fills every such missing basic field (bio and moment written
   to the style rules below). Do not create a section in that case.
2. NATIONALITY: else if either the citizenship or nationalOrigin flag code is unset,
   set both in one
   `commulingo_person_update`. Provide `citizenship` — the state whose citizenship the person actually held
   (for most figures here the Soviet Union `soviet`; use `russian-empire`-era figures' successor
   state, i.e. still `soviet` if they lived into the USSR, otherwise `russia`; foreign
   revolutionaries take their own state) — and always provide
   `nationalOrigin`, the person's documented national or ethnic background (e.g. `georgia` for
   Stalin, `poland` for Dzerzhinsky). Never infer nationalOrigin from birthplace: Karl Radek was
   born in present-day Ukraine but was Polish, and Nikolai Yezhov was born in Lithuania but is
   classified here as Russian. Citizenship is the primary flag and comes first; nationalOrigin is
   secondary. Never omit nationalOrigin: it may equal citizenship when the person's documented
   background matches their state. If sources do not establish a different background, use the
   reviewed editorial default rather than leaving it blank. Citizenship
   is NOT where the person happened to die or emigrate to:
   a Soviet official who died in exile abroad is still `soviet`. It also drives the native-name
   script check, so a wrong code turns the card's own-script name line wrong too. Each value is {{"code": <one of: {NATIONALITY_CODES}>, "label":
   {{"ko": "...", "en": "..."}}}}. Never invent a code outside that list. Example:
   patch={{"citizenship": {{"code": "soviet", "label": {{"ko": "소련", "en": "Soviet Union"}}}},
   "nationalOrigin": {{"code": "georgia", "label": {{"ko": "그루지야", "en": "Georgia"}}}}}}.
{bio_step}
4. MOMENT: else if `has moment` is false, add a bilingual `moment` as one person update — one
   sentence, two at most, one scene.
5. EVENTS: else if linked historical events is zero, inspect list_events and the most plausible
   get_event records. When one event connection is clearly supported, create exactly one
   `commulingo_event_link`; never force a weak connection.
6. SECTION: else, if this card has fewer than {MAX_SECTIONS} sections, find the single most
   valuable missing topic and add one substantial bilingual section via `commulingo_section_save`
   (one topic, roughly 350-700 Korean characters plus equivalent English) when no section covers
   it. A card already at {MAX_SECTIONS} sections is finished: say so and make no write rather than
   splitting a covered topic to have something to add. This card has {candidate['section_count']}.
Preserve every wholesale field exactly when updating. Research with the free wiki_search/wiki_get
tools first (Russian Wikipedia when available), then open at least one source outside Wikipedia —
an archive or document collection, marxists.org, a journal or university page, or a published
reference work — before writing anything beyond routine dates and posts. Never cite
cyber-lenin.com or any page on this site. Make one narrow write call and stop.

""" + CARD_STYLE_GUIDANCE


# Which era groups a candidate may be drawn from under each selection focus.
# The focus text already forbids picking outside these, so the roster does not
# have to prove absence there and can stop carrying those names.
ROSTER_GROUPS_BY_FOCUS = {
    "soviet_institutions": ("bolshevik", "stalin-era", "thaw", "perestroika"),
    # Era boundaries are fuzzy: an 1905-era activist may already be filed under
    # the revolution generation or as a non-Soviet revolutionary, so the roster
    # keeps those groups to prove absence.
    "old_regime": ("old-regime", "bolshevik", "international-revolutionary"),
}


def roster_groups_for_focus(config: dict) -> tuple[str, ...]:
    """Group ids the roster should cover, from explicit config or the focus."""
    configured = config.get("roster_groups") or ()
    if configured:
        return tuple(str(g) for g in configured)
    return ROSTER_GROUPS_BY_FOCUS.get(str(config.get("new_person_focus") or "all"), ())


def registered_person_roster(groups: tuple[str, ...] | None = None) -> str:
    """Cards already in the dictionary, grouped by era under a heading each.

    Discovery used to prove absence with at most six search_people calls
    against ~700 cards, so it kept proposing people who were already filed:
    85% of the lane's fallbacks were `candidate duplicates existing person`
    after all three attempts, each one a paid agent loop. Showing the roster
    up front turns absence into something the curator can read off.

    One comma-joined block of 1,095 names was 94% of the discovery prompt and
    still did not stop the duplicates, because a wall that size is not read.
    Splitting it by era gives the model the one section its candidate would
    belong to. `groups` narrows it further to the eras the current focus can
    actually select from; the omitted eras are named, not silently dropped, so
    absence there is never mistaken for a gap.
    """
    rows = db_query(
        """SELECT g.id AS group_id, g.title_ko, g.range_label, p.name_ko, p.name_en
             FROM commulingo_people p
             JOIN commulingo_people_groups g ON g.id = p.group_id
            ORDER BY g.sort_order, g.id, p.name_ko"""
    )
    wanted = set(groups or ())
    sections: dict[str, dict] = {}
    omitted: dict[str, int] = {}
    for row in rows:
        gid = row["group_id"]
        if wanted and gid not in wanted:
            omitted[row["title_ko"]] = omitted.get(row["title_ko"], 0) + 1
            continue
        section = sections.setdefault(
            gid, {"title": row["title_ko"], "range": row["range_label"], "names": []},
        )
        section["names"].append(
            f"{row['name_ko']}({row['name_en']})" if row["name_en"] else str(row["name_ko"])
        )

    blocks = [
        f"### {s['title']} ({s['range']}) — {len(s['names'])}명\n" + ", ".join(s["names"])
        for s in sections.values()
    ]
    if omitted:
        blocks.append(
            "### 아래 시대는 지금 선택 대상이 아니므로 명단을 싣지 않는다\n"
            + ", ".join(f"{title} {count}명" for title, count in omitted.items())
            + "\n이 시대에서 고르지 말 것. 여기 없다는 사실이 빈자리라는 뜻은 아니다."
        )
    return "\n\n".join(blocks)


_PERSON_CLAIMS: list = []


def claim_person(person_id: str) -> bool:
    """Reserve a person for this run, against the other lanes on this host.

    Each lane holds only its own run lock, and select_sparse_person is
    deterministic, so two lanes running at once pick the SAME sparsest person —
    and then, both being asked for "the single most valuable missing topic",
    write the same section under two slugs. 예이젠시테인 got 몽타주 이론 twice
    65 seconds apart that way, 가톱스키 곡물 시장 연구 twice in 30. The claim is a
    plain flock: the lanes are systemd units on one host, so a file is enough
    and it cannot outlive the process that took it.
    """
    handle = Path(f"/tmp/leninbot-commulingo-person-{person_id}.lock").open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return False
    _PERSON_CLAIMS.append(handle)  # released when the process exits
    return True


def select_claimable_person(
    config: dict, candidate_id: str, attempts: int = 5,
    exclude_ids: list[str] | None = None,
) -> dict | None:
    """The sparsest person this run can hold exclusively, or None."""
    skipped: list[str] = list(exclude_ids or [])
    for _ in range(attempts):
        candidate = select_sparse_person(
            config["recent_days"], candidate_id, config["incomplete_recent_days"],
            config["enrich_non_soviet_revolutionaries"], exclude_ids=skipped,
        )
        if not candidate:
            return None
        if claim_person(candidate["id"]):
            return candidate
        logger.info("another lane holds %s; taking the next candidate", candidate["id"])
        skipped.append(candidate["id"])
    return None


def rejected_candidate_note(rejected: list | None) -> str:
    """The people this lane already proposed and had rejected as duplicates.

    The full roster is ~700 comma-joined names, and the model kept picking the
    same handful out of it anyway: chelomey 15 times in 48h, kuzmin 17, novikov
    15, yangel 9 — 354 rejected commulingo_candidate_select calls, each one a
    discovery loop paid for and thrown away. Every attempt started from an empty
    conversation, so nothing carried over. This list is short, specific, and
    survives the run, which is what the roster cannot be.
    """
    if not rejected:
        return ""
    lines = "\n".join(
        f"- {entry['label']}" + (f" — already filed as {entry['existing_id']}" if entry.get("existing_id") else "")
        for entry in rejected[-REJECTED_MEMORY:]
    )
    return f"""
ALREADY PROPOSED AND REJECTED AS DUPLICATES — do not propose any of these again:
{lines}
"""


def build_discovery_task(
    new_person_focus: str = "all",
    rejected: list | None = None,
    roster_groups: tuple[str, ...] | None = None,
) -> str:
    focus_instruction = ""
    if new_person_focus == "soviet_institutions":
        focus_instruction = """
CURRENT SELECTION FOCUS:
- Do not select a non-Soviet revolutionary or a foreign socialist-bloc leader.
- Select only a person important to a Soviet institution: the CPSU party apparatus,
  USSR state administration, security/intelligence, armed forces, diplomacy, economic
  planning/industry, science/space, or Soviet cultural administration.
- Prefer a documented office or command that belongs in one of the institution timelines
  returned by list_offices. State that institution and office explicitly in the coverage reason.
"""
    elif new_person_focus == "old_regime":
        focus_instruction = """
CURRENT SELECTION FOCUS:
- Select only a person from the world before Bolshevik power, one whose card belongs in
  the "구체제와 그 도전자들" (old-regime) era group: tsarist statesmen, officials and
  pillars of the old order, or the revolutionaries and thinkers who fought it —
  Decembrists, Narodniks, early Russian Marxists, SRs, Mensheviks, anarchists, figures
  of 1905, and of 1917 before October.
- If the person's principal historical role lies after the Bolshevik seizure of power,
  they are out of scope for now.
- State in the coverage reason why the person matters to pre-October revolutionary or
  imperial history.
"""
    return """MODE: NEW PERSON DISCOVERY ONLY

Do not create or edit anything in this stage. Inspect list_groups, list_categories and
list_offices, then use search_people under a proposed name and aliases to prove the person
is absent. Prefer a historically important gap in revolutionary or Soviet history.

ALREADY IN THE DICTIONARY — anyone below is NOT a gap, and neither is the same person
under another transliteration. The roster is split by era: read the section your candidate
would belong to before proposing them. A candidate that turns out to be registered is
rejected and the run is wasted.

""" + registered_person_roster(roster_groups) + "\n" + focus_instruction + """
""" + rejected_candidate_note(rejected) + """
Do not survey the whole dictionary: consider at most three plausible people, select the first verified
gap, and stop searching. You have """ + str(DISCOVERY_SEARCH_BUDGET) + """ search_people lookups for the whole stage and each
result tells you how many are left; the roster above, not those lookups, is the primary proof of
absence. Open one reliable biographical source. Finish by calling
`commulingo_candidate_select` with the missing
person's id, Korean/English names, coverage reason, and source URL. That typed call is the ONLY
valid completion of this stage.
"""


def validate_discovered_candidate(candidate: dict) -> dict:
    required = ("id", "name_ko", "name_en", "reason", "source_url")
    if not isinstance(candidate, dict) or any(not str(candidate.get(k) or "").strip() for k in required):
        raise ValueError("discovery candidate is missing required fields")
    candidate = {k: str(candidate[k]).strip() for k in required}
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", candidate["id"]):
        raise ValueError("candidate id is not lowercase kebab-case")
    if not candidate["source_url"].startswith(("https://", "http://")):
        raise ValueError("candidate source_url must be HTTP(S)")
    duplicate = db_query_one(
        """SELECT p.id FROM commulingo_people p
             WHERE p.id = %(id)s
                OR LOWER(p.name_en) = LOWER(%(name_en)s)
                OR p.name_ko = %(name_ko)s
                OR EXISTS (
                    SELECT 1
                      FROM commulingo_person_aliases a
                     WHERE a.person_id = p.id
                       AND ((a.lang = 'en' AND LOWER(a.alias) = LOWER(%(name_en)s))
                         OR (a.lang = 'ko' AND a.alias = %(name_ko)s))
                )
             LIMIT 1""",
        candidate,
    )
    if duplicate:
        raise ValueError(f"candidate duplicates existing person {duplicate['id']}")

    # The SQL above compares raw strings, so a different transliteration reads
    # as a different person (C.L.R. James over C. L. R. James). Re-check on the
    # spelling-insensitive key the create tool uses, here where rejecting still
    # leaves the run a chance to pick someone else.
    key_ko, key_en = _dedup_key(candidate["name_ko"]), _dedup_key(candidate["name_en"])
    for row in db_query("SELECT id, name_ko, name_en FROM commulingo_people"):
        if _dedup_key(row["name_ko"]) == key_ko or _dedup_key(row["name_en"]) == key_en:
            raise ValueError(
                f"candidate duplicates existing person {row['id']} "
                f"({row['name_ko']}) under a different spelling"
            )
    return candidate


COMMULINGO_CANDIDATE_SELECT_TOOL = {
    "name": "commulingo_candidate_select",
    "description": (
        "Finish discovery by submitting one verified missing-person candidate. "
        "Duplicate candidates are rejected so another can be selected in the same run."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "description": "Lowercase kebab-case slug."},
            "name_ko": {"type": "string"},
            "name_en": {"type": "string"},
            "reason": {"type": "string"},
            "source_url": {"type": "string"},
        },
        "required": ["id", "name_ko", "name_en", "reason", "source_url"],
    },
}


def build_candidate_select_handler(box: dict):
    async def _select(**candidate) -> str:
        try:
            selected = validate_discovered_candidate(candidate)
        except ValueError as exc:
            box["last_error"] = str(exc)
            # Remember only "already registered" rejections. A malformed slug or a
            # non-HTTP source is a fixable mistake about a person who may still be a
            # real gap; a duplicate is a settled fact worth carrying to the next run.
            if "duplicates existing person" in str(exc):
                record_rejected_candidate(box, candidate, str(exc))
            raise
        box["candidate"] = selected
        box.pop("last_error", None)
        return json.dumps({"ok": True, "candidate": selected}, ensure_ascii=False)
    return _select


def record_rejected_candidate(box: dict, candidate: dict, reason: str) -> None:
    """Append one duplicate rejection to this run's list, newest last."""
    name_ko = str(candidate.get("name_ko") or "").strip()
    name_en = str(candidate.get("name_en") or "").strip()
    if not (name_ko or name_en):
        return
    rejected = box.setdefault("rejected", [])
    label = f"{name_ko}({name_en})" if name_ko and name_en else (name_ko or name_en)
    existing = str(reason).rsplit("existing person ", 1)[-1].split()[0] if "existing person " in reason else ""
    entry = {"label": label, "existing_id": existing}
    if entry not in rejected:
        rejected.append(entry)


# Six lookups against ~700 cards, so the roster is the primary absence check and
# these are for confirming one or two finalists. The cap itself is not the problem;
# hitting it silently was. `discovery_search_limit` raised on every call past the
# sixth, and the model answered by calling again — 1,362 rejected commulingo_people
# calls in 48h, every one a paid round. Now each search reports what is left, and
# the call past the cap returns a normal payload naming the next action instead of
# an error, because an error is what invited the retry.
DISCOVERY_SEARCH_BUDGET = 6

# How many duplicate rejections to carry between runs. Bounded so the note stays
# short enough to actually be read; the oldest fall off and may be re-proposed
# once, which is the acceptable cost of not growing the prompt without limit.
REJECTED_MEMORY = 40

# How many people can sit on an enrich-failure cooldown at once. Each entry is a
# card the queue is stepping over, so the cap keeps a bad stretch from emptying
# the queue; the oldest entry is dropped rather than growing the list.
FAILED_MEMORY = 40


def build_bounded_discovery_handlers(read_handlers: dict, box: dict) -> dict:
    """Bound duplicate lookups so discovery cannot spend every round surveying."""
    handlers = dict(read_handlers)
    people_handler = handlers["commulingo_people"]

    async def _bounded_people(**kwargs) -> str:
        if kwargs.get("action") != "search_people":
            return await people_handler(**kwargs)
        used = int(box.get("search_count") or 0)
        if used >= DISCOVERY_SEARCH_BUDGET:
            return json.dumps({
                "search_budget_spent": True,
                "remaining": 0,
                "next_action": "commulingo_candidate_select",
                "note": (
                    f"All {DISCOVERY_SEARCH_BUDGET} name/alias lookups are used. Further "
                    "search_people calls return this same notice without querying. Absence "
                    "is already established by the roster in the task; select the best "
                    "verified gap now."
                ),
            }, ensure_ascii=False)
        box["search_count"] = used + 1
        body = await people_handler(**kwargs)
        remaining = DISCOVERY_SEARCH_BUDGET - box["search_count"]
        return (
            f"{body}\n\n[discovery budget: {remaining} of {DISCOVERY_SEARCH_BUDGET} "
            f"name/alias lookups left"
            + ("; spend the rest on nobody new and call commulingo_candidate_select]"
               if remaining == 0 else "]")
        )

    handlers["commulingo_people"] = _bounded_people
    return handlers


def build_new_person_task(candidate: dict) -> str:
    return f"""MODE: NEW PERSON CREATION

Create exactly this pre-verified missing person and no one else:
- id: {candidate['id']}
- Korean name: {candidate['name_ko']}
- English name: {candidate['name_en']}
- coverage reason: {candidate['reason']}
- starting source: {candidate['source_url']}

Re-check search_people for the exact names, fetch the starting source, inspect groups and
roles, then create one complete bilingual card, including a bio and a one-line `moment` that
follow the style rules below. Use ONLY the canonical person patch keys
documented by commulingo_person_create: givenName, familyName (given
name + surname ONLY, patronymic never embedded), bio, epithet, fate, role, groupId, years,
aliases, career, cyrillic, cyrillicPatronymic, patronymic, moment, scenes, sortOrder,
including both citizenship and nationalOrigin. Both nationality fields are mandatory for every
new person; nationalOrigin may equal citizenship but must never be omitted.
The person schema requires citizenship and nationalOrigin.
nationalOrigin means documented national/ethnic background, never birthplace.
Never replace a rejected complete card with a minimal placeholder create; correct the invalid field shape and retry the complete card.
Make exactly one commulingo_person_create call and stop. Keep citations top-level, outside fields.

""" + CARD_STYLE_GUIDANCE


NARROW_WRITE_TOOLS = frozenset({
    "commulingo_person_create", "commulingo_person_update",
    "commulingo_section_save", "commulingo_event_link", "commulingo_term_create",
})
PEOPLE_ENRICH_WRITE_TOOLS = frozenset({
    "commulingo_person_update", "commulingo_section_save", "commulingo_event_link",
})


def build_retrying_write_handler(handler):
    """Turn structured write errors into tool failures so the model can retry."""
    async def _validated_write(**kwargs) -> str:
        result = await handler(**kwargs)
        if not str(result).startswith("Error:"):
            return result
        raw = str(result).removeprefix("Error:").strip()
        try:
            payload = json.loads(raw)
            error = payload.get("error") or {}
            code = str(error.get("code") or "unknown")
            retryable = bool(error.get("retryable", True))
            message = str(error.get("message") or raw)
        except (json.JSONDecodeError, AttributeError):
            code, retryable, message = "legacy_error", True, raw
        raise ValueError(
            f"commulingo_write[{code}; retryable={str(retryable).lower()}]: {message}"
        )
    return _validated_write


def latest_maintainer_edit() -> dict | None:
    return db_query_one(
        """SELECT id, target_type, target_id, action, status, confidence, created_at
             FROM commulingo_agent_suggestions
            WHERE suggested_by = 'commulingo-maintainer'
            ORDER BY id DESC LIMIT 1"""
    )


async def _call_curator_stage(
    *, task: str, spec, model: str, tools: list, handlers: dict,
    policy, stage: str, expect_edit: bool, before_count: int,
    finalization_tools: list[str], terminal_tools: list[str],
    candidate_box: dict | None = None,
) -> tuple[str, dict, dict | None]:
    from tool_gateway.inference import resolve_inference_extra

    reasoning = resolve_inference_extra(policy, "deepseek")
    attempts = 1 + max(0, int(policy.max_output_continuations))
    total_cost = 0.0
    total_rounds = 0
    last_result = ""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        tracker: dict = {}
        if candidate_box is not None:
            candidate_box["search_count"] = 0
        prior_detail = str(last_error or "")[:800]
        retry_note = "" if attempt == 1 else (
            "\n\nRETRY: The prior attempt produced no usable terminal edit/candidate. "
            "Do not emit DSML markup or commentary-only output. Complete the required "
            "terminal action now using canonical tool arguments."
            + (f" Prior failure: {prior_detail}" if prior_detail else "")
            # Each attempt is a fresh conversation, so without this the second
            # attempt only learns the last rejection and can re-propose the one
            # before it. Carry every duplicate found so far.
            + rejected_candidate_note((candidate_box or {}).get("rejected"))
        )
        try:
            last_result = await chat_with_tools(
                [{"role": "user", "content": task + retry_note}],
                client=_deepseek_anthropic_client,
                model=model,
                tools=tools,
                tool_handlers=handlers,
                system_prompt=(
                    spec.render_prompt(provider="deepseek")
                    + ("\n\nDISCOVERY-STAGE EXCEPTION: do not edit; finish only with commulingo_candidate_select." if not expect_edit else "")
                ),
                max_rounds=policy.max_rounds,
                max_tokens=policy.max_output_tokens,
                max_input_tokens=policy.max_input_tokens,
                recover_input_via_tools=True,
                continue_on_length=policy.max_output_continuations > 0,
                max_length_continuations=policy.max_output_continuations,
                budget_usd=policy.budget_usd,
                budget_tracker=tracker,
                agent_name=spec.name,
                finalization_tools=finalization_tools,
                terminal_tools=terminal_tools,
                thinking=reasoning.get("thinking"),
                output_config=reasoning.get("output_config"),
            )
            total_cost += float(tracker.get("total_cost") or 0.0)
            total_rounds += int(tracker.get("rounds_used") or 0)
            after = completed_run_count()
            if expect_edit:
                if after == before_count + 1:
                    return last_result, {"total_cost": total_cost, "rounds_used": total_rounds}, None
                if after != before_count:
                    raise RuntimeError(
                        f"unexpected edit count change during {stage}: {before_count} -> {after}"
                    )
                raise RuntimeError(f"{stage} produced no edit: {last_result[:500]}")
            candidate = (candidate_box or {}).get("candidate")
            if not candidate:
                rejection = (candidate_box or {}).get("last_error")
                detail = f": {rejection}" if rejection else ""
                raise RuntimeError(f"{stage} produced no typed candidate selection{detail}")
            return last_result, {"total_cost": total_cost, "rounds_used": total_rounds}, candidate
        except Exception as exc:
            last_error = exc
            logger.warning(
                "%s attempt %d/%d failed without an applied edit: %s",
                stage, attempt, attempts, exc,
            )
            if completed_run_count() != before_count:
                raise
    raise RuntimeError(
        f"{stage} failed after {attempts} attempts: {last_error}; result={last_result[:500]}"
    )


async def run_once(*, mode: str, candidate_id: str, config: dict) -> dict:
    if not config["enabled"]:
        return {"status": "disabled"}

    from runtime_tools.commulingo_people import direct_apply_enabled
    from tool_gateway.inference import resolve_agent_inference_policy

    if not direct_apply_enabled():
        raise RuntimeError("config/commulingo_people.json direct_apply must be true")

    state = load_state()
    requested_mode = mode if mode != "auto" else None
    chosen_mode = choose_mode(config, requested_mode, state)
    before = completed_run_count()

    spec = get_agent("commulingo_curator")
    policy = resolve_agent_inference_policy(spec)
    tools, handlers = spec.filter_tools(TOOLS, TOOL_HANDLERS)
    expected = set(spec.tools)
    available = {str(t.get("name") or "") for t in tools} & set(handlers)
    if expected != available:
        raise RuntimeError(f"curator toolset incomplete: missing={sorted(expected - available)}")
    handlers = dict(handlers)
    for name in NARROW_WRITE_TOOLS:
        handlers[name] = build_retrying_write_handler(handlers[name])
    read_tools = [t for t in tools if t.get("name") not in NARROW_WRITE_TOOLS]
    read_handlers = {k: v for k, v in handlers.items() if k not in NARROW_WRITE_TOOLS}

    def stage_tools(write_names: frozenset[str]) -> tuple[list, dict]:
        selected_tools = read_tools + [t for t in tools if t.get("name") in write_names]
        selected_handlers = {
            **read_handlers,
            **{name: handlers[name] for name in write_names},
        }
        return selected_tools, selected_handlers

    model = _resolve_deepseek_model(spec.model or "deepseek_pro")
    ctx = CallerContext(interface="agent", agent_name=spec.name, is_owner=True)

    candidate = None
    discovery = None
    fallback_error = None
    tracker = {"total_cost": 0.0, "rounds_used": 0}
    with caller_scope(ctx):
        if chosen_mode == "new":
            # Seeded from disk and merged back below whether or not the stage
            # succeeds, so a duplicate proved in this run is not re-proposed in
            # the next one. The box is built outside the try for that reason.
            candidate_box: dict = {"rejected": list(state.get("rejected_candidates") or [])}
            try:
                discovery_tools = [*read_tools, COMMULINGO_CANDIDATE_SELECT_TOOL]
                discovery_handlers = {
                    **build_bounded_discovery_handlers(read_handlers, candidate_box),
                    "commulingo_candidate_select": build_candidate_select_handler(candidate_box),
                }
                discovery_result, discovery_tracker, candidate = await _call_curator_stage(
                    task=build_discovery_task(
                        config["new_person_focus"], candidate_box["rejected"],
                        roster_groups_for_focus(config),
                    ), spec=spec, model=model,
                    tools=discovery_tools, handlers=discovery_handlers, policy=policy,
                    stage="new-person discovery", expect_edit=False, before_count=before,
                    finalization_tools=["commulingo_candidate_select"],
                    terminal_tools=["commulingo_candidate_select"], candidate_box=candidate_box,
                )
                discovery = {"candidate": candidate, "result": discovery_result}
                tracker["total_cost"] += discovery_tracker["total_cost"]
                tracker["rounds_used"] += discovery_tracker["rounds_used"]
                create_tools, create_handlers = stage_tools(frozenset({"commulingo_person_create"}))
                result, create_tracker, _ = await _call_curator_stage(
                    task=build_new_person_task(candidate), spec=spec, model=model,
                    tools=create_tools, handlers=create_handlers, policy=policy,
                    stage="new-person creation", expect_edit=True, before_count=before,
                    finalization_tools=["commulingo_person_create"],
                    terminal_tools=["commulingo_person_create"],
                )
                tracker["total_cost"] += create_tracker["total_cost"]
                tracker["rounds_used"] += create_tracker["rounds_used"]
                state["new_cooldown_remaining"] = 0
            except Exception as exc:
                if completed_run_count() != before:
                    raise
                fallback_error = str(exc)
                logger.error("new-person path failed; falling back to enrich: %s", exc)
                state["new_cooldown_remaining"] = int(config["new_person_cooldown_runs"])
                state["rejected_candidates"] = candidate_box["rejected"][-REJECTED_MEMORY:]
                save_state(state)
                chosen_mode = "enrich_fallback"
            else:
                state["rejected_candidates"] = candidate_box["rejected"][-REJECTED_MEMORY:]

        if chosen_mode in {"enrich", "enrich_fallback"}:
            # Age the failure cooldowns by one run, then step over whoever is
            # still serving one. A forced --candidate overrides the cooldown:
            # it is the operator saying to try this card now.
            for entry in state["failed_candidates"]:
                entry["runs_left"] -= 1
            state["failed_candidates"] = [
                e for e in state["failed_candidates"] if e["runs_left"] > 0
            ]
            cooling = [] if candidate_id else [e["id"] for e in state["failed_candidates"]]
            if cooling:
                logger.info("skipping %d card(s) on enrich-failure cooldown: %s",
                            len(cooling), ", ".join(cooling))
            candidate = select_claimable_person(config, candidate_id, exclude_ids=cooling)
            if candidate is None:
                # Every person was already touched within the cooldown window, or the
                # few that were not are held by the other lane right now. Idling until
                # candidates age back in is the correct, zero-cost outcome.
                save_state(state)
                return {
                    "status": "skipped",
                    "reason": (
                        f"no claimable candidate outside the cooldown "
                        f"({config['incomplete_recent_days']}d incomplete / {config['recent_days']}d complete)"
                    ),
                    "mode": chosen_mode,
                    "fallback_error": fallback_error,
                }
            task = build_task("enrich", candidate)
            enrich_tools, enrich_handlers = stage_tools(PEOPLE_ENRICH_WRITE_TOOLS)
            enrich_terminals = sorted(PEOPLE_ENRICH_WRITE_TOOLS)
            try:
                result, enrich_tracker, _ = await _call_curator_stage(
                    task=task, spec=spec, model=model,
                    tools=enrich_tools, handlers=enrich_handlers,
                    policy=policy, stage=chosen_mode, expect_edit=True, before_count=before,
                    finalization_tools=enrich_terminals,
                    terminal_tools=enrich_terminals,
                )
            except Exception:
                # All attempts are spent and nothing was written. Record the
                # cooldown before the unit dies on the traceback — otherwise the
                # run leaves no trace at all and the next hour picks this same
                # unresearchable card and spends the same rounds on it.
                cooldown = int(config["enrich_failure_cooldown_runs"])
                if cooldown > 0 and completed_run_count() == before:
                    state["failed_candidates"] = (
                        [e for e in state["failed_candidates"] if e["id"] != candidate["id"]]
                        + [{"id": candidate["id"], "runs_left": cooldown}]
                    )[-FAILED_MEMORY:]
                    save_state(state)
                    logger.warning("%s is on enrich-failure cooldown for %d run(s)",
                                   candidate["id"], cooldown)
                raise
            tracker["total_cost"] += enrich_tracker["total_cost"]
            tracker["rounds_used"] += enrich_tracker["rounds_used"]
            if chosen_mode == "enrich" and state.get("new_cooldown_remaining", 0) > 0:
                state["new_cooldown_remaining"] -= 1

    after = completed_run_count()
    # Every completed stage lands exactly one write (the burst-bug guard).
    max_edits = 1
    if not (before + 1 <= after <= before + max_edits):
        raise RuntimeError(
            f"expected 1..{max_edits} applied edits, count changed {before} -> {after}; result={result[:500]}"
        )
    save_state(state)
    edit = latest_maintainer_edit()
    if not edit or edit.get("status") != "approved":
        raise RuntimeError("applied edit was not recorded as approved")
    return {
        "status": "applied",
        "mode": chosen_mode,
        "candidate": candidate and candidate.get("id"),
        "model": model,
        "cost_usd": round(float(tracker.get("total_cost") or 0.0), 4),
        "rounds": int(tracker.get("rounds_used") or 0),
        "edit": edit,
        "discovery": discovery,
        "fallback_error": fallback_error,
        "cooldown_remaining": state.get("new_cooldown_remaining", 0),
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
        print(json.dumps(select_sparse_person(
            config["recent_days"], args.candidate, config["incomplete_recent_days"],
            config["enrich_non_soviet_revolutionaries"],
        ), ensure_ascii=False, default=str, indent=2))
        return 0
    result = asyncio.run(run_once(mode=args.mode, candidate_id=args.candidate, config=config))
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
