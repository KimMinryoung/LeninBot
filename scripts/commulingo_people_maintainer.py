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
        # Parallel-lane switch: when false, the dedicated new-person lane
        # (COMMULINGO_SUGGESTED_BY=commulingo-maintainer-new) no-ops so all
        # maintenance effort concentrates on enriching existing cards.
        "new_lane_enabled": True,
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
    cfg["new_lane_enabled"] = bool(cfg["new_lane_enabled"])
    return cfg


def load_state(path: Path = STATE_PATH) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"new_cooldown_remaining": 0}
    return {"new_cooldown_remaining": max(0, int(raw.get("new_cooldown_remaining", 0)))}


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


# Prominence tiering scales the bio-length band to a person's historical weight
# (linked events + offices held). A stub bio for a major figure like Stalin reads as a defect,
# and an inflated bio for an obscure functionary is equally a defect. The personal band by tier:
#   major (prominence >= MAJOR_PROMINENCE):   MAJOR_BIO_FLOOR..320   (fill it out)
#   standard (prominence 2..3):               120..320              (whatever the material warrants)
#   minor (prominence <= MINOR_PROMINENCE_MAX): MINOR_BIO_FLOOR..MINOR_BIO_CEILING (keep it short, under 120)
MAJOR_PROMINENCE = 4
MINOR_PROMINENCE_MAX = 1
MAJOR_BIO_FLOOR = 260
STANDARD_BIO_FLOOR = 120
BIO_CEILING = 320
MINOR_BIO_FLOOR = 60
MINOR_BIO_CEILING = 115

# Nationality flag codes the frontend has vendored SVGs for (data/commulingo/flag-icons.js).
# The curator must pick citizenship_code / origin_code from this set or the card shows no flag.
NATIONALITY_CODES = (
    "soviet, russia, ukraine, georgia, armenia, azerbaijan, belarus, kazakhstan, "
    "latvia, lithuania, estonia, uzbekistan, moldova, turkmenistan, tajikistan, "
    "kyrgyzstan, poland, finland, germany, east-germany, austria, hungary, czechia, "
    "romania, bulgaria, france, italy, spain, uk, netherlands, usa, "
    "cuba, argentina, chile, china, japan, india, turkey, vietnam, north-korea, south-korea, "
    "albania, angola, burkina-faso, congo, ghana, guinea-bissau, indonesia, "
    "mozambique, peru, trinidad, portugal"
)


def person_tier(candidate: dict) -> dict:
    """Derive the bio-length band for a candidate from its prominence signals."""
    prominence = int(candidate.get("event_count") or 0) + int(candidate.get("office_count") or 0)
    is_major = prominence >= MAJOR_PROMINENCE
    is_minor = prominence <= MINOR_PROMINENCE_MAX
    tier = "major" if is_major else "minor" if is_minor else "standard"
    if is_major:
        bio_floor, bio_ceiling = MAJOR_BIO_FLOOR, BIO_CEILING
    elif is_minor:
        bio_floor, bio_ceiling = MINOR_BIO_FLOOR, MINOR_BIO_CEILING
    else:
        bio_floor, bio_ceiling = STANDARD_BIO_FLOOR, BIO_CEILING
    return {
        "tier": tier,
        "is_major": is_major,
        "is_minor": is_minor,
        "prominence": prominence,
        "bio_floor": bio_floor,
        "bio_ceiling": bio_ceiling,
    }


def select_sparse_person(recent_days: int, forced_id: str = "", incomplete_recent_days: int | None = None) -> dict | None:
    params = {
        "recent_days": recent_days,
        "incomplete_days": incomplete_recent_days if incomplete_recent_days is not None else recent_days,
        "forced_id": forced_id.strip(),
        "major_prom": MAJOR_PROMINENCE,
        "minor_max": MINOR_PROMINENCE_MAX,
        "major_floor": MAJOR_BIO_FLOOR,
        "std_floor": STANDARD_BIO_FLOOR,
        "ceiling": BIO_CEILING,
        "minor_floor": MINOR_BIO_FLOOR,
        "minor_ceiling": MINOR_BIO_CEILING,
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
            GROUP BY p.id, p.group_id, p.name_ko, p.name_en, p.bio_ko, p.epithet_ko, p.moment_ko,
                     p.citizenship_code, p.origin_code, r.person_id
           HAVING %(forced_id)s <> ''
               OR COALESCE((
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
            ORDER BY
                  CASE WHEN COALESCE(p.bio_ko, '') = '' OR COALESCE(p.epithet_ko, '') = ''
                         OR COUNT(DISTINCT c.id) = 0 OR r.person_id IS NULL THEN 0 ELSE 1 END ASC,
                  CASE WHEN COALESCE(p.citizenship_code, '') = '' THEN 0 ELSE 1 END ASC,
                  CASE WHEN LENGTH(COALESCE(p.bio_ko, '')) <
                             CASE WHEN COUNT(DISTINCT ep.event_id) + COUNT(DISTINCT o.id) >= %(major_prom)s
                                       THEN %(major_floor)s
                                  WHEN COUNT(DISTINCT ep.event_id) + COUNT(DISTINCT o.id) <= %(minor_max)s
                                       THEN %(minor_floor)s
                                  ELSE %(std_floor)s END
                         OR LENGTH(COALESCE(p.bio_ko, '')) >
                             CASE WHEN COUNT(DISTINCT ep.event_id) + COUNT(DISTINCT o.id) <= %(minor_max)s
                                  THEN %(minor_ceiling)s ELSE %(ceiling)s END
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
    "- Korean bio length scales with the person's historical weight, and both a too-thin and a "
    "too-long bio are defects. A MAJOR figure (head of state, party leader, or someone central to "
    "many events) fills roughly 260-320 characters — a thin 150-180 character bio for that weight "
    "reads as a defect; 320 is the hard ceiling, never run past it. A STANDARD figure sits in "
    "120-320 as the material warrants. A MINOR/obscure figure stays short, under 120 characters "
    "(roughly 60-115) — inflating an obscure functionary's bio is a defect. The enrich task states "
    "the exact target band for the specific person. Give the English bio comparable substance. "
    "Never leave a one-line stub.\n"
    "- The bio states who the person essentially was and why they matter — their core "
    "significance and defining tension. It is NOT a chronological list of posts, dates, and "
    "ministries: the detailed career timeline already lists positions year by year, so do not "
    "duplicate that in the bio. Capture the essence, not the resume.\n"
    "- `moment`: one or two vivid lines of about 40-140 Korean characters (with an English "
    "equivalent) capturing a single defining scene or turn; do not exceed ~150 characters.\n"
    "- The epithet stays a short phrase. If any field runs long, tighten it rather than pad it."
)


def build_task(mode: str, candidate: dict | None) -> str:
    if mode == "new":
        return """MODE: NEW PERSON

Identify one historically important person missing from CommuLingo whose inclusion would
materially improve coverage of revolutionary or Soviet history. Inspect list_groups,
list_categories and list_offices, then search_people under the proposed name and aliases to
prove there is no duplicate. Research with the free wiki_search/wiki_get tools first (Russian
Wikipedia when available); use the paid web_search only for facts Wikipedia lacks. One opened
source is enough for routine card facts; use a second only for disputed or consequential claims. Create one
complete bilingual person card with a correct group and one primary role, including a bio and a
one-line `moment` that follow the style rules below. Make exactly one
`commulingo_edit(target_type='person', action='create', ...)` call and stop. Do not create a
section or office row in this run.

""" + CARD_STYLE_GUIDANCE
    if not candidate:
        raise RuntimeError("no eligible sparse person found")
    tier = person_tier(candidate)
    bio_floor = tier["bio_floor"]
    bio_ceiling = tier["bio_ceiling"]
    if tier["is_major"]:
        tier_line = (
            f"- prominence tier: MAJOR (linked events + offices = {tier['prominence']}). "
            f"Target Korean bio {bio_floor}-{bio_ceiling} characters."
        )
        bio_step = (
            f"3. BIO DEPTH/STYLE: else if the Korean bio is under {bio_floor} or over {bio_ceiling} "
            f"characters, or reads as a list of posts and dates rather than the person's core "
            f"significance, rewrite it in both languages into the {bio_floor}-{bio_ceiling} band and "
            f"essence-first style as one person update. This is a MAJOR figure: a thin bio is a "
            f"defect, so expand it toward the upper end with added significance, defining tensions, "
            f"and historical weight — never padding, and never exceed {bio_ceiling}. Keep the facts."
        )
    elif tier["is_minor"]:
        tier_line = (
            f"- prominence tier: MINOR (linked events + offices = {tier['prominence']}). "
            f"Target Korean bio {bio_floor}-{bio_ceiling} characters — keep it short."
        )
        bio_step = (
            f"3. BIO SIZE/STYLE: else if the Korean bio is under {bio_floor} or over {bio_ceiling} "
            f"characters, or reads as a list of posts and dates rather than the person's core "
            f"significance, rewrite it in both languages into the {bio_floor}-{bio_ceiling} band and "
            f"essence-first style as one person update. This is a MINOR figure: keep the bio short "
            f"and tight — a long bio here is a defect, so trim to the essentials and do not exceed "
            f"{bio_ceiling}. Keep the facts."
        )
    else:
        tier_line = (
            f"- prominence tier: standard. Target Korean bio {bio_floor}-{bio_ceiling} characters."
        )
        bio_step = (
            f"3. BIO SIZE/STYLE: else if the Korean bio is under {bio_floor} or over {bio_ceiling} "
            f"characters, or reads as a list of posts and dates rather than the person's core "
            f"significance, rewrite the bio in both languages into the target band and essence-first "
            f"style as one person update — keep the facts, just resize and refocus."
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
- origin flag code: {candidate.get('origin_code') or '(unset)'}
{tier_line}

Call get_person and get_sections first, then make exactly one commulingo_edit, choosing the
first step below that applies.

CONTENT PRESERVATION: before writing anything, compare your draft against what the card
already holds. Default to building on the existing content — keep accurate, in-style facts and
prose, and fold them into any rewrite rather than regenerating a field from scratch. You MAY
remove or replace existing material, but only on a judged reason (factually wrong, contradicted
by sources, duplicated elsewhere on the card, or clearly violating the style rules) — never as
an accidental side effect of a rewrite. If the existing content already satisfies a step, that
step does not apply; move to the next one.
1. BASIC COMPLETENESS: if bio or epithet is empty, career has no rows, or the primary role is
   missing, one person update that fills every such missing basic field (bio and moment written
   to the style rules below). Do not create a section in that case.
2. NATIONALITY: else if the citizenship flag code is unset, set the person's nationality in one
   person update. Provide `citizenship` — the state whose citizenship the person actually held
   (for most figures here the Soviet Union `soviet`; use `russian-empire`-era figures' successor
   state, i.e. still `soviet` if they lived into the USSR, otherwise `russia`; foreign
   revolutionaries take their own state) — and, only when it is a DIFFERENT nation, `origin`, the
   birthplace people/nation (e.g. `georgia` for Stalin, `poland` for Dzerzhinsky). Citizenship is
   the primary flag and comes first; origin is secondary. Omit origin when it equals citizenship
   or is genuinely unknown. Each value is {{"code": <one of: {NATIONALITY_CODES}>, "label":
   {{"ko": "...", "en": "..."}}}}. Never invent a code outside that list. Example:
   patch={{"citizenship": {{"code": "soviet", "label": {{"ko": "소련", "en": "Soviet Union"}}}},
   "origin": {{"code": "georgia", "label": {{"ko": "조지아", "en": "Georgia"}}}}}}.
{bio_step}
4. MOMENT: else if `has moment` is false, add a bilingual `moment` (target band) as one person
   update.
5. EVENTS: else if linked historical events is zero, inspect list_events and the most plausible
   get_event records. When one event connection is clearly supported, create exactly one
   history_event_person relation; never force a weak connection.
6. SECTION: else find the single most valuable missing topic and add one substantial bilingual
   `person_section` (one topic, roughly 350-700 Korean characters plus equivalent English) when
   no section covers it.
Preserve every wholesale field exactly when updating. Research with the free wiki_search/wiki_get
tools first (Russian Wikipedia when available); use the paid web_search only for facts Wikipedia
lacks. One opened source is enough for routine card facts; use a second only for disputed or
consequential claims. Make one commulingo_edit call and stop.

""" + CARD_STYLE_GUIDANCE


def build_discovery_task() -> str:
    return """MODE: NEW PERSON DISCOVERY ONLY

Do not create or edit anything in this stage. Inspect list_groups, list_categories and
list_offices, then use search_people under a proposed name and aliases to prove the person
is absent. Prefer a historically important gap in revolutionary or Soviet history. Open one
reliable biographical source. End with exactly one machine-readable line and no text after it:
CANDIDATE_JSON: {"id":"lowercase-kebab-slug","name_ko":"...","name_en":"...",
"reason":"...","source_url":"https://..."}
"""


def parse_discovered_candidate(text: str) -> dict:
    matches = re.findall(r"CANDIDATE_JSON:\s*(\{[^\n]+\})", text or "")
    if not matches:
        raise ValueError("discovery did not return CANDIDATE_JSON")
    candidate = json.loads(matches[-1])
    required = ("id", "name_ko", "name_en", "reason", "source_url")
    if not isinstance(candidate, dict) or any(not str(candidate.get(k) or "").strip() for k in required):
        raise ValueError("discovery candidate is missing required fields")
    candidate = {k: str(candidate[k]).strip() for k in required}
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", candidate["id"]):
        raise ValueError("candidate id is not lowercase kebab-case")
    if not candidate["source_url"].startswith(("https://", "http://")):
        raise ValueError("candidate source_url must be HTTP(S)")
    duplicate = db_query_one(
        """SELECT id FROM commulingo_people
             WHERE id = %(id)s
                OR LOWER(name_en) = LOWER(%(name_en)s)
                OR name_ko = %(name_ko)s
             LIMIT 1""",
        candidate,
    )
    if duplicate:
        raise ValueError(f"candidate duplicates existing person {duplicate['id']}")
    return candidate


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
documented by commulingo_edit: name, bio, epithet, fate, role, groupId, years, aliases,
career, cyrillic, cyrillicPatronymic, patronymic, moment, scenes, sortOrder.
Never replace a rejected complete card with a minimal placeholder create; correct the invalid field shape and retry the complete card.
Make exactly one commulingo_edit(target_type='person', action='create') call and stop.

""" + CARD_STYLE_GUIDANCE


_PERSON_PATCH_KEYS = frozenset({
    "id", "group", "groupId", "sortOrder", "cyrillic",
    "cyrillicPatronymic", "years", "name", "epithet", "bio", "moment",
    "fate", "patronymic", "aliases", "career", "role", "scenes", "office_rows",
    "citizenship", "origin",
})
_SECTION_PATCH_KEYS = frozenset({"slug", "heading", "body", "sortOrder", "sources"})


def normalize_commulingo_patch(target_type: str, target_id: str, patch: dict | None) -> tuple[dict, list[str]]:
    normalized = dict(patch or {})
    repairs: list[str] = []
    if target_type != "person":
        return normalized, repairs

    localized = {
        "name": ("nameKo", "nameEn"),
        "bio": ("bioKo", "bioEn"),
        "epithet": ("epithetKo", "epithetEn"),
        "moment": ("momentKo", "momentEn"),
        "patronymic": ("patronymicKo", "patronymicEn"),
    }
    for canonical, (ko_key, en_key) in localized.items():
        if ko_key in normalized or en_key in normalized:
            current = normalized.get(canonical) if isinstance(normalized.get(canonical), dict) else {}
            normalized[canonical] = {
                "ko": normalized.pop(ko_key, current.get("ko", "")),
                "en": normalized.pop(en_key, current.get("en", "")),
            }
            repairs.append(f"{ko_key}/{en_key}->{canonical}")
    if "yearsLabel" in normalized:
        normalized.setdefault("years", normalized.pop("yearsLabel"))
        repairs.append("yearsLabel->years")
    if "fateKind" in normalized:
        fate = normalized.get("fate") if isinstance(normalized.get("fate"), dict) else {}
        fate["kind"] = normalized.pop("fateKind")
        normalized["fate"] = fate
        repairs.append("fateKind->fate.kind")
    if "category" in normalized or "officeId" in normalized:
        role = normalized.get("role") if isinstance(normalized.get("role"), dict) else {}
        if "officeId" in normalized:
            role["officeId"] = normalized.pop("officeId")
        elif "category" in normalized:
            role["category"] = normalized.pop("category")
        normalized["role"] = role
        repairs.append("category/officeId->role")
    if normalized.get("slug") == target_id:
        normalized.pop("slug", None)
        repairs.append("dropped redundant slug")
    return normalized, repairs


def build_validating_edit_handler(handler):
    async def _validated_edit(
        target_type: str, action: str, target_id: str, sources: list,
        patch: dict | None = None, confidence: float | None = None,
    ) -> str:
        normalized, repairs = normalize_commulingo_patch(target_type, target_id, patch)
        allowed = _PERSON_PATCH_KEYS if target_type == "person" else _SECTION_PATCH_KEYS if target_type == "person_section" else None
        if allowed is not None:
            unknown = sorted(set(normalized) - allowed)
            if unknown:
                raise ValueError(
                    f"maintainer preflight rejected unknown patch key(s): {', '.join(unknown)}. "
                    f"Allowed: {', '.join(sorted(allowed))}. Retry once with canonical keys."
                )
        if repairs:
            logger.warning("auto-corrected commulingo_edit patch once: %s", ", ".join(repairs))
        result = await handler(
            target_type=target_type, action=action, target_id=target_id,
            sources=sources, patch=normalized, confidence=confidence,
        )
        if str(result).startswith("Error:"):
            raise ValueError(str(result))
        return result
    return _validated_edit


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
        retry_note = "" if attempt == 1 else (
            "\n\nRETRY: The prior attempt produced no usable terminal edit/candidate. "
            "Do not emit DSML markup or commentary-only output. Complete the required "
            "terminal action now using canonical tool arguments."
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
                    + ("\n\nDISCOVERY-STAGE EXCEPTION: do not edit in this stage; return only CANDIDATE_JSON." if not expect_edit else "")
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
                finalization_tools=spec.finalization_tools if expect_edit else [],
                terminal_tools=spec.terminal_tools if expect_edit else [],
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
            candidate = parse_discovered_candidate(last_result)
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
    handlers["commulingo_edit"] = build_validating_edit_handler(handlers["commulingo_edit"])
    model = _resolve_deepseek_model(spec.model or "deepseek_pro")
    ctx = CallerContext(interface="agent", agent_name=spec.name, is_owner=True)

    candidate = None
    discovery = None
    fallback_error = None
    tracker = {"total_cost": 0.0, "rounds_used": 0}
    with caller_scope(ctx):
        if chosen_mode == "new":
            try:
                discovery_tools = [t for t in tools if t.get("name") != "commulingo_edit"]
                discovery_handlers = {k: v for k, v in handlers.items() if k != "commulingo_edit"}
                discovery_result, discovery_tracker, candidate = await _call_curator_stage(
                    task=build_discovery_task(), spec=spec, model=model,
                    tools=discovery_tools, handlers=discovery_handlers, policy=policy,
                    stage="new-person discovery", expect_edit=False, before_count=before,
                )
                discovery = {"candidate": candidate, "result": discovery_result}
                tracker["total_cost"] += discovery_tracker["total_cost"]
                tracker["rounds_used"] += discovery_tracker["rounds_used"]
                result, create_tracker, _ = await _call_curator_stage(
                    task=build_new_person_task(candidate), spec=spec, model=model,
                    tools=tools, handlers=handlers, policy=policy,
                    stage="new-person creation", expect_edit=True, before_count=before,
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
                save_state(state)
                chosen_mode = "enrich_fallback"

        if chosen_mode in {"enrich", "enrich_fallback"}:
            candidate = select_sparse_person(
                config["recent_days"], candidate_id, config["incomplete_recent_days"]
            )
            if candidate is None:
                # Every person was already touched within the cooldown window;
                # idling until candidates age back in is the correct, zero-cost outcome.
                return {
                    "status": "skipped",
                    "reason": (
                        f"no candidate outside the cooldown "
                        f"({config['incomplete_recent_days']}d incomplete / {config['recent_days']}d complete)"
                    ),
                    "mode": chosen_mode,
                    "fallback_error": fallback_error,
                }
            task = build_task("enrich", candidate)
            result, enrich_tracker, _ = await _call_curator_stage(
                task=task, spec=spec, model=model, tools=tools, handlers=handlers,
                policy=policy, stage=chosen_mode, expect_edit=True, before_count=before,
            )
            tracker["total_cost"] += enrich_tracker["total_cost"]
            tracker["rounds_used"] += enrich_tracker["rounds_used"]
            if chosen_mode == "enrich" and state.get("new_cooldown_remaining", 0) > 0:
                state["new_cooldown_remaining"] -= 1

    after = completed_run_count()
    if after != before + 1:
        raise RuntimeError(
            f"expected exactly one applied edit, count changed {before} -> {after}; result={result[:500]}"
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
        print(json.dumps(select_sparse_person(config["recent_days"], args.candidate, config["incomplete_recent_days"]), ensure_ascii=False, default=str, indent=2))
        return 0
    result = asyncio.run(run_once(mode=args.mode, candidate_id=args.candidate, config=config))
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
