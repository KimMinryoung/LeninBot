"""commulingo.people — CommuLingo dictionary read + narrow write tools.

The people dictionary at cyber-lenin.com/commulingo/people is DB-backed
(commulingo_* tables in the main Postgres; see
frontend/dev_docs/commulingo_people_handoff.md).

The public tool surface contains one read/search tool and target-specific narrow
write tools for people, sections, event links, and terms. Every write requires
citations and shares the same validation and transaction core.

Writes run in one of two modes, switched by
config/commulingo_people.json → {"direct_apply": true|false} (mtime-cached,
no restart needed):

- direct_apply=true  — the edit is applied at once by the CommuLingo editorial
  service, which records a revision snapshot and an auto-approved row in
  commulingo_agent_suggestions so provenance (sources, confidence) is always
  on record.
- direct_apply=false — the edit is staged as a pending suggestion; the
  operator reviews it with scripts/commulingo_suggestions.py.

CommuLingo is a separate service: every write and approval goes through its
admin MCP (commulingo/mcp_client.py, frontend dev_docs/commulingo-admin-mcp.md).
This module only checks curator policy first and phrases the result for the
model. The frontend serves a periodically refreshed snapshot (default 60 seconds).
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import logging
import os
import re
from datetime import date, datetime
from decimal import Decimal


from ops.paths import commulingo_data_file
from tool_gateway.results import ToolFailure
from commulingo.periods import PERIOD_SCHEMA, period_columns
from commulingo.person_service import call_person_service
from commulingo.reads import reads as _reads

logger = logging.getLogger(__name__)


# Letters NFKD cannot decompose to ASCII. Without these, folding drops the
# letter whole (Piłsudski -> pisudski) instead of romanizing it.
_FOLD_MAP = str.maketrans({
    "ł": "l", "Ł": "l", "ø": "o", "Ø": "o", "đ": "d", "Đ": "d",
    "ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
    "þ": "th", "Þ": "th", "ð": "d", "Ð": "d", "ı": "i",
})


def _fold_slug(value: str) -> str:
    """ASCII kebab-case form of a model-written id: fold diacritics, drop
    apostrophes, collapse everything else to single dashes."""
    import unicodedata
    value = (value or "").translate(_FOLD_MAP)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"['’]", "", value)
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value

_SUGGESTED_BY = os.getenv("COMMULINGO_SUGGESTED_BY", "cyber-lenin").strip() or "cyber-lenin"

_TARGET_TYPES = (
    "person", "office_row", "person_section", "history_event_person", "term",
    "history_event", "history_event_section",
)

from commulingo.relation_kinds import HISTORY_RELATION_KINDS as _HISTORY_RELATION_KINDS  # noqa: E402
_ACTIONS = ("create", "update", "delete")

_FATE_KINDS = (
    "",  # Unknown/unclassified; preserve the supported uncertainty in label.
    "executed", "assassinated", "murdered", "killed", "suicide",
    "deposed", "exile", "natural",
)

_PERSON_PATCH_KEYS = frozenset({
    "id", "group", "groupId", "sortOrder", "cyrillic", "years",
    "name", "givenName", "familyName", "nameQualifier",
    "epithet", "bio", "moment", "fate", "patronymic", "cyrillicPatronymic",
    "aliases", "scenes", "career", "activities", "citizenship", "origin", "nationalOrigin",
    "expectedRevision", "evidence", "reviewFlags", "aliasEdits", "careerEdits", "sceneEdits",
    "office_rows", "sections",  # read-only echoes from get_person; tolerated and ignored
    "role",  # the retired legacy role; tolerated so old records round-trip, dropped before submit
})

# Flag codes the frontend has vendored SVGs for (data/commulingo/flag-icons.js).
# commulingo/people_lane.py derives its prompt NATIONALITY_CODES from this set.
# Codes map to country flags, not an exhaustive ethnicity taxonomy. Use sourced
# national background; never assign Russia or a neighbouring state by default.
# Preserve documented ethnic and mixed backgrounds in bilingual labels.
# Still code, and knowingly so: a nationality needs a vendored flag SVG under
# the frontend's public/flags/, which is baked into that image, so a new nation
# needs a deploy wherever this list lives. The frontend keeps the same set in
# data/commulingo/flag-icons.js — adding one means both repositories plus the
# asset. Everything else about a person's nationality is DB.
_NATIONALITY_CODES = frozenset({
    "soviet", "russia", "ukraine", "georgia", "armenia", "azerbaijan", "belarus",
    "kazakhstan", "latvia", "lithuania", "estonia", "uzbekistan", "moldova",
    "turkmenistan", "tajikistan", "kyrgyzstan", "poland", "finland", "germany",
    "east-germany", "austria", "hungary", "czechia", "romania", "bulgaria",
    "yugoslavia", "serbia", "croatia", "slovenia", "montenegro", "bosnia-herzegovina", "switzerland",
    "france", "italy", "spain", "uk", "netherlands", "belgium", "usa", "cuba", "argentina",
    "chile", "china", "japan", "india", "turkey", "greece", "vietnam", "north-korea",
    "south-korea", "albania", "angola", "burkina-faso", "congo", "ghana",
    "guinea-bissau", "indonesia", "mozambique", "peru", "trinidad", "portugal",
    "brazil", "el-salvador", "grenada", "guyana", "nicaragua", "south-africa",
    "tanzania", "ireland", "slovakia", "czechoslovakia", "korea", "martinique",
    "israel", "afghanistan",
    "algeria", "andorra", "antigua-and-barbuda", "australia",
    "bahamas", "bahrain", "bangladesh", "barbados",
    "belize", "benin", "bhutan", "bolivia",
    "botswana", "brunei", "burundi", "cambodia",
    "cameroon", "canada", "cape-verde", "central-african-republic",
    "chad", "colombia", "comoros", "costa-rica",
    "cyprus", "denmark", "djibouti", "dominica",
    "dominican-republic", "ecuador", "egypt", "equatorial-guinea",
    "eritrea", "eswatini", "ethiopia", "fiji",
    "gabon", "gambia", "guatemala", "guinea",
    "haiti", "honduras", "iceland", "iran",
    "iraq", "ivory-coast", "jamaica", "jordan",
    "kenya", "kiribati", "kosovo", "kuwait",
    "laos", "lebanon", "lesotho", "liberia",
    "libya", "liechtenstein", "luxembourg", "madagascar",
    "malawi", "malaysia", "maldives", "mali",
    "malta", "marshall-islands", "mauritania", "mauritius",
    "mexico", "micronesia", "monaco", "mongolia",
    "morocco", "myanmar", "namibia", "nauru",
    "nepal", "new-zealand", "niger", "nigeria",
    "north-macedonia", "norway", "oman", "pakistan",
    "palau", "palestine", "panama", "papua-new-guinea",
    "paraguay", "philippines", "qatar", "republic-of-congo",
    "rwanda", "saint-kitts-and-nevis", "saint-lucia", "saint-vincent-and-the-grenadines",
    "samoa", "san-marino", "sao-tome-and-principe", "saudi-arabia",
    "senegal", "seychelles", "sierra-leone", "singapore",
    "solomon-islands", "somalia", "south-sudan", "sri-lanka",
    "sudan", "suriname", "sweden", "syria",
    "taiwan", "thailand", "timor-leste", "togo",
    "tonga", "tunisia", "tuvalu", "uganda",
    "united-arab-emirates", "uruguay", "vanuatu", "vatican-city",
    "venezuela", "yemen", "zambia", "zimbabwe",
})


# Scripts that never belong in this site's prose, in ANY field. The writing
# model sometimes drops a token of its own multilingual vocabulary into the
# middle of a sentence — threefold-war's definition_ko carried 'संघर्ष' for
# 전쟁, operation-shingle 'उपलब्ध' for 쓸 수 있었다, and a Metaxas alias was
# spelled '메타كساس 체제' (2026-08-23). Unlike _SCRIPT_RANGES above, which
# decides WHICH script a native name should use, membership here is grounds
# for outright rejection: the site quotes Cyrillic, Greek, Han, Kana and the
# Caucasus scripts legitimately, but has no Hindi, Arabic or Thai content, so
# any such character is leakage. A native-name/original field that one day
# genuinely needs one of these scripts is exempted by key in
# _collect_checked_prose, not by loosening this list.
_FOREIGN_SCRIPT_RANGES = (
    ("Devanagari", re.compile(r"[ऀ-ॿ]")),
    ("Bengali", re.compile(r"[ঀ-৿]")),
    ("Gurmukhi/Gujarati/Oriya", re.compile(r"[਀-୿]")),
    ("Tamil/Telugu/Kannada/Malayalam/Sinhala", re.compile(r"[஀-෿]")),
    ("Thai/Lao", re.compile(r"[฀-໿]")),
    ("Tibetan/Myanmar", re.compile(r"[ༀ-႟]")),
    ("Khmer", re.compile(r"[ក-៿]")),
    ("Arabic", re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")),
)

# A Hangul syllable running straight into three or more lowercase Latin letters
# is the half-transliterated cousin of the same leak: term_ko '산acja 체제',
# '아프간tsy', '아isne 방어선', '이반ovo-보즈네센스크'. Three letters, not one,
# because Korean prose legitimately glues short Latin units and acronym tails
# onto Hangul (2만km); the legitimate acronym pattern (친PDPA, 좌파SR, 중앙TV)
# is uppercase and never matches.
_HANGUL_LATIN_MIX_RE = re.compile(r"[가-힯][a-z]{3}")

# Patch keys whose values legitimately carry a foreign script: the native-name
# line and its patronymic twin, a term's original native-script form, and
# source references, where a work's own title stays in the work's own script.
_SCRIPT_CHECK_EXEMPT_KEYS = frozenset({
    "original", "cyrillic", "cyrillicPatronymic", "native",
    "sources", "citations", "url",
})


_LOCALIZED_OFFICE_ROW_KEYS = ("body", "name", "note")

_OFFICE_ROW_PATCH_KEYS = frozenset({
    "sortOrder", "period", "body", "personId", "name", "note",
})

# Long-form detail sections rendered on /commulingo/people/<id>.
_SECTION_PATCH_KEYS = frozenset({"slug", "heading", "body", "sortOrder", "sources", "expectedRevision", "evidence", "reviewFlags"})

_HISTORY_EVENT_PERSON_PATCH_KEYS = frozenset({
    "personId", "sortOrder", "relationKind", "relation", "note", "side",
})
_SIDE_ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# The event card's short fields (frontend /commulingo/events/<id>). Title and
# period are deliberately absent: they are the id's public identity and the
# chronological sort key, and an unattended run has no business moving either.
_HISTORY_EVENT_PATCH_KEYS = frozenset({
    "question", "summary", "outcome", "timeline", "sources",
})
_LOCALIZED_EVENT_KEYS = ("question", "summary", "outcome")

# One `## ` section of an event's long-form body. The body is a single markdown
# column rather than a table of rows, so a section write is a splice into that
# markdown — the CommuLingo editorial service splices it. Modelled on person sections for the same
# reason they exist: a narrative this long is written one part per run, and a
# run that had to restate the whole body would spend most of its output
# reproducing text it is not changing (and would regress it whenever it drifted).
_HISTORY_EVENT_SECTION_PATCH_KEYS = frozenset({"heading", "body", "after"})

# Glossary terms (frontend /commulingo/terms, tables from frontend migration
# 061). `aliases` are the exact strings prose uses and feed the auto-linking
# pipeline; `people`/`events` are lists of related ids.
#
# `period` is the career-period shape ({start, end, startQual, endQual,
# ongoing, label}) stored as columns (frontend migration 364), or null for an
# undated term; the years come from it, so there are no startYear/endYear
# fields. `periodNote` is an optional bilingual remark that qualifies the
# period ("1917–1921 (전성기)"). A term written without a category shows up on
# the glossary as 'Uncategorized', so it is required on create rather than
# backfilled later by hand.
_TERM_PATCH_KEYS = frozenset({
    "id", "sortOrder", "term", "original", "period", "periodNote",
    "category", "region", "definition", "body", "aliases", "people", "events", "sources",
    "parentId", "expectedRevision", "evidence",
})

# The one allow-list every write path reads. It used to be spelled out twice —
# once in normalize_commulingo_write and once in _validate — which is one copy
# too many for a table that grows a row per target type.
_PATCH_KEYS_BY_TARGET = {
    "person": _PERSON_PATCH_KEYS,
    "office_row": _OFFICE_ROW_PATCH_KEYS,
    "person_section": _SECTION_PATCH_KEYS,
    "history_event_person": _HISTORY_EVENT_PERSON_PATCH_KEYS,
    "history_event": _HISTORY_EVENT_PATCH_KEYS,
    "history_event_section": _HISTORY_EVENT_SECTION_PATCH_KEYS,
    "term": _TERM_PATCH_KEYS,
}

# The glossary category registry lives in commulingo_term_categories, the same
# table the site reads (frontend migration 115). It used to be this tuple here
# and a second copy in the frontend's term-categories.js, so adding a category
# meant editing two repositories and deploying one of them.
#
# Read once per process: the curation lanes are oneshot units, so they pick up a
# new category on their next run. A long-running service that imported this
# module keeps the list it started with until it restarts — acceptable for a
# registry that changes a few times a year, and the fallback below keeps the
# tool usable if the DB is unreachable at import.
# FALLBACK ONLY — commulingo_term_categories is the source of truth. Adding or
# renaming a category is one statement against that table and needs no code
# change in either repository; editing this tuple changes nothing while the DB
# is reachable, and the loader logs a warning on every fall back so a stale copy
# announces itself.
_TERM_CATEGORY_FALLBACK = (
    ("theory", "Ideology and theory"), ("philosophy", "Philosophy"),
    ("historiography", "Historiography and historical debates"), ("parties", "Parties and organizations"),
    ("factions", "Party organs and line struggles"), ("state", "State and government"),
    ("events", "Revolutions and movements"), ("labor", "Labour and the labour movement"),
    ("military", "War and the military"), ("diplomacy", "Diplomacy and world order"),
    ("repression", "Repression and law"), ("economy", "Economy"), ("nationalities", "Nations and religion"),
    ("society", "Society and everyday life"), ("gender", "Sex and gender"), ("arts", "Literature and the arts"),
    ("culture", "Media, education and culture"), ("science", "Science and technology"),
)


def _load_term_categories() -> tuple[tuple[str, ...], str]:
    """(slugs, hint line) for the schema enum and the rejection messages.

    The list mirrors CommuLingo's commulingo_term_categories (18 kinds since
    2026-10-29). It is not fetched at import: CommuLingo is reached
    only through its admin MCP, and its editorial store rejects an unknown
    category on save, so a stale copy here can only make a message less exact.
    """
    pairs = list(_TERM_CATEGORY_FALLBACK)
    hint = ", ".join(
        slug if slug.replace("-", " ") == label.lower() else f"{slug} ({label.lower()})"
        for slug, label in pairs
    )
    return tuple(slug for slug, _ in pairs), hint


_TERM_CATEGORIES, _TERM_CATEGORY_HINT = _load_term_categories()

# ── Mode switch (config/commulingo_people.json, mtime-cached) ─────────

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "commulingo_people.json",
)
_config_cache: dict | None = None
_config_mtime: float = -1.0

# ── Name-spelling normalization (config/commulingo_name_normalization.json) ──
# variant -> canonical per language. Prose using a variant outside direct
# quotation marks is rejected by _validate so every card spells other people
# the way their own card does.

_NAME_NORM_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "commulingo_name_normalization.json",
)
_name_norm_cache: dict | None = None
_name_norm_mtime: float = -1.0

# Spans inside these quote pairs keep their original spelling (direct
# quotations of period documents and speech).
QUOTED_SPAN_RE = re.compile(r'"[^"]*"|“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|《[^》]*》')


def name_normalization() -> dict:
    """{'ko': {variant: canonical}, 'en': {...}, 'blocked': {'ko': [...], 'en': [...]}}.

    'blocked' strings merely contain a variant (시베리아 ⊃ 베리아) and are
    masked before matching so they never trigger.
    """
    global _name_norm_cache, _name_norm_mtime
    empty = {"ko": {}, "en": {}, "blocked": {"ko": [], "en": []}}
    try:
        mtime = os.path.getmtime(_NAME_NORM_PATH)
    except OSError:
        return empty
    if _name_norm_cache is None or mtime != _name_norm_mtime:
        try:
            with open(_NAME_NORM_PATH, encoding="utf-8") as f:
                data = json.load(f)
            blocked = data.get("blocked") or {}
            _name_norm_cache = {
                "ko": {str(k): str(v) for k, v in (data.get("ko") or {}).items()},
                "en": {str(k): str(v) for k, v in (data.get("en") or {}).items()},
                "blocked": {
                    "ko": [str(s) for s in (blocked.get("ko") or [])],
                    "en": [str(s) for s in (blocked.get("en") or [])],
                },
            }
            _name_norm_mtime = mtime
        except Exception as e:
            logger.warning("commulingo name normalization config unreadable: %s", e)
            return empty
    return _name_norm_cache


def _collect_localized_strings(node, out: list) -> None:
    """Recursively gather ('ko'|'en', text) pairs from {ko, en} dicts in a patch."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("ko", "en") and isinstance(value, str):
                out.append((key, value))
            else:
                _collect_localized_strings(value, out)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_localized_strings(item, out)


def find_spelling_variants_in_text(text: str, lang: str) -> dict[str, str]:
    """variant -> canonical spellings used in `text` outside quotation marks.

    Shared with the research-document writer (publishing/research.py) so
    reports spell dictionary people and glossary terms the way their cards do.
    """
    norm = name_normalization()
    table = norm.get(lang) or {}
    hits: dict[str, str] = {}
    if not table or not text:
        return hits
    scannable = QUOTED_SPAN_RE.sub(" ", str(text))
    for blocked in norm["blocked"].get(lang) or []:
        scannable = scannable.replace(blocked, " ")
    for variant, canonical in table.items():
        if variant in scannable:
            hits[variant] = canonical
    return hits


def normalize_spellings_in_text(text: str, lang: str) -> tuple[str, dict]:
    """Auto-correct variant spellings to their dictionary-card forms.

    Quoted spans (direct quotations) and blocked compounds (시베리아 ⊃ 베리아)
    are masked and restored untouched. Returns (fixed_text, {variant: canonical}
    applied).
    """
    norm = name_normalization()
    table = norm.get(lang) or {}
    if not table or not text:
        return str(text or ""), {}
    placeholders: dict[str, str] = {}

    def _stash(match):
        key = f"\x00Q{len(placeholders)}\x00"
        placeholders[key] = match.group(0)
        return key

    masked = QUOTED_SPAN_RE.sub(_stash, str(text))
    blocked = norm["blocked"].get(lang) or []
    for index, compound in enumerate(blocked):
        masked = masked.replace(compound, f"\x00B{index}\x00")
    applied: dict[str, str] = {}
    for variant, canonical in table.items():
        if variant in masked:
            applied[variant] = canonical
            masked = masked.replace(variant, canonical)
    for index, compound in enumerate(blocked):
        masked = masked.replace(f"\x00B{index}\x00", compound)
    for key, original in placeholders.items():
        masked = masked.replace(key, original)
    return masked, applied


def _find_name_variants(patch: dict) -> list[tuple[str, str]]:
    """(variant, canonical) pairs used outside quotation marks, deduped."""
    texts: list[tuple[str, str]] = []
    _collect_localized_strings(patch, texts)
    hits: dict[str, str] = {}
    for lang, text in texts:
        hits.update(find_spelling_variants_in_text(text, lang))
    return sorted(hits.items())


def direct_apply_enabled() -> bool:
    """True when edits apply immediately; False stages them for review.

    Defaults to False (staging) when the config file is missing or broken —
    the tool's response always says which mode ran, so a silent fallback is
    visible to the operator.
    """
    global _config_cache, _config_mtime
    try:
        mtime = os.path.getmtime(_CONFIG_PATH)
    except OSError:
        return False
    if _config_cache is None or mtime != _config_mtime:
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                _config_cache = json.load(f)
            _config_mtime = mtime
        except Exception as e:
            logger.warning("commulingo_people config unreadable: %s", e)
            return False
    return bool(_config_cache.get("direct_apply"))


def _json_default(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    return str(o)


def _dumps(obj) -> str:
    return json.dumps(obj, default=_json_default, ensure_ascii=False, indent=2)


# ── Read queries ──────────────────────────────────────────────────────

def _list_groups() -> list[dict]:
    # The blurb is what actually defines a group; the title is a label for it.
    # Without it the curator classified on the title alone, and 「비소련 반혁명
    # 진영」 collected every non-Soviet politician who ever opposed Moscow —
    # Roosevelt, Marshall, Paasikivi, and a Finnish foreign minister — when the
    # blurb says the group is for those who took up arms against a revolution.
    return _reads.groups()


def _search_people(q: str, group_id: str, limit: int, function_id: str = "", affiliation_id: str = "") -> list[dict]:
    affiliations = ()
    if function_id or affiliation_id:
        from commulingo.activities import activity_search_params
        params = activity_search_params(function_id, affiliation_id)
        affiliations = params["descendants"] if affiliation_id else ()
    rows = _reads.people_search(q, group_id, limit, function_id, affiliations)
    keys = ("id", "group_id", "name_ko", "name_en", "name_qualifier_ko", "name_qualifier_en", "cyrillic", "years_label",
            "epithet_ko", "fate_kind")
    return [{k: row.get(k) for k in keys} for row in rows]


def _office_snapshot(office_id: str) -> dict | None:
    office = _reads.office(office_id)
    if not office:
        return None
    return {
        "id": office["id"], "range_label": office.get("range", ""),
        "title_ko": office["title"]["ko"], "title_en": office["title"]["en"],
        "blurb_ko": office["blurb"]["ko"], "blurb_en": office["blurb"]["en"],
        "rows": [{
            "row_id": int(row["id"]), "period": row.get("period"), "years": (row.get("years") or {}).get("ko", ""),
            "body_ko": row["body"]["ko"], "body_en": row["body"]["en"], "person_id": row.get("personId") or None,
            "name_ko": row["name"]["ko"], "name_en": row["name"]["en"],
            "note_ko": row["note"]["ko"], "note_en": row["note"]["en"],
        } for row in office.get("rows", [])],
    }


def _get_person(person_id: str) -> dict | None:
    # The revision and fields are read in the shared store's one MVCC snapshot.
    person = _reads.person(person_id)
    person = dict(person) if person else None
    if person:
        person["office_rows"] = [
            {"row_id": r["id"], "office_id": r["officeId"], "office_title_ko": r["officeTitle"]["ko"],
             "period": r.get("period"), "years": (r.get("years") or {}).get("ko", ""), "body_ko": r["body"]["ko"], "note_ko": r["note"]["ko"]}
            for r in person.get("institutionRows", [])
        ]
        person["sections"] = [
            {"slug": s["slug"], "heading": s["heading"], "sortOrder": s["sortOrder"],
             "bodyChars": {lang: len(s["body"].get(lang) or "") for lang in ("ko", "en")}}
            for s in person["sections"]
        ]
    if person:
        # Cross-store hub: facts other pipelines (news, research, analysts) attached
        # to this person's graph node. Flag-gated with the KG recall feature; one
        # cheap Cypher by external id, never fatal.
        try:
            from kg_runtime.recall import enabled as _kg_recall_enabled
            if _kg_recall_enabled():
                from kg_runtime.search import kg_facts_for_external_id
                facts = kg_facts_for_external_id(f"commulingo:person:{person_id}", limit=5)
                if facts:
                    person["kg_facts"] = facts
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("commulingo get_person kg_facts skipped: %s", exc)
    return person


def _search_all(q: str, limit: int) -> dict:
    """Unified substring search across people, glossary terms, events, and offices.

    Callers (models) routinely guess the wrong category — a faction like
    "노동자 반대파" lives in the glossary, not in history events — so one search
    over every category removes the guess entirely. A single match includes its
    full record inline (`detail`): summary rows alone carry no substance, so
    forcing a follow-up get_* call for an unambiguous hit was pure overhead.
    Only a multi-match result needs the follow-up, and only then is `hint` sent.
    """
    needle = q.casefold()
    events = [{"id": e["id"], "period_label": e.get("period") or "", "title_ko": e["title"]["ko"], "title_en": e["title"]["en"]}
              for e in _reads.events(q, limit)][:limit]
    offices = [{"id": o["id"], "range_label": o.get("range", ""), "title_ko": o["title"]["ko"], "title_en": o["title"]["en"]}
               for o in _reads.offices()
               if needle in o["id"].casefold() or needle in o["title"]["ko"].casefold() or needle in o["title"]["en"].casefold()][:limit]
    result = {
        "people": _search_people(q, "", limit),
        "terms": _list_terms(q, limit)[:limit],
        "events": events,
        "offices": offices,
    }
    total = sum(len(rows) for rows in result.values())
    if total == 1:
        if result["people"]:
            record = _get_person(result["people"][0]["id"])
            kind = "person"
        elif result["terms"]:
            record = _get_term(result["terms"][0]["id"])
            kind = "term"
        elif result["events"]:
            record = _get_event(result["events"][0]["id"])
            kind = "event"
        else:
            record = _get_office(result["offices"][0]["id"])
            kind = "office"
        result["detail"] = {"kind": kind, "record": record}
    elif total > 1:
        result["hint"] = (
            "multiple matches — fetch one in full with get_person(person_id) / "
            "get_term(term_id) / get_event(event_id) / get_office(office_id)"
        )
    return result


def _list_offices() -> list[dict]:
    return [{"id": o["id"], "range_label": o.get("range", ""), "title_ko": o["title"]["ko"], "title_en": o["title"]["en"],
             "row_count": o.get("rowCount", 0)} for o in _reads.offices()]


def _get_office(office_id: str) -> dict | None:
    return _office_snapshot(office_id)


def _get_sections(person_id: str) -> list[dict] | None:
    """Sections in the exact person_section patch shape (read → edit → write back)."""
    person = _reads.person(person_id)
    if not person:
        return None
    return [
        {"slug": s["slug"], "sortOrder": s["sortOrder"], "heading": s["heading"], "body": s["body"],
         "sources": s.get("sources") or []}
        for s in person.get("sections") or []
    ]


def _list_events() -> list[dict]:
    return [{"id": e["id"], "period_label": e.get("period") or "", "title_ko": e["title"]["ko"], "title_en": e["title"]["en"],
             "people_count": e.get("people", 0), "body_ko_chars": e.get("bodyKoChars", 0),
             "timeline_entries": e.get("timelineEntries", 0)} for e in _reads.events()]


def _get_event(event_id: str) -> dict | None:
    event = _reads.event(event_id)
    if not event:
        return None
    event = dict(event)
    # The bodies are the longest text on the site and the reader only needs to
    # know which parts exist and how long each one runs — returning them whole
    # would spend most of a curator's context re-reading what it already wrote.
    for lang in ("ko", "en"):
        body = event.pop(f"body_{lang}") or ""
        event[f"body_{lang}_chars"] = len(body)
        event[f"body_{lang}_sections"] = [
            {"heading": heading, "chars": len(block)}
            for heading, block in _split_event_body(body)
        ]
    event["people"] = [
        {
            "personId": row["person_id"],
            "sortOrder": row["sort_order"],
            "relationKind": row["relation_kind"],
            "side": row["side"],
            "relation": {"ko": row["relation_ko"], "en": row["relation_en"]},
            "note": {"ko": row["note_ko"], "en": row["note_en"]},
        }
        for row in event.get("people") or []
    ]
    return event


def _list_terms(q: str = "", limit: int | None = None) -> list[dict]:
    # Unfiltered this is ~1300 terms with their aliases. A curator checking
    # whether one candidate is already registered used to pull that whole blob,
    # hit the tool-result cap, and pull it again — three rounds spent to not get
    # an answer. 'q' answers the same question in one small result.
    return [{"id": t["id"], "term": t["term"], "original": t.get("original") or "", "aliases": t.get("aliases") or []}
            for t in _reads.terms(q, limit)]


def _get_term(term_id: str) -> dict | None:
    return _reads.term_editorial(term_id)


def _list_suggestions(status: str, limit: int) -> list[dict]:
    keys = ("id", "target_type", "target_id", "action", "status", "confidence", "reviewer", "review_note", "created_at", "reviewed_at")
    return [{k: row.get(k) for k in keys} for row in _reads.suggestions(status, limit)]


COMMULINGO_PEOPLE_TOOL = {
    "name": "commulingo_people",
    "description": (
        "Read the CommuLingo people dictionary: Soviet and Chinese revolutionary "
        "figures with bios and career timelines, glossary terms, historical events and "
        "Soviet institution (office) leadership timelines, bilingual ko/en. Actions: "
        "`search` (q across people, terms, events and offices; a single match returns "
        "its full record; use when the category is uncertain), "
        "`list_activity_catalog` (shared functions and activity affiliations; activities bind function + organization + period + evidence, separate from citizenship), "
        "`list_groups` (era groups on the Soviet, China and world shelves + people counts), "
        "`search_people` (q matches id/name/cyrillic; optional group_id/function_id/affiliation_id; function and affiliation must match the same activity), "
        "`get_person` (full record in the person-writer field shape; office_rows "
        "and sections are read-only), "
        "`list_offices` (institution timelines + row counts), "
        "`get_office` (one institution's full leadership timeline), "
        "`get_sections` (a person's detail sections in the section patch shape), "
        "`list_events` (historical event ids, titles, and linked-person counts), "
        "`get_event` (one event and all current person relationships), "
        "`list_terms` (term ids, names and aliases; check before registering a term; "
        "pass `q` to match one candidate), "
        "`get_term` (one glossary term in the exact term patch shape), "
        "`list_suggestions` (narrow-write edit history/queue; "
        "optional status filter: pending/approved/rejected/superseded). "
        "Always read the current record before calling a narrow write tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_activity_catalog", "search", "list_groups", "search_people", "get_person",
                    "list_offices", "get_office",
                    "get_sections", "list_events", "get_event",
                    "list_terms", "get_term", "list_suggestions",
                ],
            },
            "q": {
                "type": "string",
                "description": (
                    "Substring. search: people, terms, events and offices. "
                    "search_people: id/name/cyrillic. list_terms: term id/ko/en/original/alias."
                ),
            },
            "group_id": {
                "type": "string",
                "description": "search_people: restrict to one group id.",
            },
            "function_id": {"type": "string", "description": "search_people: activity function id from list_activity_catalog."},
            "affiliation_id": {"type": "string", "description": "search_people: actual service affiliation, including descendants; matched within the same activity as function_id."},
            "person_id": {"type": "string", "description": "get_person/get_sections: person id."},
            "office_id": {"type": "string", "description": "get_office: office id."},
            "event_id": {"type": "string", "description": "get_event: historical event id."},
            "term_id": {"type": "string", "description": "get_term: glossary term id."},
            "status": {
                "type": "string",
                "description": "list_suggestions: filter (pending/approved/rejected/superseded). Default: all.",
            },
            "limit": {
                "type": "integer",
                "description": "search_people/list_suggestions: max rows. Default 30, max 100.",
            },
        },
        "required": ["action"],
    },
}


async def _exec_commulingo_people(
    action: str,
    q: str = "",
    group_id: str = "",
    person_id: str = "",
    office_id: str = "",
    event_id: str = "",
    term_id: str = "",
    status: str = "",
    limit: int = 30,
    function_id: str = "",
    affiliation_id: str = "",
) -> str:
    try:
        limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit = 30
    try:
        if action == "search":
            if not (q or "").strip():
                return "Error: q is required for search."
            result = await asyncio.to_thread(_search_all, q.strip(), limit)
        elif action == "list_activity_catalog":
            from commulingo.activities import load_catalog
            result = load_catalog()
        elif action == "list_groups":
            result = await asyncio.to_thread(_list_groups)
        elif action == "search_people":
            search_args = [(q or "").strip(), (group_id or "").strip(), limit]
            if function_id or affiliation_id:
                search_args.extend([(function_id or "").strip(), (affiliation_id or "").strip()])
            result = await asyncio.to_thread(_search_people, *search_args)
        elif action == "get_person":
            if not person_id:
                return "Error: person_id is required for get_person."
            result = await asyncio.to_thread(_get_person, person_id.strip())
            if result is None:
                return f"Error: person '{person_id}' not found. Use search_people to find the id."
        elif action == "list_offices":
            result = await asyncio.to_thread(_list_offices)
        elif action == "get_sections":
            if not person_id:
                return "Error: person_id is required for get_sections."
            result = await asyncio.to_thread(_get_sections, person_id.strip())
            if result is None:
                return f"Error: person '{person_id}' not found."
        elif action == "get_office":
            if not office_id:
                return "Error: office_id is required for get_office."
            result = await asyncio.to_thread(_get_office, office_id.strip())
            if result is None:
                return f"Error: office '{office_id}' not found. Use list_offices to find the id."
        elif action == "list_events":
            result = await asyncio.to_thread(_list_events)
        elif action == "get_event":
            if not event_id:
                return "Error: event_id is required for get_event."
            result = await asyncio.to_thread(_get_event, event_id.strip())
            if result is None:
                return f"Error: event '{event_id}' not found. Use list_events to find the id."
        elif action == "list_terms":
            result = await asyncio.to_thread(_list_terms, (q or "").strip())
        elif action == "get_term":
            if not term_id:
                return "Error: term_id is required for get_term."
            result = await asyncio.to_thread(_get_term, term_id.strip())
            if result is None:
                return f"Error: term '{term_id}' not found. Use list_terms to find the id."
        elif action == "list_suggestions":
            result = await asyncio.to_thread(_list_suggestions, (status or "").strip(), limit)
        else:
            return f"Error: unknown action '{action}'."
        body = _dumps(result)
        if len(body) > 24000:
            body = body[:24000] + "\n…(truncated at 24000 chars)"
        return body
    except Exception as e:
        logger.warning("commulingo_people error: %s", e)
        return ToolFailure(f"Error: {type(e).__name__}: {e}")


# ── Patch application (Python port of the frontend admin store) ──────
# Mirrors frontend/data/commulingo/people-admin-store.js: one transaction per
# edit, wholesale replacement of aliases/scenes/career when provided, and a
# revision snapshot in commulingo_people_revisions.


def _contains_north_korea(value) -> bool:
    if isinstance(value, str):
        return "북한" in value
    if isinstance(value, dict):
        return any(_contains_north_korea(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_north_korea(item) for item in value)
    return False


def _em_dash_problem(patch: dict) -> str | None:
    """Reject the em dash in prose, allowing it inside a quoted title.

    It is not house style and never was: on 2026-08-09 the only em dash in any
    hand-written text on the site was inside 「스페인의 교훈 — 마지막 경고」, the
    title of a Trotsky pamphlet. The 522 person sections and 11 event bodies that
    carried one had all been written by these lanes, which reach for it in both
    languages no matter what the prompt says. The prompt rule stayed and this is
    what makes it stick, the way the 북한 and 조지아 rules do.

    Quoted spans are exempt because that pamphlet title is the legitimate case:
    a work's own name is reproduced, not restyled.
    """
    strings: list[tuple[str, str]] = []
    _collect_localized_strings(patch, strings)
    for _lang, text in strings:
        if "—" not in text:
            continue
        if "—" in QUOTED_SPAN_RE.sub("", text):
            excerpt = text[max(0, text.find("—") - 40):text.find("—") + 40]
            return (
                "Error: Korean and English copy do not use the em dash (—). Rewrite the "
                "clause with a comma, a colon, parentheses, or two sentences: "
                f"…{excerpt}… It is allowed only inside a quoted title, where it is part "
                "of the work's own name."
            )
    return None


def _collect_checked_prose(node, out: list, key: str = "") -> None:
    """Every string in a patch except native-script and source fields.

    Wider than _collect_localized_strings on purpose: the Metaxas leak sat in
    an alias list ({"ko": ["메타كساس 체제"]}), whose members are plain strings
    under a list, not {ko, en} dicts — a collector keyed on localized pairs
    walks straight past them.
    """
    if key in _SCRIPT_CHECK_EXEMPT_KEYS:
        return
    if isinstance(node, str):
        out.append((key, node))
    elif isinstance(node, dict):
        for child_key, value in node.items():
            _collect_checked_prose(value, out, str(child_key))
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_checked_prose(item, out, key)


def _script_leak_problem(patch: dict) -> str | None:
    """Reject model-token leakage: foreign-script words and half-transliterations.

    See _FOREIGN_SCRIPT_RANGES for the incident this guards against. The check
    runs on every write, so a leak now costs the model a retry with the exact
    bad word quoted back, instead of costing a reader a 'संघर्ष' in a Korean
    definition until someone happens to open the page.
    """
    strings: list[tuple[str, str]] = []
    _collect_checked_prose(patch, strings)
    for field, text in strings:
        for script, pattern in _FOREIGN_SCRIPT_RANGES:
            hit = pattern.search(text)
            if hit:
                start = max(0, hit.start() - 40)
                return (
                    f"Error: '{field}' contains {script} characters: "
                    f"…{text[start:hit.end() + 40]}… This site has no {script} "
                    "content — the word is a token leak from the writing model, "
                    "not a quotation. Rewrite the affected word in the language "
                    "of the field (한국어 필드는 한국어로, English fields in "
                    "English) and resend the same edit."
                )
        mix = _HANGUL_LATIN_MIX_RE.search(text)
        if mix:
            start = max(0, mix.start() - 40)
            return (
                f"Error: '{field}' glues Hangul straight into lowercase Latin: "
                f"…{text[start:mix.end() + 40]}… That is a half-transliterated "
                "word ('산acja 체제', '아프간tsy'), not valid Korean. Write the "
                "whole word in Hangul per 외래어 표기법 ('사나차 체제'), keeping "
                "any Latin original in parentheses after it if needed."
            )
    return None


_NON_NAME_CHARS = re.compile(r"[^0-9a-z가-힣]+")


def _dedup_key(name: str) -> str:
    """Comparison key for name identity: casefold, drop spacing and punctuation.

    'C.L.R. James' and 'C. L. R. James', '본치-브루예비치' and '본치브루예비치'
    are one person written two ways, so raw-string equality misses them.
    """
    return _NON_NAME_CHARS.sub("", (name or "").casefold())


# These messages are read by a model that holds one narrow write tool, not the
# generic commulingo_edit(action=...) tool the narrow ones replaced. Telling it
# to "use action 'update'" sent it looking for a parameter that no longer exists
# on any schema, which is exactly how a term rewrite dead-ended. Name the call
# that does the job instead.
_WRITE_TOOL_CALLS = {
    ("person", "create"): "commulingo_person_create",
    ("person", "update"): "commulingo_person_update",
    ("term", "create"): "commulingo_term_create",
    ("term", "update"): "commulingo_term_update",
    ("person_section", "create"): "commulingo_section_save(action='create')",
    ("person_section", "update"): "commulingo_section_save(action='update')",
    ("office_row", "create"): "commulingo_office_row_save(action='create')",
    ("office_row", "update"): "commulingo_office_row_save(action='update')",
    ("history_event_person", "create"): "commulingo_event_link",
    ("history_event", "update"): "commulingo_event_update",
    ("history_event_section", "create"): "commulingo_event_section_save(action='create')",
    ("history_event_section", "update"): "commulingo_event_section_save(action='update')",
}


def _write_tool_call(target_type: str, action: str) -> str:
    return _WRITE_TOOL_CALLS.get((target_type, action)) or f"the {target_type} {action} tool"


def _reader_call(action: str) -> str:
    return f"commulingo_people(action='{action}')"


def _validate_history_event_person(cur, action: str, target_id: str, patch: dict) -> str | None:
    if action == "delete":
        return "Error: history_event_person deletion is not available to the unattended curator."
    event_row = _reads.event(target_id)
    if not event_row:
        return (
            f"Error: history event {target_id} not found. Find the id with "
            f"{_reader_call('list_events')}."
        )
    # Named sides (frontend migration 193): the camp this person acted for.
    side_ids = [s.get("id") for s in (event_row["sides"] or []) if isinstance(s, dict)]
    side = patch.get("side")
    if side is not None:
        if not isinstance(side, str) or not _SIDE_ID_RE.match(side):
            return "Error: side must be one of the event's side ids, or null."
        if side not in side_ids:
            return (
                f"Error: side '{side}' is not one of this event's sides"
                + (f" ({', '.join(side_ids)})." if side_ids else "; this event names no sides, so omit side.")
            )
    person_id = str(patch.get("personId") or "").strip()
    if not person_id:
        return "Error: history_event_person patch.personId is required."
    if not _reads.exists("person", person_id):
        return (
            f"Error: person {person_id} not found. Find the id with "
            f"{_reader_call('search_people')}."
        )
    kind = str(patch.get("relationKind") or "").strip()
    if kind not in _HISTORY_RELATION_KINDS:
        return f"Error: relationKind must be one of {', '.join(_HISTORY_RELATION_KINDS)}."
    if side_ids and kind == "opponent":
        return (
            f"Error: this event names its sides ({', '.join(side_ids)}); use the person's own "
            "side with their role in it (leader, executor, participant, target) instead of opponent."
        )
    for key in ("relation", "note"):
        value = patch.get(key)
        if not isinstance(value, dict) or not value.get("ko") or not value.get("en"):
            return f"Error: {key}.ko and {key}.en are required."
    note = patch["note"]
    nt_ko, nt_en = FIELD_LIMITS["event_note"]
    if len(note.get("ko") or "") > nt_ko or len(note.get("en") or "") > nt_en:
        return (
            f"Error: note is too long (ko {len(note.get('ko') or '')}/{nt_ko}, "
            f"en {len(note.get('en') or '')}/{nt_en} characters). The note is a "
            "one-or-two-sentence caption under the person on the event page — "
            "move depth to a person_section."
        )
    # Every other target treats a non-int sortOrder as "append"; this one used
    # to reject null outright, so the same patch shape passed for a person and
    # failed for an event link.
    if patch.get("sortOrder") is not None and not isinstance(patch["sortOrder"], int):
        return "Error: sortOrder must be an integer, or null to append."
    return None


def _validate_history_event(cur, action: str, target_id: str, patch: dict) -> str | None:
    if action != "update":
        return (
            "Error: history events are created and retired by hand. The curator may "
            "only update one that already exists."
        )
    stored_event = _reads.event(target_id)
    event_row = {"skeleton": not (stored_event.get("summary_ko") or "")} if stored_event else None
    if not event_row:
        return (
            f"Error: history event '{target_id}' not found. Find the id with "
            f"{_reader_call('list_events')}."
        )
    if event_row["skeleton"]:
        # A row with no summary is a hand-seeded skeleton: its first write is
        # the whole card, in one call. A partial fill would publish the page
        # (the store keys visibility on summary) with the rest still blank —
        # the empty sources box ships to readers.
        missing = [
            key for key in ("question", "summary", "outcome", "timeline", "sources")
            if not patch.get(key)
        ]
        if missing:
            return (
                "Error: this event is a skeleton, so its first write must carry the "
                f"whole card. Missing: {', '.join(missing)}. Resend ONE call with "
                "question, summary, outcome, timeline and sources together — sources "
                "is the works you actually used, the same ones as your citations."
            )
    for key in _LOCALIZED_EVENT_KEYS:
        value = patch.get(key)
        if key not in patch or value is None:
            continue
        if not isinstance(value, dict):
            return (
                f"Error: {key} must be an object {{\"ko\": \"...\", \"en\": \"...\"}} — "
                "a plain string would silently blank the other language."
            )
        ko_max, en_max = FIELD_LIMITS[f"event_{key}"]
        ko_len, en_len = len(value.get("ko") or ""), len(value.get("en") or "")
        if ko_len > ko_max or en_len > en_max:
            return (
                f"Error: {key} is too long (ko {ko_len}/{ko_max}, en {en_len}/{en_max} "
                f"characters). It is the card text that sits above the body — put the "
                f"depth in a body section with "
                f"{_write_tool_call('history_event_section', 'create')} instead."
            )
    timeline = patch.get("timeline")
    if timeline is not None:
        if not isinstance(timeline, list) or not timeline:
            return "Error: timeline must be a non-empty list; omit it to leave it unchanged."
        for index, item in enumerate(timeline):
            if not isinstance(item, dict) or set(item) - {"date", "title", "body"}:
                return (
                    f"Error: timeline[{index}] must be exactly "
                    '{"date": "1936.02", "title": {"ko","en"}, "body": {"ko","en"}}.'
                )
            if not str(item.get("date") or "").strip():
                return (
                    f"Error: timeline[{index}].date is required — '1936', '1936.02' or "
                    "'1936.07.18' as the source supports."
                )
            for key in ("title", "body"):
                part = item.get(key)
                if not isinstance(part, dict) or not (part.get("ko") and part.get("en")):
                    return f"Error: timeline[{index}].{key} needs both a ko and an en string."
    sources = patch.get("sources")
    if sources is not None and (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(s, str) and s.strip() for s in sources)
    ):
        return (
            "Error: sources must be a non-empty list of reference strings. Sending it "
            "replaces the stored list whole, so include the entries already there."
        )
    return None


def _validate_history_event_section(cur, action: str, target_id: str, patch: dict) -> str | None:
    if action == "delete":
        return "Error: event body sections are not deleted by the unattended curator."
    row = _reads.event(target_id)
    if not row:
        return (
            f"Error: history event '{target_id}' not found. Find the id with "
            f"{_reader_call('list_events')}."
        )
    for key in ("heading", "body"):
        value = patch.get(key)
        if not isinstance(value, dict) or not (value.get("ko") and value.get("en")):
            return (
                f"Error: {key} must be an object with a non-empty ko and en — a section "
                "written in one language only leaves the other page with a gap."
            )
    for lang in ("ko", "en"):
        text = patch["body"].get(lang) or ""
        # The save path renders '## {heading}\n\n{body}', so a heading line
        # inside the body ships as a duplicated or smuggled extra section.
        if re.search(r"(^|\n)\s*#{1,6} ", text):
            return (
                f"Error: body.{lang} contains a markdown heading line. The heading goes "
                "in the heading field and one call is one section — remove the '## ' "
                "line (do not repeat the heading inside the body, and do not pack a "
                "second section into this call)."
            )
        if re.search(r"중략|이하 생략|원문이 길어|\|\|\|", text):
            return (
                f"Error: body.{lang} contains a truncation placeholder. The page ships "
                "exactly what you send — write the section in full, ending it cleanly "
                "within the length target instead of cutting it with a marker."
            )
    after = patch.get("after")
    if after is not None and not isinstance(after, dict):
        return (
            'Error: after must be {"ko": "<an existing ## heading>", "en": "..."}, '
            "or omitted to append the section at the end."
        )
    ko_max, en_max = FIELD_LIMITS["event_section_body"]
    ko_len = len(patch["body"].get("ko") or "")
    en_len = len(patch["body"].get("en") or "")
    if ko_len > ko_max or en_len > en_max:
        return (
            f"Error: section body is too long (ko {ko_len}/{ko_max}, en {en_len}/{en_max} "
            f"characters). The target is {EVENT_SECTION_TARGET[0]}-{EVENT_SECTION_TARGET[1]} "
            f"Korean characters; a body this size is two sections — file the second one "
            f"as its own section instead of trimming this one to fit."
        )
    ko_parts = _split_event_body(row["body_ko"])
    en_parts = _split_event_body(row["body_en"])
    ko_at = _find_event_section(ko_parts, patch["heading"]["ko"])
    en_at = _find_event_section(en_parts, patch["heading"]["en"])
    if action == "create":
        if ko_at >= 0 or en_at >= 0:
            existing = ko_parts[ko_at][0] if ko_at >= 0 else en_parts[en_at][0]
            return (
                f"Error: '{target_id}' already has a section '{existing}' on this topic. "
                f"Rewrite it with {_write_tool_call('history_event_section', 'update')}, "
                f"or write a genuinely different part of the story."
            )
        for lang, parts in (("ko", ko_parts), ("en", en_parts)):
            anchor = ((after or {}).get(lang) or "").strip()
            if anchor and _find_event_section(parts, anchor) < 0:
                return (
                    f"Error: after.{lang} '{anchor}' is not a heading of this event's "
                    f"{lang} body. Omit 'after' to append at the end."
                )
        ko_cap, en_cap = EVENT_BODY_CEILING
        ko_total = len(row["body_ko"]) + ko_len
        en_total = len(row["body_en"]) + en_len
        if ko_total > ko_cap or en_total > en_cap:
            return (
                f"Error: this section would take the body past the per-event ceiling "
                f"(ko {ko_total}/{ko_cap}, en {en_total}/{en_cap}). The event is already a "
                f"long article — deepen an existing section instead of adding another."
            )
    elif ko_at < 0 or en_at < 0:
        missing = "ko" if ko_at < 0 else "en"
        return (
            f"Error: '{target_id}' has no '{patch['heading'][missing]}' section in its "
            f"{missing} body. Use action 'create' to add it."
        )
    return None


def _validate_office_row(cur, action: str, target_id: str, patch: dict) -> str | None:
    if "years" in patch:
        return "Error: office row years is read-only text; send period {start: [year, month?, day?], end, ...}."
    if action == "create" or "period" in patch:
        try:
            period_columns(patch.get("period"))
        except ValueError as exc:
            return f"Error: {exc}"
    for key in _LOCALIZED_OFFICE_ROW_KEYS:
        if key in patch and patch[key] is not None and not isinstance(patch[key], dict):
            return (
                f"Error: {key} must be an object {{\"ko\": \"...\", \"en\": \"...\"}} — "
                "a plain string would silently blank the other language."
            )
    if action == "create":
        if not _reads.exists("office", target_id):
            return f"Error: office '{target_id}' not found (office_row create targets an office id)."
    else:
        if not target_id.isdigit():
            return (
                f"Error: office row '{target_id}' not found (office_row update/delete "
                "targets a numeric row id from get_office/get_person)."
            )
        if not _reads.exists("office_row", target_id):
            return f"Error: office row '{target_id}' not found."
    if patch.get("personId"):
        if not _reads.exists("person", patch["personId"]):
            return f"Error: personId '{patch['personId']}' does not exist."
    return None


# Person, person-section and term writes are checked by the CommuLingo JS
# store (_run_edit hands them over before this point); the curator checks
# below remain for the targets the store re-checks only for integrity.
_VALIDATORS_BY_TARGET = {
    "history_event_person": _validate_history_event_person,
    "history_event": _validate_history_event,
    "history_event_section": _validate_history_event_section,
    "office_row": _validate_office_row,
}


def _validate(cur, target_type: str, action: str, target_id: str, patch: dict) -> str | None:
    """Return an error string, or None when the edit is applicable."""
    if _contains_north_korea(patch):
        return (
            "Error: Korean text contains '북한'. On first reference use "
            "'조선민주주의인민공화국', then '조선'. Rewrite only the affected text."
        )
    em_dash = _em_dash_problem(patch)
    if em_dash:
        return em_dash
    script_leak = _script_leak_problem(patch)
    if script_leak:
        return script_leak
    variants = _find_name_variants(patch)
    if variants:
        fixes = "; ".join(f"'{v}' → '{c}'" for v, c in variants)
        return (
            f"Error: non-standard person-name spelling(s): {fixes}. Spell people "
            "exactly as their own dictionary card does. Keep an original spelling "
            "only inside direct quotation marks (quoted spans are already exempt)."
        )
    allowed = _PATCH_KEYS_BY_TARGET[target_type]
    unknown = set(patch) - allowed
    if unknown:
        return (
            f"Error: unknown patch key(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )
    return _VALIDATORS_BY_TARGET[target_type](cur, action, target_id, patch)


_EVENT_HEADING_RE = re.compile(r"^## +(.+?)[ \t]*$", re.M)


def _split_event_body(body: str) -> list[tuple[str, str]]:
    """Split an event body into [(heading, block)] parts.

    Each block keeps its own `## ` line, so re-joining the blocks reproduces the
    body. Any text before the first heading comes back under an empty heading
    and is never reordered — there is none today, but a hand edit could leave a
    lead paragraph there and a splice must not eat it.
    """
    text = (body or "").strip("\n")
    if not text:
        return []
    matches = list(_EVENT_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    parts: list[tuple[str, str]] = []
    lead = text[: matches[0].start()].strip("\n")
    if lead:
        parts.append(("", lead))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((match.group(1).strip(), text[match.start():end].strip("\n")))
    return parts


def _find_event_section(parts: list[tuple[str, str]], heading: str) -> int:
    """Index of the part whose heading is the same topic, or -1.

    Matched on _dedup_key rather than the raw string for the reason person
    sections are: the same topic came back under punctuation-only variants.
    """
    key = _dedup_key(heading)
    if not key:
        return -1
    for index, (existing, _) in enumerate(parts):
        if existing and _dedup_key(existing) == key:
            return index
    return -1


def _public_page(target_type: str, target_id: str) -> str:
    """The page a successful edit shows up on, for the confirmation message.

    It used to name /commulingo/people whatever had been written, so a glossary
    or event write told the curator to go look at the wrong page.
    """
    section = {
        "term": "terms",
        "history_event": "events",
        "history_event_section": "events",
        "history_event_person": "events",
        "office_row": "offices",
    }.get(target_type, "people")
    suffix = f"/{target_id}" if section in ("terms", "events", "people") else ""
    return f"cyber-lenin.com/commulingo/{section}{suffix}"


def _run_edit(target_type: str, action: str, target_id: str, patch: dict,
              sources: list[str], confidence: float | None) -> str:
    if target_type=='term':
        # Local prose checks first: a write they reject must not pay for a
        # classification on every repair retry.
        prose = {k:v for k,v in patch.items() if k not in {'evidence','sources'}}
        problems = [p for p in (_em_dash_problem(prose),_script_leak_problem(prose),
            "Error: use 조선민주주의인민공화국 or 조선" if _contains_north_korea(prose) else None) if p]
        if problems:
            return '; '.join(problems)
    if target_type=='term' and action=='create' and not patch.get('category'):
        from commulingo.classify import classify_term, fill_term_category
        classification = classify_term(patch)
        if classification is None:
            return "Error: activity classification needs cited bio/career evidence with source, locator, claim and excerpt; if already provided, retry the saved draft after the classifier recovers."
        patch = fill_term_category(patch, classification)
    if target_type=='term':
        try:
            result = call_person_service({'command':'submit','target':'term','action':action,
                'id':target_id,'fields':patch,'sources':sources,'changedBy':_SUGGESTED_BY})
        except ValueError as exc:
            return f'Error: {exc}'
        return f"OK — {result['status']}: {action} term '{target_id}'. Logged as edit #{result['suggestionId']}. Pending review; no content changed."
    if target_type in {"person", "person_section"}:
        fields = {k: v for k, v in patch.items() if k not in {"office_rows", "sections", "revision"}}
        if "origin" in fields:
            fields["nationalOrigin"] = fields.pop("origin")
            fields["evidence"] = [{**e, "field": "nationalOrigin" if e.get("field") == "origin" else e.get("field")}
                                  for e in fields.get("evidence", [])]
        # Existing normalization/terminology repairs run before this boundary;
        # all persistence validation, locks, evidence and suggestions run in JS.
        # The legacy person role was retired (activities replace it); the
        # service ignores the key, so it is never forwarded.
        fields.pop("role", None)
        activity_review_required = False
        if target_type == "person":
            # The evidence check is local; run it before any paid classification
            # so a bad citation string is rejected without a Jev decision. It is
            # repeated below on the classified fields.
            pre_sources = list(dict.fromkeys([*sources, *(e["source"] for a in fields.get("activities") or []
                                                          if isinstance(a, dict)
                                                          for e in a.get("evidence") or [] if e.get("source"))]))
            pre_errors = _person_evidence_errors(fields, pre_sources)
            if pre_errors:
                return "Error: evidence validation: " + "; ".join(pre_errors)
            from commulingo.classify import (classify_person, classify_person_card, classify_person_codes,
                                                          fill_classification, fill_person_codes, missing_person_codes)
            needs_codes = bool(missing_person_codes(fields))
            # A new person is registered with documented activities (the frontend
            # refuses a create without them); a writer-supplied legacy role never
            # not skip the activity classification.
            needs_group = action == "create" and not (fields.get("groupId") and fields.get("activities"))
            if needs_codes and needs_group:
                card = classify_person_card(fields) or {}   # one request for the whole card
                codes, classification = card.get("codes"), card.get("person")
            else:
                codes = classify_person_codes(fields) if needs_codes else None
                classification = classify_person(fields) if needs_group else None
            if needs_codes:
                fields = fill_person_codes(fields, codes)
                if missing_person_codes(fields):
                    return "Error: the classification service is unavailable right now; retry this write later."
            if needs_group:
                if classification is None:
                    return "Error: activity classification needs cited bio/career evidence with source, locator, claim and excerpt; if already provided, retry the saved draft after the classifier recovers."
                fields = fill_classification(fields, classification)
                if classification["low_confidence"]:
                    activity_review_required = bool(classification.get("activities"))
                    logger.info("person %s classified with low confidence: %s", target_id, classification["confidence"])
        if target_type == "person" and fields.get("activities"):
            # An activity list resubmitted with an appended party activity
            # (pipeline party enrichment) keeps citing the card's earlier sources.
            sources = list(dict.fromkeys([*sources, *(e["source"] for a in fields["activities"]
                                                      for e in a.get("evidence") or [] if e.get("source"))]))
        evidence_errors = _person_evidence_errors(fields, sources)
        if evidence_errors:
            return "Error: evidence validation: " + "; ".join(evidence_errors)
        try:
            result = call_person_service({"command": "submit", "target": target_type,
                "action": action, "id": target_id, "fields": fields, "sources": sources,
                "confidence": confidence, "changedBy": _SUGGESTED_BY,
                "directApply": direct_apply_enabled() and not activity_review_required})
        except ValueError as exc:
            return f"Error: {exc}"
        # get_person caches for a few seconds; a re-read right after this write
        # must see the new revision or the next write fails on revision_conflict.
        _reads.invalidate(("person", target_id))
        return (f"OK — {result['status']}: {action} {target_type} '{target_id}'. "
                f"Logged as edit #{result['suggestionId']}. "
                + ("Pending review; no content changed." if result["status"] == "pending"
                   else "Applied through the shared Admin store."))
    # History events, their sections and people, office rows. The curator
    # checks below give the model specific guidance; the CommuLingo editorial
    # service re-checks integrity and owns the write.
    error = _validate(None, target_type, action, target_id, patch)
    if error:
        return error
    pending = next((row for row in _reads.suggestions("pending", 20, target_type, target_id)
                    if row.get("action") == action), None)
    try:
        result = call_person_service({"command": "submit", "target": target_type, "action": action,
            "id": target_id, "fields": patch, "sources": sources, "confidence": confidence,
            "changedBy": _SUGGESTED_BY, "directApply": direct_apply_enabled()})
    except ValueError as exc:
        return f"Error: {exc}"
    sid = result["suggestionId"]
    if result["status"] == "approved":
        return (
            f"OK — applied: {result.get('summary')}. Logged as edit #{sid}. The change is live "
            f"on {_public_page(target_type, target_id)} within ~1 minute "
            "(server cache TTL)."
        )
    msg = (
        f"OK — staged as suggestion #{sid} ({action} {target_type} '{target_id}'). "
        "Staging mode is on: the operator reviews it with "
        "scripts/commulingo_suggestions.py before it goes live."
    )
    if pending:
        msg += (
            f" Note: suggestion #{pending['id']} for the same target/action is "
            "still pending — mention to the operator if this one supersedes it."
        )
    return msg


_BILINGUAL_TEXT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"ko": {"type": "string"}, "en": {"type": "string"}},
    "required": ["ko", "en"],
}

# Single source of truth for bilingual text ceilings, as {field: (ko, en)}.
# The tool schemas (maxLength + description), the save-time _validate checks,
# and the lane/agent prompts (which import this table) are all generated from
# here — never write one of these numbers anywhere else. The curator prompt
# quoting one ceiling while the save enforced another is exactly how runs got
# spent redrafting cards that were already written.
# Mirror of frontend data/commulingo/person-life-years.js LIFE_YEARS, so a bad
# years label fails schema validation in the writer's own call instead of as a
# store 400 after the whole draft is written (a weekly bucket of rejections).
_YEAR_PART = r"(?:\d{3,4}(?:/\d{3,4})?\??|\?)"
_LIFE_YEARS_PATTERN = rf"^(?:$|(?:c\.\s*)?{_YEAR_PART}\s*[–-]\s*(?:{_YEAR_PART}(?: 이후)?)?$)"

FIELD_LIMITS: dict[str, tuple[int, int]] = {
    "epithet": (60, 140),
    "bio": (380, 900),
    "moment": (140, 300),
    "fate_label": (22, 50),
    # A ceiling is a refusal boundary above the target, but at (90, 200) this one
    # sat exactly ON its target: sentence_budget prescribes one sentence and
    # DENSE_SENTENCE_CHARS prices one at 90/200, so a slightly-denser-than-average
    # sentence bounced on arrival. note.en was the largest length-rejection bucket
    # on the site over 2026-08-01..08-08 (15 of 40), every draft between 201 and
    # 253 characters, plus 4 note.ko between 92 and 101. Raised to one dense
    # sentence plus ~25% headroom; sentence_budget still returns 1, so the
    # prescription is unchanged and notes are not meant to get longer. (Stored
    # corpus on 2026-08-08, 1,072 notes: ko p50 58 / en p50 140 — the typical note
    # is nowhere near either number.)
    "event_note": (110, 250),
    "definition": (400, 900),
    # A section body is the one long-form field, and it was the one field left
    # out of this table: no maxLength on the tool schema, no save-time check,
    # only a character band written into the maintainer prompt. This ceiling is
    # a runaway guard, not a target — it sits ~30% above SECTION_BODY_TARGET so
    # an on-target section never lands on it. (Corpus on 2026-08-02, 1,346
    # stored sections: ko p50 724 / p95 1203 / max 2071, en p50 1579 / p95 2574
    # / max 3938 — the whole corpus stays writable under this.)
    "section_body": (2600, 5800),
    # The history event card's short fields. Ceilings sit ~20% above the longest
    # stored value on 2026-08-08 (summary ko 430 / en 952, outcome ko 404 / en
    # 921, question ko 56 / en 124) so every existing event stays rewritable and
    # a card that wants one more clause has room, without the summary growing
    # into the body's job.
    "event_question": (70, 155),
    "event_summary": (520, 1150),
    "event_outcome": (520, 1150),
    # One `## ` part of an event body. Same numbers as a person section, because
    # it is the same kind of object: one topic, written in one run, read as one
    # screen. The whole-body ceiling is EVENT_BODY_CEILING below.
    "event_section_body": (2600, 5800),
}

# The same contract is imported by the JS storage validator. No duplicated
# person ceilings in prompts/tool schemas and the persistence boundary.
_EDITORIAL_CONTRACT = json.loads(commulingo_data_file(
    "person-editorial-contract.json", "COMMULINGO_PERSON_CONTRACT").read_text())
for _field in ("epithet", "bio", "moment", "fate_label"):
    FIELD_LIMITS[_field] = tuple(_EDITORIAL_CONTRACT["limits"][_field])
FIELD_LIMITS["section_body"] = tuple(_EDITORIAL_CONTRACT["limits"]["body"])

# What one event body section should be written to, as (min, max) Korean
# characters — the event twin of SECTION_BODY_TARGET. The two bodies that
# existed before this lane (spanish-civil-war, sino-soviet-split) averaged ~690
# Korean characters per section across 7 sections; that is the floor of what
# reads as a section, not the target. A history event carries more than a person
# section does, so the band is set higher on purpose.
EVENT_SECTION_TARGET = (700, 1800)

# Runaway guard on the assembled body, not a target. At the top of
# EVENT_SECTION_TARGET this pays for roughly a dozen sections, which is a long
# encyclopedia article and well past the point where a reader wants the page
# split. Nothing should ever land on it.
EVENT_BODY_CEILING = (24000, 52000)

# What a section body should actually be written to, as (min, max) Korean
# characters; the English twin follows at the DENSE_SENTENCE_CHARS ratio
# (~2.2x), so 2000 Korean is ~4400 English. A section is the one field whose
# target cannot be derived from its ceiling the way sentence_budget() derives
# the others — the ceiling is a runaway guard well above the target, so
# deriving would prescribe dozens of sentences. It lives here, in the same
# table region as the ceiling, so the schema description and the maintainer
# prompt read one number instead of restating it.
#
# The upper bound was 700 while the corpus median was already 724 and the 95th
# percentile 1203: the prescription described nothing anyone wrote, and nothing
# enforced it either way. Raised to 2000 (2026-08-02) so a substantial section
# is written on purpose rather than over an unstated line.
SECTION_BODY_TARGET = (350, 2000)

# What one dense bilingual sentence of this register actually costs, measured
# over the 1,210 accepted bios: a Korean sentence in a major-figure card runs
# ~90 characters and its English twin ~200. (Accepted 4-sentence Korean bios
# average 296 characters, 5-sentence ones 308 — but those are the post-rejection
# survivors; first drafts of dense old-regime cards came in at 414-480.)
DENSE_SENTENCE_CHARS = (90, 200)


def sentence_budget(field: str) -> int:
    """How many dense bilingual sentences a field's ceiling actually pays for.

    The prescription and the ceiling used to be set independently, and they
    disagreed by exactly one sentence: the writer was told 4-5 sentences for a
    major figure while 380 Korean characters buys 4, so 17 of 19 curator
    rejections in a 24h window were a fifth sentence overflowing bio. Deriving
    the count from the table means raising a ceiling raises the prescription
    with it instead of leaving the two to drift apart again.
    """
    ko_max, en_max = FIELD_LIMITS[field]
    ko_cost, en_cost = DENSE_SENTENCE_CHARS
    return max(1, min(ko_max // ko_cost, en_max // en_cost))


def sentence_prescription(field: str) -> str:
    """The sentence count for a short field, as the phrase the prompts use.

    Same rule as sentence_budget, one layer up: the phrase itself is generated
    so a ceiling change rewrites every prompt that quotes it. "one sentence,
    two at most" was written by hand into six places (two curator prompt
    bullets, the event-note and moment tool descriptions, the moment save
    error, the maintainer prompt) under ceilings that pay for exactly one —
    90 Korean characters is one dense sentence, and 200 English is too. Those
    six strings made note.ko/note.en the largest single rejection bucket on the
    site: 19 of 48 length rejections in the 2026-07-30..08-02 window, every one
    of them a second sentence the ceiling never had room for.
    """
    budget = sentence_budget(field)
    if budget == 1:
        # No character counts in the phrase: every caller states the ceiling on
        # its own line already, and repeating it inside the prescription is how
        # a prompt ends up with two numbers to keep in step.
        return "exactly one sentence — a second one does not fit"
    return f"{budget - 1}-{budget} sentences"


# The ceilings are stated in each description because a bare maxLength
# only reports itself after the call is already spent — 107 of the 625
# rejected person_create calls were that error.
# They are refusal boundaries, NOT targets: length is prescribed to the writer
# as a sentence count (see CARD_STYLE_GUIDANCE), because a character band
# produces padded or truncated Korean. Say the ceiling, never ask for it.
# `target` names what the writer aims at instead of this ceiling: a sentence
# count for the short fields, a character range for the one long-form field
# (a section is markdown and may run to paragraphs, so counting its sentences
# prescribes nothing).
_CEILING = (
    "Hard ceiling {ko} Korean / {en} English characters; write to the prescribed {target},"
    " not to the ceiling. Check the English side, which overruns first, before calling."
)


def _capped_bilingual_schema(field: str, extra: str = "", target: str = "sentence count") -> dict:
    ko_max, en_max = FIELD_LIMITS[field]
    return {
        **_BILINGUAL_TEXT_SCHEMA,
        "description": _CEILING.format(ko=ko_max, en=en_max, target=target) + extra,
        "properties": {"ko": {"type": "string", "maxLength": ko_max},
                       "en": {"type": "string", "maxLength": en_max}},
    }


_EPITHET_SCHEMA = _capped_bilingual_schema(
    "epithet",
    " One clause; a second clause belongs in the bio.",
)

_BIO_SCHEMA = _capped_bilingual_schema(
    "bio",
    " Use the tier's sentence count from the task; career chronology goes in career rows."
    f" A dense sentence costs ~{DENSE_SENTENCE_CHARS[0]} Korean / ~{DENSE_SENTENCE_CHARS[1]}"
    f" English characters, so the ceiling fits {sentence_budget('bio')}. Over by a sentence:"
    " cut a sentence. Slightly over: delete the weakest clause, do not reword.",
)

_MOMENT_SCHEMA = _capped_bilingual_schema(
    "moment",
    " Pull-quote on the person list card: 44-85 Korean characters = 2 lines, 86-127 = 3."
    " Excerpt an overlong quotation to its sharpest clause with '…' or pick a shorter"
    " one; never pad.",
)

_EVENT_NOTE_SCHEMA = _capped_bilingual_schema(
    "event_note",
    " Caption under the person's name on the event page: what the person did in the"
    f" event, {sentence_prescription('event_note')}.",
)

# The section body is markdown and may carry paragraphs, so its guidance is a
# character target rather than a sentence count. Both numbers come from
# SECTION_BODY_TARGET; the ceiling in _CEILING comes from FIELD_LIMITS.
_SECTION_BODY_SCHEMA = _capped_bilingual_schema(
    "section_body",
    f" One topic, {SECTION_BODY_TARGET[0]}-{SECTION_BODY_TARGET[1]} Korean characters"
    " plus equivalent English. A body near the ceiling should be two sections or"
    " repeats the bio.",
    target="length",
)

# Fate label = cause of death only, NO death year (it renders from `years`).
# Execution → 처형/Executed; vague natural death → 자연사/Natural causes; keep a
# specific illness word (심장마비/폐암…); place with " · " (암살 · 멕시코). A
# deposed/exile fate keeps its EVENT year (실각 1964) and may append the cause
# (실각 1964 · 자연사). The death year is stripped automatically on save.
_FATE_LABEL_SCHEMA = _capped_bilingual_schema(
    "fate_label",
    " Cause of death only, no death year (dropped on save): 실각 1964 · 자연사 / "
    "Removed 1964 · natural causes. Execution=처형/Executed; natural=자연사/Natural "
    "causes; place after ' · '.",
    # A badge, not prose: there is no sentence to count.
    target="length",
)

_NATIONALITY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "code": {"type": "string", "enum": sorted(_NATIONALITY_CODES)},
        "label": _BILINGUAL_TEXT_SCHEMA,
    },
    "required": ["code", "label"],
}

_NATIONALITY_POLICY = json.loads(commulingo_data_file(
    "nationality-policy.json", "COMMULINGO_NATIONALITY_POLICY").read_text())
_NATIONAL_ORIGIN_CODES = _NATIONALITY_CODES - set(_NATIONALITY_POLICY["citizenshipOnlyCodes"])

_NATIONAL_ORIGIN_SCHEMA = {
    **_NATIONALITY_SCHEMA,
    "properties": {**_NATIONALITY_SCHEMA["properties"],
        "code": {"type": "string", "enum": sorted(_NATIONAL_ORIGIN_CODES)}},
    "description": _NATIONALITY_POLICY["originGuidance"] + (
        " Not place of death either. Examples: Radek=Poland (born in Ukraine); "
        "Yezhov=Russia (ethnic Russian born in Lithuania); a non-Russian Soviet "
        "official keeps that nation (Sillari=estonia, Gumbaridze=georgia)."
    ),
}

_COMMULINGO_FIELD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "description": (
        "Canonical patch. Scalar fields are strings; bilingual fields are {ko,en}. "
        "Person create includes givenName/familyName (or legacy name), bio, epithet, "
        "groupId, activities, years, aliases, career and native-script name fields. "
        "Empty object only for delete."
    ),
    "properties": {
        "id": {"type": "string"},
        "group": {"type": "string"},
        "groupId": {"type": "string"},
        # null is the honest way to say "no preference"; every writer below falls
        # back to MAX(sort_order)+1 when this is not an int, so null already meant
        # append. Declaring only "integer" made the model discover that by failing:
        # `None is not of type 'integer' at 'fields.sortOrder'` was the single
        # largest term_create rejection (31 of 66) and 20 more on person_create.
        "sortOrder": {
            "type": ["integer", "null"],
            "description": "Explicit position. Omit or null to append after the current last row.",
        },
        "cyrillic": {"type": "string", "description": (
            "Native-script name (legacy column name) in the person's own script per "
            "citizenship: 김무정, 彭德怀, 'Võ Nguyên Giáp', 'Kádár János', 'Владимир Ленин'. "
            "Never a Russian transliteration of a non-Russian name; Latin-script nations "
            "use the native Latin spelling with diacritics. Never omit it."
        )},
        "cyrillicPatronymic": {"type": "string"},
        "years": {"type": "string", "pattern": _LIFE_YEARS_PATTERN, "description": (
                "Life years: 1878–1943; 1987– for a living person; ?– living with unknown "
                "birth year; ?–? unknown; 1895?–1940 uncertain year; c. 1953–2001 approximate; 1917–1940 이후 when only "
                "a last known year exists. Never 현재/present, no other words.")},
        "name": _BILINGUAL_TEXT_SCHEMA,
        "givenName": _BILINGUAL_TEXT_SCHEMA,
        "familyName": _BILINGUAL_TEXT_SCHEMA,
        "epithet": _EPITHET_SCHEMA,
        "bio": _BIO_SCHEMA,
        "moment": _MOMENT_SCHEMA,
        "patronymic": _BILINGUAL_TEXT_SCHEMA,
        "nameQualifier": {**_BILINGUAL_TEXT_SCHEMA, "description": (
            "Only when another card already has this exact name: a short disambiguator shown in lists as "
            "「오토 브라운 (프로이센 총리)」 — {\"ko\": \"프로이센 총리\", \"en\": \"Prussian premier\"}, without brackets. "
            "Give the existing namesake one too. Never put brackets in name parts; other names go to aliases.")},
        "citizenship": _NATIONALITY_SCHEMA,
        "origin": _NATIONAL_ORIGIN_SCHEMA,
        "nationalOrigin": _NATIONAL_ORIGIN_SCHEMA,
        "term": _BILINGUAL_TEXT_SCHEMA,
        "original": {"type": "string", "description": "term: native-script/original-language form (ГУЛАГ, нэпман)."},
        "period": {
            **PERIOD_SCHEMA,
            "type": ["object", "null"],
            "description": (
                "term: when it applies, as numbers in the career-period shape: "
                "{\"start\": [1930], \"end\": [1960]}; {\"start\": [1980], \"startQual\": \"decade\", "
                "\"ongoing\": true} for 1980년대–현재; {\"start\": [1921, 3]} for a single date. "
                "null for an undated concept. label {ko, en} only for what the numbers cannot say "
                "(still give start when there is one): {\"start\": [1800], \"label\": {\"ko\": "
                "\"19세기 말–현재\", \"en\": \"late 19th century–present\"}}. "
                + PERIOD_SCHEMA["description"]
            ),
        },
        "periodNote": {
            **_BILINGUAL_TEXT_SCHEMA,
            "description": (
                "term: optional remark that qualifies the period, shown as 1917–1921 (전성기): "
                "{\"ko\": \"전성기\", \"en\": \"heyday\"}. Both languages or neither."
            ),
        },
        "category": {
            "type": "string", "enum": list(_TERM_CATEGORIES),
            "description": f"term: which glossary group it belongs to. One of {_TERM_CATEGORY_HINT}.",
        },
        "parentId": {
            "type": ["string", "null"],
            "description": (
                "term: id of the entry this one is a part of (예조프시나 -> great-purge). "
                "Only for a component, never an adjacent concept (those go in related "
                "terms). One nesting level: a parent cannot have a parent. null detaches."
            ),
        },
        "definition": _capped_bilingual_schema(
            "definition",
            " term: the card paragraph, 2-3 sentences"
            f" (~{DENSE_SENTENCE_CHARS[0]} Korean characters each); a third fits only if"
            " the first two are tight. Qualifications and historiography go to body.",
        ),
        # Terms accept null at the write boundary to mean "clear the body"; the
        # schema declaring only "object" made the model discover that by failing
        # (`None is not of type 'object' at 'fields.body'`).
        "body": {
            **_BILINGUAL_TEXT_SCHEMA,
            "type": ["object", "null"],
            "description": (
                "Bilingual markdown body. Omit the key to keep the stored value; "
                "for terms, null clears it."
            ),
        },
        "people": {"type": "array", "items": {"type": "string"}, "description": "term: related person ids."},
        "events": {"type": "array", "items": {"type": "string"}, "description": "term: related history event ids."},
        "aliases": {
            "type": "object", "additionalProperties": False,
            "description": (
                "Person: optional; {\"ko\": [], \"en\": []} when the person has no other name. "
                "Only genuinely different names (real name, pseudonym, maiden/married name, other "
                "established transliteration, English initials form); never the headword, the "
                "name with patronymic or the bare one-word surname; no parenthetical note, quotes "
                "or label (본명, born, aka); ko in Hangul, en in Latin script, no native script. "
                "Term: the exact variant strings prose uses."
            ),
            "properties": {
                "ko": {"type": "array", "items": {"type": "string"}},
                "en": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ko", "en"],
        },
        "career": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"period": PERIOD_SCHEMA, "r": _BILINGUAL_TEXT_SCHEMA},
                "required": ["period", "r"],
            },
        },
        "activities": json.loads(commulingo_data_file("activity-schema.json", "COMMULINGO_ACTIVITY_SCHEMA").read_text()),
        "fate": {
            "type": "object", "additionalProperties": False,
            "properties": {
                # Enumerated here so the closed vocabulary is readable up front;
                # it was only discoverable by tripping the write-boundary check.
                "kind": {"type": "string", "enum": list(_FATE_KINDS),
                         "description": "Use the empty string when the fate is unconfirmed; explain uncertainty in label. Imprisonment is not exile or execution. Never invent unknown/unconfirmed enum values."},
                "label": _FATE_LABEL_SCHEMA,
            },
            "required": ["kind", "label"],
        },
        "scenes": {"type": "array", "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}},
        "slug": {"type": "string"},
        "heading": _BILINGUAL_TEXT_SCHEMA,
        # 'body' is defined once above (nullable, for terms and sections) — a
        # second bare entry here would silently override it, like 'period' once did.
        "sources": {"type": "array", "items": {"type": "string"}},
        # 'period' is defined once above, for terms. A second bare
        # {"type": "string"} entry used to sit here and, being later in the same
        # dict literal, silently overrode it, so the term tool advertised an
        # undocumented string. Office-row patches accept `period` as an alias
        # for `years` in _validate, not through this schema.
        "personId": {"type": "string"},
        "relationKind": {"type": "string", "enum": list(_HISTORY_RELATION_KINDS)},
        "side": {"type": ["string", "null"],
                 "description": "History event link only: the camp id from the event's sides, when it names them."},
        "relation": _BILINGUAL_TEXT_SCHEMA,
        "note": _BILINGUAL_TEXT_SCHEMA,
        "office_rows": {"type": "array", "items": {"type": "object"}},
        "sections": {"type": "array", "items": {"type": "object"}},
    },
}


class CommulingoInputError(ValueError):
    """Machine-classified caller error at the shared CommuLingo write boundary."""

    def __init__(self, code: str, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _commulingo_error(code: str, message: str, *, retryable: bool = True) -> str:
    return "Error: " + json.dumps(
        {"ok": False, "error": {"code": code, "message": message, "retryable": retryable}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _classify_validation_error(message: str) -> str:
    text = str(message or "").lower()
    if "already exists" in text or "already registered" in text or "same person" in text:
        return "duplicate"
    if "not found" in text or "does not exist" in text or "unknown group" in text:
        return "invalid_reference"
    if "required" in text or "needs " in text:
        return "missing_required"
    if "too long" in text:
        return "length_limit"
    if "unknown patch key" in text:
        return "unknown_field"
    return "validation_failed"


def _normalize_soviet_korean_content(value, *, korean: bool = False, excluded: bool = False):
    """Rewrite Korean public copy while preserving modern citizenship labels."""
    if isinstance(value, str):
        if korean and not excluded:
            return value.replace("조지아", "그루지야")
        return value
    if isinstance(value, list):
        return [
            _normalize_soviet_korean_content(item, korean=korean, excluded=excluded)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _normalize_soviet_korean_content(
                item,
                korean=korean or key == "ko",
                excluded=excluded or key == "citizenship",
            )
            for key, item in value.items()
        }
    return value


_LITERAL_NEWLINE_RE = re.compile(r"\\r\\n|\\n")


def _unescape_literal_newlines(value, applied: list):
    """Turn a typed-out \\n back into the line break the writer meant.

    A curator composing markdown in a JSON tool argument can escape the
    backslash as well as the n, and then what reaches the column is two
    characters rather than a line break. Markdown has no idea what to do with
    them, so the reader sees \\n\\n printed in the middle of a paragraph — which
    is what happened to one section of `revolution-1905` on 2026-08-09, eighteen
    times in each language.

    Nothing on this site legitimately writes a backslash before an n, so the
    substitution is safe and is made here rather than refused: a refusal costs a
    paid round and risks losing the manuscript over a typing artefact.
    """
    if isinstance(value, str):
        fixed = _LITERAL_NEWLINE_RE.sub("\n", value)
        if fixed != value:
            applied.append(len(value) - len(fixed))
        return fixed
    if isinstance(value, list):
        return [_unescape_literal_newlines(item, applied) for item in value]
    if isinstance(value, dict):
        return {key: _unescape_literal_newlines(item, applied) for key, item in value.items()}
    return value


def _normalize_localized_spellings(value, applied: dict, lang: str = ""):
    """Apply the name-spelling registry to every {ko, en} string in a patch.

    normalize_spellings_in_text has done the careful part since the registry was
    added (masking direct quotations and the blocked compounds like 시베리아 ⊃
    베리아) but nothing ever called it: the only live path was the _validate
    rejection, which costs a paid round to fix a substitution with exactly one
    right answer. The 조지아 -> 그루지야 rule in the same file has always been
    applied here rather than refused, and a variant spelling is the same kind of
    rule. _validate keeps its check as a backstop for writers that reach it
    without passing through here.
    """
    if isinstance(value, str):
        if lang not in ("ko", "en"):
            return value
        fixed, hits = normalize_spellings_in_text(value, lang)
        applied.update(hits)
        return fixed
    if isinstance(value, list):
        return [_normalize_localized_spellings(item, applied, lang) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_localized_spellings(
                item, applied, key if key in ("ko", "en") else lang
            )
            for key, item in value.items()
        }
    return value


def normalize_commulingo_write(
    target_type: str,
    target_id: str,
    fields: dict | None,
    citations: list | None,
    confidence: float | None = None,
) -> tuple[dict, list[str], float | None, list[str]]:
    """Canonicalize all generic and narrow writes before DB validation.

    The old maintainer emitted several legacy key shapes and sometimes nested
    tool-level citations/confidence inside a person patch. Normalize those once
    here so every caller reaches the same strict target-specific allow-list.
    """
    if target_type not in _TARGET_TYPES:
        raise CommulingoInputError("invalid_target", f"target_type must be one of {_TARGET_TYPES}")
    if not isinstance(fields, dict):
        raise CommulingoInputError("invalid_fields", "fields must be an object")
    normalized = dict(fields)
    repairs: list[str] = []
    normalized_citations = [
        s.strip() for s in (citations or []) if isinstance(s, str) and s.strip()
    ]

    if target_type == "person":
        if "nationalOrigin" in normalized:
            if "origin" in normalized:
                legacy = normalized.get("origin") if isinstance(normalized.get("origin"), dict) else {}
                explicit = normalized.get("nationalOrigin") if isinstance(normalized.get("nationalOrigin"), dict) else {}
                if (legacy.get("code") or "") != (explicit.get("code") or ""):
                    raise CommulingoInputError(
                        "conflicting_national_origin",
                        "origin and nationalOrigin disagree; send nationalOrigin only",
                    )
            normalized["origin"] = normalized.pop("nationalOrigin")
            repairs.append("nationalOrigin->origin")
        localized = {
            "name": ("nameKo", "nameEn"),
            "givenName": ("givenNameKo", "givenNameEn"),
            "familyName": ("familyNameKo", "familyNameEn"),
            "nameQualifier": ("nameQualifierKo", "nameQualifierEn"),
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
        if normalized.get("slug") == target_id:
            normalized.pop("slug", None)
            repairs.append("dropped redundant slug")
        misplaced_sources = normalized.pop("sources", None)
        if misplaced_sources is not None:
            if not normalized_citations and isinstance(misplaced_sources, list):
                normalized_citations = [
                    s.strip() for s in misplaced_sources if isinstance(s, str) and s.strip()
                ]
            repairs.append("fields.sources->citations")
        misplaced_confidence = normalized.pop("confidence", None)
        if misplaced_confidence is not None:
            if confidence is None:
                confidence = misplaced_confidence
            repairs.append("fields.confidence->confidence")

    escaped_breaks: list = []
    normalized = _unescape_literal_newlines(normalized, escaped_breaks)
    if escaped_breaks:
        repairs.append(f"literal \\n -> line break ({len(escaped_breaks)} field(s))")

    terminology_normalized = _normalize_soviet_korean_content(normalized)
    if terminology_normalized != normalized:
        normalized = terminology_normalized
        repairs.append("조지아->그루지야 in Korean content")

    spelling_hits: dict = {}
    normalized = _normalize_localized_spellings(normalized, spelling_hits)
    if spelling_hits:
        repairs.append(
            "name spellings: "
            + ", ".join(f"{variant}->{canonical}" for variant, canonical in spelling_hits.items())
        )

    allowed = _PATCH_KEYS_BY_TARGET[target_type]
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise CommulingoInputError(
            "unknown_field",
            f"unknown {target_type} field(s): {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}",
        )
    if not normalized_citations:
        raise CommulingoInputError(
            "missing_citations",
            "at least one citation is required (URL/reference plus what it supports)",
        )
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise CommulingoInputError("invalid_confidence", "confidence must be a number between 0 and 1") from exc
        if not 0.0 <= confidence <= 1.0:
            raise CommulingoInputError("invalid_confidence", "confidence must be between 0 and 1")
    return normalized, normalized_citations, confidence, repairs


async def _exec_commulingo_write(
    target_type: str,
    action: str,
    target_id: str,
    sources: list,
    patch: dict | None = None,
    confidence: float | None = None,
) -> str:
    if target_type not in _TARGET_TYPES:
        return _commulingo_error("invalid_target", f"target_type must be one of {_TARGET_TYPES}")
    if action not in _ACTIONS:
        return _commulingo_error("invalid_action", f"action must be one of {_ACTIONS}")
    target_id = (target_id or "").strip()
    if not target_id:
        return _commulingo_error("missing_target_id", "target_id is required")
    if action == "create":
        # Models write ids straight from the person's name, diacritics included
        # (václav-havel), and the old "must be kebab-case" rejection taught them
        # to swap the offending letters for dashes: twenty /people/v-clav-havel
        # style URLs shipped on 2026-08-09. Fold the id server-side instead.
        folded = _fold_slug(target_id)
        if folded != target_id:
            patch = dict(patch or {})
            if isinstance(patch.get("id"), str):
                patch["id"] = _fold_slug(patch["id"])
            target_id = folded
    patch = patch or {}
    if action != "delete" and not patch:
        return _commulingo_error("missing_fields", "fields are required for create/update")
    try:
        patch, sources, confidence, repairs = normalize_commulingo_write(
            target_type, target_id, patch, sources, confidence
        )
        # Expand references after prose normalization so citation bytes are
        # copied exactly, including any intentional literal escape sequences.
        if target_type in {"person", "person_section", "term"} and "evidence" in patch:
            from commulingo.evidence import resolve_evidence_sources
            try:
                patch = {**patch, "evidence": resolve_evidence_sources(patch["evidence"], sources)}
            except ValueError as exc:
                return _commulingo_error("invalid_source_id", str(exc))
        result = await asyncio.to_thread(
            _run_edit, target_type, action, target_id, patch, sources, confidence
        )
        if result.startswith("Error:"):
            message = result.removeprefix("Error:").strip()
            return _commulingo_error(_classify_validation_error(message), message)
        if repairs:
            logger.info("commulingo write normalized: %s", ", ".join(repairs))
        return result
    except CommulingoInputError as e:
        return _commulingo_error(e.code, str(e), retryable=e.retryable)
    except Exception as e:
        logger.warning("commulingo write error: %s", e)
        return _commulingo_error("internal_error", f"{type(e).__name__}: {e}", retryable=True)


_EVIDENCE_SCHEMA = {
    "type": "array",
    "items": {"type": "object", "additionalProperties": False,
        "properties": {**{key: {"type": "string"} for key in
            ("field", "claim", "source", "locator", "excerpt")},
            "source_id": {"type": "string", "pattern": "^S[1-9][0-9]*$", "description": "S1=citations[0], S2=citations[1]."},
            "stance": {"type": "string", "enum": ["supports", "disputes"]}},
        "required": ["field", "claim", "locator"],
        "anyOf": [{"required": ["source_id"]}, {"required": ["source"]}]},
    "description": "Prefer source_id over source text. Each supplied factual field needs its own claim and locator. A source must equal the full citation string. stance=disputes for conflicting evidence.",
}
_PAIR_SCHEMA = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"}}
_COLLECTION_SCHEMAS = {
    "aliasEdits": {"lang": {"type": "string", "enum": ["ko", "en"]}, "value": {"type": "string"}, "replacement": {"type": "string"}},
    "sceneEdits": {"scene": _PAIR_SCHEMA, "replacement": _PAIR_SCHEMA},
    "careerEdits": {"id": {"type": "string"}, "entry": {"type": "object", "properties": {
        "period": PERIOD_SCHEMA, "r": _BILINGUAL_TEXT_SCHEMA}, "additionalProperties": False}},
}
for _key, _properties in _COLLECTION_SCHEMAS.items():
    _COMMULINGO_FIELD_SCHEMA["properties"][_key] = {"type": "array", "maxItems": 100,
        "items": {"type": "object", "additionalProperties": False,
            "properties": {"op": {"type": "string", "enum": ["add", "update", "remove"]}, **_properties}, "required": ["op"]}}

_COMMULINGO_FIELD_SCHEMA["properties"].update({
    "expectedRevision": {"type": "string", "description": "Copy revision from get_person. Required on update; never refresh it without reconciling a conflict."},
    "evidence": _EVIDENCE_SCHEMA,
    "reviewFlags": {"type": "array", "items": {"type": "string", "enum": ["identity_uncertain", "source_conflict"]}},
})

def _narrow_fields_schema(keys: tuple[str, ...], *, required: tuple[str, ...] = ()) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {key: _COMMULINGO_FIELD_SCHEMA["properties"][key] for key in keys},
        **({"required": list(required)} if required else {}),
    }


_PERSON_NARROW_KEYS = (
    "groupId", "sortOrder", "cyrillic", "cyrillicPatronymic", "years",
    "givenName", "familyName", "nameQualifier", "epithet", "bio", "moment", "patronymic",
    "citizenship", "nationalOrigin", "aliases", "career", "activities", "fate", "scenes",
    "expectedRevision", "evidence", "reviewFlags", "aliasEdits", "careerEdits", "sceneEdits",
)
_TERM_NARROW_KEYS = (
    "sortOrder", "term", "original", "period", "periodNote",
    "category", "definition", "body", "aliases", "people", "events", "parentId", "evidence", "expectedRevision",
)
# Create fills the card's source list from the top-level citations. An update
# is usually partial, so it must leave the stored list alone unless the caller
# means to replace it — hence 'sources' is writable only on update.
_TERM_UPDATE_NARROW_KEYS = _TERM_NARROW_KEYS + ("sources",)
_OFFICE_ROW_NARROW_KEYS = (
    "sortOrder", "body", "personId", "name", "note",
)

_CITATIONS_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {"type": "string"},
    "description": "Source URL/reference plus the fact it supports. Never put citations inside fields.",
}


def _person_evidence_errors(fields: dict, sources: list[str]) -> list[str]:
    """Batch actionable evidence diagnostics before RPC; the store still validates."""
    evidence = fields.get("evidence", [])
    if not isinstance(evidence, list):
        return ["evidence must be an array"]
    errors = []
    covered = set()
    for index, item in enumerate(evidence):
        path = f"evidence[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{path} must be an object")
            continue
        for key in ("field", "claim", "source", "locator"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                errors.append(f"{path}.{key} must be non-empty text")
        field = item.get("field")
        if isinstance(field, str):
            covered.add(field)
            if field not in fields and field not in _EDITORIAL_CONTRACT["factFields"]:
                errors.append(f"{path}.field={field!r} is not part of this edit; use the field the claim supports")
        if item.get("source") not in sources:
            errors.append(f"{path}.source must exactly equal one complete citations entry, including its description; copy the citation verbatim, not just its URL")
    missing = [field for field in _EDITORIAL_CONTRACT["factFields"]
               if field in fields and field not in covered]
    if missing:
        errors.append("add separate evidence items for every supplied factual field: " + ", ".join(missing))
    return errors


def _person_write_tool(name: str, action: str) -> dict:
    update_only = {'expectedRevision', 'aliasEdits', 'careerEdits', 'sceneEdits'}
    field_keys = tuple(key for key in _PERSON_NARROW_KEYS
                       if action != 'create' or key not in update_only)
    # groupId and activities are assigned by the runner (commulingo_classify): the
    # model never classifies, so a create does not even carry the fields.
    if action == "create":
        field_keys = tuple(key for key in field_keys if key not in {"group", "groupId", "activities"})
    required_fields = (
        "epithet", "bio", "career", "citizenship", "nationalOrigin", "evidence",
    ) if action == "create" else ("expectedRevision", "evidence")
    fields_schema = _narrow_fields_schema(field_keys, required=required_fields)
    # The runner assigns citizenship.code, nationalOrigin.code and fate.kind
    # from the labels (commulingo_classify.classify_person_codes); the model
    # writes labels only, so the code keys leave the schema. Copy the shared
    # objects before editing them.
    for key, code in (("citizenship", "code"), ("nationalOrigin", "code"), ("origin", "code"), ("fate", "kind")):
        if key in fields_schema["properties"]:
            stripped = deepcopy(fields_schema["properties"][key])
            stripped["properties"].pop(code, None)
            stripped["required"] = [r for r in stripped.get("required", []) if r != code]
            fields_schema["properties"][key] = stripped
    return {
        "name": name,
        "description": (
            f"{action.title()} one CommuLingo person card. Person fields only; citations are a "
            "separate top-level argument. Public text is bilingual {ko,en}. evidence and reviewFlags "
            "go inside fields; expectedRevision and collection edits are update-only. The runner "
            "assigns group, primary activity and the citizenship/nationalOrigin/fate codes from labels and text: "
            "write labels only. Read the record and reference lists first. On create, citizenship "
            "and nationalOrigin need evidence; if unknown, research or defer, never guess. Soviet "
            "and Yugoslav codes are citizenship-only. nationalOrigin is national/ethnic background, "
            "never birthplace; preserve mixed ancestry in labels. Russian-style names need a "
            "researched complete patronymic {ko,en} plus cyrillicPatronymic; omitted PATCH subfields "
            "are preserved. Every supplied bio, moment, years, citizenship and nationalOrigin needs "
            "its own fields.evidence item (that field name, claim, source, locator); evidence.source "
            "must equal a full citations entry including its description, not just the URL. Updates "
            "send only changed fields and no evidence for omitted fields. Conflicting evidence or "
            "large deletions are staged for review. A successful call ends the run."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "person_id": {"type": "string", "description": "Existing id or new lowercase kebab-case slug."},
                "fields": fields_schema,
                "citations": _CITATIONS_SCHEMA,
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["person_id", "fields", "citations"],
        },
    }


COMMULINGO_PERSON_CREATE_TOOL = _person_write_tool("commulingo_person_create", "create")
COMMULINGO_PERSON_UPDATE_TOOL = _person_write_tool("commulingo_person_update", "update")

COMMULINGO_SECTION_SAVE_TOOL = {
    "name": "commulingo_section_save",
    "description": "Create or update one bilingual long-form person section. The server generates the slug on create; supply an existing slug on update. Citations are stored on the section.",
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "expected_revision": {"type": "string", "description": "Current get_person revision (also required to create a section)."},
            "evidence": _EVIDENCE_SCHEMA,
            "action": {"type": "string", "enum": ["create", "update"]},
            "person_id": {"type": "string"},
            "slug": {"type": "string", "description": "Existing section slug for update only; omit on create."},
            "heading": _BILINGUAL_TEXT_SCHEMA,
            "body": _SECTION_BODY_SCHEMA,
            "start_year": {"type": ["integer", "null"], "minimum": -3000, "maximum": 2100, "description": (
                "Year the section's period opens (1898 for '1898-1918'); sections render in this "
                "order. For a theme, the year it begins (legacy: the year of death). null only "
                "when no year applies: create appends after the last section, update keeps the "
                "current position."
            )},
            "start_month": {"type": ["integer", "null"], "minimum": 1, "maximum": 12,
                            "description": "Month within start_year when the source gives it; otherwise omit."},
            "citations": _CITATIONS_SCHEMA,
        },
        "required": ["action", "person_id", "heading", "body", "start_year", "citations", "expected_revision", "evidence"],
    },
}

COMMULINGO_EVENT_LINK_TOOL = {
    "name": "commulingo_event_link",
    "description": "Create one sourced person-to-history-event relation. Never use for a weak connection.",
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "event_id": {"type": "string"},
            "person_id": {"type": "string"},
            "relation_kind": {"type": "string", "enum": list(_HISTORY_RELATION_KINDS)},
            "side": {"type": ["string", "null"],
                     "description": "When the event names sides: the id of the camp the person acted for "
                                    "(null for a witness, historian or someone on no side). Omit otherwise."},
            "relation": _BILINGUAL_TEXT_SCHEMA,
            "note": _EVENT_NOTE_SCHEMA,
            "sort_order": {"type": ["integer", "null"], "description": "Omit or null to append."},
            "citations": _CITATIONS_SCHEMA,
        },
        "required": ["event_id", "person_id", "relation_kind", "relation", "note", "citations"],
    },
}

COMMULINGO_OFFICE_ROW_SAVE_TOOL = {
    "name": "commulingo_office_row_save",
    "description": (
        "Create or update one sourced bilingual institution leadership row. "
        "For create, target_id is the office id; for update, target_id is the numeric row id."
    ),
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["create", "update"]},
            "target_id": {"type": "string"},
            "fields": (lambda schema: {**schema, "properties": {**schema["properties"], "period": PERIOD_SCHEMA}})(
                _narrow_fields_schema(_OFFICE_ROW_NARROW_KEYS)),
            "citations": _CITATIONS_SCHEMA,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["action", "target_id", "fields", "citations"],
    },
}

_EVENT_QUESTION_SCHEMA = _capped_bilingual_schema(
    "event_question",
    extra=(
        " The question the page answers, phrased as a question: the line under the"
        " title, not a summary."
    ),
)
_EVENT_SUMMARY_SCHEMA = _capped_bilingual_schema(
    "event_summary",
    extra=" What happened, for a reader new to the period.",
    target="short paragraph",
)
_EVENT_OUTCOME_SCHEMA = _capped_bilingual_schema(
    "event_outcome",
    extra=" Its consequences, which the rest of the site refers back to.",
    target="short paragraph",
)
_EVENT_SECTION_BODY_SCHEMA = _capped_bilingual_schema(
    "event_section_body",
    extra=(
        " Markdown: one part of the story with the dates, names and figures a reader"
        " would otherwise look up. No `#` or `##` headings inside; the heading argument"
        " supplies it."
    ),
    target=f"{EVENT_SECTION_TARGET[0]}-{EVENT_SECTION_TARGET[1]} Korean characters",
)

COMMULINGO_EVENT_UPDATE_TOOL = {
    "name": "commulingo_event_update",
    "description": (
        "Update one history event's card fields: question, summary, outcome, timeline, "
        "sources. The long-form body goes through commulingo_event_section_save. Read the "
        "event with commulingo_people(action='get_event') first. timeline and sources "
        "replace the stored lists whole, so include existing entries. A successful call "
        "ends the run."
    ),
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "event_id": {"type": "string", "description": "Existing event id from list_events."},
            "fields": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "question": _EVENT_QUESTION_SCHEMA,
                    "summary": _EVENT_SUMMARY_SCHEMA,
                    "outcome": _EVENT_OUTCOME_SCHEMA,
                    "timeline": {
                        "type": "array",
                        "minItems": 1,
                        "description": (
                            "The whole timeline in chronological order: existing entries "
                            "plus new ones; an omitted entry is deleted."
                        ),
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "properties": {
                                "date": {"type": "string", "description": "'1936', '1936.02' or '1936.07.18'."},
                                "title": _BILINGUAL_TEXT_SCHEMA,
                                "body": _BILINGUAL_TEXT_SCHEMA,
                            },
                            "required": ["date", "title", "body"],
                        },
                    },
                    "sources": {
                        "type": "array", "minItems": 1, "items": {"type": "string"},
                        "description": (
                            "The whole source list (book: author, title, publisher, year; "
                            "document: URL). Replaces the stored list."
                        ),
                    },
                },
            },
            "citations": _CITATIONS_SCHEMA,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["event_id", "fields", "citations"],
    },
}

COMMULINGO_EVENT_SECTION_SAVE_TOOL = {
    "name": "commulingo_event_section_save",
    "description": (
        "Write one `## ` section of a history event's long-form body; send only this "
        "section, never the whole body. 'create' adds a section, 'update' rewrites the one "
        "with the matching heading. The heading must say what the part is about "
        "('무기보다 먼저 온 것들'), not label it ('배경 3'). Read the event with "
        "commulingo_people(action='get_event') first to see existing sections. A "
        "successful call ends the run."
    ),
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["create", "update"]},
            "event_id": {"type": "string"},
            "heading": _BILINGUAL_TEXT_SCHEMA,
            "body": _EVENT_SECTION_BODY_SCHEMA,
            "after": {
                **_BILINGUAL_TEXT_SCHEMA,
                "description": (
                    "On create: the existing heading (per language) this section "
                    "follows. Omit to append at the end."
                ),
            },
            "citations": _CITATIONS_SCHEMA,
        },
        "required": ["action", "event_id", "heading", "body", "citations"],
    },
}


def _term_write_tool(name: str, action: str) -> dict:
    creating = action == "create"
    return {
        "name": name,
        "description": (
            f"{action.title()} one sourced bilingual CommuLingo glossary term. "
            "Term fields only; citations stay top-level."
            + (
                " New lowercase kebab-case term_id."
                if creating
                else " Send only the fields that change; omitted fields keep their stored "
                "value, and a sent field replaces it whole (aliases/people/events lists "
                "are replaced, not merged). Read the term with "
                "commulingo_people(action='get_term') first."
            )
        ),
        "input_schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "term_id": {
                    "type": "string",
                    "description": (
                        "New lowercase kebab-case slug." if creating
                        else "Existing term id — find it with commulingo_people(action='list_terms')."
                    ),
                },
                "fields": _narrow_fields_schema(
                    tuple(k for k in _TERM_NARROW_KEYS if k != "category") if creating else _TERM_UPDATE_NARROW_KEYS,
                    # category is assigned by the runner (commulingo_classify); the model never sees it on create
                    required=("term", "definition", "aliases", "period") if creating else (),
                ),
                "citations": _CITATIONS_SCHEMA,
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["term_id", "fields", "citations"],
        },
    }


COMMULINGO_TERM_CREATE_TOOL = _term_write_tool("commulingo_term_create", "create")
COMMULINGO_TERM_UPDATE_TOOL = _term_write_tool("commulingo_term_update", "update")


async def _exec_commulingo_person_create(person_id: str, fields: dict, citations: list, confidence=None) -> str:
    return await _exec_commulingo_write("person", "create", person_id, citations, fields, confidence)


async def _exec_commulingo_person_update(person_id: str, fields: dict, citations: list, confidence=None) -> str:
    return await _exec_commulingo_write("person", "update", person_id, citations, fields, confidence)


# Section slugs cost an LLM call and the write is validated only afterwards, so
# a create the curator repairs and resubmits would pay again for the same slug.
# Keyed on everything the slug depends on.
_SECTION_SLUG_CACHE: dict[str, str] = {}
_SECTION_SLUG_CACHE_MAX = 256


def _section_slug_key(person_id: str, heading: dict, body: dict, sections: list) -> str:
    existing = sorted(str((section or {}).get("slug") or "") for section in sections or [])
    payload = json.dumps([person_id, heading, body, existing], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cached_section_slug(person_id: str, heading: dict, body: dict, sections: list) -> str:
    from commulingo.section_slug import generate_section_slug

    key = _section_slug_key(person_id, heading, body, sections)
    slug = _SECTION_SLUG_CACHE.get(key)
    if slug is None:
        slug = generate_section_slug(person_id, heading, body, sections)
        if len(_SECTION_SLUG_CACHE) >= _SECTION_SLUG_CACHE_MAX:
            _SECTION_SLUG_CACHE.pop(next(iter(_SECTION_SLUG_CACHE)))
        _SECTION_SLUG_CACHE[key] = slug
    return slug


async def _exec_commulingo_section_save(
    action: str, person_id: str, heading: dict, body: dict,
    citations: list, slug: str | None = None, start_year: int | None = None,
    start_month: int | None = None, expected_revision: str | None = None, evidence: list | None = None,
) -> str:
    from commulingo.section_slug import section_sort_order
    if action == "create":
        try:
            current = await asyncio.to_thread(call_person_service, {"command": "read", "id": person_id})
            slug = await asyncio.to_thread(_cached_section_slug, person_id, heading, body,
                                           (current or {}).get("sections", []))
        except (ValueError, RuntimeError) as exc:
            return _commulingo_error("validation_failed", str(exc))
        sort_order = section_sort_order(start_year, start_month, (current or {}).get("sections", []))
    elif not slug:
        return "Error: existing section slug is required for update; read get_sections first."
    else:
        # Update: an omitted year keeps the section where it is.
        sort_order = section_sort_order(start_year, start_month) if start_year is not None else None
    fields = {"slug": slug, "heading": heading, "body": body, "sources": citations, "expectedRevision": expected_revision, "evidence": evidence or []}
    if sort_order is not None:
        fields["sortOrder"] = sort_order
    return await _exec_commulingo_write("person_section", action, person_id, citations, fields, None)


async def _exec_commulingo_event_link(
    event_id: str, person_id: str, relation_kind: str, relation: dict, note: dict,
    citations: list, sort_order: int | None = None, side: str | None = None,
) -> str:
    fields = {
        "personId": person_id, "relationKind": relation_kind,
        "relation": relation, "note": note,
    }
    if sort_order is not None:
        fields["sortOrder"] = sort_order
    if side is not None:
        fields["side"] = side
    return await _exec_commulingo_write("history_event_person", "create", event_id, citations, fields, None)


async def _exec_commulingo_office_row_save(
    action: str, target_id: str, fields: dict, citations: list, confidence=None,
) -> str:
    return await _exec_commulingo_write(
        "office_row", action, target_id, citations, fields, confidence,
    )


async def _exec_commulingo_term_create(term_id: str, fields: dict, citations: list, confidence=None) -> str:
    fields = dict(fields or {})
    # setdefault alone left the card unsourced whenever the curator sent an
    # explicit empty sources list alongside real top-level citations.
    if not fields.get("sources"):
        fields["sources"] = citations
    return await _exec_commulingo_write("term", "create", term_id, citations, fields, confidence)


async def _exec_commulingo_term_update(term_id: str, fields: dict, citations: list, confidence=None) -> str:
    # No sources default here: an omitted 'sources' must preserve the card's
    # existing list, not overwrite it with this one edit's citations.
    return await _exec_commulingo_write("term", "update", term_id, citations, fields, confidence)


async def _exec_commulingo_event_update(
    event_id: str, fields: dict, citations: list, confidence=None,
) -> str:
    return await _exec_commulingo_write(
        "history_event", "update", event_id, citations, fields, confidence,
    )


async def _exec_commulingo_event_section_save(
    action: str, event_id: str, heading: dict, body: dict, citations: list,
    after: dict | None = None,
) -> str:
    fields = {"heading": heading, "body": body}
    if after:
        fields["after"] = after
    return await _exec_commulingo_write(
        "history_event_section", action, event_id, citations, fields, None,
    )


COMMULINGO_GAP_REPORT_TOOL = {
    "name": "commulingo_gap_report",
    "description": (
        "File what the event text you just wrote needs and the site lacks: a person with "
        "no card, a term with no glossary entry, a document worth publishing in full. The "
        "people and glossary lanes work from this queue. File only what the narrative "
        "leans on; a name mentioned once in passing is not a gap. Check the reading tools "
        "first: an existing card is a gap only when too thin for its weight in the event "
        "(send its target_id). 'person' is for historical actors of the events; a scholar "
        "or journalist only when the section's historiography treats them as a subject, "
        "never because you cited their book. 'doc' has a higher bar: only primary sources "
        "a reader should read whole (decree, treaty, speech, resolution, letter, short "
        "minutes). Never the scholarship you researched from (monographs, articles, "
        "memoirs, press; those belong in your sources), never book-length records (trial "
        "transcripts, plenum stenograms, commission reports), never a document the library "
        "already carries."
    ),
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "gaps": {
                "type": "array", "minItems": 1, "maxItems": 8,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "enum": ["person", "term", "doc"]},
                        "label": {
                            **_BILINGUAL_TEXT_SCHEMA,
                            "description": "What the event text calls it, in both languages.",
                        },
                        "target_id": {
                            "type": ["string", "null"],
                            "description": (
                                "Existing entry id when the card is too thin; null when "
                                "nothing exists."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "What the event narrative needs from it; the next lane writes "
                                "from this brief, so name the specific role, not 'important "
                                "figure'."
                            ),
                        },
                        "priority": {
                            "type": "integer", "minimum": 0, "maximum": 10,
                            "description": (
                                "Load-bearing in the section just written: 8-10 unreadable "
                                "without it, 4-7 deepens it, 0-3 adjacent."
                            ),
                        },
                    },
                    "required": ["kind", "label", "target_id", "reason", "priority"],
                },
            },
            "event_id": {"type": "string", "description": "The event whose text needed them."},
        },
        "required": ["gaps", "event_id"],
    },
}


def _file_gaps(gaps: list, event_id: str) -> dict:
    """File through CommuLingo (admin MCP gap_file), which skips gaps its
    dictionaries already cover (names, aliases, bracket variants, event titles
    for terms, document titles) and re-files as no-ops."""
    from commulingo.mcp_client import call_tool
    return call_tool("gap_file", {"gaps": gaps, "eventId": event_id, "changedBy": _SUGGESTED_BY})


async def _exec_commulingo_gap_report(gaps: list, event_id: str) -> str:
    if not isinstance(gaps, list) or not gaps:
        return _commulingo_error("missing_gaps", "gaps must be a non-empty list", retryable=False)
    for index, gap in enumerate(gaps):
        if not isinstance(gap, dict):
            return _commulingo_error("invalid_gap", f"gaps[{index}] must be an object")
        label = gap.get("label")
        if not isinstance(label, dict) or not (label.get("ko") or "").strip():
            return _commulingo_error(
                "invalid_gap", f"gaps[{index}].label needs at least a Korean name"
            )
        if not (gap.get("reason") or "").strip():
            return _commulingo_error(
                "invalid_gap",
                f"gaps[{index}].reason is required — it is the brief the next lane writes from",
            )
    try:
        result = await asyncio.to_thread(_file_gaps, gaps, (event_id or "").strip())
    except Exception as e:
        logger.warning("commulingo gap report error: %s", e)
        return _commulingo_error("internal_error", f"{type(e).__name__}: {e}", retryable=True)
    parts = [f"OK — queued {len(result['filed'])} gap(s): {', '.join(result['filed']) or 'none'}."]
    if result["already_queued"]:
        parts.append(f"Already queued, not duplicated: {', '.join(result['already_queued'])}.")
    if result["already_covered"]:
        parts.append(
            "The site already covers these, so they were not queued: "
            f"{', '.join(result['already_covered'])}. Link to them in your section text "
            "instead of describing them from scratch."
        )
    return " ".join(parts)


COMMULINGO_TOOLS = [
    COMMULINGO_PEOPLE_TOOL,
    COMMULINGO_PERSON_CREATE_TOOL,
    COMMULINGO_PERSON_UPDATE_TOOL,
    COMMULINGO_SECTION_SAVE_TOOL,
    COMMULINGO_EVENT_LINK_TOOL,
    COMMULINGO_OFFICE_ROW_SAVE_TOOL,
    COMMULINGO_TERM_CREATE_TOOL,
    COMMULINGO_TERM_UPDATE_TOOL,
    COMMULINGO_EVENT_UPDATE_TOOL,
    COMMULINGO_EVENT_SECTION_SAVE_TOOL,
    COMMULINGO_GAP_REPORT_TOOL,
]
COMMULINGO_TOOL_HANDLERS = {
    "commulingo_people": _exec_commulingo_people,
    "commulingo_person_create": _exec_commulingo_person_create,
    "commulingo_person_update": _exec_commulingo_person_update,
    "commulingo_section_save": _exec_commulingo_section_save,
    "commulingo_event_link": _exec_commulingo_event_link,
    "commulingo_office_row_save": _exec_commulingo_office_row_save,
    "commulingo_term_create": _exec_commulingo_term_create,
    "commulingo_term_update": _exec_commulingo_term_update,
    "commulingo_event_update": _exec_commulingo_event_update,
    "commulingo_event_section_save": _exec_commulingo_event_section_save,
    "commulingo_gap_report": _exec_commulingo_gap_report,
}
