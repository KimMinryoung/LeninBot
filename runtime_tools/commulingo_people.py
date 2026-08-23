"""runtime_tools.commulingo_people — CommuLingo dictionary read + narrow write tools.

The people dictionary at cyber-lenin.com/commulingo/people is DB-backed
(commulingo_* tables in the main Postgres; see
frontend/dev_docs/commulingo_people_handoff.md).

The public tool surface contains one read/search tool and target-specific narrow
write tools for people, sections, event links, and terms. Every write requires
citations and shares the same validation and transaction core.

Writes run in one of two modes, switched by
config/commulingo_people.json → {"direct_apply": true|false} (mtime-cached,
no restart needed):

- direct_apply=true  — the edit is validated and applied to the DB
  immediately, inside one transaction, with a revision snapshot in
  commulingo_people_revisions (same semantics as the frontend admin store)
  plus an auto-approved row in commulingo_agent_suggestions so provenance
  (sources, confidence) is always on record.
- direct_apply=false — the edit is staged as a pending row in
  commulingo_agent_suggestions; the operator reviews and applies it with
  scripts/commulingo_suggestions.py (which reuses apply_edit below).

The frontend keeps a 30s in-process cache and the page/API a ~30s CDN
max-age, so applied edits appear on the public site within about a minute —
no service restart or cache purge required.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import date, datetime
from decimal import Decimal

from psycopg2.extras import RealDictCursor, execute_values

from db import query as db_query, query_one as db_query_one, get_conn
from tool_gateway.results import ToolFailure

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,120}$")

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

_HISTORY_RELATION_KINDS = ("leader", "participant", "executor", "target", "opponent", "witness")
_ACTIONS = ("create", "update", "delete")

_FATE_KINDS = (
    "executed", "assassinated", "murdered", "killed", "suicide",
    "deposed", "exile", "natural",
)

_PERSON_PATCH_KEYS = frozenset({
    "id", "group", "groupId", "sortOrder", "cyrillic", "years",
    "name", "givenName", "familyName",
    "epithet", "bio", "moment", "fate", "patronymic", "cyrillicPatronymic",
    "aliases", "scenes", "career", "role", "citizenship", "origin", "nationalOrigin",
    "office_rows", "sections",  # read-only echoes from get_person; tolerated and ignored
})

# Flag codes the frontend has vendored SVGs for (data/commulingo/flag-icons.js).
# Must stay in sync with NATIONALITY_CODES in scripts/commulingo_people_maintainer.py.
# This vocabulary is states, and it renders as a flag on the card. Stateless
# ethnicities are therefore absent BY DECISION, not by oversight — do not "fix"
# them in, and do not file their bearers under a neighbouring state:
#
#   Jewish  — the Soviet passport did carry еврей on the fifth line, but tagging
#             Bolsheviks with it here does not reconstruct that category, it
#             supplies the 'Jewish Bolshevik' conspiracy its raw material. There
#             is also no flag that is not an anachronism for a person who died
#             before 1948.
#   Tatar, Bashkir, Chechen, Buryat, Ossetian and the other peoples of the
#             RSFSR — no separate state, so no flag.
#
# A card whose prose names one of these keeps origin_code 'russia'; the prose is
# where that identity is carried. An audit that reports these as a gap has
# rediscovered the policy, not a bug.
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
    "yugoslavia",
    "france", "italy", "spain", "uk", "netherlands", "belgium", "usa", "cuba", "argentina",
    "chile", "china", "japan", "india", "turkey", "greece", "vietnam", "north-korea",
    "south-korea", "albania", "angola", "burkina-faso", "congo", "ghana",
    "guinea-bissau", "indonesia", "mozambique", "peru", "trinidad", "portugal",
    "brazil", "el-salvador", "grenada", "guyana", "nicaragua", "south-africa",
    "tanzania", "ireland", "slovakia", "czechoslovakia", "korea", "martinique",
    "israel",
})

# Which writing system each nationality's own names use. Port of
# frontend data/commulingo/native-script.js (NATION_SCRIPTS) — keep in sync.
# `cyrillic` is the legacy column name for the native-script name line; filling
# it with a Russian transliteration for a non-Russian filed 박헌영 as
# "Пак Хон Ён" and Kádár János as "Янош Кадар" until frontend migration 057.
_SCRIPT_RANGES = (
    ("cyrillic", re.compile(r"[Ѐ-ӿԀ-ԯ]")),
    ("greek", re.compile(r"[Ͱ-Ͽ]")),
    ("hangul", re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")),
    ("kana", re.compile(r"[぀-ヿ]")),
    ("han", re.compile(r"[㐀-䶿一-鿿]")),
    ("georgian", re.compile(r"[Ⴀ-ჿ]")),
    ("armenian", re.compile(r"[԰-֏]")),
    ("hebrew", re.compile(r"[֐-׿]")),
    ("arabic", re.compile(r"[؀-ۿ]")),
    ("devanagari", re.compile(r"[ऀ-ॿ]")),
    ("bengali", re.compile(r"[ঀ-৿]")),
    ("latin", re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]")),
)

_ROMAN_NUMERAL_RE = re.compile(r"(?:^|\s)[IVXLCDM]+(?=$|\s)")

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

_CYRILLIC_NATIONS = (
    "soviet", "russia", "ukraine", "belarus", "bulgaria",
    "kazakhstan", "kyrgyzstan", "tajikistan",
)
_LATIN_NATIONS = (
    "latvia", "lithuania", "estonia", "poland", "finland", "germany",
    "east-germany", "austria", "hungary", "czechia", "romania", "albania",
    "france", "italy", "spain", "portugal", "netherlands", "belgium", "uk", "usa",
    "turkey", "cuba", "argentina", "chile", "peru", "angola", "burkina-faso",
    "congo", "ghana", "guinea-bissau", "mozambique", "trinidad", "indonesia",
    "vietnam", "brazil", "el-salvador", "grenada", "guyana", "nicaragua",
    "south-africa", "tanzania", "ireland", "slovakia", "czechoslovakia", "martinique",
)
_NATION_SCRIPTS: dict[str, tuple[str, ...]] = {
    **{code: ("cyrillic",) for code in _CYRILLIC_NATIONS},
    **{code: ("latin",) for code in _LATIN_NATIONS},
    # Republics that changed alphabet: both the Soviet-era and modern form pass.
    "moldova": ("cyrillic", "latin"),
    "yugoslavia": ("latin", "cyrillic"),
    "uzbekistan": ("latin", "cyrillic"),
    "turkmenistan": ("latin", "cyrillic"),
    "azerbaijan": ("latin", "cyrillic"),
    "georgia": ("georgian",),
    "armenia": ("armenian",),
    "greece": ("greek",),
    "china": ("han",),
    "japan": ("kana", "han"),
    "north-korea": ("hangul", "han"),
    "south-korea": ("hangul", "han"),
    "india": ("devanagari", "bengali", "latin"),
    "korea": ("hangul", "han"),
    "israel": ("hebrew",),
}

# Nations whose people write the family name first, and the joiner between
# family and given per language. Korean text fuses Korean/Chinese/Vietnamese
# names (김무정, 펑더화이, 호찌민) and keeps the space for Japanese
# (도쿠다 규이치); English follows each nation's own romanization — family
# first for Korean/Chinese/Vietnamese (Kim Mu-chong, Peng Dehuai, Le Duan),
# given first for Japanese (Sen Katayama). Port of frontend
# data/commulingo/native-script.js (FAMILY_FIRST) — keep the two in sync.
_FAMILY_FIRST: dict[str, dict[str, str | None]] = {
    "korea": {"ko": "", "en": " "},
    "north-korea": {"ko": "", "en": " "},
    "south-korea": {"ko": "", "en": " "},
    "china": {"ko": "", "en": " "},
    "vietnam": {"ko": "", "en": " "},
    "japan": {"ko": " ", "en": None},
}


def _family_first_joiner(lang: str, codes) -> str | None:
    """The joiner between family and given when one of these citizenship codes
    puts the family name first in `lang`; None means Western given-first."""
    for code in codes or ():
        rule = _FAMILY_FIRST.get((code or "").strip())
        if rule is not None:
            return rule.get(lang)
    return None

# person patch fields that must be {ko, en} objects. Plain strings are
# rejected outright: _localized() would store them as Korean-only and
# silently blank the English side (this happened in production).
_LOCALIZED_PERSON_KEYS = ("name", "givenName", "familyName", "epithet", "bio", "moment", "patronymic")
_LOCALIZED_OFFICE_ROW_KEYS = ("body", "name", "note")

_OFFICE_ROW_PATCH_KEYS = frozenset({
    "sortOrder", "years", "period", "body", "personId", "name", "note",
})

# Long-form detail sections rendered on /commulingo/people/<id>.
_SECTION_PATCH_KEYS = frozenset({"slug", "heading", "body", "sortOrder", "sources"})
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,60}$")

_HISTORY_EVENT_PERSON_PATCH_KEYS = frozenset({
    "personId", "sortOrder", "relationKind", "relation", "note",
})

# The event card's short fields (frontend /commulingo/events/<id>). Title and
# period are deliberately absent: they are the id's public identity and the
# chronological sort key, and an unattended run has no business moving either.
_HISTORY_EVENT_PATCH_KEYS = frozenset({
    "question", "summary", "outcome", "timeline", "sources",
})
_LOCALIZED_EVENT_KEYS = ("question", "summary", "outcome")

# One `## ` section of an event's long-form body. The body is a single markdown
# column rather than a table of rows, so a section write is a splice into that
# markdown — see _splice_event_section. Modelled on person sections for the same
# reason they exist: a narrative this long is written one part per run, and a
# run that had to restate the whole body would spend most of its output
# reproducing text it is not changing (and would regress it whenever it drifted).
_HISTORY_EVENT_SECTION_PATCH_KEYS = frozenset({"heading", "body", "after"})

# Glossary terms (frontend /commulingo/terms, tables from frontend migration
# 061). `aliases` are the exact strings prose uses and feed the auto-linking
# pipeline; `people`/`events` are lists of related ids.
#
# `period` is bilingual and `category`/`startYear`/`endYear` exist because of
# frontend migration 071: the single free-text period_label leaked Korean onto
# the English page and back, and nothing could sort chronologically. A term
# written without a category shows up on the glossary as 'Uncategorized', so
# both are required on create rather than backfilled later by hand.
_TERM_PATCH_KEYS = frozenset({
    "id", "sortOrder", "term", "original", "period", "startYear", "endYear",
    "category", "definition", "body", "aliases", "people", "events", "sources",
    "parentId",
})
_LOCALIZED_TERM_KEYS = ("term", "definition", "body", "period")

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
    ("theory", "Ideology and theory"), ("economy", "Economy and planning"),
    ("party-state", "Party and state"), ("factions", "Factions and line struggles"),
    ("repression", "Repression and law"), ("nationalities", "Nationalities"),
    ("culture", "Culture and education"), ("international", "International movement"),
    ("korea", "Korean political economy"), ("contemporary", "Contemporary capitalism"),
)


def _load_term_categories() -> tuple[tuple[str, ...], str]:
    """(slugs, hint line) for the schema enum and the rejection messages."""
    rows = []
    try:
        rows = db_query(
            """SELECT id, label_en FROM commulingo_term_categories
               ORDER BY sort_order, id"""
        ) or []
    except Exception as exc:  # missing table, no DB, no credentials — all fall back
        logger.warning("term categories unavailable (%s); using the built-in list", exc)
    pairs = [(r["id"], r["label_en"]) for r in rows] or list(_TERM_CATEGORY_FALLBACK)
    hint = ", ".join(
        slug if slug.replace("-", " ") == label.lower() else f"{slug} ({label.lower()})"
        for slug, label in pairs
    )
    return tuple(slug for slug, _ in pairs), hint


_TERM_CATEGORIES, _TERM_CATEGORY_HINT = _load_term_categories()
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

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
_QUOTED_SPAN_RE = re.compile(r'"[^"]*"|“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|《[^》]*》')


def _name_normalization() -> dict:
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


def _detect_scripts(text: str) -> list[str]:
    """Writing systems present in `text`; regnal numbers (Николай II) ignored."""
    value = _ROMAN_NUMERAL_RE.sub(" ", str(text or ""))
    return [name for name, pattern in _SCRIPT_RANGES if pattern.search(value)]


def _check_native_script(text: str, codes: list[tuple[str, str]], field: str) -> str | None:
    """Error string when a native-name line is written in the wrong script.

    Both citizenship and origin count: Soviet republic officials are filed as
    citizenship 'soviet' + origin 'latvia'/'georgia'/…, and a Latvian in the USSR
    legitimately writes 'Mārtiņš Lācis' in Latin. The allowed set is the union.
    `codes` pairs each code with the public field it came from, so the error can
    say WHICH field carries the wrong code — a bare 'soviet + russia' left the
    model retrying the name instead of fixing nationalOrigin (gap 1617, Sillari).
    """
    value = str(text or "").strip()
    codes = [(label, code.strip()) for label, code in codes if code and code.strip()]
    allowed: list[str] = []
    for _label, code in codes:
        for script in _NATION_SCRIPTS.get(code, ()):
            if script not in allowed:
                allowed.append(script)
    if not value or not allowed:
        return None
    wrong = [s for s in _detect_scripts(value) if s not in allowed]
    if not wrong:
        return None
    named = " + ".join(f"{label} '{code}'" for label, code in codes)
    return (
        f"Error: {field} '{value}' is written in {'/'.join(wrong)}, but {named} "
        f"writes its own names in {' or '.join(allowed)}. {field} is the "
        "person's name in THEIR OWN script, never a Russian transliteration of it "
        "(박헌영, not 'Пак Хон Ён'; 'Kádár János', not 'Янош Кадар'; 毛泽东, not "
        "'Мао Цзэдун'). Either write the name in the right script, or fix the "
        "nationality field that is wrong — usually nationalOrigin: a Soviet "
        "official of a non-Russian nationality keeps their OWN nation there, "
        "which admits its script (soviet + estonia admits Latin, soviet + "
        "georgia admits Georgian). Never file a Soviet citizen as "
        "nationalOrigin 'russia' just because they served in the USSR."
    )


def find_spelling_variants_in_text(text: str, lang: str) -> dict[str, str]:
    """variant -> canonical spellings used in `text` outside quotation marks.

    Shared with the research-document writer (runtime_tools/research.py) so
    reports spell dictionary people and glossary terms the way their cards do.
    """
    norm = _name_normalization()
    table = norm.get(lang) or {}
    hits: dict[str, str] = {}
    if not table or not text:
        return hits
    scannable = _QUOTED_SPAN_RE.sub(" ", str(text))
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
    norm = _name_normalization()
    table = norm.get(lang) or {}
    if not table or not text:
        return str(text or ""), {}
    placeholders: dict[str, str] = {}

    def _stash(match):
        key = f"\x00Q{len(placeholders)}\x00"
        placeholders[key] = match.group(0)
        return key

    masked = _QUOTED_SPAN_RE.sub(_stash, str(text))
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
    return db_query(
        """SELECT g.id, g.range_label, g.title_ko, g.title_en,
                  g.blurb_ko, g.blurb_en,
                  COUNT(p.id) AS people_count
           FROM commulingo_people_groups g
           LEFT JOIN commulingo_people p ON p.group_id = g.id
           GROUP BY g.id, g.sort_order, g.range_label, g.title_ko, g.title_en,
                    g.blurb_ko, g.blurb_en
           ORDER BY g.sort_order, g.id"""
    )


def _search_people(q: str, group_id: str, limit: int) -> list[dict]:
    return db_query(
        """SELECT id, group_id, name_ko, name_en, cyrillic, years_label,
                  epithet_ko, fate_kind
           FROM commulingo_people
           WHERE (%(q)s = ''
                  OR id ILIKE '%%' || %(q)s || '%%'
                  OR name_ko ILIKE '%%' || %(q)s || '%%'
                  OR name_en ILIKE '%%' || %(q)s || '%%'
                  OR cyrillic ILIKE '%%' || %(q)s || '%%')
             AND (%(g)s = '' OR group_id = %(g)s)
           ORDER BY sort_order, id
           LIMIT %(limit)s""",
        {"q": q, "g": group_id, "limit": limit},
    )


def _person_snapshot(cur, person_id: str) -> dict | None:
    """Full person record via an existing cursor (transaction-consistent).

    Returned in the canonical person-field shape accepted by the narrow writers.
    """
    cur.execute(
        """SELECT id, group_id, cyrillic, years_label,
                  name_ko, name_en, given_name_ko, given_name_en, family_name_ko, family_name_en,
                  epithet_ko, epithet_en, bio_ko, bio_en,
                  moment_ko, moment_en,
                  fate_kind, fate_label_ko, fate_label_en,
                  citizenship_code, citizenship_label_ko, citizenship_label_en,
                  origin_code, origin_label_ko, origin_label_en
           FROM commulingo_people WHERE id = %s""",
        (person_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    person = {
        "id": row["id"],
        "group": row["group_id"],
        "cyrillic": row["cyrillic"],
        "years": row["years_label"],
        "name": {"ko": row["name_ko"], "en": row["name_en"]},
        "givenName": {"ko": row["given_name_ko"], "en": row["given_name_en"]},
        "familyName": {"ko": row["family_name_ko"], "en": row["family_name_en"]},
        "epithet": {"ko": row["epithet_ko"], "en": row["epithet_en"]},
        "bio": {"ko": row["bio_ko"], "en": row["bio_en"]},
        "moment": {"ko": row["moment_ko"], "en": row["moment_en"]},
        "fate": {
            "kind": row["fate_kind"],
            "label": {"ko": row["fate_label_ko"], "en": row["fate_label_en"]},
        },
        "citizenship": {
            "code": row["citizenship_code"],
            "label": {"ko": row["citizenship_label_ko"], "en": row["citizenship_label_en"]},
        },
        "origin": {
            "code": row["origin_code"],
            "label": {"ko": row["origin_label_ko"], "en": row["origin_label_en"]},
        },
    }
    person["nationalOrigin"] = person["origin"]
    cur.execute(
        """SELECT patronymic_ko, patronymic_en, cyrillic_patronymic
           FROM commulingo_person_patronymics WHERE person_id = %s""",
        (person_id,),
    )
    row = cur.fetchone()
    person["patronymic"] = {"ko": row["patronymic_ko"], "en": row["patronymic_en"]} if row else None
    person["cyrillicPatronymic"] = row["cyrillic_patronymic"] if row else ""
    cur.execute(
        """SELECT lang, alias FROM commulingo_person_aliases
           WHERE person_id = %s ORDER BY lang, sort_order, alias""",
        (person_id,),
    )
    person["aliases"] = {"ko": [], "en": []}
    for r in cur.fetchall():
        person["aliases"][r["lang"]].append(r["alias"])
    cur.execute(
        """SELECT collection_id, episode_id FROM commulingo_person_scenes
           WHERE person_id = %s ORDER BY sort_order""",
        (person_id,),
    )
    person["scenes"] = [[r["collection_id"], r["episode_id"]] for r in cur.fetchall()]
    cur.execute(
        """SELECT period_label, role_ko, role_en
           FROM commulingo_person_career_entries
           WHERE person_id = %s ORDER BY sort_order, id""",
        (person_id,),
    )
    person["career"] = [
        {"y": r["period_label"], "r": {"ko": r["role_ko"], "en": r["role_en"]}}
        for r in cur.fetchall()
    ]
    cur.execute(
        """SELECT r.office_id, r.category_id,
                  COALESCE(NULLIF(c.icon, ''), NULLIF(r.icon, ''), o.icon, '') AS resolved_icon,
                  COALESCE(NULLIF(c.label_ko, ''), NULLIF(r.label_ko, ''), o.title_ko, '') AS label_ko,
                  COALESCE(NULLIF(c.label_en, ''), NULLIF(r.label_en, ''), o.title_en, '') AS label_en
           FROM commulingo_person_roles r
           LEFT JOIN commulingo_offices o ON o.id = r.office_id
           LEFT JOIN commulingo_role_categories c ON c.id = r.category_id
           WHERE r.person_id = %s""",
        (person_id,),
    )
    row = cur.fetchone()
    person["role"] = {
        "officeId": row["office_id"] or "",
        "category": row["category_id"] or "",
        "label": {"ko": row["label_ko"], "en": row["label_en"]},
        "resolvedIcon": row["resolved_icon"],
    } if row else None
    cur.execute(
        """SELECT slug, sort_order, heading_ko, heading_en,
                  length(body_ko) AS body_ko_chars, length(body_en) AS body_en_chars
           FROM commulingo_person_sections
           WHERE person_id = %s ORDER BY sort_order, id""",
        (person_id,),
    )
    person["sections"] = [
        {
            "slug": r["slug"],
            "sortOrder": r["sort_order"],
            "heading": {"ko": r["heading_ko"], "en": r["heading_en"]},
            "bodyChars": {"ko": r["body_ko_chars"], "en": r["body_en_chars"]},
        }
        for r in cur.fetchall()
    ]
    return person


def _office_snapshot(cur, office_id: str) -> dict | None:
    cur.execute(
        """SELECT id, range_label, title_ko, title_en, blurb_ko, blurb_en
           FROM commulingo_offices WHERE id = %s""",
        (office_id,),
    )
    office = cur.fetchone()
    if not office:
        return None
    office = dict(office)
    cur.execute(
        """SELECT id AS row_id, period_label, body_ko, body_en, person_id,
                  name_ko, name_en, note_ko, note_en
           FROM commulingo_office_rows
           WHERE office_id = %s ORDER BY sort_order, id""",
        (office_id,),
    )
    office["rows"] = [dict(r) for r in cur.fetchall()]
    return office


def _get_person(person_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            person = _person_snapshot(cur, person_id)
            if person is None:
                return None
            cur.execute(
                """SELECT r.id AS row_id, r.office_id, o.title_ko AS office_title_ko,
                          r.period_label, r.body_ko, r.note_ko
                   FROM commulingo_office_rows r
                   JOIN commulingo_offices o ON o.id = r.office_id
                   WHERE r.person_id = %s
                   ORDER BY o.sort_order, r.sort_order, r.id""",
                (person_id,),
            )
            person["office_rows"] = [dict(r) for r in cur.fetchall()]
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
    like = f"%{q}%"
    events = db_query(
        """SELECT id, period_label, title_ko, title_en
             FROM commulingo_history_events
            WHERE id ILIKE %(like)s OR title_ko ILIKE %(like)s OR title_en ILIKE %(like)s
            ORDER BY sort_order, id LIMIT %(limit)s""",
        {"like": like, "limit": limit},
    )
    offices = db_query(
        """SELECT id, range_label, title_ko, title_en
             FROM commulingo_offices
            WHERE id ILIKE %(like)s OR title_ko ILIKE %(like)s OR title_en ILIKE %(like)s
            ORDER BY sort_order, id LIMIT %(limit)s""",
        {"like": like, "limit": limit},
    )
    result = {
        "people": _search_people(q, "", limit),
        "terms": _list_terms(q)[:limit],
        "events": [dict(r) for r in events],
        "offices": [dict(r) for r in offices],
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
    return db_query(
        """SELECT o.id, o.range_label, o.title_ko, o.title_en,
                  COUNT(r.id) AS row_count
           FROM commulingo_offices o
           LEFT JOIN commulingo_office_rows r ON r.office_id = o.id
           GROUP BY o.id, o.sort_order, o.range_label, o.title_ko, o.title_en
           ORDER BY o.sort_order, o.id"""
    )


def _get_office(office_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            return _office_snapshot(cur, office_id)


def _list_categories() -> list[dict]:
    return db_query(
        """SELECT c.id, c.icon, c.label_ko, c.label_en,
                  COUNT(r.person_id) AS people_count
           FROM commulingo_role_categories c
           LEFT JOIN commulingo_person_roles r ON r.category_id = c.id
           GROUP BY c.id, c.sort_order, c.icon, c.label_ko, c.label_en
           ORDER BY c.sort_order, c.id"""
    )


def _get_sections(person_id: str) -> list[dict] | None:
    """Sections in the exact person_section patch shape (read → edit → write back)."""
    if not db_query_one("SELECT 1 FROM commulingo_people WHERE id = %s", (person_id,)):
        return None
    rows = db_query(
        """SELECT slug, sort_order, heading_ko, heading_en, body_ko, body_en, sources
           FROM commulingo_person_sections
           WHERE person_id = %s ORDER BY sort_order, id""",
        (person_id,),
    )
    return [
        {
            "slug": r["slug"],
            "sortOrder": r["sort_order"],
            "heading": {"ko": r["heading_ko"], "en": r["heading_en"]},
            "body": {"ko": r["body_ko"], "en": r["body_en"]},
            "sources": r["sources"],
        }
        for r in rows
    ]


def _list_events() -> list[dict]:
    return db_query(
        """SELECT e.id, e.period_label, e.title_ko, e.title_en,
                  COUNT(ep.person_id)::int AS people_count,
                  length(e.body_ko) AS body_ko_chars,
                  jsonb_array_length(e.timeline) AS timeline_entries
             FROM commulingo_history_events e
             LEFT JOIN commulingo_history_event_people ep ON ep.event_id = e.id
            GROUP BY e.id, e.sort_order, e.period_label, e.title_ko, e.title_en,
                     e.body_ko, e.timeline
            ORDER BY e.sort_order, e.id"""
    )


def _get_event(event_id: str) -> dict | None:
    event = db_query_one(
        """SELECT id, period_label, title_ko, title_en, question_ko, question_en,
                  summary_ko, summary_en, outcome_ko, outcome_en,
                  body_ko, body_en, timeline, sources
             FROM commulingo_history_events WHERE id = %s""",
        (event_id,),
    )
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
            "relation": {"ko": row["relation_ko"], "en": row["relation_en"]},
            "note": {"ko": row["note_ko"], "en": row["note_en"]},
        }
        for row in db_query(
            """SELECT person_id, sort_order, relation_kind,
                      relation_ko, relation_en, note_ko, note_en
                 FROM commulingo_history_event_people
                WHERE event_id = %s ORDER BY sort_order, person_id""",
            (event_id,),
        )
    ]
    return event


def _term_snapshot(cur, term_id: str) -> dict | None:
    """Full glossary term via an existing cursor, in the term patch shape."""
    cur.execute(
        """SELECT id, term_ko, term_en, original, period_label,
                  period_ko, period_en, start_year, end_year, category,
                  definition_ko, definition_en, body_ko, body_en, sources,
                  parent_id
           FROM commulingo_terms WHERE id = %s""",
        (term_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    term = {
        "id": row["id"],
        "term": {"ko": row["term_ko"], "en": row["term_en"]},
        "original": row["original"],
        "period": {
            "ko": row["period_ko"] or row["period_label"],
            "en": row["period_en"] or row["period_label"],
        },
        "startYear": row["start_year"],
        "endYear": row["end_year"],
        "category": row["category"],
        "definition": {"ko": row["definition_ko"], "en": row["definition_en"]},
        "body": {"ko": row["body_ko"], "en": row["body_en"]},
        "sources": row["sources"] if isinstance(row["sources"], list) else [],
        "parentId": row["parent_id"],
    }
    cur.execute(
        """SELECT lang, alias FROM commulingo_term_aliases
           WHERE term_id = %s ORDER BY lang, sort_order, alias""",
        (term_id,),
    )
    term["aliases"] = {"ko": [], "en": []}
    for alias_row in cur.fetchall():
        if alias_row["lang"] in term["aliases"]:
            term["aliases"][alias_row["lang"]].append(alias_row["alias"])
    cur.execute(
        "SELECT person_id FROM commulingo_term_people WHERE term_id = %s ORDER BY sort_order, person_id",
        (term_id,),
    )
    term["people"] = [r["person_id"] for r in cur.fetchall()]
    cur.execute(
        "SELECT event_id FROM commulingo_term_events WHERE term_id = %s ORDER BY sort_order, event_id",
        (term_id,),
    )
    term["events"] = [r["event_id"] for r in cur.fetchall()]
    return term


def _list_terms(q: str = "") -> list[dict]:
    # Unfiltered this is ~150 terms carrying ~1300 aliases. A curator checking
    # whether one candidate is already registered used to pull that whole blob,
    # hit the tool-result cap, and pull it again — three rounds spent to not get
    # an answer. 'q' answers the same question in one small result.
    rows = db_query(
        """SELECT t.id, t.term_ko, t.term_en, t.original,
                  COALESCE(a.aliases, ARRAY[]::text[]) AS aliases
             FROM commulingo_terms t
             LEFT JOIN (
                 SELECT term_id, array_agg(alias ORDER BY lang, sort_order) AS aliases
                   FROM commulingo_term_aliases GROUP BY term_id
             ) a ON a.term_id = t.id
            WHERE %(q)s = ''
               OR t.id ILIKE %(like)s
               OR t.term_ko ILIKE %(like)s
               OR t.term_en ILIKE %(like)s
               OR t.original ILIKE %(like)s
               OR EXISTS (SELECT 1 FROM commulingo_term_aliases x
                           WHERE x.term_id = t.id AND x.alias ILIKE %(like)s)
            ORDER BY t.sort_order, t.id""",
        {"q": q, "like": f"%{q}%"},
    )
    return [
        {
            "id": row["id"],
            "term": {"ko": row["term_ko"], "en": row["term_en"]},
            "original": row["original"],
            "aliases": list(row["aliases"] or []),
        }
        for row in rows
    ]


def _get_term(term_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            return _term_snapshot(cur, term_id)


def _list_suggestions(status: str, limit: int) -> list[dict]:
    return db_query(
        """SELECT id, target_type, target_id, action, status, confidence,
                  reviewer, review_note, created_at, reviewed_at
           FROM commulingo_agent_suggestions
           WHERE (%(s)s = '' OR status = %(s)s)
           ORDER BY created_at DESC
           LIMIT %(limit)s""",
        {"s": status, "limit": limit},
    )


COMMULINGO_PEOPLE_TOOL = {
    "name": "commulingo_people",
    "description": (
        "Read the CommuLingo people dictionary (cyber-lenin.com/commulingo/people): "
        "Soviet-history figures with bios, career timelines, and institution "
        "(office) leadership timelines, all bilingual ko/en. Actions: "
        "`search` (q matched across people, glossary terms, historical events, "
        "and offices at once — a single match returns its full record inline; "
        "use this when the category is uncertain), "
        "`list_groups` (era groups + people counts), "
        "`search_people` (q matches id/name/cyrillic; optional group_id), "
        "`get_person` (full record — returned in the canonical person-field shape "
        "accepted by the narrow person writers; "
        "office_rows, sections and role.resolvedIcon are read-only info), "
        "`list_offices` (institution timelines + row counts), "
        "`get_office` (one institution's full leadership timeline), "
        "`list_categories` (office-less role categories for role {category}), "
        "`get_sections` (a person's full detail-page sections, returned in the "
        "exact person_section patch shape — edit and send back), "
        "`list_events` (historical event ids, titles, and linked-person counts), "
        "`get_event` (one event and all current person relationships), "
        "`list_terms` (glossary term ids, names, and every registered alias — "
        "check this before registering a term; pass `q` to match one candidate "
        "instead of listing all of them), "
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
                    "search", "list_groups", "search_people", "get_person",
                    "list_offices", "get_office", "list_categories",
                    "get_sections", "list_events", "get_event",
                    "list_terms", "get_term", "list_suggestions",
                ],
            },
            "q": {
                "type": "string",
                "description": (
                    "search: substring matched across people, glossary terms, "
                    "historical events, and offices at once. "
                    "search_people: substring matched against id/name/cyrillic. "
                    "list_terms: substring matched against term id/ko/en/original/alias — "
                    "use it to check one candidate instead of pulling the whole glossary."
                ),
            },
            "group_id": {
                "type": "string",
                "description": "search_people: restrict to one group id.",
            },
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
        elif action == "list_groups":
            result = await asyncio.to_thread(_list_groups)
        elif action == "search_people":
            result = await asyncio.to_thread(_search_people, (q or "").strip(), (group_id or "").strip(), limit)
        elif action == "get_person":
            if not person_id:
                return "Error: person_id is required for get_person."
            result = await asyncio.to_thread(_get_person, person_id.strip())
            if result is None:
                return f"Error: person '{person_id}' not found. Use search_people to find the id."
        elif action == "list_offices":
            result = await asyncio.to_thread(_list_offices)
        elif action == "list_categories":
            result = await asyncio.to_thread(_list_categories)
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

def _localized(value, lang: str) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value if lang == "ko" else ""
    if isinstance(value, dict):
        return value.get(lang) or ""
    return ""


_SECTION_YEAR_RE = re.compile(r"(1[6-9]\d\d|20[0-2]\d)")


def _section_sort_order(patch: dict, heading, fallback: int) -> int:
    """Chronological key for a person section, as YYYYMM (MM=00 for year-only).

    Sections render in sort_order, so this is what makes a life story read front
    to back. An explicit sortOrder wins; otherwise the earliest year in the
    heading stands in, which is right for the "... (1898-1918)" and "1991년 8월,
    ..." heading shapes the sections actually use. Appending after the current
    last row is the last resort, and it is the one that used to put a newly
    written childhood section below a death scene.
    """
    if isinstance(patch.get("sortOrder"), int):
        return patch["sortOrder"]
    for lang in ("ko", "en"):
        match = _SECTION_YEAR_RE.search(_localized(heading, lang))
        if match:
            return int(match.group(1)) * 100
    return fallback


def _nationality_values(patch: dict, key: str):
    """Extract (code, label_ko, label_en) for a citizenship/origin patch node.

    Returns None when the key is absent so the caller leaves the columns
    untouched; an explicit empty {} clears the fields.
    """
    if key not in patch:
        return None
    node = patch.get(key) or {}
    if not isinstance(node, dict):
        return "", "", ""
    label = node.get("label") if isinstance(node.get("label"), dict) else None
    code = str(node.get("code") or "").strip()
    label_ko = _localized(label, "ko")
    label_en = _localized(label, "en")
    if key == "origin" and code == "georgia":
        label_ko = "그루지야"
    return code, label_ko, label_en


def _merge_patronymic_patch(patch: dict, before: dict | None = None) -> dict:
    """Field-wise patronymic PATCH merge; omitted values are never blanked."""
    before = before or {}
    state = {
        "ko": _collapse_spaces(before.get("ko")),
        "en": _collapse_spaces(before.get("en")),
        "native": _collapse_spaces(before.get("native")),
        "touched": "patronymic" in patch or "cyrillicPatronymic" in patch,
        "invalid": "",
    }
    if "patronymic" in patch:
        node = patch.get("patronymic")
        if node is None:
            state["ko"] = ""
            state["en"] = ""
        elif not isinstance(node, dict):
            state["invalid"] = "patronymic must be an object {ko,en} or null"
            return state
        else:
            for lang in ("ko", "en"):
                if lang in node:
                    state[lang] = _collapse_spaces(node.get(lang))
    if "cyrillicPatronymic" in patch:
        state["native"] = _collapse_spaces(patch.get("cyrillicPatronymic"))
    return state


def _contains_name_component(full_name: str, component: str) -> bool:
    full = f" {_collapse_spaces(full_name).casefold()} "
    part = _collapse_spaces(component).casefold()
    return bool(part and f" {part} " in full)


def _patronymic_problem(state: dict, native_name: str) -> str | None:
    if state.get("invalid"):
        return state["invalid"]
    ko, en, native = state.get("ko", ""), state.get("en", ""), state.get("native", "")
    if bool(ko) != bool(en):
        return "patronymic.ko and patronymic.en must be supplied together"
    if native and not (ko and en):
        return "cyrillicPatronymic requires both patronymic.ko and patronymic.en"
    if "cyrillic" in _detect_scripts(native_name) and ko and not native:
        return "a Cyrillic native name with a patronymic requires cyrillicPatronymic"
    if native and _contains_name_component(native_name, native):
        return (
            "cyrillic/nativeName already embeds cyrillicPatronymic; keep the "
            "patronymic only in its separate field"
        )
    return None


def _stored_patronymic_state(cur, person_id: str) -> dict:
    cur.execute(
        """SELECT patronymic_ko, patronymic_en, cyrillic_patronymic
           FROM commulingo_person_patronymics WHERE person_id = %s""",
        (person_id,),
    )
    row = cur.fetchone() or {}
    return {
        "ko": row.get("patronymic_ko") or "",
        "en": row.get("patronymic_en") or "",
        "native": row.get("cyrillic_patronymic") or "",
    }


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
        if "—" in _QUOTED_SPAN_RE.sub("", text):
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


def _parse_life_years(label: str) -> tuple[int | None, int | None]:
    m = re.match(r"^(\d{3,4})[–-](\d{3,4})$", label or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _surname(name: str) -> str:
    parts = [p for p in (name or "").replace("·", " ").split() if p]
    return parts[-1] if parts else ""


_NON_NAME_CHARS = re.compile(r"[^0-9a-z가-힣]+")


def _dedup_key(name: str) -> str:
    """Comparison key for name identity: casefold, drop spacing and punctuation.

    'C.L.R. James' and 'C. L. R. James', '본치-브루예비치' and '본치브루예비치'
    are one person written two ways, so raw-string equality misses them.
    """
    return _NON_NAME_CHARS.sub("", (name or "").casefold())


def _within_edits(a: str, b: str, limit: int) -> bool:
    """True when `a` and `b` are within `limit` single-character edits."""
    if abs(len(a) - len(b)) > limit:
        return False
    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        current = [i]
        for j, ch_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ch_a != ch_b)))
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


def _same_surname(a: str, b: str) -> bool:
    """Surnames match once transliteration noise is removed: Bonch-Bruyevich
    and Bonch-Bruevich are one spelling choice apart, not two people. Only ever
    consulted alongside an exact life-year match, so a loose threshold is safe."""
    key_a, key_b = _dedup_key(_surname(a)), _dedup_key(_surname(b))
    if not key_a or not key_b:
        return False
    if key_a == key_b:
        return True
    return _within_edits(key_a, key_b, 2 if min(len(key_a), len(key_b)) >= 5 else 1)


def _collapse_spaces(value) -> str:
    return " ".join(str(value or "").split())


def _split_full_name(full: str, lang: str = "en", codes=()) -> tuple[str, str]:
    """(given, family) from a full name: family = last token, given = the rest,
    except family-first nationalities (korea/china/vietnam/japan), which lead
    with the family name (Kim Mu-chong, 도쿠다 규이치).
    Single-token names (김일성, 카모) go wholly to family."""
    name = _collapse_spaces(full)
    if not name:
        return "", ""
    if " " not in name:
        return "", name
    if _family_first_joiner(lang, codes) is not None:
        family, given = name.split(" ", 1)
        return given, family
    given, family = name.rsplit(" ", 1)
    return given, family


def _compose_full_name(given: str, family: str, lang: str, codes=()) -> str:
    """The derived full name in the nationality's own order: 김+무정 → 김무정,
    Peng+Dehuai → Peng Dehuai, everyone else given-first with a space."""
    joiner = _family_first_joiner(lang, codes)
    if joiner is not None and given and family:
        return f"{family}{joiner}{given}"
    return " ".join(p for p in (given, family) if p)


def _name_order_codes(patch: dict, stored: dict | None = None) -> tuple[str, ...]:
    """The citizenship code that decides name order: the patch's when it sets
    one, else the stored row's (`citizenship_code` key)."""
    vals = _nationality_values(patch, "citizenship")
    if vals is not None:
        return (vals[0],) if vals[0] else ()
    code = (stored or {}).get("citizenship_code") or ""
    return (code,) if code else ()


def _patch_name_parts(patch: dict, lang: str, stored: dict | None = None,
                      codes=()) -> tuple[str, str, str]:
    """Effective (given, family, full) for one language after applying `patch`.

    Structured givenName/familyName win; a legacy full `name` is split; a
    partial parts patch falls back to `stored` (given_name_*/family_name_*
    row) for the missing side. `codes` are the citizenship codes that decide
    name order (see _compose_full_name). Mirrors frontend people-admin-store.js.
    """
    stored = stored or {}
    if "givenName" in patch or "familyName" in patch:
        given = (_collapse_spaces(_localized(patch.get("givenName"), lang))
                 if "givenName" in patch else _collapse_spaces(stored.get(f"given_name_{lang}")))
        family = (_collapse_spaces(_localized(patch.get("familyName"), lang))
                  if "familyName" in patch else _collapse_spaces(stored.get(f"family_name_{lang}")))
    elif "name" in patch:
        given, family = _split_full_name(_localized(patch.get("name"), lang), lang, codes)
    else:
        given = _collapse_spaces(stored.get(f"given_name_{lang}"))
        family = _collapse_spaces(stored.get(f"family_name_{lang}"))
    full = _compose_full_name(given, family, lang, codes)
    return given, family, full


def _existing_person_match(cur, target_id: str, patch: dict) -> dict | None:
    """Find a card that is the same person as `patch` under a different slug.

    The slug-uniqueness check alone let three duplicate pairs into the
    dictionary (오토 쿠시넨/오토 빌레 쿠시넨, 표트르/페테리스 스투치카,
    흐리스티안/크리스티안 라콥스키) — the curator picked a different Korean
    transliteration, so it picked a different slug, and nothing objected.
    Comparing name strings raw then let two more through (C.L.R. James over
    C. L. R. James, 본치브루예비치 over 본치-브루예비치), so every name comparison
    here runs on _dedup_key. Four signals: the English name, the Korean name,
    the life-year pair together with a near-matching surname, and a slug that is
    a segment-wise subset of an existing one.
    Returns the matched row plus a `why` phrase, or None.
    """
    codes = _name_order_codes(patch)
    _, _, name_ko = _patch_name_parts(patch, "ko", codes=codes)
    _, _, name_en = _patch_name_parts(patch, "en", codes=codes)
    birth, death = _parse_life_years(patch.get("years") or "")
    key_en, key_ko = _dedup_key(name_en), _dedup_key(name_ko)
    segments = set(target_id.split("-"))

    cur.execute("SELECT id, name_ko, name_en, birth_year, death_year FROM commulingo_people")
    rows = cur.fetchall()

    for row in rows:
        if key_en and _dedup_key(row["name_en"]) == key_en:
            return {**row, "why": f"registered under the same English name '{row['name_en']}'"}
        if key_ko and _dedup_key(row["name_ko"]) == key_ko:
            return {**row, "why": f"registered under the same Korean name '{row['name_ko']}'"}

    if birth:
        for row in rows:
            if row["birth_year"] != birth or row["death_year"] != death:
                continue
            if _same_surname(row["name_ko"], name_ko) or (
                name_en and _same_surname(row["name_en"], name_en)
            ):
                return {**row, "why": f"registered with the same life years "
                                      f"({patch.get('years')}) and surname"}

    # kuusinen vs otto-kuusinen — one slug is the other plus a given name.
    # Differing known birth years settle it: 야코블레프 1923–2005 and 1896–1938
    # share a slug segment and a surname but are plainly two people.
    for row in rows:
        other = set(row["id"].split("-"))
        if not (segments < other or other < segments):
            continue
        if birth and row["birth_year"] and birth != row["birth_year"]:
            continue
        return {**row, "why": f"registered under the overlapping slug '{row['id']}'"}
    return None


def _normalize_fate_label(label: str, death_year: int | None) -> str:
    """Strip the death year from a fate label — it already lives in `years` /
    deathYear and must not be repeated on the card. Political-event years (실각
    1964) differ from the death year and are preserved; only the death-year token
    is removed, then "년"/parens/dates/legacy "d." artifacts and separators are
    tidied. Keep in sync with frontend/data/commulingo/people-standard.js
    normalizeFateLabel — the CommuLingo fate standard both enforce."""
    text = (label or "").strip()
    if not text or not death_year:
        return text
    y = str(death_year)
    out = text
    out = re.sub(r"\(\s*" + y + r"\s*\)", "", out)                              # (1980)
    out = re.sub(y + r"\s*년(?:\s*\d{1,2}\s*월)?(?:\s*\d{1,2}\s*일)?", "", out)  # 1956년 4월 20일
    out = re.sub(r"\d{1,2}\s+[A-Z][a-z]+\s+" + y, "", out)                      # 20 April 1956
    out = re.sub(r"[A-Z][a-z]+\s+\d{1,2},?\s+" + y, "", out)                    # April 20, 1956
    out = re.sub(r"(?<![0-9])" + y + r"(?![0-9])", "", out)                     # bare death year
    out = re.sub(r"\bd\.\s*", "", out)                                          # legacy EN "d." tail
    out = re.sub(r"\(\s*\)", "", out)                                           # empty parens
    out = re.sub(r"(^|[\s·,])년(?=[\s·,]|$)", r"\1", out)                        # orphan 년
    out = re.sub(r"\s*·\s*", " · ", out)
    out = re.sub(r"\s*,\s*", ", ", out)
    out = re.sub(r"([·,])(?:\s*[·,])+", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"^[\s·,]+|[\s·,]+$", "", out).strip()
    return out


def _parse_date_token(token: str, fallback_year: int | None):
    token = (token or "").strip()
    if not token:
        return None
    if re.match(r"^\d{1,2}$", token) and fallback_year:
        return {"year": fallback_year, "month": int(token)}
    m = re.match(r"^(\d{3,4})(?:\.(\d{1,2}))?$", token)
    if not m:
        return None
    return {"year": int(m.group(1)), "month": int(m.group(2)) if m.group(2) else None}


def _period_columns(label: str) -> tuple:
    """(start_year, start_month, end_year, end_month) — port of parsePeriod."""
    first = (label or "").split(",")[0].strip()
    parts = [p.strip() for p in re.split(r"[–-]", first) if p.strip()]
    if not parts:
        return None, None, None, None
    start = _parse_date_token(parts[0], None)
    end = _parse_date_token(parts[1] if len(parts) > 1 else "", start["year"] if start else None)
    return (
        start["year"] if start else None,
        start["month"] if start else None,
        end["year"] if end else None,
        end["month"] if end else None,
    )


def _write_revision(cur, entity_type: str, entity_id: str, note: str, snapshot, changed_by: str):
    cur.execute(
        """INSERT INTO commulingo_people_revisions
              (entity_type, entity_id, revision_note, snapshot, changed_by)
           VALUES (%s, %s, %s, %s::jsonb, %s)""",
        (entity_type, entity_id, note, json.dumps(snapshot or {}, default=_json_default, ensure_ascii=False), changed_by),
    )


def _replace_patronymic(cur, person_id: str, state: dict):
    cur.execute("DELETE FROM commulingo_person_patronymics WHERE person_id = %s", (person_id,))
    ko, en, cyr = state.get("ko", ""), state.get("en", ""), state.get("native", "")
    if not (ko or en or cyr):
        return
    cur.execute(
        """INSERT INTO commulingo_person_patronymics
              (person_id, patronymic_ko, patronymic_en, cyrillic_patronymic, updated_at)
           VALUES (%s, %s, %s, %s, NOW())""",
        (person_id, ko, en, cyr),
    )


def _replace_aliases(cur, person_id: str, aliases: dict):
    cur.execute("DELETE FROM commulingo_person_aliases WHERE person_id = %s", (person_id,))
    # Dedupe on the conflict key keeping the LAST occurrence: a single
    # execute_values statement may not touch the same row twice
    # (CardinalityViolation), while the old per-row loop let the later
    # duplicate win via ON CONFLICT.
    rows: dict[tuple, tuple] = {}
    for lang in ("ko", "en"):
        values = aliases.get(lang) if isinstance(aliases, dict) else None
        for index, alias in enumerate(values or []):
            alias = alias.strip() if isinstance(alias, str) else ""
            if not alias:
                continue
            rows[(person_id, lang, alias)] = (person_id, lang, alias, index)
    if rows:
        execute_values(
            cur,
            """INSERT INTO commulingo_person_aliases (person_id, lang, alias, sort_order)
               VALUES %s
               ON CONFLICT (person_id, lang, alias)
               DO UPDATE SET sort_order = EXCLUDED.sort_order""",
            list(rows.values()),
        )


def _replace_scenes(cur, person_id: str, scenes: list):
    cur.execute("DELETE FROM commulingo_person_scenes WHERE person_id = %s", (person_id,))
    rows = []
    for index, scene in enumerate(scenes or []):
        if not isinstance(scene, (list, tuple)) or len(scene) < 2:
            continue
        collection_id = scene[0].strip() if isinstance(scene[0], str) else ""
        episode_id = scene[1].strip() if isinstance(scene[1], str) else ""
        if not collection_id or not episode_id:
            continue
        rows.append((person_id, collection_id, episode_id, index))
    if rows:
        execute_values(
            cur,
            """INSERT INTO commulingo_person_scenes
                  (person_id, collection_id, episode_id, sort_order)
               VALUES %s""",
            rows,
        )


def _apply_person_role(cur, person_id: str, role):
    """Upsert (dict) or clear (None) the person's role mapping.

    Only officeId/category are taken from the dict; icon/label render from
    the linked office or category (legacy per-person icon/label columns are
    written empty)."""
    if role is None:
        cur.execute("DELETE FROM commulingo_person_roles WHERE person_id = %s", (person_id,))
        return
    cur.execute(
        """INSERT INTO commulingo_person_roles
              (person_id, icon, office_id, category_id, label_ko, label_en, updated_at)
           VALUES (%s, '', NULLIF(%s, ''), NULLIF(%s, ''), '', '', NOW())
           ON CONFLICT (person_id) DO UPDATE SET
              icon = '', office_id = EXCLUDED.office_id,
              category_id = EXCLUDED.category_id,
              label_ko = '', label_en = '', updated_at = NOW()""",
        (
            person_id,
            role.get("officeId") or "",
            role.get("category") or role.get("categoryId") or "",
        ),
    )


def _replace_career(cur, person_id: str, career: list):
    cur.execute("DELETE FROM commulingo_person_career_entries WHERE person_id = %s", (person_id,))
    rows = []
    for index, entry in enumerate(career or []):
        if not isinstance(entry, dict):
            continue
        label = entry.get("y") or entry.get("period") or ""
        sy, sm, ey, em = _period_columns(label)
        role = entry.get("r") or entry.get("role") or {}
        rows.append((person_id, index, label, sy, sm, ey, em,
                     _localized(role, "ko"), _localized(role, "en")))
    if rows:
        execute_values(
            cur,
            """INSERT INTO commulingo_person_career_entries
                  (person_id, sort_order, period_label, start_year, start_month,
                   end_year, end_month, role_ko, role_en, updated_at)
               VALUES %s""",
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
        )


def _apply_person_create(cur, person_id: str, patch: dict) -> None:
    cur.execute("SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_sort FROM commulingo_people")
    next_sort = cur.fetchone()["next_sort"]
    sort_order = patch["sortOrder"] if isinstance(patch.get("sortOrder"), int) else next_sort
    birth, death = _parse_life_years(patch.get("years") or "")
    # Structured name parts (givenName/familyName win, legacy `name` is split);
    # name_ko/en are stored as the DERIVED full name — never written separately.
    codes = _name_order_codes(patch)
    given_ko, family_ko, name_ko = _patch_name_parts(patch, "ko", codes=codes)
    given_en, family_en, name_en = _patch_name_parts(patch, "en", codes=codes)
    fate = patch.get("fate") or {}
    citizenship = _nationality_values(patch, "citizenship") or ("", "", "")
    origin = _nationality_values(patch, "origin") or ("", "", "")
    cur.execute(
        """INSERT INTO commulingo_people
              (id, group_id, sort_order, initial, cyrillic, years_label, birth_year, death_year,
               name_ko, name_en, given_name_ko, given_name_en, family_name_ko, family_name_en,
               epithet_ko, epithet_en, bio_ko, bio_en,
               moment_ko, moment_en,
               fate_kind, fate_label_ko, fate_label_en,
               citizenship_code, citizenship_label_ko, citizenship_label_en,
               origin_code, origin_label_ko, origin_label_en, updated_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s, %s, NOW())""",
        (
            person_id,
            patch.get("groupId") or patch.get("group"),
            sort_order,
            "",
            patch.get("cyrillic") or "",
            patch.get("years") or "",
            birth, death,
            name_ko, name_en, given_ko, given_en, family_ko, family_en,
            _localized(patch.get("epithet"), "ko"), _localized(patch.get("epithet"), "en"),
            _localized(patch.get("bio"), "ko"), _localized(patch.get("bio"), "en"),
            _localized(patch.get("moment"), "ko"), _localized(patch.get("moment"), "en"),
            fate.get("kind") or "" if isinstance(fate, dict) else "",
            _normalize_fate_label(_localized(fate.get("label") if isinstance(fate, dict) else None, "ko"), death),
            _normalize_fate_label(_localized(fate.get("label") if isinstance(fate, dict) else None, "en"), death),
            citizenship[0], citizenship[1], citizenship[2],
            origin[0], origin[1], origin[2],
        ),
    )
    _replace_patronymic(cur, person_id, _merge_patronymic_patch(patch))
    _replace_aliases(cur, person_id, patch.get("aliases") or {
        "ko": [name_ko], "en": [name_en],
    })
    _replace_scenes(cur, person_id, patch.get("scenes") or [])
    _replace_career(cur, person_id, patch.get("career") or [])
    if patch.get("role"):
        _apply_person_role(cur, person_id, patch["role"])


def _apply_person_update(cur, person_id: str, patch: dict) -> None:
    sets, values = [], []

    def set_col(column, value):
        values.append(value)
        sets.append(f"{column} = %s")

    if "group" in patch or "groupId" in patch:
        set_col("group_id", patch.get("groupId") or patch.get("group"))
    if "cyrillic" in patch:
        set_col("cyrillic", patch.get("cyrillic") or "")
    if "years" in patch:
        birth, death = _parse_life_years(patch.get("years") or "")
        set_col("years_label", patch.get("years") or "")
        set_col("birth_year", birth)
        set_col("death_year", death)
    if "name" in patch or "givenName" in patch or "familyName" in patch:
        # Any name field recomputes all six name columns so the structured
        # parts and the derived full name never diverge.
        cur.execute(
            """SELECT given_name_ko, given_name_en, family_name_ko, family_name_en,
                      citizenship_code
               FROM commulingo_people WHERE id = %s""",
            (person_id,),
        )
        stored_name = dict(cur.fetchone() or {})
        codes = _name_order_codes(patch, stored_name)
        for lang in ("ko", "en"):
            given, family, full = _patch_name_parts(patch, lang, stored_name, codes=codes)
            set_col(f"name_{lang}", full)
            set_col(f"given_name_{lang}", given)
            set_col(f"family_name_{lang}", family)
    if "epithet" in patch:
        set_col("epithet_ko", _localized(patch.get("epithet"), "ko"))
        set_col("epithet_en", _localized(patch.get("epithet"), "en"))
    if "bio" in patch:
        set_col("bio_ko", _localized(patch.get("bio"), "ko"))
        set_col("bio_en", _localized(patch.get("bio"), "en"))
    if "moment" in patch:
        set_col("moment_ko", _localized(patch.get("moment"), "ko"))
        set_col("moment_en", _localized(patch.get("moment"), "en"))
    if "fate" in patch:
        fate = patch.get("fate") or {}
        # Death year comes from an incoming years patch if present, else the
        # stored record, so the fate label is stripped against the right year.
        if "years" in patch:
            _, death = _parse_life_years(patch.get("years") or "")
        else:
            cur.execute("SELECT death_year FROM commulingo_people WHERE id = %s", (person_id,))
            row = cur.fetchone()
            death = row["death_year"] if row else None
        set_col("fate_kind", fate.get("kind") or "" if isinstance(fate, dict) else "")
        set_col("fate_label_ko", _normalize_fate_label(_localized(fate.get("label") if isinstance(fate, dict) else None, "ko"), death))
        set_col("fate_label_en", _normalize_fate_label(_localized(fate.get("label") if isinstance(fate, dict) else None, "en"), death))
    if "sortOrder" in patch and isinstance(patch.get("sortOrder"), int):
        set_col("sort_order", patch["sortOrder"])
    for key, cols in (
        ("citizenship", ("citizenship_code", "citizenship_label_ko", "citizenship_label_en")),
        ("origin", ("origin_code", "origin_label_ko", "origin_label_en")),
    ):
        vals = _nationality_values(patch, key)
        if vals is not None:
            set_col(cols[0], vals[0])
            set_col(cols[1], vals[1])
            set_col(cols[2], vals[2])
    if sets:
        sets.append("updated_at = NOW()")
        values.append(person_id)
        cur.execute(f"UPDATE commulingo_people SET {', '.join(sets)} WHERE id = %s", values)
    if "patronymic" in patch or "cyrillicPatronymic" in patch:
        before_patronymic = _stored_patronymic_state(cur, person_id)
        _replace_patronymic(cur, person_id, _merge_patronymic_patch(patch, before_patronymic))
    if "aliases" in patch:
        _replace_aliases(cur, person_id, patch.get("aliases") or {})
    if "scenes" in patch:
        _replace_scenes(cur, person_id, patch.get("scenes") or [])
    if "career" in patch:
        _replace_career(cur, person_id, patch.get("career") or [])
    if "role" in patch:
        _apply_person_role(cur, person_id, patch.get("role"))


def _apply_office_row_create(cur, office_id: str, patch: dict) -> int:
    cur.execute(
        "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_sort FROM commulingo_office_rows WHERE office_id = %s",
        (office_id,),
    )
    next_sort = cur.fetchone()["next_sort"]
    sort_order = patch["sortOrder"] if isinstance(patch.get("sortOrder"), int) else next_sort
    label = patch.get("years") or patch.get("period") or ""
    sy, sm, ey, em = _period_columns(label)
    cur.execute(
        """INSERT INTO commulingo_office_rows
              (office_id, sort_order, period_label, start_year, start_month, end_year, end_month,
               body_ko, body_en, person_id, name_ko, name_en, note_ko, note_en, updated_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULLIF(%s, ''), %s, %s, %s, %s, NOW())
           RETURNING id""",
        (
            office_id, sort_order, label, sy, sm, ey, em,
            _localized(patch.get("body"), "ko"), _localized(patch.get("body"), "en"),
            patch.get("personId") or "",
            _localized(patch.get("name"), "ko"), _localized(patch.get("name"), "en"),
            _localized(patch.get("note"), "ko"), _localized(patch.get("note"), "en"),
        ),
    )
    return cur.fetchone()["id"]


def _apply_office_row_update(cur, row_id: int, patch: dict) -> None:
    sets, values = [], []

    def set_col(column, value):
        values.append(value)
        sets.append(f"{column} = %s")

    if "sortOrder" in patch:
        try:
            set_col("sort_order", int(patch.get("sortOrder")))
        except (TypeError, ValueError):
            set_col("sort_order", 0)
    if "years" in patch or "period" in patch:
        label = patch.get("years") or patch.get("period") or ""
        sy, sm, ey, em = _period_columns(label)
        set_col("period_label", label)
        set_col("start_year", sy)
        set_col("start_month", sm)
        set_col("end_year", ey)
        set_col("end_month", em)
    if "body" in patch:
        set_col("body_ko", _localized(patch.get("body"), "ko"))
        set_col("body_en", _localized(patch.get("body"), "en"))
    if "personId" in patch:
        set_col("person_id", patch.get("personId") or None)
    if "name" in patch:
        set_col("name_ko", _localized(patch.get("name"), "ko"))
        set_col("name_en", _localized(patch.get("name"), "en"))
    if "note" in patch:
        set_col("note_ko", _localized(patch.get("note"), "ko"))
        set_col("note_en", _localized(patch.get("note"), "en"))
    if sets:
        sets.append("updated_at = NOW()")
        values.append(row_id)
        cur.execute(f"UPDATE commulingo_office_rows SET {', '.join(sets)} WHERE id = %s", values)


def _person_create_nationality_problem(patch: dict) -> str | None:
    """Return a fail-closed error when a new card lacks either flag."""
    for key, public_key in (("citizenship", "citizenship"), ("origin", "nationalOrigin")):
        node = patch.get(key)
        if not isinstance(node, dict) or not str(node.get("code") or "").strip():
            return f"patch.{public_key}.code is required for person create"
        label = node.get("label")
        if (not isinstance(label, dict)
                or not str(label.get("ko") or "").strip()
                or not str(label.get("en") or "").strip()):
            return f"patch.{public_key}.label.ko and label.en are required for person create"
    return None


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
    if target_type == "person":
        for key in ("id", "group", "groupId", "cyrillic", "cyrillicPatronymic", "years"):
            if key in patch and patch[key] is not None and not isinstance(patch[key], str):
                return (
                    f"Error: {key} must be a plain string, not an object or list. "
                    "Only bilingual public text fields use {ko, en}."
                )
        stored = {}
        if action != "create":
            cur.execute(
                "SELECT cyrillic, citizenship_code, origin_code FROM commulingo_people WHERE id = %s",
                (target_id,),
            )
            stored = dict(cur.fetchone() or {})
        cyrillic = str(
            patch.get("cyrillic") if "cyrillic" in patch else stored.get("cyrillic") or ""
        ).strip()
        patronymic_state = _merge_patronymic_patch(
            patch,
            _stored_patronymic_state(cur, target_id) if action != "create" else {},
        )
        patronymic_error = _patronymic_problem(patronymic_state, cyrillic)
        if patronymic_error:
            return f"Error: {patronymic_error}."
        cyrillic_patronymic = patronymic_state["native"]
        # The native-name line must use the person's own script. Check it against
        # the citizenship the record will HAVE after this patch, so correcting a
        # wrong citizenship and the name together is accepted.
        nationality_codes: list[tuple[str, str]] = []
        for key, public, column in (
            ("citizenship", "citizenship", "citizenship_code"),
            ("origin", "nationalOrigin", "origin_code"),
        ):
            if isinstance(patch.get(key), dict):
                nationality_codes.append((public, str(patch[key].get("code") or "").strip()))
            elif key not in patch:
                nationality_codes.append((public, str(stored.get(column) or "")))
        for field, value in (("cyrillic", cyrillic), ("cyrillicPatronymic", cyrillic_patronymic)):
            problem = _check_native_script(value, nationality_codes, field)
            if problem:
                return problem
        # Same rule for the ko/en side: the name must never embed the patronymic.
        # The frontend composes given + patronymic + family on render, so an
        # embedded one doubles (오토 율리예비치 율리예비치 시미트 — the bug that
        # led to structured name parts, frontend migration 060). Checked against
        # the state the record will HAVE after this patch.
        name_touched = any(k in patch for k in ("name", "givenName", "familyName"))
        if name_touched or "patronymic" in patch:
            stored_name = {}
            if action != "create":
                cur.execute(
                    """SELECT p.given_name_ko, p.given_name_en, p.family_name_ko, p.family_name_en,
                              pa.patronymic_ko, pa.patronymic_en
                       FROM commulingo_people p
                       LEFT JOIN commulingo_person_patronymics pa ON pa.person_id = p.id
                       WHERE p.id = %s""",
                    (target_id,),
                )
                stored_name = dict(cur.fetchone() or {})
            for lang in ("ko", "en"):
                _, _, full = _patch_name_parts(patch, lang, stored_name)
                pat = patronymic_state[lang]
                if not pat or not full:
                    continue
                tokens = full.split()
                embedded = (pat.lower() in [t.lower() for t in tokens]) if lang == "en" else (pat in tokens)
                if embedded:
                    return (
                        f"Error: the {lang} name embeds the patronymic '{pat}'. name / "
                        "givenName+familyName carry given name + surname ONLY — the "
                        "patronymic goes only in patronymic {ko,en} and renders between "
                        "them automatically. A Western middle name is part of givenName, "
                        "not a patronymic."
                    )
        for key in _LOCALIZED_PERSON_KEYS:
            if key in patch and patch[key] is not None and not isinstance(patch[key], dict):
                return (
                    f"Error: {key} must be an object {{\"ko\": \"...\", \"en\": \"...\"}} — "
                    "a plain string is rejected because the site is bilingual and the "
                    "other language would be silently lost."
                )
        # `moment` had no limit at all, which is how 308-character moments reached the
        # card. These ceilings exist to refuse an overflowing field, not to be written
        # toward — length is prescribed to the curator as a sentence count.
        for key, (ko_max, en_max), overflow in (
            ("epithet", FIELD_LIMITS["epithet"], "Keep career chronology in career rows."),
            ("bio", FIELD_LIMITS["bio"], "Keep career chronology in career rows."),
            ("moment", FIELD_LIMITS["moment"], f"A moment is {sentence_prescription('moment')} — "
                                               "pick a sharper scene instead of explaining this one."),
        ):
            value = patch.get(key)
            if not isinstance(value, dict):
                continue
            ko_len = len(value.get("ko") or "")
            en_len = len(value.get("en") or "")
            if ko_len > ko_max or en_len > en_max:
                return (
                    f"Error: {key} is too long (ko {ko_len}/{ko_max}, en {en_len}/{en_max} "
                    f"characters). Cut the stated overflow; do not redraft from scratch. "
                    f"{overflow}"
                )
        for key in ("citizenship", "origin"):
            if key not in patch or patch[key] is None:
                continue
            node = patch[key]
            if not isinstance(node, dict):
                return (
                    f"Error: {key} must be {{\"code\": \"soviet\", "
                    f"\"label\": {{\"ko\": \"소련\", \"en\": \"Soviet Union\"}}}} or {{}} to clear."
                )
            code = str(node.get("code") or "").strip()
            if code and code not in _NATIONALITY_CODES:
                return (
                    f"Error: {key}.code '{code}' has no flag icon on the site. "
                    f"Use one of: {', '.join(sorted(_NATIONALITY_CODES))}. "
                    f"For create, choose a reviewed supported classification; the field cannot be omitted."
                )
        if "aliases" in patch and patch["aliases"] is not None:
            aliases = patch["aliases"]
            if (not isinstance(aliases, dict)
                    or not set(aliases) <= {"ko", "en"}
                    or not all(isinstance(v, list) for v in aliases.values())):
                return (
                    "Error: aliases must be {\"ko\": [\"수슬로프\"], \"en\": [\"Suslov\"]} — "
                    "lists per language of the exact strings used in book text."
                )
        if "career" in patch and patch["career"] is not None:
            if not isinstance(patch["career"], list):
                return "Error: career must be a list of {y, r} entries."
            for i, entry in enumerate(patch["career"]):
                if (not isinstance(entry, dict)
                        or not (entry.get("y") or entry.get("period"))
                        or not isinstance(entry.get("r") or entry.get("role"), dict)):
                    return (
                        f"Error: career[{i}] must be {{\"y\": \"1922–1953\", "
                        "\"r\": {\"ko\": \"...\", \"en\": \"...\"}}} — other shapes would "
                        "be stored as empty rows."
                    )
        if "fate" in patch and patch["fate"] is not None:
            fate = patch["fate"]
            if not isinstance(fate, dict):
                return "Error: fate must be {kind, label: {ko, en}} or null."
            if fate.get("label") is not None and not isinstance(fate["label"], dict):
                return "Error: fate.label must be {\"ko\": \"처형\", \"en\": \"Executed\"}."
            label = fate.get("label") or {}
            fl_ko, fl_en = FIELD_LIMITS["fate_label"]
            if (len(label.get("ko") or "") > fl_ko
                    or len(label.get("en") or "") > fl_en):
                return (
                    f"Error: fate.label is too long (ko {len(label.get('ko') or '')}/{fl_ko}, "
                    f"en {len(label.get('en') or '')}/{fl_en} characters). Write the cause of death only, WITHOUT "
                    "the death year (it renders from `years`): 처형/Executed, 자연사/"
                    "Natural causes, a specific illness (심장마비/Heart attack), place "
                    "with ' · ' (암살 · 멕시코). A deposed/exile fate keeps its event "
                    "year (실각 1964). Move burial and other detail to bio or sections."
                )
        if action == "create":
            nationality_problem = _person_create_nationality_problem(patch)
            if nationality_problem:
                return (
                    f"Error: {nationality_problem}. Both citizenship and nationalOrigin "
                    "are mandatory; nationalOrigin may equal citizenship but must not be blank."
                )
            if patch.get("id") and patch["id"] != target_id:
                return f"Error: patch.id '{patch['id']}' conflicts with target_id '{target_id}' — they must match (or omit patch.id)."
            if not _ID_RE.match(target_id):
                return "Error: target_id must be a lowercase kebab-case slug (e.g. 'ordzhonikidze')."
            cur.execute("SELECT 1 FROM commulingo_people WHERE id = %s", (target_id,))
            if cur.fetchone():
                return (
                    f"Error: person '{target_id}' already exists — edit it with "
                    f"{_write_tool_call('person', 'update')}(person_id='{target_id}', ...)."
                )
            duplicate = _existing_person_match(cur, target_id, patch)
            if duplicate:
                return (
                    f"Error: '{duplicate['id']}' ({duplicate['name_ko']}) is already "
                    f"{duplicate['why']} — this is the same person under a different "
                    f"slug. Call {_write_tool_call('person', 'update')}(person_id="
                    f"'{duplicate['id']}', ...) on that id, putting any alternate "
                    "spelling in the 'aliases' field of the same patch. If they are "
                    "genuinely two different people, give the new card an English name "
                    "and slug that do not collide with the existing one."
                )
            group = patch.get("groupId") or patch.get("group") or ""
            cur.execute("SELECT 1 FROM commulingo_people_groups WHERE id = %s", (group,))
            if not cur.fetchone():
                return f"Error: unknown group '{group}'. Check commulingo_people(action='list_groups')."
            for lang in ("ko", "en"):
                _, _, full = _patch_name_parts(patch, lang)
                if not full:
                    return (
                        "Error: person create requires a name per language — either "
                        "name {ko,en} or givenName/familyName {ko,en} (single-token "
                        "East Asian names go wholly in familyName)."
                    )
            for key in ("bio", "epithet"):
                value = patch.get(key) or {}
                if not (isinstance(value, dict) and value.get("ko") and value.get("en")):
                    return f"Error: patch.{key}.ko and patch.{key}.en are required for person create."
            if not patch.get("career"):
                return "Error: at least one bilingual career entry is required for person create."
            role = patch.get("role")
            if not isinstance(role, dict) or not (
                role.get("officeId") or role.get("category") or role.get("categoryId")
            ):
                return (
                    "Error: a primary role with officeId, category, or categoryId "
                    "is required for person create."
                )
        else:
            cur.execute("SELECT 1 FROM commulingo_people WHERE id = %s", (target_id,))
            if not cur.fetchone():
                return (
                    f"Error: person '{target_id}' not found. Find the id with "
                    f"{_reader_call('search_people')}."
                )
            if action == "update" and ("group" in patch or "groupId" in patch):
                group = patch.get("groupId") or patch.get("group") or ""
                cur.execute("SELECT 1 FROM commulingo_people_groups WHERE id = %s", (group,))
                if not cur.fetchone():
                    return f"Error: unknown group '{group}'."
        fate = patch.get("fate")
        if isinstance(fate, dict) and fate.get("kind") and fate["kind"] not in _FATE_KINDS:
            return f"Error: fate.kind must be one of {', '.join(_FATE_KINDS)}."
        if "role" in patch and patch["role"] is not None:
            role = patch["role"]
            if not isinstance(role, dict):
                return "Error: role must be {officeId} or {category}, or null to clear."
            office_id = role.get("officeId") or ""
            category = role.get("category") or role.get("categoryId") or ""
            if office_id and category:
                return "Error: role takes exactly one of officeId or category, not both."
            if not office_id and not category:
                return (
                    "Error: role needs officeId or category (icon/label render from "
                    "them — see commulingo_people action='list_categories' / 'list_offices')."
                )
            if office_id:
                cur.execute("SELECT 1 FROM commulingo_offices WHERE id = %s", (office_id,))
                if not cur.fetchone():
                    return f"Error: role.officeId '{office_id}' does not exist."
            else:
                cur.execute("SELECT 1 FROM commulingo_role_categories WHERE id = %s", (category,))
                if not cur.fetchone():
                    cur.execute("SELECT id FROM commulingo_role_categories ORDER BY sort_order")
                    valid = ", ".join(r["id"] for r in cur.fetchall())
                    return f"Error: unknown role category '{category}'. Valid: {valid}."
    elif target_type == "person_section":
        cur.execute("SELECT 1 FROM commulingo_people WHERE id = %s", (target_id,))
        if not cur.fetchone():
            return f"Error: person '{target_id}' not found (person_section targets a person id)."
        slug = patch.get("slug") or ""
        if not _SLUG_RE.match(slug):
            return "Error: patch.slug is required — a short kebab-case id like 'early-life' or 'purge-role'."
        for key in ("heading", "body"):
            if key in patch and patch[key] is not None and not isinstance(patch[key], dict):
                return f"Error: {key} must be an object {{\"ko\": \"...\", \"en\": \"...\"}}."
        # body was the one long-form field with no ceiling on either side of the
        # call — not in the tool schema, not here. The tool schema now carries a
        # maxLength; this mirrors it for the writer paths that reach _validate
        # without going through the schema, exactly as bio/epithet/moment do.
        body_patch = patch.get("body")
        if isinstance(body_patch, dict):
            ko_max, en_max = FIELD_LIMITS["section_body"]
            ko_len = len(body_patch.get("ko") or "")
            en_len = len(body_patch.get("en") or "")
            if ko_len > ko_max or en_len > en_max:
                return (
                    f"Error: section body is too long (ko {ko_len}/{ko_max}, en "
                    f"{en_len}/{en_max} characters). The target is "
                    f"{SECTION_BODY_TARGET[0]}-{SECTION_BODY_TARGET[1]} Korean characters; "
                    f"a body this size is two topics — file the second one as its own "
                    f"section instead of trimming this one to fit."
                )
        cur.execute(
            "SELECT 1 FROM commulingo_person_sections WHERE person_id = %s AND slug = %s",
            (target_id, slug),
        )
        exists = bool(cur.fetchone())
        if action == "create":
            if exists:
                return (
                    f"Error: section '{slug}' already exists for '{target_id}'. Use action "
                    f"'update' on that slug, or pick a genuinely different topic. Do NOT "
                    f"retry this create under a modified slug — that files the same topic "
                    f"twice."
                )
            # Slug uniqueness alone let the same topic in twice under two slugs when both
            # lanes enriched one person at once: 예이젠시테인 got 몽타주 이론 as
            # montage-theory and montage-theory-collision 65 seconds apart. Headings are
            # the topic, so they are what a duplicate has to be caught on.
            heading = patch.get("heading") or {}
            key_ko, key_en = _dedup_key(heading.get("ko")), _dedup_key(heading.get("en"))
            if key_ko or key_en:
                cur.execute(
                    "SELECT slug, heading_ko, heading_en FROM commulingo_person_sections "
                    "WHERE person_id = %s", (target_id,)
                )
                for row in cur.fetchall():
                    if (key_ko and _dedup_key(row["heading_ko"]) == key_ko) or (
                        key_en and _dedup_key(row["heading_en"]) == key_en
                    ):
                        return (
                            f"Error: section '{row['slug']}' already covers this topic for "
                            f"'{target_id}' (heading '{row['heading_ko']}'). Rewrite it with "
                            f"{_write_tool_call('person_section', 'update')} on slug "
                            f"'{row['slug']}', or choose a different topic."
                        )
            body = patch.get("body") or {}
            if not (body.get("ko") or body.get("en")):
                return "Error: body.ko or body.en (markdown) is required for section create."
        elif not exists:
            return (
                f"Error: section '{slug}' not found for '{target_id}'. "
                f"{_reader_call('get_sections')} lists the existing slugs."
            )
    elif target_type == "history_event_person":
        if action == "delete":
            return "Error: history_event_person deletion is not available to the unattended curator."
        cur.execute("SELECT 1 FROM commulingo_history_events WHERE id = %s", (target_id,))
        if not cur.fetchone():
            return (
                f"Error: history event {target_id} not found. Find the id with "
                f"{_reader_call('list_events')}."
            )
        person_id = str(patch.get("personId") or "").strip()
        if not person_id:
            return "Error: history_event_person patch.personId is required."
        cur.execute("SELECT 1 FROM commulingo_people WHERE id = %s", (person_id,))
        if not cur.fetchone():
            return (
                f"Error: person {person_id} not found. Find the id with "
                f"{_reader_call('search_people')}."
            )
        kind = str(patch.get("relationKind") or "").strip()
        if kind not in _HISTORY_RELATION_KINDS:
            return f"Error: relationKind must be one of {', '.join(_HISTORY_RELATION_KINDS)}."
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
    elif target_type == "history_event":
        if action != "update":
            return (
                "Error: history events are created and retired by hand. The curator may "
                "only update one that already exists."
            )
        cur.execute(
            "SELECT COALESCE(summary_ko, '') = '' AS skeleton "
            "FROM commulingo_history_events WHERE id = %s",
            (target_id,),
        )
        event_row = cur.fetchone()
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

    elif target_type == "history_event_section":
        if action == "delete":
            return "Error: event body sections are not deleted by the unattended curator."
        cur.execute(
            "SELECT body_ko, body_en FROM commulingo_history_events WHERE id = %s",
            (target_id,),
        )
        row = cur.fetchone()
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

    elif target_type == "term":
        for key in ("id", "original"):
            if key in patch and patch[key] is not None and not isinstance(patch[key], str):
                return f"Error: {key} must be a plain string."
        for key in _LOCALIZED_TERM_KEYS:
            if key in patch and patch[key] is not None and not isinstance(patch[key], dict):
                return (
                    f"Error: {key} must be an object {{\"ko\": \"...\", \"en\": \"...\"}} — "
                    "a plain string would silently blank the other language."
                )
        if "category" in patch:
            if patch["category"] not in _TERM_CATEGORIES:
                return (
                    f"Error: category must be one of {_TERM_CATEGORY_HINT}. Without it "
                    "the entry shows up on the glossary under 'Uncategorized'."
                )
        elif action == "create":
            return f"Error: category is required on create. One of {_TERM_CATEGORY_HINT}."
        if action == "create" and not patch.get("period"):
            return (
                "Error: period is required on create, as "
                "{\"ko\": \"1930–1960\", \"en\": \"1930–1960\"} or "
                "{\"ko\": \"개념\", \"en\": \"Concept\"} when undated."
            )
        for key in ("startYear", "endYear"):
            value = patch.get(key)
            if key in patch and value is not None and not isinstance(value, int):
                return f"Error: {key} must be an integer year or null."
        start, end = patch.get("startYear"), patch.get("endYear")
        if isinstance(start, int) and isinstance(end, int) and end < start:
            return f"Error: endYear ({end}) is before startYear ({start})."
        # A dated label with no startYear sorts to the end of the chronological
        # view, which is why the years are asked for alongside the label.
        period = patch.get("period")
        if action == "create" and isinstance(period, dict) and start is None:
            labels = f"{period.get('ko') or ''} {period.get('en') or ''}"
            if _YEAR_RE.search(labels):
                return (
                    f"Error: period '{labels.strip()}' names a year, so startYear is "
                    "required for chronological sorting. Use the decade or century start "
                    "for a label like 1980년대 (1980) or 19세기 (1800)."
                )
        definition = patch.get("definition")
        if isinstance(definition, dict):
            df_ko, df_en = FIELD_LIMITS["definition"]
            if len(definition.get("ko") or "") > df_ko or len(definition.get("en") or "") > df_en:
                return (
                    f"Error: definition is too long (ko {len(definition.get('ko') or '')}/{df_ko}, "
                    f"en {len(definition.get('en') or '')}/{df_en} characters). It is the card "
                    "paragraph — move depth to body (markdown)."
                )
        if "aliases" in patch and patch["aliases"] is not None:
            aliases = patch["aliases"]
            if (not isinstance(aliases, dict)
                    or not set(aliases) <= {"ko", "en"}
                    or not all(isinstance(v, list) for v in aliases.values())):
                return (
                    "Error: aliases must be {\"ko\": [\"굴라크\"], \"en\": [\"Gulag\"]} — the exact "
                    "strings prose uses; they drive site-wide auto-linking."
                )
        if "parentId" in patch and patch["parentId"] is not None:
            parent = str(patch["parentId"]).strip()
            if not parent:
                return "Error: parentId must be a term id, or null to detach the entry."
            if parent == target_id:
                return f"Error: parentId '{parent}' is the entry itself."
            cur.execute("SELECT parent_id FROM commulingo_terms WHERE id = %s", (parent,))
            parent_row = cur.fetchone()
            if not parent_row:
                return (
                    f"Error: parentId '{parent}' is not a registered term. Find it with "
                    f"{_reader_call('list_terms')}."
                )
            # One level only, matching the entry page: a child cannot be a parent.
            if parent_row["parent_id"]:
                return (
                    f"Error: term '{parent}' is itself nested under "
                    f"'{parent_row['parent_id']}', and the glossary nests one level only. "
                    f"Use '{parent_row['parent_id']}' as the parent, or leave this entry flat."
                )
            cur.execute("SELECT id FROM commulingo_terms WHERE parent_id = %s LIMIT 1", (target_id,))
            child = cur.fetchone()
            if child:
                return (
                    f"Error: '{target_id}' already has '{child['id']}' nested under it, so it "
                    "cannot become a child itself (the glossary nests one level only)."
                )
        for key, table, reader in (("people", "commulingo_people", "search_people"),
                                   ("events", "commulingo_history_events", "list_events")):
            if key not in patch or patch[key] is None:
                continue
            value = patch[key]
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                return f"Error: {key} must be a list of {table} ids."
            for item in value:
                cur.execute(f"SELECT 1 FROM {table} WHERE id = %s", (item,))
                if not cur.fetchone():
                    return (
                        f"Error: {key} id '{item}' not found. Find it with "
                        f"{_reader_call(reader)}."
                    )
        if action == "create":
            if patch.get("id") and patch["id"] != target_id:
                return f"Error: patch.id '{patch['id']}' conflicts with target_id '{target_id}'."
            if not _ID_RE.match(target_id):
                return "Error: target_id must be a lowercase kebab-case slug (e.g. 'nomenklatura')."
            cur.execute("SELECT 1 FROM commulingo_terms WHERE id = %s", (target_id,))
            if cur.fetchone():
                return (
                    f"Error: term '{target_id}' already exists — edit it with "
                    f"{_write_tool_call('term', 'update')}(term_id='{target_id}', ...), "
                    f"sending only the fields that change. Read it first with "
                    f"{_reader_call('get_term')}."
                )
            term = patch.get("term") or {}
            if not (isinstance(term, dict) and term.get("ko") and term.get("en")):
                return "Error: patch.term.ko and patch.term.en are required for term create."
            if not (isinstance(definition, dict) and definition.get("ko") and definition.get("en")):
                return "Error: patch.definition.ko and patch.definition.en are required for term create."
            # An alias or name colliding with an existing term means this is the
            # same concept under a different slug.
            candidates = {term.get("ko"), term.get("en")}
            aliases = patch.get("aliases") or {}
            for values in (aliases.get("ko") or [], aliases.get("en") or []):
                candidates.update(v for v in values if isinstance(v, str))
            candidates.discard(None)
            for candidate in candidates:
                cur.execute(
                    """SELECT t.id FROM commulingo_terms t
                        WHERE lower(btrim(t.term_ko)) = lower(btrim(%(c)s))
                           OR lower(btrim(t.term_en)) = lower(btrim(%(c)s))
                       UNION
                       SELECT a.term_id FROM commulingo_term_aliases a
                        WHERE lower(btrim(a.alias)) = lower(btrim(%(c)s))""",
                    {"c": candidate},
                )
                row = cur.fetchone()
                if row:
                    return (
                        f"Error: '{candidate}' is already registered on term "
                        f"'{row['id']}' — this is the same concept, so it is not a gap. "
                        "Move to a different candidate (or answer NO_CANDIDATE if the "
                        f"material has none). To revise that card instead, call "
                        f"{_write_tool_call('term', 'update')}(term_id='{row['id']}', ...) "
                        "if you hold that tool."
                    )
        else:
            cur.execute("SELECT 1 FROM commulingo_terms WHERE id = %s", (target_id,))
            if not cur.fetchone():
                return (
                    f"Error: term '{target_id}' not found. Find the id with "
                    f"{_reader_call('list_terms')}."
                )
    else:  # office_row
        for key in _LOCALIZED_OFFICE_ROW_KEYS:
            if key in patch and patch[key] is not None and not isinstance(patch[key], dict):
                return (
                    f"Error: {key} must be an object {{\"ko\": \"...\", \"en\": \"...\"}} — "
                    "a plain string would silently blank the other language."
                )
        if action == "create":
            cur.execute("SELECT 1 FROM commulingo_offices WHERE id = %s", (target_id,))
            if not cur.fetchone():
                return f"Error: office '{target_id}' not found (office_row create targets an office id)."
        else:
            if not target_id.isdigit():
                return (
                    f"Error: office row '{target_id}' not found (office_row update/delete "
                    "targets a numeric row id from get_office/get_person)."
                )
            cur.execute("SELECT 1 FROM commulingo_office_rows WHERE id = %s", (int(target_id),))
            if not cur.fetchone():
                return f"Error: office row '{target_id}' not found."
        if patch.get("personId"):
            cur.execute("SELECT 1 FROM commulingo_people WHERE id = %s", (patch["personId"],))
            if not cur.fetchone():
                return f"Error: personId '{patch['personId']}' does not exist."
    return None


def _replace_term_aliases(cur, term_id: str, aliases: dict):
    cur.execute("DELETE FROM commulingo_term_aliases WHERE term_id = %s", (term_id,))
    # Dedupe keeping the last occurrence — see _replace_aliases.
    rows: dict[tuple, tuple] = {}
    for lang in ("ko", "en"):
        for index, alias in enumerate((aliases or {}).get(lang) or []):
            value = alias.strip() if isinstance(alias, str) else ""
            if not value:
                continue
            rows[(term_id, lang, value)] = (term_id, lang, value, index)
    if rows:
        execute_values(
            cur,
            """INSERT INTO commulingo_term_aliases (term_id, lang, alias, sort_order)
               VALUES %s
               ON CONFLICT (term_id, lang, alias) DO UPDATE SET sort_order = EXCLUDED.sort_order""",
            list(rows.values()),
        )


def _replace_term_links(cur, term_id: str, table: str, column: str, ids: list):
    cur.execute(f"DELETE FROM {table} WHERE term_id = %s", (term_id,))
    rows = []
    for index, item in enumerate(ids or []):
        value = item.strip() if isinstance(item, str) else ""
        if not value:
            continue
        rows.append((term_id, value, index))
    if rows:
        execute_values(
            cur,
            f"""INSERT INTO {table} (term_id, {column}, sort_order)
                VALUES %s ON CONFLICT DO NOTHING""",
            rows,
        )


def _apply_term_create(cur, term_id: str, patch: dict) -> None:
    cur.execute("SELECT COALESCE(MAX(sort_order), -1) + 10 AS next_sort FROM commulingo_terms")
    next_sort = cur.fetchone()["next_sort"]
    term = patch.get("term") or {}
    period = patch.get("period")
    cur.execute(
        """INSERT INTO commulingo_terms
              (id, sort_order, term_ko, term_en, original,
               period_label, period_ko, period_en, start_year, end_year, category,
               definition_ko, definition_en, body_ko, body_en, sources, parent_id, updated_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, NOW())""",
        (
            term_id,
            patch["sortOrder"] if isinstance(patch.get("sortOrder"), int) else next_sort,
            _localized(term, "ko"), _localized(term, "en"),
            patch.get("original") or "",
            # period_label is the frozen pre-071 column; the frontend store reads
            # period_ko/en and only falls back to it, so keep it populated with
            # the Korean label for anything that still looks at the old column.
            _localized(period, "ko"),
            _localized(period, "ko"), _localized(period, "en"),
            patch.get("startYear"), patch.get("endYear"),
            patch.get("category") or "",
            _localized(patch.get("definition"), "ko"), _localized(patch.get("definition"), "en"),
            _localized(patch.get("body"), "ko"), _localized(patch.get("body"), "en"),
            json.dumps(patch.get("sources") or [], ensure_ascii=False),
            (patch.get("parentId") or None),
        ),
    )
    _replace_term_aliases(cur, term_id, patch.get("aliases") or {
        "ko": [_localized(term, "ko")], "en": [_localized(term, "en")],
    })
    _replace_term_links(cur, term_id, "commulingo_term_people", "person_id", patch.get("people") or [])
    _replace_term_links(cur, term_id, "commulingo_term_events", "event_id", patch.get("events") or [])


def _apply_term_update(cur, term_id: str, patch: dict) -> None:
    sets, values = [], []

    def set_col(column, value):
        values.append(value)
        sets.append(f"{column} = %s")

    if "term" in patch:
        set_col("term_ko", _localized(patch.get("term"), "ko"))
        set_col("term_en", _localized(patch.get("term"), "en"))
    if "original" in patch:
        set_col("original", patch.get("original") or "")
    if "period" in patch:
        period = patch.get("period")
        set_col("period_label", _localized(period, "ko"))
        set_col("period_ko", _localized(period, "ko"))
        set_col("period_en", _localized(period, "en"))
    for key, column in (("startYear", "start_year"), ("endYear", "end_year")):
        if key in patch:
            set_col(column, patch.get(key))
    if "category" in patch:
        set_col("category", patch.get("category") or "")
    if "definition" in patch:
        set_col("definition_ko", _localized(patch.get("definition"), "ko"))
        set_col("definition_en", _localized(patch.get("definition"), "en"))
    if "body" in patch:
        set_col("body_ko", _localized(patch.get("body"), "ko"))
        set_col("body_en", _localized(patch.get("body"), "en"))
    if "sortOrder" in patch and isinstance(patch.get("sortOrder"), int):
        set_col("sort_order", patch["sortOrder"])
    if "parentId" in patch:
        set_col("parent_id", patch.get("parentId") or None)
    if "sources" in patch and patch.get("sources") is not None:
        values.append(json.dumps(patch["sources"], ensure_ascii=False))
        sets.append("sources = %s::jsonb")
    if sets:
        sets.append("updated_at = NOW()")
        values.append(term_id)
        cur.execute(f"UPDATE commulingo_terms SET {', '.join(sets)} WHERE id = %s", values)
    if "aliases" in patch:
        _replace_term_aliases(cur, term_id, patch.get("aliases") or {})
    if "people" in patch:
        _replace_term_links(cur, term_id, "commulingo_term_people", "person_id", patch.get("people") or [])
    if "events" in patch:
        _replace_term_links(cur, term_id, "commulingo_term_events", "event_id", patch.get("events") or [])


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


def event_section_headings(body: str) -> list[str]:
    """The `## ` headings of an event body, in order (used by the lane prompt)."""
    return [heading for heading, _ in _split_event_body(body) if heading]


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


def _splice_event_section(body: str, heading: str, section_body: str,
                          after: str, action: str) -> str:
    """Write one `## heading` part into an event body and return the new body."""
    parts = _split_event_body(body)
    heading = (heading or "").strip()
    block = f"## {heading}\n\n{(section_body or '').strip()}"
    at = _find_event_section(parts, heading)
    if action == "update":
        # _validate has already established the section exists in both languages.
        parts[at] = (heading, block)
    else:
        anchor = _find_event_section(parts, after) if after else -1
        parts.insert(anchor + 1 if anchor >= 0 else len(parts), (heading, block))
    return "\n\n".join(block for _, block in parts if block.strip())


def _event_snapshot(cur, event_id: str) -> dict | None:
    """Full history event via an existing cursor, for the revision log."""
    cur.execute(
        """SELECT id, period_label, title_ko, title_en, question_ko, question_en,
                  summary_ko, summary_en, outcome_ko, outcome_en,
                  body_ko, body_en, timeline, sources
             FROM commulingo_history_events WHERE id = %s""",
        (event_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def _apply_event_update(cur, event_id: str, patch: dict) -> None:
    sets: list[str] = []
    params: list = []
    for key, (ko_col, en_col) in (
        ("question", ("question_ko", "question_en")),
        ("summary", ("summary_ko", "summary_en")),
        ("outcome", ("outcome_ko", "outcome_en")),
    ):
        value = patch.get(key)
        if isinstance(value, dict):
            sets += [f"{ko_col} = %s", f"{en_col} = %s"]
            params += [value.get("ko") or "", value.get("en") or ""]
    for key in ("timeline", "sources"):
        if patch.get(key) is not None:
            sets.append(f"{key} = %s::jsonb")
            params.append(json.dumps(patch[key], ensure_ascii=False))
    if not sets:
        return
    sets.append("updated_at = NOW()")
    params.append(event_id)
    cur.execute(
        f"UPDATE commulingo_history_events SET {', '.join(sets)} WHERE id = %s", params
    )


def _apply_event_section(cur, event_id: str, patch: dict, action: str) -> None:
    cur.execute(
        "SELECT body_ko, body_en FROM commulingo_history_events WHERE id = %s",
        (event_id,),
    )
    row = cur.fetchone()
    heading, body = patch["heading"], patch["body"]
    after = patch.get("after") or {}
    cur.execute(
        """UPDATE commulingo_history_events
              SET body_ko = %s, body_en = %s, updated_at = NOW()
            WHERE id = %s""",
        (
            _splice_event_section(row["body_ko"], heading.get("ko", ""),
                                  body.get("ko", ""), after.get("ko", ""), action),
            _splice_event_section(row["body_en"], heading.get("en", ""),
                                  body.get("en", ""), after.get("en", ""), action),
            event_id,
        ),
    )


def apply_edit(cur, target_type: str, action: str, target_id: str, patch: dict, changed_by: str) -> str:
    """Apply a validated edit via an open RealDictCursor. Returns a summary.

    Caller owns the transaction: everything here (including the revision
    snapshot) commits or rolls back together.
    """
    if target_type == "person":
        if action == "create":
            _apply_person_create(cur, target_id, patch)
            snapshot = _person_snapshot(cur, target_id)
            _write_revision(cur, "person", target_id, "create person", snapshot, changed_by)
            return f"created person '{target_id}'"
        if action == "update":
            before = _person_snapshot(cur, target_id)
            _apply_person_update(cur, target_id, patch)
            after = _person_snapshot(cur, target_id)
            _write_revision(cur, "person", target_id, "update person", {"before": before, "after": after}, changed_by)
            return f"updated person '{target_id}' ({', '.join(sorted(patch)) or 'no fields'})"
        before = _person_snapshot(cur, target_id)
        cur.execute("DELETE FROM commulingo_people WHERE id = %s", (target_id,))
        _write_revision(cur, "person", target_id, "delete person", before, changed_by)
        return f"deleted person '{target_id}'"

    if target_type == "term":
        if action == "create":
            _apply_term_create(cur, target_id, patch)
            _write_revision(cur, "term", target_id, "create term", _term_snapshot(cur, target_id), changed_by)
            return f"created term '{target_id}'"
        if action == "update":
            before = _term_snapshot(cur, target_id)
            _apply_term_update(cur, target_id, patch)
            _write_revision(cur, "term", target_id, "update term",
                            {"before": before, "after": _term_snapshot(cur, target_id)}, changed_by)
            return f"updated term '{target_id}' ({', '.join(sorted(patch)) or 'no fields'})"
        before = _term_snapshot(cur, target_id)
        cur.execute("DELETE FROM commulingo_terms WHERE id = %s", (target_id,))
        _write_revision(cur, "term", target_id, "delete term", before, changed_by)
        return f"deleted term '{target_id}'"

    if target_type == "history_event":
        before = _event_snapshot(cur, target_id)
        _apply_event_update(cur, target_id, patch)
        _write_revision(cur, "history_event", target_id, "update event",
                        {"before": before, "after": _event_snapshot(cur, target_id)}, changed_by)
        return f"updated event '{target_id}' ({', '.join(sorted(patch)) or 'no fields'})"

    if target_type == "history_event_section":
        heading = (patch["heading"].get("ko") or patch["heading"].get("en") or "").strip()
        before = _event_snapshot(cur, target_id)
        _apply_event_section(cur, target_id, patch, action)
        _write_revision(cur, "history_event", target_id, f"{action} body section '{heading}'",
                        {"before": before, "after": _event_snapshot(cur, target_id)}, changed_by)
        verb = "added" if action == "create" else "rewrote"
        return f"{verb} body section '{heading}' of event '{target_id}'"

    if target_type == "person_section":
        slug = patch["slug"]
        entity_id = f"{target_id}/{slug}"

        def section_row():
            cur.execute(
                """SELECT slug, sort_order, heading_ko, heading_en, body_ko, body_en, sources
                   FROM commulingo_person_sections WHERE person_id = %s AND slug = %s""",
                (target_id, slug),
            )
            row = cur.fetchone()
            return dict(row) if row else None

        if action == "delete":
            before = section_row()
            cur.execute(
                "DELETE FROM commulingo_person_sections WHERE person_id = %s AND slug = %s",
                (target_id, slug),
            )
            _write_revision(cur, "person_section", entity_id, "delete section", before, changed_by)
            return f"deleted section '{slug}' of '{target_id}'"

        before = section_row()
        heading = patch.get("heading") or {}
        body = patch.get("body") or {}
        if action == "create":
            cur.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_sort FROM commulingo_person_sections WHERE person_id = %s",
                (target_id,),
            )
            next_sort = _section_sort_order(patch, heading, cur.fetchone()["next_sort"])
            cur.execute(
                """INSERT INTO commulingo_person_sections
                      (person_id, slug, sort_order, heading_ko, heading_en,
                       body_ko, body_en, sources, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())""",
                (
                    target_id, slug, next_sort,
                    _localized(heading, "ko"), _localized(heading, "en"),
                    _localized(body, "ko"), _localized(body, "en"),
                    json.dumps(patch.get("sources") or [], ensure_ascii=False),
                ),
            )
            _write_revision(cur, "person_section", entity_id, "create section", section_row(), changed_by)
            return f"created section '{slug}' of '{target_id}'"

        sets, values = [], []
        if "heading" in patch:
            sets += ["heading_ko = %s", "heading_en = %s"]
            values += [_localized(heading, "ko"), _localized(heading, "en")]
        if "body" in patch:
            sets += ["body_ko = %s", "body_en = %s"]
            values += [_localized(body, "ko"), _localized(body, "en")]
        if isinstance(patch.get("sortOrder"), int):
            sets.append("sort_order = %s")
            values.append(patch["sortOrder"])
        if patch.get("sources"):
            sets.append("sources = %s::jsonb")
            values.append(json.dumps(patch["sources"], ensure_ascii=False))
        if sets:
            sets.append("updated_at = NOW()")
            values += [target_id, slug]
            cur.execute(
                f"UPDATE commulingo_person_sections SET {', '.join(sets)} WHERE person_id = %s AND slug = %s",
                values,
            )
        _write_revision(cur, "person_section", entity_id, "update section",
                        {"before": before, "after": section_row()}, changed_by)
        return f"updated section '{slug}' of '{target_id}'"

    if target_type == "history_event_person":
        person_id = patch["personId"]
        entity_id = f"{target_id}/{person_id}"
        cur.execute(
            """SELECT event_id, person_id, sort_order, relation_kind,
                      relation_ko, relation_en, note_ko, note_en
                 FROM commulingo_history_event_people
                WHERE event_id = %s AND person_id = %s""",
            (target_id, person_id),
        )
        row = cur.fetchone()
        before = dict(row) if row else None
        if isinstance(patch.get("sortOrder"), int):
            sort_order = patch["sortOrder"]
        else:
            cur.execute(
                """SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_sort
                     FROM commulingo_history_event_people WHERE event_id = %s""",
                (target_id,),
            )
            sort_order = cur.fetchone()["next_sort"]
        relation = patch["relation"]
        note = patch["note"]
        cur.execute(
            """INSERT INTO commulingo_history_event_people
                      (event_id, person_id, sort_order, relation_kind,
                       relation_ko, relation_en, note_ko, note_en)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                 ON CONFLICT (event_id, person_id) DO UPDATE SET
                     sort_order = EXCLUDED.sort_order,
                     relation_kind = EXCLUDED.relation_kind,
                     relation_ko = EXCLUDED.relation_ko,
                     relation_en = EXCLUDED.relation_en,
                     note_ko = EXCLUDED.note_ko,
                     note_en = EXCLUDED.note_en""",
            (target_id, person_id, sort_order, patch["relationKind"],
             relation["ko"], relation["en"], note["ko"], note["en"]),
        )
        after = {**patch, "eventId": target_id, "sortOrder": sort_order}
        _write_revision(cur, "history_event_person", entity_id,
                        "upsert history event person", {"before": before, "after": after}, changed_by)
        return f"linked person '{person_id}' to history event '{target_id}'"

    if action == "create":
        before = _office_snapshot(cur, target_id)
        row_id = _apply_office_row_create(cur, target_id, patch)
        after = _office_snapshot(cur, target_id)
        _write_revision(cur, "office", target_id, "create office row", {"before": before, "after": after}, changed_by)
        return f"created office row #{row_id} in '{target_id}'"

    row_id = int(target_id)
    cur.execute("SELECT office_id FROM commulingo_office_rows WHERE id = %s", (row_id,))
    office_id = cur.fetchone()["office_id"]
    before = _office_snapshot(cur, office_id)
    if action == "update":
        _apply_office_row_update(cur, row_id, patch)
        after = _office_snapshot(cur, office_id)
        _write_revision(cur, "office", office_id, "update office row", {"before": before, "after": after}, changed_by)
        return f"updated office row #{row_id} in '{office_id}'"
    cur.execute("DELETE FROM commulingo_office_rows WHERE id = %s", (row_id,))
    after = _office_snapshot(cur, office_id)
    _write_revision(cur, "office", office_id, "delete office row", {"before": before, "after": after}, changed_by)
    return f"deleted office row #{row_id} from '{office_id}'"


def _record_suggestion(cur, target_type, action, target_id, patch, sources, confidence,
                       status: str, reviewer: str = "", review_note: str = "") -> int:
    cur.execute(
        """INSERT INTO commulingo_agent_suggestions
              (target_type, target_id, action, patch_json, source_refs, confidence,
               suggested_by, status, reviewer, review_note, reviewed_at)
           VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s,
                   CASE WHEN %s = 'pending' THEN NULL ELSE NOW() END)
           RETURNING id""",
        (
            target_type, target_id, action,
            json.dumps(patch, ensure_ascii=False),
            json.dumps(sources, ensure_ascii=False),
            confidence, _SUGGESTED_BY, status, reviewer, review_note, status,
        ),
    )
    return cur.fetchone()["id"]


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
    if target_type == "person_section" and not patch.get("sources"):
        # Section rows carry their own sources column; reuse the tool-level
        # citations so they survive into the rendered detail page data.
        patch = {**patch, "sources": sources}
    direct = direct_apply_enabled()
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            error = _validate(cur, target_type, action, target_id, patch)
            if error:
                return error
            if direct:
                summary = apply_edit(cur, target_type, action, target_id, patch, _SUGGESTED_BY)
                sid = _record_suggestion(
                    cur, target_type, action, target_id, patch, sources, confidence,
                    status="approved", reviewer="auto:direct_apply",
                    review_note="applied directly (direct_apply mode)",
                )
                return (
                    f"OK — applied: {summary}. Logged as edit #{sid}. The change is live "
                    f"on {_public_page(target_type, target_id)} within ~1 minute "
                    "(server cache TTL)."
                )
            cur.execute(
                """SELECT id FROM commulingo_agent_suggestions
                   WHERE target_type = %s AND target_id = %s AND action = %s AND status = 'pending'
                   ORDER BY created_at DESC LIMIT 1""",
                (target_type, target_id, action),
            )
            pending = cur.fetchone()
            sid = _record_suggestion(
                cur, target_type, action, target_id, patch, sources, confidence, status="pending"
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
    "Hard ceiling {ko} Korean / {en} English characters — write to the prescribed {target},"
    " not to this number, but COUNT YOUR DRAFT AGAINST IT BEFORE CALLING: a rejected"
    " write costs a full round. Count the ENGLISH side first — it is the side that"
    " actually overruns (every length rejection on 2026-08-03/04 was an English field"
    " landing 3-9% over). An English draft within ~50 characters of its ceiling loses"
    " its weakest clause before the call, not after the reject."
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
    " One clause. A characterization that needs a second clause after a dash"
    " belongs in the bio.",
)

_BIO_SCHEMA = _capped_bilingual_schema(
    "bio",
    " Write to the tier's sentence count (the commissioned task states it);"
    " keep career chronology in career rows, not the bio."
    f" A dense sentence of this register costs ~{DENSE_SENTENCE_CHARS[0]} Korean"
    f" and ~{DENSE_SENTENCE_CHARS[1]} English characters, so the ceiling pays for"
    f" {sentence_budget('bio')} of them. Over by about that much or more: cut a"
    " sentence rather than squeezing every clause. Over by only a little: delete"
    " the weakest clause — do not reword at the same length.",
)

_MOMENT_SCHEMA = _capped_bilingual_schema(
    "moment",
    " This is the pull-quote on the person LIST card, so it is budgeted in"
    " rendered lines: 44-85 Korean characters is 2 lines, 86-127 is 3."
    " A quotation too long to fit is excerpted to its sharpest clause with '…',"
    " or traded for a shorter one — never padded out to the ceiling.",
)

_EVENT_NOTE_SCHEMA = _capped_bilingual_schema(
    "event_note",
    " The note is a caption under the person's name on the event page, stating what"
    f" the person did in the event: {sentence_prescription('event_note')}.",
)

# The section body is markdown and may carry paragraphs, so its guidance is a
# character target rather than a sentence count. Both numbers come from
# SECTION_BODY_TARGET; the ceiling in _CEILING comes from FIELD_LIMITS.
_SECTION_BODY_SCHEMA = _capped_bilingual_schema(
    "section_body",
    f" One topic, {SECTION_BODY_TARGET[0]}-{SECTION_BODY_TARGET[1]} Korean characters"
    " plus the equivalent English. The ceiling above is a runaway guard: a body"
    " approaching it is a section that should have been two, or a topic the bio"
    " already covers.",
    target="length",
)

# Fate label = cause of death only, NO death year (it renders from `years`).
# Execution → 처형/Executed; vague natural death → 자연사/Natural causes; keep a
# specific illness word (심장마비/폐암…); place with " · " (암살 · 멕시코). A
# deposed/exile fate keeps its EVENT year (실각 1964) and may append the cause
# (실각 1964 · 자연사). The death year is stripped automatically on save.
_FATE_LABEL_SCHEMA = _capped_bilingual_schema(
    "fate_label",
    " Cause of death only, no death year (실각 1964 · 자연사 / Removed 1964 · "
    "natural causes). Execution=처형/Executed; natural=자연사/Natural causes; "
    "place with ' · '. The death year is dropped automatically on save.",
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

_NATIONAL_ORIGIN_SCHEMA = {
    **_NATIONALITY_SCHEMA,
    "description": (
        "National or ethnic background, not birthplace and not place of death. "
        "For example Radek=Poland although born in present-day Ukraine; "
        "Yezhov=Russia although born in Lithuania (an ethnic Russian). The "
        "other direction matters just as much: a Soviet official OF a "
        "non-Russian nationality keeps that nation (Sillari=estonia, "
        "Gumbaridze=georgia), never a blanket 'russia'."
    ),
}

_COMMULINGO_FIELD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "description": (
        "Canonical patch. Scalar fields stay strings; bilingual fields are {ko,en}. "
        "For person create include givenName/familyName (or legacy name), bio, epithet, "
        "groupId, role, years, aliases, career, and native-script name fields. "
        "Empty object is only for delete."
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
            "Native-script name line (the column name is legacy): the person's name in "
            "THEIR OWN script per citizenship — 김무정, 彭德怀, 'Võ Nguyên Giáp', "
            "'Kádár János', 'Владимир Ленин'. Never a Russian transliteration of a "
            "non-Russian name, and never omitted because the right script is not "
            "Cyrillic: for Latin-script nations this line carries the native Latin "
            "spelling with its diacritics."
        )},
        "cyrillicPatronymic": {"type": "string"},
        "years": {"type": "string", "description": "Display range, e.g. 1878–1943."},
        "name": _BILINGUAL_TEXT_SCHEMA,
        "givenName": _BILINGUAL_TEXT_SCHEMA,
        "familyName": _BILINGUAL_TEXT_SCHEMA,
        "epithet": _EPITHET_SCHEMA,
        "bio": _BIO_SCHEMA,
        "moment": _MOMENT_SCHEMA,
        "patronymic": _BILINGUAL_TEXT_SCHEMA,
        "citizenship": _NATIONALITY_SCHEMA,
        "origin": _NATIONAL_ORIGIN_SCHEMA,
        "nationalOrigin": _NATIONAL_ORIGIN_SCHEMA,
        "term": _BILINGUAL_TEXT_SCHEMA,
        "original": {"type": "string", "description": "term: native-script/original-language form (ГУЛАГ, нэпман)."},
        "period": {
            "type": "object", "additionalProperties": False,
            "properties": {"ko": {"type": "string"}, "en": {"type": "string"}},
            "description": (
                "term: bilingual period label, e.g. {\"ko\": \"1930–1960\", \"en\": "
                "\"1930–1960\"} or {\"ko\": \"1980년대–현재\", \"en\": \"1980s–present\"}. "
                "Use {\"ko\": \"개념\", \"en\": \"Concept\"} for an undated concept. "
                "One shared string leaks Korean onto the English page."
            ),
        },
        "startYear": {
            "type": ["integer", "null"],
            "description": (
                "term: first year of the period, for chronological sorting. Required "
                "whenever the label names a year; null only for undated concepts. "
                "A decade label resolves to the decade start (1980년대 -> 1980), a "
                "century label to the century start (19세기 -> 1800)."
            ),
        },
        "endYear": {
            "type": ["integer", "null"],
            "description": "term: last year of the period; null if still current or undated.",
        },
        "category": {
            "type": "string", "enum": list(_TERM_CATEGORIES),
            "description": f"term: which glossary group it belongs to. One of {_TERM_CATEGORY_HINT}.",
        },
        "parentId": {
            "type": ["string", "null"],
            "description": (
                "term: the id of the entry this one is a PART of, so it nests under it "
                "(예조프시나 -> great-purge). Only for a component of the parent, never for "
                "a merely adjacent concept — those belong in the flat related-terms list. "
                "The glossary nests one level: a parent may not itself have a parent. "
                "null detaches an entry."
            ),
        },
        "definition": _capped_bilingual_schema(
            "definition",
            " term: the card paragraph (2-3 sentences); depth goes to body (markdown)."
            f" A dense sentence costs ~{DENSE_SENTENCE_CHARS[0]} Korean characters, so a"
            " third sentence only fits when the first two stay tight — move the"
            " qualifications and the historiography to body.",
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
                "properties": {"y": {"type": "string"}, "r": _BILINGUAL_TEXT_SCHEMA},
                "required": ["y", "r"],
            },
        },
        # Exactly one key, enforced in the schema rather than only at the write
        # boundary: "role takes exactly one of officeId or category, not both"
        # was 41 rejected person_create calls, and min/maxProperties stops those
        # before the call is spent. null (to clear the role) still validates
        # because the property constraints only apply to the object form.
        "role": {
            "type": ["object", "null"], "additionalProperties": False,
            "minProperties": 1, "maxProperties": 1,
            "description": (
                "Exactly ONE of officeId or category (categoryId is an alias for "
                "category) — never both, never neither. Valid category ids come from "
                "commulingo_people(action='list_categories'), office ids from "
                "action='list_offices'. null clears the role."
            ),
            "properties": {"officeId": {"type": "string"}, "category": {"type": "string"}, "categoryId": {"type": "string"}},
        },
        "fate": {
            "type": "object", "additionalProperties": False,
            "properties": {
                # Enumerated here so the closed vocabulary is readable up front;
                # it was only discoverable by tripping the write-boundary check.
                "kind": {"type": "string", "enum": list(_FATE_KINDS)},
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
    if "not found" in text or "does not exist" in text or "unknown group" in text or "unknown role" in text:
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


def _narrow_fields_schema(keys: tuple[str, ...], *, required: tuple[str, ...] = ()) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {key: _COMMULINGO_FIELD_SCHEMA["properties"][key] for key in keys},
        **({"required": list(required)} if required else {}),
    }


_PERSON_NARROW_KEYS = (
    "groupId", "sortOrder", "cyrillic", "cyrillicPatronymic", "years",
    "givenName", "familyName", "epithet", "bio", "moment", "patronymic",
    "citizenship", "nationalOrigin", "aliases", "career", "role", "fate", "scenes",
)
_TERM_NARROW_KEYS = (
    "sortOrder", "term", "original", "period", "startYear", "endYear",
    "category", "definition", "body", "aliases", "people", "events", "parentId",
)
# Create fills the card's source list from the top-level citations. An update
# is usually partial, so it must leave the stored list alone unless the caller
# means to replace it — hence 'sources' is writable only on update.
_TERM_UPDATE_NARROW_KEYS = _TERM_NARROW_KEYS + ("sources",)
_OFFICE_ROW_NARROW_KEYS = (
    "sortOrder", "years", "body", "personId", "name", "note",
)

_CITATIONS_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {"type": "string"},
    "description": "Source URL/reference plus the fact it supports. Never put citations inside fields.",
}


def _person_write_tool(name: str, action: str) -> dict:
    required_fields = (
        "groupId", "epithet", "bio", "career", "role", "citizenship", "nationalOrigin",
    ) if action == "create" else ()
    return {
        "name": name,
        "description": (
            f"{action.title()} one CommuLingo person card. This tool accepts person fields only; "
            "citations are a separate top-level argument. Public text is bilingual {ko,en}. "
            "Read the record and reference lists first. On create, citizenship and "
            "nationalOrigin are both mandatory; nationalOrigin may equal citizenship. "
            "nationalOrigin means national/ethnic "
            "background, never birthplace. Russian-style names must research and include a "
            "complete patronymic {ko,en} plus cyrillicPatronymic; omitted PATCH subfields are preserved. "
            "A successful call ends the run."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "person_id": {"type": "string", "description": "Existing id or new lowercase kebab-case slug."},
                "fields": _narrow_fields_schema(_PERSON_NARROW_KEYS, required=required_fields),
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
    "description": "Create or update one bilingual long-form person section. Citations are stored on the section.",
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["create", "update"]},
            "person_id": {"type": "string"},
            "slug": {"type": "string"},
            "heading": _BILINGUAL_TEXT_SCHEMA,
            "body": _SECTION_BODY_SCHEMA,
            "sort_order": {"type": ["integer", "null"], "description": (
                "Chronological key: YYYYMM of the period the section opens on, with MM=00 "
                "when only the year is known (1898 -> 189800, 1991-08 -> 199108). Sections "
                "render in this order, so a life story reads front to back no matter which "
                "one was written first. Omit or null only when the section has no period at "
                "all; the year in the heading is then used, and a bare append is the last "
                "resort."
            )},
            "citations": _CITATIONS_SCHEMA,
        },
        "required": ["action", "person_id", "slug", "heading", "body", "citations"],
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
            "fields": _narrow_fields_schema(_OFFICE_ROW_NARROW_KEYS),
            "citations": _CITATIONS_SCHEMA,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["action", "target_id", "fields", "citations"],
    },
}

_EVENT_QUESTION_SCHEMA = _capped_bilingual_schema(
    "event_question",
    extra=(
        " The question the page answers, written as a question. It is the line under"
        " the title, not a summary."
    ),
)
_EVENT_SUMMARY_SCHEMA = _capped_bilingual_schema(
    "event_summary",
    extra=" What happened, for a reader who knows nothing about the period yet.",
    target="short paragraph",
)
_EVENT_OUTCOME_SCHEMA = _capped_bilingual_schema(
    "event_outcome",
    extra=" What it left behind — the consequences the rest of the site refers back to.",
    target="short paragraph",
)
_EVENT_SECTION_BODY_SCHEMA = _capped_bilingual_schema(
    "event_section_body",
    extra=(
        f" Markdown. Write to {EVENT_SECTION_TARGET[0]}-{EVENT_SECTION_TARGET[1]} Korean"
        " characters: one part of the story, told with the dates, names and figures a"
        " reader would otherwise have to look up. No `#` or `##` headings inside the"
        " body — the heading argument supplies the only one."
    ),
    target=f"{EVENT_SECTION_TARGET[0]}-{EVENT_SECTION_TARGET[1]} Korean characters",
)

COMMULINGO_EVENT_UPDATE_TOOL = {
    "name": "commulingo_event_update",
    "description": (
        "Update one history event's card fields: the question it answers, the summary, "
        "the outcome, the timeline, the source list. The long-form body is NOT written "
        "here — use commulingo_event_section_save for that. Read the event with "
        "commulingo_people(action='get_event') first. Sending timeline or sources "
        "replaces the stored list whole, so include the entries that are already there. "
        "A successful call ends the run."
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
                            "The whole timeline, in chronological order. Every entry the "
                            "event already has plus the ones being added — an omitted "
                            "entry is a deleted entry."
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
                            "The whole source list: author, title, publisher, year for a "
                            "book; a URL for a document. Replaces the stored list."
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
        "Write one `## ` section of a history event's long-form body. The body is built "
        "one section per run, the way a person's detail sections are, so this call sends "
        "ONLY the new part — never the whole body. 'create' adds a section, 'update' "
        "rewrites the one whose heading matches. The heading must be a real heading, not "
        "a label: '무기보다 먼저 온 것들' tells a reader what the part is about, "
        "'배경 3' does not. Read the event with commulingo_people(action='get_event') "
        "first to see which sections exist. A successful call ends the run."
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
                    "On create: the existing heading this section goes after, per "
                    "language. Omit to append at the end, which is right whenever the "
                    "body is being written front to back."
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
                    _TERM_NARROW_KEYS if creating else _TERM_UPDATE_NARROW_KEYS,
                    required=("term", "definition", "aliases", "period", "category") if creating else (),
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


async def _exec_commulingo_section_save(
    action: str, person_id: str, slug: str, heading: dict, body: dict,
    citations: list, sort_order: int | None = None,
) -> str:
    fields = {"slug": slug, "heading": heading, "body": body, "sources": citations}
    if sort_order is not None:
        fields["sortOrder"] = sort_order
    return await _exec_commulingo_write("person_section", action, person_id, citations, fields, None)


async def _exec_commulingo_event_link(
    event_id: str, person_id: str, relation_kind: str, relation: dict, note: dict,
    citations: list, sort_order: int | None = None,
) -> str:
    fields = {
        "personId": person_id, "relationKind": relation_kind,
        "relation": relation, "note": note,
    }
    if sort_order is not None:
        fields["sortOrder"] = sort_order
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
        "File what the event text you just wrote needed and the site does not have: a "
        "person with no card, a term with no glossary entry, a document worth publishing "
        "in full. The people and glossary lanes work from this queue, so a gap filed here "
        "becomes a card, and the link in your section resolves. File only what the "
        "narrative actually leans on — a name mentioned once in passing is not a gap. "
        "Reading tools first: a card that already exists is not a gap unless it is too "
        "thin to carry the weight the event puts on it, in which case send its target_id. "
        "A 'person' gap is for HISTORICAL ACTORS of the events — file a scholar or "
        "journalist only when the section's historiography discussion treats that person "
        "as a subject in their own right, never merely because you cited their book. "
        "A 'doc' gap has a HIGHER bar than the others: the reference library republishes "
        "PRIMARY sources in full — a decree, treaty, speech, resolution, letter, or short "
        "set of minutes a reader should be able to read whole. Never file the scholarship "
        "you researched from (monographs, journal articles, memoirs, press pieces — those "
        "belong in your sources and nowhere else), never a book-length record (trial "
        "transcripts, plenum stenograms, commission reports), and never a document the "
        "library already carries. On 2026-08-17 the queue held 175 doc rows and a third "
        "were copyrighted secondary literature, one was a novel; every one of those rows "
        "wasted a human triage pass."
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
                                "The existing entry id when the gap is 'this card is too "
                                "thin'. Null when nothing exists yet."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "What the event narrative needs from it. This is the brief "
                                "the next lane writes from, so name the specific role, not "
                                "'important figure'."
                            ),
                        },
                        "priority": {
                            "type": "integer", "minimum": 0, "maximum": 10,
                            "description": (
                                "How load-bearing it is in the section just written: 8-10 "
                                "the reader cannot follow the section without it, 4-7 it "
                                "deepens the section, 0-3 it is adjacent."
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


_PAREN_RE = re.compile(r"[（(][^）)]*[)）]")


def _label_variants(label: str) -> list[str]:
    """The forms of a filed label worth looking up, most specific first.

    Labels arrive as the event text writes them, and the dictionaries store
    headwords. The two disagree in small ways that an exact match cannot see:
    「소브나르호스(국민경제회의)」 is the entry `sovnarkhoz` plus its own alias in
    brackets, 「식량 배급제 (프로드라즈베르스트카)」 is `prodrazvyorstka` behind a
    gloss, 「국가비상사태위원회 (GKChP)」 is `gkchp` with the acronym spelled out.
    Eleven of the twenty-two glossary gaps dismissed on 2026-08-09 were of this
    shape, and each cost a full curator run to look up and reject.

    So: the label itself, the label without its bracket, and the bracket's own
    contents (often the registered alias), each also without spaces.
    """
    forms: list[str] = []
    raw = (label or "").strip()
    if not raw:
        return forms
    candidates = [raw, _PAREN_RE.sub(" ", raw).strip()]
    candidates += [inner.strip() for inner in re.findall(r"[（(]([^）)]*)[)）]", raw)]
    for candidate in candidates:
        for form in (candidate, candidate.replace(" ", "")):
            if form and form not in forms:
                forms.append(form)
    return forms


_DOC_MANIFEST = os.path.join(
    os.getenv("FRONTEND_DIR", "/home/grass/frontend"),
    "data", "commulingo", "docs", "manifest.json",
)
# Document names are dressed in title marks that the same title wears
# inconsistently: the manifest lists 「대전환의 해」 and 대전환의 해 as separate
# aliases of one entry, and a curator writing a gap picks whichever its sentence
# used. The marks carry no identity, so neither side keeps them.
_DOC_TITLE_MARKS = dict.fromkeys(map(ord, "『』「」《》〈〉<>\"'“”‘’ "))
_doc_index_cache: tuple[float, dict] = (0.0, {})


def _doc_key(value: str) -> str:
    return (value or "").strip().lower().translate(_DOC_TITLE_MARKS)


def _doc_index() -> dict:
    """Every name a published reference document answers to, mapped to its id.

    Documents are the one dictionary that does not live in a table: they are
    files listed in the frontend manifest. So the coverage check below had
    nothing to ask and let every document gap through, and the queue collected
    requests for documents already on the site — four of the thirty published by
    2026-08-09 were sitting in it a second time.

    Re-read when the file's mtime moves, which is exactly when a document is
    published. A manifest that is missing or malformed yields no index, and the
    gap is filed as before rather than lost.
    """
    global _doc_index_cache
    try:
        stamp = os.path.getmtime(_DOC_MANIFEST)
    except OSError:
        return {}
    cached_stamp, cached = _doc_index_cache
    if cached and stamp == cached_stamp:
        return cached
    try:
        with open(_DOC_MANIFEST, encoding="utf-8") as handle:
            docs = json.load(handle).get("docs") or []
    except (OSError, ValueError):
        logger.warning("commulingo doc manifest unreadable at %s", _DOC_MANIFEST)
        return {}
    index: dict = {}
    for doc in docs:
        doc_id = str(doc.get("id") or "").strip()
        if not doc_id:
            continue
        names = [doc_id]
        for field in ("title", "aliases"):
            block = doc.get(field) or {}
            for lang in ("ko", "en"):
                value = block.get(lang)
                names += value if isinstance(value, list) else [value or ""]
        for name in names:
            key = _doc_key(str(name))
            if key:
                index.setdefault(key, doc_id)
    _doc_index_cache = (stamp, index)
    return index


def registered_event_labels() -> list[str]:
    """Event-dictionary titles as 'ko (en)', for prompts that must not re-register one.

    Lives here rather than in the glossary lane because two lanes need it and
    that module cannot be imported to borrow it: it sets COMMULINGO_SUGGESTED_BY
    at import time, so importing it from another lane would relabel that lane's
    writes.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT title_ko, title_en FROM commulingo_history_events")
            rows = cur.fetchall() or []
    labels = []
    for row in rows:
        ko = str(row.get("title_ko") or "").strip()
        en = str(row.get("title_en") or "").strip()
        labels.append(f"{ko} ({en})" if ko and en else ko or en)
    return sorted(label for label in labels if label)


def _already_covered(cur, kind: str, label: dict, target_id: str) -> str:
    """The id of an existing entry this gap is asking for, or ''.

    The event curator files gaps from what its own text leans on, and a good
    section leans on plenty the dictionaries already have — the first day of the
    lane queued 블라디미르 레닌 as a missing person. Each of those costs a full
    curator run to look up and dismiss, so they are matched here, where the check
    is one indexed query.
    """
    if target_id:
        return ""  # an explicit target means "this exists but is too thin"
    ko = (label.get("ko") or "").strip()
    en = (label.get("en") or "").strip()
    if not (ko or en):
        return ""
    if kind == "person":
        sql = """SELECT p.id FROM commulingo_people p
                  WHERE replace(lower(p.name_ko), ' ', '') = replace(lower(%(v)s), ' ', '')
                     OR replace(lower(p.name_en), ' ', '') = replace(lower(%(v)s), ' ', '')
                  UNION
                 SELECT a.person_id FROM commulingo_person_aliases a
                  WHERE replace(lower(a.alias), ' ', '') = replace(lower(%(v)s), ' ', '')
                  LIMIT 1"""
    elif kind == "term":
        sql = """SELECT t.id FROM commulingo_terms t
                  WHERE replace(lower(t.term_ko), ' ', '') = replace(lower(%(v)s), ' ', '')
                     OR replace(lower(t.term_en), ' ', '') = replace(lower(%(v)s), ' ', '')
                  UNION
                 SELECT a.term_id FROM commulingo_term_aliases a
                  WHERE replace(lower(a.alias), ' ', '') = replace(lower(%(v)s), ' ', '')
                  LIMIT 1"""
    elif kind == "doc":
        sql = ""
    else:
        return ""

    # A history event is not a glossary term. The event curator files its gaps
    # from what its own prose calls things, so an event it narrates comes back as
    # a missing concept: on 2026-08-09 「홀로도모르와 집단화 기근 (1932~1933)」 and
    # 「소련-일본 국경 전쟁과 중립조약 (1938~1941)」 were registered as terms beside
    # the events of the same name, and the reader got the same account twice in
    # two dictionaries. The events dictionary is part of what "the site already
    # covers this" means, so it is probed here too. The answer is returned as
    # 'event:<id>' because the id names a row in another table, and callers write
    # it into a resolution note rather than dereferencing it as a term.
    event_sql = "" if kind != "term" else """
        SELECT e.id FROM commulingo_history_events e
         WHERE replace(lower(e.title_ko), ' ', '') = replace(lower(%(v)s), ' ', '')
            OR replace(lower(e.title_en), ' ', '') = replace(lower(%(v)s), ' ', '')
         LIMIT 1"""

    if kind == "doc":
        index = _doc_index()

        def probe(value: str) -> str:
            return index.get(_doc_key(value), "")
    else:
        def probe(value: str) -> str:
            cur.execute(sql, {"v": value})
            row = cur.fetchone()
            if row:
                return str(row["id"])
            if not event_sql:
                return ""
            cur.execute(event_sql, {"v": value})
            row = cur.fetchone()
            return f"event:{row['id']}" if row else ""

    # Try the whole label in either language first; only then the stripped forms,
    # so an entry that matches outright always wins over one reached by peeling a
    # bracket off. A hit from a stripped form has to agree across languages when
    # both are given, because that is where a disambiguating bracket could
    # otherwise collapse two entries into one (인민전선 (소련 말기) and the 1930s
    # Popular Front are not the same entry).
    for whole in (v for v in (ko, en) if v):
        found = probe(whole)
        if found:
            return found
    hits = {}
    for lang, raw in (("ko", ko), ("en", en)):
        for form in _label_variants(raw)[1:]:  # [0] is the whole label, tried above
            found = probe(form)
            if found:
                hits[lang] = found
                break
    if len(hits) == 2:
        return hits["ko"] if hits["ko"] == hits["en"] else ""
    if len(hits) == 1 and not (ko and en):
        return next(iter(hits.values()))
    if len(hits) == 1:
        # One language resolved and the other said nothing. Accept it only when
        # the silent side has no headword of its own to contradict with.
        lang, found = next(iter(hits.items()))
        other = en if lang == "ko" else ko
        return "" if probe(other) else found
    return ""


def _file_gaps(gaps: list, event_id: str) -> dict:
    filed, duplicates, covered = [], [], []
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for gap in gaps:
                label = gap.get("label") or {}
                existing = _already_covered(
                    cur, gap["kind"], label, (gap.get("target_id") or "").strip()
                )
                if existing:
                    covered.append(f"{label.get('ko') or label.get('en')} -> {existing}")
                    continue
                cur.execute(
                    """INSERT INTO commulingo_curation_gaps
                          (kind, event_id, target_id, label_ko, label_en, reason,
                           priority, created_by)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT DO NOTHING
                       RETURNING id""",
                    (
                        gap["kind"], event_id, (gap.get("target_id") or "").strip(),
                        (label.get("ko") or "").strip(), (label.get("en") or "").strip(),
                        (gap.get("reason") or "").strip(), int(gap.get("priority") or 0),
                        _SUGGESTED_BY,
                    ),
                )
                row = cur.fetchone()
                # The partial unique index makes a re-file a no-op, which is the
                # point: two sections of one event needing the same person must
                # not queue that person twice.
                (filed if row else duplicates).append(label.get("ko") or label.get("en") or "?")
    return {"filed": filed, "already_queued": duplicates, "already_covered": covered}


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
