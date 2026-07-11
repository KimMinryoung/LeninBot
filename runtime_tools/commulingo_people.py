"""runtime_tools.commulingo_people — CommuLingo 인물 사전 read + edit tools.

The people dictionary at cyber-lenin.com/commulingo/people is DB-backed
(commulingo_* tables in the main Postgres; see
frontend/dev_docs/commulingo_people_handoff.md).

Two tools:

- commulingo_people — read/search the dictionary + edit history/queue status
- commulingo_edit   — propose or apply an edit (sources required)

commulingo_edit runs in one of two modes, switched by
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

from psycopg2.extras import RealDictCursor

from db import query as db_query, query_one as db_query_one, get_conn

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,120}$")

_SUGGESTED_BY = os.getenv("COMMULINGO_SUGGESTED_BY", "cyber-lenin").strip() or "cyber-lenin"

_TARGET_TYPES = ("person", "office_row", "person_section")
_ACTIONS = ("create", "update", "delete")

_FATE_KINDS = (
    "executed", "assassinated", "murdered", "killed",
    "deposed", "exile", "natural",
)

_PERSON_PATCH_KEYS = frozenset({
    "id", "group", "groupId", "sortOrder", "initial", "cyrillic", "years",
    "name", "epithet", "bio", "moment", "fate", "patronymic", "cyrillicPatronymic",
    "aliases", "scenes", "career", "role",
    "office_rows", "sections",  # read-only echoes from get_person; tolerated and ignored
})

# person patch fields that must be {ko, en} objects. Plain strings are
# rejected outright: _localized() would store them as Korean-only and
# silently blank the English side (this happened in production).
_LOCALIZED_PERSON_KEYS = ("name", "epithet", "bio", "moment", "patronymic")
_LOCALIZED_OFFICE_ROW_KEYS = ("body", "name", "note")

_OFFICE_ROW_PATCH_KEYS = frozenset({
    "sortOrder", "years", "period", "body", "personId", "name", "note",
})

# Long-form detail sections rendered on /commulingo/people/<id>.
_SECTION_PATCH_KEYS = frozenset({"slug", "heading", "body", "sortOrder", "sources"})
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,60}$")

# ── Mode switch (config/commulingo_people.json, mtime-cached) ─────────

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "commulingo_people.json",
)
_config_cache: dict | None = None
_config_mtime: float = -1.0


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
    return db_query(
        """SELECT g.id, g.range_label, g.title_ko, g.title_en,
                  COUNT(p.id) AS people_count
           FROM commulingo_people_groups g
           LEFT JOIN commulingo_people p ON p.group_id = g.id
           GROUP BY g.id, g.sort_order, g.range_label, g.title_ko, g.title_en
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

    Returned in the SAME shape commulingo_edit's patch accepts, so agents can
    read a record, modify fields, and send them straight back.
    """
    cur.execute(
        """SELECT id, group_id, initial, cyrillic, years_label,
                  name_ko, name_en, epithet_ko, epithet_en, bio_ko, bio_en,
                  moment_ko, moment_en,
                  fate_kind, fate_label_ko, fate_label_en
           FROM commulingo_people WHERE id = %s""",
        (person_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    person = {
        "id": row["id"],
        "group": row["group_id"],
        "initial": row["initial"],
        "cyrillic": row["cyrillic"],
        "years": row["years_label"],
        "name": {"ko": row["name_ko"], "en": row["name_en"]},
        "epithet": {"ko": row["epithet_ko"], "en": row["epithet_en"]},
        "bio": {"ko": row["bio_ko"], "en": row["bio_en"]},
        "moment": {"ko": row["moment_ko"], "en": row["moment_en"]},
        "fate": {
            "kind": row["fate_kind"],
            "label": {"ko": row["fate_label_ko"], "en": row["fate_label_en"]},
        },
    }
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
        "`list_groups` (era groups + people counts), "
        "`search_people` (q matches id/name/cyrillic; optional group_id), "
        "`get_person` (full record — returned in the exact patch shape "
        "commulingo_edit accepts, so you can edit fields and send them back; "
        "office_rows, sections and role.resolvedIcon are read-only info), "
        "`list_offices` (institution timelines + row counts), "
        "`get_office` (one institution's full leadership timeline), "
        "`list_categories` (office-less role categories for role {category}), "
        "`get_sections` (a person's full detail-page sections, returned in the "
        "exact person_section patch shape — edit and send back), "
        "`list_suggestions` (edit history/queue from commulingo_edit; "
        "optional status filter: pending/approved/rejected/superseded). "
        "Always read the current record before editing with commulingo_edit."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_groups", "search_people", "get_person",
                    "list_offices", "get_office", "list_categories",
                    "get_sections", "list_suggestions",
                ],
            },
            "q": {
                "type": "string",
                "description": "search_people: substring matched against id/name/cyrillic.",
            },
            "group_id": {
                "type": "string",
                "description": "search_people: restrict to one group id.",
            },
            "person_id": {"type": "string", "description": "get_person/get_sections: person id."},
            "office_id": {"type": "string", "description": "get_office: office id."},
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
    status: str = "",
    limit: int = 30,
) -> str:
    try:
        limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit = 30
    try:
        if action == "list_groups":
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
        return f"Error: {type(e).__name__}: {e}"


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


def _parse_life_years(label: str) -> tuple[int | None, int | None]:
    m = re.match(r"^(\d{3,4})[–-](\d{3,4})$", label or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


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


def _replace_patronymic(cur, person_id: str, patch: dict):
    cur.execute("DELETE FROM commulingo_person_patronymics WHERE person_id = %s", (person_id,))
    ko = _localized(patch.get("patronymic"), "ko")
    en = _localized(patch.get("patronymic"), "en")
    cyr = patch.get("cyrillicPatronymic") or ""
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
    for lang in ("ko", "en"):
        values = aliases.get(lang) if isinstance(aliases, dict) else None
        for index, alias in enumerate(values or []):
            alias = alias.strip() if isinstance(alias, str) else ""
            if not alias:
                continue
            cur.execute(
                """INSERT INTO commulingo_person_aliases (person_id, lang, alias, sort_order)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (person_id, lang, alias)
                   DO UPDATE SET sort_order = EXCLUDED.sort_order""",
                (person_id, lang, alias, index),
            )


def _replace_scenes(cur, person_id: str, scenes: list):
    cur.execute("DELETE FROM commulingo_person_scenes WHERE person_id = %s", (person_id,))
    for index, scene in enumerate(scenes or []):
        if not isinstance(scene, (list, tuple)) or len(scene) < 2:
            continue
        collection_id = scene[0].strip() if isinstance(scene[0], str) else ""
        episode_id = scene[1].strip() if isinstance(scene[1], str) else ""
        if not collection_id or not episode_id:
            continue
        cur.execute(
            """INSERT INTO commulingo_person_scenes
                  (person_id, collection_id, episode_id, sort_order)
               VALUES (%s, %s, %s, %s)""",
            (person_id, collection_id, episode_id, index),
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
    for index, entry in enumerate(career or []):
        if not isinstance(entry, dict):
            continue
        label = entry.get("y") or entry.get("period") or ""
        sy, sm, ey, em = _period_columns(label)
        role = entry.get("r") or entry.get("role") or {}
        cur.execute(
            """INSERT INTO commulingo_person_career_entries
                  (person_id, sort_order, period_label, start_year, start_month,
                   end_year, end_month, role_ko, role_en, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
            (person_id, index, label, sy, sm, ey, em,
             _localized(role, "ko"), _localized(role, "en")),
        )


def _apply_person_create(cur, person_id: str, patch: dict) -> None:
    cur.execute("SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_sort FROM commulingo_people")
    next_sort = cur.fetchone()["next_sort"]
    sort_order = patch["sortOrder"] if isinstance(patch.get("sortOrder"), int) else next_sort
    birth, death = _parse_life_years(patch.get("years") or "")
    name = patch.get("name") or {}
    fate = patch.get("fate") or {}
    cur.execute(
        """INSERT INTO commulingo_people
              (id, group_id, sort_order, initial, cyrillic, years_label, birth_year, death_year,
               name_ko, name_en, epithet_ko, epithet_en, bio_ko, bio_en,
               moment_ko, moment_en,
               fate_kind, fate_label_ko, fate_label_en, updated_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
        (
            person_id,
            patch.get("groupId") or patch.get("group"),
            sort_order,
            patch.get("initial") or "",
            patch.get("cyrillic") or "",
            patch.get("years") or "",
            birth, death,
            _localized(name, "ko"), _localized(name, "en"),
            _localized(patch.get("epithet"), "ko"), _localized(patch.get("epithet"), "en"),
            _localized(patch.get("bio"), "ko"), _localized(patch.get("bio"), "en"),
            _localized(patch.get("moment"), "ko"), _localized(patch.get("moment"), "en"),
            fate.get("kind") or "" if isinstance(fate, dict) else "",
            _localized(fate.get("label") if isinstance(fate, dict) else None, "ko"),
            _localized(fate.get("label") if isinstance(fate, dict) else None, "en"),
        ),
    )
    _replace_patronymic(cur, person_id, patch)
    _replace_aliases(cur, person_id, patch.get("aliases") or {
        "ko": [_localized(name, "ko")], "en": [_localized(name, "en")],
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
    if "initial" in patch:
        set_col("initial", patch.get("initial") or "")
    if "cyrillic" in patch:
        set_col("cyrillic", patch.get("cyrillic") or "")
    if "years" in patch:
        birth, death = _parse_life_years(patch.get("years") or "")
        set_col("years_label", patch.get("years") or "")
        set_col("birth_year", birth)
        set_col("death_year", death)
    if "name" in patch:
        set_col("name_ko", _localized(patch.get("name"), "ko"))
        set_col("name_en", _localized(patch.get("name"), "en"))
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
        set_col("fate_kind", fate.get("kind") or "" if isinstance(fate, dict) else "")
        set_col("fate_label_ko", _localized(fate.get("label") if isinstance(fate, dict) else None, "ko"))
        set_col("fate_label_en", _localized(fate.get("label") if isinstance(fate, dict) else None, "en"))
    if "sortOrder" in patch and isinstance(patch.get("sortOrder"), int):
        set_col("sort_order", patch["sortOrder"])
    if sets:
        sets.append("updated_at = NOW()")
        values.append(person_id)
        cur.execute(f"UPDATE commulingo_people SET {', '.join(sets)} WHERE id = %s", values)
    if "patronymic" in patch or "cyrillicPatronymic" in patch:
        _replace_patronymic(cur, person_id, patch)
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


def _validate(cur, target_type: str, action: str, target_id: str, patch: dict) -> str | None:
    """Return an error string, or None when the edit is applicable."""
    allowed = {
        "person": _PERSON_PATCH_KEYS,
        "office_row": _OFFICE_ROW_PATCH_KEYS,
        "person_section": _SECTION_PATCH_KEYS,
    }[target_type]
    unknown = set(patch) - allowed
    if unknown:
        return (
            f"Error: unknown patch key(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )
    if target_type == "person":
        for key in _LOCALIZED_PERSON_KEYS:
            if key in patch and patch[key] is not None and not isinstance(patch[key], dict):
                return (
                    f"Error: {key} must be an object {{\"ko\": \"...\", \"en\": \"...\"}} — "
                    "a plain string is rejected because the site is bilingual and the "
                    "other language would be silently lost."
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
                return "Error: fate.label must be {\"ko\": \"처형 1938\", \"en\": \"Shot, 1938\"}."
        if action == "create":
            if patch.get("id") and patch["id"] != target_id:
                return f"Error: patch.id '{patch['id']}' conflicts with target_id '{target_id}' — they must match (or omit patch.id)."
            if not _ID_RE.match(target_id):
                return "Error: target_id must be a lowercase kebab-case slug (e.g. 'ordzhonikidze')."
            cur.execute("SELECT 1 FROM commulingo_people WHERE id = %s", (target_id,))
            if cur.fetchone():
                return f"Error: person '{target_id}' already exists — use action 'update'."
            group = patch.get("groupId") or patch.get("group") or ""
            cur.execute("SELECT 1 FROM commulingo_people_groups WHERE id = %s", (group,))
            if not cur.fetchone():
                return f"Error: unknown group '{group}'. Check commulingo_people(action='list_groups')."
            name = patch.get("name") or {}
            if not (isinstance(name, dict) and name.get("ko") and name.get("en")):
                return "Error: patch.name.ko and patch.name.en are required for person create."
        else:
            cur.execute("SELECT 1 FROM commulingo_people WHERE id = %s", (target_id,))
            if not cur.fetchone():
                return f"Error: person '{target_id}' not found. Use search_people to find the id."
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
        cur.execute(
            "SELECT 1 FROM commulingo_person_sections WHERE person_id = %s AND slug = %s",
            (target_id, slug),
        )
        exists = bool(cur.fetchone())
        if action == "create":
            if exists:
                return f"Error: section '{slug}' already exists for '{target_id}' — use action 'update'."
            body = patch.get("body") or {}
            if not (body.get("ko") or body.get("en")):
                return "Error: body.ko or body.en (markdown) is required for section create."
        elif not exists:
            return f"Error: section '{slug}' not found for '{target_id}' (get_person lists existing slugs)."
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
            next_sort = cur.fetchone()["next_sort"]
            cur.execute(
                """INSERT INTO commulingo_person_sections
                      (person_id, slug, sort_order, heading_ko, heading_en,
                       body_ko, body_en, sources, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())""",
                (
                    target_id, slug,
                    patch["sortOrder"] if isinstance(patch.get("sortOrder"), int) else next_sort,
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
                    "on cyber-lenin.com/commulingo/people within ~1 minute (server cache TTL)."
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


COMMULINGO_EDIT_TOOL = {
    "name": "commulingo_edit",
    "description": (
        "Edit the CommuLingo people dictionary. Depending on the operator-set "
        "mode, the edit either applies immediately (with a revision snapshot, "
        "so it is reversible) or is staged for operator review — the response "
        "says which happened. Read the current record with commulingo_people "
        "first, and cite at least one source per edit. `patch` fields (include "
        "only what you change): person — group, initial (one letter of the "
        "native-script name), cyrillic (native-script name: Cyrillic for "
        "Soviet figures, hanzi/Latin/etc. for non-Soviet ones, e.g. 毛泽东), "
        "years ('1878–1953', en dash), name/epithet/bio/moment {ko,en}, fate "
        "{kind, label {ko,en}} (kind: executed/assassinated/murdered/killed/"
        "deposed/exile/natural), patronymic {ko,en}, cyrillicPatronymic, aliases "
        "{ko:[],en:[]}, career [{y:'1922–1953', r:{ko,en}}], role {officeId} "
        "OR {category} (the card's ONE primary marker; exactly one of the two — "
        "icon and label render from the linked institution or category, see "
        "list_offices/list_categories; ALWAYS set on person create; null "
        "clears; full multi-institution history belongs in office_rows/career); "
        "office_row — years, body {ko,en}, personId, name {ko,en}, note "
        "{ko,en}; person_section — slug (kebab-case section id), heading "
        "{ko,en}, body {ko,en} (MARKDOWN, replaced wholesale per section), "
        "sortOrder — long-form detail beyond the card, rendered on the "
        "person's detail page /commulingo/people/<id>; use sections for depth "
        "the card can't hold, one topic per section. CAUTION: aliases/"
        "career/scenes are replaced wholesale — send the complete new list, not "
        "just additions. All text fields are bilingual {ko,en} objects; plain "
        "strings are rejected. Targets: person create → target_id = new slug; "
        "person update/delete → person id; person_section (all actions) → "
        "person id + patch.slug; office_row create → "
        "office id; office_row update/delete → numeric row id (from get_office). "
        "HOUSE STYLE — each card is a miniature story: epithet = a "
        "characterization with tension or irony, never a job title; bio = one "
        "short paragraph per language that opens with a scene or contradiction "
        "and favors one concrete detail (a habit, a decision, a scene) over "
        "abstractions; moment = ONE defining quote or scene line per language, "
        "rendered as a pull-quote between epithet and bio — no dates-and-posts "
        "summary, that's what career is for. moment MUST be a real, verifiable "
        "quote or documented scene with a source; NEVER compose or embellish "
        "one — if nothing solid exists for a person, leave moment empty. An "
        "empty moment is correct; an invented one is vandalism."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "target_type": {"type": "string", "enum": list(_TARGET_TYPES)},
            "action": {"type": "string", "enum": list(_ACTIONS)},
            "target_id": {
                "type": "string",
                "description": "Person id, new person slug, office id, or numeric office-row id (see description).",
            },
            "patch": {
                "type": "object",
                "description": "Fields to set (admin-store shape). Empty object for delete.",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Required, non-empty. Each entry: a URL or citation plus what it "
                    "supports, e.g. 'https://... — 1937 arrest date'."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "Optional self-assessed confidence, 0–1.",
            },
        },
        "required": ["target_type", "action", "target_id", "sources"],
    },
}


async def _exec_commulingo_edit(
    target_type: str,
    action: str,
    target_id: str,
    sources: list,
    patch: dict | None = None,
    confidence: float | None = None,
) -> str:
    if target_type not in _TARGET_TYPES:
        return f"Error: target_type must be one of {_TARGET_TYPES}."
    if action not in _ACTIONS:
        return f"Error: action must be one of {_ACTIONS}."
    target_id = (target_id or "").strip()
    if not target_id:
        return "Error: target_id is required."
    patch = patch or {}
    if not isinstance(patch, dict):
        return "Error: patch must be an object."
    if action != "delete" and not patch:
        return "Error: patch is required for create/update."
    sources = [s.strip() for s in (sources or []) if isinstance(s, str) and s.strip()]
    if not sources:
        return "Error: at least one source citation is required (URL/reference + what it supports)."
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return "Error: confidence must be a number between 0 and 1."
        if not 0.0 <= confidence <= 1.0:
            return "Error: confidence must be between 0 and 1."
    try:
        return await asyncio.to_thread(
            _run_edit, target_type, action, target_id, patch, sources, confidence
        )
    except Exception as e:
        logger.warning("commulingo_edit error: %s", e)
        return f"Error: {type(e).__name__}: {e}"


COMMULINGO_TOOLS = [COMMULINGO_PEOPLE_TOOL, COMMULINGO_EDIT_TOOL]
COMMULINGO_TOOL_HANDLERS = {
    "commulingo_people": _exec_commulingo_people,
    "commulingo_edit": _exec_commulingo_edit,
}
