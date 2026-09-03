"""CommuLingo (Postgres) → knowledge graph mirror. Deterministic, no LLM.

Mapping (external id → node type; every edge carries ``attributes.sync_key``
so re-runs are idempotent and full passes can expire vanished rows):

  commulingo:person:<id>         Person     name=name_ko, aliases=name_en/cyrillic/person_aliases
  commulingo:office:<id>         Role       title_ko (office lineage, e.g. 국가보안 기관)
  commulingo:role-category:<id>  Role       role category (이론가, 좌익 반대파 …)
  commulingo:people-group:<id>   Concept    era group (스탈린 시대의 사람들 …)
  commulingo:event:<id>          Incident   title_ko, summary_ko, period_label
  commulingo:location:<slug>     Location   event map pins (label.ko / label.en)
  commulingo:term:<id>           Concept    term_ko, aliases term_en/original/term_aliases
  commulingo:term-category:<id>  Concept    term category (이념·이론 …)

  Person→Role       Affiliation   person_roles.category_id / office_id, office_rows (valid_at/invalid_at)
  Person→Concept    Reference     people_group, person_term
  Person→Incident   Involvement   history_event_people (role_in_incident = relation_kind)
  Incident→Location Presence      history_events.locations
  Concept→Concept   Reference     term_relations (related_term), parent_id (parent_term), category
  Concept→Incident  Reference     term_events (event_term)

Career entries (17k free-text rows) are folded into the Person summary, not
materialized as edges. ``commulingo_id_redirects`` are honoured: a node still
carrying the old id is merged into the canonical one.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime

from db import query as db_query

logger = logging.getLogger(__name__)

GROUP_ID = "commulingo"
AGENT = "kg_sync_commulingo"
TRUST_TIER = "anchor"
PROVENANCE = "source: CommuLingo curation tables (Postgres) — deterministic mirror by jobs/kg_sync_commulingo"
BATCH_SIZE = 200
SUMMARY_MAX = 700
CAREER_LINES = 6

# External-id / sync-key namespace. Smoke tests point this at a scratch
# namespace so they never touch the real "commulingo:" keys.
NAMESPACE = "commulingo"


def ext_id(kind: str, ident: str) -> str:
    return f"{NAMESPACE}:{kind}:{ident}"


def sync_key(kind: str, *parts) -> str:
    return ":".join([NAMESPACE, kind, *[str(p) for p in parts]])


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text or "unknown"


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _truncate(text: str, limit: int = SUMMARY_MAX) -> str:
    text = _clean(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _year_date(year, month=None, *, end: bool = False) -> str | None:
    if not year:
        return None
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    if month:
        try:
            m = int(month)
            return f"{y:04d}-{m:02d}-01"
        except (TypeError, ValueError):
            pass
    return f"{y:04d}-12-31" if end else f"{y:04d}-01-01"


# ── Side (entity) hints ───────────────────────────────────────────────────────

def person_side(p: dict, aliases: list[str], career: list[dict]) -> dict:
    name = _clean(p.get("name_ko")) or _clean(p.get("name_en")) or p["id"]
    alias_list = [a for a in [
        _clean(p.get("name_en")), _clean(p.get("cyrillic")),
        _clean(p.get("given_name_ko")) + " " + _clean(p.get("family_name_ko")) if p.get("family_name_ko") else "",
        *aliases,
    ] if a and a != name]
    return {
        "name": name, "type": "Person", "external_id": ext_id("person", p["id"]),
        "aliases": alias_list, "summary": person_summary(p, career),
        "name_ko": _clean(p.get("name_ko")) or None, "name_en": _clean(p.get("name_en")) or None,
    }


def person_summary(p: dict, career: list[dict]) -> str:
    parts = []
    if p.get("epithet_ko"):
        parts.append(_clean(p["epithet_ko"]))
    years = _clean(p.get("years_label"))
    if years:
        parts.append(years)
    if p.get("bio_ko"):
        parts.append(_clean(p["bio_ko"]))
    if p.get("fate_label_ko"):
        parts.append(f"최후: {_clean(p['fate_label_ko'])}")
    lines = []
    for c in career[:CAREER_LINES]:
        role = _clean(c.get("role_ko"))
        if not role:
            continue
        period = _clean(c.get("period_label"))
        lines.append(f"{period} {role}".strip())
    if lines:
        parts.append("주요 경력: " + "; ".join(lines))
    return _truncate(" ".join(parts))


def event_side(e: dict) -> dict:
    name = _clean(e.get("title_ko")) or _clean(e.get("title_en")) or e["id"]
    aliases = [a for a in [_clean(e.get("title_en"))] if a and a != name]
    period = _clean(e.get("period_label"))
    summary = _truncate(((period + ". ") if period else "") + _clean(e.get("summary_ko") or e.get("summary_en")))
    return {
        "name": name, "type": "Incident", "external_id": ext_id("event", e["id"]),
        "aliases": aliases, "summary": summary,
        "name_ko": _clean(e.get("title_ko")) or None, "name_en": _clean(e.get("title_en")) or None,
    }


def term_side(t: dict, aliases: list[str]) -> dict:
    name = _clean(t.get("term_ko")) or _clean(t.get("term_en")) or t["id"]
    alias_list = [a for a in [_clean(t.get("term_en")), _clean(t.get("original")), *aliases] if a and a != name]
    period = _clean(t.get("period_label"))
    summary = _truncate(((period + ". ") if period else "") + _clean(t.get("definition_ko") or t.get("definition_en")))
    return {
        "name": name, "type": "Concept", "external_id": ext_id("term", t["id"]),
        "aliases": alias_list, "summary": summary,
        "name_ko": _clean(t.get("term_ko")) or None, "name_en": _clean(t.get("term_en")) or None,
    }


def office_side(o: dict) -> dict:
    name = _clean(o.get("title_ko")) or _clean(o.get("title_en")) or o["id"]
    return {
        "name": name, "type": "Role", "external_id": ext_id("office", o["id"]),
        "aliases": [a for a in [_clean(o.get("title_en"))] if a and a != name],
        "summary": _truncate(_clean(o.get("blurb_ko") or o.get("blurb_en"))),
        "name_ko": _clean(o.get("title_ko")) or None, "name_en": _clean(o.get("title_en")) or None,
    }


def role_category_side(c: dict) -> dict:
    name = _clean(c.get("label_ko")) or _clean(c.get("label_en")) or c["id"]
    return {
        "name": name, "type": "Role", "external_id": ext_id("role-category", c["id"]),
        "aliases": [a for a in [_clean(c.get("label_en"))] if a and a != name],
        "summary": f"CommuLingo 인물 역할 범주: {name}",
        "name_ko": _clean(c.get("label_ko")) or None, "name_en": _clean(c.get("label_en")) or None,
    }


def people_group_side(g: dict) -> dict:
    name = _clean(g.get("title_ko")) or _clean(g.get("title_en")) or g["id"]
    period = _clean(g.get("range_label"))
    return {
        "name": name, "type": "Concept", "external_id": ext_id("people-group", g["id"]),
        "aliases": [a for a in [_clean(g.get("title_en"))] if a and a != name],
        "summary": _truncate(((period + ". ") if period else "") + _clean(g.get("blurb_ko") or g.get("blurb_en"))),
        "name_ko": _clean(g.get("title_ko")) or None, "name_en": _clean(g.get("title_en")) or None,
    }


def josa(word: str, pair: str) -> str:
    """Korean particle by final consonant: josa("레닌", "은/는") -> "은"."""
    with_batchim, without = pair.split("/")
    if not word:
        return without
    ch = word[-1]
    if "가" <= ch <= "힣":
        return with_batchim if (ord(ch) - 0xAC00) % 28 else without
    if ch.isdigit():
        return with_batchim if ch in "01367" else without
    return with_batchim if ch.lower() in "lmnr" else without


def term_category_side(c: dict) -> dict:
    name = _clean(c.get("label_ko")) or _clean(c.get("label_en")) or c["id"]
    return {
        "name": name, "type": "Concept", "external_id": ext_id("term-category", c["id"]),
        "aliases": [a for a in [_clean(c.get("label_en"))] if a and a != name],
        "summary": f"CommuLingo 용어 범주: {name}",
        "name_ko": _clean(c.get("label_ko")) or None, "name_en": _clean(c.get("label_en")) or None,
    }


def location_side(loc: dict) -> dict | None:
    label = loc.get("label") or {}
    if isinstance(label, str):
        label = {"ko": label}
    ko, en = _clean(label.get("ko")), _clean(label.get("en"))
    name = ko or en
    if not name:
        return None
    return {
        "name": name, "type": "Location", "external_id": ext_id("location", slugify(en or ko)),
        "aliases": [a for a in [en] if a and a != name], "summary": "",
        "name_ko": ko or None, "name_en": en or None,
    }


# ── Fact assembly ─────────────────────────────────────────────────────────────

def make_fact(subject: dict, predicate: str, obj: dict, fact: str, *, sync_key: str,
              attributes: dict | None = None, valid_at: str | None = None,
              invalid_at: str | None = None) -> dict:
    attrs = {"sync_key": sync_key}
    for k, v in (attributes or {}).items():
        if v not in (None, ""):
            attrs[k] = v
    f = {
        "subject_name": subject["name"], "subject_type": subject["type"],
        "predicate": predicate,
        "object_name": obj["name"], "object_type": obj["type"],
        "fact": _truncate(fact, 600),
        "attributes": attrs,
    }
    for side, hints in (("subject", subject), ("object", obj)):
        for key in ("external_id", "aliases", "summary", "name_ko", "name_en"):
            if hints.get(key):
                f[f"{side}_{key}"] = hints[key]
    if valid_at:
        f["valid_at"] = valid_at
    if invalid_at:
        f["invalid_at"] = invalid_at
    return f


class Source:
    """All CommuLingo rows needed for the mapping, keyed by id."""

    def __init__(self, **tables):
        self.people = {p["id"]: p for p in tables.get("people", [])}
        self.person_aliases: dict[str, list[str]] = {}
        for a in tables.get("person_aliases", []):
            self.person_aliases.setdefault(a["person_id"], []).append(_clean(a["alias"]))
        self.career: dict[str, list[dict]] = {}
        for c in sorted(tables.get("career", []), key=lambda r: (r.get("sort_order") or 0)):
            self.career.setdefault(c["person_id"], []).append(c)
        self.roles = {r["person_id"]: r for r in tables.get("person_roles", [])}
        self.role_categories = {c["id"]: c for c in tables.get("role_categories", [])}
        self.groups = {g["id"]: g for g in tables.get("people_groups", [])}
        self.offices = {o["id"]: o for o in tables.get("offices", [])}
        self.office_rows = tables.get("office_rows", [])
        self.events = {e["id"]: e for e in tables.get("events", [])}
        self.event_people = tables.get("event_people", [])
        self.terms = {t["id"]: t for t in tables.get("terms", [])}
        self.term_aliases: dict[str, list[str]] = {}
        for a in tables.get("term_aliases", []):
            self.term_aliases.setdefault(a["term_id"], []).append(_clean(a["alias"]))
        self.term_categories = {c["id"]: c for c in tables.get("term_categories", [])}
        self.term_relations = tables.get("term_relations", [])
        self.term_people = tables.get("term_people", [])
        self.term_events = tables.get("term_events", [])
        self.redirects = tables.get("redirects", [])

    def person(self, pid: str) -> dict | None:
        p = self.people.get(pid)
        return person_side(p, self.person_aliases.get(pid, []), self.career.get(pid, [])) if p else None

    def event(self, eid: str) -> dict | None:
        e = self.events.get(eid)
        return event_side(e) if e else None

    def term(self, tid: str) -> dict | None:
        t = self.terms.get(tid)
        return term_side(t, self.term_aliases.get(tid, [])) if t else None


def build_facts(src: Source, *, changed: dict[str, set[str]] | None = None) -> list[dict]:
    """All mirror facts. ``changed`` = {"person": ids, "event": ids, "term": ids,
    "office": ids} restricts output to facts touching a changed entity."""

    def touched(kind: str, ident: str) -> bool:
        return changed is None or ident in changed.get(kind, set())

    facts: list[dict] = []

    # people → role category / office lineage / era group
    for pid, p in src.people.items():
        if not touched("person", pid):
            continue
        ps = src.person(pid)
        role = src.roles.get(pid) or {}
        produced = 0
        cat = src.role_categories.get(role.get("category_id"))
        if cat:
            facts.append(make_fact(
                ps, "Affiliation", role_category_side(cat),
                f"{ps['name']}: {_clean(role.get('label_ko')) or cat.get('label_ko')}",
                sync_key=sync_key("person_role", pid),
                attributes={"position": _clean(role.get("label_ko")), "affiliation_type": "role_category"},
            ))
            produced += 1
        office = src.offices.get(role.get("office_id"))
        if office:
            facts.append(make_fact(
                ps, "Affiliation", office_side(office),
                f"{ps['name']}{josa(ps['name'], '은/는')} {office_side(office)['name']} 계보에 속한다"
                + (f" — {_clean(role.get('label_ko'))}" if role.get("label_ko") else ""),
                sync_key=sync_key("person_office", pid, office['id']),
                attributes={"position": _clean(role.get("label_ko")), "affiliation_type": "office_lineage"},
            ))
            produced += 1
        group = src.groups.get(p.get("group_id"))
        if group:
            gs = people_group_side(group)
            facts.append(make_fact(
                ps, "Reference", gs,
                f"{ps['name']}{josa(ps['name'], '은/는')} CommuLingo 인물사전의 '{gs['name']}' ({_clean(group.get('range_label'))}) 그룹에 수록되어 있다",
                sync_key=sync_key("person_group", pid),
                attributes={"reference_type": "people_group"},
            ))
            produced += 1
        if produced == 0:
            facts.append(make_fact(
                ps, "Reference",
                {"name": "CommuLingo 인물사전", "type": "Concept",
                 "external_id": ext_id("collection", "people"), "aliases": ["CommuLingo people"],
                 "summary": "cyber-lenin.com CommuLingo 인물사전 수록 인물", "name_ko": "CommuLingo 인물사전",
                 "name_en": "CommuLingo people"},
                f"{ps['name']}{josa(ps['name'], '은/는')} CommuLingo 인물사전에 수록되어 있다",
                sync_key=sync_key("person_collection", pid),
                attributes={"reference_type": "people_group"},
            ))

    # office rows → dated Person→Role affiliations
    for row in src.office_rows:
        pid, oid = row.get("person_id"), row.get("office_id")
        office = src.offices.get(oid)
        if not (pid and office and pid in src.people):
            continue
        if not (touched("person", pid) or touched("office", oid)):
            continue
        ps = src.person(pid)
        os_ = office_side(office)
        period = _clean(row.get("period_label"))
        body = _clean(row.get("body_ko") or row.get("body_en"))
        facts.append(make_fact(
            ps, "Affiliation", os_,
            f"{ps['name']}: {os_['name']} — {body}" + (f" ({period})" if period else ""),
            sync_key=sync_key("office_row", row['id']),
            attributes={"position": body, "affiliation_type": "office_row", "period_label": period},
            valid_at=_year_date(row.get("start_year"), row.get("start_month")),
            invalid_at=_year_date(row.get("end_year"), row.get("end_month"), end=True),
        ))

    # events → people involvement, locations
    for eid, e in src.events.items():
        if not touched("event", eid):
            continue
        es = event_side(e)
        locations = e.get("locations") or []
        if isinstance(locations, str):
            try:
                locations = json.loads(locations)
            except ValueError:
                locations = []
        for loc in locations:
            ls = location_side(loc) if isinstance(loc, dict) else None
            if not ls:
                continue
            kind = "주 무대" if (loc.get("kind") == "main") else "관련 장소"
            facts.append(make_fact(
                es, "Presence", ls,
                f"{es['name']}의 {kind}: {ls['name']}",
                sync_key=sync_key("event_location", eid, ls['external_id'].rsplit(':', 1)[-1]),
                attributes={"presence_type": "main" if loc.get("kind") == "main" else "related"},
            ))
    for link in src.event_people:
        eid, pid = link["event_id"], link["person_id"]
        if eid not in src.events or pid not in src.people:
            continue
        if not (touched("event", eid) or touched("person", pid)):
            continue
        ps, es = src.person(pid), src.event(eid)
        relation = _clean(link.get("relation_ko") or link.get("relation_en"))
        note = _clean(link.get("note_ko") or link.get("note_en"))
        text = f"{ps['name']} — {es['name']}"
        if relation:
            text += f": {relation}"
        if note:
            text += f". {note}"
        facts.append(make_fact(
            ps, "Involvement", es, text,
            sync_key=sync_key("event_person", eid, pid),
            attributes={"role_in_incident": link.get("relation_kind") or relation, "relation_label": relation},
        ))

    # terms → category, parent, related, people, events
    for tid, t in src.terms.items():
        if not touched("term", tid):
            continue
        ts = src.term(tid)
        cat = src.term_categories.get(t.get("category"))
        if cat:
            cs = term_category_side(cat)
            facts.append(make_fact(
                ts, "Reference", cs, f"{ts['name']}{josa(ts['name'], '은/는')} '{cs['name']}' 범주의 용어이다",
                sync_key=sync_key("term_category", tid), attributes={"reference_type": "category"},
            ))
        parent = src.term(t["parent_id"]) if t.get("parent_id") else None
        if parent:
            facts.append(make_fact(
                ts, "Reference", parent, f"{ts['name']}의 상위 용어는 {parent['name']}이다",
                sync_key=sync_key("term_parent", tid), attributes={"reference_type": "parent_term"},
            ))
    for rel in src.term_relations:
        a, b = rel["term_id"], rel["related_id"]
        if a not in src.terms or b not in src.terms:
            continue
        if not (touched("term", a) or touched("term", b)):
            continue
        ts, rs = src.term(a), src.term(b)
        facts.append(make_fact(
            ts, "Reference", rs, f"{ts['name']}{josa(ts['name'], '은/는')} {rs['name']}{josa(rs['name'], '과/와')} 연관된 용어이다",
            sync_key=sync_key("term_relation", a, b), attributes={"reference_type": "related_term"},
        ))
    for link in src.term_people:
        tid, pid = link["term_id"], link["person_id"]
        if tid not in src.terms or pid not in src.people:
            continue
        if not (touched("term", tid) or touched("person", pid)):
            continue
        ps, ts = src.person(pid), src.term(tid)
        facts.append(make_fact(
            ps, "Reference", ts, f"{ps['name']}{josa(ps['name'], '은/는')} 용어 '{ts['name']}'{josa(ts['name'], '과/와')} 연관된 인물이다",
            sync_key=sync_key("term_person", tid, pid), attributes={"reference_type": "person_term"},
        ))
    for link in src.term_events:
        tid, eid = link["term_id"], link["event_id"]
        if tid not in src.terms or eid not in src.events:
            continue
        if not (touched("term", tid) or touched("event", eid)):
            continue
        ts, es = src.term(tid), src.event(eid)
        facts.append(make_fact(
            ts, "Reference", es, f"용어 '{ts['name']}'{josa(ts['name'], '은/는')} 사건 '{es['name']}'{josa(es['name'], '과/와')} 연관된다",
            sync_key=sync_key("term_event", tid, eid),
            attributes={"reference_type": "event_term", "same_subject": bool(link.get("same_subject"))},
        ))

    return facts


# ── Postgres loading ──────────────────────────────────────────────────────────

def load_source() -> Source:
    q = db_query
    return Source(
        people=q("SELECT * FROM commulingo_people"),
        person_aliases=q("SELECT person_id, lang, alias FROM commulingo_person_aliases ORDER BY sort_order"),
        career=q("SELECT person_id, sort_order, period_label, start_year, end_year, role_ko, role_en FROM commulingo_person_career_entries"),
        person_roles=q("SELECT * FROM commulingo_person_roles"),
        role_categories=q("SELECT * FROM commulingo_role_categories"),
        people_groups=q("SELECT * FROM commulingo_people_groups"),
        offices=q("SELECT id, sort_order, range_label, title_ko, title_en, blurb_ko, blurb_en FROM commulingo_offices"),
        office_rows=q("SELECT * FROM commulingo_office_rows"),
        events=q("SELECT id, period_label, title_ko, title_en, summary_ko, summary_en, locations, updated_at FROM commulingo_history_events"),
        event_people=q("SELECT * FROM commulingo_history_event_people"),
        terms=q("SELECT id, term_ko, term_en, original, period_label, definition_ko, definition_en, category, parent_id, updated_at FROM commulingo_terms"),
        term_aliases=q("SELECT term_id, lang, alias FROM commulingo_term_aliases ORDER BY sort_order"),
        term_categories=q("SELECT * FROM commulingo_term_categories"),
        term_relations=q("SELECT term_id, related_id FROM commulingo_term_relations"),
        term_people=q("SELECT term_id, person_id FROM commulingo_term_people"),
        term_events=q("SELECT term_id, event_id, same_subject FROM commulingo_term_events"),
        redirects=q("SELECT entity_type, from_id, to_id FROM commulingo_id_redirects"),
    )


_REVISION_KIND = {
    "person": "person", "person_career_entry": "person", "person_section": "person",
    "history_event": "event", "history_event_person": "event",
    "term": "term", "office": "office",
}


def changed_since(since: datetime) -> dict[str, set[str]]:
    """Entity ids touched after ``since`` (updated_at columns + revision log).
    Revision ids like ``event/person`` or ``person/section`` map to their parent."""
    changed = {"person": set(), "event": set(), "term": set(), "office": set()}
    for kind, sql in (
        ("person", "SELECT id FROM commulingo_people WHERE updated_at > %s"),
        ("event", "SELECT id FROM commulingo_history_events WHERE updated_at > %s"),
        ("term", "SELECT id FROM commulingo_terms WHERE updated_at > %s"),
        ("office", "SELECT id FROM commulingo_offices WHERE updated_at > %s"),
        ("office", "SELECT office_id AS id FROM commulingo_office_rows WHERE updated_at > %s"),
        ("person", "SELECT person_id AS id FROM commulingo_office_rows WHERE updated_at > %s AND person_id IS NOT NULL"),
        ("person", "SELECT person_id AS id FROM commulingo_person_roles WHERE updated_at > %s"),
        ("person", "SELECT person_id AS id FROM commulingo_person_career_entries WHERE updated_at > %s"),
    ):
        for r in db_query(sql, (since,)):
            if r.get("id"):
                changed[kind].add(str(r["id"]))
    for r in db_query(
        "SELECT entity_type, entity_id FROM commulingo_people_revisions WHERE created_at > %s", (since,)
    ):
        kind = _REVISION_KIND.get(r["entity_type"])
        if kind and r.get("entity_id"):
            changed[kind].add(str(r["entity_id"]).split("/", 1)[0])
    return changed


# ── Graph side ────────────────────────────────────────────────────────────────

def _neo4j_session():
    from kg_runtime.search import _get_neo4j_sync_driver
    return _get_neo4j_sync_driver()


def existing_sync_edges(prefix: str | None = None) -> dict[str, dict]:
    prefix = prefix or (NAMESPACE + ":")
    with _neo4j_session() as (drv, db):
        with drv.session(database=db) as s:
            rows = s.run(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.sync_key STARTS WITH $prefix "
                "RETURN r.sync_key AS key, r.uuid AS uuid, r.fact AS fact, r.expired_at AS expired_at",
                prefix=prefix,
            )
            return {r["key"]: {"uuid": r["uuid"], "fact": r["fact"], "expired": r["expired_at"] is not None}
                    for r in rows}


def expire_edges(uuids: list[str]) -> int:
    if not uuids:
        return 0
    with _neo4j_session() as (drv, db):
        with drv.session(database=db) as s:
            rec = s.run(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.uuid IN $uuids AND r.expired_at IS NULL "
                "SET r.expired_at = datetime() RETURN count(r) AS cnt",
                uuids=uuids,
            ).single()
            return rec["cnt"] if rec else 0


def apply_redirects(redirects: list[dict]) -> dict:
    """Merge nodes still carrying a redirected id into the canonical node, or
    record the old id on the canonical node so both ids resolve."""
    from kg_runtime.identity import merge_entity_nodes_sync, upsert_identity_sync

    kinds = {"person": "person", "term": "term", "history_event": "event", "event": "event"}
    stats = {"merged": 0, "aliased": 0}
    with _neo4j_session() as (drv, db):
        with drv.session(database=db) as s:
            for r in redirects:
                kind = kinds.get(r.get("entity_type"))
                if not kind:
                    continue
                old_id, new_id = ext_id(kind, r["from_id"]), ext_id(kind, r["to_id"])
                rows = {row["eid"]: row["uuid"] for row in s.run(
                    "MATCH (n:Entity) WHERE $old IN coalesce(n.external_ids, []) OR $new IN coalesce(n.external_ids, []) "
                    "UNWIND [x IN n.external_ids WHERE x IN [$old, $new]] AS eid RETURN eid, n.uuid AS uuid",
                    old=old_id, new=new_id,
                )}
                if old_id in rows and new_id in rows and rows[old_id] != rows[new_id]:
                    merge_entity_nodes_sync(s, rows[new_id], [rows[old_id]])
                    stats["merged"] += 1
                elif new_id in rows and old_id not in rows:
                    upsert_identity_sync(s, rows[new_id], external_ids=[old_id])
                    stats["aliased"] += 1
    return stats


def write_facts(facts: list[dict], *, batch_size: int = BATCH_SIZE, group_id: str = GROUP_ID) -> dict:
    from kg_runtime.writes import add_kg_structured

    stats = {"written": 0, "rejected": 0, "new_entities": 0, "reused_entities": 0, "batches": 0, "errors": []}
    for i in range(0, len(facts), batch_size):
        batch = facts[i:i + batch_size]
        res = add_kg_structured(
            batch, group_id=group_id, agent=AGENT, trust_tier=TRUST_TIER,
            provenance_footer=PROVENANCE, allow_sync_predicates=True,
        )
        stats["batches"] += 1
        if res.get("status") == "error":
            stats["errors"].append(res.get("message", "unknown error"))
            logger.error("[kg-sync commulingo] batch %d failed: %s", stats["batches"], res.get("message"))
            continue
        stats["written"] += res.get("facts_written", 0)
        stats["rejected"] += res.get("facts_rejected", 0)
        stats["new_entities"] += res.get("new_entities", 0)
        stats["reused_entities"] += res.get("reused_entities", 0)
        for rej in (res.get("rejected_facts") or [])[:5]:
            logger.warning("[kg-sync commulingo] rejected: %s", rej.get("reason"))
    return stats


# ── Runner ────────────────────────────────────────────────────────────────────

def run(*, since: datetime | None = None, full: bool = False, limit: int | None = None,
        dry_run: bool = False) -> dict:
    src = load_source()
    changed = None if (full or since is None) else changed_since(since)
    facts = build_facts(src, changed=changed)
    stats: dict = {
        "source_rows": {"people": len(src.people), "events": len(src.events), "terms": len(src.terms),
                        "offices": len(src.offices), "office_rows": len(src.office_rows),
                        "event_people": len(src.event_people)},
        "changed": None if changed is None else {k: len(v) for k, v in changed.items()},
        "facts_total": len(facts),
    }

    existing = {} if dry_run else existing_sync_edges()
    to_write, to_expire = [], []
    for f in facts:
        key = f["attributes"]["sync_key"]
        old = existing.get(key)
        if old is None:
            to_write.append(f)
        elif old["expired"] or (old["fact"] or "") != f["fact"]:
            to_expire.append(old["uuid"])
            to_write.append(f)
    if full and existing:
        current_keys = {f["attributes"]["sync_key"] for f in facts}
        stale = [v["uuid"] for k, v in existing.items() if k not in current_keys and not v["expired"]]
        to_expire.extend(stale)
        stats["stale_expired"] = len(stale)
    if limit is not None:
        to_write = to_write[:limit]
    stats.update({"facts_new_or_changed": len(to_write), "edges_to_expire": len(to_expire)})

    if dry_run:
        stats["sample"] = [
            {k: v for k, v in f.items() if k in ("subject_name", "predicate", "object_name", "fact", "valid_at")}
            for f in to_write[:10]
        ]
        return stats

    stats["expired"] = expire_edges(to_expire)
    stats["write"] = write_facts(to_write)
    # After the writes so a first (full) run already sees the canonical nodes.
    stats["redirects"] = apply_redirects(src.redirects)
    if stats["write"]["errors"]:
        stats["error"] = f"{len(stats['write']['errors'])} batch(es) failed: {stats['write']['errors'][0][:200]}"
    return stats
