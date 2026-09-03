"""Document → knowledge graph: Document nodes, curated links, cheap extraction.

Three document stores feed the graph:

  research:<slug>   research_documents (public)          Postgres
  archival:<id>     archival translation manifest        $FRONTEND_DIR/data/commulingo/docs
  autonote:<id>     autonomous_project_notes (synthesis) Postgres

Two layers per document:

1. Deterministic (no LLM): a ``Document`` node (title, description/summary,
   content_sha256, url) plus ``Reference`` edges — ``about`` for the archival
   manifest's curated people/terms/events (resolved through their
   ``commulingo:*`` external ids, which is the archival ↔ CommuLingo hub
   link) and ``mentions`` for entities whose alias occurs in the title /
   description / opening text (in-process alias index, no embedding).
2. LLM extraction (``KG_DOC_EXTRACT_LLM=1``): registry call site
   ``kg_document_extraction`` turns the opening 12k chars into ≤15
   ``write_kg_structured``-style facts (agent schema only — no Document /
   Reference), each carrying ``attributes.doc_ref`` for provenance; every
   entity so asserted also gets a Document→Entity ``Reference(mentions)``.

Idempotency: the Document node stores ``content_sha256``; an unchanged hash
is skipped, a changed one expires the document's previous edges (matched by
``sync_key`` prefix ``doc:<ref>:``) before re-extraction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


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


GROUP_ID = "documents"
AGENT = "kg_sync_documents"
LLM_FEATURE = "kg_document_extraction"
MAX_LLM_CHARS = 12000
MAX_LLM_FACTS = 15
MAX_MENTIONS = 15
MENTION_SCAN_CHARS = 4000
# Mention links need a bit more than the search matcher: short Korean common
# nouns ("전환", "노동") are real Concept nodes but link every document.
MENTION_MIN_HANGUL = 3
MENTION_BLOCKLIST_KEYS = {"cyber lenin", "cyber lenin 자율지능플랫폼", "leninbot", "사이버 레닌", "사이버레닌"}

COLLECTION = {
    "research": {"name": "Cyber-Lenin 리서치 문서", "external_id": "collection:research",
                 "aliases": ["Cyber-Lenin research documents"], "summary": "cyber-lenin.com에 발행된 리서치·분석 보고서"},
    "archival": {"name": "CommuLingo 사료 아카이브", "external_id": "collection:archival",
                 "aliases": ["CommuLingo archival documents"], "summary": "cyber-lenin.com CommuLingo 사료 번역 발행본"},
    "autonote": {"name": "자율 프로젝트 종합 노트", "external_id": "collection:autonote",
                 "aliases": ["autonomous project synthesis notes"], "summary": "자율 프로젝트 루프가 남긴 종합(synthesis) 노트"},
}

TRUST_TIER = {"research": "corroborated", "archival": "corroborated", "autonote": "single"}


def llm_enabled() -> bool:
    return os.getenv("KG_DOC_EXTRACT_LLM", "0").strip().lower() in ("1", "true", "yes", "on")


def sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def html_to_text(html: str) -> str:
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html or "", flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>|</p>|</h\d>|</li>|</tr>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def markdown_to_text(md: str) -> str:
    text = re.sub(r"```.*?```", " ", md or "", flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[*_`>|]+", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


# ── Document records ──────────────────────────────────────────────────────────

class DocRecord(dict):
    """kind, ident, ref, title, description, text, url, lang, published_at,
    sha, aliases (list), links ({"person": [...], "term": [...], "event": [...]})."""

    @property
    def ref(self) -> str:
        return self["ref"]


def research_record(row: dict) -> DocRecord:
    md = row.get("markdown") or ""
    text = markdown_to_text(md)
    tags = row.get("tags") or []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except ValueError:
            tags = []
    slug = row.get("slug") or row.get("filename") or str(row.get("id"))
    return DocRecord(
        kind="research", ident=slug, ref=f"research:{slug}",
        title=_clean(row.get("title")) or slug,
        description=_clean(row.get("summary")) or text[:300],
        text=text, url=f"https://cyber-lenin.com/research/{slug}",
        lang=row.get("lang") or "ko",
        published_at=str(row.get("published_at") or "")[:10] or None,
        sha=row.get("content_sha256") or sha256(md),
        aliases=[_clean(row.get("title_en"))] if row.get("title_en") else [],
        links={"person": [], "term": [], "event": []},
        tags=[str(t) for t in tags if t],
    )


def archival_record(doc: dict, html: str | None) -> DocRecord:
    title = doc.get("title") or {}
    desc = doc.get("description") or {}
    aliases = doc.get("aliases") or {}
    alias_list = []
    for lang in ("ko", "en"):
        alias_list.extend([_clean(a) for a in (aliases.get(lang) or []) if a])
    if title.get("en"):
        alias_list.append(_clean(title["en"]))
    text = html_to_text(html) if html else ""
    body_for_sha = html if html is not None else json.dumps(doc, ensure_ascii=False, sort_keys=True)
    return DocRecord(
        kind="archival", ident=doc["id"], ref=f"archival:{doc['id']}",
        title=_clean(title.get("ko") or title.get("en")) or doc["id"],
        description=_clean(desc.get("ko") or desc.get("en")),
        text=text, url=f"https://cyber-lenin.com/commulingo/docs/{doc['id']}",
        lang=doc.get("docLang") or "ko",
        published_at=(doc.get("addedAt") or None),
        sha=sha256(body_for_sha),
        aliases=[a for a in alias_list if a],
        links={"person": list(doc.get("people") or []), "term": list(doc.get("terms") or []),
               "event": list(doc.get("events") or [])},
        source=_clean(doc.get("source")), date=doc.get("date"),
        kind_label=_clean((doc.get("kind") or {}).get("ko")),
    )


def autonote_record(row: dict) -> DocRecord:
    text_md = row.get("text") or ""
    text = markdown_to_text(text_md)
    first = next((l.strip("# ").strip() for l in text_md.splitlines() if l.strip()), "")
    title = _clean(first)[:120] or f"autonomous note #{row['id']}"
    return DocRecord(
        kind="autonote", ident=str(row["id"]), ref=f"autonote:{row['id']}",
        title=title, description=text[:300], text=text, url=None, lang="ko",
        published_at=str(row.get("created_at") or "")[:10] or None,
        sha=sha256(text_md), aliases=[], links={"person": [], "term": [], "event": []},
        project_id=row.get("project_id"),
    )


# ── Fact assembly ─────────────────────────────────────────────────────────────

def document_side(rec: DocRecord) -> dict:
    summary = rec.get("description") or ""
    if rec["kind"] == "archival":
        bits = [b for b in (rec.get("kind_label"), rec.get("date")) if b]
        if bits:
            summary = " · ".join(bits) + ". " + summary
    return {
        "name": rec["title"], "type": "Document", "external_id": rec.ref,
        "aliases": [a for a in rec.get("aliases", []) if a and a != rec["title"]],
        "summary": summary[:700],
    }


def _doc_attrs(rec: DocRecord, key: str, **extra) -> dict:
    attrs = {"sync_key": f"doc:{rec.ref}:{key}", "doc_ref": rec.ref}
    attrs.update({k: v for k, v in extra.items() if v not in (None, "")})
    return attrs


def _fact(subject: dict, predicate: str, obj: dict, text: str, attrs: dict, **more) -> dict:
    f = {
        "subject_name": subject["name"], "subject_type": subject["type"], "predicate": predicate,
        "object_name": obj["name"], "object_type": obj["type"], "fact": text[:600], "attributes": attrs,
    }
    for side, hints in (("subject", subject), ("object", obj)):
        for k in ("external_id", "aliases", "summary", "name_ko", "name_en"):
            if hints.get(k):
                f[f"{side}_{k}"] = hints[k]
    f.update({k: v for k, v in more.items() if v})
    return f


def collection_fact(rec: DocRecord) -> dict:
    col = COLLECTION[rec["kind"]]
    obj = {"name": col["name"], "type": "Concept", "external_id": col["external_id"],
           "aliases": col["aliases"], "summary": col["summary"]}
    return _fact(document_side(rec), "Reference", obj,
                 f"'{rec['title']}'{'' if rec.get('published_at') is None else ' (' + rec['published_at'] + ')'}{josa(rec['title'], '은/는')} {col['name']}에 수록된 문서이다",
                 _doc_attrs(rec, "collection", reference_type="collection"),
                 valid_at=rec.get("published_at"))


def curated_link_facts(rec: DocRecord, names: dict[str, dict[str, str]]) -> list[dict]:
    """Document→Entity Reference(about) for manifest-curated people/terms/events.
    ``names`` maps kind → {slug: display name} (from CommuLingo tables) so the
    node is created with its Korean name even before the CommuLingo mirror ran."""
    doc = document_side(rec)
    facts = []
    types = {"person": "Person", "term": "Concept", "event": "Incident"}
    for kind, slugs in (rec.get("links") or {}).items():
        for slug in slugs:
            name = (names.get(kind) or {}).get(slug)
            if not name:
                logger.debug("[doc-extract] %s: unknown %s slug %s", rec.ref, kind, slug)
                continue
            obj = {"name": name, "type": types[kind], "external_id": f"commulingo:{kind}:{slug}"}
            facts.append(_fact(
                doc, "Reference", obj, f"문서 '{rec['title']}'{josa(rec['title'], '은/는')} {name}{josa(name, '을/를')} 다룬다",
                _doc_attrs(rec, f"about:{kind}:{slug}", reference_type="about"),
                valid_at=rec.get("published_at"),
            ))
    return facts


def mention_facts(rec: DocRecord, alias_index) -> list[dict]:
    """Document→Entity Reference(mentions) for alias-index hits in the title,
    description and opening text. No embedding, no LLM."""
    if alias_index is None:
        return []
    scan = " ".join([rec["title"], rec.get("description") or "", (rec.get("text") or "")[:MENTION_SCAN_CHARS]])
    hits = alias_index.match(scan, limit=MAX_MENTIONS * 2)
    doc = document_side(rec)
    facts = []
    seen_names: set[str] = set()
    for h in hits:
        label = h.labels[0] if h.labels else "Concept"
        if label in ("Document", "Entity"):
            continue
        if h.key in MENTION_BLOCKLIST_KEYS or h.name in seen_names:
            continue
        hangul = re.sub(r"[^가-힣]", "", h.key)
        if hangul and len(hangul) < MENTION_MIN_HANGUL and label != "Person":
            continue
        seen_names.add(h.name)
        if len(facts) >= MAX_MENTIONS:
            break
        obj = {"name": h.name, "type": label}
        facts.append(_fact(
            doc, "Reference", obj, f"문서 '{rec['title']}'에 {h.name}{josa(h.name, '이/가')} 언급된다",
            _doc_attrs(rec, f"mention:{h.uuid[:8]}", reference_type="mentions"),
            valid_at=rec.get("published_at"),
        ))
    return facts


# ── LLM extraction ────────────────────────────────────────────────────────────

_ENTITY_TYPES = "Person, Organization, Location, Asset, Incident, Policy, Campaign, Concept, Role, Industry"
_PREDICATES = ("Affiliation, PersonalRelation, OrgRelation, Funding, AssetTransfer, ThreatAction, "
               "Involvement, Presence, PolicyEffect, Participation, Statement, Causation")

EXTRACTION_SYSTEM = """You extract knowledge-graph facts from a document for a Korean-language political-economy knowledge base.
Return ONLY a JSON object: {"facts": [ ... ]}. Each fact:
{"subject_name": str, "subject_type": T, "predicate": P, "object_name": str, "object_type": T,
 "fact": str, "valid_at": "YYYY-MM-DD" | null, "subject_aliases": [str], "object_aliases": [str]}

T ∈ {%s}
P ∈ {%s}
Predicate rules by (subject_type → object_type):
  Affiliation: Person→Organization, Person→Role, Role→Organization, Organization→Industry
  PersonalRelation: Person→Person | OrgRelation: Organization→Organization ONLY
  Involvement: subject→Incident or subject→Campaign ONLY | Participation: Person/Organization→Campaign
  Presence: any→Location | PolicyEffect: Policy→any, Organization→Policy, Campaign→Policy
  ThreatAction: Person→Organization, Organization→Organization/Person, Campaign→Organization/Asset/Industry
  Funding, AssetTransfer, Statement, Causation: any→any (wildcards). Use Statement for "X said/argued/published Y".
Guidelines:
- At most %d facts; prefer the document's central, specific, dated claims. Skip trivia and rhetoric.
- Entity names: Korean canonical form for Korean people/organizations (e.g. 민주노총, 김문수); well-known
  international entities in English (e.g. United States, Anthropic, Nikita Khrushchev). Put the other
  language form in *_aliases. Countries and governments are Organization, not Location.
- Entities must be specific named things (proper nouns: a person, a named organization, a place, a
  named policy/event/work). NEVER use a generic common noun as an entity — not 국가, 정부, 개인, 경찰,
  청년, 주주, 기관, 외국인, "the state", "individuals", "workers" — and never a type label as a name.
  Attach such claims to the concrete actor the document names (e.g. 마르크스 —Statement→ 『국가와 혁명』),
  or drop the fact. Facts using generic entities are discarded.
- "fact" must be a self-contained sentence in the document's language, with dates/numbers when present.
- Never extract the document itself, its author's persona, internal task ids, file names or code.
""" % (_ENTITY_TYPES, _PREDICATES, MAX_LLM_FACTS)


def build_llm_prompt(rec: DocRecord) -> str:
    text = (rec.get("text") or "")[:MAX_LLM_CHARS]
    head = [f"Document: {rec['title']}", f"Kind: {rec['kind']}"]
    if rec.get("published_at"):
        head.append(f"Published: {rec['published_at']}")
    if rec.get("description"):
        head.append(f"Description: {rec['description'][:500]}")
    return "\n".join(head) + "\n\n---\n" + text


def parse_llm_facts(raw: str) -> list[dict]:
    """Parse the model's JSON; tolerate fences and a bare list."""
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except ValueError:
            return []
    facts = data.get("facts") if isinstance(data, dict) else data
    return [f for f in (facts or []) if isinstance(f, dict)]


def llm_facts(rec: DocRecord, raw_facts: list[dict]) -> list[dict]:
    """Validate model output against the agent schema (Document/Reference are
    NOT allowed here), stamp provenance, and add Document→Entity mentions."""
    from graph_memory.structured_writer import validate_fact
    from kg_runtime.identity import is_generic_entity_name

    doc = document_side(rec)
    out, seen_entities = [], set()
    for i, f in enumerate(raw_facts[:MAX_LLM_FACTS]):
        generic = [f.get(k) for k in ("subject_name", "object_name") if is_generic_entity_name(f.get(k))]
        if generic:
            logger.info("[doc-extract] %s: dropped fact %d: generic entity %s", rec.ref, i, generic)
            continue
        fact = {k: f.get(k) for k in ("subject_name", "subject_type", "predicate", "object_name",
                                      "object_type", "fact", "valid_at")}
        for side in ("subject", "object"):
            al = f.get(f"{side}_aliases")
            if isinstance(al, list):
                fact[f"{side}_aliases"] = [_clean(a) for a in al if a and _clean(a) != fact.get(f"{side}_name")]
        if not fact.get("valid_at"):
            fact.pop("valid_at", None)
        else:
            fact["valid_at"] = str(fact["valid_at"])[:10]
        fact["attributes"] = _doc_attrs(rec, f"llm:{i}", extraction="llm")
        err = validate_fact(fact, i)
        if err:
            logger.info("[doc-extract] %s: dropped fact %d: %s", rec.ref, i, err)
            continue
        out.append(fact)
        for side in ("subject", "object"):
            key = (fact[f"{side}_name"], fact[f"{side}_type"])
            if key in seen_entities:
                continue
            seen_entities.add(key)
            obj = {"name": key[0], "type": key[1], "aliases": fact.get(f"{side}_aliases") or []}
            out.append(_fact(
                doc, "Reference", obj, f"문서 '{rec['title']}'에 {key[0]}{josa(key[0], '이/가')} 언급된다",
                _doc_attrs(rec, f"llm-mention:{i}:{side}", reference_type="mentions"),
                valid_at=rec.get("published_at"),
            ))
    return out


def run_llm_extraction(rec: DocRecord) -> list[dict]:
    from llm.call_registry import generate_sync

    raw = generate_sync(LLM_FEATURE, build_llm_prompt(rec), system=EXTRACTION_SYSTEM)
    if not raw:
        logger.warning("[doc-extract] %s: LLM returned nothing", rec.ref)
        return []
    return llm_facts(rec, parse_llm_facts(raw))


# ── Per-document pipeline ─────────────────────────────────────────────────────

def build_document_facts(rec: DocRecord, *, names: dict[str, dict[str, str]] | None = None,
                         alias_index=None, use_llm: bool = False) -> list[dict]:
    facts = [collection_fact(rec)]
    facts.extend(curated_link_facts(rec, names or {}))
    # A curated "about" link already says the document covers the entity;
    # a weaker "mentions" link for the same name is noise.
    linked_names = {f["object_name"] for f in facts}
    facts.extend(f for f in mention_facts(rec, alias_index) if f["object_name"] not in linked_names)
    if use_llm:
        linked_names |= {f["object_name"] for f in facts if f["predicate"] == "Reference"}
        for f in run_llm_extraction(rec):
            if f["predicate"] == "Reference" and f["object_name"] in linked_names:
                continue
            facts.append(f)
    # dedupe by sync_key
    seen, out = set(), []
    for f in facts:
        k = f["attributes"]["sync_key"]
        if k in seen:
            continue
        seen.add(k)
        out.append(f)
    return out


def write_document_facts(rec: DocRecord, facts: list[dict]) -> dict:
    from kg_runtime.writes import add_kg_structured

    if not facts:
        return {"status": "ok", "facts_written": 0}
    provenance = f"document: {rec.ref} (sha256:{rec['sha'][:8]})" + (f"\nurl: {rec['url']}" if rec.get("url") else "")
    res = add_kg_structured(
        facts, group_id=GROUP_ID, agent=AGENT, trust_tier=TRUST_TIER[rec["kind"]],
        provenance_footer=provenance, allow_sync_predicates=True,
    )
    return res


def stamp_document_node(rec: DocRecord) -> None:
    """Store hash/url/kind on the Document node (queried for idempotency)."""
    from kg_runtime.search import _get_neo4j_sync_driver
    with _get_neo4j_sync_driver() as (drv, db):
        with drv.session(database=db) as s:
            s.run(
                "MATCH (n:Entity:Document) WHERE $ref IN coalesce(n.external_ids, []) "
                "SET n.content_sha256 = $sha, n.doc_kind = $kind, n.slug = $ident, n.url = $url, "
                "    n.lang = $lang, n.published_at = $pub, n.extracted_at = datetime()",
                ref=rec.ref, sha=rec["sha"], kind=rec["kind"], ident=rec["ident"], url=rec.get("url"),
                lang=rec.get("lang"), pub=rec.get("published_at"),
            ).consume()


def expire_document_edges(ref: str) -> int:
    from kg_runtime.search import _get_neo4j_sync_driver
    with _get_neo4j_sync_driver() as (drv, db):
        with drv.session(database=db) as s:
            rec = s.run(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.doc_ref = $ref AND r.expired_at IS NULL "
                "SET r.expired_at = datetime() RETURN count(r) AS cnt", ref=ref,
            ).single()
            return rec["cnt"] if rec else 0


def existing_document_hashes(prefix: str) -> dict[str, str]:
    from kg_runtime.search import _get_neo4j_sync_driver
    with _get_neo4j_sync_driver() as (drv, db):
        with drv.session(database=db) as s:
            rows = s.run(
                "MATCH (n:Entity:Document) UNWIND coalesce(n.external_ids, []) AS ref "
                "WITH n, ref WHERE ref STARTS WITH $p RETURN ref, n.content_sha256 AS sha", p=prefix,
            )
            return {r["ref"]: r["sha"] for r in rows}


def extract_document(rec: DocRecord, *, names=None, alias_index=None, use_llm: bool | None = None,
                     force: bool = False, existing_sha: str | None = None) -> dict:
    """Full per-document pipeline: skip unchanged, expire old edges when the
    content changed, build facts, write, stamp the node."""
    use_llm = llm_enabled() if use_llm is None else use_llm
    if not force and existing_sha and existing_sha == rec["sha"]:
        return {"ref": rec.ref, "status": "unchanged"}
    expired = expire_document_edges(rec.ref) if existing_sha else 0
    facts = build_document_facts(rec, names=names, alias_index=alias_index, use_llm=use_llm)
    res = write_document_facts(rec, facts)
    stamp_document_node(rec)
    return {
        "ref": rec.ref, "status": res.get("status"), "facts": len(facts), "written": res.get("facts_written", 0),
        "rejected": res.get("facts_rejected", 0), "expired": expired, "llm": use_llm,
        "message": res.get("message", "")[:200],
    }


# ── Convenience for publish hooks ─────────────────────────────────────────────

def extract_research_by_slug(slug: str, *, use_llm: bool | None = None) -> dict:
    """Publish-hook entry: extract one public research document by slug.
    Never raises (returns {"error": ...}); the nightly job is the backstop."""
    try:
        from research_store import get_document
        from kg_runtime.identity import get_alias_index
        row = get_document(slug)
        if not row or row.get("status") != "public":
            return {"ref": f"research:{slug}", "status": "skipped", "reason": "not public"}
        rec = research_record(row)
        idx = get_alias_index()
        idx.ensure_loaded()
        existing = existing_document_hashes(rec.ref)
        return extract_document(rec, alias_index=idx, use_llm=use_llm, existing_sha=existing.get(rec.ref))
    except Exception as exc:
        logger.warning("[doc-extract] research %s failed: %s", slug, exc)
        return {"ref": f"research:{slug}", "error": str(exc)}
