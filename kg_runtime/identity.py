"""Entity identity layer for the knowledge graph.

Every Entity node may carry identity properties beyond graphiti's own:

- ``external_ids``  list[str]  stable ids from other stores
                                (``commulingo:person:karl-marx``, ``research:<slug>``)
- ``aliases``       list[str]  display aliases in any language
- ``alias_keys``    list[str]  normalized keys (see ``normalize_alias_key``) for
                                the name, name_ko, name_en and every alias
- ``name_ko`` / ``name_en``    canonical names per language when known
- ``alias_text``    str        aliases joined with " / " (fulltext-indexable)

The resolver here is deterministic and used by both the structured writer
and the sync jobs, so the same real-world entity converges on one node
regardless of which store or agent mentioned it first. Order:

1. external id membership (definitive, label ignored)
2. normalized alias key or exact name match, any group_id — same label
   required; a different-label hit is logged and ignored
3. optional name-embedding nearest neighbour (async only, gated by
   ``KG_RESOLVE_EMBEDDING_NN=1``, same label, cosine >= threshold)

graphiti saves nodes with ``SET n = $entity_data`` where ``entity_data``
includes ``node.attributes``; the patched record loader in
``graph_memory/graphiti_patches.py`` keeps unknown properties as
attributes, so these props survive graphiti re-saves.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field

from graph_memory.config import NAME_NORMALIZATION

logger = logging.getLogger(__name__)

IDENTITY_PROPS = ("external_ids", "aliases", "alias_keys", "name_ko", "name_en", "alias_text")

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\"'“”‘’`´.,;:!?()\[\]{}<>«»、。·‧#*|/\\]")
_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")

EMBED_NN_THRESHOLD = float(os.getenv("KG_RESOLVE_EMBEDDING_NN_THRESHOLD", "0.92"))


# ── Normalization ────────────────────────────────────────────────────────────

def normalize_alias_key(value) -> str:
    """Return the normalized lookup key for a name or alias.

    NFKC, lowercase, punctuation and hyphen/underscore to spaces, whitespace
    collapsed. ``NAME_NORMALIZATION`` (abbreviation -> canonical) is applied as
    the final step so ``"US"`` and ``"United States"`` share one key.
    """
    if value is None:
        return ""
    lowered = unicodedata.normalize("NFKC", str(value)).strip().lower()
    if not lowered:
        return ""
    text = _strip_key(lowered)
    # NAME_NORMALIZATION keys are raw lowercase forms ("u.s.a.", "usa", "rok");
    # try the raw, dot-less and stripped forms before giving up.
    for candidate in (lowered, lowered.replace(".", ""), text):
        canonical = NAME_NORMALIZATION.get(candidate)
        if canonical:
            return _strip_key(unicodedata.normalize("NFKC", canonical).lower())
    return text


def _strip_key(text: str) -> str:
    text = _PUNCT_RE.sub(" ", text)
    text = text.replace("-", " ").replace("_", " ").replace("–", " ").replace("—", " ")
    return _WS_RE.sub(" ", text).strip()


def alias_keys_for(*values) -> list[str]:
    """Normalized, deduplicated, order-preserving keys for the given strings."""
    keys: list[str] = []
    for v in values:
        candidates = alias_keys_for(*v) if isinstance(v, (list, tuple, set)) else [normalize_alias_key(v)]
        for k in candidates:
            if k and k not in keys:
                keys.append(k)
    return keys


def _dedupe_strings(values) -> list[str]:
    out: list[str] = []
    for v in values or ():
        if v is None:
            continue
        s = str(v).strip()
        if s and s not in out:
            out.append(s)
    return out


def build_identity_props(
    name: str,
    *,
    aliases=(),
    external_ids=(),
    name_ko: str | None = None,
    name_en: str | None = None,
) -> dict:
    """Identity attributes for a NEW node (stored via ``EntityNode.attributes``)."""
    alias_list = _dedupe_strings([a for a in aliases if a and str(a).strip() != name])
    for extra in (name_ko, name_en):
        if extra and extra != name and extra not in alias_list:
            alias_list.append(extra)
    props = {
        "external_ids": _dedupe_strings(external_ids),
        "aliases": alias_list,
        "alias_keys": alias_keys_for(name, name_ko, name_en, alias_list),
        "alias_text": " / ".join(alias_list),
    }
    if name_ko:
        props["name_ko"] = name_ko
    if name_en:
        props["name_en"] = name_en
    return props


# ── Cypher (shared by sync and async runners) ────────────────────────────────

CYPHER_RESOLVE_BY_EXTERNAL_ID = """
MATCH (n:Entity)
WHERE $eid IN coalesce(n.external_ids, [])
OPTIONAL MATCH (n)-[r:RELATES_TO]-()
WITH n, count(r) AS rels
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, rels
ORDER BY rels DESC
LIMIT 1
"""

CYPHER_RESOLVE_BY_KEY = """
MATCH (n:Entity)
WHERE n.name IN $names
   OR any(k IN $keys WHERE k IN coalesce(n.alias_keys, []))
   OR toLower(n.name) IN $keys
OPTIONAL MATCH (n)-[r:RELATES_TO]-()
WITH n, count(r) AS rels
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, rels,
       ($etype IN labels(n)) AS same_label
ORDER BY same_label DESC, rels DESC
LIMIT 5
"""

CYPHER_RESOLVE_BY_EMBEDDING = """
CALL db.index.vector.queryNodes('entity_name_embedding', 5, $vec)
YIELD node, score
WHERE $etype IN labels(node) AND score >= $threshold
RETURN node.uuid AS uuid, node.name AS name, labels(node) AS labels, score
ORDER BY score DESC
LIMIT 1
"""

CYPHER_UPSERT_IDENTITY = """
MATCH (n:Entity {uuid: $uuid})
WITH n,
     coalesce(n.external_ids, []) + $external_ids AS ids_raw,
     coalesce(n.aliases, []) + $aliases AS aliases_raw,
     coalesce(n.alias_keys, []) + $alias_keys AS keys_raw
WITH n,
     reduce(acc = [], x IN ids_raw | CASE WHEN x IS NULL OR x = '' OR x IN acc THEN acc ELSE acc + x END) AS ids,
     reduce(acc = [], x IN aliases_raw | CASE WHEN x IS NULL OR x = '' OR x = n.name OR x IN acc THEN acc ELSE acc + x END) AS als,
     reduce(acc = [], x IN keys_raw | CASE WHEN x IS NULL OR x = '' OR x IN acc THEN acc ELSE acc + x END) AS keys
SET n.external_ids = ids,
    n.aliases = als,
    n.alias_keys = keys,
    n.alias_text = reduce(s = '', a IN als | s + CASE WHEN s = '' THEN '' ELSE ' / ' END + a),
    n.name_ko = coalesce($name_ko, n.name_ko),
    n.name_en = coalesce($name_en, n.name_en),
    n.summary = CASE WHEN $summary IS NOT NULL AND $summary <> '' AND coalesce(n.summary, '') = ''
                     THEN $summary ELSE n.summary END
RETURN n.uuid AS uuid, size(ids) AS external_ids, size(als) AS aliases
"""

# Merge: move RELATES_TO (both directions, skipping same-predicate duplicates
# to the same neighbour), move MENTIONS, fill summary, union identity, delete.
CYPHER_MERGE_OUT = """
MATCH (dup:Entity {uuid: $dup_uuid})-[r:RELATES_TO]->(t)
WHERE t.uuid <> $canon_uuid
WITH dup, r, t
OPTIONAL MATCH (:Entity {uuid: $canon_uuid})-[existing:RELATES_TO]->(t)
WHERE existing.name = r.name
WITH dup, r, t, existing
WHERE existing IS NULL
MATCH (canon:Entity {uuid: $canon_uuid})
CREATE (canon)-[r2:RELATES_TO]->(t)
SET r2 = properties(r)
DELETE r
RETURN count(r2) AS cnt
"""

CYPHER_MERGE_IN = """
MATCH (src)-[r:RELATES_TO]->(dup:Entity {uuid: $dup_uuid})
WHERE src.uuid <> $canon_uuid
WITH src, r, dup
OPTIONAL MATCH (src)-[existing:RELATES_TO]->(:Entity {uuid: $canon_uuid})
WHERE existing.name = r.name
WITH src, r, dup, existing
WHERE existing IS NULL
MATCH (canon:Entity {uuid: $canon_uuid})
CREATE (src)-[r2:RELATES_TO]->(canon)
SET r2 = properties(r)
DELETE r
RETURN count(r2) AS cnt
"""

CYPHER_MERGE_MENTIONS = """
MATCH (ep)-[r:MENTIONS]->(dup:Entity {uuid: $dup_uuid})
WITH ep, r, dup
OPTIONAL MATCH (ep)-[existing:MENTIONS]->(:Entity {uuid: $canon_uuid})
WITH ep, r, dup, existing
WHERE existing IS NULL
MATCH (canon:Entity {uuid: $canon_uuid})
CREATE (ep)-[r2:MENTIONS]->(canon)
SET r2 = properties(r)
DELETE r
RETURN count(r2) AS cnt
"""

CYPHER_MERGE_IDENTITY = """
MATCH (canon:Entity {uuid: $canon_uuid}), (dup:Entity {uuid: $dup_uuid})
WITH canon, dup,
     coalesce(canon.external_ids, []) + coalesce(dup.external_ids, []) AS ids_raw,
     coalesce(canon.aliases, []) + [dup.name] + coalesce(dup.aliases, []) AS aliases_raw,
     coalesce(canon.alias_keys, []) + [toLower(dup.name)] + coalesce(dup.alias_keys, []) AS keys_raw
WITH canon, dup,
     reduce(acc = [], x IN ids_raw | CASE WHEN x IS NULL OR x = '' OR x IN acc THEN acc ELSE acc + x END) AS ids,
     reduce(acc = [], x IN aliases_raw | CASE WHEN x IS NULL OR x = '' OR x = canon.name OR x IN acc THEN acc ELSE acc + x END) AS als,
     reduce(acc = [], x IN keys_raw | CASE WHEN x IS NULL OR x = '' OR x IN acc THEN acc ELSE acc + x END) AS keys
SET canon.external_ids = ids,
    canon.aliases = als,
    canon.alias_keys = keys,
    canon.alias_text = reduce(s = '', a IN als | s + CASE WHEN s = '' THEN '' ELSE ' / ' END + a),
    canon.name_ko = coalesce(canon.name_ko, dup.name_ko),
    canon.name_en = coalesce(canon.name_en, dup.name_en),
    canon.summary = CASE WHEN coalesce(canon.summary, '') = '' THEN coalesce(dup.summary, '') ELSE canon.summary END
RETURN canon.uuid AS uuid
"""

CYPHER_MERGE_DELETE = """
MATCH (dup:Entity {uuid: $dup_uuid})
DETACH DELETE dup
RETURN count(*) AS cnt
"""

CYPHER_ALIAS_INDEX_LOAD = """
MATCH (n:Entity)
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels,
       coalesce(n.alias_keys, []) AS keys
"""

IDENTITY_INDEX_STATEMENTS = (
    "CREATE FULLTEXT INDEX entity_alias_text IF NOT EXISTS "
    "FOR (n:Entity) ON EACH [n.alias_text]",
)


# ── Resolution ───────────────────────────────────────────────────────────────

@dataclass
class ResolveResult:
    uuid: str | None
    method: str  # external_id | alias | embedding | none | label_conflict
    name: str | None = None
    labels: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.uuid is not None


def _lookup_params(name: str, aliases=()) -> tuple[list[str], list[str]]:
    names = _dedupe_strings([name, *aliases])
    keys = alias_keys_for(names)
    return names, keys


def _pick_key_hit(rows: list[dict], entity_type: str, name: str) -> ResolveResult:
    if not rows:
        return ResolveResult(None, "none")
    best = rows[0]
    if best.get("same_label"):
        return ResolveResult(best["uuid"], "alias", best.get("name"), list(best.get("labels") or []))
    logger.info(
        "[KG identity] label conflict: '%s' (%s) matches '%s' %s — not reused",
        name, entity_type, best.get("name"), [l for l in (best.get("labels") or []) if l != "Entity"],
    )
    return ResolveResult(None, "label_conflict", best.get("name"), list(best.get("labels") or []))


def resolve_entity_sync(
    session,
    *,
    name: str,
    entity_type: str,
    external_id: str | None = None,
    aliases=(),
) -> ResolveResult:
    """Deterministic resolution on a sync neo4j session (jobs, scripts)."""
    if external_id:
        rec = session.run(CYPHER_RESOLVE_BY_EXTERNAL_ID, eid=external_id).single()
        if rec:
            return ResolveResult(rec["uuid"], "external_id", rec["name"], list(rec["labels"] or []))
    names, keys = _lookup_params(name, aliases)
    if not names:
        return ResolveResult(None, "none")
    rows = [dict(r) for r in session.run(CYPHER_RESOLVE_BY_KEY, names=names, keys=keys, etype=entity_type)]
    return _pick_key_hit(rows, entity_type, name)


async def resolve_entity_async(
    session,
    *,
    name: str,
    entity_type: str,
    external_id: str | None = None,
    aliases=(),
    embedder=None,
) -> ResolveResult:
    """Same as ``resolve_entity_sync`` on an async session, plus the optional
    name-embedding nearest-neighbour step when ``KG_RESOLVE_EMBEDDING_NN=1``."""
    if external_id:
        result = await session.run(CYPHER_RESOLVE_BY_EXTERNAL_ID, eid=external_id)
        rec = await result.single()
        if rec:
            return ResolveResult(rec["uuid"], "external_id", rec["name"], list(rec["labels"] or []))
    names, keys = _lookup_params(name, aliases)
    if not names:
        return ResolveResult(None, "none")
    result = await session.run(CYPHER_RESOLVE_BY_KEY, names=names, keys=keys, etype=entity_type)
    rows = [dict(r) async for r in result]
    hit = _pick_key_hit(rows, entity_type, name)
    if hit.found or hit.method == "label_conflict":
        return hit
    if embedder is not None and embedding_nn_enabled():
        try:
            vec = await embedder.create(input_data=[name])
            result = await session.run(
                CYPHER_RESOLVE_BY_EMBEDDING, vec=vec, etype=entity_type, threshold=EMBED_NN_THRESHOLD,
            )
            rec = await result.single()
            if rec:
                logger.info("[KG identity] embedding NN reuse: '%s' -> '%s' (%.3f)", name, rec["name"], rec["score"])
                return ResolveResult(rec["uuid"], "embedding", rec["name"], list(rec["labels"] or []))
        except Exception as exc:  # NN is best-effort
            logger.debug("[KG identity] embedding NN skipped: %s", exc)
    return ResolveResult(None, "none")


def embedding_nn_enabled() -> bool:
    return os.getenv("KG_RESOLVE_EMBEDDING_NN", "0").strip().lower() in ("1", "true", "yes", "on")


# ── Identity upsert ──────────────────────────────────────────────────────────

def _upsert_params(uuid, external_ids, aliases, name_ko, name_en, summary) -> dict:
    alias_list = _dedupe_strings(aliases)
    for extra in (name_ko, name_en):
        if extra and extra not in alias_list:
            alias_list.append(extra)
    return {
        "uuid": uuid,
        "external_ids": _dedupe_strings(external_ids),
        "aliases": alias_list,
        "alias_keys": alias_keys_for(alias_list, name_ko, name_en),
        "name_ko": name_ko,
        "name_en": name_en,
        "summary": summary,
    }


def upsert_identity_sync(session, uuid: str, *, external_ids=(), aliases=(),
                         name_ko=None, name_en=None, summary=None) -> dict | None:
    """Union external ids / aliases into an existing node (sync session)."""
    params = _upsert_params(uuid, external_ids, aliases, name_ko, name_en, summary)
    rec = session.run(CYPHER_UPSERT_IDENTITY, **params).single()
    return dict(rec) if rec else None


async def upsert_identity_async(session, uuid: str, *, external_ids=(), aliases=(),
                                name_ko=None, name_en=None, summary=None) -> dict | None:
    params = _upsert_params(uuid, external_ids, aliases, name_ko, name_en, summary)
    result = await session.run(CYPHER_UPSERT_IDENTITY, **params)
    rec = await result.single()
    return dict(rec) if rec else None


# ── Merge ────────────────────────────────────────────────────────────────────

def _merge_stats(canonical_uuid: str) -> dict:
    return {"canonical_uuid": canonical_uuid, "merged": [], "edges_moved": 0, "mentions_moved": 0}


def merge_entity_nodes_sync(session, canonical_uuid: str, dup_uuids) -> dict:
    """Fold ``dup_uuids`` into ``canonical_uuid`` (sync session). Returns stats.

    Edges are moved (same-predicate edges to the same neighbour are dropped as
    duplicates), MENTIONS follow, the canonical summary is filled when empty,
    identity lists are unioned and the duplicate is DETACH DELETEd.
    """
    stats = _merge_stats(canonical_uuid)
    for dup_uuid in dup_uuids:
        if not dup_uuid or dup_uuid == canonical_uuid:
            continue
        params = {"canon_uuid": canonical_uuid, "dup_uuid": dup_uuid}
        stats["edges_moved"] += session.run(CYPHER_MERGE_OUT, **params).single()["cnt"]
        stats["edges_moved"] += session.run(CYPHER_MERGE_IN, **params).single()["cnt"]
        stats["mentions_moved"] += session.run(CYPHER_MERGE_MENTIONS, **params).single()["cnt"]
        session.run(CYPHER_MERGE_IDENTITY, **params).consume()
        session.run(CYPHER_MERGE_DELETE, **params).consume()
        stats["merged"].append(dup_uuid)
    return stats


async def merge_entity_nodes_async(session, canonical_uuid: str, dup_uuids) -> dict:
    stats = _merge_stats(canonical_uuid)
    for dup_uuid in dup_uuids:
        if not dup_uuid or dup_uuid == canonical_uuid:
            continue
        params = {"canon_uuid": canonical_uuid, "dup_uuid": dup_uuid}
        for cypher, key in ((CYPHER_MERGE_OUT, "edges_moved"), (CYPHER_MERGE_IN, "edges_moved"),
                            (CYPHER_MERGE_MENTIONS, "mentions_moved")):
            result = await session.run(cypher, **params)
            rec = await result.single()
            stats[key] += rec["cnt"] if rec else 0
        for cypher in (CYPHER_MERGE_IDENTITY, CYPHER_MERGE_DELETE):
            result = await session.run(cypher, **params)
            await result.consume()
        stats["merged"].append(dup_uuid)
    return stats


async def post_episode_merge(session, nodes) -> list[dict]:
    """After a graphiti free-text episode, fold freshly created nodes into
    existing same-label nodes that share an alias key across group_ids.

    graphiti only searches resolution candidates inside the episode's own
    group_id, which is how one entity became five nodes. Deterministic; no LLM.
    """
    merged: list[dict] = []
    for node in nodes or ():
        uuid = getattr(node, "uuid", None)
        name = getattr(node, "name", None)
        labels = [l for l in (getattr(node, "labels", None) or []) if l != "Entity"]
        if not uuid or not name:
            continue
        etype = labels[0] if labels else "Entity"
        names, keys = _lookup_params(name)
        result = await session.run(CYPHER_RESOLVE_BY_KEY, names=names, keys=keys, etype=etype)
        rows = [dict(r) async for r in result]
        candidates = [r for r in rows if r["uuid"] != uuid and r.get("same_label")]
        if not candidates:
            continue
        canonical = candidates[0]
        stats = await merge_entity_nodes_async(session, canonical["uuid"], [uuid])
        logger.info("[KG identity] post-episode merge: '%s' -> '%s' (%s)", name, canonical["name"], canonical["uuid"][:8])
        merged.append({"name": name, "into": canonical["name"], "canonical_uuid": canonical["uuid"], **stats})
    return merged


# ── Indexes ──────────────────────────────────────────────────────────────────

async def ensure_identity_indexes(driver_client, database: str) -> None:
    """Create the identity fulltext index (idempotent). Called after graphiti's
    own ``build_indices_and_constraints``."""
    async with driver_client.session(database=database) as session:
        for stmt in IDENTITY_INDEX_STATEMENTS:
            result = await session.run(stmt)
            await result.consume()


# ── In-process alias index (for search / entity-gated recall) ────────────────

@dataclass
class AliasHit:
    uuid: str
    name: str
    labels: list[str]
    key: str


class AliasIndex:
    """Cache ``alias_key -> [(uuid, name, labels)]`` for cheap text matching.

    Loaded with one Cypher; refreshed after ``ttl`` seconds. ``match()`` needs
    no embedding call: Korean keys match as substrings (particles attach to
    names), Latin keys match as whole tokens.
    """

    def __init__(self, ttl: float = 600.0, min_hangul: int = 2, min_latin: int = 4):
        self.ttl = ttl
        self.min_hangul = min_hangul
        self.min_latin = min_latin
        self._loaded_at = 0.0
        self._keys: dict[str, list[tuple[str, str, list[str]]]] = {}
        self._lock = threading.Lock()

    # loading -------------------------------------------------------------
    def load_rows(self, rows) -> None:
        keys: dict[str, list[tuple[str, str, list[str]]]] = {}
        for r in rows:
            uuid, name = r.get("uuid"), r.get("name")
            if not uuid or not name:
                continue
            labels = [l for l in (r.get("labels") or []) if l != "Entity"]
            for k in alias_keys_for(name, *(r.get("keys") or [])):
                keys.setdefault(k, []).append((uuid, name, labels))
        with self._lock:
            self._keys = keys
            self._loaded_at = time.monotonic()

    def is_stale(self) -> bool:
        return (time.monotonic() - self._loaded_at) > self.ttl or not self._keys

    def refresh_from_neo4j(self) -> None:
        from kg_runtime.search import _get_neo4j_sync_driver
        with _get_neo4j_sync_driver() as (drv, db):
            with drv.session(database=db) as s:
                rows = [dict(r) for r in s.run(CYPHER_ALIAS_INDEX_LOAD)]
        self.load_rows(rows)

    def ensure_loaded(self) -> bool:
        if not self.is_stale():
            return True
        try:
            self.refresh_from_neo4j()
            return True
        except Exception as exc:
            logger.debug("[KG identity] alias index refresh failed: %s", exc)
            return bool(self._keys)

    # matching ------------------------------------------------------------
    def _eligible(self, key: str) -> bool:
        if _HANGUL_RE.search(key):
            return len(key.replace(" ", "")) >= self.min_hangul
        return len(key) >= self.min_latin

    def match(self, text: str, limit: int = 5) -> list[AliasHit]:
        """Entities whose alias key occurs in ``text``; longest keys first,
        overlapping shorter keys dropped."""
        norm = normalize_alias_key(text)
        if not norm:
            return []
        padded = f" {norm} "
        latin_tokens = set(_LATIN_TOKEN_RE.findall(norm))
        with self._lock:
            keys = list(self._keys.items())
        hits: list[tuple[str, list[tuple[str, str, list[str]]]]] = []
        for key, entries in keys:
            if not self._eligible(key):
                continue
            if _HANGUL_RE.search(key):
                if key in norm:
                    hits.append((key, entries))
            elif " " in key:
                if f" {key} " in padded:
                    hits.append((key, entries))
            elif key in latin_tokens:
                hits.append((key, entries))
        hits.sort(key=lambda kv: -len(kv[0]))
        out: list[AliasHit] = []
        covered: list[str] = []
        seen_uuid: set[str] = set()
        for key, entries in hits:
            if any(key in c for c in covered):
                continue
            covered.append(key)
            for uuid, name, labels in entries:
                if uuid in seen_uuid:
                    continue
                seen_uuid.add(uuid)
                out.append(AliasHit(uuid, name, labels, key))
                if len(out) >= limit:
                    return out
        return out


_alias_index: AliasIndex | None = None


def get_alias_index() -> AliasIndex:
    global _alias_index
    if _alias_index is None:
        _alias_index = AliasIndex()
    return _alias_index
