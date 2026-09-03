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

IDENTITY_PROPS = ("external_ids", "aliases", "alias_keys", "weak_keys", "name_ko", "name_en", "alias_text")

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


def is_weak_alias(alias, name) -> bool:
    """A *weak* alias is one that cannot identify the entity on its own: a
    single token of a multi-token name (surnames: "카스트로", "Khrushchev",
    "Хрущёв") or a very short single token. Weak aliases are kept for display
    and for unambiguous search matching, but never drive entity resolution —
    on 2026-09-03 surname aliases merged Fidel and Raúl Castro into one node.
    """
    ak, nk = normalize_alias_key(alias), normalize_alias_key(name)
    if not ak or ak == nk:
        return False
    a_tokens, n_tokens = ak.split(" "), nk.split(" ")
    if len(a_tokens) == 1 and len(n_tokens) >= 2:
        return True
    if len(a_tokens) == 1 and len(ak) <= 3:
        return True
    return False


def split_alias_keys(name: str, aliases, *extra_names) -> tuple[list[str], list[str]]:
    """(strong_keys, weak_keys) for a node: the name and any full-form alias
    (and name_ko/name_en) are strong; surname-like aliases are weak."""
    strong = alias_keys_for(name, *[x for x in extra_names if x])
    weak: list[str] = []
    for a in aliases or ():
        if not a:
            continue
        k = normalize_alias_key(a)
        if not k or k in strong or k in weak:
            continue
        if is_weak_alias(a, name):
            weak.append(k)
        else:
            strong.append(k)
    return strong, weak


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
    strong, weak = split_alias_keys(name, alias_list, name_ko, name_en)
    props = {
        "external_ids": _dedupe_strings(external_ids),
        "aliases": alias_list,
        "alias_keys": strong,
        "weak_keys": weak,
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
       ($etype IN labels(n)) AS same_label, coalesce(n.external_ids, []) AS external_ids
ORDER BY same_label DESC, rels DESC
LIMIT 5
"""

# Weak-key fallback: only when exactly ONE same-label node carries the key
# (an unambiguous surname). Two Castros → no match → a new node.
CYPHER_RESOLVE_BY_WEAK_KEY = """
MATCH (n:Entity)
WHERE $etype IN labels(n) AND any(k IN $keys WHERE k IN coalesce(n.weak_keys, []))
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, coalesce(n.external_ids, []) AS external_ids
LIMIT 2
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
     coalesce(n.alias_keys, []) + $alias_keys AS keys_raw,
     coalesce(n.weak_keys, []) + $weak_keys AS weak_raw
WITH n,
     reduce(acc = [], x IN ids_raw | CASE WHEN x IS NULL OR x = '' OR x IN acc THEN acc ELSE acc + x END) AS ids,
     reduce(acc = [], x IN aliases_raw | CASE WHEN x IS NULL OR x = '' OR x = n.name OR x IN acc THEN acc ELSE acc + x END) AS als,
     reduce(acc = [], x IN keys_raw | CASE WHEN x IS NULL OR x = '' OR x IN acc THEN acc ELSE acc + x END) AS keys,
     reduce(acc = [], x IN weak_raw | CASE WHEN x IS NULL OR x = '' OR x IN acc THEN acc ELSE acc + x END) AS weak
SET n.external_ids = ids,
    n.aliases = als,
    n.alias_keys = keys,
    n.weak_keys = [w IN weak WHERE NOT w IN keys],
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
     coalesce(canon.alias_keys, []) + [toLower(dup.name)] + coalesce(dup.alias_keys, []) AS keys_raw,
     coalesce(canon.weak_keys, []) + coalesce(dup.weak_keys, []) AS weak_raw
WITH canon, dup, weak_raw,
     reduce(acc = [], x IN ids_raw | CASE WHEN x IS NULL OR x = '' OR x IN acc THEN acc ELSE acc + x END) AS ids,
     reduce(acc = [], x IN aliases_raw | CASE WHEN x IS NULL OR x = '' OR x = canon.name OR x IN acc THEN acc ELSE acc + x END) AS als,
     reduce(acc = [], x IN keys_raw | CASE WHEN x IS NULL OR x = '' OR x IN acc THEN acc ELSE acc + x END) AS keys
SET canon.external_ids = ids,
    canon.aliases = als,
    canon.alias_keys = keys,
    canon.weak_keys = [w IN reduce(acc = [], x IN weak_raw | CASE WHEN x IS NULL OR x = '' OR x IN acc THEN acc ELSE acc + x END) WHERE NOT w IN keys],
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
       coalesce(n.alias_keys, []) AS keys, coalesce(n.weak_keys, []) AS weak_keys
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
    """Exact names and strong keys used for lookup. Weak (surname-like)
    aliases of the incoming entity are NOT used — they would attach it to
    any namesake."""
    strong_aliases = [a for a in _dedupe_strings(aliases) if not is_weak_alias(a, name)]
    names = _dedupe_strings([name, *strong_aliases])
    keys = alias_keys_for(names)
    return names, keys


def _id_namespace(external_id: str | None) -> str | None:
    """'commulingo:person:khrushchev' -> 'commulingo:person:' (one id per node per namespace)."""
    if not external_id or ":" not in external_id:
        return None
    return external_id.rsplit(":", 1)[0] + ":"


def _namespace_conflict(row: dict, external_id: str | None) -> bool:
    """True when the candidate already carries a *different* id from the same
    namespace — a CommuLingo namesake (two 'Vladimir Komarov' entries) must
    never fold into one node, whatever their aliases say."""
    ns = _id_namespace(external_id)
    if not ns:
        return False
    return any(x.startswith(ns) and x != external_id for x in (row.get("external_ids") or []))


def _filter_rows(rows: list[dict], *, exclude_uuid: str | None, external_id: str | None, name: str) -> list[dict]:
    out = []
    for r in rows:
        if exclude_uuid and r.get("uuid") == exclude_uuid:
            continue
        if _namespace_conflict(r, external_id):
            logger.info("[KG identity] '%s' (%s) matches '%s' which already carries another id in that namespace — not reused",
                        name, external_id, r.get("name"))
            continue
        out.append(r)
    return out


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


def _pick_weak_hit(rows: list[dict], name: str) -> ResolveResult:
    if len(rows) == 1:
        r = rows[0]
        return ResolveResult(r["uuid"], "weak_alias", r.get("name"), list(r.get("labels") or []))
    if len(rows) > 1:
        logger.info("[KG identity] '%s' matches several nodes by surname alias — not reused", name)
    return ResolveResult(None, "none")


def resolve_entity_sync(
    session,
    *,
    name: str,
    entity_type: str,
    external_id: str | None = None,
    aliases=(),
    exclude_uuid: str | None = None,
) -> ResolveResult:
    """Deterministic resolution on a sync neo4j session (jobs, scripts).
    ``exclude_uuid`` lets maintenance passes resolve a node against everything
    but itself."""
    if external_id:
        rec = session.run(CYPHER_RESOLVE_BY_EXTERNAL_ID, eid=external_id).single()
        if rec:
            return ResolveResult(rec["uuid"], "external_id", rec["name"], list(rec["labels"] or []))
    names, keys = _lookup_params(name, aliases)
    if not names:
        return ResolveResult(None, "none")
    rows = [dict(r) for r in session.run(CYPHER_RESOLVE_BY_KEY, names=names, keys=keys, etype=entity_type)]
    rows = _filter_rows(rows, exclude_uuid=exclude_uuid, external_id=external_id, name=name)
    hit = _pick_key_hit(rows, entity_type, name)
    if hit.found or hit.method == "label_conflict":
        return hit
    weak_rows = [dict(r) for r in session.run(CYPHER_RESOLVE_BY_WEAK_KEY, keys=keys, etype=entity_type)]
    weak_rows = _filter_rows(weak_rows, exclude_uuid=exclude_uuid, external_id=external_id, name=name)
    return _pick_weak_hit(weak_rows, name)


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
    rows = _filter_rows([dict(r) async for r in result], exclude_uuid=None, external_id=external_id, name=name)
    hit = _pick_key_hit(rows, entity_type, name)
    if hit.found or hit.method == "label_conflict":
        return hit
    result = await session.run(CYPHER_RESOLVE_BY_WEAK_KEY, keys=keys, etype=entity_type)
    weak_rows = _filter_rows([dict(r) async for r in result], exclude_uuid=None, external_id=external_id, name=name)
    hit = _pick_weak_hit(weak_rows, name)
    if hit.found:
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

def _upsert_params(uuid, external_ids, aliases, name_ko, name_en, summary, name=None) -> dict:
    alias_list = _dedupe_strings(aliases)
    for extra in (name_ko, name_en):
        if extra and extra not in alias_list:
            alias_list.append(extra)
    strong, weak = split_alias_keys(name or "", alias_list, name_ko, name_en)
    if not name:
        # unknown canonical name: nothing can be judged weak against it, but a
        # single short token is still too ambiguous to resolve on
        strong = [k for k in alias_keys_for(alias_list, name_ko, name_en) if len(k.split(" ")) > 1 or len(k) > 3]
        weak = [k for k in alias_keys_for(alias_list, name_ko, name_en) if k not in strong]
    return {
        "uuid": uuid,
        "external_ids": _dedupe_strings(external_ids),
        "aliases": alias_list,
        "alias_keys": strong,
        "weak_keys": weak,
        "name_ko": name_ko,
        "name_en": name_en,
        "summary": summary,
    }


def upsert_identity_sync(session, uuid: str, *, external_ids=(), aliases=(),
                         name_ko=None, name_en=None, summary=None, name=None) -> dict | None:
    """Union external ids / aliases into an existing node (sync session).
    ``name`` (the node's canonical name, if known) decides which aliases are
    weak; without it single short tokens are treated as weak."""
    params = _upsert_params(uuid, external_ids, aliases, name_ko, name_en, summary, name)
    rec = session.run(CYPHER_UPSERT_IDENTITY, **params).single()
    return dict(rec) if rec else None


async def upsert_identity_async(session, uuid: str, *, external_ids=(), aliases=(),
                                name_ko=None, name_en=None, summary=None, name=None) -> dict | None:
    params = _upsert_params(uuid, external_ids, aliases, name_ko, name_en, summary, name)
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
        weak: dict[str, list[tuple[str, str, list[str]]]] = {}
        for r in rows:
            uuid, name = r.get("uuid"), r.get("name")
            if not uuid or not name:
                continue
            labels = [l for l in (r.get("labels") or []) if l != "Entity"]
            entry = (uuid, name, labels)
            for k in alias_keys_for(name, *(r.get("keys") or [])):
                keys.setdefault(k, []).append(entry)
            for k in alias_keys_for(*(r.get("weak_keys") or [])):
                weak.setdefault(k, []).append(entry)
        # Surname-style keys only count when they point at exactly one entity
        # and no entity owns that key as a strong one.
        for k, entries in weak.items():
            if k not in keys and len({e[0] for e in entries}) == 1:
                keys[k] = entries[:1]
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
