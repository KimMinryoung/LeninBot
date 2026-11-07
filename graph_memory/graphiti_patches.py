"""
Graphiti 런타임 패치 — .venv 수정 없이 프롬프트/직렬화 동작을 오버라이드
====================================================================

Render 등 원격 배포 환경에서는 pip install 시 원본 코드가 설치되므로,
.venv 파일 수정 대신 런타임 몽키패치로 동일 효과를 달성한다.

패치 목록:
1. to_prompt_json — Neo4j DateTime JSON 직렬화 지원
2. extract_edges.Edge — relation_type 필드 SCREAMING_SNAKE_CASE → PascalCase
3. extract_edges.edge() — RELATION TYPE RULES 프롬프트 교체

service.py의 initialize()에서 1회 호출.
"""

import json
import logging
import re
from datetime import datetime, date
from typing import Any

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False

# ── 원본 텍스트 (Graphiti 기본값) ─────────────────────────────────────
_ORIGINAL_RELATION_RULES = (
    "- If FACT_TYPES are provided and the relationship matches one of the types "
    "(considering the entity type signature), use that fact_type_name as the `relation_type`.\n"
    "- Otherwise, derive a `relation_type` from the relationship predicate in "
    "SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH)."
)

_PATCHED_RELATION_RULES = (
    "- If FACT_TYPES are provided, you MUST pick the closest matching fact_type_name "
    "as the `relation_type`. Always prefer a FACT_TYPE over inventing a new type.\n"
    "- Only if no FACT_TYPE is remotely applicable, use a concise PascalCase label "
    "(e.g., MilitaryAction, TradeDispute). Do NOT use SCREAMING_SNAKE_CASE."
)

_PATCHED_FIELD_DESC = (
    "The type of relationship between the entities. "
    "Use the FACT_TYPES if provided; otherwise use a concise PascalCase label."
)


# ── Neo4j DateTime 인코더 ────────────────────────────────────────────
class _Neo4jDateTimeEncoder(json.JSONEncoder):
    """Neo4j DateTime / Python datetime → ISO 문자열 직렬화."""
    def default(self, obj: Any) -> Any:
        if hasattr(obj, "iso_format"):
            return obj.iso_format()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def _patched_to_prompt_json(
    data: Any, ensure_ascii: bool = False, indent: int | None = None
) -> str:
    return json.dumps(
        data, ensure_ascii=ensure_ascii, indent=indent, cls=_Neo4jDateTimeEncoder
    )


# ── 패치 적용 ────────────────────────────────────────────────────────
def apply_graphiti_patches() -> None:
    """Graphiti 런타임 패치 적용. 여러 번 호출해도 안전(멱등)."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    _patch_to_prompt_json()
    _patch_extract_edges()
    _patch_record_parsers()
    _patch_vector_search()
    _build_name_normalization_regex()

    _PATCHES_APPLIED = True


# ── Vector-index similarity search ──────────────────────────────────
# graphiti's Neo4j path scores every node/relation with
# vector.similarity.cosine. At 18k nodes / 57k relations (2026-11-07) that was
# 1.3 s + 2.8 s of each semantic knowledge_graph_search, although the
# entity_name_embedding / edge_fact_embedding HNSW indexes exist. The index
# returns the K nearest candidates, and graphiti's filters (expired_at,
# validity, labels, group) apply to those. K over-fetches so filtered-out
# candidates rarely starve the result; endpoint-anchored edge searches keep
# graphiti's exact path.

VECTOR_SEARCH_MIN_K = 200
VECTOR_SEARCH_OVERFETCH = 20
VECTOR_SEARCH_MAX_K = 1000


def _vector_k(limit: int) -> int:
    return max(VECTOR_SEARCH_MIN_K, min(VECTOR_SEARCH_MAX_K, int(limit) * VECTOR_SEARCH_OVERFETCH))


def _index_score(min_score: float) -> float:
    # Neo4j cosine vector indexes report (1 + cosine) / 2.
    return (1.0 + float(min_score)) / 2.0


def _patch_vector_search() -> None:
    from graphiti_core.driver.driver import GraphProvider
    from graphiti_core.search import search as search_mod
    from graphiti_core.search import search_utils as su

    orig_edge = su.edge_similarity_search
    orig_node = su.node_similarity_search

    def _uses_index(driver) -> bool:
        return driver.search_interface is None and driver.provider == GraphProvider.NEO4J

    async def edge_similarity_search(driver, search_vector, source_node_uuid, target_node_uuid,
                                     search_filter, group_ids=None, limit=su.RELEVANT_SCHEMA_LIMIT,
                                     min_score=su.DEFAULT_MIN_SCORE):
        if not _uses_index(driver) or source_node_uuid is not None or target_node_uuid is not None:
            return await orig_edge(driver, search_vector, source_node_uuid, target_node_uuid,
                                   search_filter, group_ids, limit, min_score)
        filters, params = su.edge_search_filter_query_constructor(search_filter, driver.provider)
        if group_ids is not None:
            filters.append('e.group_id IN $group_ids')
            params['group_ids'] = group_ids
        where = (' WHERE ' + ' AND '.join(filters)) if filters else ''
        query = (
            "CALL db.index.vector.queryRelationships('edge_fact_embedding', $k, $search_vector) "
            "YIELD relationship AS e, score WHERE score > $index_min_score "
            "MATCH (n:Entity)-[e]->(m:Entity)" + where +
            " RETURN " + su.get_entity_edge_return_query(driver.provider) +
            " ORDER BY score DESC LIMIT $limit"
        )
        records, _, _ = await driver.execute_query(
            query, search_vector=search_vector, k=_vector_k(limit), limit=limit,
            index_min_score=_index_score(min_score), routing_='r', **params,
        )
        return [su.get_entity_edge_from_record(r, driver.provider) for r in records]

    async def node_similarity_search(driver, search_vector, search_filter, group_ids=None,
                                     limit=su.RELEVANT_SCHEMA_LIMIT, min_score=su.DEFAULT_MIN_SCORE):
        if not _uses_index(driver):
            return await orig_node(driver, search_vector, search_filter, group_ids, limit, min_score)
        filters, params = su.node_search_filter_query_constructor(search_filter, driver.provider)
        if group_ids is not None:
            filters.append('n.group_id IN $group_ids')
            params['group_ids'] = group_ids
        where = (' AND ' + ' AND '.join(filters)) if filters else ''
        query = (
            "CALL db.index.vector.queryNodes('entity_name_embedding', $k, $search_vector) "
            "YIELD node AS n, score WHERE n:Entity AND score > $index_min_score" + where +
            " RETURN " + su.get_entity_node_return_query(driver.provider) +
            " ORDER BY score DESC LIMIT $limit"
        )
        records, _, _ = await driver.execute_query(
            query, search_vector=search_vector, k=_vector_k(limit), limit=limit,
            index_min_score=_index_score(min_score), routing_='r', **params,
        )
        return [su.get_entity_node_from_record(r, driver.provider) for r in records]

    su.edge_similarity_search = edge_similarity_search
    su.node_similarity_search = node_similarity_search
    search_mod.edge_similarity_search = edge_similarity_search
    search_mod.node_similarity_search = node_similarity_search


# ── 레거시 엣지 방어 ─────────────────────────────────────────────────
# AuraDB→local 마이그레이션 이전 스키마의 edge 중 일부에 group_id/created_at이
# null로 남아 있음. 이 값들은 graphiti-core의 EntityEdge 모델에서 required라서
# knowledge_graph_search가 pydantic 검증에서 통째로 실패한다. 데이터 영구 수정
# (scripts/fix_legacy_kg_edges.py)과 별개로, 런타임에서도 placeholder로 채워서
# 검색이 안 깨지게 방어.

_LEGACY_GROUP_ID = "legacy"


def _patch_record_parsers() -> None:
    """entity_edge_from_record / entity_node_from_record — null 내구성 추가."""
    from datetime import datetime, timezone

    from graphiti_core.driver import record_parsers as rp_mod
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode
    from graphiti_core.helpers import parse_db_date

    _epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def _patched_entity_edge_from_record(record):
        attributes = dict(record['attributes'] or {})
        for k in ('uuid', 'source_node_uuid', 'target_node_uuid', 'fact',
                  'fact_embedding', 'name', 'group_id', 'episodes',
                  'created_at', 'expired_at', 'valid_at', 'invalid_at'):
            attributes.pop(k, None)

        group_id = record['group_id'] or _LEGACY_GROUP_ID
        created_at = parse_db_date(record['created_at']) or _epoch
        return EntityEdge(
            uuid=record['uuid'],
            source_node_uuid=record['source_node_uuid'],
            target_node_uuid=record['target_node_uuid'],
            fact=record['fact'] or '',
            fact_embedding=record.get('fact_embedding'),
            name=record['name'] or 'RELATES_TO',
            group_id=group_id,
            episodes=record['episodes'] or [],
            created_at=created_at,
            expired_at=parse_db_date(record['expired_at']),
            valid_at=parse_db_date(record['valid_at']),
            invalid_at=parse_db_date(record['invalid_at']),
            attributes=attributes,
        )

    def _patched_entity_node_from_record(record):
        attributes = dict(record['attributes'] or {})
        for k in ('uuid', 'name', 'group_id', 'name_embedding',
                  'summary', 'created_at', 'labels'):
            attributes.pop(k, None)
        group_id = record['group_id'] or _LEGACY_GROUP_ID
        labels = list(record.get('labels') or [])
        if group_id:
            dynamic_label = 'Entity_' + str(group_id).replace('-', '')
            if dynamic_label in labels:
                labels.remove(dynamic_label)
        return EntityNode(
            uuid=record['uuid'],
            name=record['name'] or '',
            name_embedding=record.get('name_embedding'),
            group_id=group_id,
            labels=labels,
            created_at=parse_db_date(record['created_at']) or _epoch,
            summary=record.get('summary') or '',
            attributes=attributes,
        )

    rp_mod.entity_edge_from_record = _patched_entity_edge_from_record
    rp_mod.entity_node_from_record = _patched_entity_node_from_record

    # Modules that imported the symbol by name need their local binding refreshed.
    for mod_path in (
        'graphiti_core.driver.neo4j.operations.search_ops',
        'graphiti_core.driver.neo4j.operations.entity_edge_ops',
        'graphiti_core.driver.neo4j.operations.entity_node_ops',
        'graphiti_core.driver.falkordb.operations.entity_edge_ops',
        'graphiti_core.driver.falkordb.operations.entity_node_ops',
        'graphiti_core.driver.neptune.operations.entity_edge_ops',
        'graphiti_core.driver.neptune.operations.entity_node_ops',
    ):
        try:
            import importlib
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        if hasattr(mod, 'entity_edge_from_record'):
            mod.entity_edge_from_record = _patched_entity_edge_from_record
        if hasattr(mod, 'entity_node_from_record'):
            mod.entity_node_from_record = _patched_entity_node_from_record


def _patch_to_prompt_json() -> None:
    """to_prompt_json에 Neo4j DateTime 인코더 주입."""
    import graphiti_core.prompts.prompt_helpers as helpers_mod

    helpers_mod.to_prompt_json = _patched_to_prompt_json

    # to_prompt_json을 `from ... import`한 모듈의 로컬 바인딩도 교체
    import graphiti_core.prompts.extract_edges as m1
    import graphiti_core.prompts.extract_nodes as m2
    import graphiti_core.prompts.dedupe_nodes as m3
    import graphiti_core.prompts.eval as m4
    import graphiti_core.prompts.summarize_nodes as m5
    import graphiti_core.search.search_helpers as m6

    for mod in [m1, m2, m3, m4, m5, m6]:
        if hasattr(mod, "to_prompt_json"):
            mod.to_prompt_json = _patched_to_prompt_json


def _patch_extract_edges() -> None:
    """Edge 모델 + edge() 프롬프트 패치."""
    from graphiti_core.prompts.extract_edges import Edge

    # 1) Pydantic 필드 description 교체 (클래스 수준 — 모든 참조에 전파)
    current_desc = Edge.model_fields["relation_type"].description or ""
    if "SCREAMING_SNAKE_CASE" in current_desc:
        Edge.model_fields["relation_type"].description = _PATCHED_FIELD_DESC
        Edge.model_rebuild()

    # 2) edge() 함수 래핑 — RELATION TYPE RULES 텍스트 교체
    import graphiti_core.prompts.extract_edges as edges_mod

    _original_edge_fn = edges_mod.edge

    def _patched_edge(context: dict[str, Any]):
        messages = _original_edge_fn(context)
        patched = False
        for msg in messages:
            if hasattr(msg, "content") and _ORIGINAL_RELATION_RULES in msg.content:
                msg.content = msg.content.replace(
                    _ORIGINAL_RELATION_RULES, _PATCHED_RELATION_RULES
                )
                patched = True
        if not patched:
            logger.warning("[graphiti_patches] edge prompt patch target not found — upstream may have changed")
        return messages

    # 모듈 레벨 교체
    edges_mod.edge = _patched_edge
    edges_mod.versions["edge"] = _patched_edge

    # prompt_library 인스턴스의 VersionWrapper.func 교체
    try:
        from graphiti_core.prompts.lib import prompt_library

        wrapper = getattr(
            getattr(prompt_library, "extract_edges", None), "edge", None
        )
        if wrapper and hasattr(wrapper, "func"):
            wrapper.func = _patched_edge
    except Exception as e:
        logger.warning("[graphiti_patches] prompt_library patch skipped: %s", e)


# ── 엔티티 이름 정규화 (약어 → 정식명) ──────────────────────────
# Graphiti의 entity resolution이 짧은 이름(US, UK 등)에서 실패하므로,
# 에피소드 본문 텍스트를 Graphiti에 전달하기 전에 약어를 정식명으로 치환.

_NAME_NORM_PATTERN: re.Pattern | None = None
_NAME_NORM_MAP: dict[str, str] = {}


def _build_name_normalization_regex() -> None:
    """NAME_NORMALIZATION 딕셔너리로부터 word-boundary 정규식 컴파일."""
    global _NAME_NORM_PATTERN, _NAME_NORM_MAP
    from .config import NAME_NORMALIZATION

    if not NAME_NORMALIZATION:
        return

    _NAME_NORM_MAP = {k.lower(): v for k, v in NAME_NORMALIZATION.items()}

    # 긴 패턴 먼저 매칭 (e.g., "u.s.a." before "u.s.")
    sorted_keys = sorted(_NAME_NORM_MAP.keys(), key=len, reverse=True)
    escaped = [re.escape(k) for k in sorted_keys]
    # \b 대신 ASCII word boundary 사용 — 마침표 포함 약어(U.S.)와
    # 한국어 조사("비숑은", "비숑 동지와") 양쪽에서 안정적으로 동작.
    pattern_str = r"(?<![A-Za-z0-9_])(?:" + "|".join(escaped) + r")(?![A-Za-z0-9_])"
    _NAME_NORM_PATTERN = re.compile(pattern_str, re.IGNORECASE)


def normalize_entity_names_in_text(text: str) -> str:
    """에피소드 본문 내 약어를 정식명으로 치환. service.py에서 호출."""
    if _NAME_NORM_PATTERN is None or not text:
        return text

    def _replace(match: re.Match) -> str:
        raw = match.group(0)
        key = raw.lower()
        # A short Latin abbreviation (us/uk/un/who/imf …) is only an
        # abbreviation when written in capitals. Case-insensitive matching
        # turned "Kim Jong Un" into "Kim Jong United Nations", "telling an
        # employee who …" into "… World Health Organization …" and the pronoun
        # "us" into "United States" (2026-09-04 audit).
        if _is_short_abbreviation(key) and raw.replace(".", "") != raw.replace(".", "").upper():
            return raw
        return _NAME_NORM_MAP.get(key, raw)

    return _NAME_NORM_PATTERN.sub(_replace, text)


def _is_short_abbreviation(key: str) -> bool:
    letters = key.replace(".", "")
    return letters.isascii() and letters.isalpha() and len(letters) <= 4
