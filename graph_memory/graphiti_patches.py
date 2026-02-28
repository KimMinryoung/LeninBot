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
from datetime import datetime, date
from typing import Any

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

    _PATCHES_APPLIED = True


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
        for msg in messages:
            if hasattr(msg, "content") and _ORIGINAL_RELATION_RULES in msg.content:
                msg.content = msg.content.replace(
                    _ORIGINAL_RELATION_RULES, _PATCHED_RELATION_RULES
                )
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
    except Exception:
        pass  # prompt_library 구조 변경 시 무시 — custom_extraction_instructions로 여전히 커버됨
