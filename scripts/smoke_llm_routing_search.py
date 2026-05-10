#!/usr/bin/env python3
"""Smoke checks for LLM-assisted routing/search fallbacks."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Doc:
    def __init__(self, content: str, source: str, chunk_index: int, layer: str):
        self.page_content = content
        self.metadata = {
            "source": source,
            "chunk_index": chunk_index,
            "chunk_count": 1,
            "layer": layer,
            "title": source,
        }


async def _assert_vector_translation_fallbacks() -> None:
    import corpus.store as store
    import runtime_tools.registry as registry

    original_similarity = store.similarity_search
    original_translate = registry._llm_translate_search_query
    calls: list[tuple[str, int, str | None, bool]] = []

    def fake_similarity(query: str, k: int = 5, layer: str | None = None, rerank: bool = False):
        calls.append((query, k, layer, rerank))
        return [_Doc(f"result for {query}", f"src-{len(calls)}", 0, layer or "all")]

    try:
        store.similarity_search = fake_similarity

        async def no_translation(*_args, **_kwargs):
            return None

        registry._llm_translate_search_query = no_translation
        docs = await registry._search_corpus_multilingual("국가와 혁명", 3, "core_theory")
        assert len(docs) == 1
        assert calls == [("국가와 혁명", 3, "core_theory", True)]

        calls.clear()
        docs = await registry._search_corpus_multilingual("imperialism finance capital", 3, "modern_analysis")
        assert len(docs) == 1
        assert calls == [("imperialism finance capital", 3, "modern_analysis", True)]
    finally:
        store.similarity_search = original_similarity
        registry._llm_translate_search_query = original_translate


async def _assert_vector_translation_parallel_merge() -> None:
    import corpus.store as store
    import runtime_tools.registry as registry

    original_similarity = store.similarity_search
    original_translate = registry._llm_translate_search_query
    original_rerank = registry._rerank_merged_docs
    calls: list[tuple[str, int, str | None, bool]] = []

    def fake_similarity(query: str, k: int = 5, layer: str | None = None, rerank: bool = False):
        calls.append((query, k, layer, rerank))
        return [
            _Doc(f"shared {query}", "shared-source", 0, layer or "all"),
            _Doc(f"unique {query}", f"src-{query}", 0, layer or "all"),
        ]

    async def translate(_query: str, target_language: str, _layer: str):
        return "state and revolution" if target_language == "English" else "제국주의 금융자본"

    try:
        store.similarity_search = fake_similarity
        registry._llm_translate_search_query = translate
        registry._rerank_merged_docs = lambda _query, docs, k: docs[:k]
        docs = await registry._search_corpus_multilingual("국가와 혁명", 4, "core_theory")
        assert calls == [
            ("국가와 혁명", 8, "core_theory", False),
            ("state and revolution", 8, "core_theory", False),
        ]
        assert len(docs) == 3, "shared source+chunk should be deduped"
    finally:
        store.similarity_search = original_similarity
        registry._llm_translate_search_query = original_translate
        registry._rerank_merged_docs = original_rerank


async def _assert_vector_metadata_filter_inference() -> None:
    import corpus.store as store
    import runtime_tools.registry as registry

    original_similarity = store.similarity_search
    original_translate = registry._llm_translate_search_query
    calls: list[dict] = []

    def fake_similarity(
        query: str,
        k: int = 5,
        layer: str | None = None,
        rerank: bool = False,
        **filters,
    ):
        calls.append({"query": query, "k": k, "layer": layer, "rerank": rerank, **filters})
        if filters.get("year"):
            return []
        return [_Doc("stalin national question", "stalin-source", 0, layer or "all")]

    try:
        store.similarity_search = fake_similarity

        async def no_translation(*_args, **_kwargs):
            return None

        registry._llm_translate_search_query = no_translation
        docs = await registry._search_corpus_multilingual("1913년 스탈린 민족 문제 문헌", 3, "core_theory")
        assert len(docs) == 1
        assert calls[0]["author"] == "Stalin"
        assert calls[0]["title"] == "National Question"
        assert calls[0]["year"] == 1913
        assert calls[1]["author"] == "Stalin"
        assert calls[1]["title"] == "National Question"
        assert "year" not in calls[1]
    finally:
        store.similarity_search = original_similarity
        registry._llm_translate_search_query = original_translate


async def _assert_route_task_fallbacks() -> None:
    import self_runtime.tools as tools

    original_classifier = tools._classify_route_with_llm
    try:
        async def invalid_json_fallback(_task: str, _candidates=None):
            return None

        tools._classify_route_with_llm = invalid_json_fallback
        public = json.loads(await tools._exec_route_task("이 공개 연구 글의 오타를 고쳐줘", include_store_guide=False))
        assert public["recommendation"]["recommended_agent"] == "analyst"
        assert public["classifier"]["fallback"] == "heuristic"

        code = json.loads(await tools._exec_route_task("서비스 traceback 고치고 테스트 돌려줘", include_store_guide=False))
        assert code["recommendation"]["recommended_agent"] == "programmer"

        async def programmer_misroute(_task: str, _candidates=None):
            return {
                "recommended_agent": "programmer",
                "confidence": "high",
                "reason": "bad classifier result",
                "content_type": "research_document",
                "needs_identifier": True,
                "required_capabilities": [],
                "forbidden_assumptions": [],
                "source": "llm_classifier",
            }

        tools._classify_route_with_llm = programmer_misroute
        guarded = json.loads(await tools._exec_route_task("공개 연구 글의 문구를 수정해줘", include_store_guide=False))
        assert guarded["recommendation"]["recommended_agent"] == "programmer"
        assert "Probable misroute" in guarded["recommendation"]["warning"]
    finally:
        tools._classify_route_with_llm = original_classifier


def _assert_operator_trust_tier() -> None:
    from provenance.runtime import init_provenance_buffer

    operator = init_provenance_buffer(agent="orchestrator")
    assert operator.infer_trust_tier() == "anchor"
    assert operator.trust_source_note() == "trusted_operator_chat_or_operator_task"

    web_like = init_provenance_buffer(agent="agent")
    assert web_like.infer_trust_tier() == "unverified"


async def main() -> int:
    await _assert_vector_translation_fallbacks()
    await _assert_vector_translation_parallel_merge()
    await _assert_vector_metadata_filter_inference()
    await _assert_route_task_fallbacks()
    _assert_operator_trust_tier()
    print("llm routing/search smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
