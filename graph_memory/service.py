"""
GraphMemoryService — Graphiti 기반 지식 그래프 서비스
====================================================

Cyber-Lenin의 정보 에이전트 기능을 위한 핵심 서비스 레이어.
Neo4j + Gemini를 사용하여 에피소드 수집, 검색, 브리핑 생성을 수행.

chatbot.py와 독립적으로 사용 가능. 통합은 별도 작업.
"""

import os
import json
from datetime import datetime, timezone
import re
from typing import Any

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config import (
    SearchConfig,
    EdgeSearchConfig,
    NodeSearchConfig,
    EdgeSearchMethod,
    NodeSearchMethod,
    EdgeReranker,
    NodeReranker,
)
from graphiti_core.search.search_filters import SearchFilters

from graphiti_core.prompts.models import Message

from .entities import ENTITY_TYPES
from .edges import EDGE_TYPES
from .config import (
    EDGE_TYPE_MAP,
    EPISODE_SOURCE_MAP,
    EXCLUDED_ENTITY_TYPES,
    NEWS_PREPROCESS_PROMPT_TEMPLATE,
    CUSTOM_EXTRACTION_INSTRUCTIONS,
)


class GraphMemoryService:
    """Graphiti 기반 지식 그래프 서비스. 레닌봇의 정보 에이전트 기능."""

    def __init__(self):
        self._graphiti: Graphiti | None = None
        self._llm_client: GeminiClient | None = None

    async def initialize(self) -> None:
        """Neo4j 연결 + Gemini LLM 초기화 + 인덱스/제약조건 설정."""
        if self._graphiti is not None:
            return

        load_dotenv()

        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")

        llm_client = GeminiClient(
            config=LLMConfig(
                api_key=gemini_api_key,
                model="gemini-2.5-flash",
                small_model="gemini-2.5-flash-lite",
            )
        )

        embedder = GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=gemini_api_key,
                embedding_model="gemini-embedding-001",
            )
        )

        cross_encoder = GeminiRerankerClient(
            config=LLMConfig(
                api_key=gemini_api_key,
                model="gemini-2.5-flash-lite",
            )
        )

        graph_driver = Neo4jDriver(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
        )

        self._llm_client = llm_client

        self._graphiti = Graphiti(
            uri=None,
            user=None,
            password=None,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            graph_driver=graph_driver,
        )

        await self._graphiti.build_indices_and_constraints()
        print("✅ [GraphMemory] 지식 그래프 서비스 초기화 완료")

    def _ensure_initialized(self) -> Graphiti:
        """초기화 확인 후 Graphiti 인스턴스 반환."""
        if self._graphiti is None:
            raise RuntimeError(
                "GraphMemoryService가 초기화되지 않았습니다. "
                "await service.initialize()를 먼저 호출하세요."
            )
        return self._graphiti

    def _sanitize_episode_name(self, name: str, source_type: str) -> str:
        """엔티티 오염을 줄이기 위해 에피소드 이름을 중립 ID 형태로 정규화."""
        clean = name.strip()
        # day1464, 2026-02-27 같은 메타 패턴은 제거
        clean = re.sub(r"(?i)day\d+", "", clean)
        clean = re.sub(r"\d{4}-\d{2}-\d{2}", "", clean)
        clean = re.sub(r"[^a-zA-Z0-9_-]+", "-", clean).strip("-_")

        if not clean:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            return f"{source_type}-{ts}"

        return clean[:80]

    def _extract_text_from_llm_response(self, response: Any) -> str:
        """LLM 응답 타입별 텍스트 추출(문자열/객체 모두 대응)."""
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()

        for attr in ("text", "content", "output_text", "response"):
            value = getattr(response, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # dict 유사 객체 fallback
        if isinstance(response, dict):
            for key in ("text", "content", "output_text", "response"):
                value = response.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return str(response).strip()

    def _parse_json_from_llm_response(self, response_text: str) -> dict | list | None:
        """LLM 텍스트 응답에서 JSON 블록을 안전하게 파싱."""
        raw = response_text.strip()
        if not raw:
            return None

        # ```json ... ``` 코드펜스 제거
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None

        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None

        return None


    async def preprocess_news_article(self, raw_article: str) -> str:
        """뉴스 원문을 그래프 수집 친화적 팩트 목록으로 정제."""
        self._ensure_initialized()
        if self._llm_client is None:
            return raw_article

        print(f"    [preprocess] LLM 전처리 시작 (본문 {len(raw_article)}자)...", flush=True)
        prompt = NEWS_PREPROCESS_PROMPT_TEMPLATE.format(article=raw_article)
        response = await self._llm_client.generate_response([Message(role="user", content=prompt)])
        processed = self._extract_text_from_llm_response(response)
        print(f"    [preprocess] 완료 → {len(processed)}자", flush=True)
        return processed or raw_article

    async def ingest_episode(
        self,
        name: str,
        body: str,
        source_type: str,
        reference_time: datetime,
        group_id: str,
        source_description: str | None = None,
        preprocess_news: bool = True,
        max_body_chars: int | None = None,
    ) -> None:
        """정보 에피소드 1건 수집. 엔티티/엣지 자동 추출.

        Args:
            name: 에피소드 식별 이름.
            body: 에피소드 본문 텍스트 또는 JSON.
            source_type: EPISODE_SOURCE_MAP 키 (e.g., 'osint_news').
            reference_time: 정보의 기준 시각 (timezone-aware).
            group_id: 논리적 그룹 ID (e.g., 'osint_semiconductor').
            source_description: 소스 설명 오버라이드. None이면 SOURCE_MAP에서 자동.
            preprocess_news: osint_news인 경우 본문을 LLM으로 사전 정제할지 여부.
            max_body_chars: 전처리 후 본문 최대 길이. 초과 시 truncate.
        """
        graphiti = self._ensure_initialized()

        # source_type → EpisodeType + source_description 매핑
        mapping = EPISODE_SOURCE_MAP.get(source_type)
        if mapping:
            episode_type_str, default_desc = mapping
            episode_type = EpisodeType(episode_type_str)
            if source_description is None:
                source_description = default_desc
        else:
            episode_type = EpisodeType.text
            if source_description is None:
                source_description = f"Unknown source type: {source_type}"

        sanitized_name = self._sanitize_episode_name(name, source_type)
        ingest_body = body

        if preprocess_news and source_type == "osint_news":
            ingest_body = await self.preprocess_news_article(body)

        # 전처리 후 본문 길이 제한 (Graphiti output token 초과 방지)
        if max_body_chars and len(ingest_body) > max_body_chars:
            print(f"    [truncate] {len(ingest_body)}자 → {max_body_chars}자로 잘라냄", flush=True)
            ingest_body = ingest_body[:max_body_chars]

        print(f"    [graphiti] add_episode 시작 (name={sanitized_name}, body {len(ingest_body)}자)...", flush=True)
        await graphiti.add_episode(
            name=sanitized_name,
            episode_body=ingest_body,
            source=episode_type,
            source_description=source_description,
            reference_time=reference_time,
            group_id=group_id,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
            excluded_entity_types=EXCLUDED_ENTITY_TYPES,
            custom_extraction_instructions=CUSTOM_EXTRACTION_INSTRUCTIONS,
        )
        print(f"    [graphiti] add_episode 완료", flush=True)

    async def ingest_episodes_bulk(self, episodes: list[dict]) -> None:
        """대량 에피소드 일괄 수집.

        Args:
            episodes: 각 항목은 ingest_episode()의 인자를 담은 dict.
                필수 키: name, body, source_type, reference_time, group_id
                선택 키: source_description
        """
        for ep in episodes:
            await self.ingest_episode(
                name=ep["name"],
                body=ep["body"],
                source_type=ep["source_type"],
                reference_time=ep["reference_time"],
                group_id=ep["group_id"],
                source_description=ep.get("source_description"),
                preprocess_news=ep.get("preprocess_news", True),
            )

    async def search(
        self,
        query: str,
        group_ids: list[str] | None = None,
        edge_types: list[str] | None = None,
        node_labels: list[str] | None = None,
        center_node_uuid: str | None = None,
        num_results: int = 10,
    ) -> dict:
        """지식 그래프 하이브리드 검색.

        Returns:
            {"nodes": [...], "edges": [...]} 형태의 결과.
            각 node: {"name", "labels", "summary", "uuid"}
            각 edge: {"fact", "valid_at", "invalid_at", "uuid"}
        """
        graphiti = self._ensure_initialized()

        # SearchFilters 구성
        search_filter = SearchFilters()
        if edge_types:
            search_filter.edge_types = edge_types
        if node_labels:
            search_filter.node_labels = node_labels

        config = SearchConfig(
            edge_config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                reranker=EdgeReranker.rrf,
            ),
            node_config=NodeSearchConfig(
                search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
                reranker=NodeReranker.rrf,
            ),
            limit=num_results,
        )

        results = await graphiti.search_(
            query=query,
            config=config,
            group_ids=group_ids,
            search_filter=search_filter,
            center_node_uuid=center_node_uuid,
        )

        # 결과를 직렬화 가능한 dict로 변환
        nodes = [
            {
                "name": node.name,
                "labels": node.labels,
                "summary": node.summary,
                "uuid": node.uuid,
            }
            for node in results.nodes
        ]
        edges = [
            {
                "fact": edge.fact,
                "valid_at": str(edge.valid_at) if edge.valid_at else None,
                "invalid_at": str(edge.invalid_at) if edge.invalid_at else None,
                "uuid": edge.uuid,
            }
            for edge in results.edges
        ]

        return {"nodes": nodes, "edges": edges}

    async def generate_briefing(
        self,
        topic: str,
        group_ids: list[str],
    ) -> dict:
        """주제 기반 전략 브리핑 데이터 수집.

        Returns:
            {
                "topic": str,
                "key_entities": [...],
                "relationships": [...],
                "incidents": [...],
                "timeline": [...]
            }
        """
        graphiti = self._ensure_initialized()

        briefing_data = {
            "topic": topic,
            "key_entities": [],
            "relationships": [],
            "incidents": [],
            "timeline": [],
        }

        # 1. 주제 관련 핵심 엔티티 + 관계 검색
        main_results = await graphiti.search_(
            query=topic,
            config=SearchConfig(limit=30),
            group_ids=group_ids,
        )

        for node in main_results.nodes:
            briefing_data["key_entities"].append({
                "name": node.name,
                "type": node.labels,
                "summary": node.summary,
            })

        for edge in main_results.edges:
            entry = {
                "fact": edge.fact,
                "valid_from": str(edge.valid_at) if edge.valid_at else None,
                "valid_until": str(edge.invalid_at) if edge.invalid_at else None,
            }
            briefing_data["relationships"].append(entry)

            if edge.valid_at:
                briefing_data["timeline"].append({
                    "date": str(edge.valid_at),
                    "event": edge.fact,
                })

        # 2. 관련 특이사건 검색
        incident_results = await graphiti.search_(
            query=topic,
            config=SearchConfig(limit=10),
            search_filter=SearchFilters(
                node_labels=["Incident"],
                edge_types=["Involvement", "ThreatAction"],
            ),
            group_ids=group_ids,
        )

        for node in incident_results.nodes:
            briefing_data["incidents"].append({
                "name": node.name,
                "summary": node.summary,
            })

        # 3. 타임라인 정렬
        briefing_data["timeline"].sort(key=lambda x: x["date"])

        return briefing_data

    async def query_chatbot(
        self,
        query: str,
        group_ids: list[str] | None = None,
        num_results: int = 12,
    ) -> dict:
        """지식그래프 기반 테스트용 챗봇 질의.

        1) 그래프에서 관련 노드/엣지를 검색하고
        2) 검색 결과만 근거로 LLM이 간단 답변을 생성한다.

        Returns:
            {
                "query": str,
                "answer": str,
                "context": {"nodes": [...], "edges": [...]},
            }
        """
        context = await self.search(
            query=query,
            group_ids=group_ids,
            num_results=num_results,
        )

        if self._llm_client is None:
            return {
                "query": query,
                "answer": "LLM client가 초기화되지 않아 검색 결과만 반환합니다.",
                "context": context,
            }

        node_lines = [
            f"- {n['name']} ({', '.join(n['labels'])}) | {n.get('summary') or 'no summary'}"
            for n in context["nodes"]
        ]
        edge_lines = [
            f"- {e['fact']} | valid_at={e.get('valid_at')}"
            for e in context["edges"]
        ]

        prompt = (
            "너는 지식그래프 기반 분석 보조 챗봇이다.\n"
            "아래 컨텍스트에 없는 내용은 추정하지 말고 '근거 없음'이라고 답하라.\n"
            "답변은 한국어로 간결하게 작성하라.\n\n"
            f"[질문]\n{query}\n\n"
            "[그래프 노드]\n"
            f"{chr(10).join(node_lines) if node_lines else '- (없음)'}\n\n"
            "[그래프 엣지]\n"
            f"{chr(10).join(edge_lines) if edge_lines else '- (없음)'}\n"
        )

        response = await self._llm_client.generate_response([Message(role="user", content=prompt)])
        answer = self._extract_text_from_llm_response(response)

        if not answer:
            answer = "관련 컨텍스트를 찾지 못했거나 답변 생성에 실패했습니다."

        return {
            "query": query,
            "answer": answer,
            "context": context,
        }

    async def query_active_wars(
        self,
        group_ids: list[str] | None = None,
        num_results: int = 30,
    ) -> dict:
        """지식그래프에서 '현재 진행 중인 전쟁'과 개시일을 구조화 추출한다."""
        context = await self.search(
            query=(
                "ongoing war current conflict invasion military campaign "
                "active hostilities start date belligerents"
            ),
            group_ids=group_ids,
            edge_types=["ThreatAction", "Participation", "Involvement", "Presence"],
            node_labels=["Campaign", "Incident", "Organization", "Location"],
            num_results=num_results,
        )

        if self._llm_client is None:
            return {
                "items": [],
                "note": "LLM client가 초기화되지 않아 구조화 추출을 수행하지 못했습니다.",
                "context": context,
            }

        prompt = (
            "아래 지식그래프 컨텍스트만 사용해서 현재 진행 중인 전쟁 목록을 JSON으로 추출하라.\n"
            "컨텍스트 밖 지식은 사용 금지. 근거가 부족하면 unknown으로 채워라.\n"
            "반드시 JSON 배열만 출력하라. 스키마:\n"
            "[{'war_name': str, 'countries': [str], 'start_date': 'YYYY-MM-DD|YYYY-MM|YYYY|unknown', 'status': 'ongoing|unknown', 'evidence': [str]}]\n\n"
            f"[nodes] {json.dumps(context['nodes'], ensure_ascii=False)}\n"
            f"[edges] {json.dumps(context['edges'], ensure_ascii=False)}\n"
        )

        response = await self._llm_client.generate_response([Message(role="user", content=prompt)])
        text = self._extract_text_from_llm_response(response)
        parsed = self._parse_json_from_llm_response(text)

        items: list[dict] = []
        if isinstance(parsed, list):
            items = [item for item in parsed if isinstance(item, dict)]

        return {
            "items": items,
            "raw_response": text,
            "context": context,
        }

    async def close(self) -> None:
        """Graphiti 연결 종료."""
        if self._graphiti is not None:
            await self._graphiti.close()
            self._graphiti = None
            self._llm_client = None
            print("✅ [GraphMemory] 지식 그래프 서비스 종료")
