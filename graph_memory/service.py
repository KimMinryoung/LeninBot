"""
GraphMemoryService — Graphiti 기반 지식 그래프 서비스
====================================================

Cyber-Lenin의 정보 에이전트 기능을 위한 핵심 서비스 레이어.
Neo4j + Gemini를 사용하여 에피소드 수집, 검색, 브리핑 생성을 수행.

chatbot.py와 독립적으로 사용 가능. 통합은 별도 작업.
"""

import os
from datetime import datetime, timezone

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

from .entities import ENTITY_TYPES
from .edges import EDGE_TYPES
from .config import EDGE_TYPE_MAP, EPISODE_SOURCE_MAP


class GraphMemoryService:
    """Graphiti 기반 지식 그래프 서비스. 레닌봇의 정보 에이전트 기능."""

    def __init__(self):
        self._graphiti: Graphiti | None = None

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
                small_model="gemini-2.0-flash-lite",
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
                model="gemini-2.0-flash-lite",
            )
        )

        graph_driver = Neo4jDriver(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
        )

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

    async def ingest_episode(
        self,
        name: str,
        body: str,
        source_type: str,
        reference_time: datetime,
        group_id: str,
        source_description: str | None = None,
    ) -> None:
        """정보 에피소드 1건 수집. 엔티티/엣지 자동 추출.

        Args:
            name: 에피소드 식별 이름.
            body: 에피소드 본문 텍스트 또는 JSON.
            source_type: EPISODE_SOURCE_MAP 키 (e.g., 'osint_news').
            reference_time: 정보의 기준 시각 (timezone-aware).
            group_id: 논리적 그룹 ID (e.g., 'osint_semiconductor').
            source_description: 소스 설명 오버라이드. None이면 SOURCE_MAP에서 자동.
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

        await graphiti.add_episode(
            name=name,
            episode_body=body,
            source=episode_type,
            source_description=source_description,
            reference_time=reference_time,
            group_id=group_id,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )

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

    async def close(self) -> None:
        """Graphiti 연결 종료."""
        if self._graphiti is not None:
            await self._graphiti.close()
            self._graphiti = None
            print("✅ [GraphMemory] 지식 그래프 서비스 종료")
