"""
GraphMemoryService — Graphiti 기반 지식 그래프 서비스
====================================================

Cyber-Lenin의 정보 에이전트 기능을 위한 핵심 서비스 레이어.
Neo4j + Gemini를 사용하여 에피소드 수집, 검색, 브리핑 생성을 수행.

모든 인터페이스(web_chat, telegram_bot, agents)에서 shared.py를 통해 사용.
"""

import asyncio
import os
import json
import logging
import time
from datetime import datetime, timezone
import re
from typing import Any

# Graphiti 내부 semaphore_gather 동시성 — graphiti import 전에 설정해야 유효.
# gemini-2.5-flash/lite는 RPM 여유가 있으므로 기본 20 사용.
# rate limit 발생 시 호출부에서 SEMAPHORE_LIMIT=1 등으로 오버라이드 가능.
os.environ.setdefault("SEMAPHORE_LIMIT", "20")

from dotenv import load_dotenv

from .graphiti_patches import apply_graphiti_patches, normalize_entity_names_in_text
apply_graphiti_patches()

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
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
from .conformance import validate_episode_result, apply_hard_fixes

logger = logging.getLogger(__name__)


_EMBED_RETRYABLE_KEYWORDS = (
    "429",
    "resource_exhausted",
    "rate limit",
    "quota",
    "503",
    "unavailable",
)


def _is_retryable_embedding_error(exc: Exception) -> bool:
    err = str(exc).lower()
    return any(keyword in err for keyword in _EMBED_RETRYABLE_KEYWORDS)


def _embedding_retry_delays() -> list[float]:
    raw = os.getenv("KG_EMBED_RETRY_DELAYS", "5,15,45")
    delays: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            delay = float(part)
        except ValueError:
            continue
        if delay >= 0:
            delays.append(delay)
    return delays or [5.0, 15.0, 45.0]


class _EmbedRateLimiter:
    """Client-side request pacer for embedding calls (KG_EMBED_MAX_RPS).

    Batch producers (scout→KG ingest, kg_enricher, curation ingest) used to
    burst SEMAPHORE_LIMIT-wide into the Gemini quota and rely on the retry
    wrapper to paper over the resulting 429s. Spacing requests at the client
    keeps quota headroom for interactive searches instead.

    Simple serializing scheduler: each acquire reserves the next free slot
    1/rate seconds after the previous one. Rate is re-read from the env on
    every acquire (cheap, and lets ops tune without restart); <= 0 disables.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    @staticmethod
    def _rate() -> float:
        try:
            return float(os.getenv("KG_EMBED_MAX_RPS", "2"))
        except (TypeError, ValueError):
            return 2.0

    async def acquire(self) -> float:
        """Wait for the next request slot; returns the seconds slept."""
        rate = self._rate()
        if rate <= 0:
            return 0.0
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next_at)
            self._next_at = start + (1.0 / rate)
            wait = start - now
        if wait > 0:
            await asyncio.sleep(wait)
        return wait


_EMBED_LIMITER = _EmbedRateLimiter()


def _resolve_kg_gemini_key() -> str:
    """Gemini key for the KG service (LLM extraction + embeddings).

    KG_GEMINI_API_KEY isolates KG traffic onto its own quota; it falls back
    to the shared GEMINI_API_KEY so the separation can be provisioned later
    (a key from the SAME Google project shares quota and gains nothing —
    it must come from a separate project/account)."""
    from secrets_loader import get_secret

    return (get_secret("KG_GEMINI_API_KEY", "") or "").strip() or (get_secret("GEMINI_API_KEY", "") or "")


class RetryingGeminiEmbedder(GeminiEmbedder):
    """Gemini embedder with client-side pacing plus bounded retry for
    transient Vertex/Gemini limits."""

    async def _with_retry(self, operation: str, call):
        delays = _embedding_retry_delays()
        max_attempts = len(delays) + 1
        for attempt in range(1, max_attempts + 1):
            try:
                waited = await _EMBED_LIMITER.acquire()
                if waited > 1.0:
                    logger.debug(
                        "[KG] embedding %s rate-limited client-side; waited %.1fs",
                        operation, waited,
                    )
                return await call()
            except Exception as exc:
                if attempt >= max_attempts or not _is_retryable_embedding_error(exc):
                    raise
                delay = delays[attempt - 1]
                logger.warning(
                    "[KG] Gemini embedding %s failed with retryable error "
                    "(attempt %s/%s); sleeping %.1fs: %s",
                    operation,
                    attempt,
                    max_attempts,
                    delay,
                    str(exc)[:300],
                )
                await asyncio.sleep(delay)

    async def create(self, input_data):
        return await self._with_retry(
            "create",
            lambda: super(RetryingGeminiEmbedder, self).create(input_data),
        )

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return await self._with_retry(
            "create_batch",
            lambda: super(RetryingGeminiEmbedder, self).create_batch(input_data_list),
        )


class GraphMemoryService:
    """Graphiti 기반 지식 그래프 서비스. 레닌봇의 정보 에이전트 기능."""

    def __init__(self):
        self._graphiti: Graphiti | None = None
        self._llm_client: GeminiClient | None = None
        self._init_lock = asyncio.Lock()

    @property
    def graphiti(self) -> Graphiti:
        """Public access to the Graphiti instance (e.g. for driver access)."""
        return self._ensure_initialized()

    async def initialize(self) -> None:
        """Neo4j 연결 + Gemini LLM 초기화 + 인덱스/제약조건 설정."""
        if self._graphiti is not None:
            return

        async with self._init_lock:
            if self._graphiti is not None:
                return

            from secrets_loader import get_secret

            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = get_secret("NEO4J_PASSWORD", "") or ""
            neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
            gemini_api_key = _resolve_kg_gemini_key()

            llm_client = GeminiClient(
                config=LLMConfig(
                    api_key=gemini_api_key,
                    model="gemini-3.1-flash-lite",
                    small_model="gemini-2.5-flash-lite",
                )
            )

            embedder = RetryingGeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=gemini_api_key,
                    embedding_model="gemini-embedding-001",
                )
            )

            graph_driver = Neo4jDriver(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                database=neo4j_database,
            )
            # graphiti의 Neo4jDriver가 keepalive/lifetime 설정 없이 드라이버를 생성하므로,
            # 내부 client를 재생성하여 유휴 연결 끊김 방지
            from neo4j import AsyncGraphDatabase
            graph_driver.client = AsyncGraphDatabase.driver(
                uri=neo4j_uri,
                auth=(neo4j_user or '', neo4j_password or ''),
                keep_alive=True,
                max_connection_lifetime=300,       # 5분마다 연결 갱신
                liveness_check_timeout=30,         # 유휴 연결 사용 전 30초 내 liveness 확인
                connection_acquisition_timeout=30,
                max_connection_pool_size=50,
            )

            graphiti = Graphiti(
                uri=None,
                user=None,
                password=None,
                llm_client=llm_client,
                embedder=embedder,
                graph_driver=graph_driver,
            )

            # _graphiti는 인덱스 빌드까지 성공한 뒤에만 할당 — 실패하면
            # 반쯤 초기화된 인스턴스가 "완료"로 남지 않도록 드라이버를 닫는다.
            try:
                await graphiti.build_indices_and_constraints()
            except BaseException:
                await graph_driver.client.close()
                raise

            self._llm_client = llm_client
            self._graphiti = graphiti
            logger.info("✅ [GraphMemory] 지식 그래프 서비스 초기화 완료")

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

        logger.info("[preprocess] LLM 전처리 시작 (본문 %d자)", len(raw_article))
        prompt = NEWS_PREPROCESS_PROMPT_TEMPLATE.format(article=raw_article)
        response = await self._llm_client.generate_response([Message(role="user", content=prompt)])
        processed = self._extract_text_from_llm_response(response)
        logger.info("[preprocess] 완료 → %d자", len(processed))
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

        # 약어 → 정식명 치환 (Graphiti entity resolution 보강)
        ingest_body = normalize_entity_names_in_text(ingest_body)

        # 전처리 후 본문 길이 제한 (Graphiti output token 초과 방지)
        if max_body_chars and len(ingest_body) > max_body_chars:
            logger.info("[truncate] %d자 → %d자로 잘라냄", len(ingest_body), max_body_chars)
            ingest_body = ingest_body[:max_body_chars]

        logger.info("[graphiti] add_episode 시작 (name=%s, body %d자)", sanitized_name, len(ingest_body))
        result = await graphiti.add_episode(
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
        logger.info("[graphiti] add_episode 완료")

        # ── Conformance gate ──
        # Validate the just-created entities/edges against the schema. Hard
        # violations (self-loops, non-Entity endpoints) are auto-deleted; soft
        # violations (non-standard edge names, EDGE_TYPE_MAP mismatches,
        # untyped nodes) are logged for the daily scanner to pick up.
        try:
            report = validate_episode_result(result, log_audit=True)
            if report.hard_violation_count() > 0:
                await apply_hard_fixes(
                    graphiti.driver.client,
                    os.getenv("NEO4J_DATABASE", "neo4j"),
                    report,
                )
        except Exception as exc:
            # Conformance is best-effort — never break ingestion on validator errors
            logger.warning("[conformance] check failed (non-fatal): %s", exc)

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

    async def close(self) -> None:
        """Graphiti 연결 종료."""
        if self._graphiti is not None:
            await self._graphiti.close()
            self._graphiti = None
            self._llm_client = None
            logger.info("[GraphMemory] 지식 그래프 서비스 종료")
