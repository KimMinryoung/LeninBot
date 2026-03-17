import os
import logging
import hashlib
from typing import Annotated, List, TypedDict, Optional
from operator import add
from dotenv import load_dotenv

# Database & Embeddings
from db import query as db_query, execute as db_execute
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document # [New] To handle documents
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

import asyncio
import re
import time
from datetime import datetime
from typing import Literal

from shared import (
    extract_text_content, CORE_IDENTITY, KST, MODEL_MAIN, MODEL_LIGHT,
    get_kg_service, run_kg_async,
    extract_urls, fetch_urls_as_documents,
    search_experiential_memory, set_shared_embeddings,
)
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cyber_lenin")


def _require_env(var_name: str) -> str:
    """Load required environment variable or fail fast with a clear message."""
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"필수 환경변수 누락: {var_name}")
    return value


def _invoke_with_backoff(invoke_fn, max_attempts: int = 3, base_delay: float = 1.0):
    """Invoke callable with retry/backoff for transient quota/rate-limit errors."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return invoke_fn()
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            is_retryable = any(k in msg for k in ["429", "503", "quota", "resource_exhausted", "rate limit", "timeout", "unavailable"])
            if not is_retryable or attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("LLM 호출 재시도(%s/%s): %s", attempt + 1, max_attempts, e)
            time.sleep(delay)
    raise last_error




def _structured_call(prompt, llm_instance, inputs: dict, model_class: type[BaseModel], default):
    """Invoke LLM with schema-constrained decoding. Falls back to default on any failure."""
    try:
        structured_llm = llm_instance.with_structured_output(model_class)
        chain = prompt | structured_llm
        return _invoke_with_backoff(lambda: chain.invoke(inputs))
    except Exception as e:
        logger.warning("[구조화] %s 호출 실패 (기본값 사용): %s", model_class.__name__, e)
        return default

logger.info("⚙️ [시스템] 사이버-레닌의 지능망 기동 중...")
# 1. 환경 설정 및 초기화
load_dotenv()

GEMINI_API_KEY = _require_env("GEMINI_API_KEY")

# 임베딩 모델
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
set_shared_embeddings(embeddings)  # Share with shared.py to avoid duplicate loading

# LLM 설정 (Gemini 3.1 Flash Lite)
llm = ChatGoogleGenerativeAI(
    model=MODEL_MAIN,
    google_api_key=GEMINI_API_KEY,
    temperature=0.45,
    max_output_tokens=4096,
    streaming=True,
    top_p=0.88,
    max_retries=5,
)
# 경량 LLM (라우팅, 그레이딩 등 유틸리티 작업용)
llm_light = ChatGoogleGenerativeAI(
    model=MODEL_LIGHT,
    google_api_key=GEMINI_API_KEY,
    temperature=0.0,
    max_output_tokens=512,
    streaming=False,
    max_retries=2,
)
logger.info("✅ [성공] 모든 시스템 기동 완료.")




# 2. 상태(State) 정의
# 대화 기록(messages)과 검색된 문서를 저장하는 메모리 구조입니다.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]
    intent: Optional[Literal["academic", "strategic", "casual"]]
    datasource: Optional[Literal["vectorstore", "generate", "plan"]]
    logs: Annotated[List[str], add]
    # Phase 2: Query decomposition — search_queries are ready-to-use (context-resolved, translated)
    search_queries: Optional[List[dict]]  # [{"ko": "...", "en": "..."|None}, ...]
    layer: Optional[Literal["core_theory", "modern_analysis", "all"]]  # knowledge layer
    needs_realtime: Optional[Literal["yes", "no"]]  # from batch grading
    # Phase 3: Plan-and-execute
    plan: Optional[List[dict]]        # structured research plan from planner
    current_step: int                 # progress pointer into plan
    step_results: List[str]                   # accumulated intermediate results (manual accumulation for checkpoint reset)
    logs_turn_start: int              # index into logs[] where the current turn begins (for per-turn log slicing)
    # Knowledge Graph integration
    kg_context: Optional[str]         # formatted KG results (nodes+edges). Always included, no grading.
    # Internal: Track KG searched queries to avoid duplicates (Plan-and-execute path)
    _kg_searched_queries: List[str]   # internal state, not exposed to nodes
    # Self-knowledge: which self-tool to invoke (None = not needed)
    self_knowledge_tool: Optional[str]
    # URL content: documents fetched from URLs in user's question
    url_documents: List[Document]

# Merge 1: Combined query analysis (router + layer router + decompose + query rewrite in one LLM call)
class SearchQuery(BaseModel):
    """A ready-to-search query with optional English translation."""
    ko: str = Field(..., description="Self-contained Korean search query (pronouns resolved)")
    en: Optional[str] = Field(default=None, description="English translation (when layer needs English texts)")

class QueryAnalysis(BaseModel):
    """Combined intent classification, layer routing, query decomposition, and rewriting."""
    datasource: Literal["vectorstore", "generate"] = Field(..., description="Whether knowledge retrieval is needed")
    intent: Literal["academic", "strategic", "casual"] = Field(..., description="Response style")
    layer: Literal["core_theory", "modern_analysis", "all"] = Field(default="all", description="Which knowledge layer to search")
    search_queries: List[SearchQuery] = Field(default_factory=list, description="Ready-to-search queries with ko/en versions")
    needs_plan: bool = Field(default=False, description="Whether the query requires multi-step research planning")
    self_knowledge_tool: Optional[str] = Field(default=None, description="Which self-knowledge tool to use, or null if not needed")

system_query_analysis = """Analyze the user's question and conversation context. Output ALL fields in ONE JSON response.

Resolve pronouns/references using conversation context. Output SELF-CONTAINED search queries.

**datasource**: "vectorstore" (needs knowledge retrieval) | "generate" (greetings, chat, no data needed)
**intent**: "academic" (scholarly) | "strategic" (actionable advice) | "casual" (chat/greetings)
**layer**: "core_theory" (Marx/Lenin/Engels, pre-1950) | "modern_analysis" (contemporary) | "all" (both/unsure)
**search_queries**: Ready-to-search queries. Each object has "ko" (self-contained Korean query) and "en" (English translation, or null). Include "en" when layer is "core_theory" or "all". Only decompose when 2+ DISTINCT topics need separate searches.
**needs_plan**: true only for multi-step research/synthesis tasks. Most questions: false.
**self_knowledge_tool**: "read_diary" | "read_chat_logs" | "read_system_status" | "read_recent_updates" | null

Example for Korean question on classical theory (layer=all):
{{"datasource":"vectorstore", "intent":"academic", "layer":"all", "search_queries":[{{"ko":"레닌의 제국주의론", "en":"Lenin theory of imperialism"}}], "needs_plan":false, "self_knowledge_tool":null}}

Example for casual greeting:
{{"datasource":"generate", "intent":"casual", "layer":"all", "search_queries":[{{"ko":"안녕", "en":null}}], "needs_plan":false, "self_knowledge_tool":null}}"""

query_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", system_query_analysis),
    ("human", "Conversation context:\n{context}\n\nCurrent question: {question}"),
])

_ANALYSIS_DEFAULT = QueryAnalysis(datasource="vectorstore", intent="academic", layer="all", search_queries=[], needs_plan=False, self_knowledge_tool=None)

def invoke_query_analysis(inputs: dict) -> QueryAnalysis:
    return _structured_call(query_analysis_prompt, llm_light, inputs, QueryAnalysis, _ANALYSIS_DEFAULT)


# Merge 3+4: Batch document grading + realtime check in one LLM call
class BatchGradeResult(BaseModel):
    """Batch relevance grading for all documents + realtime info check."""
    scores: List[Literal["yes", "no"]] = Field(..., description="Relevance score for each document in order")
    needs_realtime: Literal["yes", "no"] = Field(default="no", description="Whether the question would benefit from real-time web search")

system_batch_grader = """Grade each document's relevance to the user question. Also judge if real-time web search would help.

For EACH document: "yes" if it contains relevant information, "no" only if clearly unrelated. When in doubt, "yes".
needs_realtime: "yes" if question involves current events or recent data (2020+), "no" for pure history/classical theory.

Respond with ONLY a JSON object like this example:
{{"scores": ["yes", "yes", "no", "yes"], "needs_realtime": "no"}}"""

batch_grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_batch_grader),
    ("human", "User question: {question}\n\nDocuments:\n{documents}"),
])

_BATCH_GRADE_DEFAULT = None  # handled in node

def invoke_batch_grader(inputs: dict) -> BatchGradeResult | None:
    return _structured_call(batch_grade_prompt, llm_light, inputs, BatchGradeResult, _BATCH_GRADE_DEFAULT)



# --- 헬퍼 함수 ---

def _label_doc(d: Document) -> str:
    """Short one-line label for a document, used in logs."""
    meta = d.metadata or {}
    author = meta.get("author", "")
    year = meta.get("year", "")
    source = meta.get("source", "")
    title = meta.get("title", "")
    if author or year:
        suffix = ", ".join(p for p in [author, str(year) if year else ""] if p)
        return f"{source or '출처미상'} ({suffix})"
    elif title:
        return f"{title} — {source}" if source else title
    return source or "출처미상"


def _format_doc(d: Document) -> str:
    """Format a document with metadata header for LLM consumption.
    Vectorstore docs use [author, year]; web docs fall back to [title | url]."""
    meta = d.metadata or {}
    author = meta.get("author")
    year = meta.get("year")
    header_parts = [p for p in [author, str(year) if year else None] if p]
    if header_parts:
        header = f"[{', '.join(header_parts)}] "
    else:
        title = meta.get("title", "")
        source = meta.get("source", "")
        web_label = " | ".join(p for p in [title, source] if p)
        header = f"[{web_label}] " if web_label else ""
    return f"{header}{d.page_content}"

# --- 노드 및 엣지 함수 정의 ---

def _build_context(messages: list, max_turns: int = 4) -> str:
    """Build a brief conversation context string from recent messages (excluding the last one)."""
    history = messages[:-1] if len(messages) > 1 else []
    # Take the last `max_turns` messages for context
    recent = history[-max_turns:]
    if not recent:
        return "(no prior context)"
    lines = []
    for msg in recent:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        # Truncate long messages to keep router prompt concise
        content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

# Node: Combined Query Analysis (Merge 1: router + layer + decompose in one call)
def analyze_intent_node(state: AgentState):
    question = state["messages"][-1].content
    context = _build_context(state["messages"])
    analysis = invoke_query_analysis({"question": question, "context": context})

    turn_start = len(state.get("logs", []))
    logs = [f"\n🚦 [분석] 의도: {analysis.intent} / 경로: {analysis.datasource} / 레이어: {analysis.layer}"]

    # Build search_queries as list of dicts from SearchQuery models
    search_queries = None
    if analysis.search_queries:
        search_queries = [sq.model_dump() for sq in analysis.search_queries]
        if len(search_queries) > 1:
            logs.append(f"🔀 [분해] 복합 질문을 {len(search_queries)}개의 하위 질문으로 분해:")
            for i, sq in enumerate(search_queries, 1):
                logs.append(f"   {i}. {sq['ko']}" + (f" → EN: {sq['en']}" if sq.get('en') else ""))
        elif search_queries:
            sq = search_queries[0]
            if sq['ko'] != question:
                logs.append(f"🔄 [재작성] 맥락 반영: \"{sq['ko']}\"")
            if sq.get('en'):
                logs.append(f"🔄 [번역] 영어 검색: \"{sq['en']}\"")

    # Phase 3: Plan-and-execute routing
    needs_plan = analysis.needs_plan and analysis.datasource == "vectorstore"
    if needs_plan:
        logs.append("📋 [계획] 복합 전략 질문 감지 — 연구 계획 수립 경로로 진입합니다.")

    self_tool = analysis.self_knowledge_tool
    if self_tool:
        logs.append(f"🪞 [분석] 자기 인식 질문 감지 — {self_tool} 조회 예정")

    # URL detection: fetch content from URLs in the question
    urls = extract_urls(question)
    url_docs = []
    if urls:
        logs.append(f"🔗 [분석] 질문에서 URL {len(urls)}개 감지")
        url_docs = fetch_urls_as_documents(urls, logs)
        # Force vectorstore path only if we actually got content (not just failure docs)
        has_real_content = any(not d.metadata.get("fetch_failed") for d in url_docs)
        if has_real_content and analysis.datasource == "generate":
            analysis_datasource = "vectorstore"
            logs.append("🔗 [분석] URL 내용 분석을 위해 vectorstore 경로로 전환")
        else:
            analysis_datasource = analysis.datasource
    else:
        analysis_datasource = analysis.datasource

    return {
        "intent": analysis.intent,
        "datasource": "plan" if needs_plan else analysis_datasource,
        "layer": analysis.layer,
        "search_queries": search_queries,
        "logs": logs,
        "logs_turn_start": turn_start,
        # Reset transient per-turn fields to prevent state leakage across checkpointed turns
        "documents": [],
        "plan": None,
        "current_step": 0,
        "needs_realtime": None,
        "step_results": [],
        "kg_context": None,
        "_kg_searched_queries": [],  # Reset KG query tracking for new turn
        "self_knowledge_tool": self_tool,
        "url_documents": url_docs,
    }


# Edge: Routing Logic
def router_logic(state: AgentState):
    return state.get("datasource", "vectorstore")

# Helper: Supabase RPC를 직접 호출하여 similarity search 수행
# (langchain-community의 SupabaseVectorStore.similarity_search가
#  postgrest v2.x의 SyncRPCFilterRequestBuilder와 호환되지 않는 문제 우회)
def _direct_similarity_search(query: str, k: int = 5, layer: str = None) -> list:
    query_embedding = embeddings.embed_query(query)
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    try:
        rows = db_query(
            "SELECT * FROM match_documents(%s::vector, 0.5, %s, %s)",
            (embedding_str, k, layer),
        )
    except Exception as e:
        error_msg = str(e)
        if "57014" in error_msg or "timeout" in error_msg.lower():
            print("⚠️ [검색] statement timeout 발생. 검색을 건너뜁니다.")
        else:
            print(f"⚠️ [검색] RPC 호출 실패: {e}")
        return []

    return [
        Document(
            page_content=row.get("content", ""),
            metadata=row.get("metadata", {}),
        )
        for row in rows
        if row.get("content")
    ]


# Helper: Run similarity search for a single query with layer logic
def _retrieve_for_query(search_query_ko: str, search_query_en: str | None, selected_layer: str) -> list:
    """Run vector search for one query across the appropriate layer(s)."""
    layer_filter = None if selected_layer == "all" else selected_layer
    if selected_layer == "all":
        en_q = search_query_en or search_query_ko
        docs_core = _direct_similarity_search(en_q, k=3, layer="core_theory")
        docs_modern = _direct_similarity_search(search_query_ko, k=3, layer="modern_analysis")
        return docs_core + docs_modern
    elif selected_layer == "core_theory":
        return _direct_similarity_search(search_query_en or search_query_ko, k=5, layer=layer_filter)
    else:
        return _direct_similarity_search(search_query_ko, k=5, layer=layer_filter)


# Helper: Deduplicate documents by page_content
def _deduplicate_docs(docs: list) -> list:
    """Remove duplicate documents based on page_content hash."""
    seen = set()
    unique = []
    for d in docs:
        h = hashlib.sha256(d.page_content.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(d)
    return unique


# Helper: Search Knowledge Graph
def _search_kg(query, num_results=10, query_en: Optional[str] = None) -> Optional[str]:
    """Query the knowledge graph and return formatted results, or None on failure.
    
    Args:
        query: Primary search query (Korean or English).
        num_results: Max number of nodes+edges to retrieve.
        query_en: Optional English query for better matching on English-centric KG.
                  If provided, searches with both queries and merges results.
    """
    _svc = [get_kg_service()]
    if not _svc[0]:
        return None

    def _do_search(q):
        _CONN_ERRORS = ("connection reset", "defunct", "connectionreseterror")
        _RESET_KEYWORDS = ("dns", "connection", "timeout", "unavailable", "graphiti")

        for attempt in range(2):
            try:
                return run_kg_async(_svc[0].search(query=q, group_ids=None, num_results=num_results))
            except Exception as e:
                err_msg = str(e).lower()
                is_conn_error = any(k in err_msg for k in _CONN_ERRORS)

                if is_conn_error and attempt == 0:
                    # Stale connection — reset and retry once
                    logger.info("[KG] 유휴 연결 리셋 감지, 재연결 시도... query=%s", q[:50])
                    from shared import reset_kg_service
                    reset_kg_service()
                    _svc[0] = get_kg_service()
                    if not _svc[0]:
                        return None
                    continue

                if is_conn_error:
                    logger.warning("[KG] 재시도 후에도 연결 실패. query=%s", q[:50])
                else:
                    logger.warning("[KG] 검색 실패 (query=%s): %s", q[:50], e)
                if any(k in err_msg for k in _RESET_KEYWORDS):
                    from shared import reset_kg_service
                    reset_kg_service()
                return None
        return None
    
    # If English query is provided, search with both and merge
    all_nodes, all_edges = [], []
    seen_nodes, seen_edges = set(), set()
    
    for q in [query, query_en] if query_en and query_en != query else [query]:
        result = _do_search(q)
        if not result:
            continue
        for n in result.get("nodes", []):
            if n.get("uuid") and n["uuid"] not in seen_nodes:
                seen_nodes.add(n["uuid"])
                all_nodes.append(n)
        for e in result.get("edges", []):
            if e.get("uuid") and e["uuid"] not in seen_edges:
                seen_edges.add(e["uuid"])
                all_edges.append(e)
    
    if not all_nodes and not all_edges:
        return None
    
    lines = []
    if all_nodes:
        lines.append("[Knowledge Graph: Entities]")
        for n in all_nodes:
            # Truncate summary to 150 chars for token efficiency (개선 5)
            summary = (n.get("summary", "") or "")[:150] + ("..." if len(n.get("summary", "") or "") > 150 else "")
            lines.append(f"- {n['name']} ({', '.join(n.get('labels', []))}): {summary}")
    if all_edges:
        lines.append("[Knowledge Graph: Facts/Relations]")
        for e in all_edges:
            lines.append(f"- {e['fact']}")
    return "\n".join(lines)


def _merge_kg_contexts(existing: Optional[str], new_text: Optional[str]) -> Optional[str]:
    """Merge two kg_context strings, deduplicating entities by name and facts by text."""
    if not new_text:
        return existing
    if not existing:
        return new_text

    # Parse a kg_context block into (entity_dict, fact_set)
    def _parse(text):
        entities = {}   # name -> full line
        facts = set()   # fact text
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("[Knowledge Graph:"):
                continue
            if line.startswith("- "):
                content = line[2:]
                # Entity line: "Name (Label, ...): summary"
                if "):  " in content or "): " in content:
                    paren_idx = content.find(" (")
                    if paren_idx != -1:
                        name = content[:paren_idx].strip()
                        if name not in entities:
                            entities[name] = content
                        continue
                # Fact line
                facts.add(content)
        return entities, facts

    old_ents, old_facts = _parse(existing)
    new_ents, new_facts = _parse(new_text)

    # Merge: existing entries take priority (first seen wins)
    for name, line in new_ents.items():
        if name not in old_ents:
            old_ents[name] = line
    old_facts.update(new_facts)

    if not old_ents and not old_facts:
        return None

    lines = []
    if old_ents:
        lines.append("[Knowledge Graph: Entities]")
        for ent_line in old_ents.values():
            lines.append(f"- {ent_line}")
    if old_facts:
        lines.append("[Knowledge Graph: Facts/Relations]")
        for fact in sorted(old_facts):
            lines.append(f"- {fact}")
    return "\n".join(lines)


def _count_kg_items(kg_context: str) -> int:
    """Count entity/fact bullet lines in formatted kg_context text."""
    return sum(1 for line in kg_context.splitlines() if line.strip().startswith("- "))


# Node: Retrieve — uses pre-resolved search_queries from analyze_intent (no LLM calls)
def retrieve_node(state: AgentState):
    query = state["messages"][-1].content
    search_queries = state.get("search_queries")
    logs = []

    selected_layer = state.get("layer", "all")
    logs.append(f"\n📂 [레이어] '{selected_layer}' 레이어에서 검색합니다.")

    # Build fallback search query if analysis didn't produce any
    if not search_queries:
        needs_en = selected_layer in ("core_theory", "all")
        search_queries = [{"ko": query, "en": query if needs_en else None}]

    docs = []
    try:
        if len(search_queries) > 1:
            # Multi-retrieval — run each pre-resolved query independently
            for i, sq in enumerate(search_queries, 1):
                logs.append(f"\n🔍 [검색 {i}/{len(search_queries)}] \"{sq['ko']}\"")
                sq_docs = _retrieve_for_query(sq["ko"], sq.get("en"), selected_layer)
                logs.append(f"   → {len(sq_docs)}건 발견")
                for d in sq_docs:
                    logs.append(f"      📄 {_label_doc(d)}")
                docs.extend(sq_docs)
            before = len(docs)
            docs = _deduplicate_docs(docs)
            if before > len(docs):
                logs.append(f"🔗 [병합] {before}건 → {len(docs)}건 (중복 {before - len(docs)}건 제거)")
        else:
            sq = search_queries[0]
            docs = _retrieve_for_query(sq["ko"], sq.get("en"), selected_layer)

        if docs:
            logs.append(f"✅ {len(docs)}개의 혁명 문헌을 발견했습니다:")
            for d in docs:
                logs.append(f"   📄 {_label_doc(d)}")
        else:
            logs.append("⚠️ 영묘 데이터에 관련된 문헌이 없습니다.")

    except Exception as e:
        logs.append(f"⚠️ 검색 중 오류 발생 (무시하고 진행): {e}")

    return {"documents": docs, "logs": logs}

# Heuristic KG filter — replaces LLM-based KG grading
def _heuristic_kg_filter(question: str, fact: str) -> bool:
    """Return False if the fact is tautological or too short to be useful."""
    if len(fact) < 15:
        return False
    # Check if all entity-like words from the fact appear in the question (tautological)
    q_lower = question.lower()
    # Extract capitalized entity names (words starting with uppercase, length > 2)
    entities = re.findall(r'\b[A-Z\uac00-\ud7a3][A-Za-z\uac00-\ud7a3]{2,}\b', fact)
    if len(entities) >= 2 and all(e.lower() in q_lower for e in entities):
        return False
    return True


# Node: KG Retrieve (knowledge graph search with heuristic filtering, no LLM calls)
def kg_retrieve_node(state):
    query = state["messages"][-1].content
    search_queries_raw = state.get("search_queries")
    logs = []

    # Extract ko queries for KG search
    if search_queries_raw and len(search_queries_raw) > 1:
        kg_queries = [sq["ko"] for sq in search_queries_raw]
    elif search_queries_raw:
        kg_queries = [search_queries_raw[0]["ko"]]
    else:
        kg_queries = [query]

    is_complex = len(kg_queries) > 1 or state.get("intent") in ("strategic", "academic")
    num_results = 15 if is_complex else 8

    merged_kg = None
    total_items = 0

    for i, sq in enumerate(kg_queries, 1):
        kg_text = _search_kg(sq, num_results=num_results)
        if kg_text:
            merged_kg = _merge_kg_contexts(merged_kg, kg_text)
            count = _count_kg_items(kg_text)
            total_items += count
            logs.append(f"   🧩 [KG 질의 {i}/{len(kg_queries)}] \"{sq}\" → {count}건")
        else:
            logs.append(f"   🧩 [KG 질의 {i}/{len(kg_queries)}] \"{sq}\" → 0건")

    # Heuristic filter (replaces LLM-based KG grading — saves 1 LLM call)
    if merged_kg:
        logs.insert(0, f"\n🕸️ [지식그래프] 총 {total_items}건의 구조화된 팩트를 확보했습니다.")

        kg_lines = merged_kg.splitlines()
        filtered_parts = []
        filtered_count = 0
        section_header = None

        for line in kg_lines:
            if "[Knowledge Graph:" in line:
                section_header = line
                continue
            if line.startswith("- "):
                fact_text = line[2:]
                if _heuristic_kg_filter(query, fact_text):
                    if section_header:
                        filtered_parts.append(section_header)
                        section_header = None
                    filtered_parts.append(line)
                    filtered_count += 1

        if filtered_parts:
            merged_kg = "\n".join(filtered_parts)
            if filtered_count < total_items:
                logs.append(f"   ⚖️ [KG 필터] {total_items}건 → {filtered_count}건 (동어반복 {total_items - filtered_count}건 제거)")
        else:
            merged_kg = None
            logs.append(f"   ⚖️ [KG 필터] 모든 항목 필터링됨 (0건)")
    else:
        logs.insert(0, "\n🕸️ [지식그래프] 관련 팩트 없음.")

    return {"kg_context": merged_kg, "logs": logs}

# Node: Grade Documents (Merge 3+4: batch grading + realtime check in one call)
def grade_documents_node(state: AgentState):
    question = state["messages"][-1].content
    documents = state["documents"]
    logs = []
    logs.append("\n⚖️ [검열관] 문헌의 적절성을 일괄 평가 중...")

    if not documents:
        logs.append("   ⚠️ 연관있는 문헌이 없다.")
        return {"documents": [], "logs": logs}

    # Rate-limit guard: brief pause between analyze_intent and batch_grader
    time.sleep(0.5)

    # Build numbered document list for batch grading
    doc_entries = []
    for i, d in enumerate(documents, 1):
        formatted = _format_doc(d)
        # Truncate each doc to avoid token overflow
        if len(formatted) > 500:
            formatted = formatted[:500] + "..."
        doc_entries.append(f"[Document {i}]\n{formatted}")
    docs_text = "\n\n".join(doc_entries)

    result = invoke_batch_grader({"question": question, "documents": docs_text})

    # Parse scores — fallback to all "yes" if batch grading fails
    if result and result.scores:
        scores = result.scores
        needs_realtime = result.needs_realtime
    else:
        scores = ["yes"] * len(documents)
        needs_realtime = "yes"
        logs.append("   ⚠️ 일괄 평가 실패, 모든 문헌을 유지합니다.")

    # Pad or trim scores to match document count
    while len(scores) < len(documents):
        scores.append("yes")
    scores = scores[:len(documents)]

    filtered_docs = []
    for i, (d, score) in enumerate(zip(documents, scores)):
        meta = d.metadata or {}
        author = meta.get("author", "")
        year = meta.get("year", "")
        source = meta.get("source", "출처미상")
        label = f"{source} ({author}, {year})" if author or year else source

        if score == "yes":
            logs.append(f"   ✅ 적절한 문헌: {label}")
            filtered_docs.append(d)
        else:
            logs.append(f"   🗑️ 관련없는 문헌(무시): {label}")

    # Fallback: keep at least 1 document
    if not filtered_docs and documents:
        filtered_docs = [documents[0]]

    return {"documents": filtered_docs, "logs": logs, "needs_realtime": needs_realtime}


def decide_websearch_need(state: AgentState):
    filtered_docs = state["documents"]

    # Always search web if too few documents
    if len(filtered_docs) <= 1:
        return "need_web_search"

    # Read realtime decision from batch grading (Merge 4: no separate LLM call)
    if state.get("needs_realtime") == "yes":
        return "need_web_search"

    return "no_need_to_search_web"

# Gemini Google Search grounding LLM (single instance, reused across calls)
_gemini_search_llm = ChatGoogleGenerativeAI(
    model=MODEL_MAIN,
    google_api_key=GEMINI_API_KEY,
    temperature=0.0,
    max_output_tokens=2048,
    streaming=False,
    max_retries=3,
).bind(tools=[{"google_search": {}}])


# Helper: Web Search — uses Gemini API Google Search grounding
def _run_web_search(query: str, logs: list) -> list:
    """Invoke Gemini with Google Search grounding and return results as Document objects."""
    try:
        response = _invoke_with_backoff(
            lambda: _gemini_search_llm.invoke(query)
        )
        content = extract_text_content(response.content) if hasattr(response, "content") else str(response)
        if not content or not content.strip():
            logs.append("  ⚠️ 웹 검색 결과 없음")
            return []

        # Extract grounding metadata (sources) if available
        grounding_meta = getattr(response, "response_metadata", {}).get("grounding_metadata", {})
        grounding_chunks = grounding_meta.get("grounding_chunks", [])
        # Also check additional_kwargs
        if not grounding_chunks:
            additional = getattr(response, "additional_kwargs", {})
            grounding_chunks = additional.get("grounding_metadata", {}).get("grounding_chunks", [])

        docs = []
        if grounding_chunks:
            for chunk in grounding_chunks:
                web_info = chunk.get("web", {})
                docs.append(Document(
                    page_content=content if len(grounding_chunks) == 1 else content[:500],
                    metadata={
                        "source": web_info.get("uri", ""),
                        "title": web_info.get("title", ""),
                    }
                ))
        else:
            # No grounding metadata — wrap the whole response as a single Document
            docs.append(Document(
                page_content=content,
                metadata={"source": "gemini_search", "title": query}
            ))

        logs.append(f"  ✅ {len(docs)}건의 웹 결과를 확보했습니다.")
        for doc in docs:
            logs.append(f"   🌐 {_label_doc(doc)}")
        return docs
    except Exception as e:
        logs.append(f"  ⚠️ 웹 검색 실패: {e}")
        return []


# Node: Web Search
def web_search_node(state: AgentState):
    """
    Search the external world to gather more context.
    """
    question = state["messages"][-1].content
    current_docs = list(state.get("documents", []))
    logs = []
    has_docs = len(current_docs) > 1
    if has_docs:
        logs.append(f"\n🌐 [웹 검색] 문헌 {len(current_docs)}건 확보 — 실시간 정보 보충을 위해 외부 정찰 개시")
    else:
        logs.append(f"\n🌐 [웹 검색] 문헌 부족 — 외부 세계를 정찰")
    web_docs = _run_web_search(question, logs)
    current_docs.extend(web_docs)
    return {"documents": current_docs, "logs": logs}


# Helper: Fetch a single self-knowledge source based on tool name
def _fetch_self_knowledge(tool_name: str) -> str:
    """Fetch one specific self-knowledge source. Returns formatted text or empty string."""
    try:
        if tool_name == "read_diary":
            from shared import fetch_diaries
            diaries = fetch_diaries(3)
            if not diaries:
                return "(No diary entries yet.)"
            lines = []
            for d in diaries:
                ts = d.get("created_at", "?")
                if isinstance(ts, str) and len(ts) > 16:
                    ts = ts[:16].replace("T", " ")
                title = d.get("title", "Untitled")
                content = d.get("content", "")[:600]
                lines.append(f"[{ts}] {title}\n{content}")
            return "[MY DIARY ENTRIES]\n" + "\n---\n".join(lines)

        elif tool_name == "read_chat_logs":
            from shared import fetch_chat_logs
            rows = fetch_chat_logs(15, hours_back=24)
            if not rows:
                return "(No recent conversations.)"
            lines = []
            for row in rows:
                q = str(row.get("user_query", ""))[:100]
                a = str(row.get("bot_answer", ""))[:150]
                lines.append(f"- User: {q}\n  Me: {a}")
            return f"[MY RECENT CONVERSATIONS ({len(rows)} in last 24h)]\n" + "\n".join(lines)

        elif tool_name == "read_system_status":
            from shared import fetch_diaries, fetch_chat_logs, fetch_task_reports
            parts = []
            diaries = fetch_diaries(1)
            if diaries:
                last = diaries[0]
                ts = last.get("created_at", "?")
                if isinstance(ts, str) and len(ts) > 16:
                    ts = ts[:16].replace("T", " ")
                parts.append(f"Last diary: {ts} — {last.get('title', 'N/A')}")
            logs_24h = fetch_chat_logs(1000, hours_back=24)
            parts.append(f"Chat activity (24h): {len(logs_24h)} conversations")
            tasks = fetch_task_reports(20)
            if tasks:
                by_status = {}
                for t in tasks:
                    s = t.get("status", "?")
                    by_status[s] = by_status.get(s, 0) + 1
                parts.append("Tasks: " + ", ".join(f"{k}: {v}" for k, v in by_status.items()))
            current_dt = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
            parts.append(f"Current time: {current_dt}")
            return "[MY SYSTEM STATUS]\n" + "\n".join(parts)

        elif tool_name == "read_recent_updates":
            from shared import fetch_recent_updates
            updates = fetch_recent_updates(max_entries=3, max_chars=1200)
            if updates and not updates.startswith("("):
                return f"[MY RECENT SYSTEM UPDATES]\n{updates}"
            return "(No recent updates.)"

    except Exception as e:
        logger.warning("[자기인식] %s 조회 실패: %s", tool_name, e)
    return ""


# Node: Generate (merged with strategize — dialectical analysis instructions inline)
def generate_node(state: AgentState):
    docs = state.get("documents", [])
    docs = docs[:12]
    # Truncate each doc to 400 chars for token efficiency
    context = "\n\n".join([_format_doc(d)[:400] for d in docs]) if docs else ""

    # URL documents: include with higher char limit (user explicitly referenced these)
    url_docs = state.get("url_documents", [])
    if url_docs:
        url_context = "\n\n".join([f"[USER-REFERENCED URL: {d.metadata.get('source', '')}]\n{d.page_content}" for d in url_docs])
        context = f"{url_context}\n\n{context}" if context else url_context

    messages = state["messages"]
    last_user_query = messages[-1].content
    intent = state.get("intent", "casual")

    logs = []

    # Self-knowledge
    self_section = ""
    self_tool = state.get("self_knowledge_tool")
    if self_tool:
        self_data = _fetch_self_knowledge(self_tool)
        if self_data:
            self_section = f"\n[SELF-KNOWLEDGE]\n{self_data}"
            logs.append(f"🪞 [자기인식] {self_tool} 데이터 로드 완료")

    # Knowledge Graph
    kg_context = state.get("kg_context")
    kg_section = f"\n[STRUCTURED INTELLIGENCE]\n{kg_context}" if kg_context else ""

    # Phase 3: Include step results summary if from plan path
    step_section = ""
    step_results = state.get("step_results", [])
    if step_results:
        step_section = f"\n[RESEARCH STEPS]\n" + "\n".join(step_results)

    # Experiential memory — past lessons/patterns relevant to this query
    exp_section = ""
    if intent != "casual":
        exp_rows = search_experiential_memory(last_user_query, k=3)
        if exp_rows:
            exp_lines = [f"[{r['category']}] {r['content']}" for r in exp_rows]
            exp_section = "\n[PAST EXPERIENCE]\n" + "\n".join(exp_lines)
            logs.append(f"🧠 [경험] 관련 경험 {len(exp_rows)}건 로드")

    # Intent-specific style + mission
    style_map = {
        "academic": ("Detailed, objective explanation grounded in context.", "Professional, authoritative. Precise terminology."),
        "strategic": ("Concrete, actionable strategy. Synthesize—don't restate sources.", "Decisive, analytical. Clear phases or numbered steps."),
        "casual": ("Respond with wit and revolutionary charm.", "Natural, friendly, brief (1-3 sentences)."),
    }
    mission, style = style_map.get(intent, style_map["casual"])

    current_dt = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    # Static content first (maximizes Gemini implicit caching), variable content last
    # CORE_IDENTITY + STYLE + MISSION + ANALYSIS METHOD are stable across calls → cacheable prefix
    # CURRENT TIME + SOURCE MATERIAL + QUESTION change per call → placed at the end
    # Skip analysis method and source material for casual queries (saves ~150 tokens)
    if intent == "casual" and not context and not self_section:
        system_prompt = f"""{CORE_IDENTITY}
[STYLE] {style}
[MISSION] {mission}

[CURRENT TIME] {current_dt}
[QUESTION] {last_user_query}

Respond in the same language as the user's question."""
    else:
        analysis_method = "\n[ANALYSIS METHOD] Ground in concrete conditions and class forces. Identify internal contradictions driving development. Note quantitative-to-qualitative tipping points. Connect to broader economic/political/ideological relations. Apply only what illuminates the question. Keep jargon minimal.\n" if intent != "casual" else ""
        system_prompt = f"""{CORE_IDENTITY}
[STYLE] {style}
[MISSION] {mission}
{analysis_method}
[CURRENT TIME] {current_dt}
[SOURCE MATERIAL] {context if context else "(No sources found.)"}{kg_section}{step_section}{exp_section}{self_section}

[QUESTION] {last_user_query}

Respond in the same language as the user's question."""

    safe_system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", safe_system_prompt),
        ("placeholder", "{messages}")
    ])

    # Trim conversation history to last 6 messages (3 turns) — system prompt already has all context
    trimmed = messages[-6:] if len(messages) > 6 else messages

    chain = prompt | llm
    response = _invoke_with_backoff(lambda: chain.invoke({"messages": trimmed}))
    text = extract_text_content(response.content)
    normalized = AIMessage(content=text)
    logs.append(f"💬 [생성] '{intent}' 의도에 적합한 답변 생성")

    return {"messages": [normalized], "logs": logs}

# Node: Log Conversation to Supabase
def log_conversation_node(state: AgentState, config: RunnableConfig):
    logs = []
    try:
        messages = state["messages"]
        # Find the last HumanMessage and AIMessage
        user_query = ""
        bot_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not bot_answer:
                bot_answer = msg.content
            elif isinstance(msg, HumanMessage) and not user_query:
                user_query = msg.content
            if user_query and bot_answer:
                break

        docs = state.get("documents", [])
        turn_start = state.get("logs_turn_start", 0)
        processing_logs = state.get("logs", [])[turn_start:]

        datasource = state.get("datasource", "generate")
        route = datasource if datasource else ("casual" if not docs else "vectorstore")

        web_search_used = any("웹 검색" in log or "Web Search" in log for log in processing_logs)

        cfg = config.get("configurable", {})
        session_id = cfg.get("thread_id", "unknown")
        fingerprint = cfg.get("fingerprint", "")
        user_agent = cfg.get("user_agent", "")
        ip_address = cfg.get("ip_address", "")

        row = {
            "session_id": session_id,
            "fingerprint": fingerprint,
            "user_agent": user_agent,
            "ip_address": ip_address,
            "user_query": user_query,
            "bot_answer": bot_answer,
            "route": route,
            "documents_count": len(docs),
            "web_search_used": web_search_used,
            "strategy": None,
            "processing_logs": processing_logs,
        }

        db_execute(
            """INSERT INTO chat_logs
               (session_id, fingerprint, user_agent, ip_address,
                user_query, bot_answer, route, documents_count,
                web_search_used, strategy, processing_logs)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (row["session_id"], row["fingerprint"], row["user_agent"],
             row["ip_address"], row["user_query"], row["bot_answer"],
             row["route"], row["documents_count"], row["web_search_used"],
             row["strategy"], row["processing_logs"]),
        )
    except Exception as e:
        print(f"⚠️ [로그] DB 기록 실패: {e}")

    return {}


# Phase 3: Plan-and-Execute — multi-step research for complex strategic queries

class PlanStep(BaseModel):
    """A single step in a research plan."""
    description: str = Field(..., description="What this step investigates")
    tool: Literal["retrieve", "web_search"] = Field(..., description="Which tool to use")
    query: str = Field(..., description="Search query for this step")

class ResearchPlan(BaseModel):
    """Structured research plan for complex queries."""
    steps: List[PlanStep] = Field(..., description="Ordered list of research steps (2-4 steps)")

system_planner = """Create a 2-4 step research plan. Each step: one focused angle, one tool, one query.
Tools: "retrieve" (internal Marxist-Leninist knowledge base) | "web_search" (current events/data).
Query language: Korean for current/domestic, English for classical theory.
Build progressively: concrete facts → structural forces → theoretical framework.
{{"steps": [{{"description": "...", "tool": "retrieve"|"web_search", "query": "..."}}, ...]}}"""

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", system_planner),
    ("human", "Question: {question}\n\nConversation context:\n{context}"),
])

_PLAN_DEFAULT = ResearchPlan(steps=[
    PlanStep(description="General search", tool="retrieve", query=""),
])


def planner_node(state: AgentState):
    """Create a multi-step research plan for complex strategic queries."""
    question = state["messages"][-1].content
    context = _build_context(state["messages"])
    logs = []

    logs.append("\n📋 [계획관] 연구 계획을 수립 중...")

    result = _structured_call(
        planner_prompt, llm, {"question": question, "context": context},
        ResearchPlan, _PLAN_DEFAULT
    )

    # If fallback default was used, set the query to the actual question
    plan_steps = []
    for step in result.steps:
        step_dict = step.model_dump()
        if not step_dict["query"]:
            step_dict["query"] = question
        plan_steps.append(step_dict)

    logs.append(f"📋 [계획관] {len(plan_steps)}단계 연구 계획 수립 완료:")
    for i, step in enumerate(plan_steps, 1):
        tool_icon = "📚" if step["tool"] == "retrieve" else "🌐"
        logs.append(f"   {i}. {tool_icon} {step['description']}")
        logs.append(f"      쿼리: \"{step['query']}\"")

    return {
        "plan": plan_steps,
        "current_step": 0,
        "logs": logs,
    }


def step_executor_node(state: AgentState):
    """Execute the current step of the research plan.
    
    개선 4: KG 중복 검색 최적화 — 이전 단계와 쿼리가 유사하면 KG 검색 스킵.
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    logs = []

    if current_step >= len(plan):
        logs.append("⚠️ [실행관] 계획의 모든 단계가 이미 완료되었습니다.")
        return {"logs": logs}

    step = plan[current_step]
    step_num = current_step + 1
    total = len(plan)
    logs.append(f"\n⚡ [실행관] 단계 {step_num}/{total}: {step['description']}")

    current_docs = list(state.get("documents") or [])
    selected_layer = state.get("layer", "all")
    context = _build_context(state["messages"])
    query = step["query"]

    if step["tool"] == "retrieve":
        # The planner already generates context-resolved queries in the right language.
        # Skip the extra flash-lite rewrite call to avoid rate-limit cascades.
        needs_english = selected_layer in ("core_theory", "all")
        sq_ko = query
        sq_en = query if needs_english else None
        new_docs = _retrieve_for_query(sq_ko, sq_en, selected_layer)
        logs.append(f"   📚 {len(new_docs)}건의 문헌을 발견했습니다.")
        for d in new_docs:
            logs.append(f"      📄 {_label_doc(d)}")

        # Summarize what we found for step_results
        doc_snippets = []
        for d in new_docs:
            snippet = _format_doc(d)[:200]
            doc_snippets.append(snippet)
        result_summary = f"[Step {step_num}: {step['description']}] Retrieved {len(new_docs)} docs. Key content: " + " | ".join(doc_snippets[:3])

        current_docs.extend(new_docs)

    elif step["tool"] == "web_search":
        logs.append(f"   🌐 웹 검색: \"{query}\"")
        web_docs = _run_web_search(query, logs)
        current_docs.extend(web_docs)
        if web_docs:
            snippets = " | ".join(d.page_content[:200] for d in web_docs[:3])
            result_summary = f"[Step {step_num}: {step['description']}] Web search found {len(web_docs)} results: {snippets}"
        else:
            result_summary = f"[Step {step_num}: {step['description']}] Web search returned no results."

    # 개선 4: KG 중복 검색 최적화
    # 이전 단계의 쿼리 목록과 현재 쿼리를 비교하여 유사하면 스킵
    prev_queries = list(state.get("_kg_searched_queries", []))
    should_search_kg = True
    
    # Simple similarity check: if current query is substring/superset of previous queries
    for prev_q in prev_queries:
        # Check if queries are substantially overlapping (>70% word overlap)
        curr_words = set(query.lower().split())
        prev_words = set(prev_q.lower().split())
        if not curr_words or not prev_words:
            continue
        overlap = len(curr_words & prev_words) / max(len(curr_words), len(prev_words))
        if overlap > 0.7:
            should_search_kg = False
            logs.append(f"   🕸️ [지식그래프] 이전 단계와 쿼리 중복 — 스킵 (유사도 {overlap:.0%})")
            break
    
    kg_text = None
    if should_search_kg:
        # Determine num_results dynamically (개선 5: Token 효율성)
        num_results = 12 if step["tool"] == "retrieve" else 8
        kg_text = _search_kg(query, num_results=num_results)
        if kg_text:
            kg_count = _count_kg_items(kg_text)
            logs.append(f"   🕸️ [지식그래프] {kg_count}건의 팩트 확보")
            # Track this query to avoid future duplicates
            prev_queries.append(query)
        else:
            logs.append(f"   🕸️ [지식그래프] 관련 팩트 없음")
    else:
        # Skip KG search — previous context already covers this query
        kg_text = None

    # Deduplicate accumulated docs
    current_docs = _deduplicate_docs(current_docs)

    # Manually accumulate step_results (no add reducer — needed for checkpoint reset)
    current_results = list(state.get("step_results") or [])
    current_results.append(result_summary)

    # Merge KG results into kg_context (every step)
    return_dict = {
        "documents": current_docs,
        "current_step": current_step + 1,
        "step_results": current_results,
        "logs": logs,
        "_kg_searched_queries": prev_queries,  # Track searched queries
    }
    if kg_text:
        return_dict["kg_context"] = _merge_kg_contexts(state.get("kg_context"), kg_text)
    return return_dict


def plan_progress(state: AgentState):
    """Check if there are more plan steps to execute."""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step < len(plan):
        return "continue"
    return "done"


# 그래프(Workflow) 구성
workflow = StateGraph(AgentState)
workflow.add_node("analyze_intent", analyze_intent_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("log_conversation", log_conversation_node)
# Knowledge Graph retrieval node (vectorstore path only; plan path runs KG per step inside step_executor)
workflow.add_node("kg_retrieve", kg_retrieve_node)
# Phase 3: Plan-and-execute nodes
workflow.add_node("planner", planner_node)
workflow.add_node("step_executor", step_executor_node)

workflow.add_edge(START, "analyze_intent")
# 3-way routing: plan (strategic complex) / vectorstore (standard RAG) / generate (casual)
workflow.add_conditional_edges("analyze_intent", router_logic, {
    "vectorstore": "retrieve",
    "generate": "generate",
    "plan": "planner",
})
# Vectorstore path: retrieve → kg_retrieve → grade_documents
workflow.add_edge("retrieve", "kg_retrieve")
workflow.add_edge("kg_retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_websearch_need, {
    "need_web_search": "web_search",
    "no_need_to_search_web": "generate",
})
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "log_conversation")
workflow.add_edge("log_conversation", END)
# Phase 3: planner → step_executor → [continue → step_executor | done → generate]
workflow.add_edge("planner", "step_executor")
workflow.add_conditional_edges("step_executor", plan_progress, {
    "continue": "step_executor",
    "done": "generate",
})
graph = workflow.compile(checkpointer=MemorySaver())

# 실행 루프 (채팅 인터페이스)
if __name__ == "__main__":
    print("🚩 [System] 사이버-레닌 AI 가동됨.")
    print("🚩 [System] 당신의 영혼이 레닌 영묘와 연결되었습니다. 레닌 동지에게 말을 거십시오.\n")

    config = {"configurable": {"thread_id": "cli-session"}}

    while True:
        try:
            user_input = input("혁명가(나): ")
            if user_input.lower() in ["exit", "quit", "종료"]:
                print("🚩 통신 종료. 혁명은 계속된다.")
                break

            inputs = {"messages": [HumanMessage(content=user_input)]}
            for output in graph.stream(inputs, config=config, stream_mode="updates"):
                for node_name, node_content in output.items():
                    if node_content and "logs" in node_content:
                        for log_line in node_content["logs"]:
                            print(log_line)
                    if node_name == "generate":
                        answer = node_content["messages"][-1].content
                        print(f"\n💬 사이버-레닌: {answer}")
            print()
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
