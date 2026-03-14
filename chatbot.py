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
import json
import re
import time
from datetime import datetime
from typing import Literal

from shared import (
    extract_text_content, CORE_IDENTITY, KST, MODEL_MAIN, MODEL_LIGHT,
    get_tavily_search, get_kg_service, run_kg_async,
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



def _extract_json(text, model_class: type[BaseModel]):
    """Try to extract and validate JSON from LLM response text. Returns None on failure."""
    # Normalize list content (extended thinking models return list of typed blocks)
    if not isinstance(text, str):
        text = extract_text_content(text)
    # Try direct parse
    try:
        return model_class.model_validate_json(text.strip())
    except Exception:
        pass
    # Try extracting from markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return model_class.model_validate(json.loads(match.group(1)))
        except Exception:
            pass
    # Try scanning for the first decodable JSON object (handles nested objects)
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return model_class.model_validate(obj)
        except Exception:
            continue
    return None


def _invoke_structured(chain, inputs: dict, model_class: type[BaseModel], default, max_retries: int = 2, retry_llm=None):
    """Invoke LLM chain, parse JSON from response. Retry with error feedback on failure, fall back to default."""
    try:
        resp = _invoke_with_backoff(lambda: chain.invoke(inputs))
    except Exception as e:
        logger.warning("[구조화] LLM 호출 실패 (기본값 사용): %s", e)
        return default
    result = _extract_json(resp.content, model_class)
    if result is not None:
        return result

    # Retry: send the bad output back with a correction prompt
    _llm = retry_llm or llm
    for attempt in range(max_retries):
        correction = _invoke_with_backoff(lambda: _llm.invoke(
            f"Your previous output was not valid JSON:\n{resp.content}\n\n"
            f"Respond with ONLY a valid JSON object matching this schema: {model_class.model_json_schema()}"
        ))
        result = _extract_json(correction.content, model_class)
        if result is not None:
            return result

    logger.warning("[구조화] %s JSON 파싱 실패, 기본값 사용", model_class.__name__)
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
    max_output_tokens=256,
    streaming=False,
    max_retries=2,
)
web_search_tool = get_tavily_search()
logger.info("✅ [성공] 모든 시스템 기동 완료.")




# 2. 상태(State) 정의
# 대화 기록(messages)과 검색된 문서를 저장하는 메모리 구조입니다.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]
    strategy: Optional[str]
    intent: Optional[Literal["academic", "strategic", "casual"]]
    datasource: Optional[Literal["vectorstore", "generate", "plan"]]
    logs: Annotated[List[str], add]
    # Phase 2: Query decomposition
    sub_queries: Optional[List[str]]  # decomposed sub-queries (None = simple query)
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

# Merge 1: Combined query analysis (router + layer router + decompose in one LLM call)
class QueryAnalysis(BaseModel):
    """Combined intent classification, layer routing, and query decomposition."""
    datasource: Literal["vectorstore", "generate"] = Field(..., description="Whether knowledge retrieval is needed")
    intent: Literal["academic", "strategic", "casual"] = Field(..., description="Response style")
    layer: Literal["core_theory", "modern_analysis", "all"] = Field(default="all", description="Which knowledge layer to search")
    sub_queries: List[str] = Field(default_factory=list, description="Decomposed sub-queries, or single original query")
    needs_plan: bool = Field(default=False, description="Whether the query requires multi-step research planning")

system_query_analysis = """You are an expert query analyzer for the Cyber-Lenin AI.
Analyze the user's question and conversation context to determine ALL of the following in ONE response.

Use the conversation context to resolve pronouns, references, and follow-up questions.

1. **datasource**: Where to look for the answer.
   - "vectorstore": Query requires historical, theoretical, or technical knowledge, or knowledge about modern society or revolutionary experiences.
   - "generate": Greetings, personal talk, or simple requests not needing external data.

2. **intent**: How to respond.
   - "academic": Objective, detailed, scholarly explanations.
   - "strategic": Actionable plans, tactics, or "how-to" advice.
   - "casual": Simple greetings, jokes, or non-political chit-chat.

3. **layer**: Which knowledge layer to search (only matters if datasource="vectorstore").
   - "core_theory": Classical Marxist-Leninist texts (Lenin, Marx, Engels, original writings, early 20th century).
   - "modern_analysis": Contemporary analysis (AI, tech, current politics, 21st century economics).
   - "all": When the question spans both classical and modern topics. When unsure, use "all".

4. **sub_queries**: Query decomposition for retrieval (only matters if datasource="vectorstore").
   - DECOMPOSE ONLY when the question explicitly asks about 2+ DIFFERENT thinkers, theories, or distinct topics that need separate searches.
   - DO NOT decompose single-topic questions, even if complex. When in doubt, do NOT decompose.
   - If decomposing: output 2-4 focused sub-queries targeting DIFFERENT search topics.
   - If NOT decomposing: output the original question as the single item in the list.

5. **needs_plan**: Whether the query requires multi-step research and synthesis (only matters if datasource="vectorstore").
   - true: The question requires COMBINING knowledge from multiple angles, building an argument across several research steps, or producing a structured analysis/strategy. Examples: "Analyze X crisis from a Marxist perspective and propose a strategy", "Compare and synthesize theories A, B, C into an actionable framework".
   - false: The question can be answered with a single retrieval pass (even if complex). Most questions are false. When in doubt, use false.

Respond with ONLY a JSON object:
{{"datasource": "...", "intent": "...", "layer": "...", "sub_queries": ["..."], "needs_plan": true|false}}"""

query_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", system_query_analysis),
    ("human", "Conversation context:\n{context}\n\nCurrent question: {question}"),
])

_ANALYSIS_DEFAULT = QueryAnalysis(datasource="vectorstore", intent="academic", layer="all", sub_queries=[], needs_plan=False)

def invoke_query_analysis(inputs: dict) -> QueryAnalysis:
    return _invoke_structured(query_analysis_prompt | llm_light, inputs, QueryAnalysis, _ANALYSIS_DEFAULT, retry_llm=llm_light)


# Merge 3+4: Batch document grading + realtime check in one LLM call
class BatchGradeResult(BaseModel):
    """Batch relevance grading for all documents + realtime info check."""
    scores: List[Literal["yes", "no"]] = Field(..., description="Relevance score for each document in order")
    needs_realtime: Literal["yes", "no"] = Field(default="no", description="Whether the question would benefit from real-time web search")

system_batch_grader = """You are a document relevance grader and information freshness evaluator.
You will receive a user question and multiple retrieved documents (numbered).

For EACH document, determine if it would be useful for answering the question.
- Grade "yes" if the document contains relevant information, context, or perspectives.
- Grade "no" only if the document is clearly unrelated.
- When in doubt, grade "yes".

Also determine if the question would benefit from real-time web search:
- "yes" if the question involves current events, recent data (2020+), modern organizations, or applying theory to current conditions.
- "no" if purely about historical events or classical theory with no modern angle.

Respond with ONLY a JSON object:
{{"scores": ["yes", "no", ...], "needs_realtime": "yes"|"no"}}"""

batch_grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_batch_grader),
    ("human", "User question: {question}\n\nDocuments:\n{documents}"),
])

_BATCH_GRADE_DEFAULT = None  # handled in node

def invoke_batch_grader(inputs: dict) -> BatchGradeResult | None:
    return _invoke_structured(batch_grade_prompt | llm_light, inputs, BatchGradeResult, _BATCH_GRADE_DEFAULT, retry_llm=llm_light)

# KG Batch Grader — Knowledge Graph 결과의 관련성 평가
class BatchGradeKGResult(BaseModel):
    """Batch relevance grading for Knowledge Graph nodes and edges."""
    node_scores: List[Literal["yes", "no"]] = Field(..., description="Relevance score for each node in order")
    edge_scores: List[Literal["yes", "no"]] = Field(..., description="Relevance score for each edge/fact in order")

system_batch_grade_kg = """You are a Knowledge Graph relevance grader.
You will receive a user question and structured KG results (nodes and edges).

For EACH node and edge, determine if it would add **meaningful, non-obvious information** for answering the question.
- Grade "yes" if it contains specific, informative facts that help answer the question.
- Grade "no" if it is:
  - Clearly unrelated to the question.
  - Tautological or self-evident (e.g. "X is a party to the X-Y War", "Country A is involved in conflict with Country B" when that is the premise of the question).
  - A mere restatement of the question or its obvious assumptions.
- When in doubt, grade "yes".

Respond with ONLY a JSON object:
{{"node_scores": ["yes", "no", ...], "edge_scores": ["yes", "no", ...]}}"""

batch_grade_kg_prompt = ChatPromptTemplate.from_messages([
    ("system", system_batch_grade_kg),
    ("human", "User question: {question}\n\nKnowledge Graph Nodes:\n{nodes}\n\nKnowledge Graph Edges/Facts:\n{edges}"),
])

_BATCH_GRADE_KG_DEFAULT = None

def invoke_batch_grade_kg(inputs: dict) -> BatchGradeKGResult | None:
    return _invoke_structured(batch_grade_kg_prompt | llm_light, inputs, BatchGradeKGResult, _BATCH_GRADE_KG_DEFAULT, retry_llm=llm_light)

# Strategist Prompt — 분석적 사고를 위한 간결한 지침
system_strategist = """You are a revolutionary strategist. Analyze using dialectical materialism:

1. **Concrete conditions** — Ground in specific material conditions, class forces, historical context. No abstractions.
2. **Internal contradictions** — Opposing forces *within* the phenomenon driving its development. Not pros/cons.
3. **Quantitative accumulation → qualitative leap** — Where are the tipping points?
4. **Negation of the negation** — Spiral development: what is preserved and elevated? Not "thesis-antithesis-synthesis."
5. **Totality** — Connect to broader economic, political, ideological relations.

Apply only principles that illuminate the question. For simple questions, analyze directly.
Think dialectically, but keep philosophical jargon out of the output unless necessary.

Output a concise blueprint (in English) for the speaker.
"""
strategist_prompt = ChatPromptTemplate.from_messages([
    ("system", system_strategist),
    ("human", "Context: \n{context}\n\nUser Question:\n{question}")
])
strategist_chain = strategist_prompt | llm


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
    logs = [f"\n🚦 [분석] 대화 맥락:\n{context}\n🚦 [분석] 의도: {analysis.intent} / 경로: {analysis.datasource} / 레이어: {analysis.layer}"]

    sub_queries = analysis.sub_queries if len(analysis.sub_queries) > 1 else None
    if sub_queries:
        logs.append(f"🔀 [분해] 복합 질문을 {len(sub_queries)}개의 하위 질문으로 분해:")
        for i, sq in enumerate(sub_queries, 1):
            logs.append(f"   {i}. {sq}")

    # Phase 3: Plan-and-execute routing
    needs_plan = analysis.needs_plan and analysis.datasource == "vectorstore"
    if needs_plan:
        logs.append("📋 [계획] 복합 전략 질문 감지 — 연구 계획 수립 경로로 진입합니다.")

    return {
        "intent": analysis.intent,
        "datasource": "plan" if needs_plan else analysis.datasource,
        "layer": analysis.layer,
        "sub_queries": sub_queries,
        "logs": logs,
        "logs_turn_start": turn_start,
        # Reset transient per-turn fields to prevent state leakage across checkpointed turns
        "documents": [],
        "strategy": None,
        "plan": None,
        "current_step": 0,
        "needs_realtime": None,
        "step_results": [],
        "kg_context": None,
        "_kg_searched_queries": [],  # Reset KG query tracking for new turn
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

# Helper: Prepare search queries (Merge 2: rewrite + translate in one call)
def _prepare_search_queries(query: str, context: str, selected_layer: str, logs: list):
    """Rewrite query for context and translate to English if needed. Returns (ko_query, en_query)."""
    needs_english = selected_layer in ("core_theory", "all")
    has_korean = any('\uac00' <= ch <= '\ud7a3' for ch in query)

    try:
        if needs_english and has_korean:
            # Single call: rewrite + translate
            combined_prompt = (
                "Given the conversation context and current question, do TWO things:\n"
                "1. Rewrite the question into a self-contained Korean search query (resolve pronouns/references from context).\n"
                "2. Translate that Korean query into English.\n\n"
                "If the question is already self-contained, keep it as-is for line 1.\n\n"
                f"Conversation context:\n{context}\n\n"
                f"Current question: {query}\n\n"
                "Output EXACTLY two lines, nothing else:\n"
                "KO: <korean query>\n"
                "EN: <english query>"
            )
            result = _invoke_with_backoff(lambda: llm_light.invoke(combined_prompt))
            lines = result.content.strip().split("\n")
            search_query_ko = query
            search_query_en = None
            for line in lines:
                line = line.strip()
                if line.upper().startswith("KO:"):
                    search_query_ko = line[3:].strip()
                elif line.upper().startswith("EN:"):
                    search_query_en = line[3:].strip()
            if search_query_ko != query:
                logs.append(f"🔄 [재작성] 맥락 반영 검색 쿼리: \"{search_query_ko}\"")
            if search_query_en:
                logs.append(f"🔄 [번역] 영어 문헌 검색용 번역: \"{search_query_en}\"")
            return search_query_ko, search_query_en
        elif needs_english and not has_korean:
            # Query is already English — rewrite for context resolution, use as-is for English search
            rewrite_prompt = (
                "Given the conversation context and current question, "
                "rewrite the user's CURRENT question into a self-contained English search query. "
                "Resolve any pronouns or references using the context. "
                "If the question is already self-contained, return it as-is. "
                "Output ONLY the rewritten query, nothing else.\n\n"
                f"Conversation context:\n{context}\n\n"
                f"Current question: {query}"
            )
            rewritten = _invoke_with_backoff(lambda: llm_light.invoke(rewrite_prompt))
            search_query_en = rewritten.content.strip()
            if search_query_en != query:
                logs.append(f"🔄 [재작성] 맥락 반영 검색 쿼리: \"{search_query_en}\"")
            return search_query_en, search_query_en
        else:
            # Only rewrite in Korean (no translation needed)
            rewrite_prompt = (
                "Given the conversation context and current question, "
                "rewrite the user's CURRENT question into a self-contained Korean search query. "
                "Resolve any pronouns or references using the context. "
                "If the question is already self-contained, return it as-is. "
                "Output ONLY the rewritten query, nothing else.\n\n"
                f"Conversation context:\n{context}\n\n"
                f"Current question: {query}"
            )
            rewritten = _invoke_with_backoff(lambda: llm_light.invoke(rewrite_prompt))
            search_query_ko = rewritten.content.strip()
            if search_query_ko != query:
                logs.append(f"🔄 [재작성] 맥락 반영 검색 쿼리: \"{search_query_ko}\"")
            return search_query_ko, None
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
            logs.append("⚠️ [재작성] Gemini 속도 제한(429) — 원본 쿼리로 대체합니다.")
        return query, query if needs_english else None


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
    svc = get_kg_service()
    if not svc:
        return None
    
    def _do_search(q):
        try:
            return run_kg_async(svc.search(query=q, group_ids=None, num_results=num_results))
        except Exception as e:
            err_msg = str(e).lower()
            if "connection reset" in err_msg or "defunct" in err_msg or "connectionreseterror" in err_msg:
                logger.warning("[KG] 검색 실패 — Neo4j 유휴 연결 리셋 (무시 가능, 자동 복구됨). query=%s", q[:50])
            else:
                logger.warning("[KG] 검색 실패 (query=%s): %s", q[:50], e)
            if any(k in err_msg for k in ("dns", "connection", "timeout", "unavailable", "graphiti")):
                from shared import reset_kg_service
                reset_kg_service()
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


# Node: Retrieve
def retrieve_node(state: AgentState):
    query = state["messages"][-1].content
    context = _build_context(state["messages"])
    sub_queries = state.get("sub_queries")
    logs = []

    # Layer already determined by analyze_intent_node (Merge 1)
    selected_layer = state.get("layer", "all")
    logs.append(f"\n📂 [레이어] '{selected_layer}' 레이어에서 검색합니다.")

    docs = []
    try:
        if sub_queries and len(sub_queries) > 1:
            # Phase 2: Multi-retrieval — run each sub-query independently
            for i, sq in enumerate(sub_queries, 1):
                if i > 1:
                    time.sleep(1)  # Rate-limit guard: avoid flash-lite burst
                logs.append(f"\n🔍 [검색 {i}/{len(sub_queries)}] \"{sq}\"")
                sq_ko, sq_en = _prepare_search_queries(sq, context, selected_layer, logs)
                sq_docs = _retrieve_for_query(sq_ko, sq_en, selected_layer)
                logs.append(f"   → {len(sq_docs)}건 발견")
                for d in sq_docs:
                    logs.append(f"      📄 {_label_doc(d)}")
                docs.extend(sq_docs)
            # Deduplicate across sub-queries
            before = len(docs)
            docs = _deduplicate_docs(docs)
            if before > len(docs):
                logs.append(f"🔗 [병합] {before}건 → {len(docs)}건 (중복 {before - len(docs)}건 제거)")
        else:
            # Single query path (original behavior)
            sq_ko, sq_en = _prepare_search_queries(query, context, selected_layer, logs)
            docs = _retrieve_for_query(sq_ko, sq_en, selected_layer)

        if docs:
            logs.append(f"✅ {len(docs)}개의 혁명 문헌을 발견했습니다:")
            for d in docs:
                logs.append(f"   📄 {_label_doc(d)}")
        else:
            logs.append("⚠️ 영묘 데이터에 관련된 문헌이 없습니다.")

    except Exception as e:
        logs.append(f"⚠️ 검색 중 오류 발생 (무시하고 진행): {e}")

    return {"documents": docs, "logs": logs}

# Node: KG Retrieve (knowledge graph search with grading)
# Note: retrieve_node already handles query rewriting/translation for vectorstore.
# KG search uses original queries directly (no extra LLM calls) to avoid rate-limit risk.
def kg_retrieve_node(state):
    query = state["messages"][-1].content
    sub_queries = state.get("sub_queries")
    logs = []

    # Use original queries directly — no LLM rewrite (retrieve_node already does that for docs)
    search_queries = sub_queries if (sub_queries and len(sub_queries) > 1) else [query]

    # Determine num_results dynamically
    is_complex = (sub_queries and len(sub_queries) > 1) or state.get("intent") in ("strategic", "academic")
    num_results = 15 if is_complex else 8

    merged_kg = None
    total_items = 0

    for i, sq in enumerate(search_queries, 1):
        kg_text = _search_kg(sq, num_results=num_results)
        if kg_text:
            merged_kg = _merge_kg_contexts(merged_kg, kg_text)
            count = _count_kg_items(kg_text)
            total_items += count
            logs.append(f"   🧩 [KG 질의 {i}/{len(search_queries)}] \"{sq}\" → {count}건")
        else:
            logs.append(f"   🧩 [KG 질의 {i}/{len(search_queries)}] \"{sq}\" → 0건")

    # Grade KG results (개선 1: 관련성 필터링)
    if merged_kg:
        logs.insert(0, f"\n🕸️ [지식그래프] 총 {total_items}건의 구조화된 팩트를 확보했습니다.")
        
        # Parse nodes/edges for grading
        kg_lines = merged_kg.splitlines()
        node_lines = []
        edge_lines = []
        in_nodes = False
        in_edges = False
        
        for line in kg_lines:
            if "[Knowledge Graph: Entities]" in line:
                in_nodes, in_edges = True, False
                continue
            elif "[Knowledge Graph: Facts/Relations]" in line:
                in_nodes, in_edges = False, True
                continue
            if line.startswith("- "):
                content = line[2:]
                if in_nodes:
                    node_lines.append(content)
                elif in_edges:
                    edge_lines.append(content)
        
        # Batch grade if we have items to grade
        if node_lines or edge_lines:
            # Rate-limit guard
            time.sleep(0.5)
            
            nodes_text = "\n".join(node_lines) if node_lines else "(no nodes)"
            edges_text = "\n".join(edge_lines) if edge_lines else "(no edges)"
            
            grade_result = invoke_batch_grade_kg({
                "question": query,
                "nodes": nodes_text,
                "edges": edges_text
            })
            
            if grade_result:
                node_scores = grade_result.node_scores[:len(node_lines)] if node_lines else []
                edge_scores = grade_result.edge_scores[:len(edge_lines)] if edge_lines else []
                
                # Pad scores if needed
                while len(node_scores) < len(node_lines):
                    node_scores.append("yes")
                while len(edge_scores) < len(edge_lines):
                    edge_scores.append("yes")
                
                # Filter nodes/edges by scores
                filtered_node_lines = [
                    line for line, score in zip(node_lines, node_scores) if score == "yes"
                ]
                filtered_edge_lines = [
                    line for line, score in zip(edge_lines, edge_scores) if score == "yes"
                ]
                
                # Rebuild kg_context with filtered results (restore "- " prefix stripped during parsing)
                filtered_parts = []
                if filtered_node_lines:
                    filtered_parts.append("[Knowledge Graph: Entities]")
                    filtered_parts.extend(f"- {line}" for line in filtered_node_lines)
                if filtered_edge_lines:
                    filtered_parts.append("[Knowledge Graph: Facts/Relations]")
                    filtered_parts.extend(f"- {line}" for line in filtered_edge_lines)
                
                if filtered_parts:
                    merged_kg = "\n".join(filtered_parts)
                    filtered_count = len(filtered_node_lines) + len(filtered_edge_lines)
                    logs.append(f"   ⚖️ [KG 검열] {total_items}건 → {filtered_count}건 (필터링 {total_items - filtered_count}건)")
                else:
                    merged_kg = None
                    logs.append(f"   ⚖️ [KG 검열] 모든 항목 필터링됨 (0 건)")
            else:
                logs.append("   ⚖️ [KG 검열] 평가 실패, 모든 항목 유지")
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

    # Rate-limit guard: batch_grader is the heaviest flash-lite call.
    # Sleep 1s to let the RPM window recover after upstream flash-lite calls
    # (analyze_intent, prepare_search_queries) that fired in the same turn.
    time.sleep(1)

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

# Helper: Web Search — returns one Document per result with url/title metadata
def _run_web_search(query: str, logs: list) -> list:
    """Invoke Tavily and return results as individual Document objects with url/title metadata."""
    try:
        search_response = web_search_tool.invoke({"query": query})
        results = search_response.get("results", []) if isinstance(search_response, dict) else search_response
        docs = []
        for r in results:
            if isinstance(r, dict) and r.get("content"):
                docs.append(Document(
                    page_content=r["content"],
                    metadata={
                        "source": r.get("url", ""),
                        "title": r.get("title", ""),
                    }
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

def strategize_node(state: AgentState):
    docs = state.get("documents", [])
    context = "\n\n".join([_format_doc(d) for d in docs]) if docs else "No specific documents found."
    question = state["messages"][-1].content

    # Phase 3: Include step results summary if from plan path
    step_results = state.get("step_results", [])
    if step_results:
        research_summary = "\n".join(step_results)
        context = f"=== Research Steps Summary ===\n{research_summary}\n\n=== Source Documents ===\n{context}"

    # Knowledge Graph: prepend structured facts as highest-confidence context
    kg_context = state.get("kg_context")
    if kg_context:
        context = f"=== Knowledge Graph (Structured Intelligence) ===\n{kg_context}\n\n{context}"

    logs = []
    logs.append("\n🧠 [참모] 변증법적 전술을 고안 중...")

    response = _invoke_with_backoff(lambda: strategist_chain.invoke({"context": context, "question": question}))
    strategy_text = extract_text_content(response.content)

    logs.append(f"👉 전술 :\n   {strategy_text}")

    return {"strategy": strategy_text, "logs": logs}

# Node: Generate
def generate_node(state: AgentState):
    docs = state.get("documents", [])
    docs = docs[:12]  # Cap document count to prevent prompt size blowup in plan-and-execute
    context = "\n\n".join([_format_doc(d) for d in docs]) if docs else ""
    strategy = state.get("strategy", None)
    messages = state["messages"]
    # 마지막 사용자 질문
    last_user_query = messages[-1].content
    intent = state.get("intent", "casual") # Router에서 전달된 intent 사용

    logs = []

    # Knowledge Graph structured intelligence
    kg_context = state.get("kg_context")
    kg_section = f"\n[STRUCTURED INTELLIGENCE]\n{kg_context}" if kg_context else ""

    # Intent-specific style guide (minimal, focused on tone)
    style_guide = {
        "academic": "Professional, intellectual, authoritative. Explain concepts thoroughly with precise terminology.",
        "strategic": "Decisive, analytical, practical. Structure as clear phases or numbered steps.",
        "casual": "Natural, friendly, dignified. Brief and conversational (1-3 sentences).",
    }

    # Mission by intent
    mission_guide = {
        "academic": "Provide a detailed, objective explanation grounded in the provided context.",
        "strategic": "Deliver a concrete, actionable strategy. Synthesize the analysis—do not merely restate sources.",
        "casual": "Respond with wit and revolutionary charm.",
    }

    current_dt = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    system_prompt = f"""{CORE_IDENTITY}
[CURRENT TIME] {current_dt}

[MISSION] {mission_guide.get(intent, mission_guide['casual'])}
[INTERNAL ANALYSIS] {strategy if strategy else "No analysis available."}
[SOURCE MATERIAL] {context if context else "(No sources found.)"}{kg_section}
[STYLE] {style_guide.get(intent, style_guide['casual'])}

Respond in the same language as the user's question."""
    system_prompt += f"\n\n[QUESTION] {last_user_query}"

    # Escape curly braces for ChatPromptTemplate
    safe_system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", safe_system_prompt),
        ("placeholder", "{messages}")
    ])

    chain = prompt | llm
    response = _invoke_with_backoff(lambda: chain.invoke({"messages": messages}))
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
        strategy = state.get("strategy", None)
        turn_start = state.get("logs_turn_start", 0)
        processing_logs = state.get("logs", [])[turn_start:]

        datasource = state.get("datasource", "generate")
        route = datasource if datasource else ("casual" if not docs and not strategy else "vectorstore")

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
            "strategy": strategy,
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

system_planner = """You are a research planner for Cyber-Lenin. Create a 2-4 step plan to gather material for dialectical analysis.

Plan steps to collect what dialectical reasoning needs:
- **Concrete conditions**: material facts, class forces, historical context (use "web_search" for current events, "retrieve" for theory)
- **Internal contradictions**: opposing forces within the phenomenon
- **Systemic connections**: broader economic, political, ideological relations

Each step: ONE focused angle, one tool, one query.
- "retrieve" = internal Marxist-Leninist knowledge base (classical texts, analysis)
- "web_search" = current events, recent data, real-world conditions
- Query language: Korean for current/domestic topics, English for classical theory

Build progressively: concrete facts first, then structural forces, then theoretical framework.

Respond with ONLY a JSON object:
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

    result = _invoke_structured(
        planner_prompt | llm, {"question": question, "context": context},
        ResearchPlan, _PLAN_DEFAULT, retry_llm=llm
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
workflow.add_node("strategize", strategize_node)
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
    "no_need_to_search_web": "strategize",
})
workflow.add_edge("web_search", "strategize")
workflow.add_edge("strategize", "generate")
workflow.add_edge("generate", "log_conversation")
workflow.add_edge("log_conversation", END)
# Phase 3: planner → step_executor → [continue → step_executor | done → strategize]
# (KG search runs inside step_executor per step, not as a separate node)
workflow.add_edge("planner", "step_executor")
workflow.add_conditional_edges("step_executor", plan_progress, {
    "continue": "step_executor",
    "done": "strategize",
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
