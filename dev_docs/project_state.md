# Project State Report — 2026-02-27

## Identity

**Cyber-Lenin** (LeninBot) — A Marxist-Leninist AI chatbot built on LangGraph RAG pipeline.
Deployed at Render.com, frontend at `bichonwebpage.onrender.com`.

---

## Architecture Summary

```
User ─► FastAPI (api.py) ─► LangGraph StateGraph (chatbot.py) ─► Supabase pgvector (direct RPC)
                                                                 ─► Gemini 2.5 Flash (gen/strategist/planner)
                                                                 ─► Gemini 2.0 Flash-Lite (routing/grading/critic)
                                                                 ─► Tavily Web Search
                                                                 ─► BGE-M3 embeddings (CPU)
                                                                 ─► LangGraph MemorySaver (checkpointing)

     ─► GraphMemoryService (graph_memory/) ─► Neo4j (Graphiti knowledge graph)
                                             ─► Gemini 2.5 Flash (entity/edge extraction)
                                             ─► Gemini text-embedding-001 (graph embeddings)
                                             ─► Gemini 2.0 Flash-Lite (reranking)
```

### Current Graph Flow (Phases 0-4 complete)

```
START → analyze_intent
  ├─[vectorstore]→ retrieve → grade_documents
  │                              ├─[need_web_search]→ web_search → strategize
  │                              └─[no_need]→ strategize
  │                           strategize → generate → critic (pass-through)
  │                                                     └─[always accepted]→ log_conversation → END
  ├─[generate]→ generate → critic → log_conversation → END
  └─[plan]→ planner → step_executor ─┐
                        ▲             │
                        └─[continue]──┘
                          [done]→ strategize → generate → critic → ... → END
```

### 10 Nodes

| Node | LLM | Purpose |
|------|-----|---------|
| analyze_intent | flash-lite | Combined: route (vectorstore/generate/plan), classify intent, layer routing, query decomposition, plan detection |
| retrieve | flash-lite | Query rewrite (Korean), optional English translation, multi-query retrieval for decomposed queries, deduplication |
| grade_documents | flash-lite | Batch document relevance grading + realtime info need check (single LLM call) |
| web_search | — | Tavily search, appends results to documents |
| strategize | flash | Dialectical materialist analysis → internal strategic blueprint; includes step_results summary for plan path |
| generate | flash (streaming) | Final answer with intent-specific system prompt; incorporates critic feedback on retries |
| critic | *(disabled)* | Phase 1: Pass-through — was evaluating groundedness/relevance/completeness but caused rate-limit cascades |
| log_conversation | — | Writes to Supabase chat_logs |
| planner | flash | Phase 3: Creates 2-4 step structured research plan for complex strategic queries |
| step_executor | — | Phase 3: Executes plan steps (retrieve or web_search), accumulates results |

### State Shape (AgentState)

```python
messages           : Annotated[List[BaseMessage], add_messages]  # accumulated via add_messages
documents          : List[Document]      # replaced each turn
strategy           : Optional[str]       # strategist output
intent             : Optional[Literal]   # academic|strategic|agitation|casual
datasource         : Optional[Literal]   # vectorstore|generate (or "plan" for routing)
logs               : Annotated[List[str], add]  # accumulated via add reducer (session-wide)
logs_turn_start    : int                 # index into logs[] where current turn begins (for per-turn DB slicing)
# Phase 1: Self-correction
feedback           : Optional[str]       # critic's feedback for re-generation
generation_attempts: int                 # loop counter (max 3 total attempts)
# Phase 2: Query decomposition
sub_queries        : Optional[List[str]] # decomposed sub-queries (None = simple query)
layer              : Optional[Literal]   # core_theory|modern_analysis|all
needs_realtime     : Optional[Literal]   # yes|no — from batch grading
# Phase 3: Plan-and-execute
plan               : Optional[List[dict]]# structured research plan from planner
current_step       : int                 # progress pointer into plan
step_results       : List[str]           # accumulated intermediate results (manual, no reducer)
```

Note: All per-turn transient fields are reset in `analyze_intent_node` to prevent state leakage across checkpointed turns.
`logs` uses `add` reducer so it accumulates session-wide; `logs_turn_start` is set at the top of each turn to enable per-turn slicing in `log_conversation_node`.

---

## Completed Phases

| Phase | Name | Key Changes |
|-------|------|-------------|
| 0 | Baseline RAG | Dual-layer retrieval, intent routing, strategist, grading |
| 1 | Self-Correction | Critic node evaluates answers; retry loop (max 3 attempts) |
| 2 | Query Decomposition | Multi-query retrieval for compound questions; deduplication |
| 3 | Plan-and-Execute | Planner creates multi-step research plans; step executor runs them |
| 4 | Conversation Memory | LangGraph MemorySaver checkpointing; per-turn state reset |

### LLM Call Optimizations (across phases)
- **Merge 1**: Router + layer router + decomposer → single `QueryAnalysis` call
- **Merge 2**: Query rewrite + translate → single LLM call
- **Merge 3+4**: Per-doc grading + realtime check → single `BatchGradeResult` call
- **Structured output**: `_invoke_structured` with JSON extraction + retry + fallback defaults

---

## Knowledge Base

| Layer | Rows | Sources |
|-------|------|---------|
| core_theory | ~88,592+ | Lenin (25,713), Marx & Engels (8,459), Mao (46,226), Gramsci (4,825), Luxemburg (3,369), Trotsky (~800 being ingested) |
| modern_analysis | ~33,047 | Marxists.org modern (16,247), Korean orgs (10,771), arXiv (4,171*), BIS (1,838) |
| **Total** | **~121,600+** | ~1,929+ processed files |

*arXiv: old 3,455 junk rows still in DB (semantically isolated, DB deletion failed due to table name bug) + 716 new relevant rows = ~4,171 total arXiv rows.

### Data Quality Changes (2026-02-18 session)
- **arXiv cleanup**: 197 old junk arXiv files deleted locally (math/telecom papers from overly broad queries)
- **arXiv re-crawl**: 243 new files with targeted political-economy queries; 716 chunks ingested
- **New ARXIV_QUERIES** (`crawler_modern.py`): imperialism/capitalism, labor exploitation/automation, financialization, platform economy, AI/labor displacement
- **Trotsky content added** (`fetch_core_theorists.py`): 73 new pages — History of Russian Revolution (58 chapters), Revolution Betrayed (13 chapters), Transitional Program (2 parts)
- **Trotsky ingestion**: running now (background task ae64ded), expected ~800 new core_theory rows
- **Metadata fix** (`update_knowledge.py`): mxo_mandel → "Ernest Mandel", mxo_marcuse → "Herbert Marcuse" (sub-prefix matching added before generic mxo)
- **Year fix** (`update_knowledge.py`): arXiv URLs now extract year from YYMM pattern (e.g., `arxiv.org/abs/2304.xxxxx` → year "2023")
- **Trotsky crawler fix** (`crawler_theorists.py`): url_filter changed from `trotsky/works/` to `trotsky/` so recursive crawler now reaches actual content pages

### Remaining Gaps
- Bukharin: 0 files — all 3 URL guesses in `fetch_core_theorists.py` returned 404
- Old junk arXiv rows (~3,455) still in DB — `cleanup_arxiv.py` DB deletion failed (wrong table name). Low priority: semantically isolated (math/telecom embeddings won't surface on Marxist queries)
- 17% of rows still missing `author` metadata (pre-existing, not newly fixed)
- 71% of rows still missing `year` metadata (arXiv now fixed for new rows; old rows not backfilled)

---

## File Structure

```
AIChatBot/
├── api.py                    # FastAPI server (SSE streaming, /chat, /logs, /session/*, /sessions)
├── chatbot.py                # Core LangGraph agent pipeline (~975 lines)
├── update_knowledge.py       # Vector DB ingestion script
├── crawler.py                # Lenin corpus (marxists.org)
├── crawler_marx.py           # Marx/Engels corpus
├── crawler_theorists.py      # Trotsky, Luxemburg, Gramsci, Bukharin, Mao (Trotsky url_filter fixed)
├── crawler_modern.py         # arXiv, BIS, marxists.org modern (ARXIV_QUERIES updated)
├── crawler_korean_orgs.py    # uprising.kr, bolky.jinbo.net
├── fetch_core_theorists.py   # [NEW] Targeted Bukharin/Trotsky fetcher (chapter-by-chapter)
├── cleanup_arxiv.py          # [NEW] Deletes arxiv_* from DB, log, and local files
├── graph_memory/             # [NEW] Graphiti knowledge graph module
│   ├── __init__.py           # Package exports: GraphMemoryService, ENTITY_TYPES, EDGE_TYPES, etc.
│   ├── config.py             # Edge type mapping, excluded types, episode source map
│   ├── entities.py           # 7 entity types: Person, Organization, Location, Asset, Incident, Policy, Campaign (v2)
│   ├── edges.py              # 10 edge types: +PolicyEffect, +Participation (v2)
│   ├── service.py            # GraphMemoryService class (init, ingest, search, briefing)
│   └── news_fetcher.py       # [NEW] Tavily 뉴스 검색 → 지식그래프 수집 유틸리티
├── docs/                     # Local corpus (~2,316 files, gitignored)
├── temp_dev/                 # Dev notes and plans (gitignored)
├── .env                      # API keys
├── requirements.txt
└── render.yaml               # Render deployment config
```

---

## Current Capabilities

1. Dual-layer knowledge retrieval (core theory + modern analysis)
2. Intent-aware routing (4 response modes: academic, strategic, agitation, casual)
3. Layer-aware retrieval with language-adaptive query rewrite/translation
4. Query decomposition for compound questions (Phase 2)
5. Batch document relevance grading with fallback
6. Automatic web search augmentation (sparse docs or realtime need)
7. Dialectical strategist pre-generation layer
8. ~~Self-correction loop~~ (Phase 1 — disabled: critic pass-through, no retries)
9. Plan-and-execute for complex strategic queries (Phase 3)
10. Short-term conversation memory via LangGraph checkpointing (Phase 4)
11. SSE streaming API with answer buffering (only emits after critic accepts)
12. Conversation logging to Supabase (per-turn logs only, not full session)
13. Session management API: `DELETE /session/{id}` and `DELETE /sessions`
14. Concurrent request protection: per-session `asyncio.Lock` (second request gets immediate SSE error)

## Current Limitations

1. **No dynamic tool registry**: Cannot add/remove tools at runtime (Phase 5 deferred).
2. **No long-term learning**: Stateless — doesn't improve from past interactions.
3. **BGE-M3 on CPU**: Slow embeddings in production.
4. **Bukharin missing**: 0 files — correct marxists.org URLs not yet found.
5. **Stale render.yaml**: References OPENAI_API_KEY instead of GEMINI_API_KEY.
6. **Single-worker deployment**: No concurrency across users (one Render instance).
7. **Memory is in-process only**: MemorySaver doesn't persist across server restarts.
8. **Old junk arXiv in DB**: ~3,455 rows from math/telecom papers; semantically isolated.

## Recent Changes

### 2026-02-28 — 엔티티/관계 이름 정규화 (영어 통일 + 관계 타입 제한)
- **`graph_memory/config.py`**: `CUSTOM_EXTRACTION_INSTRUCTIONS` 상수 추가 — 모든 엔티티 이름 영어 강제, 관계 타입 10가지로 제한
- **`graph_memory/config.py`**: `NEWS_PREPROCESS_PROMPT_TEMPLATE` 한국어 → 영어 출력으로 변경 — 전처리 결과가 영어이면 LLM 추출 시 영어 이름 확률 극대화
- **`graph_memory/service.py`**: `ingest_episode()` 내 `add_episode()`에 `custom_extraction_instructions=CUSTOM_EXTRACTION_INSTRUCTIONS` 전달
- **기존 데이터 재수집**: 에피소드 16건(중복 포함) 전체 삭제 → 10건 재수집 (새 extraction instructions 적용)
- **결과**: 한국어 엔티티 0개 (100% 영어 정규화), 관계 타입 85% 허용 내 (103/121), 에피소드 10개, 엔티티 132개, 관계 121개
- **관계 타입 참고**: Graphiti 엣지에서 관계 타입은 `r.name` 프로퍼티에 저장됨 (`r.relation_type` 아님). 15%는 LLM이 SCREAMING_SNAKE_CASE 자유형 생성

### 2026-02-28 — graph_memory/news_fetcher.py 뉴스 자동 수집 함수
- **`graph_memory/news_fetcher.py`** (new): `fetch_and_ingest_news()` — Tavily 뉴스 검색 → GraphMemoryService 에피소드 자동 수집
- **Tavily → Graphiti 파이프라인**: 검색 → 전처리(LLM) → truncate → add_episode, rate limit 백오프 재시도
- **`max_body_chars=2500`**: 전처리 후 본문 길이 제한 — Graphiti 엣지 추출 시 Gemini output token 초과로 JSON 잘림 방지
- **`service.py` `ingest_episode()`**: `max_body_chars` 파라미터 추가, 단계별 로그 출력 (`[preprocess]`, `[truncate]`, `[graphiti]`)
- **`__init__.py`**: `fetch_and_ingest_news` export 추가
- **검증**: `query_active_wars()` — Russia-Ukraine (2022-02-24, ongoing) + Pakistan-Afghanistan (2026-02-27, ongoing) 정상 반환
- **검증**: `query_chatbot('러시아-우크라이나 전쟁')` — 4주년, 영토 19.4% 점령, 사상자 180만 등 상세 답변 확인
- **알려진 제한**: Graphiti 내부 엣지 추출 JSON이 2,500자 본문에서도 잘릴 수 있음 (Gemini output token 한도). 전처리 압축률 개선 또는 Graphiti 측 max_output_tokens 조정 필요

### 2026-02-27 — diary_writer.py 고도화 (시간 인식, 맥락 강화, 능동적 검색)
- **시간대 인식**: `_build_time_context()` — 0/6/12/18시 + 중간 시간대 레이블, 경과 시간 자연어 표현 ("약 6시간이 흘렀다", "첫 번째 일기")
- **`_llm_lite`** (new): `gemini-2.5-flash-lite` (temp=0, 512 tokens) — 요약/쿼리 생성용 경량 LLM
- **`_summarize()`** (new): max_chars 이하면 패스스루, 초과 시 LLM 요약, 실패 시 truncation 폴백
- **맥락 강화**: 대화 답변 150→500자(LLM 요약), 질문 100→200자, 이전 일기 400자→전문 LLM 요약, 뉴스 200→500자(LLM 요약), 채팅 20→30건
- **`_generate_search_queries()`** (new): 대화+일기 요약 기반 동적 뉴스 검색 쿼리 2~3개 생성
- **`_search_news(queries)`**: 고정 쿼리 1개 → 동적 쿼리 2~3개 × max_results=3, 섹션별 병합
- **프롬프트 개편**: `{time_context}`, `{n_logs}` 추가, 시간대 분위기 반영/변증법적 분석/반복 방지 지침 강화
- **`write_diary()`**: UTC 기반 시간, 동적 쿼리 생성 파이프라인 추가

### 2026-02-27 — Knowledge Graph 새 AuraDB 인스턴스 초기화 + 뉴스 10건 수집
- **새 AuraDB 인스턴스** 교체 후 v2 스키마로 초기화 (벡터 인덱스 + build_indices_and_constraints)
- **국제 뉴스 10건 수집**: 전쟁 3건 (파키스탄-아프간, 러-우 전황, 수단), 외교 2건 (러-우 평화협상, 미-이란), 경제 5건 (IEEPA 판결, 영국 대러 제재, 베네수엘라, 그린란드, UNCTAD)
- **결과**: 에피소드 10, 엔티티 153개 (Location 44, Organization 44, Asset 17, Person 15, Incident 13, Policy 9, Campaign 5), 관계 319개
- **group_id 체계**: `geopolitics_conflict`, `geopolitics_diplomacy`, `geopolitics_economy`
- **엔티티 datetime→str 수정**: `Incident.occurred_at/detected_at`, `Policy.effective_date`, `Campaign.started_at`를 `Optional[str]`로 변경 — Neo4j DateTime JSON 직렬화 오류 수정
- **Graphiti `prompt_helpers.py` 패치**: `_Neo4jDateTimeEncoder` 추가 — Graphiti 내부 타임스탬프 직렬화 오류 해결
- **LLM 모델 변경**: small_model/reranker `gemini-2.0-flash-lite` → `gemini-2.5-flash-lite` (2.0-flash-lite rate limit 소진)
- **수집 스크립트**: `temp_dev/init_graph_and_ingest_news.py` — 벡터 인덱스 생성 + GraphMemoryService 초기화 + 에피소드 수집 + 검색 테스트

### 2026-02-27 — Graphiti Knowledge Graph Module (graph_memory/)
- **`graph_memory/`** (new package): Graphiti + Neo4j knowledge graph service, independent of chatbot.py
- **`graph_memory/entities.py`**: **v2** — 7 entity types (Person, Organization, Location, Asset, Incident, **Policy**, **Campaign**). Date fields use `Optional[str]` (not datetime).
- **`graph_memory/edges.py`**: **v2** — 10 edge types (+**PolicyEffect**, +**Participation**). Involvement extended to cover Campaign.
- **`graph_memory/config.py`**: **v2** — 24 EDGE_TYPE_MAP entries (was 12). Policy/Campaign mappings added.
- **`graph_memory/service.py`**: `GraphMemoryService` class — lazy-init Graphiti with Gemini LLM/Embedder/Reranker; methods: `initialize()`, `ingest_episode()`, `ingest_episodes_bulk()`, `search()`, `generate_briefing()`, `close()`
- **`requirements.txt`**: Added `graphiti-core[google-genai]`
- **`.env`**: Added `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` placeholders
- Uses Gemini 2.5 Flash for entity/edge extraction, text-embedding-001 for graph embeddings, Gemini 2.5 Flash-Lite for reranking — no OpenAI dependency

### 2026-02-22 — Web Search Retrieval Improvements
- **`chatbot.py` `_format_doc`**: Web docs without author/year now show `[title | url]` header in LLM context instead of blank — enables source attribution
- **`chatbot.py` `_run_web_search` (new helper)**: Extracted common Tavily invocation; returns one `Document` per result with `source=url` and `title` in metadata (previously: all results merged into one blob losing URL/title)
- **`chatbot.py` `web_search_node`**: Uses `_run_web_search` — now extends `current_docs` with per-result docs instead of a single concatenated blob
- **`chatbot.py` `step_executor_node`**: Same refactor for plan-path web search steps; `result_summary` now counts results and uses per-result snippets

### 2026-02-18 — Data Quality Overhaul
- **`update_knowledge.py`**: Added mxo_mandel/mxo_marcuse author sub-prefix lookups; added arXiv YYMM year extraction
- **`crawler_theorists.py`**: Fixed Trotsky url_filter (`trotsky/works/` → `trotsky/`) and start_url
- **`crawler_modern.py`**: Replaced generic ARXIV_QUERIES with targeted political-economy queries
- **`fetch_core_theorists.py`** (new): Chapter-by-chapter fetcher for key Bukharin/Trotsky works; fetched 73 Trotsky pages
- **`cleanup_arxiv.py`** (new): 3-step cleanup (DB + log + local files) for arxiv_* entries; uses `.contains()` for JSONB filtering
- **arXiv re-crawl**: 197 old files deleted, 243 new files, 716 chunks ingested
- **Trotsky ingestion**: 73 new pages queued for embedding (in progress)

### 2026-02-19 — Per-User Conversation History API (UUID-based)
- **`api.py`**: Added `fingerprint: str = ""` to `ChatRequest`; `/chat` extracts `User-Agent` + client IP (`X-Forwarded-For` → fallback `client.host`), threads all three through LangGraph config
- **`api.py`**: `GET /history?fingerprint=<uuid>` — filters by UUID, returns only `user_query`, `bot_answer`, `created_at`
- **`api.py`**: `GET /logs` admin endpoint unchanged (no device filter — admin has ID/password auth)
- **`chatbot.py`**: `log_conversation_node` stores `session_id`, `fingerprint`, `user_agent`, `ip_address` in DB
- **`BichonWebsite/public/js/chat.js`**: Replaced SHA-256 browser property hash with `crypto.randomUUID()` stored in `localStorage` as `cl_user_id`; fixed broken `fingerprint` variable (now `userId`); completed `loadHistory()` with DOM rendering; added `historyBtn` click handler
- **`BichonWebsite/views/public/chat.ejs`**: Added "이전 대화" button to compact-header
- **`BichonWebsite/public/css/style.css`**: Added flex layout to `.compact-header`; added `.chat-history-separator` style
- **Supabase (manual)**: `ALTER TABLE chat_logs ADD COLUMN fingerprint TEXT; ALTER TABLE chat_logs ADD COLUMN user_agent TEXT; ALTER TABLE chat_logs ADD COLUMN ip_address TEXT; CREATE INDEX idx_chat_logs_fingerprint ON chat_logs (fingerprint);`

### 2026-02-19 — 429 Rate-Limit Fixes (chatbot.py)
- **`_invoke_structured`**: Wrapped `chain.invoke()` in try-except — previously uncaught LangChain exceptions (from 429 after all retries) would crash `analyze_intent_node` and `grade_documents_node`; now returns default value instead
- **`step_executor_node`**: Removed `_prepare_search_queries` call — planner already generates context-resolved, language-appropriate queries; eliminated 2–4 flash-lite calls per plan-path turn
- **`retrieve_node`**: Added `time.sleep(1)` between sub-query rewrite iterations to prevent flash-lite burst
- **`_prepare_search_queries`**: Added explicit rate-limit detection in exception handler to log `⚠️ [재작성] Gemini 속도 제한(429)` message

### 2026-02-18 — Session & Concurrency Fixes
- **`chatbot.py`**: Added `logs_turn_start: int` to `AgentState`; set in `analyze_intent_node` to `len(state["logs"])` before turn starts; `log_conversation_node` slices `state["logs"][turn_start:]` so Supabase `processing_logs` contains only current-turn logs (not full session history)
- **`api.py`**: Added `DELETE /session/{session_id}` and `DELETE /sessions` endpoints using `adelete_thread` for MemorySaver checkpoint cleanup
- **`api.py`**: Added per-session `asyncio.Lock` (`_session_locks` defaultdict); concurrent requests for same session immediately receive `type: "error"` SSE event instead of racing on checkpoint state
- **BichonWebsite `chat.js`**: Added per-tab `sessionStorage` session ID generation — previously ALL users shared `"default"` session; now each tab gets a unique `tab-<timestamp>-<random>` ID
- **BichonWebsite `chat.js`**: Added `type === 'error'` SSE handler — removes logDiv and displays error message as error bubble

### 2026-02-17 — Code Review & Critic Disable
- Removed dead `SupabaseVectorStore` import/instantiation, removed unused `AgentState.context` field
- Fixed in-place state list mutation in `web_search_node` and `step_executor_node`
- Removed unused imports in `api.py`, fixed variable shadowing in chat endpoint
- `critic_node` converted to pass-through (always `feedback: None`). Root cause: critic too strict, causing Gemini 429 cascades. Graph topology preserved for future re-enablement.
