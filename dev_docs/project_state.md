# Project State Report — 2026-03-01

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
│   ├── __main__.py           # python -m graph_memory "query" 엔트리포인트
│   ├── cli.py                # KG 질의 CLI (run_query, main)
│   ├── config.py             # Edge type mapping, excluded types, episode source map, extraction instructions
│   ├── entities.py           # 7 entity types: Person, Organization, Location, Asset, Incident, Policy, Campaign (v2)
│   ├── edges.py              # 10 edge types: +PolicyEffect, +Participation (v2)
│   ├── graphiti_patches.py   # [NEW] Graphiti 런타임 몽키패치 (DateTime 직렬화, 엣지 프롬프트)
│   ├── kr_news_fetcher.py    # 한국 국내 뉴스 수집 + 인물 프로파일 보강 파이프라인
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

### 2026-03-01 — temp_dev 코드 → graph_memory 모듈 이전
- **`graph_memory/kr_news_fetcher.py`** (new): `temp_dev/ingest_kr_news.py` 코드를 패키지 내부로 이전. 상대 임포트(`from .service`) 사용, `sys.path.insert` 제거.
- **`graph_memory/cli.py`** (new): `temp_dev/query_kg.py` 코드를 패키지 내부로 이전. `run_query()` + `main()` argparse 엔트리포인트.
- **`graph_memory/__main__.py`** (new): `python -m graph_memory "query"` 실행 지원.
- **`graph_memory/__init__.py`**: `fetch_kr_news`, `extract_persons_from_articles`, `fetch_person_profile`, `run_full_pipeline`, `cli_main` export 추가.
- **`temp_dev/ingest_kr_news.py`**: thin wrapper로 교체 — `graph_memory.kr_news_fetcher`에서 re-export.
- **`temp_dev/query_kg.py`**: thin wrapper로 교체 — `graph_memory.cli.main()` 호출.
- **`temp_dev/test_kr_news_ingest.py`**: import를 `graph_memory.kr_news_fetcher`로 변경. 테스트 통과 확인.

### 2026-03-01 — 한국 국내 뉴스 + 인물 프로파일 지식그래프 수집 (TDD)
- **TDD 방식 개발**: `temp_dev/test_kr_news_ingest.py` (6 테스트) 먼저 작성 → 구현 → 전체 통과
- **`temp_dev/ingest_kr_news.py`** (new): 한국 국내 뉴스 수집 + 인물 추출 + 프로파일 보강 파이프라인
  - `fetch_kr_news()`: Tavily 뉴스 검색 → 기사 리스트 반환
  - `extract_persons_from_articles()`: LLM(Gemini)으로 기사에서 한국 인물명 추출 (name_ko, name_en, role)
  - `fetch_person_profile()`: Tavily 일반 검색 + LLM 종합으로 인물 프로파일 구조화
  - `ingest_news_to_kg()`: 뉴스 기사를 GraphMemoryService 에피소드로 수집
  - `ingest_profile_to_kg()`: 인물 프로파일을 KG 에피소드로 수집
  - `run_full_pipeline()`: 전체 파이프라인 오케스트레이션 (standalone 실행 가능)
- **수집 결과**: 뉴스 3건 + 이재명(Lee Jae-myung) 프로파일 KG 수집 완료
  - 노드 15개: South Korea, Democratic Party of Korea, Lee Jae-myung, Seongnam, Gyeonggi Province 등
  - 엣지 15개: 더불어민주당 대표, 성남시장, 경기도지사, 국회의원, 암살 미수 피해 등
  - `group_id="korea_domestic"` — 기존 국제 뉴스(`geopolitics_*`)와 분리
- **KG 현황**: 에피소드 14건 (기존 10 + 뉴스 3 + 프로파일 1), 한국 인물/조직 엔티티 추가됨

### 2026-02-28 — diary_writer.py 뉴스 → 지식그래프 자동 수집
- **`diary_writer.py` `_search_news()`**: 반환 타입 `str` → `tuple[str, list[dict]]` — 기존 요약 문자열 + 원문 `[{"title", "url", "content"}]` 함께 반환. URL 중복 제거 포함.
- **`diary_writer.py` `_ingest_news_to_graph()`** (new): `GraphMemoryService.ingest_episode()`로 뉴스 원문을 KG에 수집. `group_id="diary_news"`, `source_type="osint_news"`, `max_body_chars=1500`. 개별 기사 실패 시 로그 후 계속 (best-effort). 전체 try/except로 KG 장애가 일기 파이프라인에 전파되지 않음.
- **`diary_writer.py` `write_diary()`**: `_save_diary()` 성공 후에만 `_ingest_news_to_graph()` 호출. 일기 작성할 때마다 검색한 뉴스가 자동으로 지식그래프에 축적됨.

### 2026-02-28 — 엔티티/관계 정규화 + 런타임 패치 + 성능 최적화
- **`graph_memory/config.py`**: `CUSTOM_EXTRACTION_INSTRUCTIONS` 상수 추가 — 엔티티 영어 강제, 관계 타입 10종 제한. `NEWS_PREPROCESS_PROMPT_TEMPLATE` 영어 출력으로 변경.
- **`graph_memory/service.py`**: `add_episode()`에 `custom_extraction_instructions` 전달. `SEMAPHORE_LIMIT` 기본 20으로 설정.
- **`graph_memory/graphiti_patches.py`** (new): Graphiti 런타임 몽키패치 — .venv 수정 없이 Render 등 원격 배포 환경에서도 동작
  - `to_prompt_json`: Neo4j DateTime JSON 직렬화 (`_Neo4jDateTimeEncoder`)
  - `extract_edges.Edge`: relation_type SCREAMING_SNAKE_CASE → PascalCase + FACT_TYPES 우선
  - `extract_edges.edge()`: RELATION TYPE RULES 프롬프트 교체
- **`graph_memory/news_fetcher.py`**: `DEFAULT_DELAY_BETWEEN` 30→5초
- **기존 데이터 재수집**: 전체 삭제 후 10건 재수집. 한국어 엔티티 0개, 관계 타입 85%+ 정규화
- **`requirements.txt`**: 프로덕션 의존성만 유지 (크롤러 전용 4개 패키지 제거)

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
