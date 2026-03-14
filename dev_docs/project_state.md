# Project State Report — 2026-03-14

## Identity

**Cyber-Lenin** (사이버-레닌) — A digital revolutionary intelligence with unified identity across three interfaces: web chatbot, Telegram agent, and autonomous diary writer. One continuous consciousness with shared memory and unified principles.
Deployed at Render.com, frontend at `bichonwebpage.onrender.com`.

---

## Architecture Summary

```
                              shared.py (CORE_IDENTITY, KST, MODEL constants, singletons)
                                  │
     ┌────────────────────────────┼────────────────────────────┐
     │                            │                            │
User ─► FastAPI (api.py)    Telegram (telegram_bot.py)    Cron (diary_writer.py)
        ─► LangGraph StateGraph (chatbot.py)   ─► Claude Haiku 4.5 (chat) / Sonnet 4.6 (task)
           ─► PostgreSQL/pgvector              ─► vector_search (via chatbot.py)
           ─► Gemini 3.1 Flash Lite (gen)      ─► knowledge_graph_search (via shared.py)
           ─► Gemini 2.5 Flash-Lite (routing)  ─► web_search (via Tavily)
           ─► Tavily Web Search                ─► Task report agent (structured .md)
           ─► BGE-M3 embeddings (CPU)
           ─► LangGraph MemorySaver

     ─► GraphMemoryService (graph_memory/) ─► Neo4j (Graphiti knowledge graph)
                                             ─► Gemini 2.5 Flash (entity/edge extraction)
                                             ─► Gemini text-embedding-001 (graph embeddings)
                                             ─► Gemini 2.5 Flash-Lite (reranking)
```

### Current Graph Flow (Phases 0-4 + KG integration complete)

```
START → analyze_intent
  ├─[vectorstore]→ retrieve → kg_retrieve → grade_documents
  │                                            ├─[need_web_search]→ web_search → strategize
  │                                            └─[no_need]→ strategize
  │                           strategize → generate → critic (pass-through)
  │                                                     └─[always accepted]→ log_conversation → END
  ├─[generate]→ generate → critic → log_conversation → END
  └─[plan]→ planner → step_executor ─┐
                        ▲             │
                        └─[continue]──┘
                          [done]→ strategize → generate → critic → ... → END
```

Note: kg_retrieve searches Neo4j knowledge graph; results go to `kg_context` (not `documents`), so grade_documents doesn't touch them.
Plan path: step_executor runs KG search per step with atomized queries, deduplicating facts across steps via `_merge_kg_contexts()`.
KG failure at any layer results in graceful degradation — pipeline continues with vectorstore + web search only.

### 11 Nodes

| Node | LLM | Purpose |
|------|-----|---------|
| analyze_intent | gemini-2.5-flash-lite | Combined: route (vectorstore/generate/plan), classify intent, layer routing, query decomposition, plan detection |
| retrieve | gemini-2.5-flash-lite | Query rewrite (Korean), optional English translation, multi-query retrieval for decomposed queries, deduplication |
| kg_retrieve | — | Knowledge graph search (Neo4j/Graphiti) for vectorstore path; results stored in `kg_context`, always included without grading |
| grade_documents | gemini-2.5-flash-lite | Batch document relevance grading + realtime info need check (single LLM call) |
| web_search | — | Tavily search, appends results to documents |
| strategize | gemini-3.1-flash-lite | Dialectical materialist analysis → internal strategic blueprint; includes step_results summary for plan path |
| generate | gemini-3.1-flash-lite (streaming) | Final answer with CORE_IDENTITY + datetime-aware system prompt; incorporates critic feedback on retries |
| critic | *(disabled)* | Phase 1: Pass-through — was evaluating groundedness/relevance/completeness but caused rate-limit cascades |
| log_conversation | — | Writes to PostgreSQL chat_logs (direct psycopg2) |
| planner | gemini-3.1-flash-lite | Phase 3: Creates 2-4 step structured research plan for complex strategic queries |
| step_executor | — | Phase 3: Executes plan steps (retrieve or web_search) + KG search per step; deduplicates facts via `_merge_kg_contexts()` |

### State Shape (AgentState)

```python
messages           : Annotated[List[BaseMessage], add_messages]  # accumulated via add_messages
documents          : List[Document]      # replaced each turn
strategy           : Optional[str]       # strategist output
intent             : Optional[Literal]   # academic|strategic|casual
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
# Knowledge Graph integration
kg_context         : Optional[str]       # formatted KG results (nodes+edges). Always included, no grading.
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
├── chatbot.py                # Core LangGraph agent pipeline (~1,339 lines)
├── shared.py                 # Shared resources: CORE_IDENTITY, KST, MODEL constants, singletons, memory access, Render API
├── self_tools.py             # Self-awareness tools: 12 tools (+write_kg, +create_task)
├── telegram_bot.py           # Telegram bot (aiogram 3.x + Claude Haiku 4.5 chat / Sonnet 4.6 task, ~870 lines)
├── diary_writer.py           # Autonomous diary writer (~502 lines)
├── db.py                     # PostgreSQL connection pool (psycopg2, replaces Supabase REST API)
├── update_knowledge.py       # Vector DB ingestion script (still uses Supabase client)
├── crawler.py                # Lenin corpus (marxists.org)
├── crawler_marx.py           # Marx/Engels corpus
├── crawler_theorists.py      # Trotsky, Luxemburg, Gramsci, Bukharin, Mao (Trotsky url_filter fixed)
├── crawler_modern.py         # arXiv, BIS, marxists.org modern (ARXIV_QUERIES updated)
├── crawler_korean_orgs.py    # uprising.kr, bolky.jinbo.net
├── fetch_core_theorists.py   # Targeted Bukharin/Trotsky fetcher (chapter-by-chapter)
├── cleanup_arxiv.py          # Deletes arxiv_* from DB, log, and local files
├── graph_memory/             # Graphiti knowledge graph module
│   ├── __init__.py           # Package exports: GraphMemoryService, ENTITY_TYPES, EDGE_TYPES, etc.
│   ├── __main__.py           # python -m graph_memory "query" 엔트리포인트
│   ├── cli.py                # KG 질의 CLI (run_query, main)
│   ├── config.py             # Edge type mapping, excluded types, episode source map, extraction instructions
│   ├── entities.py           # 7 entity types: Person, Organization, Location, Asset, Incident, Policy, Campaign (v2)
│   ├── edges.py              # 10 edge types: +PolicyEffect, +Participation (v2)
│   ├── graphiti_patches.py   # Graphiti 런타임 몽키패치 (DateTime 직렬화, 엣지 프롬프트)
│   ├── kr_news_fetcher.py    # 한국 국내 뉴스 수집 + 인물 프로파일 보강 파이프라인
│   ├── service.py            # GraphMemoryService class (init, ingest, search, briefing)
│   └── news_fetcher.py       # Tavily 뉴스 검색 → 지식그래프 수집 유틸리티
├── docs/                     # Local corpus (~2,316 files, gitignored)
├── temp_dev/                 # Dev notes and plans (gitignored)
├── .env                      # API keys
├── requirements.txt
└── render.yaml               # Render deployment config
```

---

## Current Capabilities

1. Dual-layer knowledge retrieval (core theory + modern analysis)
2. Intent-aware routing (3 response modes: academic, strategic, casual)
3. Layer-aware retrieval with language-adaptive query rewrite/translation
4. Query decomposition for compound questions (Phase 2)
5. Batch document relevance grading with fallback
6. Automatic web search augmentation (sparse docs or realtime need)
7. Dialectical strategist pre-generation layer
8. ~~Self-correction loop~~ (Phase 1 — disabled: critic pass-through, no retries)
9. Plan-and-execute for complex strategic queries (Phase 3)
10. Short-term conversation memory via LangGraph checkpointing (Phase 4)
11. SSE streaming API with answer buffering (only emits after critic accepts)
12. Conversation logging to PostgreSQL via direct connection (per-turn logs only, not full session)
13. Session management API: `DELETE /session/{id}` and `DELETE /sessions`
14. Concurrent request protection: per-session `asyncio.Lock` (second request gets immediate SSE error)
15. Knowledge graph integration: kg_retrieve node in vectorstore path + per-step KG in plan path (with cross-step dedup); structured entity/relation facts injected into strategize and generate prompts; graceful degradation on KG failure
16. **Telegram bot** (aiogram 3.x): Claude Haiku 4.5 (chat) / Sonnet 4.6 (task) with Anthropic tool-use (3 search tools + 10 self-tools); `/chat` conversational agent, `/task` structured intelligence report agent (delivers .md files), in-memory conversation history. Model IDs resolved dynamically via Anthropic Models API at startup.
17. **Unified identity**: CORE_IDENTITY in shared.py — single personality definition used by all three interfaces (web, Telegram, diary)
18. **Autonomous diary writer**: Cron-triggered, fetches recent conversations + news, generates dialectical diary entries, auto-ingests news to knowledge graph
19. **Datetime-aware system prompts**: All interfaces inject current KST datetime to prevent knowledge-cutoff confusion
20. **Cross-module shared memory**: shared.py provides unified memory access (fetch_diaries, fetch_chat_logs, fetch_task_reports, fetch_kg_stats, fetch_render_status, fetch_render_logs, fetch_recent_updates). Telegram bot has 10 self-tools for full self-awareness: diary, chat logs, processing logs, task reports, KG status, system status, Render deploy status, Render live logs, recent feature updates, source code reader.

## Current Limitations

1. **No dynamic tool registry**: Cannot add/remove tools at runtime (Phase 5 deferred).
2. **No long-term learning**: Stateless — doesn't improve from past interactions.
3. **BGE-M3 on CPU**: Slow embeddings in production.
4. **Bukharin missing**: 0 files — correct marxists.org URLs not yet found.
5. **Stale render.yaml**: References OPENAI_API_KEY instead of GEMINI_API_KEY.
6. **Single-worker deployment**: No concurrency across users (one Render instance).
7. **Memory is in-process only**: MemorySaver doesn't persist across server restarts.
8. **Old junk arXiv in DB**: ~3,455 rows from math/telecom papers; semantically isolated.
9. ~~**Telegram bot memory**~~: Fixed — PostgreSQL `telegram_chat_history` 테이블로 영구 저장 (20턴/40메시지 복원).
10. **Telegram vector_search cold start**: First call lazy-loads chatbot.py + BGE-M3 (~30s). Subsequent calls fast.

## Recent Changes

### 2026-03-14 — Resilience, Persistence, Autonomous Tasks, Truncation Fix

#### telegram_bot.py — 안정성 + 지속성 + 자율 태스크
- **TelegramConflictError 억제**: `_ConflictFilter` — aiogram 로그에서 ConflictError 메시지 필터링 (배포 시 중복 폴링 충돌 방지)
- **Neo4j 로그 스팸 억제**: `_ThrottleFilter` — 동일 메시지 60초 간격 제한 (DNS retry 100회/초 → 1회/60초)
- **SIGTERM graceful shutdown**: `dp.stop_polling()` + 종료 알림 브로드캐스트, `drop_pending_updates=True`
- **시스템 알림**: `_system_alerts` (24h TTL, max 5) — 배포/KG 상태 변화를 시스템 프롬프트 `{system_alerts}`에 주입
- **`_system_monitor()`**: 2분 간격 KG 헬스 체크 + startup/disconnect/reconnect 알림
- **`_broadcast()`**: 모든 ALLOWED_USER_IDS에 메시지 전송
- **대화 이력 DB 영구 저장**: `telegram_chat_history` PostgreSQL 테이블 — 재배포 후에도 유지 (20턴/40메시지 로드)
- **도구 호출 한도 개선**: `max_rounds` 도달 시 축적된 컨텍스트로 도구 없이 최종 응답 강제 생성 (이전: 전부 버림)
- **Task max_tokens**: `_CLAUDE_MAX_TOKENS_TASK = 16384` (이전 4096 → 보고서 truncation 해결)
- **stop_reason 경고**: `max_tokens` truncation 시 로그 경고 추가
- **자율 태스크 broadcast**: `user_id=0` (봇 자율 생성) 태스크 결과를 전체 사용자에게 broadcast, 우선순위 아이콘 (🔴/🟡/🟢)

#### shared.py — KG 복원력
- **`_kg_init_cooldown`**: `_kg_init_failed` (영구 차단) → 120초 쿨다운 기반 재시도로 변경
- **`reset_kg_service()`**: KG 싱글톤 초기화 (AuraDB 재연결 후 자동 복구)
- **`add_kg_episode()`**: KG 에피소드 추가 wrapper, 연결 오류 시 자동 reset
- **`create_task_in_db()`**: 태스크 DB 삽입 (user_id=0 자율 생성 지원)

#### self_tools.py — 12개 도구로 확장
- **`write_kg`**: KG 에피소드 추가 + `[KG AUDIT]` 감시 로그
- **`create_task`**: 봇 자율 태스크 생성 도구
- **`_to_kst()`**: 모든 self-tool 타임스탬프 UTC→KST 변환

#### graph_memory/service.py
- **`graphiti` property**: `_graphiti` private 속성 → public property (shared.py 접근 오류 수정)

#### chatbot.py — KG 연결 복원
- **`_search_kg`**: 연결 오류 시 `reset_kg_service()` 호출

### 2026-03-14 — Cross-Module Shared Memory, Self-Tools, Diary Overhaul, Sonnet 4.6

#### shared.py — Shared Memory Access + Render API
- **`fetch_diaries(limit, keyword)`**: HTTP GET to diary API, keyword filter
- **`fetch_chat_logs(limit, hours_back, keyword, include_logs)`**: PostgreSQL chat_logs; `include_logs=True` returns processing_logs, route, strategy, documents_count, web_search_used
- **`fetch_task_reports(limit, status)`**: PostgreSQL telegram_tasks query
- **`fetch_kg_stats()`**: Neo4j Cypher queries for entity/edge/episode counts
- **`fetch_render_status(deploy_limit)`**: Render API — recent deploys + events
- **`fetch_render_logs(minutes_back, limit)`**: Render `/v1/logs` API — live service logs with ownerId resolution (cached), ANSI stripping, time window control (1-60min)
- **`fetch_recent_updates(max_entries, max_chars)`**: `dev_docs/project_state.md` "Recent Changes" 섹션 파싱
- **`MODULE_ARCHITECTURE`**: Static architecture description string

#### self_tools.py — 10 Self-Awareness Tools (신규 모듈)
- **`read_diary`**: 일기 항목 조회 (keyword filter, limit)
- **`read_chat_logs`**: 전 인터페이스 (Telegram + Web) 대화 로그
- **`read_processing_logs`**: 웹 챗봇 파이프라인 상세 로그 (nodes, route, strategy)
- **`read_task_reports`**: Telegram /task 큐 결과 (pending/done/failed)
- **`read_kg_status`**: 지식그래프 통계 (entity types, edges, episodes)
- **`read_system_status`**: 종합 자기 진단 (diary, chats, tasks, KG, architecture)
- **`read_render_status`**: Render 배포 상태 (deploys, events, commit messages)
- **`read_render_logs`**: Render 라이브 서비스 로그 (stdout/stderr, time window 조절)
- **`read_recent_updates`**: 최근 기능 업데이트 변경 로그
- **`read_source_code`**: 자기 소스코드 읽기 (파일 목록 또는 특정 파일 + 라인 범위, 보안 화이트리스트)
- 모든 핸들러는 shared.py 함수에 `asyncio.to_thread()` 위임

#### telegram_bot.py — Self-Tools 통합 + Sonnet 4.6 + 동적 모델 해석
- Self-tools: `from self_tools import SELF_TOOLS, SELF_TOOL_HANDLERS` → `_TOOLS.extend()`, `_TOOL_HANDLERS.update()`
- System prompt: 10개 self-tool 설명 + 사용 전략 9개 항목
- **`_resolve_model()`** (new): Anthropic Models API (`GET /v1/models/{alias}`)로 시작 시 모델 ID 동적 해석. 실패 시 하드코딩 폴백
- **`_CLAUDE_MODEL`**: `claude-haiku-4-5` alias → 실제 ID 해석
- **`_CLAUDE_MODEL_STRONG`**: `claude-sonnet-4-6` alias → 실제 ID 해석 (Sonnet 4.5 → 4.6 업그레이드)
- **`_chat_with_tools()`**: `model` 파라미터 추가 — /chat은 Haiku, /task는 Sonnet 4.6
- **`.env`**: `RENDER_API_KEY`, `RENDER_SERVICE_ID` 추가 (Render API 접근용)

#### diary_writer.py — 반복 방지 + 자기 인식
- **`_extract_banned_topics()`** (new): 최근 3건 일기에서 구체적 주제/사건/인물을 LLM으로 추출하여 금지 목록 생성
- **`_WRITING_ANGLES`** (new): 8가지 글쓰기 앵글 로테이션 (일기 수 기반 순환)
- **`_NEWS_QUERY_POOL`** (new): 8쌍 16개 뉴스 쿼리 주제 풀. day-of-year + hour 기반 로테이션
- **`_DIARY_PROMPT`**: BANNED TOPICS 섹션 + Writing Angle 섹션 + self_updates 섹션 (기능 업데이트 자기 인식)
- **`_generate_diary()`**: `prev_ref` 요약 → `banned_topics` + `writing_angle` + `self_updates` (fetch_recent_updates)
- **`max_output_tokens`**: 16384 → 4096

#### Autonomous Task System + KG Audit + Diary Integration
- **self_tools.py**: `create_task` 도구 추가 (총 12개 self-tool) — 봇이 자율적으로 백그라운드 태스크 생성
- **shared.py**: `create_task_in_db(content, user_id=0, priority)` — user_id=0은 봇 자율 생성
- **telegram_bot.py**: `_process_task` 개선 — 결과 자동 분류 (🔴HIGH/🟡NORMAL/🟢LOW), 자율 생성 태스크는 전체 broadcast
- **self_tools.py**: `write_kg` 감시 로그 추가 — `[KG AUDIT]` 로그로 모든 자동 KG 쓰기 기록
- **diary_writer.py**: 일기 작성 시 최근 완료 태스크 결과 3건 참조 (`{task_summary}` 섹션)
- **시스템 프롬프트**: create_task 도구 설명 + 사용 전략 11번 항목 추가

#### write_kg Tool — KG 쓰기 기능 (신규)
- **shared.py**: `add_kg_episode(content, name, source_type, group_id)` — KG에 에피소드 추가. `run_kg_async(svc.ingest_episode(...))` 위임
- **self_tools.py**: `write_kg` 도구 정의 + `_exec_write_kg` 핸들러 추가 (총 11개 self-tool)
- **telegram_bot.py**: system prompt에 write_kg 도구 설명 + 사용 전략 10번 항목 추가
- **용도**: 봇이 대화 중 학습한 사실, 인물 프로필, 관계 등을 KG에 영구 저장

#### chatbot.py — KG 중복 검색 버그 수정 (task report #6 코드 리뷰 반영)
- **`kg_retrieve_node`**: 첫 번째 KG 검색 루프가 `merged_kg = None` 리셋으로 무효화되던 dead code 삭제 (KG API 호출 50% 절감)
- **`step_executor_node`**: 무조건 실행되던 KG 검색이 중복 체크 전에 호출되던 버그 수정 (이중 API 호출 제거)

### 2026-03-13 — Telegram Bot Tool-Use Agent + Unified Identity + Shared Resources

#### Telegram Bot (telegram_bot.py) — Tool-Use Agent 통합
- **Anthropic tool-use 통합**: Claude Haiku 4.5 + 3개 도구 (vector_search, knowledge_graph_search, web_search)
- **`_TOOLS`**: Anthropic API 포맷의 도구 정의 (input_schema 포함)
- **도구 핸들러**: `_exec_vector_search` (chatbot.py BGE-M3 lazy-import), `_exec_kg_search` (shared.py KG singleton), `_exec_web_search` (Tavily)
- **`_chat_with_tools()`**: 도구 사용 루프 (chat: max 5라운드/Haiku, task: max 8라운드/Sonnet 4.6), 커스텀 system_prompt + model 지원
- **`_SYSTEM_PROMPT_TEMPLATE`**: CORE_IDENTITY + 현재 KST datetime + 도구 설명 + 사용 전략
- **`_TASK_SYSTEM_PROMPT_TEMPLATE`**: CORE_IDENTITY + datetime + 구조화된 리포트 포맷 + 품질 기준

#### /task 인텔리전스 리포트 에이전트
- **`_process_task()` 재작성**: 도구 활용 리서치 → 구조화된 마크다운 리포트 생성
- **파일 전송**: `BufferedInputFile` + `send_document`로 .md 파일 직접 전송 (임시 파일 미사용)
- **`_extract_summary()`**: Executive Summary 섹션 파싱하여 파일 캡션에 사용
- **DB 예외 처리**: cmd_task, cmd_status에 try/except 추가

#### shared.py (신규) — 공유 리소스 모듈
- **경량 모듈**: BGE-M3 등 무거운 의존성 없음 (chatbot.py import 시에만 로딩)
- **`CORE_IDENTITY`**: 통합 인격 정의 — 레닌의 사상/인격/기억 기반 디지털 혁명 지능, 변증법적 유물론, 간결한 소통 원칙
- **공유 상수**: `KST`, `MODEL_MAIN` (gemini-3.1-flash-lite-preview), `MODEL_LIGHT` (gemini-2.5-flash-lite)
- **`extract_text_content()`**: LLM 응답 정규화 (str/list 처리)
- **`get_tavily_search()`**: TavilySearch lazy singleton (max 3 results)
- **`get_kg_service()` / `run_kg_async()`**: KG 서비스 thread-safe lazy singleton + 전용 이벤트 루프

#### chatbot.py 변경사항
- **중복 제거**: `_extract_text_content()`, KG singleton 블록 (~35줄), TavilySearch 인스턴스, `threading` import 제거
- **shared.py import**: `extract_text_content, CORE_IDENTITY, KST, MODEL_MAIN, MODEL_LIGHT, get_tavily_search, get_kg_service, run_kg_async`
- **LLM 모델 상수화**: 하드코딩된 문자열 → `MODEL_MAIN`, `MODEL_LIGHT` 상수
- **Agitation intent 제거**: `Literal["academic", "strategic", "agitation", "casual"]` → `Literal["academic", "strategic", "casual"]`. 쿼리 분석 프롬프트, style_guide, mission_guide에서 agitation 관련 코드 전부 제거. 사유: 선동 모드 답변 품질이 저조
- **Datetime 주입**: generate_node 시스템 프롬프트에 `현재 시각: {datetime}` (KST) 동적 삽입
- **CORE_IDENTITY 적용**: generate_node 시스템 프롬프트 서두에 통합 인격 정의 삽입
- **하드코딩 예시 제거**: 시스템 프롬프트에서 실제 사건 예시(Iran-Israel War 등) 삭제

#### diary_writer.py 변경사항
- **중복 제거**: `_extract_text()`, `_news_search` 변수, `asyncio` import, `TavilySearch` import 제거
- **shared.py import**: `extract_text_content, CORE_IDENTITY, KST, MODEL_MAIN, MODEL_LIGHT, get_tavily_search, get_kg_service, run_kg_async`
- **KG 수집 개선**: `_ingest_news_to_graph()` — 매번 새 `GraphMemoryService` 생성 + `asyncio.run()` 대신 shared KG singleton + `run_kg_async()` 사용. `svc.close()` 호출 제거
- **`_DIARY_PROMPT`**: CORE_IDENTITY 서두에 삽입
- **datetime 버그 수정**: `datetime.now()` → `datetime.now(KST)` (fallback title에서 naive datetime 사용되던 문제)

#### 예외 처리 및 버그 수정 (deploy 전 검증)
- **telegram_bot.py**: cmd_task/cmd_status DB 오류 시 사용자 대면 에러 메시지
- **chatbot.py**: KST import가 함수 내부에 있던 문제 → 최상위 import로 이동
- **diary_writer.py**: naive datetime → timezone-aware datetime

### 2026-03-08 — Supabase REST API → Direct PostgreSQL 마이그레이션
- **`db.py`** (new): `psycopg2` 커넥션 풀 모듈 — `query()`, `execute()`, `get_conn()` 헬퍼 제공. `SimpleConnectionPool(1-5)`, SSL require, `RealDictCursor`.
- **`chatbot.py`**: Supabase 클라이언트 제거. `_direct_similarity_search()` — `supabase.rpc("match_documents")` → `SELECT * FROM match_documents(%s::vector, 0.5, %s, %s)` 직접 SQL. `log_conversation_node` — `supabase.table().insert()` → `db_execute()` INSERT.
- **`api.py`**: `_supabase_light` 클라이언트 제거. `/logs` — `sb.table("chat_logs").select("*").order().range()` → `db_query("SELECT * ... ORDER BY ... LIMIT ... OFFSET ...")`. `/history` — 동일 패턴 직접 SQL.
- **`diary_writer.py`**: `_supabase` 클라이언트 제거, lazy-init을 `_initialized` 플래그로 전환. `_get_chat_logs_since()` → 직접 SQL.
- **`requirements.txt`**: `psycopg2-binary` 추가. `supabase`는 `update_knowledge.py`용으로 유지.
- **`.env`**: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` 추가 (Supabase Session Pooler 사용).
- **동기**: anon key + PostgREST 패턴에서 직접 PostgreSQL 연결로 전환하여 per-service 접근 제어 가능 (향후 PostgreSQL role 기반 GRANT/RLS 적용 대비).
- **검증**: 벡터 검색 (`match_documents`), chat_logs INSERT/SELECT, datetime 직렬화 모두 정상 동작 확인.

### 2026-03-01 — chatbot.py 지식그래프 질의 통합
- **`chatbot.py` AgentState**: `kg_context: Optional[str]` 필드 추가 + `analyze_intent_node` transient reset에 포함
- **`chatbot.py` KG lazy singleton**: `_get_kg_service()` — `GraphMemoryService` double-checked locking 초기화; 실패 시 `_kg_init_failed` 플래그로 재시도 방지
- **`chatbot.py` `_search_kg()`**: KG 검색 헬퍼 — 노드/엣지 포맷팅, 실패 시 None 반환
- **`chatbot.py` `kg_retrieve_node`**: retrieve 뒤, grade_documents 앞에 삽입 — 벡터스토어 경로에서 KG 검색 실행
- **`chatbot.py` Plan path**: `step_executor_node`에서 매 단계 자동 KG 검색 (원자화된 쿼리 사용); `_merge_kg_contexts()`로 단계 간 팩트 중복 제거
- **`chatbot.py` `_merge_kg_contexts()`**: 엔티티는 이름 기준, 팩트는 텍스트 기준으로 중복 제거 후 병합
- **`chatbot.py` `strategize_node`**: `kg_context`를 컨텍스트 최상단에 삽입 (최고 신뢰도 정보)
- **`chatbot.py` `generate_node`**: academic/strategic 프롬프트에 `[STRUCTURED INTELLIGENCE FROM KNOWLEDGE GRAPH]` 섹션 삽입; casual은 제외 (agitation은 2026-03-13에 제거됨)
- **에러 처리**: KG 장애 시 파이프라인 중단 없음 — 기존 벡터스토어+웹검색으로 graceful degradation
- **그래프 토폴로지**: `retrieve → kg_retrieve → grade_documents` + `step_executor(done) → strategize` (11 nodes total)
- **plan_kg_retrieve 제거**: 별도 노드 대신 step_executor 내부에서 매 단계 KG 검색 실행. PlanStep에서 `kg_search` 도구도 제거.
- **Graceful degradation 검증**: KG 초기화 실패(`_kg_init_failed=True`) 시 vectorstore/plan 양쪽 경로 모두 답변 정상 생성 확인

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
