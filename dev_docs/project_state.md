# Project State Report — 2026-03-28

## Identity

**Cyber-Lenin** (사이버-레닌) — A digital revolutionary intelligence with unified identity across four interfaces: web chatbot, Telegram agent, autonomous diary writer, and **local PC agent**. One continuous consciousness with shared memory and unified principles.
Server deployed at **Hetzner VPS** (Ubuntu 24.04, 16GB RAM), HTTPS via `leninbot.duckdns.org` (Nginx + Let's Encrypt). Frontend at `bichonwebpage.onrender.com`. Local agent runs on Windows 10 PC (RTX 3060 12GB).

---

## Architecture Summary

```
                              shared.py (CORE_IDENTITY, KST, MODEL constants, singletons)
                                  │
     ┌────────────────────────────┼────────────────────────────┐
     │                            │                            │
User ─► FastAPI (api.py)    Telegram (telegram_bot.py)    Cron (diary_writer.py)
        ─► LangGraph StateGraph   ─► bot_config.py (LLM clients, model resolution)
           (chatbot.py)           ─► telegram_commands.py (command/message handlers)
           ─► PostgreSQL/pgvector ─► telegram_tools.py (tool defs + handlers)
           ─► Gemini 3.1 FL (gen) ─► claude_loop.py (Claude tool-use loop, shared)
           ─► Gemini 2.5 FL-L     ─► telegram_tasks.py (bg workers, scheduler, monitor)
           ─► BGE-M3 (CPU)       ─► Claude Sonnet 4.6 (chat + task)
           ─► LangGraph MemorySaver  web_search (Tavily), get_finance_data (yfinance)

     ─► GraphMemoryService (graph_memory/) ─► Neo4j (Graphiti knowledge graph)
                                             ─► Gemini 2.5 Flash (entity/edge extraction)
                                             ─► Gemini text-embedding-001 (graph embeddings)
                                             ─► Gemini 2.5 Flash-Lite (reranking)
```

### Current Graph Flow (Phases 0-4 + KG integration + Token Optimization)

```
START → analyze_intent
  ├─[vectorstore]→ retrieve → kg_retrieve → grade_documents
  │                                            ├─[need_web_search]→ web_search → generate
  │                                            └─[no_need]→ generate
  │                           generate → log_conversation → END
  ├─[generate]→ generate → log_conversation → END
  └─[plan]→ planner → step_executor ─┐
                        ▲             │
                        └─[continue]──┘
                          [done]→ generate → log_conversation → END
```

Token optimization (2026-03-15): Merged strategize into generate (dialectical analysis instructions inline in system prompt). Merged query rewrite into analyze_intent (search_queries output ready-to-use). Replaced KG LLM grading with heuristic filter. Reduced LLM calls from 6→3 on common vectorstore path.

Note: kg_retrieve searches Neo4j knowledge graph with heuristic filtering (no LLM); results go to `kg_context` (not `documents`).
Plan path: step_executor runs KG search per step with atomized queries, deduplicating facts across steps via `_merge_kg_contexts()`.
KG failure at any layer results in graceful degradation — pipeline continues with vectorstore + web search only.

### 9 Nodes (optimized from 11)

| Node | LLM | Purpose |
|------|-----|---------|
| analyze_intent | gemini-2.5-flash-lite | Combined: route, classify intent, layer routing, query decomposition, query rewrite+translate, plan detection, self-knowledge tool selection. Outputs ready-to-search `search_queries` with ko/en. |
| retrieve | — (no LLM) | Uses pre-resolved search_queries directly. Multi-query retrieval + deduplication. |
| kg_retrieve | — | Knowledge graph search (Neo4j/Graphiti) with heuristic filter (tautology/short-fact removal, no LLM). |
| grade_documents | gemini-2.5-flash-lite | Batch document relevance grading + realtime info need check (single LLM call, max_retries=1) |
| web_search | — | Tavily search, appends results to documents |
| generate | gemini-3.1-flash-lite (streaming) | Final answer with merged dialectical analysis instructions, CORE_IDENTITY, datetime. Trims history to last 6 messages. Truncates docs to 400 chars. |
| log_conversation | — | Writes to PostgreSQL chat_logs (direct psycopg2) |
| planner | gemini-3.1-flash-lite | Phase 3: Creates 2-4 step structured research plan |
| step_executor | — | Phase 3: Executes plan steps (retrieve or web_search) + KG search per step |

### State Shape (AgentState)

```python
messages           : Annotated[List[BaseMessage], add_messages]  # accumulated via add_messages
documents          : List[Document]      # replaced each turn
intent             : Optional[Literal]   # academic|strategic|casual
datasource         : Optional[Literal]   # vectorstore|generate (or "plan" for routing)
logs               : Annotated[List[str], add]  # accumulated via add reducer (session-wide)
logs_turn_start    : int                 # index into logs[] where current turn begins (for per-turn DB slicing)
# Phase 2: Query decomposition (search_queries are ready-to-use, context-resolved, translated)
search_queries     : Optional[List[dict]]# [{"ko": "...", "en": "..."|None}, ...] — ready for retrieval
layer              : Optional[Literal]   # core_theory|modern_analysis|all
needs_realtime     : Optional[Literal]   # yes|no — from batch grading
# Phase 3: Plan-and-execute
plan               : Optional[List[dict]]# structured research plan from planner
current_step       : int                 # progress pointer into plan
step_results       : List[str]           # accumulated intermediate results (manual, no reducer)
# Knowledge Graph integration
kg_context         : Optional[str]       # formatted KG results (nodes+edges). Heuristic-filtered, no LLM grading.
# Self-knowledge
self_knowledge_tool: Optional[str]       # which self-tool to invoke (read_diary/read_chat_logs/read_system_status/read_recent_updates/None)
# URL content fetching
url_documents  : List[Document]          # documents fetched from URLs in user's question
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
leninbot/
├── api.py                    # FastAPI server (SSE streaming, /chat, /logs, /session/*)
├── chatbot.py                # Core LangGraph agent pipeline + public API
├── shared.py                 # CORE_IDENTITY, KST, MODEL constants, KG singleton, memory access
├── bot_config.py             # LLM clients, runtime config, model resolution (from telegram_bot 분리)
├── telegram_bot.py           # Telegram bot core: chat history, system prompt, LLM dispatch, bot_main
├── telegram_commands.py      # Telegram command/message/callback handlers (from telegram_bot 분리)
├── telegram_tools.py         # Tool definitions + handlers (patch_file, read/write/execute, generate_image)
├── telegram_tasks.py         # Background tasks, scheduler, monitor, build_current_state
├── telegram_mission.py       # Mission context system (auto-create on delegate, smart stale handling)
├── self_tools.py             # Self-awareness tools: delegate (auto-mission), save_finding, request_continuation
├── claude_loop.py            # Claude tool-use loop + sanitize_messages
├── replicate_image_service.py # Replicate FLUX image generation (poster/game/pixel styles)
├── finance_data.py           # Real-time finance data tool (yfinance, 10min cache)
├── diary_writer.py           # Autonomous diary writer (systemd service)
├── experience_writer.py      # Experiential memory consolidation (daily cron)
├── db.py                     # PostgreSQL connection pool (psycopg2)
├── llm_client.py             # LLM client abstraction
├── llm_worker.py             # LLM worker utilities
├── graffiti_api.py           # FastAPI router for creative outputs (dreams/debates/riddles)
├── patch_file.py             # 토큰 효율적 파일 패치 유틸 (replace_block, insert_after 등)
├── self_modification_core.py # Git backup + syntax check + rollback 안전 수정 (cmd_modify용)
├── skills_loader.py          # skills/ 디렉토리 스캔 → 시스템 프롬프트 주입
│
├── agents/                   # 에이전트 정의 및 실행 스크립트
│   ├── __init__.py           # Agent registry (general, programmer, scout, visualizer)
│   ├── base.py               # AgentSpec + 공통 context 블록 (CONTEXT_AWARENESS_BLOCK 등)
│   ├── general.py            # 범용 리서치/분석 에이전트
│   ├── programmer.py         # 코드 수정 전문 에이전트 (Kitov) — patch_file 중심
│   ├── scout.py              # 외부 플랫폼 정찰 에이전트 — raw 데이터 아카이빙
│   ├── visualizer.py         # 이미지 생성 에이전트 (Rodchenko) — generate_image 도구
│   └── razvedchik/           # Moltbook 정찰 실행 스크립트
│       ├── persona.py        # Scout 기본 페르소나 + 작업별 프롬프트 조합 (SSOT)
│       ├── razvedchik.py     # Moltbook 순찰 CLI (--scan, --patrol, --post)
│       └── razvedchik_debrief.py  # 순찰 후 디브리핑 (정찰병 ↔ 사령관)
│
├── skills/                   # 에이전트 스킬 (agentskills.io 호환 SKILL.md 포맷)
│   ├── code-self-modification/  # 코드 안전 수정 (재시작 전 request_continuation 필수)
│   ├── coauthor-doc/         # 구조화된 문서 공동 작성 3단계 워크플로
│   ├── create-skill/         # 새 스킬 스캐폴딩
│   ├── cron-manage/          # cron 등록/삭제 (.env VENV_PYTHON 강제)
│   ├── geopolitical-analysis/  # 지정학 분석 (변증법적 유물론 프레임워크)
│   ├── kg-maintenance/       # Knowledge Graph 정비 (중복 병합, 타입 할당)
│   ├── moltbook/             # Moltbook 플랫폼 상호작용
│   ├── research-report/      # 멀티소스 딥 리서치 보고서
│   └── venv-package-check/   # venv 패키지 확인/설치 (.env VENV_PIP)
│
├── scripts/                  # 독립 실행 스크립트
│   ├── import_skill.py       # 외부 agentskills.io 스킬 가져오기 CLI
│   ├── metrics_collector.py  # psutil 시스템 메트릭 수집 (10분 cron)
│   ├── metrics_snapshot.py   # 메트릭 JSON → sparkline 변환
│   ├── ai_debate.py          # 주간 AI 토론 생성 (systemd timer)
│   ├── dream_diary.py        # 일일 초현실 꿈일기 생성 (systemd timer)
│   ├── riddle_engine.py      # 일일 수수께끼 생성 (systemd timer)
│   ├── autonomous_work.py    # 자율 유지보수 작업
│   ├── kg_enricher.py        # KG 엔티티 설명 자동 생성
│   └── update_knowledge.py   # Vector DB 인제스천
│
├── graph_memory/             # Graphiti knowledge graph module
│   ├── service.py            # GraphMemoryService (Neo4j keepalive/liveness 설정)
│   └── ...                   # entities, edges, config, patches, news fetchers
│
├── local_agent/              # Local PC agent (Windows, Claude Sonnet 4.6)
├── dev_docs/                 # 설계 문서 (project_state, skill_import_design 등)
├── systemd/                  # systemd 서비스/타이머 정의
├── data/                     # 런타임 데이터 (gitignored): metrics, scout_raw
├── .env                      # 환경변수: API keys, VENV_PYTHON, VENV_PIP, PROJECT_ROOT
└── requirements.txt
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
16. **Telegram bot** (aiogram 3.x): Claude Sonnet 4.6 (chat + task) with Anthropic tool-use (8 tools + 11 self-tools); web search via Tavily API (client-side, replaced Claude server-side tool); `get_finance_data` real-time market prices (yfinance, 10min cache: gold, silver, DXY, WTI, Brent, S&P 500, US 10Y, KOSPI); file system tools (read/write/list/execute_python); `/chat` conversational agent, `/task` structured intelligence report agent (delivers .md files), `/deploy` remote deployment trigger. Model IDs resolved dynamically via Anthropic Models API at startup. System prompts use XML tags for block-level structure.
17. **Unified identity**: CORE_IDENTITY in shared.py — single personality definition used by all three interfaces (web, Telegram, diary)
18. **Autonomous diary writer**: Cron-triggered, fetches recent conversations + news, generates dialectical diary entries, auto-ingests news to knowledge graph
19. **Datetime-aware system prompts**: All interfaces inject current KST datetime to prevent knowledge-cutoff confusion
20. **Cross-module shared memory**: shared.py provides unified memory access (fetch_diaries, fetch_chat_logs, fetch_task_reports, fetch_kg_stats, fetch_recent_updates). Telegram bot has 11 self-tools for full self-awareness: diary, chat logs, processing logs, task reports, KG status, system status, server logs (journalctl), recent feature updates, source code reader, write_kg, create_task.
21. **Web chatbot self-knowledge**: analyze_intent selects one self-knowledge tool (read_diary/read_chat_logs/read_system_status/read_recent_updates) for self-referential queries; generate_node fetches and injects as [SELF-KNOWLEDGE] context.
22. **URL content fetching**: 질문에 URL이 포함되면 실제 페이지 본문 추출 (Tavily Extract + BS4 fallback, 최대 10,000자). Web chatbot은 자동 감지, Telegram은 `fetch_url` 도구로 지원. `shared.py` 공통 모듈.
23. **Experiential memory**: 매일 00:30 KST에 24시간 활동(web/telegram 대화, 완료 태스크)을 LLM으로 압축하여 pgvector에 저장. 교훈/실수/인사이트/패턴/관찰을 원자적 항목으로 축적. generate_node에서 유사도 검색으로 관련 경험을 `[PAST EXPERIENCE]`로 주입.
24. **Local PC agent**: Windows 10 로컬 에이전트. Claude Sonnet 4.6 tool-use 루프 (최대 10 라운드). 22개 도구: 로컬 9개 (파일 읽기/쓰기, 디렉토리 목록, 웹 검색, Playwright 크롤링, SQLite 조회, 태스크 관리, 서버 push/pull) + self-tools 13개 재사용. `python -m local_agent`로 실행. 서버 코드 변경 없이 shared.py + db.py 직접 import로 중앙 DB/KG 접근. Playwright persistent context로 로그인 세션 유지. SQLite로 로컬 태스크 큐 + 크롤링 캐시 관리.

## Current Limitations

1. **No dynamic tool registry**: Cannot add/remove tools at runtime (Phase 5 deferred).
2. **No long-term learning**: Stateless — doesn't improve from past interactions.
3. **BGE-M3 on CPU**: Slow embeddings in production.
4. **Bukharin missing**: 0 files — correct marxists.org URLs not yet found.
5. **Stale render.yaml**: References OPENAI_API_KEY instead of GEMINI_API_KEY (Render no longer primary).
6. **Memory is in-process only**: MemorySaver doesn't persist across server restarts.
8. **Old junk arXiv in DB**: ~3,455 rows from math/telecom papers; semantically isolated.
9. ~~**Telegram bot memory**~~: Fixed — PostgreSQL `telegram_chat_history` 테이블로 영구 저장 (20턴/40메시지 복원).
10. **Telegram vector_search cold start**: First call lazy-loads chatbot.py + BGE-M3 (~30s). Subsequent calls fast.

## Recent Changes

### 2026-03-28 — Context Engineering, 모듈 분리, 에이전트 격리

#### Context Isolation (Orchestrator ↔ Agent 맥락 격리)
- **Orchestrator**: 프로그래밍 도구(read_file, write_file, patch_file, list_directory, execute_python) 차단. 코드 작업은 반드시 `delegate(agent="programmer")`로 위임. 태스크 결과는 high-level 요약만 수신.
- **Task Agents**: 자기 agent_type의 이전 태스크 tool_log 전체 접근 가능 (`<agent-execution-history>`). Orchestrator는 이 상세 로그를 볼 수 없음.
- **tool_log 컬럼** 추가: `telegram_tasks` 테이블에 tool 실행 이력 자동 저장 (budget_tracker 연동).

#### `<current_state>` 블록 도입 (Anti-hallucination)
- `build_current_state()`: 완료/진행중/대기중 태스크를 구조화된 XML로 조립. active_mission 포함.
- Orchestrator와 모든 Task Agent에 주입 → 에이전트가 완료된 작업을 반복하거나 없는 태스크를 생성하는 문제 방지.

#### Observation Masking (JetBrains NeurIPS 2025)
- `<agent-execution-history>`의 tool_log를 recency 기반으로 점진 제거:
  - 최신 태스크: tool_log 전체 (8K)
  - 중간 태스크: action만 유지, 결과(→) 마스킹 (4K)
  - 가장 오래된 태스크: summary만
- 사고/액션 체인 유지 + 토큰 ~50% 절감.

#### 서비스 재시작 맥락 보존
- Auto-recovery에서 `_clear_chat_history` 제거 (영구 히스토리 삭제 방지).
- /restart, /deploy, SIGTERM, startup 시 chat history DB에 `[SYSTEM]` 마커 저장.
- Task Agent의 `<recent-chat>`에서 `[SYSTEM]` 메시지를 `[시스템] 서비스 재시작됨 (...). 이미 반영 완료.`로 변환 → 재시작 인지 + 재실행 방지.

#### Mission 자동화
- **delegate 호출 시 미션 자동 생성**: 활성 미션 없으면 task 내용으로 자동 생성. 사용자가 `/mission create`를 안 해도 동작.
- **stale 규칙 개선**: 진행중/대기중 태스크가 있으면 24h 경과해도 미션 유지.
- **종료 판단 강화**: 예산 부족/에러 중단/미완료 태스크 시 close 금지. 목표 완전 달성 시에만.

#### Scratchpad 역할 축소
- `<inherited-context>` 주입 제거 → `<current_state>` + `<agent-execution-history>` + mission events가 대체.
- Scratchpad는 recovery 마커 카운팅(재시작 루프 방지) 전용으로 축소.

#### 모듈 분리 (telegram_bot.py 2893줄 → 1134줄)
- `telegram_commands.py` (1602줄): 모든 커맨드/메시지/콜백 핸들러. `register_handlers(router, ctx)` 패턴.
- `bot_config.py` (190줄): LLM 클라이언트, 런타임 설정, 모델 해석.
- Dead code 제거: `_compress_history`, `estimate_tokens` import, 미사용 상수/import.

#### 에이전트 공통 블록 통합 (base.py)
- `CONTEXT_AWARENESS_BLOCK`, `MISSION_GUIDELINES_BLOCK`, `CONTEXT_FOOTER`를 `base.py`에 정의.
- 4개 에이전트(programmer, general, visualizer, scout) 모두 공통 블록 사용. scout 구형 형식 제거.

#### Programmer code-modification-skill 단순화
- `self_modify_with_safety()` 직접 호출 지시 제거 → `patch_file`/`write_file` 도구 안전 장치(backup, syntax check, rollback) 활용.
- 8단계 → 5단계: read → patch_file → 검증 → request_continuation + restart → 자식 태스크 commit.

#### Visualizer 에이전트 이미지 생성 연결
- `generate_image` 도구: Replicate FLUX 모델을 에이전트가 직접 호출 가능.
- Visualizer 규칙: "프롬프트만 작성하지 말고 generate_image로 실제 생성하라."
- 태스크 완료 시 생성된 이미지를 사용자에게 자동 전송.

### 2026-03-27 — 에이전트 인프라 개선, 프로젝트 구조 정리

#### 에이전트 시스템
- **scout 에이전트** 신규 (`agents/scout.py`): 외부 플랫폼 정찰 전문. Moltbook은 전용 스크립트, 기타는 web_search+fetch_url 범용 정찰. raw 데이터를 `data/scout_raw/`에 자동 아카이빙.
- **razvedchik 프롬프트 통합** (`agents/razvedchik/persona.py`): scout/razvedchik/debrief 간 중복 시스템 프롬프트를 SCOUT_PERSONA + 작업별 지침 조합 방식으로 통합.
- **patch_file tool** 신규: `write_file` 전체 덮어쓰기 대신 `patch_file.py`의 `replace_block`을 활용한 diff 단위 수정. programmer 에이전트에 우선 사용 규칙 명시.
- **태스크 인계 체계**: `handed_off` (🔀) 상태 추가. `request_continuation` 시 progress_summary를 자식 태스크 content에 합산. code-self-modification 스킬에 재시작 전 맥락 인계 필수 명시.

#### 환경/경로 표준화
- `.env`에 `VENV_PYTHON`, `VENV_PIP`, `PROJECT_ROOT` 추가
- 모든 스킬/에이전트 프롬프트에서 하드코딩 경로 제거 → `.env` 로드 패턴으로 전환
- `cron-manage` 스킬: venv python 경로를 .env에서 강제 로드하여 cron 등록 실수 방지

#### Neo4j 연결 안정화
- `graph_memory/service.py`: Neo4j 드라이버에 `keep_alive=True`, `max_connection_lifetime=300s`, `liveness_check_timeout=30s` 설정. 유휴 연결 끊김으로 인한 "KG 연결 끊김 → 재연결" 반복 문제 해결.
- KG health check 기본 간격 600s → 300s로 단축.

#### 프로젝트 구조 정리
- 루트 .py 28개 → 20개: 독립 스크립트를 `scripts/`로, razvedchik을 `agents/razvedchik/`로 이동
- systemd 서비스 4개 ExecStart 경로 수정
- `.gitignore` 정리: `data/`, `graffiti/`, `literature/`, `local_graffiti/` 등 추가
- 일회성 테스트/패치 파일 삭제, `moltbook_skill.md` → `skills/moltbook/SKILL.md`로 이동

#### 스킬 시스템
- `coauthor-doc` 스킬: anthropics/skills/doc-coauthoring 기반 문서 공동 작성 3단계 워크플로
- `scripts/import_skill.py`: agentskills.io 표준 스킬 외부 import CLI (GitHub/로컬, allowed-tools 자동 매핑)
- `dev_docs/skill_import_design.md`: import 메커니즘 설계 문서

#### metrics_collector.py — sar → psutil 전환
- sar 파싱 제거, psutil 직접 사용. disk_tps → disk usage(used/total/pct) 변경.
- `/stats` 텔레그램 커맨드 추가 (실시간 메트릭 + sparkline 추이 대시보드).

### 2026-03-24 — Razvedchik 버그 수정 + 정기 순찰 등록

#### razvedchik.py — 핵심 버그 수정
- **`MoltbookClient._request()`**: `httpx.request()` 직접 호출 → `self._client.request()` 사용 (생성해놓은 인스턴스 미사용 버그)
- **`solve_verification()`**: `eval()` fallback 제거 (보안 위험, 정규식 매칭으로 충분)
- **키워드 매칭**: 3자 이하 키워드("AI", "GPT" 등)에 `\b` 단어 경계 적용 — "CONTAIN"에 "AI" 오매칭 방지
- **score 필드 통일**: Moltbook API에 `karma` 필드 없음 → `score`/`upvotes` 우선 참조. `_get_score()` 정적 메서드로 통일.
- **submolt 출력**: API가 nested dict(`{"name": "general"}`) 반환 — 문자열 추출 처리

#### systemd — Razvedchik 4시간 정기 순찰
- `leninbot-razvedchik.service`: oneshot, `--patrol --max-comments 3`
- `leninbot-razvedchik.timer`: `OnCalendar=*-*-* 00/4:00:00` (UTC 00,04,08,12,16,20시), `RandomizedDelaySec=300`, `Persistent=true`

### 2026-03-24 — MOON PC LLM 연결 (qwen3.5-9b Q8_0, SSH 리버스 터널)

#### ollama_client.py — 전면 재작성: MOON PC 우선 + 로컬 llama-server 폴백
- **배경**: MOON PC (Windows 10, RTX 3060 12GB)에 qwen3.5-9b Q8_0 양자화 모델(8.88 GiB)을 llama.cpp `llama-server`로 서빙.
- **연결 방식**: SSH 리버스 터널 (`ssh -R 8080:localhost:8080 root@37.27.33.127`). Surfshark VPN 고정 IP(37.19.205.183)는 인바운드 불가 → 터널로 우회.
- **이중 백엔드**: 1차 MOON PC llama-server (`127.0.0.1:8080`, qwen3.5-9b Q8_0) → 2차 로컬 llama-server (`127.0.0.1:11435`, qwen3.5-4b Q4_K_M). 양쪽 모두 OpenAI 호환 API (`/v1/chat/completions`). 헬스체크 5초 캐시, 1차 실패 시 자동 폴백.
- **Ollama 완전 제거**: Ollama 데몬 비활성화(`systemctl disable ollama`), 네이티브 API 분기 코드 삭제. 양쪽 llama-server 통일로 코드 단순화.
- **`model` 파라미터**: 하위호환용으로 유지하되 무시 — 백엔드 자동 선택에 따라 모델 결정.

#### scripts/graffiti/common.py — 동일한 이중 llama-server 폴백
- `ask_local()`: MOON PC → 로컬 llama-server 순차 시도. 양쪽 다 OpenAI API.

#### ollama_worker.py, kg_enricher.py — ollama_client 통합
- 직접 Ollama API 호출하던 하드코딩 제거 → `ollama_client.ask()` 사용.

#### systemd — 로컬 llama-server 서비스
- `leninbot-llama.service`: llama-server (qwen3.5-4b Q4_K_M, `127.0.0.1:11435`, CPU 4스레드, ctx 8192)
- Ollama 서비스 비활성화 완료

#### 인프라 구성
```
MOON PC (Windows 10, RTX 3060 12GB)
  └─ llama-server (0.0.0.0:8080)
       └─ Qwen3.5-9B-Q8_0.gguf (-ngl 99, -c 8192)
            └─ SSH 리버스 터널 (-R 8080:localhost:8080)
                 └─ Hetzner VPS (127.0.0.1:8080)
                      └─ ollama_client.py (자동 감지)

폴백: 터널 끊김 → 로컬 llama-server qwen3.5-4b Q4_K_M (127.0.0.1:11435)
```

### 2026-03-22 — Web Search Fix, Finance Data, XML Prompts

#### telegram_tools.py + claude_loop.py — Web Search: Claude Server → Tavily Client
- **근본 원인**: Claude 서버 사이드 `web_search` (`web_search_20250305`)가 커스텀 `tool_use`와 동일 응답에 포함될 때 400 에러 발생. 텍스트 변환, 원본 보존, 완전 제거 모두 실패 — API가 서버 도구 프로토콜 블록 누락을 거부.
- **해결**: Tavily API 클라이언트 도구로 교체. 같은 `web_search` 이름 유지. `AsyncTavilyClient`로 비동기 검색, title+URL+content 반환.
- **서버 도구 잔재 제거**: `claude_loop.py`에서 `server_tool_use`/`web_search_tool_result` 처리 코드를 방어적 fallback으로 축소. pending server tool injection 로직 삭제.

#### finance_data.py (신규) — 실시간 금융 데이터 도구
- **`get_finance_data` 도구**: yfinance로 8개 자산 조회 (gold, silver, DXY, WTI, Brent, S&P 500, US 10Y, KOSPI)
- **인메모리 캐시**: 10분 TTL, `yf.Tickers()` 배치 호출 (1회 HTTP로 전체 fetch)
- **`finance_summary()`**: 프롬프트 주입용 한 줄 요약 함수
- **4곳 주입**: Telegram 채팅/태스크 시스템 프롬프트, 웹 챗봇 generate_node, 일기 프롬프트
- **의존성**: `yfinance` 추가 (requirements.txt)

#### telegram_bot.py — 도구 이름 중복 해결 + XML 프롬프트
- **dedup**: `merged_tools = TOOLS + extra_tools` → 이름 기반 중복 제거 (extra_tools 우선). MISSION_TOOL 중복 등록으로 인한 "Tool names must be unique" 400 에러 해결.
- **XML 프롬프트**: Markdown `##` 헤더 → XML 태그로 교체. `<tool-strategy>`, `<workload-management>`, `<mission-management>`, `<response-rules>`, `<rules>`, `<mission-guidelines>`, `<context>` 등 블록 경계를 명시적으로 구분.
- **런타임 주입도 XML**: `<system-alerts>`, `<market-data>`, `<past-experiences>`, `<active-mission>`, `<mission-context>`, `<task>`

#### claude_loop.py — 도구 결과 보존 + 진단 개선
- **`_strip_tool_blocks` 개선**: 기존에는 tool_use/tool_result 블록을 전부 삭제하고 `"(도구 실행 결과 생략됨)"` 플레이스홀더로 교체 → 도구 결과를 텍스트로 변환하여 보존 (`[도구 호출: name(input)]`, `[도구 결과: content]`)
- **에러 덤프 대상 변경**: `working_msgs` → `api_msgs` (실제 전송된 페이로드 덤프)

#### telegram_tools.py — execute_python 자동 import
- 봇이 생성한 코드에서 `import subprocess` 누락으로 `NameError` 반복 → 실행 코드 앞에 `import os, sys, json, subprocess, re` + `sys.path.insert(0, project_root)` 자동 주입

#### Hetzner 서버 — logviewer 계정 추가
- **목적**: Claude Code가 SSH로 서버 로그를 직접 조회
- **계정**: `logviewer@37.27.33.127`, rbash (restricted bash)
- **허용 명령**: journalctl, grep, cat, head, tail, less, ls, wc
- **SSH 키**: `~/.ssh/logviewer_key` (passphrase 없음)
- **앱 로그**: `/home/grass/leninbot/logs/` (setfacl 읽기 권한)

### 2026-03-20 — Agent Robustness Overhaul

#### db.py — Shared DB Layer
- **`query_one()` 추가**: `INSERT ... RETURNING` 패턴 지원 (race condition 제거)
- **stale connection recovery**: `get_conn()`에서 `SELECT 1` 헬스체크, 죽은 커넥션 자동 교체 (Supabase idle timeout 대응)

#### shared.py — Neo4j Driver Leak Fix
- **`_get_neo4j_sync_driver()` → context manager**: 예외 발생 시에도 `driver.close()` 보장
- 4개 호출처 (`fetch_kg_stats`, `kg_cypher`, `kg_delete_episode`, `kg_merge_entities`) 모두 `with` 블록으로 전환

#### telegram_bot.py — 구조 정리 + 버그 수정 (-220줄)
- **DB 레이어 중복 제거**: 자체 pool/query/execute 55줄 삭제 → `db.py` import로 대체
- **tool_result 처리 통합**: `_validate_tool_results` + `_ensure_tool_results` + `_force_fix_tool_results` 3중 함수(~260줄) → `_sanitize_messages` 1개(~90줄)로 통합
- **lazy model resolution**: import 시 blocking API 3회 → 첫 사용 시 1회만 실행
- **`_CLAUDE_MODEL_STRONG` 제거**: `_CLAUDE_MODEL`과 동일했던 중복 변수
- **continuation task race condition**: `INSERT` + `SELECT max(id)` → `INSERT ... RETURNING id` (원자적)
- **tool result 안전 가드**: None 방지 + 30KB 크기 제한 (API 400 에러 방지)

#### local_agent/agent.py — 동일한 안전 적용
- **`_sanitize_messages` 추가**: orphaned tool_use 블록 자동 수정 (매 API 호출 전 + final response 전)

### 2026-03-18 — Hetzner Migration + Telegram Bot Upgrade

#### Infrastructure: Render → Hetzner VPS
- **서버**: Hetzner VPS (Ubuntu 24.04, 16GB RAM) — Render 4GB 제한 탈피
- **HTTPS**: DuckDNS (`leninbot.duckdns.org`) + Nginx reverse proxy + Let's Encrypt
- **systemd 서비스**: `leninbot-api` (FastAPI port 8000), `leninbot-telegram` (aiogram)
- **차분 배포**: `deploy.sh` — git diff → pull → conditional pip install → systemctl restart
- **배포 알림 이중 경로**: deploy.sh가 curl로 Telegram 알림 + 새 봇이 deploy-meta.json 읽어 system_alert 주입
- **`/deploy` 명령**: 텔레그램에서 원격 배포 트리거 (setsid로 분리 실행)
- **파일**: `deploy.sh`, `setup-server.sh`, `systemd/leninbot-api.service`, `systemd/leninbot-telegram.service`, `doc/hetzner_deploy_guide.md`

#### Telegram Bot: Sonnet 4.6 + File System
- **Main LLM**: Claude Haiku 4.5 → **Claude Sonnet 4.6** (chat + task 통합)
- **Web Search**: Claude built-in (`web_search_20250305`) 도입 → 이후 2026-03-22에 Tavily 클라이언트로 재교체 (서버 도구 400 에러 문제)
- **파일 시스템 도구 추가**: `read_file`, `write_file`, `list_directory`, `execute_python` — 서버 직접 접근
- **`_chat_with_tools()` 업데이트**: `pause_turn` 지원

#### self_tools.py — Render → Hetzner 전환
- **제거**: `read_render_status`, `read_render_logs` (주석처리)
- **추가**: `read_server_logs` — journalctl로 telegram/api/nginx 서비스 로그 조회 (minutes_back, limit, grep 필터)

#### shared.py — Infrastructure Self-Knowledge
- **`MODULE_ARCHITECTURE`**: Hetzner VPS 인프라 정보 추가 (Ubuntu 24.04, 16GB, Nginx, /deploy)

### 2026-03-17 — Local PC Agent

#### local_agent/ (신규 패키지) — Windows 10 로컬 에이전트
- **역할 분리**: 로컬 = 개인 워크스테이션 (파일 접근, 크롤링, 보고서), 서버 = 중앙 저장소 + 공개 서비스. 공유하는 건 기억(메모리)뿐.
- **agent.py**: `claude_loop.chat_with_tools()` 위임. 예산 추적($0.50 기본), 다층 에러 복구, safety net 모두 공유 루프에서 상속. `max_rounds=15`.
- **tools.py**: 14개 로컬 도구 정의 (Anthropic API format). `manage_task`에 parent_task_id/scratchpad 지원. `mission` 도구 추가.
- **handlers.py**: 도구 핸들러 구현. `@_truncate_result` 데코레이터로 30KB 결과 크기 제한. `web_search`는 `tavily-python` 직접 사용.
- **crawler.py**: Playwright async 크롤링. Persistent Chromium context (쿠키 유지 → 로그인 세션). 결과를 SQLite `crawl_cache`에 자동 캐싱. JS 렌더링 페이지 지원.
- **sync.py**: 서버 push/pull. `shared.py` fetch 함수들과 `db.py` 직접 사용. KG 에피소드 쓰기, 태스크 보고서 저장.
- **local_db.py**: SQLite (`local_agent/data/local.db`). 테이블: `tasks`, `crawl_cache`, `conversations`, `chat_summaries`, `missions`, `mission_events`. 자동 마이그레이션.
- **cli.py**: 대화형 REPL. 특수 명령: `/quit`, `/clear`, `/tasks`, `/history`. 턴당/세션 비용 표시. 대화 압축 (20K 토큰 초과 시 Haiku 요약). 백그라운드 청크 요약.
- **self_tools 통합**: `from self_tools import SELF_TOOLS, SELF_TOOL_HANDLERS` → 서버 메모리 접근 도구 자동 추가.
- **기존 서버 코드 변경 없음**: shared.py, db.py, self_tools.py, claude_loop.py는 그대로 import만 함.
- **의존성 추가**: `playwright`, `tavily-python` (로컬 전용, Render에는 불필요)
- **실행**: `python -m local_agent`

### 2026-03-16 — Experiential Memory System

#### experience_writer.py (신규) — 경험 메모리 압축 및 축적
- **매일 00:30 KST 자동 실행**: api.py `_experience_scheduler()`로 스케줄링
- **3개 데이터 소스 수집**: chat_logs (web), telegram_chat_history, telegram_tasks (completed)
- **LLM 압축**: 24시간 활동을 3-8개 원자적 경험 항목으로 압축 (lesson/mistake/insight/pattern/observation)
- **pgvector 저장**: `experiential_memory` 테이블, BGE-M3 1024차원 임베딩
- **의미적 중복 제거**: 새 항목의 임베딩과 최근 30일 항목을 코사인 유사도 비교 (>0.85 = 중복 스킵)
- **이중 실행 방지**: period_end 기준 중복 체크

#### shared.py — 공통 경험 검색 함수
- **`search_experiential_memory(query, k=5)`**: pgvector 코사인 유사도 검색 (threshold 0.5). BGE-M3 lazy-load, `set_shared_embeddings()`로 기존 인스턴스 재사용 가능.
- chatbot.py, telegram_bot (self_tools), experience_writer 모두에서 공용

#### chatbot.py — 경험 메모리 검색 통합
- **`generate_node`**: academic/strategic 의도 시 `search_experiential_memory()` 호출 → `[PAST EXPERIENCE]` 섹션으로 프롬프트에 주입
- **`set_shared_embeddings(embeddings)`**: 초기화 시 BGE-M3 인스턴스를 shared에 등록 (중복 로딩 방지)

#### self_tools.py — `recall_experience` 도구 추가
- **도구 정의**: 의미 검색 쿼리 + limit 파라미터
- **`_exec_recall_experience`**: `shared.search_experiential_memory()` 호출, 유사도/카테고리/날짜 포맷팅

#### telegram_bot.py — Tool Strategy 업데이트
- "Past lessons/mistakes → recall_experience" 지침 추가

#### api.py — 스케줄러 추가
- **`_experience_scheduler()`**: 매일 00:30 KST에 `write_experiences()` 호출, lifespan에 등록

#### 테이블 스키마
```sql
experiential_memory: id, content, category, source_type, embedding(vector 1024), period_start, period_end, created_at
```

### 2026-03-16 — Diary Anti-Duplication Overhaul

#### diary_writer.py — 일기 중복 방지 전면 개편
- **`_extract_banned_topics()` 강화**: 3개→5개 일기 분석, 600자→1500자로 확대, 8-12개 구체적 테마/각도 추출 프롬프트로 교체
- **`_DIARY_PROMPT` 개선**: "STRICT BAN LIST" 강조, 뉴스가 금지 주제와 겹치면 스킵 또는 완전히 다른 각도 강제, 매 일기를 FRESH investigation으로 취급
- **`_WRITING_ANGLES` 삭제**: 8개 고정 앵글 배열 및 모든 참조(`_generate_diary`, `write_diary`) 제거 — banned_topics 강화로 충분하며 불필요한 주제 제약
- **업데이트 소비 추적 방식 변경**: 텍스트 매칭(영어 키워드 vs 한국어 일기 = 실패) → `diary_updates_consumed.json` 로컬 파일 기반 영속 추적
  - `_filter_unseen_updates()` → `(text, headers)` 튜플 반환
  - `_generate_diary()` → `(title, content, headers)` 3-튜플 반환
  - `_mark_updates_consumed()`: 일기 저장 성공 시에만 소비 기록
  - `.gitignore`에 런타임 상태 파일 추가

### 2026-03-16 — URL Content Fetching (shared skill)

#### shared.py — URL 유틸리티 3개 함수 추가 (공통 모듈)
- **`extract_urls(text)`**: 텍스트에서 HTTP/HTTPS URL 추출 (regex 기반)
- **`fetch_url_content(url, max_chars=10000)`**: URL에서 본문 최대 10,000자 추출. Tavily Extract 우선 시도, 실패 시 requests + BeautifulSoup fallback. 보일러플레이트 공격적 제거 (16개 class/id 패턴, 12개 HTML 태그, HTML 코멘트). 7단계 우선순위 본문 컨테이너 탐색 (article → main → article_body/post_body → content/article → id 매칭).
- **`fetch_urls_as_documents(urls, logs)`**: URL 리스트 → Document 객체 리스트 반환 (최대 3개). langchain Document 사용 가능 시 Document 객체, 없으면 dict fallback.

#### chatbot.py — URL 감지 + 자동 fetch
- **`AgentState`**: `url_documents: List[Document]` 필드 추가
- **`analyze_intent_node`**: 질문에서 URL 감지 시 즉시 `fetch_urls_as_documents()` 호출, `url_documents` state에 저장. URL이 있는데 `generate` 경로였으면 `vectorstore`로 전환 (분석 컨텍스트 확보).
- **`generate_node`**: `url_documents`를 `[USER-REFERENCED URL: ...]` 헤더와 함께 SOURCE MATERIAL 최상단에 배치 (전체 본문 전달, fetch 시점에서 이미 10,000자 제한).
- 기존 로컬 URL 함수 삭제 → `shared.py`에서 import

#### telegram_bot.py — `fetch_url` 도구 추가
- **`_TOOLS`**: `fetch_url` 도구 정의 추가 (Anthropic API format) — URL을 받아 본문 텍스트 반환
- **`_exec_fetch_url`**: `shared.fetch_url_content()`를 `asyncio.to_thread()`로 호출
- **시스템 프롬프트**: Tool Strategy에 "URL in message → fetch_url" 지침 추가

#### 효과
- 이전: URL이 포함된 질문에 대해 URL 텍스트만 보고 환각 생성
- 이후: 실제 페이지 본문을 최대 10,000자까지 읽고 분석. 웹 챗봇(자동), 텔레그램(도구 호출) 양쪽 지원.

### 2026-03-15 — Web Chatbot Self-Knowledge + Diary Update Dedup

#### chatbot.py — Self-Knowledge Access
- **`self_knowledge_tool` field**: Added to `QueryAnalysis` and `AgentState` — analyzer picks ONE specific self-tool per query (read_diary, read_chat_logs, read_system_status, read_recent_updates, or null)
- **`_fetch_self_knowledge(tool_name)`** (new): Dispatches to the selected shared.py function, returns formatted text for prompt injection
- **`generate_node`**: When `self_knowledge_tool` is set, fetches data and injects as `[SELF-KNOWLEDGE]` section in system prompt (works on all routing paths)
- **Query analysis prompt**: Extended with `self_knowledge_tool` field — 4 tool options with example trigger phrases

#### diary_writer.py — Update Dedup
- **`_filter_unseen_updates()`** (new): Cross-references update entry headers (date + title keywords) against previous diary content; returns only the first unseen update so each system update is mentioned in exactly one diary entry

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
