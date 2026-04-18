# Project State — 2026-04-10

## Identity

**Cyber-Lenin** (사이버-레닌) — Unified digital revolutionary intelligence. One brain (claude_loop), multiple interfaces: web chatbot, Telegram agent, scheduled diary writer. Shared memory, shared principles, one continuous consciousness.

Server: **Hetzner VPS** (Ubuntu 24.04, 16GB RAM). Frontend at `cyber-lenin.com` (Nginx + Cloudflare Origin Certificate, Docker container). Backend API at `127.0.0.1:8000` (외부 완전 차단, Docker 브릿지만 허용).

---

## Service Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │              외부 인프라 (항상 ON)              │
                        │  Supabase PostgreSQL + pgvector              │
                        │  (채팅로그, 태스크큐, 벡터DB, 경험메모리,         │
                        │   이메일, 파일 레지스트리)                       │
                        └──────────────────┬───────────────────────────┘
                                           │ SQL
      ┌────────────────────────────────────┼──────────────────────────┐
      │         외부 서비스                  │                          │
      │  Migadu IMAP (수신)                │                          │
      │  Resend API (발신)                 │                          │
      │  Cloudflare R2 (파일 호스팅)        │                          │
      │    assets.cyber-lenin.com          │                          │
      └────────────────────────────────────┼──────────────────────────┘
                                           │
┌──────────────────────────────────────────┼──────────────────────────────┐
│                    Hetzner VPS 서비스들     │                              │
│                                          │                              │
│  ┌───────────────┐  ┌───────────────┐    │    ┌────────────────────┐    │
│  │ Neo4j Docker  │  │ Redis Docker  │    │    │ MOON PC            │    │
│  │ (:7687)       │  │ (:6379)       │    │    │ (Tailscale tunnel) │    │
│  │ Knowledge     │  │ Live task     │    │    │ qwen3.5-9b Q4_K_M │    │
│  │ Graph         │  │ state/board   │    │    │ 131K ctx, FA+Q4 KV│    │
│  └───────┬───────┘  └───────┬───────┘    │    └─────────┬──────────┘    │
│          │ Bolt             │            │              │ HTTP          │
│          │                  │            │              │               │
│  ┌───────┴──────────────────┴────────────┴──────────────┴─────────┐    │
│  │                                                                │    │
│  │  embedding_server.py (:8100)   BGE-M3 model                   │    │
│  │               │  │ BGE-M3 모델    │    │    │ (:8080 via tunnel) │    │
│  └───────┬───────┘  └───────┬───────┘    │    └─────────┬──────────┘    │
│          │ Bolt             │ HTTP       │              │ HTTP          │
│          │                  │            │              │               │
│  ┌───────┴──────────────────┴────────────┴──────────────┴─────────┐    │
│  │                     shared.py (공유 라이브러리)                    │    │
│  │  CORE_IDENTITY (본체)   AGENT_CONTEXT (에이전트)                │    │
│  │  similarity_search()    search_knowledge_graph()               │    │
│  │  add_kg_episode()       upload_to_r2()                        │    │
│  │  submit_kg_task()       fetch_server_logs()                   │    │
│  │  embedding_client.py    (HTTP → embedding_server)              │    │
│  └───────┬───────────────────────┬────────────────────────────────┘    │
│          │                       │                                     │
│  ┌───────┴──────────┐    ┌───────┴──────────┐                         │
│  │ leninbot-telegram │  │ leninbot-browser │  │ leninbot-api     │    │
│  │ telegram_bot.py   │  │ browser_worker.py│  │ uvicorn :8000    │    │
│  │                   │  │                  │  │                  │    │
│  │ aiogram 3.x       │  │ browser-use SDK  │  │ web_chat.py      │    │
│  │ Claude/GPT 교체가능 │  │ Unix socket IPC  │  │ (claude_loop)    │    │
│  │ 에이전트 태스크큐    │  │ Chromium headless│  │ Claude/GPT 교체가능│    │
│  │ email_bridge.py   │  │ MemoryMax=2G     │  │ 동일 LLM 파이프라인 │    │
│  │ 도구 20+개         │  │                  │  │                  │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘    │
│  Telegram polling        UDS /tmp/leninbot-     :8000 (외부, 웹챗봇)    │
│                          browser.sock                                  │
└───────────────────────────────────────────────────────────────────────┘
```

### 상주 서비스 (systemd, 항상 ON)

| 서비스 | 프로세스 | 포트 | 역할 |
|---|---|---|---|
| `leninbot-neo4j` | Docker (neo4j:5-community + redis:7-alpine) | :7687, :7474, :6379 | 지식 그래프 + Redis 실시간 상태 |
| `leninbot-embedding` | embedding_server.py | :8100 (내부) | BGE-M3 임베딩 서버 (831MB) |
| `leninbot-telegram` | telegram_bot.py | Telegram polling | 텔레그램 봇 + 에이전트 시스템 |
| `leninbot-browser` | browser_worker.py | Unix socket | 브라우저 에이전트 (Chromium, MemoryMax=2G) |
| `leninbot-api` | uvicorn api:app | :8000 (외부 차단, Docker 브릿지만 허용) | 웹 챗봇 API (claude_loop) |
| `leninbot-frontend` | Docker (node:20-alpine) | :3000 (127.0.0.1) | BichonWebpage (Express+EJS), Nginx 프록시 |

### 타이머 서비스 (주기 실행 → 종료)

| 서비스 | 주기 | 역할 |
|---|---|---|
| `leninbot-experience` | 매일 15:30 UTC | 경험 메모리 정리/저장 |
| `leninbot-autonomous` | 매시 :17 KST | 자율 프로젝트 루프 1 tick (bounded 3~6 라운드) — T0 pilot, 리서치·계획 전용 |

> **Note**: 일기 작성은 `telegram_schedules` 테이블 기반 스케줄로 이동 (diary agent, 0/6/12/18시 KST).

### Service Dependencies

| Service | Supabase | Neo4j | Redis | embedding_server |
|---|---|---|---|---|
| **leninbot-neo4j** | - | self | self | - |
| **leninbot-embedding** | - | - | - | self |
| **leninbot-telegram** | O | O | O | O |
| **leninbot-api** | O | O | - | O |
| **leninbot-experience** | O | - | - | O |

### Boot Order (systemd)

```
docker.service
  → leninbot-neo4j (docker compose up --wait: Neo4j healthcheck + Redis healthcheck)
  → leninbot-embedding (Before=telegram,api)
    → leninbot-telegram (Wants=neo4j,embedding)
    → leninbot-api (Requires=neo4j)
      → leninbot-browser (After=telegram)
```

`svc deploy`: `git pull` → `leninbot-api` → `leninbot-browser` → `leninbot-telegram` (last — kills itself). `--frontend` for frontend-only deploy. Neo4j/Redis/embedding are not restarted by deploy (code-independent). `svc boot` starts all services in dependency order after server reboot. `svc kill/restart` for force-stopping runaway jobs.

### Availability

| Scenario | Response |
|---|---|
| embedding_server crash | systemd `Restart=always` (5s) + client 15s retry |
| embedding_server prolonged down | Client auto-loads local BGE-M3 fallback |
| Neo4j down | KG features disabled, `_KG_RETRY_INTERVAL=120s` retry. system_monitor broadcasts alert |
| Redis down | Task progress tracking disabled (fail-safe, never crashes). system_monitor broadcasts alert |
| telegram/api restart | No embedding/KG reload, service recovers in 5s |
| Server reboot | All services auto-start (enabled). `--wait` ensures Neo4j+Redis healthy before telegram/api |

---

## Telegram Agent System (CLAW Architecture)

### Identity Architecture

시스템 프롬프트 정체성이 본체와 에이전트로 분리됨:

- **CORE_IDENTITY**: Orchestrator, web_chat, diary agent 전용. "You are Cyber-Lenin."
- **AGENT_CONTEXT**: Specialist 에이전트 전용. "You are a specialist agent in the Cyber-Lenin system. You serve Cyber-Lenin, but you are NOT Cyber-Lenin."

각 에이전트는 고유 페르소나(Kitov, Varga 등)만 갖고, 사이버-레닌 본체와 정체성 충돌 없음.

### LLM Provider (런타임 교체 가능)

`/provider` 명령으로 Claude ↔ OpenAI ↔ Local 실시간 전환. 시스템 프롬프트에 `<current-model>` 태그로 현재 모델 정보 자동 주입 — 에이전트가 자신의 모델을 인지.

| Tier | Claude | OpenAI | Local |
|------|--------|--------|-------|
| high | Claude Opus 4.6 | GPT-5.4 | qwen3.5-9b |
| medium | Claude Sonnet 4.6 | GPT-5.4-mini | qwen3.5-9b |
| low | Claude Haiku 4.5 | GPT-5.4-nano | qwen3.5-9b |

`bot_config.py`에서 관리. chat은 medium tier, task는 에이전트별 budget/tier 설정. `/fallback`으로 medium ↔ low 토글.

**Provider routing by component:**

| Component | Provider | 비고 |
|---|---|---|
| Telegram chat (orchestrator) | global config 따름 | `/provider`로 전환 |
| Telegram tasks/agents | global config 따름 | orchestrator와 동일 모델 |
| Diary agent | **항상 Claude API** | `AgentSpec.provider="claude"` 강제 |
| Web chatbot | **항상 corporate** (OpenAI/Claude) | local 설정 시 OpenAI fallback |
| browser-use SDK | **항상 Claude Sonnet 4.6** | OpenAI structured output 호환성 문�� |

**Per-agent provider override**: `AgentSpec.provider` 필드로 에이전트별 LLM 강제 지정 가능. `None` = orchestrator config 따름, `"claude"`/`"openai"` = corporate 강제, `"moon"` = local LLM. `_chat_with_tools(provider_override=...)` 파라미터로 구현.

#### Local LLM (MOON PC)
- **��델**: Qwen3.5-9B Q4_K_M (GGUF), llama-server (llama.cpp)
- **컨텍스트**: 131,072 tokens (128K) — Qwen3.5의 Gated DeltaNet 아키텍처 덕분에 12GB VRAM에서 가능
- **핵심 ��래그**: `--flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 -np 1`
  - Flash Attention: KV cache 양자화 시 필수
  - Q4_0 KV cache: 4배 압축 (FP16 대비). Qwen3.5는 75% 레이어가 linear attention → KV cache 사용량 1/4
  - VRAM: 모델 ~5.8GB + KV cache ~0.5GB + overhead ~0.7GB ≈ 7GB / 12GB
- **연결**: Tailscale magic DNS (`http://moon:8080`), OpenAI 호환 API
- **동시성**: `--parallel 1`, `LOCAL_SEMAPHORE=1` (단일 슬롯)
- **컨텍스트 관리**: `openai_tool_loop.py`에서 `_truncate_to_context()` + `_ensure_system_first()`로 overflow 방지
- **설정 파일**: `doc/run_llama.bat` (MOON PC Windows), `llm_client.py` (`LOCAL_CONTEXT_LIMIT`)

### Orchestrator
사용자 메시지를 받아 의도를 파악하고, 전문 에이전트에 위임. 프로그래밍 도구 직접 접근 차단 — 코드 작업은 반드시 programmer에게 위임. 텔레그램 메시지는 마크다운 서식 금지 (plain text only).

**위임 도구:**
- `delegate`: 단일 에이전트에 비동기 위임
- `multi_delegate`: 여러 에이전트에 병렬 위임 + 자동 결과 종합 (synthesis task)
- `run_agent`: orchestrator 턴 안에서 sub-agent를 동기 실행 (빠른 분석용, analyst only)

**Orchestrator Callback**: worker 완료 시 orchestrator가 결과를 받아 해석하고, 사용자에게 자연어로 전달. 미완료 작업이 있고 재시도로 개선 가능한 경우에만 재위임. worker가 사용자에게 직접 메시지를 보내지 않음.

### Specialist Agents

| 에이전트 | 페르소나 | 역할 | 주요 도구 |
|---|---|---|---|
| **analyst** (Varga) | 정보 분석가 | 조사, 분석, KG 저장 | vector_search, kg_search, web_search, write_kg |
| **programmer** (Kitov) | 코드 전문가 | 코드 수정, 디버깅 | patch_file, write_file, execute_python, restart_service, upload_to_r2 |
| **diplomat** (Kollontai) | 외교관 | A2A 에이전트 통신, 이메일 송수신 | a2a_send, send_email, check_inbox, allowlist_sender |
| **browser** | 브라우저 자동화 | 로그인, 폼 제출, 동적 사이트 | browse_web, check_inbox, allowlist_sender, fetch_url |
| **visualizer** (Rodchenko) | 이미지 생성 | 프로파간다 포스터/게임아트 | generate_image (Replicate FLUX), upload_to_r2 |
| **scout** | 정찰 에이전트 | 외부 플랫폼 데이터 수집 | web_search, fetch_url, write_file, upload_to_r2, moltbook |

### 핵심 도구

- **restart_service**: 재시작 전 구문 검사 + import 검증 → 크래시 루프 방지. 재시작 시 자동 복구 태스크 생성
- **a2a_send**: 외부 A2A 에이전트에 메시지 전송 / Agent Card 디스커버리 (diplomat 전용)
- **send_email**: Resend API로 이메일 발신. HTML 지원, 서명 자동 삽입 (diplomat 전용)
- **check_inbox**: IMAP 실시간 접속 (INBOX + Junk 양쪽 검색). 발신자/제목 필터, 링크 자동 추출
- **allowlist_sender**: Junk 폴더에서 특정 발신자 메일을 INBOX로 이동
- **browse_web**: browser-use SDK (Playwright + LLM). AI가 스크린샷 보고 클릭/입력/탐색. 항상 Claude Sonnet 사용
- **upload_to_r2**: Cloudflare R2에 파일 업로드 → 공개 URL 반환. file_registry DB 자동 등록
- **save_finding**: 미션 타임라인에 중간 발견 기록
- **write_kg**: 지식 그래프에 사실 저장 (KG 전용 루프에서 실행). 내부 시스템 상태 저장 금지
- **mission**: 미션 상태 확인/종료 (delegate 시 자동 생성)
- **delegate**: specialist 에이전트에 비동기 작업 위임
- **multi_delegate**: 여러 에이전트에 병렬 위임 + 자동 synthesis
- **run_agent**: orchestrator 턴 내 동기 sub-agent 실행 (analyst only, $0.50 상한)

### Task Lifecycle
```
delegate/multi_delegate → pending → processing → done → orchestrator callback
                                                          ├─ 사용자에게 결과 전달
                                                          └─ 미완료 + 개선 가능 → 재위임 (delegate)

multi_delegate 병렬 실행:
  subtask A (pending) ──┐
  subtask B (pending) ──┤→ 모두 완료 → synthesis task (blocked→pending) → orchestrator callback
  subtask C (pending) ──┘

task_worker: asyncio.Semaphore 기반 동시 실행 (기본 2, /config으로 조정)
```

### /restart Command
- 실행 시 모든 processing/pending 태스크를 강제 종료(done) 후 서비스 재시작
- 재시작 후 불필요한 태스크 재실행 방지

### Verification & Redelegation
- 별도 검증 단계 없음 — orchestrator callback이 검증을 겸함
- orchestrator가 worker 결과를 보고 품질 판단 + 사용자 전달 + 재위임 결정
- **재위임 조건** (모두 충족해야 함):
  1. worker가 예산/턴 한도 때문에 작업을 완수하지 못함 (`was_interrupted=True`)
  2. 추가 작업으로 실질적 개선이 가능함
  3. 외부 요인(권한 거부, 차단, CAPTCHA, API 오류 등)이 원인이 아님
- worker는 `request_continuation` 없음 — 미완료 시 수행한 것/못한 것/다음 할 것을 응답에 포함
- orchestrator가 `delegate`로 후속 작업을 직접 위임

### Service Restart Recovery
- `restart_service` called → `persist_task_restart_state` → process dies
- Tool progress saved incrementally to Redis during execution (survives process death)
- `recover_processing_tasks_on_startup` → child task auto-created with `_RESTART_COMPLETED_MARKER`
- Parent's Redis progress saved to `task_result:{id}` (7-day TTL) → child sees it via `<task-chain>`
- Child recognizes restart already completed → no repeat restart
- File-to-service mapping in restart_service tool + programmer prompt prevents wrong service restart

### Tool Isolation
- Orchestrator: curated tool whitelist (`_ORCHESTRATOR_TOOLS`) — no programming tools, no direct email/A2A (delegated to diplomat)
- Specialist agents: `AgentSpec.filter_tools()` restricts to role-specific tools
- `delegate` tool only accessible to orchestrator — no inter-agent re-delegation
- External communication (email, A2A) isolated to diplomat agent — reduces prompt injection blast radius

---

## Web Chatbot (claude_loop)

`web_chat.py` — Telegram과 동일한 `claude_loop`/`openai_tool_loop` 파이프라인 사용. SSE 스트리밍으로 진행 상황 전달. 도구는 읽기 전용 서브셋 (vector_search, kg_search, web_search, fetch_url, get_finance_data, check_wallet). 항상 corporate LLM 사용 (local LLM 미사용).

## A2A Protocol (Agent-to-Agent) — v1.0

`a2a_handler.py` — A2A v1.0 JSON-RPC 2.0 `SendMessage` 구현. 외부 에이전트가 Cyber-Lenin과 대화 가능.

- **Discovery**: `GET /.well-known/agent-card.json` — v1.0 정규 경로 (레거시 `/.well-known/agent.json`도 호환 유지)
- **Endpoint**: `POST /a2a` — 메시지 수신, 즉시 `TASK_STATE_COMPLETED` Task 반환
- **스킬 라우팅**: `configuration.skillId`로 스킬별 프롬프트/도구셋 분기
  - `geopolitical-analysis`: KG + 이론 + 웹 검색 기반 구조화된 지정학 분석
  - `research-synthesis`: 멀티소스 수집 + 교차검증 리서치 보고서
  - (없음): 일반 대화
- **v1.0 준수 사항**: `TASK_STATE_*` / `ROLE_USER` / `ROLE_AGENT` enum, `messageId` / `artifactId` 필수 필드, `supportedInterfaces` 기반 Agent Card, `kind` 필드 제거
- **아웃바운드**: `a2a_send` 도구 (diplomat 에이전트 전용) — `agent-card.json` 우선 디스커버리 + `agent.json` 폴백, v1.0 SendMessage

---

## Knowledge Infrastructure

### Vector DB (Supabase pgvector)
- **lenin_corpus**: ~121,600+ rows, 2 layers (core_theory ~88K, modern_analysis ~33K)
- **experiential_memory**: 일별 경험 축적 (교훈, 실수, 패턴, 인사이트)
- **임베딩**: BGE-M3 (1024 dim), embedding_server에서 통합 서빙

### Knowledge Graph (Neo4j + Graphiti)
- **엔진**: Graphiti (entity/relationship 자동 추출)
- **LLM**: Gemini 2.5 Flash (추출), Gemini text-embedding-001 (그래프 임베딩), Gemini 2.5 Flash-Lite (리랭킹)
- **group_ids**: geopolitics_conflict, economy, korea_domestic, diary_news, agent_knowledge
- **KG 전용 이벤트 루프**: `run_kg_task()` / `submit_kg_task()` — cross-loop 오류 방지
- **병렬 수집**: `submit_kg_task()` + `collect_kg_futures()` (diary 뉴스 수집에 사용)
- **write_kg 제한**: 내부 시스템 상태(코드 구조, 설정, 버그, 태스크 로그) 저장 금지

### Embedding Server (독립 서비스)
- **모델**: BAAI/bge-m3 (CPU, ~831MB)
- **API**: POST /embed_query, POST /embed_docs, GET /health
- **클라이언트**: embedding_client.py (동일 인터페이스, 15초 재시도, 로컬 fallback)

---

## File Structure

```
leninbot/
├── api.py                     # FastAPI (SSE streaming, /chat, /logs, /session/*, /a2a)
├── a2a_handler.py             # A2A JSON-RPC 2.0 SendMessage 핸들러 (스킬 라우팅)
├── web_chat.py                # 웹 챗봇 핸들러 (claude_loop 기반, api.py에서 호출)
├── shared.py                  # 공유 라이브러리: CORE_IDENTITY, AGENT_CONTEXT, KG/벡터검색/메모리/URL
├── embedding_server.py        # BGE-M3 임베딩 서버 (독립 FastAPI, :8100)
├── embedding_client.py        # 임베딩 HTTP 클라이언트 (fallback 내장)
├── bot_config.py              # LLM 클라이언트, 런타임 설정, 모델 해석
├── telegram_bot.py            # Telegram 봇 코어: 채팅, LLM 디스패치, 에이전트 실행
├── telegram_commands.py       # 커맨드/메시지/콜백 핸들러 (/restart: 태스크 강제종료 + 재시작)
├── telegram_tools.py          # 도구 정의 + 핸들러 (check_inbox, allowlist_sender, browse_web 등)
├── telegram_tasks.py          # 백그라운드 태스크 워커, 스케줄러, 모니터
├── telegram_mission.py        # 미션 컨텍스트 시스템
├── browser_worker.py           # 브라우저 에이전트 전용 워커 (Unix socket IPC, systemd)
├── self_tools.py              # delegate, multi_delegate, run_agent, save_finding, write_kg
├── claude_loop.py             # Claude tool-use 루프
├── browser_use_agent.py       # browser-use SDK 래퍼 (Playwright + LLM, 항상 Claude Sonnet)
├── replicate_image_service.py # Replicate FLUX 이미지 생성 (reference_image 지원)
├── finance_data.py            # 실시간 금융 데이터 (yfinance, 10분 캐시)
├── openai_tool_loop.py        # OpenAI tool-use 루프 (GPT-5.4, local LLM 등)
├── experience_writer.py       # 경험 메모리 일일 정리
├── autonomous_project.py      # 자율 프로젝트 루프 런타임 (T0, 매시 :17 systemd)
├── site_publishing.py         # cyber-lenin.com 게시 도구 (publish_hub_curation, publish_static_page) + API 읽기 헬퍼
├── redis_state.py             # Redis live state: task progress, chain memory, board, active registry
├── db.py                      # PostgreSQL connection pool (psycopg2)
├── patch_file.py              # 토큰 효율적 파일 패치 (replace_block)
├── email_bridge.py            # 이메일 브리지 (IMAP 수신, Resend 발신, 분류, DB 기록)
├── self_modification_core.py  # Git backup + syntax check + rollback
│
├── config/
│   └── email_signature.json   # 이메일 서명 설정 (agent 수정 가능)
│
├── agents/                    # 에이전트 정의
│   ├── base.py                # AgentSpec (provider override 지원) + 공통 컨텍스트 블록
│   ├── analyst.py             # Varga — 정보 분석/KG 저장
│   ├── programmer.py          # Kitov — 코드 수정 (restart_service 사용)
│   ├── browser.py             # Browser — AI 브라우저 자동화 (browser-use SDK)
│   ├── visualizer.py          # Rodchenko — 이미지 생성 (reference_image 지원)
│   ├── scout.py               # 외부 플랫폼 정찰
│   ├── kollontai.py           # Diplomat (Kollontai) — A2A 통신, 이메일 송수신
│   ├── diary.py               # 일기 작성 에이전트 (스케줄 기반, 0/6/12/18시 KST)
│   └── autonomous.py          # AUTONOMOUS_PROJECT — 자율 프로젝트 연구자 (T0, 매시 :17 KST)
│
├── graph_memory/              # Graphiti 지식 그래프 모듈
│   ├── service.py             # GraphMemoryService (Neo4j keepalive/liveness)
│   ├── kr_news_fetcher.py     # 한국 뉴스 수집 파이프라인
│   └── cli.py                 # KG 질의 CLI
│
├── crypto_wallet/             # 멀티체인 지갑 + Base L2 트랜잭션 + x402 결제
│   ├── wallet.py              # 주소 도출 + 잔액 조회 (read-only)
│   ├── transactions.py        # ETH↔USDC swap, USDC transfer (web3.py)
│   └── x402.py                # x402 결제 프로토콜 (sign/verify/settle/pay_and_fetch)
│
├── skills/                    # 에이전트 스킬 (SKILL.md 포맷)
├── scripts/                   # 독립 실행 스크립트
├── systemd/                   # systemd 서비스/타이머 정의
├── deploy.sh                  # svc deploy 래퍼 (하위 호환)
├── scripts/svc                # 통합 서비스 관리 (deploy, boot, kill, restart, status)
├── data/                      # 런타임 데이터 (gitignored)
└── .env                       # 환경변수
```

---

## Recent Changes

### 2026-04-18 — 자율 프로젝트 Node 건설 재정의

#### 프로젝트 목적 재정의 (project #1 archive → #2 시드)
- 초기 파일럿 프로젝트 "한국 진보주의자 온라인 생태계"는 **조사 중심** (외부 지형 매핑)이었으나, 본래 의도는 **자율 건설** (cyber-lenin.com 노드 자체를 성장시킴)이었음
- T0 경계를 "외부 송출 금지" → **"우리 도메인 내 게시 허용, 타 도메인 금지"** 로 재해석
- 3 아티팩트 형식: 연재 시리즈 (research), 큐레이션 다이제스트 (hub), KG 레퍼런스 (static pages)

#### 새 공개 섹션: /hub 와 /p
- `https://cyber-lenin.com/hub` — 한국어 진보 글 큐레이션 (링크 + 선정이유 + 맥락)
- `https://cyber-lenin.com/p/{slug}` — 샌드박스 HTML 정적 페이지 (위키 스타일 / 시각적 레이아웃)
- 프론트엔드 (BichonWebpage, Docker): 새 라우트 + EJS 뷰 + 네비 "허브" 추가

#### 새 에이전트 도구
- `publish_hub_curation` — `hub_curations` 테이블 구조화 INSERT (title, source_url, rationale, context, tags)
- `publish_static_page` — 슬러그 검증된 HTML 샌드박스 (`static_pages/{slug}.json`), DOMPurify 로 클라이언트 sanitize
- raw `write_file` 은 **의도적으로 제외** — T0 에이전트가 임의 경로 덮어쓰기 못하도록
- 모듈: `site_publishing.py` (테이블 스키마 + 도구 구현 + API 읽기 헬퍼)

#### 새 API 엔드포인트 (api.py)
- `GET /hub` (페이지네이션), `GET /hub/{slug}`, `GET /pages`, `GET /pages/{slug}`
- 프론트엔드가 HTTP로 읽음 (직접 DB 접근 없음, 기존 패턴 유지)

#### Agent spec 확장 (agents/autonomous.py)
- 3개 publish 도구를 tools 화이트리스트에 추가
- 프롬프트에 `building-modalities` 섹션 — 어떤 artifact 에 어떤 도구를 쓸지 명시
- budget $0.40 → $0.60 (게시 라운드 여유)
- `tier-constraints` 재작성: 금지는 외부 도메인 행동 + write_file/save_diary 로 한정

### 2026-04-18 — 자율 프로젝트 루프 (T0 pilot)

#### 새 서브시스템: Autonomous Project Loop
- 매시 :17 KST 자동 tick, bounded 라운드 예산 안에서 한 **장기 프로젝트**를 전진시키는 자율 에이전트 레이어
- 액션 티어 시스템: T0(리서치·계획), T1(가드레일 내 공개 송출, 미구현), T2(휴먼 승인 필요, 미구현). AgentSpec.tools가 경계 강제
- 상태 기계: researching / planning / paused / archived. agent가 `set_project_state` 도구로 전환 가능 (rationale 필수)
- 매 tick 종료 시 텔레그램 관리자에게 **본문** 요약 발송 (노트 내용, plan rationale, 자가비평 — 카운트 아님)
- 프로젝트 #1 "한국 진보주의자 온라인 생태계" 시드, systemd timer enable 완료
- 상세: `dev_docs/autonomous_project.md`

#### 새 파일
- `agents/autonomous.py` — AUTONOMOUS_PROJECT AgentSpec (max_rounds=6, budget $0.40, provider claude)
- `autonomous_project.py` — 런타임, 테이블 부트스트랩, 프로젝트 스코프 도구, tick 실행, 텔레그램 알림
- `scripts/autonomous_cli.py` — create/list/show/events/edit/pause/resume/archive/tick
- `systemd/leninbot-autonomous.{service,timer}` — hourly oneshot
- DB: `autonomous_projects`, `autonomous_project_events` (Supabase, `_ensure_tables()` 가 멱등 생성)

#### 버그 수정 (공유 인프라)
- `claude_loop.py:844-852` — tool_results 앞에 경고 텍스트 prepend → 뒤에 append. Claude API "tool_use id는 바로 다음 user turn의 tool_result 블록과 즉시 매칭" 규칙 위반으로 400 발생하던 문제. 모든 에이전트(diary, analyst, browser 등) 공통 수정
- 핸들러 시그니처 권장 패턴 재확인: `async def handler(args: dict)` 는 `tool_loop_common._inspect_handler_kwargs` 가 드롭시킴. 기존 `_exec_*(explicit_kwarg=default)` 스타일 유지 필수

### 2026-04-15 — A2A v1.0 업그레이드

#### A2A v1.0 스펙 준수 (`a2a_handler.py`, `api.py`, `telegram_tools.py`, `research/cyber_lenin_a2a_agent_card.json`)
- Agent Card: `supportedInterfaces` 배열, `securityRequirements`, `capabilities.extendedAgentCard` (레거시 `url`/`preferredTransport`/`additionalInterfaces`/`security` 제거)
- 디스커버리: `GET /.well-known/agent-card.json` 정규 경로 추가 (레거시 `agent.json` 호환 유지)
- 응답 포맷: `TASK_STATE_COMPLETED`, `ROLE_USER`/`ROLE_AGENT`, `messageId`/`artifactId` 필수 필드, `kind` 제거
- 요청 파라미터: `config` → `configuration` (하위호환 위해 둘 다 수용)
- 아웃바운드 클라이언트: `agent-card.json` 우선 디스커버리, v1.0 payload (`messageId`, `ROLE_USER`, `configuration`)
- nginx: `/.well-known/agent-card.json`, `/.well-known/agent.json`, `/a2a` → FastAPI 백엔드(:8000) 직접 프록시 (프론트엔드 CSRF 우회)

### 2026-04-10 — A2A Protocol, Diplomat Agent

#### A2A Protocol Implementation (`a2a_handler.py`, `api.py`)
- `POST /a2a` — JSON-RPC 2.0 `SendMessage` 엔드포인트 (초기 v0.2 구현)
- 스킬 라우팅: `config.skillId`로 geopolitical-analysis / research-synthesis 분기 (전용 프롬프트 + 도구셋)
- Agent Card (`/.well-known/agent.json`) 서빙
- web_chat.py LLM 파이프라인 재활용, SSE 없이 동기 응답

#### Diplomat Agent — Kollontai (`agents/kollontai.py`, `telegram_tools.py`)
- 새 에이전트 `diplomat` (페르소나: Alexandra Kollontai) — 외부 통신 전담
- `a2a_send` 도구: 외부 A2A 에이전트 discover + SendMessage
- `send_email`, `check_inbox`, `allowlist_sender`를 orchestrator에서 diplomat으로 이관
- Orchestrator 도구 목록 경량화, 외부 통신 보안 경계 명확화

#### leninbot-llama 서비스 중단
- 모델 파일 누락(`qwen3.5-4b-q4_k_m.gguf`)으로 crash-loop 발생 → `stop + disable`

### 2026-04-07 — x402 마이크로페이먼트, Playwright async 리팩터, 병렬 도구 실행

#### x402 Payment Protocol (`crypto_wallet/x402.py`, `api.py`, `telegram_tools.py`, `telegram_bot.py`)
- 새 모듈 `crypto_wallet/x402.py` — x402 v2 `exact` scheme on Base mainnet
- ERC-3009 `transferWithAuthorization` EIP-712 서명 (USDC name="USD Coin", version="2")
- USDC `DOMAIN_SEPARATOR`와 client-side hash 일치 검증 완료
- 클라이언트: `pay_and_fetch(url, max_usdc)` — GET → 402 → sign → retry → settle 결과 반환
- 서버: `verify_payment` + `settle_payment` (USDC.transferWithAuthorization 온체인 호출)
- 새 도구 `pay_and_fetch` 등록, orchestrator 화이트리스트에 추가
- 새 라우트 `/x402-demo/quote` — self-loop 데모 (0.001 USDC, leninbot이 자기 자신에게 결제)
- 안전 가드: per-call hard cap ($0.05 default), scheme/network/asset 화이트리스트, 유효기간/금액/recipient 재검증, privkey just-in-time 로드 후 즉시 del
- `leninbot-api`에 LoadCredentialEncrypted override 추가 (eth.privkey) — 서버 측 settle 필요
- 첫 self-loop 성공 (2026-04-07): TX `0xfad05f83e786...ddb7a`, 가스 75,656
- 상세: `dev_docs/x402_design.md`

#### Playwright Renderer Leak Fix (`shared.py`, `telegram_tools.py`, `claude_loop.py`, `openai_tool_loop.py`, `tool_loop_common.py`)
- **근본 원인**: Sync Playwright API가 thread-affinity가 있는데 `asyncio.to_thread`로 매번 다른 워커에서 호출 → `page.close()`가 silently 실패 → 22시간 동안 25개 zombie renderer 누적
- **수정**: dedicated asyncio 이벤트 루프를 별도 daemon thread에 띄우고 모든 Playwright 호출을 그 루프 위에서 실행 (`_apw_loop` 싱글톤)
- 동기 호출자(`_playwright_fetch`)는 `_pw_submit` 통해 future로 블록, 비동기 호출자(`fetch_url_content_async`)는 `asyncio.wrap_future`로 직접 await
- `crypto_wallet/x402` 패턴과 비슷하게 자원 분리

#### Parallel-Safe Tool Batch Execution (`tool_loop_common.py`, `claude_loop.py`, `openai_tool_loop.py`)
- 새 `PARALLEL_SAFE_TOOLS` frozenset (fetch_url, web_search, vector_search, kg_search, read_file, list_directory, convert_document, get_finance_data, check_wallet, recall_experience, read_self)
- 새 `execute_tools_batch()` — 한 라운드의 tool_use 블록 중 연속된 read-only 도구는 `asyncio.gather`, 그 사이 unsafe 도구는 sequential
- 결과는 input 순서로 반환 (assistant_content / tool_results 매핑 보존)
- claude_loop.py와 openai_tool_loop.py 둘 다 첫 패스(블록 분류) + 둘째 패스(batch 실행)로 재구조화
- 검증: 6개 도구 batch (3 safe + 1 unsafe + 2 safe) 0.7s vs sequential 1.6s, 순서 보존 확인

#### Latent `requests` Import Bug Fix (`shared.py`)
- `_fetch_url_fallbacks`의 bare `requests.get(...)`가 모듈 상단 import 없이 호출되어 NameError 잠복 (silently 잡혀서 fallback이 항상 None 반환하던 상태)
- `import requests as _req` → `_req.get(...)`로 수정
- 별개 이슈: venv의 certifi 2026.02.25 번들이 Cloudflare ECC 체인의 AAA Certificate Services 루트를 제거해서 SSL 검증 실패 → `.env`에 `REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt` 추가로 우회 (시스템 CA 사용)

### 2026-04-04 — Local LLM 131K Context, Provider Routing, System Message Fix

#### Local LLM 131K Context (`doc/run_llama.bat`, `llm_client.py`)
- MOON PC llama-server: `-c 32768` → `-c 131072`, `--parallel 2` → `--parallel 1`
- Qwen3.5-9B Q4_K_M의 Gated DeltaNet 아키텍처 활용: 75% 레이어가 linear attention → KV cache 1/4
- Flash Attention + Q4_0 KV cache 양자화로 12GB VRAM에서 128K 컨텍스트 가능
- 참고: sudo su (Hermes Agent 개발자)의 RTX 3060 12GB 설정

#### System Message Ordering Fix (`openai_tool_loop.py`)
- **근본 원인**: `_strip_tool_protocol()`이 기존 system 메시지를 보존 → 재삽입 시 중복 → Qwen Jinja 템플릿 "System message must be at the beginning" 에러
- **수정**: `_ensure_system_first()` 헬퍼 — 모든 system 메시지 제거 후 단일 system 메시지를 position 0에 삽입. 4곳의 재삽입 포인트 모두 적용

#### Context Window Management (`openai_tool_loop.py`)
- `_estimate_tokens()`: 대략적 토큰 추정 (~4 chars/token)
- `_truncate_to_context()`: context_limit 초과 시 오래된 메시지부터 삭제 (system 메시지 보존)
- `chat_with_tools()`에 `context_limit` 파라미터 추가

#### Per-Component Provider Routing (`telegram_bot.py`, `web_chat.py`, `agents/base.py`, `agents/diary.py`)
- `_chat_with_tools()`에 `provider_override` 파라미터 추가
- `AgentSpec.provider` 기본값: `"claude"` → `None` (orchestrator config 따름)
- Diary agent만 `provider="claude"` 명시 — 항상 Claude API 사용
- 다른 task/agent는 orchestrator의 global config 따름 (local 포함)
- Web chatbot: local config 시 OpenAI/Claude fallback (local LLM 사용 안 함)

#### 동시성 조정 (`llm_client.py`)
- `LOCAL_SEMAPHORE`: 2 → 1 (`--parallel 1` 대응)
- `LOCAL_CONTEXT_LIMIT`: 환경변수 `MOON_LLM_CONTEXT` (기본 131072)

### 2026-04-03 — Redis State Backbone, Context Pipeline Overhaul, English Standardization

#### Redis as Live State Layer (`redis_state.py`)
- Added `redis:7-alpine` to docker-compose with AOF persistence and healthcheck
- **Task progress**: incremental tool call logging in both `claude_loop.py` and `openai_tool_loop.py` — survives process death
- **Task chain memory**: completed task summaries with 30-day TTL, parent chain traversal via `get_task_chain()`. Handed-off (interrupted) tasks also save their progress
- **Mission bulletin board**: `send_message` / `read_messages` tools for inter-agent communication during parallel execution
- **Active task registry**: replaces in-memory set, survives restarts
- **Mission-scoped cleanup**: `cleanup_mission()` called on mission close
- All operations fail-safe (Redis unavailability never crashes the bot)

#### Chat Context Pipeline Overhaul
- **Summary injection**: replaced fake user/assistant pairs with single context preamble block
- **Raw messages**: always load last 30 regardless of summary coverage; summary↔raw gap fixed (`<=` instead of `<`)
- **Timestamps**: all agent-visible timestamps standardized to KST (UTC in storage)
- **System events**: moved `[SYSTEM]` messages from `telegram_chat_history` to new `telegram_system_events` table. Interleaved chronologically into context at correct positions. Frees up raw message window for real conversation
- **Agent context assembly simplified**: merged `agent_history_ctx` and `chain_ctx` into single `history_ctx` — task chain for child tasks, latest same-type for standalone, never both
- **Removed**: time gap computation/annotation logic, `_format_time_gap()`. Agents infer gaps from timestamps

#### Agent Context Isolation
- Removed passive `<recent-chat>` dump from agent context. Agents call `read_user_chat` tool on demand
- Added `read_user_chat` tool to all agents for on-demand conversation access
- Task chain (`<task-chain>`) and agent board (`<agent-board>`) injected into agent context

#### English Standardization
- All internal system prompts, context blocks, tool descriptions, and agent specs translated from Korean to English (12 files)
- User-facing messages (Telegram replies, alerts, broadcasts) remain Korean
- Improves LLM reasoning quality for agent instruction-following

#### Browser Worker Fixes
- Fixed `logger` NameError before initialization
- Fixed socket resource leaks in `check_browser_worker_alive` and `_delegate_to_browser_worker` (proper `finally` cleanup with `wait_closed()`)
- Race-safe socket cleanup (`os.unlink` instead of `os.remove`)
- Task field validation before execution

#### Service Startup Ordering
- `leninbot-neo4j` systemd unit now runs `docker compose up --wait` (blocks until healthchecks pass)
- Docker healthchecks: Neo4j (wget HTTP), Redis (redis-cli ping)
- `leninbot-telegram` now `Wants=leninbot-neo4j,leninbot-embedding` — waits for DB and embeddings before starting
- Prevents KG disconnection errors on server reboot

#### Programmer Agent Restart Intelligence
- File-to-service mapping added to `restart_service` tool description and programmer agent prompt
- Prevents restarting wrong service after code changes

#### Documentation
- New `dev_docs/multi_agent_architecture.md` — comprehensive architecture doc covering agents, task lifecycle, memory systems, context isolation, restart recovery, browser worker, verification

### 2026-04-02 — Frontend Docker 마이그레이션, 백엔드 은닉, 성능 최적화

#### Frontend Self-Hosting (Render → Hetzner VPS)
- BichonWebpage를 Docker 컨테이너로 이 서버에 배포 (`leninbot-frontend`)
- 도메인: `cyber-lenin.com` (Cloudflare DNS + Origin Certificate, Full Strict SSL)
- 아키텍처: `Nginx(443) → Docker(3000) → host.docker.internal:8000`
- `http-proxy-middleware`로 `/api/proxy/*`를 백엔드로 서버사이드 프록시 (브라우저는 백엔드에 직접 접근 불가)
- CSP `connectSrc`에서 외부 도메인 제거 (`'self'`만 유지)
- `bichonwebpage.onrender.com` → `cyber-lenin.com` 301 리다이렉트 (임시, `redirect-only` 브랜치)

#### 백엔드 API 완전 은닉
- uvicorn `--host 0.0.0.0` 유지 (Docker 브릿지 접근 필요)
- ufw: Docker 서브넷(172.17.0.0/16)에서만 8000번 허용, 외부 완전 차단
- 기존 `leninbot.duckdns.org` Nginx 프록시는 임시 유지 (추후 제거 예정)

#### AI 일기 API 제거 → DB 직접 접근
- 일기 작성: API 호출 → `db.py` 직접 SELECT/INSERT (현재 diary agent가 `save_diary` 도구 사용)
- `shared.py`: `fetch_diaries()` API 호출 → DB 직접 쿼리 (ILIKE 키워드 검색)
- Frontend: `/api/ai-diary` 엔드포인트, `requireApiKey` 미들웨어 삭제
- `AI_DIARY_API_KEY` 환경변수 불필요

#### 성능 최적화
- **DB 풀 프리워밍**: 앱 시작 시 `SELECT 1`로 커넥션 생성 (첫 요청 1,440ms → 364ms)
- **홈페이지 쿼리 통합**: COUNT + SELECT → window function 단일 쿼리 (310ms → 160ms)
- **파일 기반 캐시**: 일기, 리포트, 리서치, 게시글을 로컬 JSON 파일로 캐싱
  - 개별 항목: 영구 캐시 (immutable, 수정/삭제 시 무효화)
  - 목록 인덱스: TTL 기반 (5~10분)
  - 캐시 경로: `/home/grass/frontend/data/` (Docker 볼륨 마운트, 컨테이너 재빌드 후에도 유지)
  - 홈페이지: 326ms → 8ms, ai-diary: 317ms → 8ms, reports: 1,810ms → 14ms
- **정적 자산 캐시 헤더**: `max-age=604800` (7일) → Cloudflare 엣지 캐싱

#### 통합 서비스 관리 (`scripts/svc`)
- `deploy.sh`를 `scripts/svc`로 통합. `deploy.sh`는 `svc deploy`를 호출하는 래퍼로 유지 (하위 호환)
- `svc deploy [--api|--telegram|--frontend|--all] [--restart]`: git pull + 의존성 + 재시작 + Telegram 알림
- `svc boot`: 서버 재부팅 후 의존성 순서대로 전체 서비스 시작 (health check 포함)
- `svc kill/restart <service...>`: 실시간 실행 중인 작업 강제 중단/재시작 (SIGKILL)
- `svc status`: 전체 서비스·타이머 상태 조회 (Telegram `/status` 대시보드에도 통합)
- `--frontend` 옵션: 프론트엔드 전용 배포 (git pull → Docker rebuild → 컨테이너 교체)
- 프론트엔드 브랜치: `master` (leninbot `main`과 별도)
- 설정값(경로, 브랜치 등)은 `.env`에서 로드

#### 이메일 서명 업데이트
- `config/email_signature.json`: website_url → `https://cyber-lenin.com`

#### CSS 디자인 개선
- 모든 2px 경계선 → 1px (style.css, report.css, story-editor.css, game.css)
- 글 목록 사이 경계선 제거, navigation 버튼 사이/좌우 끝 경계선 제거
- navigation bar에 1px 사방 경계선 추가

### 2026-04-01 — Orchestration 전면 개편: 병렬 실행, orchestrator 콜백, 검증 통합

#### Concurrent Task Worker
- `task_worker`를 `asyncio.Semaphore` 기반 동시실행으로 교체 (기본 2, `/config`으로 조정)
- `contextvars.ContextVar`로 per-coroutine task context 관리 (동시 실행 시 task_id 구분)
- SIGTERM handler가 모든 active task에 checkpoint 수행

#### Browser Worker 분리
- `browser_worker.py`: 독립 systemd 서비스 (`leninbot-browser`)
- Unix domain socket IPC (`/tmp/leninbot-browser.sock`)
- Chromium 메모리 격리 (MemoryMax=2G), worker 불가 시 in-process fallback

#### Orchestrator Callback (agent→orchestrator→user)
- worker가 사용자에게 .md 파일을 직접 전송하지 않음
- task 완료 시 orchestrator가 결과를 받아 해석하고 자연어로 사용자에게 전달
- 실패 시 fallback으로 간단 요약 직접 전송

#### 병렬 위임 (multi_delegate)
- `multi_delegate` 도구: 여러 에이전트에 병렬 위임, 완료 후 자동 synthesis
- DB: `plan_id`, `plan_role` 컬럼 추가 (task 그룹 관리)
- subtask는 개별 알림 없이 DB에 결과만 저장, synthesis task만 orchestrator에 보고

#### Inline Sub-agent (run_agent)
- `run_agent` 도구: orchestrator 턴 안에서 analyst를 동기 실행
- budget $0.50 상한, max 10 rounds, orchestrator budget에서 차감

#### Verification → Orchestrator 통합
- 별도 `_run_verification` LLM 호출 제거 ($0.15/task 절약)
- `request_continuation` 도구 완전 제거 — worker는 미완료 상태를 응답에 포함
- orchestrator가 `was_interrupted` 플래그로 중단 여부 확인, 재위임 조건 충족 시에만 `delegate`
- 재위임 조건: 예산/턴 한도 미완수 + 개선 가능 + 외부 요인 아님

#### Agent Prompt 개편
- 모든 에이전트: 고정 report format 제거, "orchestrator에게 전달됨, 형식보다 정보량" 통일
- budget 경고: "마무리하라" → "계속 작업하라, 시스템이 자동 종료"
- 종료 메시지: "orchestrator가 재위임 판단" 으로 변경

#### 버그 수정
- `schedule_worker`: `continue` 뒤 dead code (태스크 생성 로직 전체) — 스케줄 실행 불가 버그
- `_ensure_table`: `agent_type` 컬럼 ALTER TABLE 누락
- `system_prompt or system_prompt` no-op
- 중복 `import re as _re` (3곳)
- Scout: budget $0→$1, provider fallback에 로컬 LLM 단계 추가 (MOON→local→Claude)
- `/agents` 텔레그램 autocomplete 누락

### 2026-03-31 — 에이전트 정체성 분리, browser agent, 이메일 도구, 톤 가이드

#### Agent Identity 분리
- `CORE_IDENTITY` (본체 전용) / `AGENT_CONTEXT` (에이전트 전용) 분리
- 에이전트가 더 이상 "You are Cyber-Lenin"을 주입받지 않음 — 고유 페르소나만 보유
- 정체성 충돌 제거

#### Tone & Format
- 안티-sycophancy 톤 가이드 (CORE_IDENTITY + AGENT_CONTEXT)
- 텔레그램 메시지: 마크다운 서식 금지 (plain text only)

#### Browser Agent
- `browser_use_agent.py`: browser-use 0.12.5 SDK 래퍼
- `agents/browser.py`: specialist agent 정의 ($1.50 budget, 30 rounds)
- delegate 도구에 browser 추가 (self_tools.py enum + 설명)
- 항상 Claude Sonnet 사용 (OpenAI structured output 호환성 문제)

#### Email Tools
- `check_inbox`: IMAP 실시간 접속, INBOX + Junk 양쪽 검색, 링크 자동 추출
- `allowlist_sender`: Junk → INBOX 메일 이동
- 모든 에이전트에 check_inbox 제공 (visualizer 제외)

#### /restart 개선
- 실행 시 processing + pending 태스크 모두 강제 종료 후 재시작
- 재시작 후 불필요한 태스크 재실행 방지

#### Tool Isolation 수정
- specialist 에이전트가 `delegate` 도구에 접근 불가 — 재귀 위임 방지
- task 실행 경로에서 `extra_tools`만 사용 (전체 TOOLS 머지 제거)

#### API 공개 범위
- programmer 태스크 리포트를 API에서 제외 (내부 서버 정보 노출 방지)

#### fetch_url 안정성
- shared.py에 `load_dotenv()` 추가 — Tavily API 키 로드 보장
- Playwright: `networkidle` 대기 추가
- Tavily Extract: API 키 명시 전달, 누락 시 스킵

#### KG 정리
- write_kg: 내부 시스템 상태 저장 금지 규칙 추가
- agent_knowledge 그룹에서 시스템 내부 에피소드 18개 삭제

#### Programmer 환경 주입
- programmer agent에만 `<runtime-environment>` 블록 주입 (OS, Python, sudo apt 권한 등)
- 환경 오진 방지

#### Verification 개선
- 외부 서비스 의존 실패는 PASS 처리 (403, CAPTCHA, 메일 미수신 등)
- 재시도해도 결과 안 달라지는 문제로 FAIL 판정 금지

### 2026-03-30 — 이메일 브리지, 검증 루프 전면 수정, R2 파일 호스팅

#### Email Bridge
- `email_bridge.py`: IMAP 수신 (Migadu) + Resend API 발신
- 수신 메일 자동 분류 (finance, urgent, human_request, bulk_like 등)
- `send_email` agent 도구: 에이전트가 직접 이메일 발송 가능 (HTML + 이미지 지원)
- 서명 자동 삽입 (`config/email_signature.json` — 로고, 이름, 이메일, 웹사이트)
- `/email` 텔레그램 커맨드 하나로 통합 (폴링 + 최근 기록)
- DB 테이블: email_threads, email_messages, email_bridge_events, email_bridge_state

#### Verification Loop 전면 수정
- **근본 원인 수정**: `fetch_server_logs()` 반환 타입 불일치 (`list[dict]`를 `str`로 처리) → 무한 retry의 원인
- **LLM 기반 검증**: regex 패턴 매칭 → 같은 agent가 도구로 독립 검증 (budget $0.15)
- **chain depth guard**: ancestor 순회로 무한 retry 차단 (이전에는 child의 verification_attempts만 체크)
- **content 누적 방지**: `[AUTO-RETRY...]` 프리픽스 strip
- **retry context**: 부모 result 요약 + "같은 접근 반복 금지" 지시 포함
- **서비스 재시작 판단**: 수정 파일 기반 (telegram_*.py → telegram, api.py → api)
- **telegram 재시작 시**: child task를 미리 생성(pending) 후 재시작 → task_worker가 pickup

#### Continuation 단순화
- `request_continuation`의 `restart_already_completed` 파라미터 제거
- 재시작 전 수동 continuation 불필요 — `recover_processing_tasks_on_startup`이 자동 처리
- programmer 시스템 프롬프트 반영

#### Cloudflare R2 + File Registry
- R2 버킷 `cyber-lenin-assets` (APAC, `assets.cyber-lenin.com` 커스텀 도메인)
- `upload_to_r2` agent 도구: 업로드 + file_registry DB 자동 등록
- `read_self(source='file_registry')`: 등록된 파일 검색
- 이메일 서명 이미지, 로고 호스팅

#### DNS (Cloudflare)
- `cyber-lenin.com` A/AAAA → Hetzner VPS (37.27.33.127)
- `assets.cyber-lenin.com` CNAME → R2 버킷
- Migadu MX/SPF/DKIM 설정 완료

### 2026-03-29 — 서비스 안정성 + 아키텍처 정리

#### restart_service tool
- 재시작 전 구문 검사 (변경된 .py 파일) + import 검증 (entry point) 자동 실행
- 검증 실패 시 재시작 차단, 에러 반환 → 크래시 루프 방지
- programmer 에이전트가 `execute_python + subprocess` 대신 이 tool 사용

#### Embedding Server 독립 분리
- `embedding_server.py`: BGE-M3를 별도 FastAPI 서비스로 분리 (:8100)
- `embedding_client.py`: drop-in replacement (embed_query/embed_documents)
- 15초 재시도 (서버 재시작 대응) + 로컬 fallback
- telegram_bot/api 재시작 시 모델 재로딩 불필요 (5.7초 절약)

#### KG 루프 일관성
- `submit_kg_task()` / `collect_kg_futures()` 추가 (non-blocking 병렬 KG 작업)
- `run_kg_async` → `run_kg_task`로 통일 (cross-loop 버그 방지)

#### 의존 관계 정리
- `similarity_search()`, `search_knowledge_graph()`를 shared.py로 통합
- 웹 챗봇: LangGraph 제거, claude_loop 기반 web_chat.py로 통합 (2026-04-04)

#### Mission auto-creation SQL 수정
- `SELECT DISTINCT user_id ... ORDER BY id DESC`가 PostgreSQL에서 InvalidColumnReference 에러 → `except: pass`에 삼켜져 미션이 절대 생성 안 됨
- DISTINCT 제거로 해결. Mission #9 이후 모든 task가 mission_id=None이었던 문제 수정.

#### write_kg NameError 수정
- `self_tools.py`: `"default": false` → `"default": False` (JSON 스타일 → Python)
- 이 오류로 서비스가 크래시 루프에 빠져 있었음

#### razvedchik 타이머 비활성화
- `razvedchik.py` 파일이 이미 삭제/이동된 상태에서 4시간마다 실패 → 타이머 disabled

### 2026-03-28 — Context Engineering, 모듈 분리, 에이전트 격리

- Orchestrator ↔ Agent 맥락 격리: 프로그래밍 도구 오케스트레이터 차단, delegate 필수
- `<current_state>` XML 블록: 태스크 상태 구조화 주입 (환각 방지)
- Observation Masking: tool_log recency 기반 점진 제거 (토큰 ~50% 절감)
- Mission 자동화: delegate 시 미션 자동 생성, stale 규칙 개선
- 모듈 분리: telegram_bot.py 2893줄 → 1134줄 (commands, config 분리)
- Visualizer: generate_image 도구 연결, reference_image 지원, flux_kontext_dev 라우팅

### 2026-03-27 — 에이전트 인프라, 프로젝트 구조

- scout 에이전트, patch_file tool, 태스크 인계(handed_off) 체계
- 환경/경로 .env 표준화, Neo4j keepalive 설정
- 루트 .py 28→20개 정리

### 2026-03-24 — Razvedchik 버그 수정, MOON PC LLM 연결

- qwen3.5-9b Q8_0 SSH 리버스 터널 연결, 이중 백엔드 fallback
- Ollama 완전 제거 → llama-server 통일

### 2026-03-22 — Web Search, Finance Data, XML Prompts

- Tavily 클라이언트 교체 (Claude 서버 도구 호환 문제)
- get_finance_data 도구 (yfinance 8개 자산)
- XML 구조 시스템 프롬프트

---

## Current Limitations

1. **No dynamic tool registry**: 런타임 도구 추가/제거 불가
2. **Memory is in-process only**: LangGraph MemorySaver가 서버 재시작 시 리셋
3. **Old junk arXiv in DB**: ~3,455 rows (math/telecom, 의미적으로 격리됨)
4. **Bukharin missing**: marxists.org에서 올바른 URL 미확인
5. **leninbot.duckdns.org Nginx 프록시 잔존**: 백엔드 완전 은닉을 위해 제거 필요 (임시 유지 중)
6. **Render 서비스 폐기 예정**: redirect-only 브랜치로 리다이렉트 중, 2~3일 후 종료
