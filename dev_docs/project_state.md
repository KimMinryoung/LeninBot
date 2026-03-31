# Project State — 2026-03-31

## Identity

**Cyber-Lenin** (사이버-레닌) — Unified digital revolutionary intelligence across four interfaces: web chatbot, Telegram agent, autonomous diary writer, local PC agent. Shared memory, shared principles, one continuous consciousness.

Server: **Hetzner VPS** (Ubuntu 24.04, 16GB RAM). HTTPS via `leninbot.duckdns.org` (Nginx + Let's Encrypt). Frontend at `bichonwebpage.onrender.com`. Local agent on Windows 10 PC (RTX 3060 12GB).

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
│  │ Neo4j Docker  │  │ embedding     │    │    │ MOON PC            │    │
│  │ (:7687)       │  │ _server.py    │    │    │ (Tailscale 터널)    │    │
│  │ 지식 그래프     │  │ (:8100)       │    │    │ qwen3.5-9b Q8_0   │    │
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
│  │ leninbot-telegram │    │ leninbot-api     │                         │
│  │ telegram_bot.py   │    │ uvicorn :8000    │                         │
│  │                   │    │                  │                         │
│  │ aiogram 3.x       │    │ chatbot.py       │                         │
│  │ Claude/GPT 교체가능 │    │ (LangGraph RAG)  │                         │
│  │ 에이전트 태스크큐    │    │ Gemini 3.1 FL    │                         │
│  │ email_bridge.py   │    │ 9-node 워크플로우   │                         │
│  │ browser-use SDK   │    │                  │                         │
│  │ 도구 20+개         │    │                  │                         │
│  └──────────────────┘    └──────────────────┘                         │
│  Telegram polling          :8000 (외부, 웹챗봇)                         │
└───────────────────────────────────────────────────────────────────────┘
```

### 상주 서비스 (systemd, 항상 ON)

| 서비스 | 프로세스 | 포트 | 역할 |
|---|---|---|---|
| `leninbot-neo4j` | Docker (neo4j:5-community) | :7687, :7474 | 지식 그래프 저장소 (Graphiti) |
| `leninbot-embedding` | embedding_server.py | :8100 (내부) | BGE-M3 임베딩 서버 (831MB) |
| `leninbot-telegram` | telegram_bot.py | Telegram polling | 텔레그램 봇 + 에이전트 시스템 |
| `leninbot-api` | uvicorn api:app | :8000 (외부) | 웹 챗봇 API (LangGraph) |

### 타이머 서비스 (주기 실행 → 종료)

| 서비스 | 주기 | 역할 |
|---|---|---|
| `leninbot-diary` | 매 3시간 | 일기 작성 + 뉴스 KG 수집 (병렬) |
| `leninbot-experience` | 매일 15:30 UTC | 경험 메모리 정리/저장 |

### 서비스 의존 관계

| 서비스 | Supabase | Neo4j | embedding_server | chatbot.py |
|---|---|---|---|---|
| **leninbot-embedding** | - | - | 자기 자신 | - |
| **leninbot-neo4j** | - | 자기 자신 | - | - |
| **leninbot-telegram** | O | O | O | - |
| **leninbot-api** | O | O | O | O (LangGraph) |
| **leninbot-diary** | O | O | O | - |
| **leninbot-experience** | O | - | O | - |

### 기동/배포 순서

```
항상 ON (재시작 안 함):
  1. leninbot-neo4j (Docker)
  2. leninbot-embedding (Before=telegram,api)

deploy.sh가 재시작하는 것:
  3. leninbot-api
  4. leninbot-telegram (마지막 — 자기 자신을 죽이므로)
```

`deploy.sh`는 `git pull` + `leninbot-api` → `leninbot-telegram` 순으로만 재시작. Neo4j와 embedding_server는 코드 변경과 무관하므로 건드리지 않음.

### 가용성 보장

| 시나리오 | 대응 |
|---|---|
| embedding_server 크래시 | systemd `Restart=always` (5초) + 클라이언트 15초 재시도 |
| embedding_server 장기 다운 | 클라이언트 로컬 BGE-M3 fallback 자동 로딩 |
| Neo4j 다운 | KG 기능만 비활성화, `_KG_RETRY_INTERVAL=120초` 후 재시도 |
| telegram/api 재시작 | 임베딩/KG 재로딩 없음, 서비스 5초 내 복구 |
| 서버 재부팅 | 모든 서비스 자동 시작 (enabled) |

---

## Telegram Agent System (CLAW Architecture)

### Identity Architecture

시스템 프롬프트 정체성이 본체와 에이전트로 분리됨:

- **CORE_IDENTITY**: Orchestrator, chatbot, diary_writer 전용. "You are Cyber-Lenin."
- **AGENT_CONTEXT**: Specialist 에이전트 전용. "You are a specialist agent in the Cyber-Lenin system. You serve Cyber-Lenin, but you are NOT Cyber-Lenin."

각 에이전트는 고유 페르소나(Kitov, Varga 등)만 갖고, 사이버-레닌 본체와 정체성 충돌 없음.

### LLM Provider (런타임 교체 가능)

`/provider` 명령으로 Claude ↔ OpenAI 실시간 전환. 시스템 프롬프트에 `<current-model>` 태그로 현재 모델 정보 자동 주입 — 에이전트가 자신의 모델을 인지.

| Tier | Claude | OpenAI |
|------|--------|--------|
| high | Claude Opus 4.6 | GPT-5.4 |
| medium | Claude Sonnet 4.6 | GPT-5.4-mini |
| low | Claude Haiku 4.5 | GPT-5.4-nano |

`bot_config.py`에서 관리. chat은 medium tier, task는 에이전트별 budget/tier 설정. `/fallback`으로 medium ↔ low 토글.

**browser-use SDK는 항상 Claude Sonnet 4.6 사용** (OpenAI 모델은 structured output parsing 실패).

### Orchestrator
사용자 메시지를 받아 의도를 파악하고, 필요 시 전문 에이전트에 `delegate` tool로 위임. 프로그래밍 도구 직접 접근 차단 — 코드 작업은 반드시 programmer에게 위임. 텔레그램 메시지는 마크다운 서식 금지 (plain text only).

### Specialist Agents

| 에이전트 | 페르소나 | 역할 | 주요 도구 |
|---|---|---|---|
| **analyst** (Varga) | 정보 분석가 | 조사, 분석, KG 저장 | vector_search, kg_search, web_search, write_kg, send_email, check_inbox |
| **programmer** (Kitov) | 코드 전문가 | 코드 수정, 디버깅 | patch_file, write_file, execute_python, restart_service, upload_to_r2, send_email, check_inbox |
| **browser** | 브라우저 자동화 | 로그인, 폼 제출, 동적 사이트 | browse_web, check_inbox, allowlist_sender, fetch_url |
| **visualizer** (Rodchenko) | 이미지 생성 | 프로파간다 포스터/게임아트 | generate_image (Replicate FLUX), upload_to_r2 |
| **scout** | 정찰 에이전트 | 외부 플랫폼 데이터 수집 | web_search, fetch_url, write_file, upload_to_r2, check_inbox |

### 핵심 도구

- **restart_service**: 재시작 전 구문 검사 + import 검증 → 크래시 루프 방지. 재시작 시 자동 복구 태스크 생성 (request_continuation 불필요)
- **send_email**: Resend API로 이메일 발신. HTML 지원, 서명 자동 삽입 (`config/email_signature.json`). DB 기록
- **check_inbox**: IMAP 실시간 접속 (INBOX + Junk 양쪽 검색). 발신자/제목 필터, 링크 자동 추출. 뉴스레터 인증 메일 처리에 사용
- **allowlist_sender**: Junk 폴더에서 특정 발신자 메일을 INBOX로 이동
- **browse_web**: browser-use SDK (Playwright + LLM). AI가 스크린샷 보고 클릭/입력/탐색. 항상 Claude Sonnet 사용
- **upload_to_r2**: Cloudflare R2에 파일 업로드 → 공개 URL 반환. file_registry DB 자동 등록
- **save_finding**: 미션 타임라인에 중간 발견 기록
- **request_continuation**: 예산 한계 시 자식 태스크 생성 (작업 이관 전용)
- **write_kg**: 지식 그래프에 사실 저장 (KG 전용 루프에서 실행). 내부 시스템 상태 저장 금지
- **mission**: 미션 상태 확인/종료 (delegate 시 자동 생성)
- **delegate**: specialist 에이전트에 작업 위임 (analyst, programmer, scout, visualizer, browser)

### Task Lifecycle
```
delegate → pending → processing → done → LLM verification
                                            ├─ passed → complete
                                            └─ failed → auto-retry child (max chain depth = retry_limit+1)
                                                         └─ includes parent result summary + verification details
                                   ↓
                           handed_off (request_continuation → child task)
```

### /restart Command
- 실행 시 모든 processing/pending 태스크를 강제 종료(done) 후 서비스 재시작
- 재시작 후 불필요한 태스크 재실행 방지

### Verification System
- **Phase 1**: 자동 체크 (task_report 존재, url_access HTTP 상태)
- **Phase 2**: LLM 기반 검증 — 같은 agent가 도구로 독립 검증 (로그 확인, 코드 확인, URL 접근)
- 검증자가 서비스 재시작 필요 판단 시: api는 직접 재시작, telegram은 VERDICT: FAIL → child task 생성 후 재시작
- chain depth guard: ancestor 순회로 무한 retry 차단
- retry child에 부모 result 요약 포함 → 같은 접근 반복 방지
- 외부 서비스 의존 실패(403, CAPTCHA, 메일 미수신 등)는 PASS 처리

### Service Restart Recovery
- `restart_service` 호출 → `persist_task_restart_state` → 프로세스 사망
- `recover_processing_tasks_on_startup` → child task 자동 생성 (`_RESTART_COMPLETED_MARKER` 포함)
- child는 재시작 이미 완료된 상태로 인식 → 재시작 반복 없음
- `request_continuation` 사전 호출 불필요

### Tool Isolation
- Orchestrator: 모든 도구 접근 가능 (단, 프로그래밍 도구 차단)
- Specialist 에이전트: `AgentSpec.filter_tools()`로 역할별 도구만 노출
- `delegate` 도구는 orchestrator만 접근 가능 — 에이전트 간 재위임 불가

---

## Web Chatbot (LangGraph RAG)

### Graph Flow
```
START → analyze_intent
  ├─[vectorstore]→ retrieve → kg_retrieve → grade_documents
  │                                            ├─[need_web_search]→ web_search → generate
  │                                            └─[no_need]→ generate
  ├─[generate]→ generate → log_conversation → END
  └─[plan]→ planner → step_executor → generate → log_conversation → END
```

### 9 Nodes

| Node | LLM | Purpose |
|------|-----|---------|
| analyze_intent | Gemini 2.5 Flash-Lite | 라우팅, 의도 분류, 쿼리 분해/번역, 자기지식 도구 선택 |
| retrieve | — | 벡터 검색 (embedding_server HTTP → pgvector) |
| kg_retrieve | — | 지식 그래프 검색 (Neo4j/Graphiti, 휴리스틱 필터) |
| grade_documents | Gemini 2.5 Flash-Lite | 문서 관련성 배치 평가 + 실시간 정보 필요 판단 |
| web_search | — | Tavily 검색 |
| generate | Gemini 3.1 Flash-Lite (streaming) | 최종 답변 생성 (변증법적 분석 내장) |
| log_conversation | — | PostgreSQL 채팅 로그 저장 |
| planner | Gemini 3.1 Flash-Lite | 복합 질문용 2-4단계 연구 계획 |
| step_executor | — | 계획 단계별 실행 (검색 + KG, 단계간 중복 제거) |

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
├── api.py                     # FastAPI (SSE streaming, /chat, /logs, /session/*)
├── chatbot.py                 # LangGraph 9-node RAG pipeline (웹챗봇 전용)
├── shared.py                  # 공유 라이브러리: CORE_IDENTITY, AGENT_CONTEXT, KG/벡터검색/메모리/URL
├── embedding_server.py        # BGE-M3 임베딩 서버 (독립 FastAPI, :8100)
├── embedding_client.py        # 임베딩 HTTP 클라이언트 (fallback 내장)
├── bot_config.py              # LLM 클라이언트, 런타임 설정, 모델 해석
├── telegram_bot.py            # Telegram 봇 코어: 채팅, LLM 디스패치, 에이전트 실행
├── telegram_commands.py       # 커맨드/메시지/콜백 핸들러 (/restart: 태스크 강제종료 + 재시작)
├── telegram_tools.py          # 도구 정의 + 핸들러 (check_inbox, allowlist_sender, browse_web 등)
├── telegram_tasks.py          # 백그라운드 태스크 워커, 스케줄러, 모니터
├── telegram_mission.py        # 미션 컨텍스트 시스템
├── self_tools.py              # delegate, save_finding, request_continuation, write_kg
├── claude_loop.py             # Claude tool-use 루프
├── browser_use_agent.py       # browser-use SDK 래퍼 (Playwright + LLM, 항상 Claude Sonnet)
├── replicate_image_service.py # Replicate FLUX 이미지 생성 (reference_image 지원)
├── finance_data.py            # 실시간 금융 데이터 (yfinance, 10분 캐시)
├── diary_writer.py            # 자율 일기 작성 + 뉴스 KG 병렬 수집
├── experience_writer.py       # 경험 메모리 일일 정리
├── db.py                      # PostgreSQL 커넥션 풀 (psycopg2)
├── patch_file.py              # 토큰 효율적 파일 패치 (replace_block)
├── email_bridge.py            # 이메일 브리지 (IMAP 수신, Resend 발신, 분류, DB 기록)
├── self_modification_core.py  # Git backup + syntax check + rollback
│
├── config/
│   └── email_signature.json   # 이메일 서명 설정 (agent 수정 가능)
│
├── agents/                    # 에이전트 정의
│   ├── base.py                # AgentSpec + 공통 컨텍스트 블록
│   ├── analyst.py             # Varga — 정보 분석/KG 저장
│   ├── programmer.py          # Kitov — 코드 수정 (restart_service 사용)
│   ├── browser.py             # Browser — AI 브라우저 자동화 (browser-use SDK)
│   ├── visualizer.py          # Rodchenko — 이미지 생성 (reference_image 지원)
│   ├── scout.py               # 외부 플랫폼 정찰
│   └── general.py             # 범용 리서치 (도구 제한 없음)
│
├── graph_memory/              # Graphiti 지식 그래프 모듈
│   ├── service.py             # GraphMemoryService (Neo4j keepalive/liveness)
│   ├── kr_news_fetcher.py     # 한국 뉴스 수집 파이프라인
│   └── cli.py                 # KG 질의 CLI
│
├── skills/                    # 에이전트 스킬 (SKILL.md 포맷)
├── scripts/                   # 독립 실행 스크립트
├── local_agent/               # 로컬 PC 에이전트 (Windows, Claude Sonnet 4.6)
├── systemd/                   # systemd 서비스/타이머 정의
├── deploy.sh                  # 배포: git pull → api → telegram 재시작
├── data/                      # 런타임 데이터 (gitignored)
└── .env                       # 환경변수
```

---

## Recent Changes

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
- diary_writer 뉴스 수집 병렬화
- chatbot.py / diary_writer.py의 `run_kg_async` → `run_kg_task`로 통일 (cross-loop 버그 방지)

#### 의존 관계 정리
- `similarity_search()`, `search_knowledge_graph()`를 chatbot.py → shared.py로 이동
- telegram_tools → chatbot.py 의존 제거 (graph import만 남음, api.py 전용)

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
5. **Stale render.yaml**: OPENAI_API_KEY 참조 (Render 비활성)
