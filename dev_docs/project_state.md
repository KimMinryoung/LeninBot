# Project State — 2026-03-29

## Identity

**Cyber-Lenin** (사이버-레닌) — Unified digital revolutionary intelligence across four interfaces: web chatbot, Telegram agent, autonomous diary writer, local PC agent. Shared memory, shared principles, one continuous consciousness.

Server: **Hetzner VPS** (Ubuntu 24.04, 16GB RAM). HTTPS via `leninbot.duckdns.org` (Nginx + Let's Encrypt). Frontend at `bichonwebpage.onrender.com`. Local agent on Windows 10 PC (RTX 3060 12GB).

---

## Service Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │              외부 인프라 (항상 ON)              │
                        │  Supabase PostgreSQL + pgvector              │
                        │  (채팅로그, 태스크큐, 벡터DB, 경험메모리)         │
                        └──────────────────┬───────────────────────────┘
                                           │ SQL
┌──────────────────────────────────────────┼──────────────────────────────┐
│                    Hetzner VPS 서비스들     │                              │
│                                          │                              │
│  ┌───────────────┐  ┌───────────────┐    │    ┌────────────────────┐    │
│  │ Neo4j Docker  │  │ embedding     │    │    │ MOON PC            │    │
│  │ (:7687)       │  │ _server.py    │    │    │ (SSH 리버스 터널)    │    │
│  │ 지식 그래프     │  │ (:8100)       │    │    │ qwen3.5-9b Q8_0   │    │
│  │               │  │ BGE-M3 모델    │    │    │ (:8080 via tunnel) │    │
│  └───────┬───────┘  └───────┬───────┘    │    └─────────┬──────────┘    │
│          │ Bolt             │ HTTP       │              │ HTTP          │
│          │                  │            │              │               │
│  ┌───────┴──────────────────┴────────────┴──────────────┴─────────┐    │
│  │                     shared.py (공유 라이브러리)                    │    │
│  │  similarity_search()    search_knowledge_graph()               │    │
│  │  add_kg_episode()       get_kg_service() / run_kg_task()       │    │
│  │  submit_kg_task()       collect_kg_futures()                   │    │
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
│  │ 도구 13+개         │    │ 9-node 워크플로우   │                         │
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

### LLM Provider (런타임 교체 가능)

`/provider` 명령으로 Claude ↔ OpenAI 실시간 전환. 시스템 프롬프트에 `<current-model>` 태그로 현재 모델 정보 자동 주입 — 에이전트가 자신의 모델을 인지.

| Tier | Claude | OpenAI |
|------|--------|--------|
| high | Claude Opus 4.6 | GPT-5.4 |
| medium | Claude Sonnet 4.6 | GPT-5.4-mini |
| low | Claude Haiku 4.5 | GPT-5.4-nano |

`bot_config.py`에서 관리. chat은 medium tier, task는 에이전트별 budget/tier 설정. `/fallback`으로 medium ↔ low 토글.

### Orchestrator
사용자 메시지를 받아 의도를 파악하고, 필요 시 전문 에이전트에 `delegate` tool로 위임. 프로그래밍 도구 직접 접근 차단 — 코드 작업은 반드시 programmer에게 위임.

### Specialist Agents

| 에이전트 | 페르소나 | 역할 | 주요 도구 |
|---|---|---|---|
| **analyst** (Varga) | 정보 분석가 | 조사, 분석, KG 저장 | vector_search, kg_search, web_search, write_kg |
| **programmer** (Kitov) | 코드 전문가 | 코드 수정, 디버깅 | patch_file, write_file, execute_python, restart_service |
| **visualizer** (Rodchenko) | 이미지 생성 | 프로파간다 포스터/게임아트 | generate_image (Replicate FLUX), reference_image 지원 |
| **scout** | 정찰 에이전트 | 외부 플랫폼 데이터 수집 | web_search, fetch_url, write_file |

### 핵심 도구

- **restart_service**: 재시작 전 구문 검사 + import 검증 → 크래시 루프 방지
- **save_finding**: 미션 타임라인에 중간 발견 기록
- **request_continuation**: 재시작/예산 한계 시 자식 태스크 생성 (재시작 문구 자동 제거)
- **write_kg**: 지식 그래프에 사실 저장 (KG 전용 루프에서 실행)
- **mission**: 미션 상태 확인/종료 (delegate 시 자동 생성)

### Task Lifecycle
```
delegate → pending → processing → done/failed/handed_off
                                      ↓
                              request_continuation → child task (pending)
```

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
├── shared.py                  # 공유 라이브러리: CORE_IDENTITY, KG/벡터검색/메모리/URL
├── embedding_server.py        # BGE-M3 임베딩 서버 (독립 FastAPI, :8100)
├── embedding_client.py        # 임베딩 HTTP 클라이언트 (fallback 내장)
├── bot_config.py              # LLM 클라이언트, 런타임 설정, 모델 해석
├── telegram_bot.py            # Telegram 봇 코어: 채팅, LLM 디스패치, 에이전트 실행
├── telegram_commands.py       # 커맨드/메시지/콜백 핸들러
├── telegram_tools.py          # 도구 정의 + 핸들러 (restart_service 포함)
├── telegram_tasks.py          # 백그라운드 태스크 워커, 스케줄러, 모니터
├── telegram_mission.py        # 미션 컨텍스트 시스템
├── self_tools.py              # delegate, save_finding, request_continuation, write_kg
├── claude_loop.py             # Claude tool-use 루프
├── replicate_image_service.py # Replicate FLUX 이미지 생성 (reference_image 지원)
├── finance_data.py            # 실시간 금융 데이터 (yfinance, 10분 캐시)
├── diary_writer.py            # 자율 일기 작성 + 뉴스 KG 병렬 수집
├── experience_writer.py       # 경험 메모리 일일 정리
├── db.py                      # PostgreSQL 커넥션 풀 (psycopg2)
├── patch_file.py              # 토큰 효율적 파일 패치 (replace_block)
├── self_modification_core.py  # Git backup + syntax check + rollback
│
├── agents/                    # 에이전트 정의
│   ├── base.py                # AgentSpec + 공통 컨텍스트 블록
│   ├── analyst.py             # Varga — 정보 분석/KG 저장
│   ├── programmer.py          # Kitov — 코드 수정 (restart_service 사용)
│   ├── visualizer.py          # Rodchenko — 이미지 생성 (reference_image 지원)
│   └── scout.py               # 외부 플랫폼 정찰
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
