# Project State

2026-09-23 저장소 코드·프런트엔드 프록시 설정·설치된 Nginx 설정을 대조했다. 서비스 활성 상태는 별도 점검 대상이다.

Cyber-Lenin은 하나의 런타임 정체성을 여러 인터페이스로 노출하는 시스템이다. 주요 사용자 인터페이스는 Telegram bot, public web chat API, scheduled autonomous/diary/background workers다. 장기 상태는 로컬 PostgreSQL(`leninbot-pg` Docker 컨테이너 — 활성 `leninbot`·`writer` DB와 읽기 전용 보관 `legacy_game` DB, `dev_docs/db_migration_plan.md`)과 Neo4j에 저장하고, Redis는 실행 중인 task 상태와 mission board 같은 단기 공유 상태를 맡는다.

## Runtime Map

```
cyber-lenin.com (Cloudflare -> Nginx, Cloudflare Origin Certificate)
        |
        v
frontend (Node/Express :3000, separate repository)
        |-- pages, CommuLingo, admin UI -> frontend routes
        |-- /api/proxy/* -> route by path prefix
        |      |-- /writer/* -> novel-writer-api :8001
        |      |-- /email/* -> leninbot-email-api :8002
        |      `-- other paths -> leninbot-api :8000
        `-- /.well-known/agent-card.json, /a2a -> leninbot-a2a-api :8003

leninbot-api (:8000, FastAPI)
        |-- /chat, /chat/feedback, /personas -> services/api.py + services/web_chat.py
        |-- admin/chat-history/report/private-report JSON -> api_routes/* modules
        |
        +--> PostgreSQL (leninbot-pg, 127.0.0.1:5434)
        +--> Neo4j KG via kg_runtime/
        +--> services/embedding_server.py (:8100)

novel-writer-api (:8001, FastAPI)
        |-- /writer -> api_routes.writer -> writer/ package
        |
        +--> writer DB in leninbot-pg via WRITER_DB_*
        +--> Anthropic/DeepSeek writer model clients

leninbot-email-api (:8002, FastAPI)
        |-- /email/* -> api_routes.email -> services/email_bridge.py
        |
        +--> PostgreSQL (leninbot-pg) email tables
        +--> IMAP/Resend email provider config

leninbot-a2a-api (:8003, FastAPI)
        |-- /.well-known/agent-card.json + /a2a -> services/a2a_api.py + services/a2a_handler.py
        |
        +--> PostgreSQL (leninbot-pg, 127.0.0.1:5434)
        +--> Neo4j KG via kg_runtime/
        +--> provider/tool runtime

Telegram
        |
        v
telegram/bot.py orchestrator (chat loop: telegram/chat_runtime.py)
        |-- runtime_tools/ registry and allow-lists
        |-- telegram/tasks.py background task worker
        |-- agents/* AgentSpec registry
        |
        +--> PostgreSQL task/chat/mission tables
        +--> Redis task progress, active task state, mission board
        +--> Neo4j KG

systemd timers
        |-- leninbot-autonomous.timer -> jobs/autonomous_project.py
        |-- leninbot-experience.timer -> jobs/experience_writer.py
        |-- leninbot-kg-integrity.timer -> scripts/check_kg_integrity.py
        |-- research-document-translation.timer -> scripts/run_translation_batch.py -> research/DB translation scripts
        |-- leninbot-email-poller.timer -> scripts/email_poll_once.py
        |-- leninbot-commulingo-pipeline.timer -> scripts/commulingo_pipeline.py tick (durable people/term stages)
        |-- (no timer) leninbot-commulingo-{maintainer,new,enrich,terms,gap}.service -> manual-start legacy lanes

developer MCP clients
        |
        v
        python -m mcp_gateway.server (stdio, on demand)
        |-- explicit MCP profile allow-list
        |-- read-only adapters over runtime_tools, dev_docs, task/corpus state
        |-- operator-only readonly_query_db and bounded_query_db via existing DB guards

LLM-consuming services -> llm_proxy (127.0.0.1:8110; text tier resolution + key injection) -> model providers
paid search/extraction -> web_gateway (127.0.0.1:8111) -> Tavily/Brave
```

## 공개 진입점과 프록시 경계

Nginx는 Cloudflare 뒤에서 `cyber-lenin.com`의 HTTPS를 받고 모든 일반 공개 경로를 프런트엔드 Node/Express(`127.0.0.1:3000`)로 보낸다. 포트 80은 HTTPS로 넘긴다. 공개 `/admin`은 404이며, Tailscale 주소의 별도 HTTP server block은 같은 프런트엔드로 전달한다. 채팅·writer SSE를 위해 Nginx 프록시 버퍼링을 끄고 읽기·쓰기 타임아웃을 둔다. Nginx의 `/api/proxy/`와 `/commulingo/api/`에는 IP별 요청 제한을 적용한다. 설정 원본은 별도 프런트엔드 저장소의 `/home/grass/frontend/nginx/leninbot-frontend.conf`이며 설치본은 `/etc/nginx/sites-available/leninbot-frontend`다.

프런트엔드 `server.js`는 세션·채팅 신원 미들웨어 다음에 `config/proxies.js`를 설치한다. `/api/proxy` 접두사를 제거한 뒤 `/writer`는 writer API, `/email`은 email API, 나머지는 main API로 보낸다. `/.well-known/agent-card.json`과 `/a2a`는 별도 A2A 프록시가 처리한다. `/api/proxy/sessions`와 `/api/proxy/history`의 일부 GET은 `routes/chat-history.js`가 프록시보다 먼저 프런트엔드 DB에서 답한다. 프런트엔드는 클라이언트가 보낸 신원·관리자 헤더를 제거하고, 확인한 세션과 자체 secret을 바탕으로 백엔드용 헤더를 주입한다. 목적지 기본값은 `config/services.js`에 있지만 실제 배포값은 프런트엔드 환경변수가 우선한다. 요청별 경로와 인증 계약은 [api_reference.md](api_reference.md)를 따른다.

Nginx·프런트엔드의 역방향 HTTP 프록시와 내부 서비스 게이트웨이는 별개다. `llm_proxy`는 loopback `:8110`에서 모델 제공자 키를 보관·주입하고 정책·감사를 수행한다([llm_gateway.md](llm_gateway.md)). `web_gateway`는 loopback `:8111`에서 유료 검색·본문 추출 키와 공용 예산을 관리한다([web_research.md](web_research.md)). 두 서비스는 공개 Nginx 라우트가 아니다. 모델이 호출하는 도구의 가시성과 실행 권한은 [tool_gateway.md](tool_gateway.md), [security_gateway.md](security_gateway.md)가 담당한다.

## Service Units

| Unit | Entrypoint | Role |
|---|---|---|
| `leninbot-neo4j.service` | Docker Compose | Neo4j, Redis, and main Postgres (`leninbot-pg`) backing services — stopping this unit takes the main DB down too |
| `leninbot-llm-proxy.service` | `uvicorn llm_proxy.app:app` | localhost key-injection passthrough, authoritative LLM policy gate, provider credential custodian; readiness blocks LLM consumers |
| `leninbot-web-gateway.service` | `uvicorn web_gateway.app:app` | localhost search/Extract service, sole search-key custodian, shared daily budget and usage ledger |
| `leninbot-embedding.service` | `services/embedding_server.py` | local embedding HTTP service |
| `leninbot-telegram.service` | `telegram/bot.py` | Telegram orchestrator and task worker |
| `leninbot-roleplay.service` | `roleplay/bot.py` | standalone roleplay companion bot, independent of Cyber-Lenin |
| `leninbot-api.service` | `uvicorn services.api:app` | web chat, chat history, admin users, task reports, private report JSON, x402 demo; remaining `api_routes/*` here are code modules in this same process |
| `novel-writer-api.service` | `uvicorn services.novel_writer_api:app` | isolated personal fiction writer API and writer SSE runs |
| `leninbot-email-api.service` | `uvicorn services.email_api:app` | admin-gated `/email/*` review, approval, draft, and manual poll API |
| `leninbot-email-poller.timer` | `scripts/email_poll_once.py` | periodic IMAP polling into the email bridge tables; the script forces `EMAIL_POLLING_ENABLED=true` at import time (a unit `Environment=` override would lose to `EnvironmentFile=`), independently of the legacy Telegram-loop setting |
| `leninbot-a2a-api.service` | `uvicorn services.a2a_api:app` | public A2A discovery and JSON-RPC endpoint |
| `leninbot-browser.service` | `browser/worker.py` | browser automation worker over Unix socket |
| `leninbot-autonomous.service` | `venv/bin/python -m jobs.autonomous_project` | one autonomous project tick |
| `leninbot-experience.service` | `jobs/experience_writer.py` | daily experience memory write |
| `leninbot-kg-integrity.service` | `scripts/check_kg_integrity.py` | KG maintenance check |
| `leninbot-kg-sync.service` | `python -m jobs.kg_sync --source commulingo,documents --documents-limit 40 --notify-on-error` | nightly 04:00 KST — CommuLingo·발행 문서를 KG로 미러 (증분, 7일마다 전체) |
| `leninbot-kg-report.service` | `scripts/kg_weekly_report.py --notify` | Mon 09:30 KST — KG 건강 리포트 (성장·중복·동기화 지연·검색 사용량) |
| `leninbot-commulingo-review.service` | `scripts/commulingo_person_reviewer.py` | 독립 timer는 비활성. pipeline이 공통 검토 함수를 사용하며 `/commulingo_review`는 수동 조회·처리용 |
| `leninbot-commulingo-pipeline.service` | `scripts/commulingo_pipeline.py tick` | 대상별 보강 묶음과 최대 12단계 연속 실행; live 반영(건수 제한 없음)·일일 공용 예산은 `config/commulingo_pipeline.json` |
| `leninbot-commulingo-maintainer.service` | `scripts/commulingo_people_maintainer.py` | 수동 실행 전용(timer 없음). one sourced CommuLingo edit or pending review. 2026-09-20 daily batch 폐지 뒤 pipeline이 정기 편집을 맡는다 |
| `leninbot-commulingo-new.service` | `scripts/commulingo_people_parallel.py --mode new` | 수동 실행 전용(timer 없음). new-person discovery/create lane |
| `leninbot-commulingo-enrich.service` | `scripts/commulingo_people_parallel.py --mode enrich` | 수동 실행 전용(timer 없음). existing-person enrichment lane |
| `leninbot-commulingo-terms.service` | `scripts/commulingo_terms_maintainer.py` | 수동 실행 전용(timer 없음). glossary-term creation lane |
| `leninbot-commulingo-gap.service` | `scripts/commulingo_gap_worker.py` | 수동 실행 전용(timer 없음). curation gap queue lane |

Dependency direction is simple: `leninbot-llm-proxy.service` waits for network-online and a credential-complete `/health`, then every LLM-consuming unit starts after it; Neo4j/Redis and embedding also start before Telegram/API; browser starts after Telegram. API can optionally run Telegram in-process only when `RUN_TELEGRAM_IN_API=true`, but production uses the dedicated Telegram unit.

## API Boundary Status

`api_routes/*` does not automatically mean a separate service. As of 2026-07-08, writer, email, and A2A are real service boundaries; the remaining extracted route modules are code-ownership boundaries inside `leninbot-api.service` and still share one Uvicorn process, one port (`:8000`), and the main API systemd credentials/runtime.

| Surface | Runtime boundary | Code owner | Notes |
|---|---|---|---|
| Writer workspace `/writer/*` | separate `novel-writer-api.service` on `:8001` | `services/novel_writer_api.py`, `api_routes/writer.py`, `writer/` | frontend `/api/proxy/writer` targets this service; `leninbot-api.service` does not include writer routes |
| Email bridge `/email/*` | separate `leninbot-email-api.service` on `:8002` plus `leninbot-email-poller.timer` | `services/email_api.py`, `api_routes/email.py`, `services/email_bridge.py`, `scripts/email_poll_once.py` | frontend `/api/proxy/email` targets this service; periodic polling no longer depends on main API requests |
| A2A discovery/JSON-RPC | separate `leninbot-a2a-api.service` on `:8003` | `services/a2a_api.py`, `services/a2a_handler.py` | frontend proxies `/.well-known/agent-card.json` and `/a2a` to this service; main API no longer includes A2A routes |
| Public web chat `/chat`, `/chat/feedback`, `/personas` | same `leninbot-api.service` process | `services/api.py`, `services/web_chat.py`, `services/web_personas.py` | kept in the main API because it shares session locks, fingerprint/proxy identity, persona visibility, feedback/regeneration, and rate limiting |
| Admin users `/admin/users*` | same `leninbot-api.service` process | `api_routes/admin_users.py` | code moved out of `services/api.py`; URLs/auth unchanged |
| Chat logs/history `/logs`, `/history`, `/sessions`, `/session/{id}` | same `leninbot-api.service` process | `api_routes/chat_history.py` | code moved out of `services/api.py`; frontend may serve some history from its local DB first |
| Task reports `/reports*` | same `leninbot-api.service` process | `api_routes/task_reports.py` | code moved out of `services/api.py`; URLs/auth unchanged |
| Private report JSON `/private-reports*` | same `leninbot-api.service` process | `api_routes/private_reports.py` | browser shell moved to frontend `/admin/private-reports`; FastAPI only serves JSON |


## Main Data Stores

| Store | Owner modules | Stored state |
|---|---|---|
| PostgreSQL (`leninbot-pg`, pgvector/pg17, `127.0.0.1:5434`; 활성 `leninbot` + `writer` DBs — Supabase에서 2026-07-28 이전, `db_migration_plan.md`) | `db.py`, `telegram/task_store.py`, `memory_store/*`, `jobs/autonomous_project.py`, `services/email_bridge.py`, `security_gateway/audit.py`, `writer/store.py` | chat logs, task queue, missions, reports, autonomous projects, email metadata, vector corpus metadata, writer projects/messages/manuscripts/revisions, `tool_audit_log` (per-call security audit) |
| PostgreSQL `legacy_game` DB (`leninbot-pg` 내부, 런타임 미사용·읽기 전용 보관) | 운영자 전용; `scripts/backup_main_db_to_r2.py`, `scripts/restore_db.py` | 옛 게임의 `story_scenes` 415행. main DB에서 2026-07-29 분리했으며 일일 로컬/R2 백업 및 DRI 복구 범위에 포함 |
| pgvector | `corpus/*`, `memory_store/experiential.py` | core theory, modern analysis, self-produced analysis, experience memory vectors |
| Neo4j | `graph_memory/*`, `kg_runtime/*`, `jobs/kg_sync*` | typed KG entities, relations, Graphiti episodes; 저장소 간 허브 — CommuLingo·리서치·사료 문서가 external_ids로 미러됨 (`dev_docs/knowledge_graph_design.md`) |
| Redis | `memory_store/redis_state.py` | live task progress/state, active task registry, mission board, task-chain summaries |
| R2 | `shared.py`, publication/runtime tools | public uploaded files and generated media |

## Vector Corpus Maintenance Backlog

`lenin_corpus` is mixed-generation data. New ingestion should record at least `layer`, `author`, `title`, `source`, `source_url`/`public_url` when available, `year`, `language`, `chunk_size`, `chunk_overlap`, `chunk_index`, and `chunk_count`. Author names should use canonical names that match KG usage where possible.

Current known cleanup targets:

- `core_theory` Marx & Engels, Lenin, Rosa Luxemburg, Trotsky, and Gramsci were reingested on the Windows GPU host from local `docs/` source files with corrected `title`, `year`, `source_url`, `language`, `chunk_size`, `chunk_overlap`, `chunk_index`, and `chunk_count` metadata. Non-work index/abstract/study-guide rows encountered during the pass were pruned.
- `modern_analysis` has been reingested from Korean organization documents under `docs/modern_analysis/` (`bolky_`, `diamat_`, `uprising_`) with `language=ko`, `chunk_size=1800`, `chunk_overlap=200`, `title`, `author`, `organization`, source URLs when present, and file paths. arXiv/BIS/MXO material is intentionally excluded from this layer.
- Mao has been removed from `core_theory` pending clean reingestion with canonical metadata and larger chunks.

Current default chunking for new corpus ingestion is language-specific in `corpus/store.py`: English/default texts use larger chunks, Korean texts use smaller chunks because Hangul text is denser per character.

## Current Module Ownership

| Area | Modules |
|---|---|
| Identity and prompt rendering | `identity/prompts.py`, `identity/agent_prompts/*.md`, `llm/prompt_renderer.py`, `agents/base.py` |
| LLM provider config | `bot_config.py`, `llm/agent_loop.py` (shared loop engine), `llm/claude_loop.py`, `llm/openai_tool_loop.py`, `llm/client.py` |
| Personal fiction workspace | `writer/` package (store/documents/models/prompts/tools/runs/stream), `writer/static/writer.html`, `/writer/*` routes in `api_routes/writer.py`, `services/novel_writer_api.py` |
| Agents | `agents/*.py`, `config/agent_runtime.json`, `api_routes/task_reports.py` |
| Tools | `runtime_tools/registry.py` (global schemas and registration), general tools in `runtime_tools/*`, domain tools in their packages (`commulingo/people.py`, `publishing/*`, `mail_runtime/tools.py`, `roleplay/memory.py`), `self_runtime/tools.py`, `crypto_wallet/*` |
| KG facade | `kg_runtime/search.py`, `kg_runtime/writes.py`, `kg_runtime/admin.py`, `kg_runtime/service_runtime.py` |
| KG implementation | `graph_memory/service.py`, `graph_memory/entities.py`, `graph_memory/edges.py`, `graph_memory/structured_writer.py` |
| Public content | `publishing/research_store.py`, `publishing/site_publishing.py`, `publishing/publication_records.py`, `publishing/research.py`, `publishing/post_edit.py`, `api_routes/private_reports.py` (JSON), frontend `/admin/private-reports` shell |
| Hub 큐레이션 (`/curate`) | `telegram/curate.py` (URL 정규화·중복 검사·태스크 등록, 쓰기 경계 검증 래퍼, 결정적 결과 DM), `agents/hub_curator.py` (DeepSeek V4.1 Flash 작성자 스펙, `CURATION_LIMITS` 단일 출처), `publishing/site_publishing.py` (`publish_hub_curation` 툴, `hub_curations` 테이블) |
| CommuLingo 인물·용어 사전 | `commulingo/people.py` (read + six target-specific narrow writes; shared normalization, structured errors; direct/staging switch in `config/commulingo_people.json`), `tool_gateway/profiles.py` + `agents/analyst.py` (Telegram direct/delegated narrow-write surfaces), `agents/commulingo_curator.py` + `scripts/commulingo_people_maintainer.py` / `scripts/commulingo_terms_maintainer.py` (typed discovery and stage-scoped scheduled direct maintenance; 레인 공용 설정·카운터·단계 실행기는 `commulingo/lane.py`, 인물 레인 선택·프롬프트·`run_once`는 `commulingo/people_lane.py`), `scripts/commulingo_suggestions.py` (staging 리뷰 CLI). 인물/절 저장·승인은 frontend 공통 서비스의 Docker RPC를 사용한다. 필수 버전·근거·검토·보강 상태는 [인물 편집 계약](commulingo_editorial.md), 데이터/렌더링은 별도 저장소 `/home/grass/frontend/dev_docs/commulingo_people_handoff.md` 참고 |
| 웹 검색 gateway | `leninbot-web-gateway.service` (`127.0.0.1:8111`) owns Tavily/Brave keys, routing, cache/coalescing and the $10/UTC-day budget. `runtime_tools/web_search.py` and paid `content_fetch/urls.py` extraction use `web_gateway/client.py`; no direct-provider fallback. Policy: `config/web_research.json`; private ledger: `/var/lib/leninbot-web-gateway/usage.sqlite3`. See [web_research.md](web_research.md). |
| Admin user API routes | `api_routes/admin_users.py` |
| Chat history/API routes | `api_routes/chat_history.py`, `services/chat_history_sanitize.py`, `services/web_chat.py` |
| Email bridge | `services/email_bridge.py`, `api_routes/email.py`, `services/email_api.py`, `scripts/email_poll_once.py` |
| A2A API | `services/a2a_api.py`, `services/a2a_handler.py` |
| Fetch/browser | `content_fetch/*`, `browser/*`, `runtime_tools/fetch.py`, `runtime_tools/media.py` |
| Inbound MCP gateway | `mcp_gateway/*`, `scripts/smoke_mcp_gateway.py` |
| Tool security gateway | `security_gateway/*`, `tool_gateway.dispatcher`/`validation`, Postgres `tool_idempotency`, `scripts/security_gateway.py`, `scripts/smoke_security_gateway.py` |

`shared.py` holds only `KST`, `upload_to_r2` and the static `MODULE_ARCHITECTURE` self-description; the old compatibility re-exports were removed (2026-09-24). Import from the domain modules above instead of growing `shared.py`. Per-turn runtime context (`current_task_ctx`, runtime prelude, context-block join/merge) lives in `llm/runtime_context.py` so the browser worker and A2A API do not import `telegram.bot`; `telegram.bot` re-exports the same objects. The owner chat runtime (`chat_with_tools`, orchestrator system prompt, system alerts, per-agent provider/model dispatch) lives in `telegram/chat_runtime.py`; `jobs`, `runtime_tools` and tests import it from there rather than from the `telegram.bot` entry point, which re-exports it under the old private names. Checkout and frontend locations come from `ops/paths.py`: `PROJECT_ROOT`/`VENV_PYTHON` derive from the checkout, `FRONTEND_DIR` (default `/home/grass/frontend`) locates the separate frontend repository, and `commulingo_data_file()` keeps the per-file `COMMULINGO_*` overrides. The four CommuLingo contract JSON files the frontend owns are vendored in `config/commulingo_contracts/` and used only when `FRONTEND_DIR` does not exist, so a lone checkout (e.g. a cloud session) imports and passes the unit suite; `scripts/sync_commulingo_contracts.py` refreshes the copies and `tests/test_commulingo_contract_copies.py` fails when they drift. Checked-in configs name paths as `${FRONTEND_DIR}/…`/`${PROJECT_ROOT}/…`, resolved by `expand_path_tokens()`. Route modules are service boundaries only when included by a dedicated service entrypoint such as `services/novel_writer_api.py`, `services/email_api.py`, or `services/a2a_api.py`; the remaining route modules are imported into `services/api.py` and run inside `leninbot-api.service`.

## Operational Entry Points

- Service status: `systemctl status leninbot-api.service novel-writer-api.service leninbot-email-api.service leninbot-a2a-api.service leninbot-telegram.service leninbot-browser.service leninbot-roleplay.service`
- Deploy: `deploy.sh` → `scripts/svc deploy [--api|--telegram|--frontend|--all] [--restart]` (Telegram `/deploy`). It pulls `main`, runs `pip install` only when `requirements*` changed, runs `daemon-reload` when `systemd/` changed, then restarts `api` and `telegram` through `scripts/restart_guard.py` (refuses while work is active), `browser` if active, and hands the frontend to the frontend repository's `scripts/deploy`. It does **not** copy unit files into `/etc/systemd/system/` (do `sudo cp systemd/<unit> /etc/systemd/system/` + `daemon-reload` yourself) and does **not** restart `leninbot-roleplay`, `leninbot-a2a-api`, `novel-writer-api`, `leninbot-email-api`, `leninbot-llm-proxy` or `leninbot-web-gateway`.
- Restart after moving or renaming a module: running processes keep old code and fail at their next function-level import of the old path, so restart every long-running service that imports the package (usually telegram, roleplay, api, a2a-api, novel-writer-api, browser, email-api), after `restart_guard.py` for telegram/api. Check that `systemctl show <unit> -p MainPID` changed. `scripts/svc restart <name>` stops rather than kills when sudo does not allow `systemctl kill` (as for `grass`); `sudo -n systemctl restart <unit>` is allowed directly.
- Service liveness alerting: `leninbot-service-health.timer` (see `monitoring.md`).
- Logs: `journalctl -u <unit> -f`
- Telegram connectivity watchdog: `telegram/bot.py` probes `get_me()` every `TELEGRAM_CONNECTIVITY_WATCHDOG_SECONDS` seconds, using `TELEGRAM_CONNECTIVITY_PROBE_TIMEOUT_SECONDS` as the per-probe timeout. Owner-facing degraded/restored notifications are emitted only after `TELEGRAM_CONNECTIVITY_NOTIFY_AFTER_FAILURES` consecutive failures.
- Static page smoke tests: `scripts/smoke_static_pages.py`
- Runtime smoke tests: `scripts/smoke_runtime.py`, `scripts/smoke_tool_allowlists.py`, `scripts/smoke_webchat_security.py`, `scripts/smoke_kg_schema_docs.py`
- MCP gateway smoke test: `scripts/smoke_mcp_gateway.py`
- Secret management: `scripts/manage_secrets.py`
- Schema migrations: `scripts/schema_migrations.py`
- Model/provider audit: `scripts/model_runtime_audit.py`
- KG maintenance: `scripts/check_kg_integrity.py`, `scripts/kg_enricher.py`, `skills/kg-maintenance/scripts/*`
- Vector corpus ingestion: `scripts/ingest_literature.py` for ad hoc modern analysis drops; ignored one-off helpers under `scripts/ingest_*` may be used for curated reingestion and should call `corpus.store.ingest_to_corpus` with the intended layer.
- Embedding runtime: `services/embedding_server.py` exposes BGE-M3 over HTTP. `EMBEDDING_DEVICE=auto` uses CUDA when available and falls back to CPU; set `EMBEDDING_OFFLINE=1` on Windows GPU ingestion hosts with cached Hugging Face models to avoid startup network checks. `EMBEDDING_PRELOAD_RERANKER=0` lets the embedding service start even when the reranker model is not cached.
- vector_search는 cross-encoder 리랭크를 쓰지 않는다 (2026-07-28 제거): CPU에서 후보 15개 리랭크에 17~40초가 걸려 p50 30초의 원인이었고, 클라이언트 30초 타임아웃으로 결과가 버려지는 경우도 많았다. pgvector 코사인 순서를 그대로 쓰고, 교차언어 병합은 metadata의 `similarity` 점수 내림차순 정렬. 측정: 단일 검색 ~0.2-0.3초, 이중언어 ~0.4초+번역 LLM. 서버의 `/rerank` 엔드포인트는 남아 있으나 미사용.

## Design Notes

- **도메인 패키지 (2026-09-25 정리)**: CommuLingo 코드는 `commulingo/`(편집 도구·레인·조사 기억·`pipeline/`), 공개 콘텐츠는 `publishing/`(보고서 저장소·게시·검토, 정적 페이지, 게시물 편집, 비공개 보고서, 게시 기록, 자율 게시 통제, Cloudflare 퍼지, 채널 방송), 역할극 봇은 `roleplay/`, 사료 번역과 번역 메모리는 `translation_runtime/`, 메일 도구는 `mail_runtime/`에 있다. `runtime_tools/`는 전역 도구 등록(`registry.py`)과 도메인에 속하지 않는 범용 도구만 둔다. 소유자 채팅 루프는 `telegram/chat_runtime.py`이며, 라이브러리 코드는 `telegram.bot` 진입점을 import하지 않는다. 그 밖에 `task_store`는 `telegram/`, `redis_state`는 `memory_store/`, `prompt_context`·`skills_loader`는 `llm/`, `audit_sink`는 `ops/`, `self_modification_core`는 `self_runtime/`에 있다. 옛 경로의 호환 shim은 없으므로 import와 patch 대상은 패키지 경로를 쓴다. 루트에 남은 Python 모듈은 `bot_config`, `db`, `secrets_loader`, `shared`뿐이다. 라이브러리 패키지는 `scripts/`를 import하지 않는다.
- **알고 유지하는 것**: `bot_config.ANTHROPIC_CLIENT_KEY`/`MOONSHOT_CLIENT_KEY`는 쓰는 곳이 없지만 OPENAI/DEEPSEEK 쪽과 모양을 맞추려고 둔다. `bot_config.set_gateway_enforce_mode`는 코드 호출부가 없지만 `security_gateway.md`의 운영 진입점이다. `_slice_text`(`self_runtime/tools.py`, `mcp_gateway/tools.py`)는 offset이 길이를 넘을 때 반환 위치가 달라 합치지 않았다. standby 백업·복원 스크립트(`scripts/backup_*_to_r2.py`, `scripts/restore_db.py`)는 독립 실행을 위해 `KST`를 자체 정의한다.
- Telegram is the only full orchestrator path. Web chat has a narrower tool set and separate webchat provider settings.
- **Hub 큐레이션 `/curate` (2026-09-06)**: 소유자가 Telegram 봇 DM에 `/curate <url> [메모]`를 보내면 `telegram/curate.py`가 SSRF 검사·`hub_curations` 중복 검사(추적 파라미터·www·끝 슬래시 정규화) 후 `telegram_tasks`에 `agent_type="hub_curator"` 행을 넣는다. 일반 태스크 워커가 `agents/hub_curator.py`(DeepSeek V4.1 Flash, terminal=`publish_hub_curation`)를 실행하고, 완료 시 오케스트레이터 LLM 보고 대신 DB 행 존재 여부로 판정한 결정적 DM(`report_curation_outcome`)을 보낸다. 자유 대화 오케스트레이터에는 publish 툴을 노출하지 않고 `/curate`로 안내만 한다. 발행 시 `maybe_broadcast_autonomous_publication`이 그대로 동작하므로 `TELEGRAM_BROADCAST_SITE_ENABLED`가 켜져 있으면 확성기 채널에도 알림이 나간다.
- **Agent tool loop (2026-08-04)**: the round-loop/forced-final control flow shared by every provider lives once in `llm.agent_loop.run_tool_loop`; `llm/claude_loop.py` (Anthropic protocol: Claude, DeepSeek) and `llm/openai_tool_loop.py` (OpenAI protocol: GPT, Kimi, local) are protocol adapters with unchanged public `chat_with_tools` signatures. Loop control-flow fixes go in the engine once — never mirrored across the two adapter files. Contract regressions are caught by `tests/test_agent_loop_engine.py` + the two `test_*_loop_rounds.py` suites; details in `llm_provider_architecture.md`.
- `config.json` stores mutable runtime config. `config/agent_runtime.json` overlays per-agent execution settings.
- Prompt text under `identity/agent_prompts/` hot-reloads on the next prompt render. Python code, tool definitions, and systemd credentials require service restart.
- Public web chat provider is pinned independently with `webchat_provider` and `webchat_model`; Telegram `/config` changes do not necessarily affect API until `leninbot-api` restarts.
- Public web-chat generation은 run 시작 뒤 서버가 소유한다. `/chat` SSE 연결은 진행 상황만 관찰하므로 browser/proxy disconnect가 LLM 생성을 취소하지 않으며, 성공한 detached run은 `chat_logs`에 저장되고 frontend가 session/persona 범위 history polling으로 복구한다. In-process run registry는 같은 session의 중복 재시도를 막고 detached run을 global concurrency 계산에 유지한다. Web chat `vector_search`에는 `WEBCHAT_VECTOR_SEARCH_TIMEOUT_SECONDS`(기본 45초) 상한이 적용된다.
- `/writer`는 `novel-writer-api.service` (`:8001`)의 개인 소설 작업 공간이다. DB는 같은 `leninbot-pg` 안의 별도 `writer` DB다. 도구·문맥·캐시·진단→저자 수정은 [writer_runtime.md](writer_runtime.md), 백업은 [db_migration_plan.md](db_migration_plan.md)를 따른다.
- Inbound MCP is an on-demand stdio gateway for developer/operator clients, not a public API route. `MCP_GATEWAY_PROFILE=inspect` is the default. `operator` adds `readonly_query_db`, `bounded_query_db`, and `kg_maintenance_run`, each delegated to existing guarded project scripts/tools.
- Services do not run startup DDL. Apply `scripts/schema_migrations.py` before deploying code that depends on new tables, columns, indexes, or constraints. The roleplay bot's tables are the `roleplay-tables` migration (`ensure_roleplay_tables` in `telegram/schema.py`).
- Provider-loop runtime tool calls funnel through `tool_gateway.dispatcher.execute_tool`, where the **tool security gateway** validates the provider-visible JSON Schema, authorizes against one policy, atomically consumes capped Redis rate limits, and emits audit events for `tool_audit_log` + structured journal output. `ops/audit_sink.py` forwards DB audit rows to the proxy; MCP has its own dispatch boundary (see `mcp_gateway.md`). Unknown taxonomy, invalid arguments, authorization errors, capped rate-store outages, and stable-scope idempotency-store outages fail closed. Successful scoped side effects are recorded in Postgres `tool_idempotency`; handler exceptions become `outcome_unknown` and are not replayed. Owner-gating and over-cap decisions still follow `gateway_enforce_mode`, while web-chat/A2A restrictions, taxonomy denial, and rate-store outage denial are always enforced. Inspect with `scripts/security_gateway.py {policy,check,audit}`. Full design: `dev_docs/security_gateway.md`.
- `leninbot-roleplay.service` is a **separate identity**, not Cyber-Lenin. It runs `roleplay/bot.py`: owner-gated, DeepSeek over the Anthropic-compatible endpoint via `claude_loop` (thinking on, kept out of replies), a hot-reloaded persona at `identity/roleplay_persona.md`, its own isolated chat tables, and a narrow read-only tool set (see `tool_allowlist_current_state.md`). Runtime config is the `ROLEPLAY_*` env vars; the bot token is `ROLEPLAY_BOT_TOKEN`.

번역 실행의 공통 소유자는 `translation_runtime/`이다. 기타 DB 번역 최신성은 원문 테이블의 `translation_source_sha256`가 관리한다. 반복 검증 실패의 임시 보류 기록은 `output/translation_failures/`에 둔다. 2026-09-07 설치한 systemd 정의는 `scripts/run_translation_batch.py`로 두 작업의 실패를 합산하며 `ignore_errors=no`를 확인했다. 사료·연구 Markdown·DB JSON 어댑터가 공통 호출/검증 재시도를 사용하고, DeepL 정적 페이지는 별도 HTTP 경로에서 HTML 검증을 공유한다. 운영 `leninbot.research_documents.markdown_en_source_sha256` 마이그레이션은 2026-09-07 적용·검증했다. 형식별 경계, 타이머 정의, 캐시·TM·원문 최신성 및 평가 방법은 [Translation Pipeline](translation_pipeline.md)을 따른다.

## Mail briefing state

`mail_runtime/` persists raw mail, per-task body coverage and per-audience Telegram delivery receipts. `check_inbox` defaults to unbriefed mail in delegated tasks; scout can stage exact per-mail summaries with the write tool `prepare_mail_briefing`. Successful task callbacks send these summaries and record each accepted message, bypassing model rewriting. See [mail_briefing.md](mail_briefing.md) for tool semantics, migration and failure behavior.
