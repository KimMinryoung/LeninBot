# Project State

최종 확인 기준: 2026-07-08 코드 트리.

Cyber-Lenin은 하나의 런타임 정체성을 여러 인터페이스로 노출하는 시스템이다. 주요 사용자 인터페이스는 Telegram bot, public web chat API, scheduled autonomous/diary/background workers다. 장기 상태는 PostgreSQL/Supabase와 Neo4j에 저장하고, Redis는 실행 중인 task 상태와 mission board 같은 단기 공유 상태를 맡는다.

## Runtime Map

```
cyber-lenin.com frontend
        |
        v
leninbot-api (:8000, FastAPI)
        |-- /chat, /chat/feedback, /personas -> api.py + web_chat.py
        |-- admin/chat-history/report/private-report JSON -> api_routes/* modules
        |
        +--> PostgreSQL/Supabase
        +--> Neo4j KG via kg_runtime/
        +--> embedding_server.py (:8100)

novel-writer-api (:8001, FastAPI)
        |-- /writer -> api_routes.writer -> writer/ package
        |
        +--> local writer PostgreSQL via WRITER_DB_*
        +--> Anthropic/DeepSeek writer model clients

leninbot-email-api (:8002, FastAPI)
        |-- /email/* -> api_routes.email -> email_bridge.py
        |
        +--> PostgreSQL/Supabase email tables
        +--> IMAP/Resend email provider config

leninbot-a2a-api (:8003, FastAPI)
        |-- /.well-known/agent-card.json + /a2a -> a2a_api.py + a2a_handler.py
        |
        +--> PostgreSQL/Supabase
        +--> Neo4j KG via kg_runtime/
        +--> provider/tool runtime

Telegram
        |
        v
telegram/bot.py orchestrator
        |-- runtime_tools/ registry and allow-lists
        |-- telegram/tasks.py background task worker
        |-- agents/* AgentSpec registry
        |
        +--> PostgreSQL task/chat/mission tables
        +--> Redis task progress, active task state, mission board
        +--> Neo4j KG

systemd timers
        |-- leninbot-autonomous.timer -> autonomous_project.py
        |-- leninbot-experience.timer -> experience_writer.py
        |-- leninbot-kg-integrity.timer -> scripts/check_kg_integrity.py
        |-- research-document-translation.timer -> scripts/static_page_translation_pipeline.py
        |-- leninbot-email-poller.timer -> scripts/email_poll_once.py
        |-- leninbot-commulingo-maintainer.timer -> scripts/commulingo_people_maintainer.py

developer MCP clients
        |
        v
        python -m mcp_gateway.server (stdio, on demand)
        |-- explicit MCP profile allow-list
        |-- read-only adapters over runtime_tools, dev_docs, task/corpus state
        |-- operator-only readonly_query_db and bounded_query_db via existing DB guards
```

## Service Units

| Unit | Entrypoint | Role |
|---|---|---|
| `leninbot-neo4j.service` | Docker Compose | Neo4j and Redis backing services |
| `leninbot-embedding.service` | `embedding_server.py` | local embedding HTTP service |
| `leninbot-telegram.service` | `telegram/bot.py` | Telegram orchestrator and task worker |
| `leninbot-roleplay.service` | `telegram/roleplay_bot.py` | standalone roleplay companion bot, independent of Cyber-Lenin |
| `leninbot-api.service` | `uvicorn api:app` | web chat, chat history, admin users, task reports, private report JSON, x402 demo; remaining `api_routes/*` here are code modules in this same process |
| `novel-writer-api.service` | `uvicorn novel_writer_api:app` | isolated personal fiction writer API and writer SSE runs |
| `leninbot-email-api.service` | `uvicorn email_api:app` | admin-gated `/email/*` review, approval, draft, and manual poll API |
| `leninbot-email-poller.timer` | `scripts/email_poll_once.py` | periodic IMAP polling into the email bridge tables |
| `leninbot-a2a-api.service` | `uvicorn a2a_api:app` | public A2A discovery and JSON-RPC endpoint |
| `leninbot-browser.service` | `browser/worker.py` | browser automation worker over Unix socket |
| `leninbot-autonomous.service` | `venv/bin/python -m autonomous_project` | one autonomous project tick |
| `leninbot-experience.service` | `experience_writer.py` | daily experience memory write |
| `leninbot-kg-integrity.service` | `scripts/check_kg_integrity.py` | KG maintenance check |
| `leninbot-commulingo-maintainer.service` | `scripts/commulingo_people_maintainer.py` | one direct, sourced CommuLingo person edit through the dedicated DeepSeek V4 Pro curator |

Dependency direction is simple: Neo4j/Redis and embedding start before Telegram/API; browser starts after Telegram. API can optionally run Telegram in-process only when `RUN_TELEGRAM_IN_API=true`, but production uses the dedicated Telegram unit.

## API Boundary Status

`api_routes/*` does not automatically mean a separate service. As of 2026-07-08, writer, email, and A2A are real service boundaries; the remaining extracted route modules are code-ownership boundaries inside `leninbot-api.service` and still share one Uvicorn process, one port (`:8000`), and the main API systemd credentials/runtime.

| Surface | Runtime boundary | Code owner | Notes |
|---|---|---|---|
| Writer workspace `/writer/*` | separate `novel-writer-api.service` on `:8001` | `novel_writer_api.py`, `api_routes/writer.py`, `writer/` | frontend `/api/proxy/writer` targets this service; `leninbot-api.service` does not include writer routes |
| Email bridge `/email/*` | separate `leninbot-email-api.service` on `:8002` plus `leninbot-email-poller.timer` | `email_api.py`, `api_routes/email.py`, `email_bridge.py`, `scripts/email_poll_once.py` | frontend `/api/proxy/email` targets this service; periodic polling no longer depends on main API requests |
| A2A discovery/JSON-RPC | separate `leninbot-a2a-api.service` on `:8003` | `a2a_api.py`, `a2a_handler.py` | frontend proxies `/.well-known/agent-card.json` and `/a2a` to this service; main API no longer includes A2A routes |
| Public web chat `/chat`, `/chat/feedback`, `/personas` | same `leninbot-api.service` process | `api.py`, `web_chat.py`, `web_personas.py` | kept in the main API because it shares session locks, fingerprint/proxy identity, persona visibility, feedback/regeneration, and rate limiting |
| Admin users `/admin/users*` | same `leninbot-api.service` process | `api_routes/admin_users.py` | code moved out of `api.py`; URLs/auth unchanged |
| Chat logs/history `/logs`, `/history`, `/sessions`, `/session/{id}` | same `leninbot-api.service` process | `api_routes/chat_history.py` | code moved out of `api.py`; frontend may serve some history from its local DB first |
| Task reports `/reports*` | same `leninbot-api.service` process | `api_routes/task_reports.py` | code moved out of `api.py`; URLs/auth unchanged |
| Private report JSON `/private-reports*` | same `leninbot-api.service` process | `api_routes/private_reports.py` | browser shell moved to frontend `/admin/private-reports`; FastAPI only serves JSON |


## Main Data Stores

| Store | Owner modules | Stored state |
|---|---|---|
| PostgreSQL/Supabase | `db.py`, `task_store.py`, `memory_store/*`, `autonomous_project.py`, `email_bridge.py`, `security_gateway/audit.py`, `writer/store.py` | chat logs, task queue, missions, reports, autonomous projects, email metadata, vector corpus metadata, writer projects/messages/manuscripts/revisions, `tool_audit_log` (per-call security audit) |
| pgvector | `corpus/*`, `memory_store/experiential.py` | core theory, modern analysis, self-produced analysis, experience memory vectors |
| Neo4j | `graph_memory/*`, `kg_runtime/*` | typed KG entities, relations, Graphiti episodes |
| Redis | `redis_state.py` | live task progress/state, active task registry, mission board, task-chain summaries |
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
| LLM provider config | `bot_config.py`, `claude_loop.py`, `openai_tool_loop.py`, `llm/client.py` |
| Personal fiction workspace | `writer/` package (store/documents/models/prompts/tools/runs/stream; `creative_writer.py` compat shim), `frontend/writer.html`, `/writer/*` routes in `api_routes/writer.py`, `novel_writer_api.py` |
| Agents | `agents/*.py`, `config/agent_runtime.json`, `api_routes/task_reports.py` |
| Tools | `runtime_tools/*`, `self_runtime/tools.py`, `crypto_wallet/*` |
| KG facade | `kg_runtime/search.py`, `kg_runtime/writes.py`, `kg_runtime/admin.py`, `kg_runtime/service_runtime.py` |
| KG implementation | `graph_memory/service.py`, `graph_memory/entities.py`, `graph_memory/edges.py`, `graph_memory/structured_writer.py` |
| Public content | `research_store.py`, `site_publishing.py`, `publication_records.py`, `runtime_tools/research.py`, `runtime_tools/post_edit.py`, `api_routes/private_reports.py` (JSON), frontend `/admin/private-reports` shell |
| CommuLingo 인물 사전 | `runtime_tools/commulingo_people.py` (read + `commulingo_edit`; direct/staging 모드 스위치 `config/commulingo_people.json`), `agents/commulingo_curator.py` + `scripts/commulingo_people_maintainer.py` (scheduled one-person direct maintenance), `scripts/commulingo_suggestions.py` (staging 리뷰 CLI). 데이터/렌더링은 frontend 저장소 — `frontend/dev_docs/commulingo_people_handoff.md` 참고 |
| Admin user API routes | `api_routes/admin_users.py` |
| Chat history/API routes | `api_routes/chat_history.py`, `chat_history_sanitize.py`, `web_chat.py` |
| Email bridge | `email_bridge.py`, `api_routes/email.py`, `email_api.py`, `scripts/email_poll_once.py` |
| A2A API | `a2a_api.py`, `a2a_handler.py` |
| Fetch/browser | `content_fetch/*`, `browser/*`, `runtime_tools/fetch.py`, `runtime_tools/media.py` |
| Inbound MCP gateway | `mcp_gateway/*`, `scripts/smoke_mcp_gateway.py` |
| Tool security gateway | `security_gateway/*`, `tool_loop_common.execute_tool` (seam), `scripts/security_gateway.py`, `scripts/smoke_security_gateway.py` |

`shared.py` is now a compatibility facade plus a small set of shared helpers. New implementation should import from the domain modules above instead of growing `shared.py`. Route modules are service boundaries only when included by a dedicated service entrypoint such as `novel_writer_api.py`, `email_api.py`, or `a2a_api.py`; the remaining route modules are imported into `api.py` and run inside `leninbot-api.service`.

## Operational Entry Points

- Service status: `systemctl status leninbot-api.service novel-writer-api.service leninbot-email-api.service leninbot-a2a-api.service leninbot-telegram.service leninbot-browser.service`
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
- Embedding runtime: `embedding_server.py` exposes BGE-M3 over HTTP. `EMBEDDING_DEVICE=auto` uses CUDA when available and falls back to CPU; set `EMBEDDING_OFFLINE=1` on Windows GPU ingestion hosts with cached Hugging Face models to avoid startup network checks. `EMBEDDING_PRELOAD_RERANKER=0` lets the embedding service start even when the reranker model is not cached.

## Design Notes

- Telegram is the only full orchestrator path. Web chat has a narrower tool set and separate webchat provider settings.
- `config.json` stores mutable runtime config. `config/agent_runtime.json` overlays per-agent execution settings.
- Prompt text under `identity/agent_prompts/` hot-reloads on the next prompt render. Python code, tool definitions, and systemd credentials require service restart.
- Public web chat provider is pinned independently with `webchat_provider` and `webchat_model`; Telegram `/config` changes do not necessarily affect API until `leninbot-api` restarts.
- `/writer` is an admin-gated personal fiction workspace, separate from the public chat/persona/model-tier surfaces. Its routes live in `api_routes/writer.py` and are exposed by dedicated `novel-writer-api.service` (`uvicorn novel_writer_api:app`, port 8001); frontend `/api/proxy/writer` targets the dedicated writer service. It stores `writer_projects`, `writer_messages`, `writer_manuscripts`, searchable manuscript chunks, manuscript revision metadata, and `writer_documents` (per-project background/reference documents — worldbuilding, character sheets, outlines — kept separate from the manuscript), audits writer tool calls through `security_gateway/audit.py` into the main `tool_audit_log` table, then calls Claude Fable 5 directly through the `writer/` package (`creative_writer.py` remains as a compatibility shim). **Writer tables live in the local Docker Postgres `leninbot-writer-pg`** (PG17, `127.0.0.1:5433`, volume `leninbot_writer_pg_data`, `WRITER_DB_*` in `.env`, pool in `db.get_writer_conn`/`writer_query*`) — migrated off Supabase 2026-07-07 because its ~560ms RTT (Hetzner Helsinki ↔ AWS AP) made every manuscript op cost 1–4s; local ops are 4–20ms. The Supabase copies were renamed to `writer_*_migrated_20260707` so a missing `WRITER_DB_*` config fails loudly instead of silently serving stale data. Durability: `leninbot-writer-backup.timer` (daily 03:20 KST) runs `scripts/backup_writer_db_to_r2.py` — `docker exec pg_dump -Fc`, TOC verify, upload to `cyber-lenin-backups` R2 (7-day retention) + local `data/writer_db_backups/` (3-day); restore drill verified. The writer's tool surface is declared as the `system.writer` profile in `tool_gateway/profiles.py`. The writer agent's tools are `search_manuscript`/`read_manuscript`/`append_to_manuscript`/`replace_in_manuscript` plus `read_document`/`search_documents`/`save_document` and `research_web` (`WRITER_WEB_SEARCH_ENABLED`, on); search and replace tolerate whitespace/quote drift (incl. 「」『』). Easy work is delegated to light DeepSeek agents (`writer/research.py`, `writer.models.resolve_light_model`; falls back to the main model when DeepSeek is unconfigured): `research_web` spawns a `deepseek_flash` sub-agent that runs the actual Tavily `web_search` rounds and returns only a distilled, `<external>`-wrapped brief into heavy-model context (sub-run cost folds into the message cost via the run registry's `extra_costs`), and the 퇴고 diagnosis/line-edit editor role runs on `deepseek_pro` regardless of the selected main model. Writer model calls use the central `WRITER_CALL_POLICIES` registry (`writer/config.py`) for role-specific input/output token ceilings, tool rounds, output continuations, and thinking policy. Heavy `main` and `revision` calls share `tool_gateway.inference` defaults (160k input, 32k output); diagnosis, line-edit, and research keep smaller role-specific limits. Diagnosis has an 8k per-response output ceiling and uses the shared `bot_config._get_deepseek_tool_thinking_params()` policy (default non-thinking; configurable through `DEEPSEEK_TOOL_THINKING_MODE`). Input overflow preserves durable state through pinned `Story so far`/background documents and replays exact prose through `read_manuscript(chapter, start_anchor, end_anchor)`; scene locators use chapter headings plus boundary sentences rather than mutable character offsets. Old large tool results are replaced by explicit replay instructions only after the configured input ceiling would otherwise be exceeded. Output `max_tokens` opens bounded continuation rounds even when truncation occurs on the final normal tool round. Generation runs as a server-side background task registered in an in-process run registry; browsers reattach to a live run via `GET /writer/projects/{id}/stream`. The admin's model choice persists in `writer_settings` (choices: `fable` adaptive-thinking effort-high, `fable_fast` effort-low, `deepseek_pro`, `deepseek_flash`); history is budgeted to ~30k chars with a **quantized window** (start advances only in 15k-char jumps, `CONTEXT_WINDOW_QUANTUM_CHARS` — a per-turn slide changed the message prefix every request and defeated prompt caching) and `pinned` documents (story-so-far synopsis) are injected in full each turn (3×8k). **Cache-aware prompt layout (2026-07-07)**: the cached 1h-TTL prefix carries only stable content — tools, the craft system prompt, and the style guide (`build_system_blocks`) — while ALL volatile manuscript state (char count, opening/tail excerpts, selection, document listing with STALE marks, pinned docs) rides in a `<manuscript_state>` block inside the current turn's user message (`manuscript_state_block`, never persisted with the prompt). Before this, the volatile state sat in system block 2: every Fable turn re-wrote ~60k cache tokens at 2× input price and read ~0 back (~$1.17 of every ~$1.30 turn was dead cache writes, incl. refusal turns that produced 3 output tokens); live-verified after the change that a manuscript edit + new history between calls leaves the prefix 100% cache-hit. **Style guide documents (kind `style`, 2×12k)** are injected in full into the writer, diagnosis, and line-edit contexts as the binding prose calibration target — exemplar passages (rhythm/diction absorption, never copying), contrast pairs (금지→지향, the writer's own corrections, ranked above generic craft rules), and operational rules; the system prompt tells the agent to distill chat style feedback into this document in the same turn (`writer/prompts.py: style_guide_parts`). The system prompt carries Korean literary craft guidance; context includes a 3.5k opening excerpt plus a 16k tail, and the document listing marks docs as STALE when the manuscript has grown ≥2k chars past their last save (the memory-discipline nudge — the prompt tells the agent to record durable facts into `Characters`/`Timeline`/etc. docs and distilled `Research — <topic>` notes the same turn). An optional per-turn 퇴고 pass (UI checkbox → `critic` flag) runs in the mode set by `WRITER_REVISION_MODE` (default **`diagnose_revise`**, 2026-07-07): stage 1, a light `deepseek_pro` editor DIAGNOSES the turn's changed spans (구조/극화/리듬/문장/문체/연속성; read-only tools; numbered notes with quoted anchors, direction-not-wording; replying `PASS` skips stage 2), then stage 2, the MAIN model revises as the author — with the FULL writer tool surface and the same system blocks as the main pass so its request reads the prompt cache the main pass just wrote (appending is forbidden by the revision request itself), plus a fresh post-edit `<manuscript_state>` (doc listing + pinned, no excerpts) since this stage owns the story-bible refresh — explicitly free to reject notes that would hurt the voice — critique flows to the strongest prose model instead of a weaker model rewriting its sentences (Reflexion-style; agents `writer_diagnosis`/`writer_revision`, `writer/stream.py: run_diagnose_revise_pass`). A failed diagnosis degrades to author self-diagnosis; legacy mode `line_edit` (single `deepseek_pro` line edit, `run_line_edit_pass`) remains as rollback. Both stages' cost/usage merge into the same assistant message and their edits into the same highlight spans (`done.critic` carries `mode`). Apply `venv/bin/python scripts/schema_migrations.py --only writer-tables` before first use or after writer schema changes.
- Inbound MCP is an on-demand stdio gateway for developer/operator clients, not a public API route. `MCP_GATEWAY_PROFILE=inspect` is the default. `operator` adds `readonly_query_db`, `bounded_query_db`, and `kg_maintenance_run`, each delegated to existing guarded project scripts/tools.
- Services do not run startup DDL. Apply `scripts/schema_migrations.py` before deploying code that depends on new tables, columns, indexes, or constraints. The roleplay bot's tables are the `roleplay-tables` migration (`ensure_roleplay_tables` in `telegram/schema.py`).
- Every tool call from every interface funnels through `tool_loop_common.execute_tool`, where the **tool security gateway** (`security_gateway/*`) authorizes the call against one unified policy and writes a `tool_audit_log` row + structured journal line. The gateway is fail-open on internal error. New rules (owner-gating, rate limits) ship in **shadow** mode by default (`gateway_enforce_mode` in `config.json`); web-chat interface restriction is always enforced (it already mirrors the pre-filter). Inspect with `scripts/security_gateway.py {policy,check,audit}`. Full design: `dev_docs/security_gateway.md`.
- `leninbot-roleplay.service` is a **separate identity**, not Cyber-Lenin. It runs `telegram/roleplay_bot.py`: owner-gated, DeepSeek over the Anthropic-compatible endpoint via `claude_loop` (thinking on, kept out of replies), a hot-reloaded persona at `identity/roleplay_persona.md`, its own isolated chat tables, and a narrow read-only tool set (see `tool_allowlist_current_state.md`). Runtime config is the `ROLEPLAY_*` env vars; the bot token is `ROLEPLAY_BOT_TOKEN`.
