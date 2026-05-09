# Project State

최종 확인 기준: 2026-05-09 코드 트리.

Cyber-Lenin은 하나의 런타임 정체성을 여러 인터페이스로 노출하는 시스템이다. 주요 사용자 인터페이스는 Telegram bot, public web chat API, scheduled autonomous/diary/background workers다. 장기 상태는 PostgreSQL/Supabase와 Neo4j에 저장하고, Redis는 실행 중인 task 상태와 mission board 같은 단기 공유 상태를 맡는다.

## Runtime Map

```
cyber-lenin.com frontend
        |
        v
leninbot-api (:8000, FastAPI)
        |-- /chat -> web_chat.py
        |-- admin APIs -> chat logs, task reports, private reports, email bridge
        |-- /.well-known/agent-card.json + /a2a
        |
        +--> PostgreSQL/Supabase
        +--> Neo4j KG via kg_runtime/
        +--> embedding_server.py (:8100)

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
```

## Service Units

| Unit | Entrypoint | Role |
|---|---|---|
| `leninbot-neo4j.service` | Docker Compose | Neo4j and Redis backing services |
| `leninbot-embedding.service` | `embedding_server.py` | local embedding HTTP service |
| `leninbot-telegram.service` | `telegram/bot.py` | Telegram orchestrator and task worker |
| `leninbot-api.service` | `uvicorn api:app` | web chat, admin API, email/A2A/private reports |
| `leninbot-browser.service` | `browser/worker.py` | browser automation worker over Unix socket |
| `leninbot-autonomous.service` | `scripts/autonomous_work.py` | one autonomous project tick |
| `leninbot-experience.service` | `experience_writer.py` | daily experience memory write |
| `leninbot-kg-integrity.service` | `scripts/check_kg_integrity.py` | KG maintenance check |

Dependency direction is simple: Neo4j/Redis and embedding start before Telegram/API; browser starts after Telegram. API can optionally run Telegram in-process only when `RUN_TELEGRAM_IN_API=true`, but production uses the dedicated Telegram unit.

## Main Data Stores

| Store | Owner modules | Stored state |
|---|---|---|
| PostgreSQL/Supabase | `db.py`, `task_store.py`, `memory_store/*`, `autonomous_project.py`, `email_bridge.py` | chat logs, task queue, missions, reports, autonomous projects, email metadata, vector corpus metadata |
| pgvector | `corpus/*`, `memory_store/experiential.py` | core theory, modern analysis, self-produced analysis, experience memory vectors |
| Neo4j | `graph_memory/*`, `kg_runtime/*` | typed KG entities, relations, Graphiti episodes |
| Redis | `redis_state.py` | live task progress/state, active task registry, mission board, task-chain summaries |
| R2 | `shared.py`, publication/runtime tools | public uploaded files and generated media |

## Current Module Ownership

| Area | Modules |
|---|---|
| Identity and prompt rendering | `identity/prompts.py`, `identity/agent_prompts/*.md`, `llm/prompt_renderer.py`, `agents/base.py` |
| LLM provider config | `bot_config.py`, `claude_loop.py`, `openai_tool_loop.py`, `llm/client.py` |
| Agents | `agents/*.py`, `config/agent_runtime.json` |
| Tools | `runtime_tools/*`, `self_runtime/tools.py`, `crypto_wallet/*` |
| KG facade | `kg_runtime/search.py`, `kg_runtime/writes.py`, `kg_runtime/admin.py`, `kg_runtime/service_runtime.py` |
| KG implementation | `graph_memory/service.py`, `graph_memory/entities.py`, `graph_memory/edges.py`, `graph_memory/structured_writer.py` |
| Public content | `research_store.py`, `site_publishing.py`, `publication_records.py`, `runtime_tools/research.py`, `runtime_tools/post_edit.py` |
| Fetch/browser | `content_fetch/*`, `browser/*`, `runtime_tools/fetch.py`, `runtime_tools/media.py` |

`shared.py` is now a compatibility facade plus a small set of shared helpers. New implementation should import from the domain modules above instead of growing `shared.py`.

## Operational Entry Points

- Service status: `systemctl status leninbot-api.service leninbot-telegram.service leninbot-browser.service`
- Logs: `journalctl -u <unit> -f`
- Static page smoke tests: `scripts/smoke_static_pages.py`
- Runtime smoke tests: `scripts/smoke_runtime.py`, `scripts/smoke_tool_allowlists.py`, `scripts/smoke_webchat_security.py`, `scripts/smoke_kg_schema_docs.py`
- Secret management: `scripts/manage_secrets.py`
- Schema migrations: `scripts/schema_migrations.py`
- Model/provider audit: `scripts/model_runtime_audit.py`
- KG maintenance: `scripts/check_kg_integrity.py`, `scripts/kg_enricher.py`, `skills/kg-maintenance/scripts/*`

## Design Notes

- Telegram is the only full orchestrator path. Web chat has a narrower tool set and separate webchat provider settings.
- `config.json` stores mutable runtime config. `config/agent_runtime.json` overlays per-agent execution settings.
- Prompt text under `identity/agent_prompts/` hot-reloads on the next prompt render. Python code, tool definitions, and systemd credentials require service restart.
- Public web chat provider is pinned independently with `webchat_provider` and `webchat_model`; Telegram `/config` changes do not necessarily affect API until `leninbot-api` restarts.
- Services do not run startup DDL. Apply `scripts/schema_migrations.py` before deploying code that depends on new tables, columns, indexes, or constraints.
