# Knowledge Graph Design

최종 확인 기준: 2026-05-09 코드 트리.

Cyber-Lenin's knowledge graph stores structured geopolitical, economic, organizational, policy, and internal agent knowledge. The public runtime talks through `kg_runtime/`; Graphiti/Neo4j implementation details live under `graph_memory/`.

## Ownership Boundary

| Layer | Modules | Responsibility |
|---|---|---|
| Runtime facade | `kg_runtime/service_runtime.py` | singleton lifecycle, dedicated event loop, retry/cooldown, health checks |
| Runtime operations | `kg_runtime/search.py`, `writes.py`, `admin.py`, `scout_ingest.py` | safe search/write/admin functions exposed to tools and scripts |
| Graph implementation | `graph_memory/service.py` | Graphiti initialization, Neo4j access, episode ingestion, search |
| Schema | `graph_memory/entities.py`, `edges.py`, `config.py`, `conformance.py`, `structured_writer.py` | entity/edge models, type mapping, deterministic structured writes, integrity checks |
| Tool surface | `runtime_tools/registry.py` | `knowledge_graph_search`, `write_kg_structured`, related handlers |

New code should import from `kg_runtime/*` unless it is changing the KG implementation itself.

## Runtime Model

Graphiti/Neo4j objects are sensitive to event-loop ownership. `kg_runtime/service_runtime.py` creates one persistent KG event loop thread and runs KG async operations on that loop:

- `get_kg_service()` lazily initializes `GraphMemoryService`
- `run_kg_task()` runs one async callable on the KG loop and blocks for the result
- `submit_kg_task()` submits work and returns a `Future`
- `collect_kg_futures()` waits for multiple submitted tasks
- transient connection failures mark the singleton unhealthy and apply a 120-second retry cooldown

`graph_memory/service.py` uses Gemini `gemini-3.1-flash-lite` as the primary Graphiti LLM and `gemini-2.5-flash-lite` as the small model. Graphiti semantic search and structured writer embeddings use `gemini-embedding-001` through a bounded retry wrapper for transient Gemini/Vertex `429 RESOURCE_EXHAUSTED` and `503 UNAVAILABLE` responses. The retry delays default to `5,15,45` seconds and can be overridden with `KG_EMBED_RETRY_DELAYS` as a comma-separated seconds list. Since 2026-07-16 (roadmap Phase 6) embedding calls are also **paced client-side** by a serializing scheduler (`KG_EMBED_MAX_RPS`, default 2 req/s, `0` disables; re-read from the env on every request) so batch producers (scout ingest, `kg_enricher`, curation) don't burst `SEMAPHORE_LIMIT`-wide into quota that retries then paper over. The KG service resolves its Gemini key as `KG_GEMINI_API_KEY` → fallback `GEMINI_API_KEY` (`_resolve_kg_gemini_key`); the dedicated key isolates KG traffic onto its own quota **only if it comes from a separate Google project/account** — same-project keys share quota. Not yet provisioned; the fallback keeps everything on the shared key until it is. Hermetic smoke: `scripts/smoke_kg_embed_limiter.py`.

API startup eagerly initializes KG in a background thread and starts a periodic health check, but callers must still tolerate `get_kg_service()` returning `None`.

Robustness rules (2026-07-17):

- `GraphMemoryService.initialize()` runs entirely inside its async init lock and only publishes `_graphiti` after `build_indices_and_constraints()` succeeds; on failure the Neo4j driver is closed before re-raising.
- `reset_kg_service()` no longer stops the KG event loop — stopping it silently killed in-flight ingests whenever a search-side connection blip triggered a reset. Only the service singleton (and its driver, closed asynchronously on the loop) is discarded.
- Transient-error keyword lists no longer contain bare module names (`neo4j`, `graphiti`) — an error message merely mentioning a module must not tear down the service.
- Legacy OSINT pipelines were removed (`news_fetcher.py`, `kr_news_fetcher.py`, `generate_briefing`, `query_active_wars`, `ingest_episodes_bulk`); `query_chatbot` stays as the CLI diagnostic path (`python -m graph_memory.cli`).

## Schema Summary

Primary entity types:

- `Person`
- `Organization`
- `Location`
- `Asset`
- `Incident`
- `Policy`
- `Campaign`
- `Concept`
- `Role`
- `Industry`

Primary relation predicates:

- `Affiliation`
- `PersonalRelation`
- `OrgRelation`
- `Funding`
- `AssetTransfer`
- `ThreatAction`
- `Involvement`
- `Presence`
- `PolicyEffect`
- `Participation`
- `Statement`
- `Causation`

The edge compatibility map lives in `graph_memory/config.py` as `EDGE_TYPE_MAP`. Unknown or broad pairs can fall back through the `("Entity", "Entity")` mapping for general predicates.

See `knowledge_graph_schema.md` for field-level schema details.

## Write Paths

Use `write_kg_structured` for new agent/tool writes. It is deterministic: the caller supplies subject name/type, predicate, object name/type, and fact text. This avoids free-form extraction drift for high-value facts and lets agents reuse exact entity names.

Typical `group_id` values:

- `geopolitics_conflict`
- `diplomacy`
- `economy`
- `korea_domestic`
- `agent_knowledge`

Use shared topic groups for reusable knowledge, including facts discovered by autonomous or diary agents. Do not create diary-specific or project-specific KG groups for ordinary facts; keep transient working notes in task logs, mission events, diary drafts, or autonomous project notes instead.

Scout report ingestion (`kg_runtime/scout_ingest.py`) routes each report into a group via a light Gemini call (`SCOUT_KG_CLASSIFY_MODEL`, default `gemini-2.5-flash-lite`, temperature 0) instead of keyword substring matching; on any failure it falls back to `agent_knowledge` rather than blocking ingestion.

## Search Paths

`knowledge_graph_search` is the normal tool-facing read path. It should preserve canonical names and languages when possible. For Korean organizations/publications or people, prefer the canonical Korean name when that is how the node is stored.

Vector corpus search is separate (`vector_search`) and should not be treated as KG evidence unless the result is explicitly written to the graph.

## Maintenance

Relevant scripts:

- `scripts/check_kg_integrity.py`
- `scripts/kg_enricher.py`
- `scripts/ingest_reports_to_kg.py`
- `scripts/ingest_pending_curations.py`
- `scripts/fix_legacy_kg_edges.py`
- `skills/kg-maintenance/scripts/*`
- `mcp_gateway.tools`: `kg_integrity_check` for inspect profile and `kg_maintenance_run` for operator-only bounded cleanup

Operational rules:

- MCP `inspect` may check KG integrity but must not mutate KG.
- MCP `operator` may run bounded maintenance scripts through `kg_maintenance_run`; mutating actions require `execute=true` plus `confirm=APPLY_KG_MAINTENANCE`, and direct mutating actions run a backup first.
- KG connection failures should degrade features, not crash Telegram/API.
- If `scripts/check_kg_integrity.py --smoke-query ...` reports degraded search with `429 RESOURCE_EXHAUSTED`, first rerun after a short wait and check service logs. A passing rerun indicates transient Gemini/Vertex rate pressure; repeated failures over several minutes usually require checking the Google AI Studio / Google Cloud quota page for the API key's `gemini-embedding-001` request quota and requesting an increase or moving KG embeddings to a separate key/project.
- Integrity changes should be tested against `graph_memory/config.py` and `knowledge_graph_schema.md`.
- Any schema expansion must update entity/edge models, `EDGE_TYPE_MAP`, write-tool descriptions, and this document.
