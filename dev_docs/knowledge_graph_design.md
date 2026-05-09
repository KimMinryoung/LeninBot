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

API startup eagerly initializes KG in a background thread and starts a periodic health check, but callers must still tolerate `get_kg_service()` returning `None`.

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
- `autonomous_project_<id>` for project-local facts

Do not store transient internal task chatter in the KG. Use task logs, mission events, or autonomous notes for that.

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

Operational rules:

- KG connection failures should degrade features, not crash Telegram/API.
- Integrity changes should be tested against `graph_memory/config.py` and `knowledge_graph_schema.md`.
- Any schema expansion must update entity/edge models, `EDGE_TYPE_MAP`, write-tool descriptions, and this document.
