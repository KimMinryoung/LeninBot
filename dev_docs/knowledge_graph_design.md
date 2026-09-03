# Knowledge Graph Design

최종 확인 기준: 2026-09-03 코드 트리 (저장소 간 허브 재설계).

Cyber-Lenin's knowledge graph is the **hub across the project's knowledge stores**: CommuLingo people/terms/events, published research and archival documents, and news/analysis facts written by agents all live in one Neo4j graph. Every mirrored node carries stable external ids, so the same real-world entity converges on one node regardless of which store or agent mentioned it first. The public runtime talks through `kg_runtime/`; Graphiti/Neo4j implementation details live under `graph_memory/`.

## Why the 2026-09-03 redesign

Measured on 2026-09-03: growth had collapsed (new edges/month 2,917 in March → 192 in August), 105 searches in 11 weeks against 95k CommuLingo tool calls, 13% orphan nodes, 42% empty summaries, the same entity split across `group_id`s (Donald Trump ×5), Korean webchat queries against an English news graph, and search results that showed facts without subject/object/date/source. The redesign attacks each: an identity layer, deterministic mirror jobs, an upgraded read path, publish hooks, and health metrics. Design rationale and the full before/after plan: `~/.claude/plans/cozy-whistling-tower.md` (session record).

## Ownership Boundary

| Layer | Modules | Responsibility |
|---|---|---|
| Runtime facade | `kg_runtime/service_runtime.py` | singleton lifecycle, dedicated event loop, retry/cooldown, health checks |
| Identity | `kg_runtime/identity.py` | alias-key normalization, deterministic entity resolution, identity upsert, node merge, in-process `AliasIndex` |
| Runtime operations | `kg_runtime/search.py`, `writes.py`, `admin.py`, `scout_ingest.py`, `recall.py`, `doc_extract.py`, `metrics.py` | search/entity view/hydration, structured writes, admin, scout episode ingest, entity-gated recall, document extraction, health metrics |
| Sync jobs | `jobs/kg_sync.py`, `jobs/kg_sync_commulingo.py`, `jobs/kg_sync_documents.py` | nightly mirrors with `kg_sync_state` watermarks |
| Graph implementation | `graph_memory/service.py` | Graphiti initialization, Neo4j access, episode ingestion, hybrid search, post-episode cross-group merge |
| Schema | `graph_memory/entities.py`, `edges.py`, `config.py`, `conformance.py`, `structured_writer.py` | entity/edge models, type mapping, sync-only subsets, deterministic structured writes, integrity checks |
| Tool surface | `runtime_tools/registry.py` (`knowledge_graph_search`), `self_runtime/tools.py` (`write_kg_structured`) | agent-facing read/write |

New code should import from `kg_runtime/*` unless it is changing the KG implementation itself.

## Runtime Model

Graphiti/Neo4j objects are sensitive to event-loop ownership. `kg_runtime/service_runtime.py` creates one persistent KG event loop thread and runs KG async operations on that loop:

- `get_kg_service()` lazily initializes `GraphMemoryService`
- `run_kg_task()` runs one async callable on the KG loop and blocks for the result
- `submit_kg_task()` submits work and returns a `Future`
- `collect_kg_futures()` waits for multiple submitted tasks
- transient connection failures mark the singleton unhealthy and apply a 120-second retry cooldown

`graph_memory/service.py` uses the models registered in `config/llm_call_sites.json` (`kg_extraction_main`, `kg_extraction_small`, `kg_embedding`; see `dev_docs/llm_call_registry.md`). Embedding calls go through a bounded retry wrapper for transient `429`/`503` (`KG_EMBED_RETRY_DELAYS`, default `5,15,45`) and a client-side pacer (`KG_EMBED_MAX_RPS`, default 2 req/s, `0` disables). The structured writer pre-embeds nodes and facts in batches of 50 (`_embed_in_batches`) so a 200-fact sync batch costs a handful of embedding requests, not hundreds. KG uses the proxy-owned shared `GEMINI_API_KEY`. Hermetic smoke: `scripts/smoke_kg_embed_limiter.py`.

Direct Cypher (no Graphiti) goes through `kg_runtime.search._get_neo4j_sync_driver()`; `NEO4J_PASSWORD` comes from `secrets_loader.get_secret` (systemd credential), so ad-hoc runs export it from `/run/credentials/<unit>/neo4j_password`.

API startup eagerly initializes KG in a background thread and starts a periodic health check, but callers must still tolerate `get_kg_service()` returning `None`.

Robustness rules (2026-07-17):

- `GraphMemoryService.initialize()` runs entirely inside its async init lock and only publishes `_graphiti` after `build_indices_and_constraints()` (and `ensure_identity_indexes()`) succeed; on failure the Neo4j driver is closed before re-raising.
- `reset_kg_service()` does not stop the KG event loop; only the service singleton is discarded.
- Transient-error keyword lists contain no bare module names.

## Identity Layer (2026-09-03)

Every Entity may carry `external_ids`, `aliases`, `alias_keys`, `name_ko`, `name_en`, `alias_text` (field-level detail in `knowledge_graph_schema.md` §1.1). They ride on `EntityNode.attributes`; graphiti saves attributes with `SET n = $entity_data` and the patched record loader (`graph_memory/graphiti_patches.py`) keeps unknown properties, so they survive graphiti re-saves. **Any graphiti upgrade must re-verify this round-trip.**

`kg_runtime.identity.resolve_entity_{sync,async}` is the single resolver for the structured writer and the sync jobs:

1. `external_id ∈ n.external_ids` — definitive, label ignored
2. normalized key (`normalize_alias_key`: NFKC, lowercase, punctuation/hyphens → space, `NAME_NORMALIZATION` as the final map) ∈ `alias_keys`, or exact/lowercase name — any `group_id`, **same label required**; a different-label hit is logged (`label_conflict`) and not reused
3. `KG_RESOLVE_EMBEDDING_NN=1` (default off): name-embedding nearest neighbour on `entity_name_embedding`, cosine ≥ `KG_RESOLVE_EMBEDDING_NN_THRESHOLD` (0.92), same label

`upsert_identity_*` unions ids/aliases into an existing node (and fills an empty summary); `merge_entity_nodes_*` is the one merge implementation (edges moved with same-predicate dedupe, MENTIONS moved, summary filled, identity unioned, duplicate DETACH DELETEd) — `skills/kg-maintenance/scripts/merge_entities.py` and `kg_runtime.admin.kg_merge_entities` delegate to it. After every free-text episode, `post_episode_merge` folds new nodes into same-label nodes that share an alias key across groups (graphiti only searches resolution candidates inside the episode's own `group_id`, which is how one entity became five nodes).

`AliasIndex` caches `alias_key → [(uuid, name, labels)]` in-process (one Cypher, 10-minute TTL). `match(text)` finds entities named in free text without any embedding call: Korean keys (≥2 chars) match as substrings because particles attach to names, Latin keys (≥4 chars) as whole tokens, longest match wins.

`group_id` is a domain tag on episodes and edges; entity resolution ignores it. Sync entities use `commulingo` / `documents` (`config.SYNC_GROUP_IDS`), which are **not** in the `write_kg_structured` enum.

## Schema Summary

Primary entity types: `Person`, `Organization`, `Location`, `Asset`, `Incident`, `Policy`, `Campaign`, `Concept`, `Role`, `Industry`, plus the sync-only `Document`.

Primary relation predicates: `Affiliation`, `PersonalRelation`, `OrgRelation`, `Funding`, `AssetTransfer`, `ThreatAction`, `Involvement`, `Presence`, `PolicyEffect`, `Participation`, `Statement`, `Causation`, plus the sync-only `Reference` (`reference_type` = about / mentions / collection / related_term / parent_term / category / person_term / event_term / people_group).

The edge compatibility map lives in `graph_memory/config.py` as `EDGE_TYPE_MAP` (unchanged, handed to the graphiti extractor). `Reference` pairs live separately in `REFERENCE_EDGE_PAIRS` / `sync_predicate_allowed()`. `SYNC_ONLY_ENTITY_TYPES` / `SYNC_ONLY_PREDICATES` are removed from the extractor schema (`service.EXTRACTION_ENTITY_TYPES` / `EXTRACTION_EDGE_TYPES`) and from `validate_fact` unless `allow_sync_predicates=True`. The conformance gate knows the same rule.

See `knowledge_graph_schema.md` for field-level schema details and the mirror mapping table.

## Write Paths

- **`write_kg_structured`** (agents): deterministic typed triples. Entities are resolved through the identity layer, so a fact about "Nikita Khrushchev" attaches to the CommuLingo node "니키타 흐루쇼프" via its `name_en` alias. Per-fact `attributes` (dict) are now stored on the edge; `invalid_at` is accepted. Sync-only identity hints (`subject_external_id`, `subject_aliases`, `subject_summary`, `subject_name_ko/en`, and the `object_*` twins — `IDENTITY_FACT_FIELDS`) are honoured by `write_structured_facts` but not exposed in the tool schema.
- **Sync jobs** (`python -m jobs.kg_sync --source commulingo,documents [--full] [--limit N] [--dry-run]`): deterministic, no LLM for CommuLingo; every edge carries `attributes.sync_key`, re-runs are idempotent, changed facts expire the old edge and write a new one, vanished rows are expired on full passes (automatic every 7 days or `--full`). `kg_sync_state` (Postgres) holds the per-source watermark. Runs nightly at 04:00 KST via `systemd/leninbot-kg-sync.timer`.
  - CommuLingo → Person/Role/Concept/Incident/Location with Korean canonical names, `name_en`/cyrillic/curated aliases, curated summaries; `commulingo_id_redirects` merge or alias the old id. Incremental runs use `updated_at` columns and `commulingo_people_revisions`.
  - Documents → `Document` nodes for public research documents, the archival manifest (`$FRONTEND_DIR/data/commulingo/docs/manifest.json`, whose curated people/terms/events slugs become `Reference(about)` edges to the `commulingo:*` nodes — the archival ↔ CommuLingo hub link) and autonomous `synthesis` notes; `Reference(mentions)` for alias-index hits in title/description/opening text. With `KG_DOC_EXTRACT_LLM=1` the registry site `kg_document_extraction` (gemini-3.5-flash-lite, JSON) adds ≤15 agent-schema facts per document with `attributes.doc_ref` provenance. Idempotent per `content_sha256` stored on the Document node.
- **Publish hook**: `runtime_tools/research.py` schedules `doc_extract.extract_research_by_slug` after a public publish/edit (fire-and-forget; the nightly job is the backstop).
- **Scout episodes**: `kg_runtime/scout_ingest.py` (free-text graphiti extraction, group classified by `scout_kg_classify`); episode names now carry `-t<task_id>`.
- `write_kg` (LLM extraction tool) stays deprecated.

Typical `group_id` values for agent writes: `geopolitics_conflict`, `diplomacy`, `economy`, `korea_domestic`, `agent_knowledge`. Do not create diary-specific or project-specific groups for ordinary facts.

## Read Paths

`knowledge_graph_search(query, num_results, entity=, mode=auto|entity|semantic)`:

1. **Alias match first** (no embedding): the query is matched against `AliasIndex`. Exactly one entity → **entity view**: the node (aliases, external id, summary) plus its 1-hop neighbourhood, active facts first then expired, capped at 25. `entity=` forces this for a named entity; `mode=entity` without a hit returns no result.
2. Otherwise Graphiti hybrid search (BM25 + cosine, RRF) as before, with a small entity view prepended for up to two alias hits.
3. Every edge is **hydrated** in one Cypher pass (`_hydrate_edges`): subject/object + labels, predicate, `valid_at`/`invalid_at`/`expired_at`, trust tier parsed from the episode name (both `[T:x]` and the sanitized `T-x` form — the pre-redesign parser only knew `T-`, so every structured-write edge showed `?`), and a source label (`commulingo`, `research:<slug>`, `scout`, `analyst`, `news`, …).
4. Output: `- [tier|expired] 주어 —Predicate→ 목적어: fact (valid 1953-01-01 → 1964-10-01; src: commulingo)` and `- 이름 [Person] (aka: …; id: commulingo:person:x): summary`.
5. Graphiti failures still fall back to the direct-Cypher text match (now including `alias_text`).

**Entity-gated recall** (`KG_ENTITY_GATED_RECALL=1`, default off): `kg_runtime.recall.entity_gated_kg_block(text, provider)` renders a `<knowledge-graph>` block (Markdown for non-Claude) for up to two entities named in a message/task — alias match + neighbourhood, zero LLM/embedding cost. Injected next to the experience block in `telegram/commands.py`, `telegram/tasks.py` and `services/web_chat.py`. The same flag lets `commulingo_people get_person` append `kg_facts` (non-mirror facts attached to the person's node).

Tool exposure: `knowledge_graph_search` is available to analyst, diary, general, autonomous, the Telegram/web/A2A/MCP surfaces, and since 2026-09-03 to both CommuLingo curators and the writer surface (`WRITER_TOOLS`). Vector corpus search (`vector_search`) stays separate.

## Maintenance

Scripts:

- `scripts/check_kg_integrity.py [--smoke-query …] [--metrics] [--notify]` — hourly timer; `--metrics` adds `kg_runtime.metrics.collect_kg_metrics()`
- `scripts/kg_weekly_report.py [--notify]` — Monday 09:30 KST timer, same Telegram channel as the integrity check
- `scripts/smoke_kg_sync.py` — live but isolated end-to-end check of the mirror (scratch namespace/group, cleaned up)
- `scripts/smoke_kg_search_modes.py [--out FILE]` — renders representative queries for before/after comparison
- `scripts/kg_backfill_summaries.py [--execute]` — fills empty summaries from the node's best facts (no LLM)
- `skills/kg-maintenance/scripts/merge_exact_name_dupes.py` / `merge_entities.py` — duplicate cleanup (now through `identity.merge_entity_nodes_sync`)
- `skills/kg-maintenance/scripts/backup_kg.py` / `restore_kg.py` — now carry every extra node property (`props`), so identity/Document props survive a restore; `scripts/backup_kg_to_r2.py` daily
- `scripts/kg_enricher.py`, `scripts/ingest_reports_to_kg.py` — dormant
- `mcp_gateway.tools`: `kg_integrity_check` (inspect) and `kg_maintenance_run` (operator, `execute=true` + `confirm=APPLY_KG_MAINTENANCE`)

Feature flags (all default off; see `.env.example`): `KG_ENTITY_GATED_RECALL`, `KG_DOC_EXTRACT_LLM`, `KG_RESOLVE_EMBEDDING_NN`.

Operational rules:

- MCP `inspect` may check KG integrity but must not mutate KG.
- KG connection failures should degrade features, not crash Telegram/API.
- Ad-hoc processes: the Postgres write guard covers `kg_sync_state` (needs `LENINBOT_ALLOW_WRITE=1` or a systemd unit); Neo4j writes are not guarded — back up first (`scripts/backup_kg_to_r2.py`) and prefer `--dry-run`.
- If `check_kg_integrity.py --smoke-query` reports degraded search with `429 RESOURCE_EXHAUSTED`, rerun after a short wait; repeated failures usually mean Gemini embedding quota pressure.
- Any schema expansion must update entity/edge models, `EDGE_TYPE_MAP` or `REFERENCE_EDGE_PAIRS`, write-tool descriptions, `knowledge_graph_schema.md` and this document (`scripts/smoke_kg_schema_docs.py` guards drift).
- Tests: `tests/test_kg_identity.py`, `test_kg_sync_mapping.py`, `test_kg_search_format.py`, `test_kg_doc_extract.py` (hermetic, `scripts/run_unit_tests.sh kg_`).
