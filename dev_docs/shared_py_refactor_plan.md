# shared.py Refactor Completion Record

Date: 2026-05-05

This document started as the handoff plan for splitting `shared.py` into clearer
runtime/service modules without changing behavior. The refactor is now complete
as of commits `cc37594`, `6fce55b`, and the follow-up tool-module move
`38ab93b`.

## Current State

`shared.py` is now a compatibility facade, not an implementation owner. New
code should import from the domain modules below rather than from `shared.py`.

As of 2026-05-05 the implementation ownership is:

- `identity/prompts.py`: `CORE_IDENTITY`, `AGENT_CONTEXT`, `EXTERNAL_SOURCE_RULE`
- `provenance/runtime.py`: `ProvenanceBuffer`, `init_provenance_buffer`, `_wrap_external`
- `runtime_tools/broadcast.py`: Telegram channel broadcast tool definition and handler
- `kg_runtime/service_runtime.py`: KG singleton lifecycle and dedicated KG event loop
- `kg_runtime/search.py`, `writes.py`, `admin.py`, `scout_ingest.py`: KG facade helpers
- `memory_store/queries.py`: PostgreSQL memory/task report queries
- `task_store.py`: `create_task_in_db`
- `ops/logs.py`: server log fetch/grep helpers
- `corpus/embeddings.py`, `store.py`, `public_index.py`: vector corpus and public-output indexing
- `memory_store/experiential.py`: experiential memory search/write
- `content_fetch/urls.py`, `browser_pool.py`, `documents.py`: URL fetching, Playwright pool, document conversion
- `shared.py`: R2 upload helper plus compatibility re-exports

The file is approximately 224 lines.

## Original KG Boundary Rationale

There is already a `graph_memory/` package, but it has a different ownership:

- `graph_memory/service.py` owns the Graphiti-backed service implementation:
  schema setup, Neo4j/Gemini initialization, episode ingestion, Graphiti search,
  conformance checks, and briefing generation.
- `graph_memory/entities.py`, `edges.py`, `config.py`, `conformance.py`, and
  `structured_writer.py` own KG domain schema and write mechanics.
- Before this refactor, `shared.py` had grown as the cross-runtime access layer used by web chat,
  Telegram, agents, scripts, and tools. It handles singleton lifecycle,
  cross-event-loop safety, sync/async wrappers, trust-tier display, DB-facing
  convenience functions, and legacy import compatibility.

So the split was not arbitrary. `graph_memory/` remains the implementation
package; `kg_runtime/` now owns the cross-runtime KG facade.

## Original Design Problems

1. `shared.py` had too many owners.

   Prompt constants, KG runtime, corpus indexing, URL fetching, task creation,
   R2 upload, and Telegram broadcast live together. This makes every import of a
   small helper conceptually depend on the whole platform.

2. The dependency direction was blurred.

   `shared.py` imported `telegram.channel_broadcast` inside
   `broadcast_to_channel`. A generic shared module should not know a Telegram
   implementation detail. Runtime tools should own tool-facing wrappers.

3. Tool wrappers and tool implementations were split oddly.

   `runtime_tools/fetch.py` defined tools, but the heavy implementation was
   in `shared.py` (`fetch_url_content_async`, `convert_document`,
   Playwright pool). That makes the tool package a thin wrapper over a monolith.

4. KG access had multiple concerns in one place.

   Dedicated loop lifecycle, service singleton, search formatting, write
   wrappers, direct Cypher admin functions, and scout ingestion should not be in
   one module.

5. “shared” was not a stable public API.

   Call sites imported many internals directly, including `_wrap_external`,
   `_get_exp_embeddings`, `run_kg_task`, and `kg_merge_entities`. This makes
   refactoring harder because every helper looks public.

## Target Shape

The target packages below were implemented. Compatibility re-exports remain in
`shared.py` for legacy callers, but migrated call sites should use the domain
modules directly.

Implemented packages:

```text
identity/
  prompts.py              # CORE_IDENTITY, AGENT_CONTEXT, EXTERNAL_SOURCE_RULE

provenance/
  runtime.py              # ProvenanceBuffer, init/get buffer, _wrap_external

kg_runtime/
  service_runtime.py      # get_kg_service, reset, healthcheck, KG event loop
  search.py               # search_knowledge_graph, tier lookup formatting
  writes.py               # add_kg_episode(_async), add_kg_structured(_async)
  admin.py                # kg_cypher, kg_delete_episode, kg_merge_entities
  scout_ingest.py         # process_scout_report_to_kg

memory_store/
  queries.py              # fetch_diaries, fetch_chat_logs, fetch_task_reports
  experiential.py         # search_experiential_memory, save_experiential_memory

corpus/
  embeddings.py           # embedding singleton
  store.py                # similarity_search, ingest_to_corpus, corpus helpers
  public_index.py         # public_self_analysis_source/index/delete

content_fetch/
  urls.py                 # fetch_url_content(_async), diagnosis, extract_urls
  browser_pool.py         # Playwright loop/context/cookie pool
  documents.py            # convert_document

ops/
  logs.py                 # fetch_server_logs and grep helpers

task_store.py             # create_task_in_db

runtime_tools/
  broadcast.py            # BROADCAST_TO_CHANNEL_TOOL, broadcast_to_channel
  db.py                   # query_db tool
  post_edit.py            # edit_public_post tool
  research.py             # research publish/edit/unpublish tools
  private_reports.py      # private report tools
  x.py                    # X/Twitter fetch tool

self_runtime/
  tools.py                # delegate/multi_delegate/run_agent/read_self/self tools
```

Names changed slightly where the implemented ownership was clearer: `task_store.py` is top-level, log helpers live in `ops/logs.py`, and root-level tool modules were moved under `runtime_tools/` or `self_runtime/`.

## Migration Record

### Phase 0 Completed: Baseline Before Editing

These were run before and during the move:

```bash
venv/bin/python -m py_compile shared.py runtime_tools/*.py self_runtime/*.py telegram/*.py web_chat.py a2a_handler.py browser/worker.py
venv/bin/python scripts/smoke_tool_allowlists.py
venv/bin/python scripts/smoke_filesystem_tools.py
```

Also run import scans:

```bash
rg -n "from shared import|import shared" --glob '*.py'
rg -n "^(def|async def|class) " shared.py
```

### Phase 1 Completed: Extract Pure Constants And Provenance

Lowest-risk first move:

- Move `CORE_IDENTITY`, `AGENT_CONTEXT`, `EXTERNAL_SOURCE_RULE` to
  `identity/prompts.py`.
- Move `ProvenanceBuffer`, `get_provenance_buffer`,
  `init_provenance_buffer`, and `_wrap_external` to `provenance/runtime.py`.
- Keep re-exports in `shared.py`:

```python
from identity.prompts import CORE_IDENTITY, AGENT_CONTEXT, EXTERNAL_SOURCE_RULE
from provenance.runtime import ProvenanceBuffer, get_provenance_buffer, init_provenance_buffer, _wrap_external
```

Then migrate call sites from `shared` to the new modules in a separate commit.

### Phase 2 Completed: Extract Content Fetch

Move URL/document fetching into `content_fetch/`:

- `extract_urls`, `diagnose_url_fetch_failure`, `fetch_url_content`,
  `fetch_url_content_async` -> `content_fetch/urls.py`
- Playwright loop/context/cookie code -> `content_fetch/browser_pool.py`
- `convert_document` -> `content_fetch/documents.py`

Important path note:

- `_PW_COOKIE_PATH` is currently next to `shared.py`, i.e.
  `/home/grass/leninbot/.pw_cookies.json`.
- If moving to a package under repo root, do not accidentally make it
  `/home/grass/leninbot/content_fetch/.pw_cookies.json`.
- Use an explicit project-root helper or env var:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PW_COOKIE_PATH = PROJECT_ROOT / ".pw_cookies.json"
```

After this phase, `runtime_tools/fetch.py` should import implementation from
`content_fetch`, not from `shared`.

### Phase 3 Completed: Extract KG Runtime Facade

Keep `graph_memory/` as the domain implementation. Extract the runtime facade
from `shared.py` into `kg_runtime/`.

Move in this order:

1. `service_runtime.py`
   - `_kg_service`, `_kg_lock`, `_kg_loop`, `_ensure_kg_loop`,
     `run_kg_task`, `submit_kg_task`, `collect_kg_futures`,
     `get_kg_service`, `reset_kg_service`, `start_kg_healthcheck`
2. `writes.py`
   - `add_kg_episode`, `add_kg_episode_async`,
     `add_kg_structured`, `add_kg_structured_async`
3. `search.py`
   - `_get_neo4j_sync_driver`, `fetch_kg_stats`,
     `search_knowledge_graph`
4. `admin.py`
   - `kg_cypher`, `kg_delete_episode`, `kg_merge_entities`
5. `scout_ingest.py`
   - `process_scout_report_to_kg`

Reason for this order: write/search/admin all depend on the service runtime.

Do not move the actual `GraphMemoryService` class out of `graph_memory/service.py`.
That package should remain the KG implementation boundary.

### Phase 4 Completed: Extract Corpus And Embeddings

Move vector/corpus concerns:

- `set_shared_embeddings`, `_get_exp_embeddings` -> `corpus/embeddings.py`
- `search_experiential_memory`, `save_experiential_memory` ->
  `memory_store/experiential.py` or `corpus/experiential.py`
- `similarity_search`, `_chunk_text`, `ingest_to_corpus`,
  `fetch_corpus_source_context`, `delete_corpus_source` ->
  `corpus/store.py`
- `save_self_produced_analysis`, `public_self_analysis_source`,
  `index_public_self_analysis`, `delete_public_self_analysis_index` ->
  `corpus/public_index.py`

After this phase, `runtime_tools/registry.py` should import vector search
implementation from `corpus`, not from `shared`.

### Phase 5 Completed: Extract DB Memory/Task Helpers

Move PostgreSQL helper functions:

- `fetch_diaries`, `fetch_chat_logs`, `fetch_task_reports` ->
  `memory_store/queries.py`
- `create_task_in_db` -> `telegram/task_store.py` or top-level
  `task_store.py`
- `fetch_server_logs`, `_normalize_grep_terms`, `grep_matches_text` ->
  `ops/logs.py` or `runtime_tools/logging_support.py`

Be careful with naming: `create_task_in_db` writes to `telegram_tasks`, but it
is used by more than Telegram command handling. A neutral `task_store.py` may be
better than putting it under `telegram/`.

### Phase 6 Completed: Move Broadcast Tool Out Of shared.py

Move:

- `BROADCAST_TO_CHANNEL_TOOL`
- `broadcast_to_channel`

to `runtime_tools/broadcast.py` or `runtime_tools/social.py`.

Then update `runtime_tools/registry.py` to import from that module. This removes
the Telegram-specific dependency from `shared.py`.

## Compatibility Rule

During the migration, `shared.py` should become a compatibility facade:

- It may re-export old symbols temporarily.
- It should not contain new implementation code.
- Each moved section should leave a short comment:

```python
# Compatibility re-export. New code should import from kg_runtime.search.
from kg_runtime.search import search_knowledge_graph
```

Only delete re-exports after all call sites are migrated and smoke tests pass.

## Validation Checklist Per Phase

Run after each phase:

```bash
venv/bin/python -m py_compile shared.py runtime_tools/*.py self_runtime/*.py telegram/*.py web_chat.py a2a_handler.py browser/worker.py
venv/bin/python scripts/smoke_tool_allowlists.py
venv/bin/python scripts/smoke_filesystem_tools.py
```

Run targeted import checks:

```bash
venv/bin/python -c "import shared; import runtime_tools.registry; print('imports ok')"
venv/bin/python -c "from kg_runtime.service_runtime import get_kg_service; print('kg runtime import ok')"
```

For path-sensitive moves, verify project-root-derived paths explicitly:

```bash
venv/bin/python -c "from content_fetch.browser_pool import PW_COOKIE_PATH; print(PW_COOKIE_PATH)"
```

If a phase changes URL fetching:

```bash
venv/bin/python -c "import asyncio; from content_fetch.urls import fetch_url_content_async; print(asyncio.run(fetch_url_content_async('https://example.com', 500))[:80])"
```

If a phase changes KG runtime:

```bash
venv/bin/python -c "from kg_runtime.search import fetch_kg_stats; print(fetch_kg_stats().keys())"
```

Do not restart services until compile/import checks pass. After runtime import changes, compile/import checks were run before service restart. Service restarts require explicit operator approval:

```bash
sudo systemctl restart leninbot-api.service leninbot-browser.service leninbot-telegram.service
systemctl is-active leninbot-api.service leninbot-browser.service leninbot-telegram.service
```

## Non-Goals

- Do not redesign Graphiti schema in this refactor.
- Do not change KG trust-tier behavior.
- Do not change tool allow-lists.
- Do not change DB schemas.
- Do not rewrite URL extraction behavior unless a test exposes a regression.
- Do not move `GraphMemoryService` out of `graph_memory/service.py`.

## Success Criteria

The refactor is successful when:

- `shared.py` is a small compatibility facade.
- New code imports domain services from `identity`, `provenance`, `content_fetch`, `kg_runtime`, `corpus`, `memory_store`, `ops`, `task_store`, `runtime_tools`, and `self_runtime`.
- `runtime_tools` imports tool implementations from focused modules, not from root-level tool files.
- `graph_memory/` remains the KG implementation package.
- `kg_runtime/` owns cross-runtime KG lifecycle, sync/async access, and search formatting.
- Web chat, Telegram, browser worker, and A2A API import cleanly after service restart.
