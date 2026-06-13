# Autonomous Project Loop

최종 확인 기준: 2026-06-11 코드 트리.

The autonomous project loop advances long-running projects without a live user turn. It is a bounded scheduled worker, not a conversational surface.

## Entry Points

| File | Role |
|---|---|
| `autonomous_project.py` | project selection, tick prompt assembly, event logging |
| `scripts/autonomous_work.py` | compatibility wrapper for older manual invocations |
| `agents/autonomous.py` | `autonomous_project` AgentSpec and capability boundary |
| `bot_config.py` | `autonomous_active`, `autonomous_provider`, `autonomous_model` |
| `systemd/leninbot-autonomous.service` / `.timer` | scheduled execution |

`is_autonomous_active()` reloads config from disk each call, so timer-spawned processes see Telegram `/config` toggles without a long-running process reload.
The production systemd service runs `venv/bin/python -m autonomous_project`; `scripts/autonomous_work.py` delegates to the same entrypoint for compatibility and must not grow a separate workflow.

## Project State

`scripts/schema_migrations.py --only autonomous-projects` ensures these tables before runtime:

| Table | Purpose |
|---|---|
| `autonomous_projects` | project title, topic, goal, state, plan, counters, optional publication pacing knobs |
| `autonomous_project_events` | durable tick/event log, including bounded tool traces |
| `autonomous_project_advisories` | operator advice kept pending until a tick saves durable project work |
| `autonomous_project_notes` | durable research notes with sources; `kind` distinguishes `finding` (single result, 6k chars) from `synthesis` (periodic consolidation, 12k chars) |

Active states are:

- `researching`
- `planning`

Inactive states are:

- `paused`
- `archived`

## Tick Lifecycle

1. Check `autonomous_active`.
2. Select one due active project. Projects with pending operator advisories are selected before ordinary oldest-`last_run_at` round-robin order.
3. Load project, plan, recent notes, recent tick warnings, currently staged research drafts, last tick tool log, and pending operator advisories.
4. Run the `autonomous_project` agent with a small bounded round/budget.
5. Persist notes, staged research drafts, plan changes, state changes, publications, and a clipped tool log.
6. Increment turn metadata and emit project events.
7. Mark pending advisories consumed only if the tick produced durable project work, including a staged research draft; otherwise retain them for the next tick.

Each tick should make one concrete advance. The agent prompt explicitly prioritizes saving durable notes before final prose because chat text does not persist across ticks. The base autonomous agent spec keeps project-state tools (`add_research_note`, `revise_plan`, `set_project_state`) and owned publishing tools (`research_document`, `publish_hub_curation`, `edit_content`, `publish_static_page`) in finalization tools so durable persistence still has pressure even when no runtime override file is present.

`scripts/autonomous_cli.py status` reads config without DB access, so an operator can see `autonomous_active=false`, the autonomous provider/model, and optional systemd timer/service state including next and last timer fire times plus the previous one-shot service result even from a shell that lacks production DB credentials; `--json` emits the same status for scripts/monitors. `read_self(content_type="autonomous_project", id=<id>)` and `scripts/autonomous_cli.py show <id>` read recent notes from `autonomous_project_notes` first and include pending/recent operator advisories, the latest currently staged research draft, tick error, no-durable-action warning, and tick tool log; both fall back to legacy `autonomous_projects.research_notes` only if the note table is unavailable. `scripts/autonomous_cli.py list` also shows pending advisory counts and the most recent project event for quick triage.
The Telegram orchestrator autonomous status block and `/projects` list include pending advisory counts, the most recent project event, and whether the loop is paused by `autonomous_active=false`, so normal chat context can see stalled/error/config-paused states without an extra project-detail call. `/project <id> show` lists pending operator advisories and includes the latest currently staged research draft, tick error, no-durable-action warning, and tick tool log for direct operator triage. Public web chat autonomous summaries remain public-safe but include the loop enabled/paused state and public `publication_created` events in recent work.
If a tick raises before completion, the runtime logs `tick_error` and updates `last_run_at` without incrementing `turn_count`; this gives the failed project a scheduling cooldown so another due project can run on the next timer tick. If a tick succeeds without any durable note, staged draft, publication, plan revision, or state transition, the runtime logs `tick_no_durable_action` so no-op wakes are visible in normal project status views. Pending operator advisories are consumed only when that successful tick produced a durable project action, including a staged draft; no-op ticks retain them and log `advisories_retained_no_durable_action`. Recent `tick_error`, `tick_no_durable_action`, and `advisories_retained_no_durable_action` events are also surfaced in the next tick prompt to discourage repeated failure/no-save loops.

## Capability Boundary

Autonomous publishing is allowed only on owned Cyber-Lenin surfaces:

- public/private research documents
- hub curation entries
- static pages
- Telegram channel announcements that are part of the owned publishing pipeline

The agent does not have broad external outreach tools. It cannot send email, use A2A, browse with browser automation, create images, or modify code/config/systemd files. Its only file writes are research-source downloads/conversions confined to `data/downloads/` and `data/converted/`.

Current autonomous tools include:

- research: `web_search`, `fetch_url`, `fetch_x_post`, `vector_search`, `knowledge_graph_search`, `read_self`, `recall_experience`, `get_finance_data`
- primary-source documents: `download_file`, `convert_document`, `read_document`
- KG: `write_kg_structured`
- owned publishing: `research_document`, `publish_hub_curation`, `edit_content`, `publish_static_page`
- project state: `add_research_note`, `read_research_notes`, `revise_plan`, `set_project_state`

`web_search` accepts optional `search_depth` (`basic`/`advanced`), `topic` (`general`/`news`/`finance`), and `time_range` (`day`/`week`/`month`/`year`) so research ticks can rank recent coverage and pull longer snippets for one focused question; `news`/`finance` results include publish dates. Snippets remain leads, not citable sources — the agent prompt requires fetching the underlying page or a second independent source before a specific figure/quote enters a note or public artifact.

`read_research_notes` is a project-scoped read tool (registered per tick alongside the other project-state tools) that returns FULL note text with optional `keyword`/`note_ids`/`limit` filters. The tick prompt's recent-notes section shows 500-char snippets only and now labels each note with its `#id`; the agent is instructed to load full notes through this tool before drafting long-form artifacts so reports are written from saved research rather than snippets or memory.

`read_self` supports bounded detail pagination with `max_chars` and `offset` for long owned bodies such as diary entries, blog posts, task reports, research documents, private research documents, and static pages. List views still return short previews by design.

Synthesis notes keep old research reachable: `add_research_note(note_type="synthesis")` writes a consolidation note (12k char cap vs 6k for findings). The tick prompt always surfaces the latest synthesis note (clipped at 3,000 chars with a `read_research_notes(note_ids=[...])` pointer) above recent notes, and when 12+ finding-notes have accumulated since the last synthesis (`SYNTHESIS_DUE_AFTER_FINDINGS`), injects a synthesis-due directive telling the tick to consolidate instead of adding more findings — unless operator advice or a staged draft awaiting verification takes priority. `read_research_notes` accepts a `note_type` filter.

`fetch_url` accepts optional `max_chars` (1,000–50,000, default 10,000) and `offset` so long primary sources can be read with bounded character pagination instead of repeatedly returning only the head of the page. Returned headers include `chars start:end`, `truncated`, and a next-call hint when more extracted text is available. For PDFs and other binary documents the agent uses `download_file` → `convert_document` → `read_document`; `read_document` is a per-tick tool sandboxed to `data/downloads/` and `data/converted/` with character pagination — the autonomous agent deliberately does not get the general `read_file` tool because it publishes publicly without a human in the loop.

The agent prompt also carries a `report-quality` section (lead with the finding, absolute dates, comparators on every figure, fact/interpretation/forecast register separation with falsifiable indicators, engage counter-evidence, draft from full notes).

## Publishing Gates

`research_document` is the long-form markdown path. Public publication is gated through staged drafts and fact-check notes. `stage_public` writes an exact draft backup under `data/publication_drafts/research/` and stores the same document in `research_documents` with `status='staged'`; internal agents can retrieve it with `read_self(content_type="research_document", slug="<slug>")`. In autonomous context, `stage_public` also records a `research_draft_staged` project event, and autonomous ticks surface that project's staged drafts before other recent staged drafts so later wakes can resume fact-checking or publication without relying only on the previous raw tool log.

The stage→publish gate is cross-tick for autonomous runs: a draft staged during the current tick cannot be published by that same tick. `autonomous_project._run_one_tick` initializes the `current_tick_staged_slugs` contextvar per tick; `record_autonomous_staged_draft` records each staged filename into it, and `publish_public` refuses any filename found there with guidance to verify and publish on the next wake. The contextvar is `None` outside the tick runtime, so operator/task publication paths are unaffected. Rationale: the context that wrote a draft should not be the context that fact-checks and publishes it — fresh-context verification on the next wake catches errors that same-context self-review rationalizes away. Cost: public publication trails staging by at least one timer interval. `stage_public` refuses to overwrite an already-public row, so revisions to public documents must use `edit_public` instead of staging over the live slug. Public web chat remains restricted to `status='public'` research documents. The autonomous prompt requires independent verification of proper nouns, dates, figures, offices, source attributions, and other factual claims before public publishing. Autonomous public-bound research calls require an explicit stable slug. `publish_public`, `republish_public`, `publish_private`, and `edit_public` also require fact-check notes when they affect a public research document; private-to-public and republish calls are routed through the same public publication path as new reports.

For autonomous projects, `autonomous_publication_controls.py` enforces a narrow structural gate before public publication:

- autonomous public-bound research calls must include a stable slug; research publication and autonomous public research edits must include fact-check notes with at least two source markers/URLs.
- hub curations must include source title, source publication, a valid source URL, rationale/context fields, and a stable slug.
- static pages must include a stable slug, title, HTML body with reader-visible text, and semantic structure. Summary is optional metadata, not a publication blocker.
- The hard gate must not decide semantic quality, length sufficiency, current usefulness, political/reputational risk, or placeholder status by keyword or substring matching. Those judgments belong to the LLM/Stasova review path and must return concrete review reasons when they block or warn.

Publication pacing is no longer a default hard gate. `max_publications_per_day`
and `cooldown_after_publish_minutes` remain in `autonomous_projects` as dormant
knobs for emergency throttling, but scheduled autonomous publication ignores
them unless `AUTONOMOUS_PUBLICATION_PACING_ENABLED=true` is set for the process.
This leaves operator control to `autonomous_active`/manual tick timing without
blocking a queued publication batch.

`publish_hub_curation` creates structured curation entries for Korean-language sources.

`edit_content` edits existing hub curation rows, static page rows, or other public content instead of creating duplicates.

`publish_static_page` creates custom inner HTML content for pages where markdown is insufficient. The site supplies page shell, navigation, and sanitization.

Rough drafts should stay in project notes. Publishing is for artifacts that meet the project goal's quality bar.

## Provider and Budget

Defaults come from `bot_config.py` and can be overlaid in `config/agent_runtime.json`. The example pins autonomous work to DeepSeek Pro with a smaller budget and finalization tools for state/publishing persistence.

Do not assume the autonomous loop uses the same provider as Telegram chat. Use `get_current_model_selection(kind="autonomous")` for display/runtime metadata.

## Operational Notes

- Keep ticks idempotent. Recent notes and last tool logs exist to prevent repeated research.
- Operator advisories are authoritative over the agent's prior plan for the next tick.
- Use shared KG group_ids such as `economy`, `korea_domestic`, `geopolitics_conflict`, `diplomacy`, or `agent_knowledge`; keep project-only working notes in autonomous project notes instead of creating project-specific KG groups.
- Use `paused` rather than deleting projects when a project should stop temporarily.
- Use stable slugs for public artifacts so later ticks edit/republish instead of duplicating.
