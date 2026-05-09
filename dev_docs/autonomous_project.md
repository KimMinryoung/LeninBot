# Autonomous Project Loop

최종 확인 기준: 2026-05-09 코드 트리.

The autonomous project loop advances long-running projects without a live user turn. It is a bounded scheduled worker, not a conversational surface.

## Entry Points

| File | Role |
|---|---|
| `autonomous_project.py` | project selection, tick prompt assembly, event logging |
| `scripts/autonomous_work.py` | systemd-friendly runner |
| `agents/autonomous.py` | `autonomous_project` AgentSpec and capability boundary |
| `bot_config.py` | `autonomous_active`, `autonomous_provider`, `autonomous_model` |
| `systemd/leninbot-autonomous.service` / `.timer` | scheduled execution |

`is_autonomous_active()` reloads config from disk each call, so timer-spawned processes see Telegram `/config` toggles without a long-running process reload.

## Project State

`scripts/schema_migrations.py --only autonomous-projects` ensures these tables before runtime:

| Table | Purpose |
|---|---|
| `autonomous_projects` | project title, topic, goal, state, plan, counters, publication throttles |
| `autonomous_project_events` | durable tick/event log, including bounded tool traces |
| `autonomous_project_advisories` | operator one-shot advice consumed by the next tick |
| `autonomous_project_notes` | durable research notes with sources |

Active states are:

- `researching`
- `planning`

Inactive states are:

- `paused`
- `archived`

## Tick Lifecycle

1. Check `autonomous_active`.
2. Select one due active project.
3. Load project, plan, recent notes, last tick tool log, and pending operator advisories.
4. Mark advisories consumed for that project.
5. Run the `autonomous_project` agent with a small bounded round/budget.
6. Persist notes, plan changes, state changes, publications, and a clipped tool log.
7. Increment turn metadata and emit project events.

Each tick should make one concrete advance. The agent prompt explicitly prioritizes saving durable notes before final prose because chat text does not persist across ticks.

## Capability Boundary

Autonomous publishing is allowed only on owned Cyber-Lenin surfaces:

- public/private research documents
- hub curation entries
- static pages
- Telegram channel announcements that are part of the owned publishing pipeline

The agent does not have broad external outreach tools. It cannot send email, use A2A, browse with browser automation, create images, write files, or modify code/config/systemd files.

Current autonomous tools include:

- research: `web_search`, `fetch_url`, `fetch_x_post`, `vector_search`, `knowledge_graph_search`, `read_self`, `recall_experience`, `get_finance_data`
- KG: `write_kg_structured`
- owned publishing: `research_document`, `publish_hub_curation`, `edit_content`, `publish_static_page`
- project state: `add_research_note`, `revise_plan`, `set_project_state`

## Publishing Gates

`research_document` is the long-form markdown path. Public publication is gated through staged drafts and fact-check notes. The autonomous prompt requires independent verification of proper nouns, dates, figures, offices, source attributions, and other factual claims before public publishing.

For autonomous projects, `autonomous_publication_controls.py` also enforces a hard quality gate before public publication:

- research reports must be substantial, sectioned markdown; `fact_check_notes` must cite at least two source markers/URLs; the notes or body must explicitly address 2026-current usefulness by updating or removing stale, low-utility framing.
- hub curations must include source title and publication, a valid source URL, substantive rationale/context, and an explanation of why the external piece remains useful now.
- static pages must have stable slugs, substantial semantic inner HTML, a reader-facing summary, and a current-usefulness explanation.
- placeholder/TODO/draft-only text blocks autonomous publication across these surfaces.

`publish_hub_curation` creates structured curation entries for Korean-language sources.

`edit_content` edits existing hub curation rows or public content instead of creating duplicates.

`publish_static_page` creates custom inner HTML content for pages where markdown is insufficient. The site supplies page shell, navigation, and sanitization.

Rough drafts should stay in project notes. Publishing is for artifacts that meet the project goal's quality bar.

## Provider and Budget

Defaults come from `bot_config.py` and can be overlaid in `config/agent_runtime.json`. The example pins autonomous work to DeepSeek Pro with a smaller budget and finalization tools for state/publishing persistence.

Do not assume the autonomous loop uses the same provider as Telegram chat. Use `get_current_model_selection(kind="autonomous")` for display/runtime metadata.

## Operational Notes

- Keep ticks idempotent. Recent notes and last tool logs exist to prevent repeated research.
- Operator advisories are authoritative over the agent's prior plan for the next tick.
- Use `paused` rather than deleting projects when a project should stop temporarily.
- Use stable slugs for public artifacts so later ticks edit/republish instead of duplicating.
