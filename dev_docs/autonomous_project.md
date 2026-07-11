# Autonomous Project Loop

최종 확인 기준: 2026-07-09 코드 트리.

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
3. Load project, plan, recent notes, recent tick warnings, currently staged research drafts, last tick tool log, and pending operator advisories. Staged drafts additionally get the editorial diagnosis pass (see Publishing Gates). A `<past-experiences>` block (experiential-memory vector recall keyed on title+topic+goal, k=3, local embeddings) is injected when relevant lessons exist — including lessons the failure hooks below wrote on earlier ticks. A `<research-trail>` block (`_research_trail`) mines the last `RESEARCH_TRAIL_LOGS` persisted tick tool logs **before** the most recent one (that one is shown in full as `<last-tick-execution>`) and lists deduped research-tool invocations (`web_search`/`fetch_url`/`vector_search`/KG/finance/document calls, args capped at 180 chars, max 40 lines) so ticks outside the visible window don't get re-run.
4. **Pre-tick Planner** (`_plan_tick_objective`, flag `autonomous_tick_planner`, default on): one bounded call (DeepSeek **high tier** / provider high tier, ≤$0.05, max_tokens 1200, **thinking disabled** — DeepSeek reasoning mode exhausted the entire token cap before any visible reply on the first live ticks — and **toolless single round**: `extra_tools=[]` matters, because `extra_tools=None` makes `_chat_with_tools` grant the full orchestrator toolset, and a planner must choose, not act) over the assembled context picks this tick's single objective — `OBJECTIVE`/`ARTIFACT`/`WHY` — logged as a `tick_objective` event and injected into the tick prompt as `<tick-objective>`. The agent may deviate but must justify it in its self-critique. Malformed replies or failures degrade to no injection. (2026-07-09: planner/critic moved from low to high tier — objective selection and progress judgment are judgment calls, and measured tick cost is ~$0.04 so the stronger model adds ~$0.01–0.02; editorial diagnosis stays low tier per the writer's validated cheap-diagnoser pattern. Tier is resolved by `_reviewer_profile(provider, tier)`.)
5. Run the `autonomous_project` agent with a small bounded round/budget.
6. Persist notes, staged research drafts, plan changes, state changes, publications, and a clipped tool log.
7. Increment turn metadata and emit project events.
8. **Post-tick Critic** (`_review_tick_outcome`, flag `autonomous_tick_critic`, default on): one bounded call (high tier, ≤$0.03, max_tokens 800, thinking disabled, toolless single round — same rationale as the planner) judges the tick's durable actions against the objective (or the one-concrete-step standard when no objective was set) — `VERDICT: advanced|partial|no-op` + one-line reason, logged as a `tick_review` event (meta: verdict, model, objective). This makes the tick's self-critique durable: partial/no-op verdicts surface in the next tick's warnings, unlike the closing self-critique paragraph that dies with the chat text. The critic does not run on `tick_error` ticks. A no-op verdict (and any `tick_error`) additionally writes a deduped `mistake` lesson to `experiential_memory` (`source_type=autonomous_tick`) so later ticks recall it via the past-experiences block. The verdict + REASON line also rides on the owner Telegram tick notification (`[크리틱]`), next to the agent's own self-critique.
9. **Stall auto-pause** (`_maybe_auto_pause_stalled_project`, Phase-4 enforcement): when the last `STALL_AUTO_PAUSE_STREAK` (3) completed ticks were all stalls (`tick_error`, or `tick_review` verdict `no-op`), the project is set to `paused` (a `state_transition` event with `meta.auto=true`), and the owner gets a Telegram alert with resume instructions. Only a no-op tick can complete the streak — a healthy tick never triggers it — and the check is implicitly gated on the tick critic flag (no verdicts → no streak). Resume by `set_project_state`/CLI back to `researching`, optionally with an advisory.
10. Mark pending advisories consumed only if the tick produced durable project work, including a staged research draft; otherwise retain them for the next tick.

Each tick should make one concrete advance. The agent prompt explicitly prioritizes saving durable notes before final prose because chat text does not persist across ticks. The base autonomous agent spec keeps project-state tools (`add_research_note`, `revise_plan`, `set_project_state`) and owned publishing tools (`research_document`, `publish_hub_curation`, `edit_content`, `publish_static_page`) in finalization tools so durable persistence still has pressure even when no runtime override file is present.

`scripts/autonomous_cli.py status` reads config without DB access, so an operator can see `autonomous_active=false`, the autonomous provider/model, and optional systemd timer/service state including next and last timer fire times plus the previous one-shot service result even from a shell that lacks production DB credentials; `--json` emits the same status for scripts/monitors. `read_self(content_type="autonomous_project", id=<id>)` and `scripts/autonomous_cli.py show <id>` read recent notes from `autonomous_project_notes` first and include pending/recent operator advisories, the latest currently staged research draft, tick error, no-durable-action warning, and tick tool log; both fall back to legacy `autonomous_projects.research_notes` only if the note table is unavailable. `scripts/autonomous_cli.py list` also shows pending advisory counts and the most recent project event for quick triage.
The Telegram orchestrator autonomous status block and `/projects` list include pending advisory counts, the most recent project event, and whether the loop is paused by `autonomous_active=false`, so normal chat context can see stalled/error/config-paused states without an extra project-detail call. `/project <id> show` lists pending operator advisories and includes the latest currently staged research draft, tick error, no-durable-action warning, and tick tool log for direct operator triage. Public web chat autonomous summaries remain public-safe but include the loop enabled/paused state and public `publication_created` events in recent work.
If a tick raises before completion, the runtime logs `tick_error` and updates `last_run_at` without incrementing `turn_count`; this gives the failed project a scheduling cooldown so another due project can run on the next timer tick. If a tick succeeds without any durable note, staged draft, publication, plan revision, or state transition, the runtime logs `tick_no_durable_action` so no-op wakes are visible in normal project status views. Pending operator advisories are consumed only when that successful tick produced a durable project action, including a staged draft; no-op ticks retain them and log `advisories_retained_no_durable_action`. Recent `tick_error`, `tick_no_durable_action`, and `advisories_retained_no_durable_action` events are also surfaced in the next tick prompt to discourage repeated failure/no-save loops, along with `tick_review` events whose verdict is `partial` or `no-op` (advanced/inconclusive reviews stay in the event log without polluting the warnings).

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
- deep-dive delegation: `research_deep_dive` (registered per tick by `_build_deep_dive_tool`)

`research_deep_dive(question, context?, budget_usd?)` commissions the **analyst** spec on a hard-filtered READ-ONLY tool subset (`web_search`, `fetch_url`, `fetch_x_post`, `vector_search`, `knowledge_graph_search`, `get_finance_data`, `read_self` — analyst's own `read_file`/`research_document`/`edit_content`/`write_kg_structured` are stripped, so the autonomous capability boundary is unchanged) for one focused research question that needs more rounds than the tick has left. Bounds: max 2 calls per tick (closure counter), ≤10 sub-rounds, budget clamped to $0.05–0.50 per call **on top of** the tick budget (separate tracker; cost reported in the tool result and a `deep_dive` project event). Failures degrade to an error string, never break the tick. Gateway risk class: `delegate`. The tick prompt instructs the agent to save deep-dive findings via `add_research_note` citing the mini-report's SOURCES and to verify independently before anything public.

`web_search` accepts optional `search_depth` (`basic`/`advanced`), `topic` (`general`/`news`/`finance`), and `time_range` (`day`/`week`/`month`/`year`) so research ticks can rank recent coverage and pull longer snippets for one focused question; `news`/`finance` results include publish dates. Snippets remain leads, not citable sources — the agent prompt requires fetching the underlying page or a second independent source before a specific figure/quote enters a note or public artifact.

`read_research_notes` is a project-scoped read tool (registered per tick alongside the other project-state tools) that returns FULL note text with optional `keyword`/`note_ids`/`limit` filters. The tick prompt's recent-notes section shows 500-char snippets only and now labels each note with its `#id`; the agent is instructed to load full notes through this tool before drafting long-form artifacts so reports are written from saved research rather than snippets or memory.

`read_self` supports bounded detail pagination with `max_chars` and `offset` for long owned bodies such as diary entries, blog posts, task reports, research documents, private research documents, and static pages. List views still return short previews by design.

Synthesis notes keep old research reachable: `add_research_note(note_type="synthesis")` writes a consolidation note (12k char cap vs 6k for findings). The tick prompt always surfaces the latest synthesis note (clipped at 3,000 chars with a `read_research_notes(note_ids=[...])` pointer) above recent notes, and when 12+ finding-notes have accumulated since the last synthesis (`SYNTHESIS_DUE_AFTER_FINDINGS`), injects a synthesis-due directive telling the tick to consolidate instead of adding more findings — unless operator advice or a staged draft awaiting verification takes priority. `read_research_notes` accepts a `note_type` filter.

`fetch_url` accepts optional `max_chars` (1,000–50,000, default 10,000) and `offset` so long primary sources can be read with bounded character pagination instead of repeatedly returning only the head of the page. Returned headers include `chars start:end`, `truncated`, and a next-call hint when more extracted text is available. For PDFs and other binary documents the agent uses `download_file` → `convert_document` → `read_document`; `read_document` is a per-tick tool sandboxed to `data/downloads/` and `data/converted/` with character pagination — the autonomous agent deliberately does not get the general `read_file` tool because it publishes publicly without a human in the loop.

The agent prompt also carries a `report-quality` section (lead with the finding, absolute dates, comparators on every figure, fact/interpretation/forecast register separation with falsifiable indicators, engage counter-evidence, draft from full notes).

## Publishing Gates

`research_document` is the long-form markdown path. Public publication is gated through staged drafts and fact-check notes. `stage_public` writes an exact draft backup under `data/publication_drafts/research/` and stores the same document in `research_documents` with `status='staged'`; internal agents can retrieve it with `read_self(content_type="research_document", slug="<slug>")`. `stage_public`/`publish_public` strip an agent-supplied leading H1/author/date scaffold before composing the canonical document (mirroring `edit_public`) and preserve the original `작성일` across re-stagings — before 2026-07-11 the header was duplicated on every staged/published report (28 existing rows were cleaned via `scripts/dedupe_research_headers.py`; backup in `data/publication_drafts/research_header_dedupe_backup_20260711.json`).

Two revision/publication paths avoid re-emitting long drafts (a ~23k-char draft does not fit the tick's 16,384-token completion cap, which is what stalled project 3 into auto-pause on 2026-07-11): `edit_staged` applies exact find/replace `edits` to the stored staged body (each `find` must match exactly once; all-or-nothing; re-runs citation validation; records a `research_draft_staged` event and re-arms the cross-tick gate), and **slug-only `publish_public`** (content omitted, `fact_check_notes` still required) publishes the stored staged text as-is. In autonomous context, `stage_public` also records a `research_draft_staged` project event, and autonomous ticks surface that project's staged drafts before other recent staged drafts so later wakes can resume fact-checking or publication without relying only on the previous raw tool log.

The stage→publish gate is cross-tick for autonomous runs: a draft staged during the current tick cannot be published by that same tick. `autonomous_project._run_one_tick` initializes the `current_tick_staged_slugs` contextvar per tick; `record_autonomous_staged_draft` records each staged filename into it, and `publish_public` refuses any filename found there with guidance to verify and publish on the next wake. The contextvar is `None` outside the tick runtime, so operator/task publication paths are unaffected. Rationale: the context that wrote a draft should not be the context that fact-checks and publishes it — fresh-context verification on the next wake catches errors that same-context self-review rationalizes away. Cost: public publication trails staging by at least one timer interval. `stage_public` refuses to overwrite an already-public row, so revisions to public documents must use `edit_public` instead of staging over the live slug. Public web chat remains restricted to `status='public'` research documents. The autonomous prompt requires independent verification of proper nouns, dates, figures, offices, source attributions, and other factual claims before public publishing. Autonomous public-bound research calls require an explicit stable slug. `publish_public`, `republish_public`, `publish_private`, and `edit_public` also require fact-check notes when they affect a public research document; private-to-public and republish calls are routed through the same public publication path as new reports.

For autonomous projects, `autonomous_publication_controls.py` enforces a narrow structural gate before public publication:

- autonomous public-bound research calls must include a stable slug; research publication and autonomous public research edits must include fact-check notes with at least two source markers/URLs.
- hub curations must include source title, source publication, a valid source URL, rationale/context fields, and a stable slug.
- static pages must include a stable slug, title, HTML body with reader-visible text, and semantic structure. Summary is optional metadata, not a publication blocker.
- The hard gate must not decide semantic quality, length sufficiency, current usefulness, political/reputational risk, or placeholder status by keyword or substring matching. Those judgments belong to the LLM/Stasova review path and must return concrete review reasons when they block or warn.

**Editorial diagnosis (Reflexion pre-publish pass).** On each tick, this project's staged drafts get an independent editorial diagnosis (`_diagnose_staged_drafts_for_tick`, using `llm/reflexion.py`): a cheap model (DeepSeek flash when available, else the tick provider's low tier) reviews the staged markdown against 사실성/논리/완결성/명료성/공개적합성 and returns numbered, quote-anchored notes or `PASS`. Non-PASS notes are injected into the tick prompt as `<editorial-diagnosis>` right after the staged-drafts listing; the tick agent, as the author, either revises via `stage_public` (same slug) or consciously rejects the notes before `publish_public`. Each diagnosis is logged as an `editorial_diagnosis` project event (meta: slug, verdict, model) and **cached per draft version** — an unchanged draft re-injects the stored notes on later ticks with no LLM call. Toggle: `reflexion_autonomous_publish` config key (default on). Any failure in the pass degrades to no injection; it never blocks a tick. This complements — does not replace — the structural gate and cross-tick stage→publish rule above.

**Revision budget (`_EDITORIAL_DIAGNOSIS_MAX_ROUNDS = 2`).** An open-ended diagnoser never says PASS — on project 3 it issued 18 diagnosis rounds and drove 11 re-stagings of one draft. After 2 non-PASS diagnosis rounds per slug (counted from `editorial_diagnosis` events, any draft version), the pass stops diagnosing that draft and instead injects a directive: publish now via slug-only `publish_public`, or fix a concrete factual error via `edit_staged` and publish on the next wake. Style/clarity/structure notes can no longer block past the cap.

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
