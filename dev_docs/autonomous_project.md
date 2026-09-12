# Autonomous Project Loop

최종 확인 기준: 2026-07-09 코드 트리.

The autonomous project loop advances long-running projects without a live user turn. It is a bounded scheduled worker, not a conversational surface.

## Entry Points

| File | Role |
|---|---|
| `jobs/autonomous_project.py` | project selection, tick prompt assembly, event logging |
| `scripts/autonomous_work.py` | compatibility wrapper for older manual invocations |
| `agents/autonomous.py` | `autonomous_project` AgentSpec and capability boundary |
| `bot_config.py` | `autonomous_active`, `autonomous_provider`, `autonomous_model` |
| `systemd/leninbot-autonomous.service` / `.timer` | scheduled execution |

`is_autonomous_active()` reloads config from disk each call, so timer-spawned processes see Telegram `/config` toggles without a long-running process reload.
The production systemd service runs `venv/bin/python -m jobs.autonomous_project`; `scripts/autonomous_work.py` delegates to the same entrypoint for compatibility and must not grow a separate workflow.

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
9. Mark pending advisories consumed only if the tick produced durable project work, including a staged research draft; otherwise retain them for the next tick.

One tick allocates a gateway `request_id` for the main autonomous run and uses
`scope_type=autonomous_project`, `scope_id=<project id>`. Planner, critic, and
`research_deep_dive` sub-runs receive the main run ID as
`parent_request_id`; all tool audits therefore remain queryable by project even
though the scheduled worker has no chat session. Scheduled CommuLingo
maintainers use the same `interface=autonomous` boundary with
`scope_type=maintenance_job` and a lane/job identifier.

`tick_error` and `tick_review` verdicts remain diagnostic signals only. Repeated
`no-op`/error ticks do not automatically pause a project; project state changes
to `paused` only through an explicit operator or agent state-change action.

Each tick should make one concrete advance. The agent prompt explicitly prioritizes saving durable notes before final prose because chat text does not persist across ticks. The base autonomous agent spec keeps project-state tools (`add_research_note`, `revise_plan`, `set_project_state`) and owned publishing tools (`research_document`, `publish_hub_curation`, `edit_content`, `publish_static_page`) in finalization tools so durable persistence still has pressure even when no runtime override file is present.

The system prompt and both provider context formats treat the latest synthesis
as a historical account, not independent or necessarily current evidence.
Research-trail reuse permits re-reading to recover missing evidence, resolve
contradictions, or check plausible changes. Successful writes should not be
repeated merely to verify them. The common AgentSpec source boundary and
execution contract also apply, while public formats, tick persistence and
terminal/finalization tool behavior keep their specialized contracts.

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

The agent prompt also carries a `report-quality` section (lead with the finding, absolute dates, comparators on every figure, fact/interpretation/forecast register separation with falsifiable indicators, engage counter-evidence, draft from full notes) and a `report-format` section fixing a small public-report FRAME while leaving analysis structure free-form per piece (2026-07-11 operator decision — 글의 다양성을 위해 중간 구조는 강제하지 않는다): related/선행-report links at the top (mandatory for series installments, linking earlier installments), `## 요약` as the first section, `## 출처` (footnote definitions) as the last; body headings `##`/`###` only, no multi-level decimal numbering, body target 6,000–15,000 chars. The mechanical parts are enforced by `validate_autonomous_research_publication` at publish time (no body H1; `## 요약` present; ≥2 `[^n]:` footnote definitions), with `is_edit=True` relaxing the frame checks (but not the body-H1 check) so factual corrections to pre-frame legacy documents are not blocked. Legacy body-H1 headings in 11 existing documents were demoted via `scripts/demote_research_body_h1.py` (backup `data/publication_drafts/research_body_h1_backup_20260711.json`).

## Publishing Gates

`research_document` is the long-form markdown path. Public publication is gated through staged drafts and fact-check notes. `stage_public` writes an exact draft backup under `data/publication_drafts/research/` and stores the same document in `research_documents` with `status='staged'`; internal agents can retrieve it with `read_self(content_type="research_document", slug="<slug>")`. `stage_public`/`publish_public` strip an agent-supplied leading H1/author/date scaffold before composing the canonical document (mirroring `edit_public`) and preserve the original `작성일` across re-stagings — before 2026-07-11 the header was duplicated on every staged/published report (28 existing rows were cleaned via `scripts/dedupe_research_headers.py`; backup in `data/publication_drafts/research_header_dedupe_backup_20260711.json`).

Two revision/publication paths avoid re-emitting long drafts (a ~23k-char draft does not fit the tick's 16,384-token completion cap): `edit_staged` applies exact find/replace `edits` to the stored staged body (each `find` must match exactly once; all-or-nothing; re-runs citation validation; records a `research_draft_staged` event and re-arms the cross-tick gate), and **slug-only `publish_public`** (content omitted, `fact_check_notes` still required) publishes the stored staged text as-is. In autonomous context, `stage_public` also records a `research_draft_staged` project event, and autonomous ticks surface that project's staged drafts before other recent staged drafts so later wakes can resume fact-checking or publication without relying only on the previous raw tool log.

The stage→publish gate is cross-tick for autonomous runs: a draft staged during the current tick cannot be published by that same tick. `jobs.autonomous_project._run_one_tick` initializes the `current_tick_staged_slugs` contextvar per tick; `record_autonomous_staged_draft` records each staged filename into it, and `publish_public` refuses any filename found there with guidance to verify and publish on the next wake. The contextvar is `None` outside the tick runtime, so operator/task publication paths are unaffected. Rationale: the context that wrote a draft should not be the context that fact-checks and publishes it — fresh-context verification on the next wake catches errors that same-context self-review rationalizes away. Cost: public publication trails staging by at least one timer interval. `stage_public` refuses to overwrite an already-public row, so revisions to public documents must use `edit_public` instead of staging over the live slug. Public web chat remains restricted to `status='public'` research documents. The autonomous prompt requires independent verification of proper nouns, dates, figures, offices, source attributions, and other factual claims before public publishing. Autonomous public-bound research calls require an explicit stable slug. `publish_public`, `republish_public`, `publish_private`, and `edit_public` also require fact-check notes when they affect a public research document; private-to-public and republish calls are routed through the same public publication path as new reports.

For autonomous projects, `jobs/autonomous_publication_controls.py` enforces a narrow structural gate before public publication:

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

`publish_hub_curation` creates structured curation entries for external progressive writing in any language (Korean-only restriction lifted 2026-07-11). Curation titles are plain headlines — meta prefixes like '왜 이 글이 지금 중요한가:' are banned from titles; that framing belongs in `selection_rationale`.

`edit_content` edits existing hub curation rows, static page rows, or other public content instead of creating duplicates.

`publish_static_page` creates custom inner HTML content for pages where markdown is insufficient. The site supplies page shell, navigation, and sanitization.

Rough drafts should stay in project notes. Publishing is for artifacts that meet the project goal's quality bar.

## Provider and Budget

Defaults come from `bot_config.py` and can be overlaid in `config/agent_runtime.json`. The example pins autonomous work to DeepSeek V4.1 Flash (`deepseek-flash`; legacy `deepseek_pro` selections resolve to Flash) with a smaller budget and finalization tools for state/publishing persistence.

Do not assume the autonomous loop uses the same provider as Telegram chat. Use `get_current_model_selection(kind="autonomous")` for display/runtime metadata.

## Operational Notes

- Keep ticks idempotent. Recent notes and last tool logs exist to prevent repeated research.
- Operator advisories are authoritative over the agent's prior plan for the next tick.
- Use shared KG group_ids such as `economy`, `korea_domestic`, `geopolitics_conflict`, `diplomacy`, or `agent_knowledge`; keep project-only working notes in autonomous project notes instead of creating project-specific KG groups.
- Use `paused` rather than deleting projects when a project should stop temporarily.
- Use stable slugs for public artifacts so later ticks edit/republish instead of duplicating.

## Runtime evidence and operator authority

The main tick sends the existing bounded project snapshot as reference context,
the project goal as an assignment, and pending advisories as separately identified
operator instructions with IDs/dates. Prior plans, notes, tool traces, staged
artifacts and critic judgments remain historical data rather than fresh actions.
Shared reality guidance distinguishes stored work, publication and goal progress.
Existing planner/critic calls, durable-action checks, advisory consumption, budgets
and publication gates are unchanged; paused projects are never enabled by this
context change. Experience recall now includes source and period metadata.

## Project #4: practice output policy

`data/project_designs/practice_output_loop.md`가 #4의 발주 명세다. 다른 프로젝트는
기존 경로를 유지한다. `jobs/practice_output.py:applies`가 적용 대상을 ID 4로 제한한다.
코드 설치는 프로젝트·설정 활성화나 advisory 소비를 수행하지 않는다.

- `objective` / `build_tools`: 별도 planner 대신 단계별 산출물 objective를 생성한다.
  tick 전용 `define_practice_output`으로 주제, 독자, 사용 목적, 전달 형태를 저장한다.
  런타임이 만든 `p4-<uuid>`가 산출물 ID이자 유일한 쓰기 허용 slug다. 주제 문자열의
  정확 일치 중복과 진행 중 descriptor 변경을 거부한다. 의미가 같은 주제의 다른
  표현을 판정하는 LLM 관문은 두지 않는다.
- `begin` / `reserve` / `complete` / `finish`: 기존 `autonomous_project_events`에
  `practice_output_state` JSON snapshot을 영속화한다. 한 산출물에 a 1회, b 2회,
  c 1회를 배정한다. b 2회는 상한이며 오류도 시도로 소비한다. 예약을 실행 전에
  커밋하므로 `turn_count`가 증가하지 않은 실패도 무료 재시도가 되지 않는다.
  프로세스 중단 시 다음 실행이 기존 예약의 실제 event ID를 먼저 회수하고 실패로
  기록한다. c 시도 후 주제를 닫고 다음 tick은 새 산출물로 이동한다. 반복 실패에
  따른 프로젝트 자동 종료/일시정지 정책은 추가하지 않았다.
- `_run_one_tick`: config와 DB 상태를 다시 확인하고 PostgreSQL advisory session
  lock으로 #4의 동시 tick을 차단한다. 상태 저장 실패는 삼키지 않는다. `paused`나
  `archived`에서는 상태 예약도 할 수 없다. 운영자 상태 변경은 이 루프에 맡기지 않는다.
  실행 도중 운영자가 중단해도 이미 예약한 시도의 종료·발행 증거는 저장한다.
  이 저장은 프로젝트를 재개하거나 새 시도를 예약하지 않는다. session lock 획득 후
  트랜잭션은 커밋하여 모델 실행 동안 idle transaction을 유지하지 않는다.
- `validate_plan` / `passive_wait`: #4의 `revise_plan` 쓰기 전 발표 대기 목표를 거절하고
  차단 사유 기록 및 수행 가능한 산출물 작업으로 전환하도록 반환한다. 날짜나 이미
  발표된 자료 활용은 허용한다. 한국어·영어의 명시적 대기/발표일 확인 표현에 대한
  보수적 어휘 필터이며 모든 자연어 우회 표현을 의미론적으로 검출하지는 않는다.
- `_execute_one_tick`: #4에서 기존 편집진단, planner, critic, `research_deep_dive`를
  실행하지 않는다. 기본 모델·권한 체계는 유지한다. 본 실행은 기존 엔진 예산 제한에
  `min(config, $0.30)`와 일반 라운드 `min(config, 12)`를 전달한다. 발행 보안검수
  `run_stasova_publication_review`는 #4에서 `min(config, $0.05)`/2라운드로 제한한다.
  본 실행에 더해 산출물별 발행 시도는 1회다. 기존 엔진은 응답 뒤 비용을 확인하고
  최종 저장 호출을 허용하므로 이 수치는 선불 결제 한도나 최종 청구액 상한이 아니다.
- `guard_handlers`: a/b에서는 현재 산출물의 초안 저장/수정만, c에서 발행 시도 1회를
  허용한다. 실제 발행 handler와 cross-tick staging·출처·권한·보안 검사는 그대로다.
  상한에 도달했다고 사실 검증을 통과한 것으로 취급하지 않는다. 기존 공개 콘텐츠
  수정 및 다른 slug 쓰기는 거절한다. c에서 새로 staging해 기존 gate에 막히면 그
  산출물을 재시도하지 않고 차단 사유를 남긴다.
- `complete` / `matches_output`: 초안(`research_draft_staged`), 검수
  (`publication_reviewed`/`publication_review_error`), 실제 발행(`publication_created`)을
  서로 다른 event ID로 저장하고 발행 관련 기록은 현재 산출물 ID로 대조한다.
  c 종료는 루프 종료이지 발행 성공이 아니다. 발행 receipt가 없으면
  `publication_unconfirmed`; receipt가 있으면 `published`다. 오류와 발행 상태는
  독립적이므로 발행 직후 오류가 나도 저장된 발행 증거를 버리지 않는다.

`value_metrics`는 웹 재사용/후속 질문, 재료 소비, 실무자 응답을 모두
`value=null, status=unknown`으로 기록한다. 현재 `chat_logs`는 대화/도구 추적 텍스트,
`web_chat_feedback`은 chat ID에 대한 반응이며, `research_documents`에는 산출물 소비
관계나 실무자 검증 요청–응답의 구조화된 식별자가 없다. 임의 문자열 매칭이나 모델
자가평가로 이 세 지표를 0 또는 달성으로 바꾸지 않는다. 실제 계측 추가는 후속 작업이다.

운영 시작 전 결정할 제안(아직 설정·자동 정책 아님): 첫 실행부터 예약된 8 tick,
즉 최대 두 산출물 순환 뒤 운영자가 방향을 검토한다. 분모는 오류/중단을 포함한
고유 `request_id` 예약 수, 초안·발행 건수는 산출물 ID로 중복 제거한다. 무진전은
이 구간의 초안/실제 발행 receipt 부재로 점검하되 가치 지표 unknown을 0으로
취급하지 않는다. 별도 LLM 비용 없이 이벤트를 읽는다. 지표 관측 기간·대상 집단과
추가 계측은 운영자가 결정해야 하며, 이 제안으로 자동 종료시키지 않는다.

검증: `venv/bin/python -m pytest -q tests/test_practice_output.py`는 DB/LLM/발행을
모킹하여 필터, 단계 상한, 재시작·오류, 발행 receipt 구분, descriptor 불변성,
비활성·동시 실행 보호 경로와 실제 tick의 예산/검토 호출 구성을 검사한다.
기존 smoke는 `scripts/smoke_autonomous_research.py`,
`scripts/smoke_autonomous_publication_gates.py`다. 운영 DB에 tick을 실행하는 검증은 아니다.
다음 승인된 timer 실행이 새 프로세스로 코드를 읽으므로 이 변경을 위해 Telegram
서비스를 재시작할 필요는 없다.
