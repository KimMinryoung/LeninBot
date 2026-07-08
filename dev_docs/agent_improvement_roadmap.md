# Agent Improvement Roadmap — Intelligence & Memory

**Date**: 2026-07-08
**Scope**: (1) Agent intelligence — evolve the ReAct-only loops toward the CLAW triad (Planner + ReAct Executor + Critic), generalized Reflexion, and Plan-and-Execute; (2) Memory & knowledge — KG resilience, vector corpus backlog, experiential memory in the working loops.
**Out of scope**: testing infrastructure overhaul, security hardening, code cleanup (small enablers allowed where a phase requires them).

Design principles (per `CLAUDE.md`): Simplicity First; Surgical Changes; each phase independently shippable and verifiable; built on existing seams (`chat_with_tools`, `tool_gateway.dispatcher.execute_tool`, `AgentSpec` registry, `telegram_tasks` queue); rollout via env flags / `config.json` keys / `config/agent_runtime.json` overlays, with shadow→enforce staging wherever output reaches users.

---

## Current-state assessment (verified against code, 2026-07-08)

The CLAUDE.md goal (CLAW / Reflexion / LATS / Plan-and-Execute) is mostly unrealized. Mapping onto existing seeds:

| CLAW component | Existing seed | Roadmap move |
|---|---|---|
| Executor (ReAct) | `claude_loop.py:418` / `openai_tool_loop.py:959` `chat_with_tools` — live everywhere | Keep as-is |
| Critic | **Dead** task verifier (below); writer `run_diagnose_revise_pass` (`writer/stream.py`, blind-eval 12/12); Stasova (advisory, publication-only) | Phase 1 wires the verifier; Phase 2 generalizes diagnose→revise |
| Planner | `multi_delegate` `plan_id`/`plan_role` machinery; `autonomous_projects.plan` + `revise_plan` — static data, no planner component | Phase 3 adds dependency DAGs; Phase 4 adds per-tick objective selection |
| Reflexion | Writer diagnose_revise; chat `_reflect_on_recent`; tick self-critique paragraph (dies with chat text) | Phases 2/4/5 make critique durable and general |
| LATS | none | Phase 8: assessed and **deferred**; bounded best-of-N pilot instead |

Code-verified facts that shape the ordering (recorded here so future sessions don't re-derive them):

1. **The task verifier is dead code.** `_run_verification` (`telegram/tasks.py:510`) and `_maybe_redelegate_after_verification_failure` (`telegram/tasks.py:768`) — independent LLM verification of task reports with tools, PASS/FAIL verdict, restart orchestration, bounded auto-retry with "take a DIFFERENT approach" re-delegation, chain-depth guards — are defined but never called. Nothing writes `metadata.verification`; `_persist_task_success` sets `verification_status='pending'` and it stays pending forever. Wiring it is the highest value/effort item in the repo.
2. **Experiential auto-recall already exists in the Telegram chat loop.** `telegram/commands.py:1495` calls `_fetch_relevant_experiences` (top-3, similarity > 0.5, local BGE-M3) on every message and injects a `<past-experiences>` block; `_reflect_on_recent` writes back every 5 exchanges; `experience_writer.py` runs a daily batch. The gap is the **task worker** and **autonomous tick**, plus structured write-back from failures.
3. **KG embedding 429 resilience is partially done.** `graph_memory/service.py:94` already wraps Gemini embeddings in bounded retry (`KG_EMBED_RETRY_DELAYS`, 5/15/45s, 429/503 only). Not done (per `knowledge_graph_design.md` Maintenance): separate key/project for `gemini-embedding-001`, client-side rate limiting, read-path degradation.
4. **The corpus reingestion backlog is mostly complete.** Per `vector_corpus_reingestion_handoff.md`: core_theory and modern_analysis reingested with full metadata; Stalin was in-flight. Remaining: Stalin completion confirmation, curated Mao manifest + reingestion, audit of legacy rows missing `chunk_size` metadata.
5. `run_agent` supports analyst only, budget capped $0.50 (`self_runtime/tools.py`). `multi_delegate` is parallel-only — no inter-subtask dependencies; the worker unblock query (`telegram/tasks.py:1906`) only unblocks `plan_role='synthesis'` rows.

Where each upgrade pays off: the **Critic** on task reports and publications (Phases 1–2); the **Planner** in autonomous ticks (known `tick_no_durable_action` loops) and multi-step delegated missions (Phases 3–4); full LATS nowhere at current cost constraints (Phase 8).

---

## Phase 1 (P0) — Activate the Task Critic (wire the dead verifier)

**Status: SHIPPED in shadow mode (2026-07-08).** Implemented as designed; see `multi_agent_architecture.md` "Post-Hoc Verification (Critic)" for the operational description and `scripts/smoke_task_verification.py` for the hermetic smoke test (25/25). Awaiting 1–2 weeks of shadow data before the enforce decision.

**Goal**: every completed delegated task gets independent post-hoc verification; failed verification triggers the existing bounded auto-retry.

**Motivation**: ~450 lines of finished Critic code with zero call sites. `verification_status` columns, retry-chain guards, restart handoff, and the `on_complete(verification_status=...)` callback signature already exist.

**Design**:
- Default-policy synthesizer in `telegram/tasks.py`: when `metadata.verification` is absent, derive one from `agent_type` — programmer → `task_report` + `server_logs`; analyst/scout/diplomat → `task_report`; visualizer/diary → none. Explicit `metadata.verification.required=false` opts out.
- Add an optional `verification` object (checks/urls/log_service/retry_limit) to the `delegate`/`multi_delegate` tool schemas so the orchestrator can set policy per delegation — `_normalize_verification_policy` already parses these fields.
- Call `_run_verification(...)` from `process_task` after `_persist_task_success`, passing `chat_with_tools_fn` and a **low-tier** `get_model_fn` (writer precedent: cheap model judges; verifier budget is already $0.15). In enforce mode, feed FAIL into `_maybe_redelegate_after_verification_failure`.
- Config key `task_verification_mode: off | shadow | enforce` in `bot_config.py` `_CONFIG_META` + `config.json` (standard shadow→enforce pattern, cf. `gateway_enforce_mode`). Shadow = run verifier, persist verdict, surface it in the orchestrator report callback, never redelegate.

**Files**: `telegram/tasks.py` (call site + default policy), `self_runtime/tools.py` (delegate schema), `bot_config.py`, `telegram/bot.py` (surface verdict), `dev_docs/multi_agent_architecture.md`.

**Rollout**: shadow 1–2 weeks of live tasks → audit `verification_details` → enforce for programmer tasks first, then analyst/scout.

**Success criteria**: ≥90% of newly completed tasks have non-`pending` `verification_status`; shadow false-FAIL rate ≲10%; at least one genuine catch (unapplied edit, broken URL) during shadow. Verify via SQL over `telegram_tasks`.

**Effort**: S (1–2 days).

## Phase 2 (P0) — Generalized Reflexion service (diagnose → author-revise)

**Status: SHIPPED (2026-07-08).** `llm/reflexion.py` created; task-report hook live in `telegram/tasks.py` (`_maybe_reflexion_revise_report`, analyst/scout ≥1500 chars, cheap-tier diagnoser = the Phase-1 verifier fns, text-only author revision with a ≥50%-length guard); autonomous editorial diagnosis live in `autonomous_project.py` (`_diagnose_staged_drafts_for_tick`, per-draft-version caching via `editorial_diagnosis` events, DeepSeek-flash diagnoser with provider fallback, `<editorial-diagnosis>` prompt injection). Config flags `reflexion_task_reports` / `reflexion_autonomous_publish` (default on, /config panel). Smoke: `scripts/smoke_reflexion.py` (28/28). Deviation from the original sketch: task-report revision is deliberately **text-only** (no tool surface) so a revision turn can never re-run side-effectful publishing tools.

**Goal**: extract the writer's proven pattern into a reusable helper and apply it to (a) autonomous staged-draft publication, (b) long task reports.

**Motivation**: `run_diagnose_revise_pass` is the system's only blind-eval-validated (12/12) quality mechanism: cheap model diagnoses with read-only tools + numbered anchored notes + PASS marker + direction-not-wording; the strongest model revises *as author*, free to reject notes; failed diagnosis degrades to self-diagnosis. A general recipe, currently locked inside `writer/`.

**Design**:
- New module `llm/reflexion.py`: `async diagnose(content, context, *, chat_fn, model_tier="low", read_only_tools=None) -> notes | PASS` plus a thin `diagnose_then_revise(...)` composition, provider-agnostic via the `chat_with_tools` interface. Port the PASS-marker/note conventions from `writer/prompts.py`; leave `writer/` untouched as the reference implementation.
- **Autonomous publish gate**: the cross-tick stage→publish design already forces a fresh-context verification wake. On the publish tick, before publishing, run a diagnosis pass (deepseek_flash) over the staged draft + fact-check notes and inject the notes as an `<editorial-diagnosis>` block into the tick prompt; the tick agent (author, main autonomous model) revises and re-stages or proceeds on PASS. Context injection, not a new control loop — complements the structural gates in `autonomous_publication_controls.py` (which deliberately refuse semantic judgment).
- **Task reports**: in the task success path, for analyst/scout reports over a size threshold, run one diagnosis; if not PASS, one author-revision turn with the same agent model within remaining budget.
- Flags: `REFLEXION_AUTONOMOUS_PUBLISH`, `REFLEXION_TASK_REPORTS`; per-agent opt-out via `config/agent_runtime.json`.

**Files**: new `llm/reflexion.py`; `autonomous_project.py` (publish-tick hook); `telegram/tasks.py`; `dev_docs/autonomous_project.md`.

**Success criteria**: replay K recent staged drafts through diagnosis and human-review note precision (writer-style blind check); post-publication correction frequency declines; per-report cost delta ≤ ~$0.05 when PASS.

**Effort**: M (3–5 days).

## Phase 3 (P1) — Plan-and-Execute: dependency DAGs + replanning

**Status: SHIPPED (2026-07-08).** `depends_on` (earlier-index-only → structurally cycle-free) in `multi_delegate` items, max 8 tasks/plan; dependents created `status='blocked'` with `metadata.depends_on_task_ids`; worker runs `_unblock_dependency_tasks_sync` each loop (missing dependency rows count as terminal; 48h watchdog fails deadlocked tasks so synthesis always completes); `<dependency-results>` injection via `prompt_context.format_dependency_results` includes failed dependencies marked by status. Planning stays in the orchestrator model as designed (no `plan_mission` tool yet — add only if orchestrator-authored plans prove weak). Replanning v1 = failure context flows to dependents/synthesis/orchestrator callback. Smoke: `scripts/smoke_plan_dag.py` (22/22). Remaining from the sketch: live 3-stage end-to-end run on production infra still to be observed.

**Goal**: upgrade `multi_delegate` from "parallel fan-out + one synthesis" to small dependency graphs, so multi-step missions (research → analyze → publish) run without the operator stitching stages.

**Design**:
- Schema-light, reusing existing columns: `multi_delegate` task entries gain optional `depends_on` (list of sibling indices). Dependent tasks are created `status='blocked'` with dependency IDs in `metadata`. Generalize the worker unblock query (`telegram/tasks.py:1906`) from `plan_role='synthesis'`-only to any blocked task whose recorded dependencies are all terminal; on unblock, inject predecessors' results as a `<dependency-results>` block (same mechanism as `<subtask-results>` in `prompt_context.format_subtask_results`).
- Planning stays in the orchestrator model first (Simplicity First): the tool schema + description teach it to express dependencies. Only if orchestrator-authored plans prove weak, add a dedicated `plan_mission` tool running a bounded medium-tier planner call that drafts the task list + per-task `success_criteria` (delegation-contract fields already exist).
- Replanning (depends on Phase 1): when a subtask exhausts verification retries, mark downstream dependents and hand the failure to the synthesis/final task and orchestrator callback. v1: surfaced for the orchestrator to decide; v2: automatic planner-revision call. Guards: max plan size (~6 tasks), max depth, reuse existing handoff loop guards.

**Files**: `self_runtime/tools.py` (`_exec_multi_delegate`), `telegram/tasks.py` (unblock + context injection), `task_store.py`, `dev_docs/multi_agent_architecture.md`; extend `scripts/smoke_runtime.py`.

**Success criteria**: a 3-stage dependent plan completes end-to-end on live infra; existing parallel-plan behavior unchanged (regression check on `plan_role='synthesis'` unblock); blocked tasks never deadlock (stale-blocked watchdog in `system_monitor`).

**Effort**: M (3–6 days).

## Phase 4 (P1) — CLAW in the autonomous loop: tick-objective Planner + durable tick Critic

**Status: SHIPPED (2026-07-08).** `_plan_tick_objective` (OBJECTIVE/ARTIFACT/WHY over the assembled tick context, capped at 20k chars, `tick_objective` event, `<tick-objective>` injection) and `_review_tick_outcome` (VERDICT advanced/partial/no-op vs `_collect_tick_actions`, `tick_review` event; partial/no-op verdicts feed the next tick's warnings via `_recent_tick_attention_events`). Both use the shared `_cheap_reviewer_profile` (DeepSeek flash / provider low tier). Deviations from the sketch: flags are **config keys** (`autonomous_tick_planner`, `autonomous_tick_critic`, default on, /config panel) rather than env vars, matching the Phase 1/2 convention; the critic does not run on `tick_error` ticks; the N≥3-consecutive-no-op enforcement hook is not yet implemented. Smoke: `scripts/smoke_tick_planner_critic.py` (21/21). Success measurement: compare `tick_no_durable_action` + no-op `tick_review` rates over the two weeks after 2026-07-08 vs the two weeks before.

**Goal**: attack the two known failure modes (`tick_error` loops, `tick_no_durable_action` no-op wakes) with a cheap planner before the wake and a cheap critic after it.

**Design**:
- **Pre-tick Planner**: before the main agent wake in `_run_one_tick`, one bounded deepseek_flash call reads what `_build_task_prompt` already assembles (plan, latest synthesis, recent notes, tick warnings, staged drafts, advisories) and emits "this tick's single objective + expected durable artifact (note/draft/plan revision/publication)". Injected as `<tick-objective>` into the tick prompt; logged as a `tick_objective` project event. The existing "advance by exactly one concrete step" prompt line becomes a concrete, externally chosen step.
- **Post-tick Critic**: after the tick, deepseek_flash compares the objective against `_collect_tick_actions` output (notes/drafts/publications/plan/state — already collected) and emits `tick_review`: advanced / partial / no-op + one-line reason, logged as a project event. Add `tick_review` to the `_recent_tick_attention_events` set feeding the next tick prompt — the current self-critique paragraph (which dies with chat text) becomes durable Reflexion across ticks.
- Enforcement (later flag): N≥3 consecutive no-op reviews → auto-create an operator advisory or pause + owner alert via existing autonomous status surfaces.
- Flags: `AUTONOMOUS_TICK_PLANNER`, `AUTONOMOUS_TICK_CRITIC` env vars (the timer process re-reads env each run — cheap rollout/rollback).

**Files**: `autonomous_project.py` (`_run_one_tick`, `_build_task_prompt`, `_collect_tick_actions`, event types), `agents/autonomous.py` (one prompt note), `dev_docs/autonomous_project.md`.

**Success criteria**: from `autonomous_project_events` — `tick_no_durable_action` rate over 2 weeks drops vs the prior 2 weeks; added cost per tick ≤ ~10% (two flash calls); no increase in `tick_error`.

**Effort**: M (2–4 days).

## Phase 5 (P1) — Memory into task & autonomous contexts + failure write-back

**Status: SHIPPED (2026-07-08).** `recall_experiences_block` + `is_duplicate_experience` + `save_experiential_memory(dedupe=)` now live in `memory_store/experiential.py`; chat's `_fetch_relevant_experiences` and `experience_writer._is_duplicate` are thin delegates. Task worker injects `<past-experiences>` keyed on task content (`_build_task_context_content`); autonomous tick injects it keyed on title+topic+goal (`_build_task_prompt`, before the warnings block). Write-back hooks: verification FAIL (`_record_failure_experience`, `source_type=task_verification`), tick_error and no-op `tick_review` (`_record_tick_experience`, `source_type=autonomous_tick`) — all `category=mistake`, deduped over 30 days, never raise. KG auto-recall deliberately excluded as designed. Smoke: `scripts/smoke_experience_recall.py` (20/20).

**Goal**: extend the already-live chat auto-recall to the surfaces doing real work, and close the loop by writing structured lessons from failures.

**Design**:
- Factor `_fetch_relevant_experiences` (`telegram/commands.py:1666`) into a shared helper (in `memory_store/experiential.py` or `prompt_context.py`) taking provider + query; chat behavior stays identical.
- **Task worker**: `_build_task_context_content` adds a bounded `<past-experiences>` block keyed on task content (k=3, local BGE-M3 → milliseconds, zero API cost).
- **Autonomous tick**: same block in `_build_task_prompt`, keyed on project topic + (Phase 4) tick objective.
- **KG auto-recall: deliberately NOT per-message/per-task.** Each KG search costs a Gemini embedding call (quota-pressured; Phase 6's subject) plus latency; KG stays tool-driven. Optional later flag: entity-gated recall (only when the input names known entities).
- **Write-back**: on verification FAIL (Phase 1), tick_error / no-op `tick_review` (Phase 4), and auto-retry exhaustion, save a structured `experiential_memory` row (`category=mistake`, `source_type=task_verification|autonomous_tick`), reusing the dedupe check in `experience_writer._is_duplicate` (move it into `memory_store/experiential.py`). Complements the daily writer with event-driven, high-signal entries.

**Files**: `memory_store/experiential.py`, `telegram/tasks.py`, `autonomous_project.py`, `telegram/commands.py` (refactor to shared helper), `experience_writer.py` (dedupe extraction only).

**Success criteria**: sampled task/tick prompts contain the block when relevant memories exist; verification-failure lessons visibly recalled on similar subsequent tasks; no latency regression (local embeddings only).

**Effort**: S–M (2–3 days).

## Phase 6 (P1, parallel ops track) — KG embedding resilience completion

**Goal**: finish what the retry wrapper started; eliminate remaining 429/503 impact per `knowledge_graph_design.md` Maintenance guidance.

**Design** (retry/backoff already exists — do not rebuild):
- **Key separation**: `KG_GEMINI_API_KEY` (falls back to the main Gemini key) consumed by `graph_memory/config.py`/`service.py`, provisioned via the systemd-creds flow in `secret_management.md`.
- **Client-side rate limiter**: small async token bucket inside `GeminiEmbedderWithRetry` (`KG_EMBED_MAX_RPS`), so batch producers (scout→KG ingest, `kg_enricher`, curation ingest) don't burst into 429s that retries then paper over.
- **Read-path degradation**: when `knowledge_graph_search` fails on embeddings after retries, fall back to Neo4j full-text/BM25 (Graphiti hybrid search has keyword components) or, minimally, return a clean "KG search degraded (rate limit) — retry shortly or use vector_search" message. Callers already tolerate `get_kg_service() is None`; extend that discipline to per-query embedding failures.
- Optional stretch (only if failures persist after key separation): deferred-write retry queue for failed episode ingestion, drained by `leninbot-kg-integrity.timer`.

**Files**: `graph_memory/service.py`, `graph_memory/config.py`, `kg_runtime/search.py`, `dev_docs/knowledge_graph_design.md`, secret provisioning.

**Success criteria**: `scripts/check_kg_integrity.py --smoke-query` shows no 429-degraded runs over a month; journal retry warnings trend to zero; no KG write losses.

**Effort**: S–M (2–3 days; +2 for the optional queue).

## Phase 7 (P2, background ops) — Vector corpus reingestion completion

**Goal**: close the documented backlog in `vector_corpus_reingestion_handoff.md`.

**Steps**: (1) confirm the Stalin server-side pass completed (validation queries in the handoff doc); (2) build the curated Mao manifest — the doc explicitly forbids blind ingestion of the ~9.7M-char crawl; (3) reingest Mao on the Windows GPU host (`temp_dev/vector_reingest.py` pattern, 3000/300 chunks, canonical `Mao` author, staging-layer → promote pattern used for modern_analysis); (4) audit remaining legacy rows lacking `chunk_size` and normalize or reingest per source family; (5) update `project_state.md` + the handoff doc.

**Files**: operational only — no runtime code changes. **Success**: validation queries return zero missing titles/chunk metadata in `core_theory`; `vector_search(author="Mao")` returns well-attributed chunks. **Effort**: M, dominated by curation time; fully parallel to all other phases.

## Phase 8 (P2, conditional) — Bounded best-of-N for hard subtasks; LATS: deferred

**LATS verdict — recommend against implementing it.** Full LATS is MCTS over agent trajectories: node expansion, value backprop, rollback. It fails here on three counts: (1) **cost** — 10–30× tokens per task against a real cost constraint (the writer cache-layout work exists precisely because ~$1.17/turn of waste was unacceptable); (2) **state** — trajectories here have side effects (publish, KG writes, tasks, email); forking/rolling back world-state would require a sandbox layer that contradicts Simplicity First; (3) **evidence** — LATS wins are on stateless benchmarks (HotpotQA, WebShop, code), not long-horizon operational agents. Recorded here so the CLAUDE.md goal line stops implying it is pending work.

**Cheap variant worth piloting instead**: best-of-N with judge, restricted to **read-only** agents (analyst/scout) so parallel trajectories can't double-publish. `delegate(..., strategy="best_of_2")` creates 2 parallel subtasks (reuse `multi_delegate` machinery; optionally different tiers/approach hints) plus a judge task (Phase 1 verifier prompt style, medium tier) that picks or merges. Trigger only on explicit orchestrator opt-in, or as the retry strategy after a verification failure (retry-as-best-of-2 instead of single retry). **Gate**: on a small set of hard research questions, best-of-2+judge must beat single-run quality at ~2.3× cost, else drop the phase permanently.

**Effort**: M (3–4 days), only after Phases 1–4 prove out.

---

## Sequencing

| Order | Phase | Priority | Depends on |
|---|---|---|---|
| 1 | 1 Task Critic wiring | P0 | — |
| 2 | 2 Generalized Reflexion | P0 | — (shares critic prompt conventions with 1) |
| 3 | 6 KG resilience | P1 | — (parallel ops track, can go anytime) |
| 4 | 4 Autonomous tick Planner/Critic | P1 | — |
| 5 | 5 Memory into task/autonomous loops | P1 | write-back triggers from 1 & 4 |
| 6 | 3 Plan-and-Execute DAGs | P1 | 1 (verification signal for replanning) |
| 7 | 7 Corpus completion | P2 | — (background, operator-driven) |
| 8 | 8 Best-of-N pilot | P2 | 1–4 outcomes |

Every phase ships behind a flag, starts in shadow/log-only mode where output reaches users, and updates its dev_docs page per CLAUDE.md.

## Open decisions

1. **Verifier model tier** (Phase 1): pin verification to low tier (haiku/deepseek_flash) for cost, or medium for judgment quality? Shadow phase should compare both on the same tasks before enforce.
2. **Cross-provider critique**: when tasks run on DeepSeek, is a DeepSeek verifier independent *enough*, or should the Critic deliberately be a different provider than the Executor? (Writer precedent: DeepSeek diagnoses Fable's prose successfully — the reverse pairing is untested.)
3. **Autonomous tick cost ceiling** (Phase 4): is +2 flash calls per hourly tick (~+10%) acceptable, and should the tick Critic run on `tick_error` ticks too?
4. **Phase 3 shape**: extend `multi_delegate` with `depends_on` (recommended, minimal) vs. a new `plan_mission` planner tool — is an explicit Planner component wanted for its own sake, or only if orchestrator-authored plans prove weak?
5. **KG key separation** (Phase 6) is an operator/billing decision (second Google AI Studio key or Cloud project), not just code — needs a go-ahead before the phase is fully effective.
6. **Mao reingestion** (Phase 7) needs the Windows GPU host and manual curation — who/when? The only backlog item requiring off-server work.
7. **Web chat memory**: should the public surface also get experience auto-recall? Risk: leaking operator-context lessons publicly. Recommendation: no (or category-filtered) unless explicitly wanted.
8. **CLAUDE.md goal text**: once the LATS-defer verdict is accepted, update the project-goal line to point at this roadmap so future sessions don't re-litigate it.
