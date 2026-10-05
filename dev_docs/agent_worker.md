# Agent worker

2026-10-05. leninbot's agents as a service for other services: a requester queues a task
(instructions, material, allowed research tools, result schema, optional validator) and
polls for a schema-shaped result with the sources it rests on. The first requester is the
frontend's CommuLingo enrichment pipeline (frontend `dev_docs/commulingo-agent-pipeline.md`):
the frontend decides what work is needed, validates and saves; the worker only researches
and writes the result. It never writes CommuLingo.

## Pieces

| Piece | Where | What |
|---|---|---|
| MCP endpoint | `worker/endpoint.py`, mounted on leninbot-api at `POST /worker/mcp` (172.17.0.1:8000, the port the frontend container already reaches) | `agent_task_submit`, `agent_task_get`, `agent_task_cancel`; stateless JSON-RPC, one message per request |
| Executor | `worker/service.py`, `leninbot-worker.service` | polls `agent_worker_tasks`, runs up to `WORKER_CONCURRENCY` (2) tasks, stops cancelled ones, re-runs a task interrupted by a restart once its 15-minute lease expires (2 attempts) |
| Run loop | `worker/runner.py` | generalized from the CommuLingo pipeline's `model_call`: AgentSpec per tier, research tools, terminal `worker_submit_result` checked against `resultSchema` and the validator; content-risk refusal retried once on the fallback provider |
| Catalog | `worker/catalog.py` | research tools, model tiers, validator tools |
| Table | `worker/schema.sql` (`scripts/schema_migrations.py --only agent-worker`) | request, status, lease, result, sources, usage, rejections |

## Request

`agent_task_submit {idempotencyKey, request}`; the same key with the same request returns
the existing task, a different request under the same key is refused (409).

- `instructions` (≤ 30,000 chars): appended to the worker's base prompt (untrusted
  material, cite source ids, submit exactly once). No political-line block.
- `input`: any JSON, attached as a runtime record.
- `tools`: subset of `wiki_search`, `wiki_get`, `web_search`, `fetch_url`,
  `knowledge_graph_search`, `vector_search`, `read_corpus_passage`.
- `resultSchema`: JSON Schema (type object) of the terminal tool. Under the OpenAI tier the
  strict wire form is derived (`commulingo/pipeline/strict_input.py`).
- `validator` (optional): `{tool, arguments}` naming a CommuLingo admin MCP check from
  `catalog.VALIDATOR_TOOLS` (`editorial_validate`, a dry run). The submitted value goes in
  `arguments.value`; a rejection is returned to the model inside the same loop.
- `sources` (optional): `[{url?, title?, text}]` shown up front as S1.. (e.g. pages an
  earlier stage retrieved).
- `tier`: `author` (GPT-6 Luna, the curator's settings, budget cap $0.60) or `review`
  (DeepSeek Flash, the reviewer's settings, cap $0.40). `budgetUsd`, `maxRounds`.

## CommuLingo session kinds

`kind: commulingo_editor` (research and draft, one author session) and
`kind: commulingo_review` (independent review) run the CommuLingo pipeline's own
`Editor`/`Review` (`worker/commulingo.py`) for a queue the frontend owns. The request
is `{kind, input: {job, artifacts}, budgetUsd ≤ 0.6}`: the job row and its artifacts as
the frontend stored them. The session reads the entry and validates drafts through the
CommuLingo admin MCP as before, keeps fetched pages in leninbot's source cache
(`commulingo_pipeline_sources`, `fetch_cache`, `job_sources`; bodies expire hourly in
`leninbot-worker`), and never writes CommuLingo. `result` is
`{stage: {value, nextStage, status, delaySeconds}, artifacts: [editor_checkpoint,
fetch_failures], notes: [{decision, reason}], usage, metrics, costComplete}`; a failed
session returns the same without `stage`, so its checkpoint and cost still reach the
frontend, which stores the artifacts, writes review notes and decides the next stage.

## Result

`agent_task_get {taskId}` → `status` (`queued|running|done|failed|cancelled`), `result`
(`{value, validation}`), `sources` (every page opened with `wiki_get`/`fetch_url`/
`read_corpus_passage` or supplied: id, tool, arguments, fetchedAt, sha256, chars, text ≤ 60k),
`usage` (`costUsd`, `modelCalls`, `providerFallback?`), `rejections`, `error`. A failed task
keeps its sources and usage. Tasks are visible only to the client that queued them.

## Clients

`scripts/worker_token.py <client>` prints a token once (stdout) and the
`WORKER_MCP_CLIENTS` entry (stderr, `name:sha256`). Add the entry to `.env` and restart
leninbot-api. The frontend holds its token as `COMMULINGO_WORKER_TOKEN` with
`COMMULINGO_WORKER_URL=http://host.docker.internal:8000/worker/mcp`.

The executor reaches CommuLingo validators with leninbot's CommuLingo admin MCP token
(`commulingo/mcp_client.py`). Tool calls run under caller context `interface=autonomous`,
`agent_name=worker`, `is_owner=false`, scope `agent_worker_task:<id>`; `worker_submit_result`
is risk class `state`.

## Operating

```bash
sudo cp systemd/leninbot-worker.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now leninbot-worker.service
journalctl -u leninbot-worker -f
```

Verified 2026-10-05 from the frontend container: a `review`-tier Wikipedia lookup ($0.001)
and an `editorial_validate` round trip where three rejections were corrected in-loop.

Not yet: citation-support checking of (claim, excerpt) pairs and passage labels (the
pipeline's `citation_gate`/`evidence.Passages`) move here when the frontend pipeline's
research stage is ported (frontend W3).
