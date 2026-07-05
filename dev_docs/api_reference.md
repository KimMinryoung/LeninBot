# API Reference

최종 확인 기준: 2026-06-19 `api.py`.

`api.py` exposes the internal FastAPI service used by the frontend, admin tools, A2A, and email bridge. Production listens on `127.0.0.1:8000` behind the frontend/Nginx boundary.

## Authentication

Admin endpoints require:

```
X-Admin-Key: <ADMIN_API_KEY>
```

Missing or invalid key returns `403`. If the server has no `ADMIN_API_KEY`, admin endpoints return `503`.

Some public web-chat requests may include frontend proxy headers such as `X-User-Fingerprints`; these are accepted only when the proxy secret path marks the request trusted in `api.py`.

Inbound A2A is controlled by non-secret env `A2A_ENABLED`. When false, `/.well-known/agent-card.json` returns `503` and `/a2a` returns a JSON-RPC error with HTTP `503` before any LLM call.

## Public Endpoints

| Method | Path | Description |
|---|---|---|
| `GET`, `HEAD` | `/` | health check |
| `GET`, `HEAD` | `/health` | health check |
| `GET`, `HEAD` | `/api/health` | health check alias |
| `POST` | `/chat` | public web chat SSE stream |
| `POST` | `/chat/feedback` | store rating/tone feedback for a web chat answer |
| `GET` | `/personas` | selectable public web-chat persona catalog |
| `GET` | `/history` | chat history visible to fingerprint/proxy identity |
| `GET` | `/sessions` | session list visible to fingerprint/proxy identity |
| `GET` | `/writer` | noindex browser shell for the admin-gated personal fiction workspace |
| `GET` | `/.well-known/agent-card.json` | public A2A discovery card |
| `POST` | `/a2a` | A2A JSON-RPC endpoint |
| `GET` | `/x402-demo/quote` | x402 demo quote route from `api_routes/x402_demo.py` |

### `POST /chat`

Request:

```json
{
  "message": "질문 텍스트",
  "session_id": "browser-session-id",
  "fingerprint": "browser-fingerprint",
  "persona": "cyber-lenin",
  "regenerate_from_id": 123,
  "tone_feedback": "more_in_character",
  "feedback_note": "더 차갑고 단정적으로"
}
```

Limits are enforced in `ChatRequest`: message 1-8000 chars, session ID 1-128 chars, fingerprint max 256 chars, persona max 64 chars. Unknown persona IDs fall back server-side to `cyber-lenin`. `regenerate_from_id`, `tone_feedback`, and `feedback_note` are optional; when `regenerate_from_id` is present, the server verifies that the target `chat_logs.id` belongs to the same fingerprint/session/persona, excludes that prior answer from the regenerated prompt history, regenerates that user turn, updates the same `chat_logs` row with the new answer, and returns the same `message_id`.

Selectable personas are defined in `web_personas.py`. Current public personas include `cyber-lenin`, `gramsci`, and `yezhov`; admin-only personas are omitted from `/personas` unless the request has a valid `X-Admin-Key`. Persona-specific chat history and feedback are scoped by the `persona` value. Gramsci's primary writings are expected to be retrieved through `vector_search(layer="core_theory", author="Gramsci")`; for Gramsci theory/concept triggers the server also performs a preflight vector lookup and injects a bounded grounding block into the current turn. His persona-only dossier under `identity/web_personas/gramsci/knowledge` is supplemental reading protocol and answer-structure material. Web chat exposes that dossier only through the active persona-bound `read_persona_context` tool, so other personas cannot read that namespace.

Response is `text/event-stream`. Event payloads are JSON:

```text
data: {"type":"log","node":"...","content":"..."}
data: {"type":"answer","message_id":123,"regenerated_from_id":null,"content":"..."}
data: {"type":"error","content":"..."}
```

Concurrency and rate controls:

- one active request per `session_id`
- per-client sliding window from `WEBCHAT_RATE_LIMIT` and `WEBCHAT_RATE_WINDOW_SECONDS`
- global active request cap from `WEBCHAT_GLOBAL_ACTIVE_LIMIT`

### `GET /personas`

Returns the persona picker catalog for web chat:

```json
{
  "personas": [
    {"id": "cyber-lenin", "display_name": "사이버-레닌", "description": "...", "default": true, "admin_only": false},
    {"id": "gramsci", "display_name": "안토니오 그람시", "description": "...", "default": false, "admin_only": false}
  ],
  "default": "cyber-lenin"
}
```

Admin-only personas are included only for valid admin requests.

### `POST /chat/feedback`

Stores explicit feedback for a completed web-chat answer. The frontend should use `message_id` from the final `/chat` SSE `answer` event. Feedback is scoped by fingerprint/session/persona. Manual `note` text is folded into the next normal `/chat` turn once, then marked consumed so free-form corrections do not keep reappearing. Dropdown `tone_feedback` is not injected as per-turn chat context; recent dropdown selections are aggregated into a standing response policy for that persona/session. Feedback passed inline with `regenerate_from_id` is applied only to that regeneration request, while any saved dropdown value still contributes to the ongoing policy aggregate.

Request:

```json
{
  "message_id": 123,
  "session_id": "browser-session-id",
  "fingerprint": "browser-fingerprint",
  "persona": "yezhov",
  "rating": 4,
  "tone_feedback": "more_in_character",
  "note": "더 위압적으로"
}
```

`tone_feedback` is optional and must be one of: `shorter`, `longer`, `warmer`, `colder`, `more_direct`, `more_in_character`, `less_formal`, `more_cited`. At least one of `rating`, `tone_feedback`, or `note` is required.

### `GET /history`

Query:

| Param | Required | Notes |
|---|---|---|
| `fingerprint` | conditional | anonymous browser identity |
| `session_id` | no | restrict to one session |
| `limit` | no | 1-200, default 50 |

Returns `{"history": [...]}` with sanitized `user_query`, `bot_answer`, and `created_at`.

### `GET /sessions`

Query:

| Param | Required | Notes |
|---|---|---|
| `fingerprint` | conditional | anonymous browser identity |
| `limit` | no | 1-200, default 50 |

Returns session IDs, first/last timestamps, message count, and a first-message preview.

## Personal Writer Endpoints

`/writer` serves a noindex HTML shell. Through the public frontend, use `/writer`; the frontend requires the existing admin login and redirects to `/api/proxy/writer`, where it injects the backend admin credential server-side. Direct backend calls to the data and generation routes require `X-Writer-Key` or `X-Admin-Key`. This workspace is deliberately separate from `/chat`, `web_chat.py`, selectable personas, and `webchat_model`. It uses `creative_writer.py`, stores state in `writer_projects` and `writer_messages`, and calls Anthropic Messages API with `model="claude-fable-5"`.

Apply the explicit migration before first use:

```bash
venv/bin/python scripts/schema_migrations.py --only writer-tables
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/writer/projects` | list writer projects plus Fable pricing metadata |
| `POST` | `/writer/projects` | create a writer project |
| `GET` | `/writer/projects/{project_id}` | project metadata and ordered messages |
| `PATCH` | `/writer/projects/{project_id}` | update title, premise, and style notes |
| `DELETE` | `/writer/projects/{project_id}` | delete a project and its messages |
| `POST` | `/writer/projects/{project_id}/messages` | stream a Claude Fable 5 answer as SSE |

Project request:

```json
{
  "title": "Novel title",
  "premise": "Core situation and continuity",
  "style_notes": "Voice, POV, tense, constraints"
}
```

Message request:

```json
{
  "prompt": "Write the next scene...",
  "request_kind": "draft",
  "max_tokens": 4096
}
```

`request_kind` accepts `draft`, `continue`, `revise`, `plan`, or `critique`; invalid values are normalized to `draft`. The message route streams `text/event-stream` payloads: `user_saved`, `text_delta`, `done`, and `error`. `done` includes `model`, `stop_reason`, token usage when provided by Anthropic, and estimated USD cost using Claude Fable 5 base input/output pricing. Fable refusals arrive as successful streams whose final `stop_reason` can be `refusal`.

## Admin Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin/private-reports` | noindex admin HTML shell; data actions still require `X-Admin-Key` through the JSON endpoints |
| `GET` | `/logs` | raw chat log admin list |
| `GET` | `/reports` | completed non-programmer task reports |
| `GET` | `/reports/{report_id}` | one completed task report |
| `GET` | `/private-reports` | private report list |
| `GET` | `/private-reports/{report_ref}` | private report detail by ID or slug |
| `POST` | `/private-reports` | create/update private report |
| `POST` | `/private-reports/{slug}/publish` | publish private report into public research documents |
| `DELETE` | `/session/{session_id}` | delete chat logs for a session |
| `POST` | `/email/poll` | run one email polling cycle |
| `GET` | `/email/inbound` | list inbound email records |
| `GET` | `/email/pending` | list outbound approvals pending |
| `GET` | `/email/messages/{message_id}` | email detail plus reply prompt input for inbound |
| `POST` | `/email/drafts` | queue outbound reply draft |
| `POST` | `/email/messages/{message_id}/approval` | approve/send, save draft, or reject outbound email |
| `POST` | `/email/messages/{message_id}/internal-approve` | mark inbound email approved for internal delivery |
| `GET` | `/email/internal-approved` | list inbound emails approved for internal delivery |
| `POST` | `/email/messages/{message_id}/internal-deliver` | deliver approved inbound email to internal input path |

### Private Report Request

`POST /private-reports`:

```json
{
  "title": "Report title",
  "slug": "stable-slug",
  "markdown_body": "# Markdown",
  "source_task_id": 123
}
```

`POST /private-reports/{slug}/publish`:

```json
{
  "title": "Optional public title",
  "body": "Optional replacement markdown"
}
```

## Email Approval Actions

`POST /email/messages/{message_id}/approval` accepts:

| `action` | Effect |
|---|---|
| `approve_send` | send queued outbound email |
| `save_draft` | keep as draft without sending |
| `reject` | reject outbound email |

## CORS

API CORS origins are read from `WEBCHAT_CORS_ORIGINS` as a comma-separated env value. `CORS_ALLOW_ORIGINS` is accepted as a compatibility alias. Default:

- `https://cyber-lenin.com`
- `http://localhost:3000`

Update `.env` or the systemd environment when frontend deployment origins change; Python code should not need to change.
