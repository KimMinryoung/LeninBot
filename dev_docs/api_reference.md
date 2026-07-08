# API Reference

최종 확인 기준: 2026-06-19 `api.py`.

`api.py` exposes the internal FastAPI service used by the frontend, admin tools, A2A, and email bridge. Production listens on `127.0.0.1:8000` behind the frontend/Nginx boundary. The personal fiction writer routes are implemented in `api_routes/writer.py` and are also exposed by `novel_writer_api.py` on port `8001`; `api.py` includes them as a temporary compatibility fallback during rollout.

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

`/writer` is owned by `api_routes/writer.py`. `novel_writer_api.py` serves the dedicated writer process (`novel-writer-api.service`, port 8001), while `api.py` still includes the same routes temporarily for compatibility. Through the public frontend, use `/writer`; the frontend requires the existing admin login and proxies `/api/proxy/writer` to the dedicated writer service with backend credentials injected server-side. Direct backend calls to the data and generation routes require `X-Writer-Key` or `X-Admin-Key`. This workspace is separate from `/chat`, `web_chat.py`, selectable personas, and `webchat_model`. It uses the `writer/` package (`creative_writer.py` is a compatibility shim), stores writer state in writer-specific PostgreSQL tables, and defaults to Anthropic Messages API with `model="claude-fable-5"` while also exposing configured DeepSeek writer model choices.

Apply the explicit migration before first use or after schema changes:

```bash
venv/bin/python scripts/schema_migrations.py --only writer-tables
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/writer/projects` | list writer projects plus Fable pricing metadata and manuscript character counts; `?status=deleted` lists the trash |
| `POST` | `/writer/projects` | create a writer project and its empty manuscript |
| `GET` | `/writer/projects/{project_id}` | project metadata and ordered message log |
| `PATCH` | `/writer/projects/{project_id}` | update title, premise, and style notes |
| `DELETE` | `/writer/projects/{project_id}` | soft delete: moves the project to trash (`status='deleted'`, data intact); `?permanent=true` hard-deletes the project, manuscript, chunks, revisions, messages, and documents |
| `POST` | `/writer/projects/{project_id}/restore` | restore a trashed project to active |
| `GET` | `/writer/projects/{project_id}/manuscript` | fetch the canonical manuscript body |
| `PUT` | `/writer/projects/{project_id}/manuscript` | replace the canonical manuscript body and rebuild searchable chunks |
| `GET` | `/writer/projects/{project_id}/manuscript/search` | search indexed manuscript chunks with `q` and optional `limit` |
| `POST` | `/writer/projects/{project_id}/manuscript/append` | append text to the manuscript and rebuild chunks |
| `POST` | `/writer/projects/{project_id}/manuscript/replace` | replace a character range in the manuscript and rebuild chunks |
| `GET` | `/writer/projects/{project_id}/manuscript/revisions` | list manuscript revision metadata |
| `GET` | `/writer/projects/{project_id}/documents` | list background documents (id, title, kind, char count) |
| `POST` | `/writer/projects/{project_id}/documents` | create or overwrite a background document by title (upsert) |
| `GET` | `/writer/projects/{project_id}/documents/{document_id}` | fetch one background document with content |
| `PUT` | `/writer/projects/{project_id}/documents/{document_id}` | update title, kind, and content of a document |
| `DELETE` | `/writer/projects/{project_id}/documents/{document_id}` | delete a background document |
| `GET` | `/writer/projects/{project_id}/stream` | reattach SSE to a live background writer run (`no_active_run` when idle) |
| `PUT` | `/writer/settings` | persist the admin's model choice (`{"model": "<choice key>"}`); returned as `selected_model` on the project list |
| `POST` | `/writer/projects/{project_id}/messages` | stream a Claude Fable 5 answer as SSE |

Project request:

```json
{
  "title": "Novel title",
  "premise": "Core situation and continuity",
  "style_notes": "Voice, POV, tense, constraints"
}
```

Manuscript save request:

```json
{
  "body": "Full manuscript text...",
  "note": "manual save"
}
```

Manuscript append request:

```json
{
  "text": "New scene text...",
  "note": "append assistant reply"
}
```

Manuscript replace request:

```json
{
  "start": 1200,
  "end": 1840,
  "replacement": "Revised passage...",
  "note": "replace selected passage"
}
```

Background document request (`POST`/`PUT`):

```json
{
  "title": "Characters",
  "kind": "character",
  "content": "Reference notes kept separate from the manuscript..."
}
```

Documents are per-project reference material (worldbuilding, character sheets, outline, research) stored in `writer_documents`, unique by `(project_id, title)`; `POST` upserts by title. A document inventory (titles, kinds, sizes) is injected into the model's manuscript-context system block, and the model reads them with the `read_document`/`search_documents` tools and maintains them with `save_document`. Documents with kind `pinned` (up to 2, 6000 chars each) are injected in full every turn; the prompt instructs the model to maintain a pinned 'Story so far' synopsis so long-novel continuity does not require re-reading the manuscript.

Model selection: the admin's choice persists server-side in `writer_settings` (`PUT /writer/settings`, loaded as `selected_model` on the project list) and is used when a message request omits `model`. Choices: `fable` (Claude Fable 5, adaptive thinking, effort high), `fable_fast` (effort low), `deepseek_pro`, `deepseek_flash`. Conversation history sent to the model is budgeted to ~30k chars (newest first, minimum 8 messages).

Message request:

```json
{
  "prompt": "Revise the selected passage so the narrator sounds colder.",
  "selection_start": 1200,
  "selection_end": 1840
}
```

`selection_start` and `selection_end` are optional character offsets into the canonical manuscript. When present, the selected range is supplied to the model alongside a bounded recent manuscript tail. Writer sends project instructions and manuscript context as separate 1-hour prompt-cache system blocks so repeated turns against unchanged context can produce `cache_read_input_tokens`; the UI shows cache read/write token counts from the final usage metadata. The visible UI does not expose `max_tokens`; `writer/config.py` owns the server-side default. The writer loop allows up to 16 model/tool rounds for continuity searches, edit retries, and final append/replace calls. The message route streams `text/event-stream` payloads: `user_saved`, `text_delta`, `budget`, `tool`, `provider_retry`, `ping`, `done`, and `error`. `provider_retry` reports a transient Anthropic/DeepSeek connection retry; `ping` is an empty keepalive event emitted while waiting on long model/tool calls. Writer generation is owned by a server-side background task registered in an in-process run registry; the browser SSE connection is only an observer. The run starts before the first SSE byte is sent, so an immediate client drop cannot prevent it. If the browser/proxy disconnects before the final event, the background task continues and persists successful manuscript edits plus final assistant/error metadata; the UI first reattaches to the live run via `GET /writer/projects/{project_id}/stream` (which replays a `run_status` event carrying `text_snapshot` — everything streamed so far — then follows live), and only falls back to polling project messages when no active run exists. A page reload also auto-reattaches to a still-running generation. If the server restarts mid-run, the cancelled run persists an explanatory assistant `error` message noting that tool edits applied before the restart are saved. For writer calls, a provider stream that accepts the request but produces no text/final event for 70 seconds is treated as a transient provider stall and retried by the server before surfacing an `error`; if no model/tool progress reaches the browser for 240 seconds, the stream returns an explanatory `error` event while the server-side writer run remains active. Writer tool calls run under the `system:writer` security-gateway caller; the allowed surface is the `system.writer` tool profile in `tool_gateway/profiles.py`, enforced by `writer.tools.build_writer_tools`, with explicit risk classes: `search_manuscript`, `read_manuscript`, `read_document`, and `search_documents` are `read`, while `append_to_manuscript`, `replace_in_manuscript`, and `save_document` are `write`. `read_manuscript` returns any character-offset slice of the saved manuscript (default: last 5000 chars, max 20000 per call). `search_manuscript` and `replace_in_manuscript` fall back to whitespace/quote-insensitive matching when a verbatim match fails, and their failure messages steer the model toward short distinctive queries instead of long quoted paragraphs. `done` includes `model`, `stop_reason`, token usage when provided by Anthropic, estimated USD cost, `manuscript_text`, `commentary_text`, and `display_text`. The model is instructed to edit the manuscript through writer tools when the request asks for an edit, and to use `<commentary>` only for questions, rationale, continuity notes, and clarification questions. Fable refusals arrive as successful streams whose final `stop_reason` can be `refusal`.

## Admin Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin/private-reports` | noindex admin HTML shell; data actions still require `X-Admin-Key` through the JSON endpoints |
| `GET` | `/logs` | raw chat log admin list |
| `GET` | `/admin/users` | list public/admin web accounts with fingerprint/passkey/chat counts |
| `GET` | `/admin/users/{user_id}` | one account detail with linked fingerprints and safe passkey metadata |
| `PATCH` | `/admin/users/{user_id}` | rename an account |
| `POST` | `/admin/users/{source_user_id}/merge` | merge a regular source account into a target account |
| `DELETE` | `/admin/users/{user_id}` | delete a regular account after username confirmation; chat logs remain fingerprint-retained |
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
