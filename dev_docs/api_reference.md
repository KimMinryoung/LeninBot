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
| `GET` | `/history` | chat history visible to fingerprint/proxy identity |
| `GET` | `/sessions` | session list visible to fingerprint/proxy identity |
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

Limits are enforced in `ChatRequest`: message 1-8000 chars, session ID 1-128 chars, fingerprint max 256 chars, persona max 64 chars. `regenerate_from_id`, `tone_feedback`, and `feedback_note` are optional; when `regenerate_from_id` is present, the server verifies that the target `chat_logs.id` belongs to the same fingerprint/session/persona, excludes that prior answer from the regenerated prompt history, regenerates that user turn, updates the same `chat_logs` row with the new answer, and returns the same `message_id`.

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

### `POST /chat/feedback`

Stores explicit feedback for a completed web-chat answer. The frontend should use `message_id` from the final `/chat` SSE `answer` event. Feedback is scoped by fingerprint/session/persona and is folded into the next normal `/chat` turn once, then marked consumed so specific correction notes do not keep reappearing. Feedback passed inline with `regenerate_from_id` is applied only to that regeneration request.

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
