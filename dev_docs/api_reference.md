# API Reference

최종 확인 기준: 2026-05-09 `api.py`.

`api.py` exposes the internal FastAPI service used by the frontend, admin tools, A2A, and email bridge. Production listens on `127.0.0.1:8000` behind the frontend/Nginx boundary.

## Authentication

Admin endpoints require:

```
X-Admin-Key: <ADMIN_API_KEY>
```

Missing or invalid key returns `403`. If the server has no `ADMIN_API_KEY`, admin endpoints return `503`.

Some public web-chat requests may include frontend proxy headers such as `X-User-Fingerprints`; these are accepted only when the proxy secret path marks the request trusted in `api.py`.

## Public Endpoints

| Method | Path | Description |
|---|---|---|
| `GET`, `HEAD` | `/` | health check |
| `GET`, `HEAD` | `/health` | health check |
| `GET`, `HEAD` | `/api/health` | health check alias |
| `POST` | `/chat` | public web chat SSE stream |
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
  "fingerprint": "browser-fingerprint"
}
```

Limits are enforced in `ChatRequest`: message 1-8000 chars, session ID 1-128 chars, fingerprint max 256 chars.

Response is `text/event-stream`. Event payloads are JSON:

```text
data: {"type":"log","node":"...","content":"..."}
data: {"type":"answer","content":"..."}
data: {"type":"error","content":"..."}
```

Concurrency and rate controls:

- one active request per `session_id`
- per-client sliding window from `WEBCHAT_RATE_LIMIT` and `WEBCHAT_RATE_WINDOW_SECONDS`
- global active request cap from `WEBCHAT_GLOBAL_ACTIVE_LIMIT`

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

Current CORS allow-list in `api.py`:

- `https://bichonwebpage.onrender.com`
- `http://localhost:3000`

Update this list in code when frontend deployment origin changes.
