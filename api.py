import asyncio
import json
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query, Request, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from db import query as db_query, query_one as db_query_one
from email_bridge import (
    build_reply_prompt_input,
    deliver_inbound_email_to_internal_input,
    get_email_message,
    list_inbound_messages,
    list_messages_approved_for_internal_delivery,
    list_pending_email_approvals,
    mark_email_for_internal_delivery,
    queue_outbound_reply,
    reject_outbound_email,
    run_polling_cycle,
    send_outbound_email,
)
from private_report_tools import (
    get_private_report_sync,
    list_private_reports_sync,
    publish_private_report_sync,
    save_private_report_sync,
)

from secrets_loader import get_secret

_LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
logging.basicConfig(level=_LOG_LEVEL, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger().setLevel(_LOG_LEVEL)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Admin API key authentication ──────────────────────────────────
_ADMIN_API_KEY = get_secret("ADMIN_API_KEY", "") or ""
_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_admin(api_key: str = Security(_admin_key_header)):
    """Dependency that enforces admin API key for sensitive endpoints."""
    if not _ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin API key not configured on server")
    if not api_key or api_key != _ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing admin API key")


class PrivateReportRequest(BaseModel):
    title: str
    slug: str
    markdown_body: str
    source_task_id: int | None = None


class PublishPrivateReportRequest(BaseModel):
    body: str | None = None
    title: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── KG Eager Init ────────────────────────────────────────────────
    # Initialize Neo4j connection immediately at startup (background thread)
    # so the first real request doesn't pay the cold-start penalty.
    def _eager_init_kg():
        from shared import get_kg_service
        svc = get_kg_service()
        if svc:
            import logging
            logging.getLogger(__name__).info("[startup] KG eager init succeeded")
        else:
            import logging
            logging.getLogger(__name__).warning("[startup] KG eager init failed — will retry on first request")

    import threading
    threading.Thread(target=_eager_init_kg, daemon=True, name="kg-eager-init").start()

    # ── KG Health Check (10 min interval) ────────────────────────────
    from shared import start_kg_healthcheck
    start_kg_healthcheck(interval=600)

    # Telegram bot should run in its dedicated systemd service by default.
    # Optional fallback for single-process dev environments:
    run_telegram_in_api = os.getenv("RUN_TELEGRAM_IN_API", "false").strip().lower() in {"1", "true", "yes", "on"}
    bot_task = None
    if run_telegram_in_api:
        from telegram.bot import bot_main
        bot_task = asyncio.create_task(bot_main())
    yield
    if bot_task is not None:
        bot_task.cancel()



app = FastAPI(title="Cyber-Lenin API", lifespan=lifespan)


# Per-session locks to prevent concurrent requests from corrupting checkpointed state.
# Uses LRU-style eviction to prevent unbounded memory growth.
_session_locks: dict[str, asyncio.Lock] = {}
_SESSION_LOCKS_MAX = 200


def _get_session_lock(session_id: str) -> asyncio.Lock:
    """Get or create a lock for a session, evicting oldest if over limit."""
    if session_id not in _session_locks:
        if len(_session_locks) >= _SESSION_LOCKS_MAX:
            # Evict oldest (first inserted) entries that are not locked
            to_remove = []
            for k, v in _session_locks.items():
                if not v.locked():
                    to_remove.append(k)
                if len(_session_locks) - len(to_remove) < _SESSION_LOCKS_MAX // 2:
                    break
            for k in to_remove:
                del _session_locks[k]
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]





@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.api_route("/api/health", methods=["GET", "HEAD"])
async def api_health():
    return {"status": "ok"}


# ── x402 demo route (self-loop) ───────────────────────────────────
# Tiny micropayment demo over USDC on Base mainnet via the x402 'exact' scheme.
# The same wallet pays and receives — net USDC change is zero, only gas is spent.
X402_DEMO_AMOUNT_ATOMIC = 1000  # 0.001 USDC (~$0.001)


@app.get("/x402-demo/quote")
async def x402_demo_quote(request: Request):
    """x402-protected demo endpoint. No payment → 402 with PaymentRequirements;
    valid PAYMENT-SIGNATURE → settle on-chain and return a Cyber-Lenin aphorism.
    """
    import base64
    from crypto_wallet import x402
    from crypto_wallet.wallet import get_addresses

    addrs = get_addresses()
    pay_to = addrs.get("base")
    if not pay_to:
        raise HTTPException(status_code=503, detail="Server wallet not configured")

    resource = str(request.url)
    requirements = x402.build_payment_requirements(
        pay_to=pay_to,
        amount_atomic=X402_DEMO_AMOUNT_ATOMIC,
        resource=resource,
        description="Cyber-Lenin x402 demo: pay tiny USDC for an aphorism",
        mime_type="application/json",
    )

    payment_header = request.headers.get("PAYMENT-SIGNATURE")
    if not payment_header:
        body = x402.build_402_body(requirements)
        return Response(
            content=json.dumps(body, ensure_ascii=False),
            status_code=402,
            media_type="application/json",
            headers={
                "PAYMENT-REQUIRED": base64.b64encode(
                    json.dumps(requirements, separators=(",", ":")).encode()
                ).decode(),
            },
        )

    # Verify signature against demanded requirements
    try:
        decoded = x402.decode_payment_header(payment_header)
        signer = x402.verify_payment(decoded, requirements)
    except Exception as e:
        raise HTTPException(status_code=402, detail=f"x402 verification failed: {e}")

    # Settle on-chain
    try:
        settlement = await x402.settle_payment(decoded["payload"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"x402 settlement failed: {e}")

    if settlement.get("status") != "success":
        raise HTTPException(status_code=502, detail=f"settlement tx not success: {settlement}")

    # Settlement OK — return paid content
    aphorism = (
        "정치란 과학이며 예술이다. 그것은 하늘에서 떨어지는 것이 아니라 "
        "노력과 투쟁을 통해 얻어지는 것이다."
    )
    body = {
        "x402Version": x402.X402_VERSION,
        "message": "결제 검증 통과 — Cyber-Lenin의 격언:",
        "aphorism": aphorism,
        "payer": signer,
        "amount_atomic": int(decoded["payload"]["authorization"]["value"]),
        "tx_hash": settlement["tx_hash"],
        "gas_used": settlement["gas_used"],
    }
    return Response(
        content=json.dumps(body, ensure_ascii=False),
        status_code=200,
        media_type="application/json",
        headers={
            "PAYMENT-RESPONSE": x402.encode_settlement_header(settlement),
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bichonwebpage.onrender.com",
    "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    fingerprint: str = ""  # Browser fingerprint from localStorage (persistent across server restarts)


class EmailDraftRequest(BaseModel):
    inbound_message_id: int
    draft_body: str
    subject: str | None = None
    to_emails: list[str] | None = None
    approver_user_id: int | None = None
    metadata: dict | None = None


class EmailApprovalRequest(BaseModel):
    action: str  # approve_send | save_draft | reject
    note: str = ""


class EmailInboundApprovalRequest(BaseModel):
    note: str = ""
    approved_by: int | None = None


def format_sse(data: dict):
    """Server-Sent Events 포맷으로 변환"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


_PRIVATE_REPORTS_ADMIN_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex,nofollow">
  <title>Cyber-Lenin Private Reports</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f6f4ef;
      --panel: #ffffff;
      --text: #171717;
      --muted: #666257;
      --line: #d9d3c7;
      --accent: #9f2727;
      --accent-dark: #741c1c;
      --code: #111111;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.82);
      backdrop-filter: blur(10px);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    .bar {
      max-width: 1440px;
      margin: 0 auto;
      padding: 14px 18px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
    }
    h1 {
      font-size: 18px;
      line-height: 1.25;
      margin: 0;
      font-weight: 720;
      letter-spacing: 0;
    }
    .auth {
      display: flex;
      gap: 8px;
      align-items: center;
      min-width: min(540px, 100%);
    }
    input, button {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font: inherit;
      font-size: 14px;
    }
    input {
      background: var(--panel);
      color: var(--text);
      padding: 0 10px;
      width: 100%;
    }
    button {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      padding: 0 12px;
      cursor: pointer;
      white-space: nowrap;
    }
    button.secondary {
      background: transparent;
      color: var(--accent-dark);
      border-color: var(--accent-dark);
    }
    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      grid-template-columns: minmax(280px, 380px) minmax(0, 1fr);
      gap: 18px;
    }
    aside, section.viewer {
      min-width: 0;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
    }
    aside {
      overflow: hidden;
    }
    .tools {
      padding: 12px;
      display: grid;
      gap: 8px;
      border-bottom: 1px solid var(--line);
    }
    .search {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
    }
    .status {
      min-height: 20px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .list {
      max-height: calc(100vh - 142px);
      overflow: auto;
    }
    .report-row {
      width: 100%;
      height: auto;
      display: block;
      text-align: left;
      background: transparent;
      color: var(--text);
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      padding: 12px;
    }
    .report-row:hover, .report-row.active {
      background: #f1e9dd;
    }
    .row-title {
      font-weight: 700;
      font-size: 14px;
      line-height: 1.35;
      margin-bottom: 6px;
      white-space: normal;
    }
    .row-meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      word-break: break-all;
    }
    .viewer-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 6px;
    }
    .viewer-title {
      font-size: 20px;
      font-weight: 760;
      line-height: 1.3;
    }
    .viewer-meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      word-break: break-all;
    }
    pre {
      margin: 0;
      padding: 18px;
      color: var(--code);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 14px/1.62 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      min-height: calc(100vh - 180px);
    }
    .empty {
      padding: 18px;
      color: var(--muted);
      line-height: 1.6;
    }
    @media (max-width: 860px) {
      .bar, main { grid-template-columns: 1fr; }
      .auth { min-width: 0; }
      main { padding: 12px; }
      .list { max-height: 36vh; }
      pre { min-height: 44vh; }
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #171615;
        --panel: #211f1d;
        --text: #f0eee9;
        --muted: #aaa39a;
        --line: #3a342e;
        --accent: #b83a35;
        --accent-dark: #e2716b;
        --code: #f0eee9;
      }
      header { background: rgba(33,31,29,0.84); }
      .report-row:hover, .report-row.active { background: #2b2621; }
    }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <h1>Cyber-Lenin Private Reports</h1>
      <div class="auth">
        <input id="adminKey" type="password" autocomplete="off" placeholder="X-Admin-Key">
        <button id="saveKey">저장</button>
        <button id="clearKey" class="secondary">삭제</button>
      </div>
    </div>
  </header>
  <main>
    <aside>
      <div class="tools">
        <div class="search">
          <input id="keyword" type="search" placeholder="검색어">
          <button id="reload">조회</button>
        </div>
        <div id="status" class="status">관리자 키를 입력하고 조회.</div>
      </div>
      <div id="list" class="list"></div>
    </aside>
    <section class="viewer">
      <div class="viewer-head">
        <div id="title" class="viewer-title">비공개 보고서</div>
        <div id="meta" class="viewer-meta">목록에서 문서를 선택.</div>
      </div>
      <pre id="body"></pre>
    </section>
  </main>
  <script>
    const keyInput = document.getElementById('adminKey');
    const keywordInput = document.getElementById('keyword');
    const statusEl = document.getElementById('status');
    const listEl = document.getElementById('list');
    const titleEl = document.getElementById('title');
    const metaEl = document.getElementById('meta');
    const bodyEl = document.getElementById('body');
    let activeSlug = '';

    keyInput.value = localStorage.getItem('cyberLeninAdminKey') || '';

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function headers() {
      const key = keyInput.value.trim();
      return key ? { 'X-Admin-Key': key } : {};
    }

    function formatDate(value) {
      if (!value) return '?';
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return String(value);
      return date.toLocaleString('ko-KR', { hour12: false });
    }

    async function api(path) {
      const res = await fetch(path, { headers: headers() });
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const data = await res.json();
          detail = data.detail || detail;
        } catch (_) {}
        throw new Error(`${res.status} ${detail}`);
      }
      return res.json();
    }

    function renderList(reports) {
      listEl.textContent = '';
      if (!reports.length) {
        const empty = document.createElement('div');
        empty.className = 'empty';
        empty.textContent = '비공개 보고서가 없다.';
        listEl.appendChild(empty);
        return;
      }
      for (const report of reports) {
        const row = document.createElement('button');
        row.className = 'report-row';
        row.dataset.slug = report.slug || '';
        if (report.slug === activeSlug) row.classList.add('active');
        const rowTitle = document.createElement('div');
        rowTitle.className = 'row-title';
        rowTitle.textContent = report.title || '(untitled)';
        const meta = document.createElement('div');
        meta.className = 'row-meta';
        meta.textContent = `${report.slug} · ${formatDate(report.updated_at)}`;
        row.append(rowTitle, meta);
        row.addEventListener('click', () => loadDetail(report.slug));
        listEl.appendChild(row);
      }
    }

    async function loadList() {
      const key = keyInput.value.trim();
      if (!key) {
        setStatus('관리자 키가 필요하다.');
        return;
      }
      setStatus('조회 중...');
      const keyword = keywordInput.value.trim();
      const qs = new URLSearchParams({ limit: '100' });
      if (keyword) qs.set('keyword', keyword);
      try {
        const data = await api(`/private-reports?${qs.toString()}`);
        renderList(data.reports || []);
        setStatus(`보고서 ${(data.reports || []).length}개`);
      } catch (err) {
        setStatus(`실패: ${err.message}`);
      }
    }

    async function loadDetail(slug) {
      activeSlug = slug;
      setStatus('본문 조회 중...');
      try {
        const data = await api(`/private-reports/${encodeURIComponent(slug)}`);
        const report = data.report || {};
        titleEl.textContent = report.title || '(untitled)';
        metaEl.textContent = `${report.slug || ''} · id=${report.id || ''} · updated=${formatDate(report.updated_at)} · sha256=${report.content_sha256 || ''}`;
        bodyEl.textContent = report.markdown || '';
        document.querySelectorAll('.report-row').forEach((el) => {
          el.classList.toggle('active', el.dataset.slug === slug);
        });
        setStatus('본문 조회 완료.');
      } catch (err) {
        setStatus(`실패: ${err.message}`);
      }
    }

    document.getElementById('saveKey').addEventListener('click', () => {
      localStorage.setItem('cyberLeninAdminKey', keyInput.value.trim());
      setStatus('관리자 키 저장됨.');
      loadList();
    });
    document.getElementById('clearKey').addEventListener('click', () => {
      localStorage.removeItem('cyberLeninAdminKey');
      keyInput.value = '';
      setStatus('관리자 키 삭제됨.');
    });
    document.getElementById('reload').addEventListener('click', loadList);
    keywordInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') loadList();
    });
    if (keyInput.value.trim()) loadList();
  </script>
</body>
</html>"""


@app.get("/admin/private-reports")
@app.get("/private")
async def private_reports_admin_page():
    return Response(
        content=_PRIVATE_REPORTS_ADMIN_HTML,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow"},
    )


def _parse_user_fingerprints(http_req: Request) -> list[str]:
    """Read X-User-Fingerprints header (CSV) — injected by the frontend proxy
    for logged-in users so their bound fingerprints (across devices) can be
    queried as one. Empty when unauthenticated."""
    raw = http_req.headers.get("x-user-fingerprints", "")
    return [f.strip() for f in raw.split(",") if f.strip()]


@app.post("/chat")
async def chat(request: ChatRequest, http_req: Request):
    """
    클라이언트에게 실시간 로그와 답변을 스트리밍합니다.
    Uses claude_loop via web_chat module.
    """
    from web_chat import handle_web_chat

    user_agent = http_req.headers.get("user-agent", "")
    forwarded = http_req.headers.get("x-forwarded-for", "")
    ip_address = forwarded.split(",")[0].strip() if forwarded else (http_req.client.host if http_req.client else "")
    user_fingerprints = _parse_user_fingerprints(http_req)

    lock = _get_session_lock(request.session_id)

    async def event_generator():
        if lock.locked():
            print(f"⚠️ [요청 거부] session={request.session_id} — 이전 요청 처리 중", flush=True)
            yield format_sse({"type": "error", "content": "이전 질문에 대한 답변이 아직 처리 중입니다. 잠시 후 다시 시도해 주세요."})
            return

        async with lock:
            print(f"\n{'='*60}", flush=True)
            print(f"📩 [요청] session={request.session_id} fp={request.fingerprint[:8] or 'none'} | \"{request.message[:80]}\"", flush=True)

            async for sse_event in handle_web_chat(
                message=request.message,
                session_id=request.session_id,
                fingerprint=request.fingerprint,
                user_agent=user_agent,
                ip_address=ip_address,
                user_fingerprints=user_fingerprints,
            ):
                yield sse_event

            print(f"{'='*60}\n", flush=True)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/logs", dependencies=[Depends(require_admin)])
async def get_logs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """
    Fetch chat logs (admin view — all fields, ordered by most recent first).
    Requires X-Admin-Key header.
    """
    rows = db_query(
        "SELECT * FROM chat_logs ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (limit, offset),
    )
    return {"logs": rows, "count": len(rows)}


@app.get("/history")
async def get_history(
    http_req: Request,
    fingerprint: str | None = Query(default=None, description="Browser fingerprint (anonymous visitors)"),
    session_id: str | None = Query(default=None, description="Restrict to a single conversation session"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Fetch conversation history.
    - Anonymous: ?fingerprint=X — only that device's turns.
    - Logged-in: frontend proxy sets X-User-Fingerprints header — union of all bound devices.
      If `session_id` is given, only that session is returned.
    """
    fps = list({f for f in (_parse_user_fingerprints(http_req) + [fingerprint or ""]) if f})
    if not fps:
        return {"history": []}

    if session_id:
        rows = db_query(
            """SELECT user_query, bot_answer, created_at
               FROM chat_logs
               WHERE session_id = %s AND fingerprint = ANY(%s)
               ORDER BY created_at ASC
               LIMIT %s""",
            (session_id, fps, limit),
        )
    else:
        rows = db_query(
            """SELECT user_query, bot_answer, created_at
               FROM chat_logs
               WHERE fingerprint = ANY(%s)
               ORDER BY created_at ASC
               LIMIT %s""",
            (fps, limit),
        )
    return {"history": rows}


@app.get("/sessions")
async def list_sessions(
    http_req: Request,
    fingerprint: str | None = Query(default=None, description="Anonymous browser fingerprint"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    List distinct chat sessions (session_id groups) visible to the caller, with a
    preview of the first user message and timestamps. Ordered by most-recent activity.
    """
    fps = list({f for f in (_parse_user_fingerprints(http_req) + [fingerprint or ""]) if f})
    if not fps:
        return {"sessions": []}

    rows = db_query(
        """WITH scoped AS (
              SELECT id, session_id, fingerprint, user_query, created_at
                FROM chat_logs
               WHERE fingerprint = ANY(%s)
            ),
            agg AS (
              SELECT session_id,
                     MIN(created_at) AS first_at,
                     MAX(created_at) AS last_at,
                     COUNT(*)::int   AS message_count
                FROM scoped
               GROUP BY session_id
            ),
            first_msg AS (
              SELECT DISTINCT ON (session_id)
                     session_id, user_query AS first_query
                FROM scoped
               ORDER BY session_id, created_at ASC
            )
            SELECT agg.session_id, agg.first_at, agg.last_at, agg.message_count,
                   first_msg.first_query
              FROM agg
              JOIN first_msg USING (session_id)
             ORDER BY agg.last_at DESC
             LIMIT %s""",
        (fps, limit),
    )
    # Truncate previews for transport
    for r in rows:
        q = r.get("first_query") or ""
        r["first_query"] = q[:120]
    return {"sessions": rows}


@app.get("/reports", dependencies=[Depends(require_admin)])
async def list_reports(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Completed task reports list (admin-only)."""
    rows = db_query(
        """SELECT id, content, result, created_at, completed_at
           FROM telegram_tasks
           WHERE status = 'done' AND result IS NOT NULL AND result != ''
             AND COALESCE(agent_type, '') != 'programmer'
           ORDER BY completed_at DESC
           LIMIT %s OFFSET %s""",
        (limit, offset),
    )
    count_rows = db_query(
        "SELECT COUNT(*) AS cnt FROM telegram_tasks WHERE status = 'done' AND result IS NOT NULL AND result != '' AND COALESCE(agent_type, '') != 'programmer'",
    )
    total = count_rows[0]["cnt"] if count_rows else 0
    return {"reports": rows, "total": total}


@app.get("/reports/{report_id}", dependencies=[Depends(require_admin)])
async def get_report(report_id: int):
    """Single task report (admin-only, full markdown)."""
    rows = db_query(
        """SELECT id, content, result, created_at, completed_at
           FROM telegram_tasks
           WHERE id = %s AND status = 'done' AND result IS NOT NULL
             AND COALESCE(agent_type, '') != 'programmer'""",
        (report_id,),
    )
    if not rows:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "Report not found"})
    return {"report": rows[0]}


@app.get("/private-reports", dependencies=[Depends(require_admin)])
async def list_private_reports(
    limit: int = Query(default=20, ge=1, le=100),
    keyword: str | None = Query(default=None),
):
    """Admin-only private report list."""
    rows = await asyncio.to_thread(list_private_reports_sync, limit=limit, keyword=keyword)
    return {"reports": rows}


@app.get("/private-reports/{report_ref}", dependencies=[Depends(require_admin)])
async def get_private_report(report_ref: str):
    """Admin-only private report detail by id or slug."""
    if report_ref.isdigit():
        row = await asyncio.to_thread(get_private_report_sync, report_id=int(report_ref))
    else:
        row = await asyncio.to_thread(get_private_report_sync, slug=report_ref)
    if not row:
        return JSONResponse(status_code=404, content={"detail": "Private report not found"})
    return {"report": row}


@app.post("/private-reports", dependencies=[Depends(require_admin)])
async def save_private_report(request: PrivateReportRequest):
    """Create or update an admin-only private report."""
    try:
        row = await asyncio.to_thread(
            save_private_report_sync,
            title=request.title,
            slug=request.slug,
            markdown_body=request.markdown_body,
            source_task_id=request.source_task_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"report": row}


@app.post("/private-reports/{slug}/publish", dependencies=[Depends(require_admin)])
async def publish_private_report(slug: str, request: PublishPrivateReportRequest):
    """Publish a private report into public research_documents."""
    try:
        result = await asyncio.to_thread(
            publish_private_report_sync,
            slug=slug,
            body=request.body,
            title=request.title,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "private_report_id": result["private_report"]["id"],
        "research_document": result["research_document"],
        "is_overwrite": result["is_overwrite"],
        "public_url": result["public_url"],
    }


AGENT_CARD_DIR = Path(__file__).parent / "research"


@app.delete("/session/{session_id}", dependencies=[Depends(require_admin)])
async def clear_session(session_id: str):
    """Clear chat history for a session (deletes from chat_logs)."""
    result = await asyncio.to_thread(
        db_query,
        "DELETE FROM chat_logs WHERE session_id = %s RETURNING id",
        (session_id,),
    )
    return {"session_id": session_id, "cleared": len(result) if result else 0}


async def _serve_agent_card():
    """Serve the public A2A Agent Card for discovery."""
    filepath = AGENT_CARD_DIR / "cyber_lenin_a2a_agent_card.json"
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Agent card not found")
    return Response(content=filepath.read_text(encoding="utf-8"), media_type="application/json; charset=utf-8")


@app.get("/.well-known/agent-card.json")
async def a2a_agent_card_v1():
    """v1.0 canonical discovery endpoint."""
    return await _serve_agent_card()


@app.get("/.well-known/agent.json")
async def a2a_agent_card_legacy():
    """Legacy discovery endpoint (v0.2 compat)."""
    return await _serve_agent_card()


@app.post("/a2a")
async def a2a_endpoint(request: Request):
    """A2A JSON-RPC 2.0 endpoint (SendMessage)."""
    from a2a_handler import handle_a2a_message
    try:
        body = await request.json()
    except Exception:
        return Response(
            content=json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}),
            media_type="application/json",
            status_code=400,
        )
    result = await handle_a2a_message(body)
    status_code = 200 if "result" in result else 400
    return Response(content=json.dumps(result, ensure_ascii=False), media_type="application/json", status_code=status_code)


@app.post("/email/poll", dependencies=[Depends(require_admin)])
async def email_poll(limit: int = Query(default=10, ge=1, le=50)):
    result = await asyncio.to_thread(run_polling_cycle, limit)
    return result


@app.get("/email/inbound", dependencies=[Depends(require_admin)])
async def email_inbound(
    limit: int = Query(default=20, ge=1, le=100),
    route: str | None = Query(default=None),
    status: str | None = Query(default=None),
):
    rows = await asyncio.to_thread(list_inbound_messages, limit, route=route, status=status)
    return {"messages": rows, "count": len(rows)}


@app.get("/email/pending", dependencies=[Depends(require_admin)])
async def email_pending(limit: int = Query(default=20, ge=1, le=100)):
    return {"pending": await asyncio.to_thread(list_pending_email_approvals, limit)}


@app.get("/email/messages/{message_id}", dependencies=[Depends(require_admin)])
async def email_message_detail(message_id: int):
    row = await asyncio.to_thread(get_email_message, message_id)
    if not row:
        raise HTTPException(status_code=404, detail="Email message not found")
    prompt_input = None
    if row.get("direction") == "inbound":
        prompt_input = build_reply_prompt_input(row)
    return {"message": row, "reply_prompt_input": prompt_input}


@app.post("/email/drafts", dependencies=[Depends(require_admin)])
async def create_email_draft(request: EmailDraftRequest):
    draft = await asyncio.to_thread(
        queue_outbound_reply,
        request.inbound_message_id,
        request.draft_body,
        approver_user_id=request.approver_user_id,
        subject=request.subject,
        to_emails=request.to_emails,
        metadata=request.metadata,
    )
    return {"draft": draft}


@app.post("/email/messages/{message_id}/approval", dependencies=[Depends(require_admin)])
async def approve_email_message(message_id: int, request: EmailApprovalRequest):
    action = request.action.strip().lower()
    if action == "approve_send":
        result = await asyncio.to_thread(send_outbound_email, message_id, True, request.note)
        return {"result": result}
    if action == "save_draft":
        result = await asyncio.to_thread(send_outbound_email, message_id, False, request.note)
        return {"result": result}
    if action == "reject":
        await asyncio.to_thread(reject_outbound_email, message_id, request.note)
        return {"result": {"status": "rejected", "message_id": message_id}}
    raise HTTPException(status_code=400, detail="Unsupported action")


@app.post("/email/messages/{message_id}/internal-approve", dependencies=[Depends(require_admin)])
async def approve_inbound_email_for_internal_input(message_id: int, request: EmailInboundApprovalRequest):
    try:
        result = await asyncio.to_thread(
            mark_email_for_internal_delivery,
            message_id,
            approved_by=request.approved_by,
            note=request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result}


@app.get("/email/internal-approved", dependencies=[Depends(require_admin)])
async def email_internal_approved(limit: int = Query(default=20, ge=1, le=100)):
    rows = await asyncio.to_thread(list_messages_approved_for_internal_delivery, limit)
    return {"messages": rows, "count": len(rows)}


@app.post("/email/messages/{message_id}/internal-deliver", dependencies=[Depends(require_admin)])
async def deliver_inbound_email_to_internal(message_id: int, request: EmailInboundApprovalRequest):
    try:
        result = await asyncio.to_thread(
            deliver_inbound_email_to_internal_input,
            message_id,
            delivered_by=f"api:{request.approved_by or 'admin'}",
            note=request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result}


if __name__ == "__main__":
    print("🚩 사이버-레닌 API 서버 가동... (Port: 8000)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
