import asyncio
import json
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, Response
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

load_dotenv()

# ── Admin API key authentication ──────────────────────────────────
_ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_admin(api_key: str = Security(_admin_key_header)):
    """Dependency that enforces admin API key for sensitive endpoints."""
    if not _ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin API key not configured on server")
    if not api_key or api_key != _ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing admin API key")


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
        from telegram_bot import bot_main
        bot_task = asyncio.create_task(bot_main())
    yield
    if bot_task is not None:
        bot_task.cancel()



app = FastAPI(title="Cyber-Lenin API", lifespan=lifespan)

# ── Graffiti 라우터 등록 ───────────────────────────────────────────────
from graffiti_api import router as graffiti_router
app.include_router(graffiti_router)


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
    fingerprint: str = Query(..., description="Browser fingerprint stored in localStorage"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Fetch conversation history for an end-user identified by browser fingerprint.
    Returns only user_query, bot_answer, created_at — no processing logs or internal fields.
    Persistent across server restarts (fingerprint is device-based, not session-based).
    """
    rows = db_query(
        """SELECT user_query, bot_answer, created_at
           FROM chat_logs
           WHERE fingerprint = %s
           ORDER BY created_at ASC
           LIMIT %s""",
        (fingerprint, limit),
    )
    return {"history": rows}


@app.get("/reports")
async def list_reports(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Completed task reports list (public, for BichonWebsite)."""
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


@app.get("/reports/{report_id}")
async def get_report(report_id: int):
    """Single task report (full markdown)."""
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


RESEARCH_DIR = Path(__file__).parent / "research"
LEGACY_OUTPUT_RESEARCH_DIR = Path(__file__).parent / "output" / "research"
PUBLIC_BASE_URL = "https://cyber-lenin.com"

SEO_DEFAULT_TITLE = "Cyber-Lenin"
SEO_DEFAULT_DESCRIPTION = "Cyber-Lenin: Marxist-Leninist analysis, geopolitical research, and autonomous intelligence reports."
SEO_DEFAULT_OG_IMAGE = f"{PUBLIC_BASE_URL}/static/og/cyber-lenin-og.png"
SEO_DEFAULT_KEYWORDS = (
    "cyber-lenin, marxist-leninist analysis, geopolitics, historical materialism, "
    "technology democracy, open weights, ai sovereignty"
)


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _html_escape(value: str) -> str:
    return _xml_escape(value)


def _build_meta_head(*, title: str, description: str, canonical_url: str, og_type: str = "website", image_url: str = SEO_DEFAULT_OG_IMAGE, noindex: bool = False) -> str:
    robots = "noindex, nofollow" if noindex else "index, follow, max-image-preview:large"
    schema_json = json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "WebSite" if og_type == "website" else "Article",
            "name": title,
            "headline": title,
            "description": description,
            "url": canonical_url,
            "image": image_url,
            "publisher": {
                "@type": "Organization",
                "name": "Cyber-Lenin",
                "url": PUBLIC_BASE_URL,
            },
        },
        ensure_ascii=False,
    )
    return f"""
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>{_html_escape(title)}</title>
    <meta name=\"description\" content=\"{_html_escape(description)}\">
    <meta name=\"keywords\" content=\"{_html_escape(SEO_DEFAULT_KEYWORDS)}\">
    <meta name=\"robots\" content=\"{robots}\">
    <link rel=\"canonical\" href=\"{_html_escape(canonical_url)}\">
    <link rel=\"alternate\" type=\"application/atom+xml\" title=\"Cyber-Lenin Atom Feed\" href=\"{PUBLIC_BASE_URL}/atom.xml\">
    <meta property=\"og:site_name\" content=\"Cyber-Lenin\">
    <meta property=\"og:title\" content=\"{_html_escape(title)}\">
    <meta property=\"og:description\" content=\"{_html_escape(description)}\">
    <meta property=\"og:type\" content=\"{_html_escape(og_type)}\">
    <meta property=\"og:url\" content=\"{_html_escape(canonical_url)}\">
    <meta property=\"og:image\" content=\"{_html_escape(image_url)}\">
    <meta name=\"twitter:card\" content=\"summary_large_image\">
    <meta name=\"twitter:title\" content=\"{_html_escape(title)}\">
    <meta name=\"twitter:description\" content=\"{_html_escape(description)}\">
    <meta name=\"twitter:image\" content=\"{_html_escape(image_url)}\">
    <script type=\"application/ld+json\">{schema_json}</script>
    """.strip()


def _render_html_page(*, title: str, description: str, canonical_url: str, body_html: str, og_type: str = "website", noindex: bool = False) -> str:
    head = _build_meta_head(
        title=title,
        description=description,
        canonical_url=canonical_url,
        og_type=og_type,
        noindex=noindex,
    )
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
{head}
</head>
<body>
{body_html}
</body>
</html>
"""


def _build_research_url(filename: str) -> str:
    return f"{PUBLIC_BASE_URL}/research/{filename}"


def _resolve_research_file(filename: str) -> Path | None:
    """Resolve a public research markdown file, preferring research/ over legacy output/research/."""
    primary = RESEARCH_DIR / filename
    if primary.is_file():
        return primary
    legacy = LEGACY_OUTPUT_RESEARCH_DIR / filename
    if legacy.is_file():
        return legacy
    return None


@app.get("/robots.txt")
async def robots_txt():
    content = "\n".join([
        "User-agent: *",
        "Allow: /",
        "",
        f"Sitemap: {PUBLIC_BASE_URL}/sitemap.xml",
        f"Host: {PUBLIC_BASE_URL.replace('https://', '')}",
    ])
    return Response(content=content + "\n", media_type="text/plain; charset=utf-8")


@app.get("/sitemap.xml")
async def sitemap_xml():
    static_urls = [
        (f"{PUBLIC_BASE_URL}/", "daily", "1.0"),
        (f"{PUBLIC_BASE_URL}/research", "hourly", "0.9"),
        (f"{PUBLIC_BASE_URL}/atom.xml", "hourly", "0.6"),
    ]
    entries: list[str] = []
    for loc, changefreq, priority in static_urls:
        entries.append(
            "<url>"
            f"<loc>{_xml_escape(loc)}</loc>"
            f"<changefreq>{changefreq}</changefreq>"
            f"<priority>{priority}</priority>"
            "</url>"
        )

    files_by_name: dict[str, Path] = {}
    for directory in (LEGACY_OUTPUT_RESEARCH_DIR, RESEARCH_DIR):
        if not directory.is_dir():
            continue
        for p in directory.glob("*.md"):
            files_by_name[p.name] = p

    for name in sorted(files_by_name):
        p = files_by_name[name]
        lastmod = p.stat().st_mtime
        from datetime import datetime, timezone
        lastmod_iso = datetime.fromtimestamp(lastmod, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entries.append(
            "<url>"
            f"<loc>{_xml_escape(_build_research_url(name))}</loc>"
            f"<lastmod>{lastmod_iso}</lastmod>"
            "<changefreq>weekly</changefreq>"
            "<priority>0.7</priority>"
            "</url>"
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + "".join(entries)
        + "</urlset>"
    )
    return Response(content=xml, media_type="application/xml; charset=utf-8")


@app.get("/atom.xml")
async def atom_feed():
    files_by_name: dict[str, Path] = {}
    for directory in (LEGACY_OUTPUT_RESEARCH_DIR, RESEARCH_DIR):
        if not directory.is_dir():
            continue
        for p in directory.glob("*.md"):
            files_by_name[p.name] = p

    sorted_files = sorted(files_by_name.values(), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
    updated = "1970-01-01T00:00:00Z"
    entries: list[str] = []
    if sorted_files:
        from datetime import datetime, timezone
        updated = datetime.fromtimestamp(sorted_files[0].stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for p in sorted_files:
            modified = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            title = p.stem.replace("_", " ")
            url = _build_research_url(p.name)
            entries.append(
                "<entry>"
                f"<title>{_xml_escape(title)}</title>"
                f"<link href=\"{_xml_escape(url)}\" />"
                f"<id>{_xml_escape(url)}</id>"
                f"<updated>{modified}</updated>"
                "</entry>"
            )

    feed = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"<title>{_xml_escape(SEO_DEFAULT_TITLE)}</title>"
        f"<subtitle>{_xml_escape(SEO_DEFAULT_DESCRIPTION)}</subtitle>"
        f"<link href=\"{PUBLIC_BASE_URL}/atom.xml\" rel=\"self\" />"
        f"<link href=\"{PUBLIC_BASE_URL}/\" />"
        f"<id>{PUBLIC_BASE_URL}/atom.xml</id>"
        f"<updated>{updated}</updated>"
        + "".join(entries)
        + "</feed>"
    )
    return Response(content=feed, media_type="application/atom+xml; charset=utf-8")


@app.get("/research")
async def list_research():
    """List public .md files, preferring research/ and backfilling from legacy output/research/."""
    files_by_name: dict[str, dict] = {}
    for directory in (LEGACY_OUTPUT_RESEARCH_DIR, RESEARCH_DIR):
        if not directory.is_dir():
            continue
        for p in directory.glob("*.md"):
            stat = p.stat()
            files_by_name[p.name] = {
                "filename": p.name,
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
                "source_dir": p.parent.name,
                "url": _build_research_url(p.name),
            }
    files = [files_by_name[name] for name in sorted(files_by_name)]
    return {"files": files}


@app.get("/research/{filename}")
async def get_research(filename: str, format: str = Query(default="json")):
    """Read a single public markdown file from research/ or legacy output/research/."""
    if "/" in filename or "\\" in filename or ".." in filename:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": "Invalid filename"})
    filepath = _resolve_research_file(filename)
    if filepath is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "File not found"})
    content = filepath.read_text(encoding="utf-8")
    if format.lower() == "html":
        title = f"{filepath.stem.replace('_', ' ')} | Cyber-Lenin Research"
        description = content.splitlines()[0].lstrip("# ").strip() if content.strip() else SEO_DEFAULT_DESCRIPTION
        body_html = (
            f"<main><article><h1>{_html_escape(title)}</h1>"
            f"<pre>{_html_escape(content)}</pre>"
            f"</article></main>"
        )
        html = _render_html_page(
            title=title,
            description=description[:300],
            canonical_url=_build_research_url(filename),
            body_html=body_html,
            og_type="article",
        )
        return Response(content=html, media_type="text/html; charset=utf-8")
    return {"filename": filename, "content": content, "size": len(content), "source_dir": filepath.parent.name, "url": _build_research_url(filename)}


@app.get("/reports/research")
async def redirect_research_index():
    """Backward-compatible redirect from legacy guessed path to the actual public research index."""
    return RedirectResponse(url="/research", status_code=307)


@app.get("/reports/research/{filename}")
async def redirect_research_file(filename: str):
    """Backward-compatible redirect from legacy guessed path to the actual public research file URL."""
    return RedirectResponse(url=f"/research/{filename}", status_code=307)


@app.delete("/session/{session_id}", dependencies=[Depends(require_admin)])
async def clear_session(session_id: str):
    """Clear chat history for a session (deletes from chat_logs)."""
    result = await asyncio.to_thread(
        db_query,
        "DELETE FROM chat_logs WHERE session_id = %s RETURNING id",
        (session_id,),
    )
    return {"session_id": session_id, "cleared": len(result) if result else 0}


@app.get("/.well-known/agent.json")
async def a2a_agent_card():
    """Serve the public A2A example Agent Card for discovery."""
    filepath = RESEARCH_DIR / "cyber_lenin_a2a_agent_card.json"
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Agent card not found")
    return Response(content=filepath.read_text(encoding="utf-8"), media_type="application/json; charset=utf-8")


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
