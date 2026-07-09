import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Query, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from api_routes.admin_users import router as admin_users_router
from api_routes.chat_history import router as chat_history_router
from api_routes.private_reports import router as private_reports_router
from api_routes.task_reports import router as task_reports_router
from api_routes.x402_demo import router as x402_demo_router
from api_security import (
    require_admin,
    is_admin_request as _is_admin_request,
    trusted_proxy_request as _trusted_proxy_request,
)
from db import query as db_query, query_one as db_query_one

_LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
logging.basicConfig(level=_LOG_LEVEL, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger().setLevel(_LOG_LEVEL)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── KG Eager Init ────────────────────────────────────────────────
    # Initialize Neo4j connection immediately at startup (background thread)
    # so the first real request doesn't pay the cold-start penalty.
    def _eager_init_kg():
        from kg_runtime.service_runtime import get_kg_service
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
    from kg_runtime.service_runtime import start_kg_healthcheck
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
app.include_router(admin_users_router)
app.include_router(chat_history_router)
app.include_router(private_reports_router)
app.include_router(task_reports_router)
app.include_router(x402_demo_router)


# Per-session locks to prevent concurrent requests from corrupting checkpointed state.
# Uses LRU-style eviction to prevent unbounded memory growth.
_session_locks: dict[str, asyncio.Lock] = {}
_SESSION_LOCKS_MAX = 200
_webchat_hits: defaultdict[str, deque[float]] = defaultdict(deque)
_webchat_active_count = 0
_WEBCHAT_RATE_LIMIT = max(1, int(os.getenv("WEBCHAT_RATE_LIMIT", "20") or "20"))
_WEBCHAT_RATE_WINDOW_SECONDS = max(10, int(os.getenv("WEBCHAT_RATE_WINDOW_SECONDS", "300") or "300"))
_WEBCHAT_GLOBAL_ACTIVE_LIMIT = max(1, int(os.getenv("WEBCHAT_GLOBAL_ACTIVE_LIMIT", "8") or "8"))
_DEFAULT_CORS_ORIGINS = "https://cyber-lenin.com,http://localhost:3000"

def _parse_cors_origins() -> list[str]:
    """Read comma-separated CORS origins from env, preserving safe defaults."""
    raw = os.getenv("WEBCHAT_CORS_ORIGINS") or os.getenv("CORS_ALLOW_ORIGINS") or _DEFAULT_CORS_ORIGINS
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or [item.strip() for item in _DEFAULT_CORS_ORIGINS.split(",")]


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
@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.api_route("/api/health", methods=["GET", "HEAD"])
async def api_health():
    return {"status": "ok"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    fingerprint: str = Field(default="", max_length=256)  # Browser fingerprint from localStorage (persistent across server restarts)
    persona: str = Field(default="cyber-lenin", max_length=64)  # Selected chat persona; unknown ids fall back to the default server-side
    regenerate_from_id: int | None = Field(default=None, gt=0)
    tone_feedback: str | None = Field(default=None, max_length=64)
    feedback_note: str | None = Field(default=None, max_length=500)


class ChatFeedbackRequest(BaseModel):
    message_id: int = Field(..., gt=0)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    fingerprint: str = Field(default="", max_length=256)
    persona: str = Field(default="cyber-lenin", max_length=64)
    rating: int | None = Field(default=None, ge=1, le=4)
    tone_feedback: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=500)


def format_sse(data: dict):
    """Server-Sent Events 포맷으로 변환"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"




def _parse_user_fingerprints(http_req: Request) -> list[str]:
    """Read X-User-Fingerprints header (CSV) — injected by the frontend proxy
    for logged-in users so their bound fingerprints (across devices) can be
    queried as one. Empty when unauthenticated."""
    if not _trusted_proxy_request(http_req):
        return []
    raw = http_req.headers.get("x-user-fingerprints", "")
    return [f.strip()[:256] for f in raw.split(",") if f.strip()][:20]


def _parse_authenticated_user_id(http_req: Request) -> int | None:
    """Read the account id stamped by the trusted frontend proxy."""
    if not _trusted_proxy_request(http_req):
        return None
    raw = (http_req.headers.get("x-authenticated-user-id") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _client_ip(http_req: Request) -> str:
    if _trusted_proxy_request(http_req):
        forwarded = http_req.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return http_req.client.host if http_req.client else ""


def _webchat_rate_key(request: ChatRequest, http_req: Request, ip_address: str) -> str:
    fingerprint = (request.fingerprint or "").strip()
    if fingerprint:
        return f"fp:{fingerprint[:128]}"
    return f"ip:{ip_address or 'unknown'}"


def _check_webchat_rate_limit(key: str) -> bool:
    now = time.monotonic()
    hits = _webchat_hits[key]
    cutoff = now - _WEBCHAT_RATE_WINDOW_SECONDS
    while hits and hits[0] < cutoff:
        hits.popleft()
    if len(hits) >= _WEBCHAT_RATE_LIMIT:
        return False
    hits.append(now)
    if len(_webchat_hits) > 5000:
        stale = [
            item_key for item_key, item_hits in _webchat_hits.items()
            if not item_hits or item_hits[-1] < cutoff
        ][:1000]
        for item_key in stale:
            _webchat_hits.pop(item_key, None)
    return True


@app.post("/chat")
async def chat(request: ChatRequest, http_req: Request):
    """
    클라이언트에게 실시간 로그와 답변을 스트리밍합니다.
    Uses claude_loop via web_chat module.
    """
    from web_chat import handle_web_chat
    from web_personas import get_persona

    user_agent = http_req.headers.get("user-agent", "")
    ip_address = _client_ip(http_req)
    user_fingerprints = _parse_user_fingerprints(http_req)
    authenticated_user_id = _parse_authenticated_user_id(http_req)
    rate_key = _webchat_rate_key(request, http_req, ip_address)

    # Admin-only personas (e.g. adult roleplay) require a valid X-Admin-Key.
    persona_spec = get_persona(request.persona)
    persona_blocked = persona_spec.admin_only and not _is_admin_request(http_req)

    lock = _get_session_lock(request.session_id)

    async def event_generator():
        global _webchat_active_count
        if persona_blocked:
            logger.warning(
                "web chat blocked admin-only persona=%s session=%s",
                request.persona, request.session_id,
            )
            yield format_sse({"type": "error", "content": "이 대화 상대는 관리자만 이용할 수 있습니다."})
            return
        if not _check_webchat_rate_limit(rate_key):
            logger.warning("web chat rate limited key=%s session=%s", rate_key[:24], request.session_id)
            yield format_sse({"type": "error", "content": "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."})
            return
        if _webchat_active_count >= _WEBCHAT_GLOBAL_ACTIVE_LIMIT:
            logger.warning("web chat global active limit reached session=%s", request.session_id)
            yield format_sse({"type": "error", "content": "현재 동시 요청이 많습니다. 잠시 후 다시 시도해 주세요."})
            return
        if lock.locked():
            logger.info("web chat rejected because session is locked session=%s", request.session_id)
            yield format_sse({"type": "error", "content": "이전 질문에 대한 답변이 아직 처리 중입니다. 잠시 후 다시 시도해 주세요."})
            return

        _webchat_active_count += 1
        try:
            async with lock:
                logger.info(
                    "web chat request session=%s fingerprint_prefix=%s chars=%d trusted_proxy=%s",
                    request.session_id,
                    (request.fingerprint or "")[:8] or "none",
                    len(request.message or ""),
                    _trusted_proxy_request(http_req),
                )

                async for sse_event in handle_web_chat(
                    message=request.message,
                    session_id=request.session_id,
                    fingerprint=request.fingerprint,
                    user_agent=user_agent,
                    ip_address=ip_address,
                    authenticated_user_id=authenticated_user_id,
                    user_fingerprints=user_fingerprints,
                    persona=request.persona,
                    regenerate_from_id=request.regenerate_from_id,
                    tone_feedback=request.tone_feedback or "",
                    feedback_note=request.feedback_note or "",
                ):
                    yield sse_event
        finally:
            _webchat_active_count = max(0, _webchat_active_count - 1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat/feedback")
async def save_chat_feedback(request: ChatFeedbackRequest, http_req: Request):
    """Store explicit feedback for one web-chat answer.

    The frontend should pass `message_id` from the final /chat SSE answer event.
    Stored feedback is folded into the next normal turn for the same
    persona/session once, then marked consumed. Regeneration feedback is applied
    only to that regeneration request.
    """
    from web_chat import (
        _FEEDBACK_TONE_LABELS,
        get_web_chat_log_for_feedback,
        normalize_web_chat_tone_feedback,
        save_web_chat_feedback,
    )

    tone_feedback = normalize_web_chat_tone_feedback(request.tone_feedback)
    if request.tone_feedback and not tone_feedback:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid tone_feedback", "allowed": sorted(_FEEDBACK_TONE_LABELS)},
        )
    note = (request.note or "").strip()
    if request.rating is None and not tone_feedback and not note:
        raise HTTPException(status_code=400, detail="rating, tone_feedback, or note is required")

    authenticated_user_id = _parse_authenticated_user_id(http_req)
    fps = list({f for f in (_parse_user_fingerprints(http_req) + [request.fingerprint or ""]) if f})
    if not authenticated_user_id and not fps:
        raise HTTPException(status_code=400, detail="fingerprint is required")

    row = await asyncio.to_thread(
        get_web_chat_log_for_feedback,
        request.message_id,
        fps,
        request.session_id,
        request.persona,
        account_user_id=authenticated_user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="chat message not found")

    feedback = await asyncio.to_thread(
        save_web_chat_feedback,
        chat_log_id=int(row["id"]),
        session_id=row["session_id"],
        fingerprint=request.fingerprint or row["fingerprint"],
        persona=row["persona"],
        rating=request.rating,
        tone_feedback=tone_feedback,
        note=note,
    )
    return {"feedback": feedback, "allowed_tone_feedback": sorted(_FEEDBACK_TONE_LABELS)}


@app.get("/personas")
async def list_chat_personas(http_req: Request):
    """Catalog of selectable chat personas for the frontend picker.

    Admin-only personas are included only when the request carries a valid
    X-Admin-Key header.
    """
    from web_personas import list_personas, DEFAULT_PERSONA_ID

    is_admin = _is_admin_request(http_req)
    return {"personas": list_personas(include_admin=is_admin), "default": DEFAULT_PERSONA_ID}


if __name__ == "__main__":
    print("🚩 사이버-레닌 API 서버 가동... (Port: 8000)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
