import asyncio
import json
import os
from collections import defaultdict
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
#from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from db import query as db_query

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

# Lazy-load chatbot so uvicorn can bind the port first
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        from chatbot import graph
        _graph = graph
    return _graph


@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.api_route("/api/health", methods=["GET", "HEAD"])
async def api_health():
    return {"status": "ok"}


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

def format_sse(data: dict):
    """Server-Sent Events 포맷으로 변환"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest, http_req: Request):
    """
    클라이언트에게 실시간 로그와 답변을 스트리밍합니다.
    """
    g = get_graph()
    user_agent = http_req.headers.get("user-agent", "")
    # X-Forwarded-For is set by Render's proxy; fall back to direct client IP
    forwarded = http_req.headers.get("x-forwarded-for", "")
    ip_address = forwarded.split(",")[0].strip() if forwarded else (http_req.client.host if http_req.client else "")

    lock = _get_session_lock(request.session_id)

    async def event_generator():
        if lock.locked():
            print(f"⚠️ [요청 거부] session={request.session_id} — 이전 요청 처리 중", flush=True)
            yield format_sse({"type": "error", "content": "이전 질문에 대한 답변이 아직 처리 중입니다. 잠시 후 다시 시도해 주세요."})
            return

        async with lock:
            inputs = {"messages": [HumanMessage(content=request.message)]}
            config = {
                "configurable": {
                    "thread_id": request.session_id,
                    "fingerprint": request.fingerprint,
                    "user_agent": user_agent,
                    "ip_address": ip_address,
                }
            }
            pending_answer = None

            print(f"\n{'='*60}", flush=True)
            print(f"📩 [요청] session={request.session_id} fp={request.fingerprint[:8] or 'none'} | \"{request.message[:80]}\"", flush=True)

            try:
                async for output in g.astream(inputs, config=config, stream_mode="updates"):
                    for node_name, node_content in output.items():
                        if node_name == "log_conversation":
                            continue

                        # Stream logs via SSE + print to console
                        if "logs" in node_content:
                            for log_line in node_content["logs"]:
                                print(f"[{node_name}] {log_line}", flush=True)
                                yield format_sse({
                                    "type": "log",
                                    "node": node_name,
                                    "content": log_line
                                })

                        if node_name == "generate":
                            last_message = node_content["messages"][-1]
                            answer = last_message.content
                            print(f"[generate] 답변 생성 완료 ({len(answer)}자)", flush=True)
                            yield format_sse({
                                "type": "answer",
                                "content": answer
                            })
            except Exception as e:
                print(f"❌ [오류] 그래프 실행 중 예외 발생: {e}", flush=True)
                yield format_sse({
                    "type": "error",
                    "content": "서버에 일시적 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
                })

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
           ORDER BY completed_at DESC
           LIMIT %s OFFSET %s""",
        (limit, offset),
    )
    count_rows = db_query(
        "SELECT COUNT(*) AS cnt FROM telegram_tasks WHERE status = 'done' AND result IS NOT NULL AND result != ''",
    )
    total = count_rows[0]["cnt"] if count_rows else 0
    return {"reports": rows, "total": total}


@app.get("/reports/{report_id}")
async def get_report(report_id: int):
    """Single task report (full markdown)."""
    rows = db_query(
        """SELECT id, content, result, created_at, completed_at
           FROM telegram_tasks
           WHERE id = %s AND status = 'done' AND result IS NOT NULL""",
        (report_id,),
    )
    if not rows:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "Report not found"})
    return {"report": rows[0]}


@app.delete("/session/{session_id}", dependencies=[Depends(require_admin)])
async def clear_session(session_id: str):
    """특정 세션의 대화 기록(체크포인트)을 삭제합니다."""
    g = get_graph()
    exists = session_id in g.checkpointer.storage
    if exists:
        await g.checkpointer.adelete_thread(session_id)
    return {"session_id": session_id, "cleared": exists}


@app.delete("/sessions", dependencies=[Depends(require_admin)])
async def clear_all_sessions():
    """모든 세션의 대화 기록(체크포인트)을 삭제합니다."""
    g = get_graph()
    session_ids = list(g.checkpointer.storage.keys())
    for sid in session_ids:
        await g.checkpointer.adelete_thread(sid)
    return {"cleared_sessions": len(session_ids), "session_ids": session_ids}


if __name__ == "__main__":
    print("🚩 사이버-레닌 API 서버 가동... (Port: 8000)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
