import asyncio
import json
import os
from collections import defaultdict
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
#from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from supabase.client import Client, create_client

load_dotenv()


# ── Diary scheduler (background task) ─────────────────────────
async def _diary_scheduler():
    """6 의 배수 시각 정각 (0, 6, 12, 18 시) 에 일기 자동 작성 (KST 기준)."""
    from datetime import datetime, timedelta, timezone

    KST = timezone(timedelta(hours=9))

    while True:
        now = datetime.now(KST)
        PERIODHOUR = 6
        # 다음 6 의 배수 정각 계산
        current_hour = now.hour
        next_hour = current_hour + (PERIODHOUR - current_hour % PERIODHOUR) if current_hour % PERIODHOUR != 0 else current_hour + PERIODHOUR
        next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=(next_hour - current_hour))
        wait_seconds = (next_run - now).total_seconds()
        print(f"📝 [일기 스케줄러] 다음 실행: {next_run.strftime('%H:%M')} (KST, {int(wait_seconds)}초 후)")

        await asyncio.sleep(wait_seconds)
        try:
            from diary_writer import write_diary
            await asyncio.to_thread(write_diary)
        except Exception as e:
            print(f"⚠️ [일기 스케줄러] 오류: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_diary_scheduler())
    yield
    task.cancel()


app = FastAPI(title="Cyber-Lenin API", lifespan=lifespan)

# Lightweight Supabase client for /logs (does NOT import chatbot module)
_supabase_light: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY"),
)

# Per-session locks to prevent concurrent requests from corrupting checkpointed state
_session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

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
    ip_address = forwarded.split(",")[0].strip() if forwarded else (http_req.client.host or "")

    lock = _session_locks[request.session_id]

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
                        pending_answer = last_message.content
                        print(f"[generate] 답변 생성 완료 ({len(pending_answer)}자)", flush=True)

                    if node_name == "critic" and pending_answer is not None:
                        feedback = node_content.get("feedback")
                        if not feedback:
                            print(f"[critic] ✅ 답변 승인 — 클라이언트로 전송", flush=True)
                            yield format_sse({
                                "type": "answer",
                                "content": pending_answer
                            })
                            pending_answer = None

            print(f"{'='*60}\n", flush=True)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/logs")
async def get_logs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """
    Fetch chat logs from Supabase (admin view — all fields, ordered by most recent first).
    """
    sb = _supabase_light
    result = (
        sb.table("chat_logs")
        .select("*")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return {"logs": result.data, "count": len(result.data)}


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
    result = (
        _supabase_light.table("chat_logs")
        .select("user_query, bot_answer, created_at")
        .eq("fingerprint", fingerprint)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return {"history": result.data}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """특정 세션의 대화 기록(체크포인트)을 삭제합니다."""
    g = get_graph()
    exists = session_id in g.checkpointer.storage
    if exists:
        await g.checkpointer.adelete_thread(session_id)
    return {"session_id": session_id, "cleared": exists}


@app.delete("/sessions")
async def clear_all_sessions():
    """모든 세션의 대화 기록(체크포인트)을 삭제합니다."""
    g = get_graph()
    session_ids = list(g.checkpointer.storage.keys())
    for sid in session_ids:
        await g.checkpointer.adelete_thread(sid)
    return {"cleared_sessions": len(session_ids), "session_ids": session_ids}


if __name__ == "__main__":
    print("🚩 사이버-레닌 API 서버 가동... (Port: 8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
