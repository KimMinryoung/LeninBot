import json
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
#from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from supabase.client import Client, create_client

load_dotenv()

app = FastAPI(title="Cyber-Lenin API")

# Lightweight Supabase client for /logs (does NOT import chatbot module)
_supabase_light: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY"),
)

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

def format_sse(data: dict):
    """Server-Sent Events 포맷으로 변환"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    클라이언트에게 실시간 로그와 답변을 스트리밍합니다.
    """
    g = get_graph()

    async def event_generator():
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": request.session_id}}
        pending_answer = None

        print(f"\n{'='*60}", flush=True)
        print(f"📩 [요청] session={request.session_id} | \"{request.message[:80]}\"", flush=True)

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
    Fetch chat logs from Supabase, ordered by most recent first.
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


if __name__ == "__main__":
    print("🚩 사이버-레닌 API 서버 가동... (Port: 8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)