import asyncio
import json
import os

import uvicorn
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
#from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage

app = FastAPI(title="Cyber-Lenin API")

# Lazy-load chatbot so uvicorn can bind the port first
_graph = None
_supabase = None


def get_graph():
    global _graph
    if _graph is None:
        from chatbot import graph
        _graph = graph
    return _graph


def get_supabase():
    global _supabase
    if _supabase is None:
        from chatbot import supabase
        _supabase = supabase
    return _supabase


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

def format_sse(data: dict):
    """Server-Sent Events 포맷으로 변환"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    클라이언트에게 실시간 로그와 답변을 스트리밍합니다.
    """
    _graph = get_graph()

    async def event_generator():
        inputs = {"messages": [HumanMessage(content=request.message)]}

        # 그래프 실행 및 로그 스트리밍 (stream_mode="updates")
        # 각 노드가 끝날 때마다 그 노드의 출력값(logs 등)을 받아옵니다.
        async for output in _graph.astream(inputs, stream_mode="updates"):
            for node_name, node_content in output.items():
                # log_conversation 노드는 내부 전용이므로 클라이언트에 노출하지 않음
                if node_name == "log_conversation":
                    continue

                # 로그가 있다면 클라이언트로 전송
                if "logs" in node_content:
                    for log_line in node_content["logs"]:
                        yield format_sse({
                            "type": "log",
                            "node": node_name,
                            "content": log_line
                        })
                
                # 최종 답변 생성 단계라면 답변 내용 전송
                if node_name == "generate":
                    last_message = node_content["messages"][-1]
                    yield format_sse({
                        "type": "answer",
                        "content": last_message.content
                    })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/logs")
async def get_logs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """
    Fetch chat logs from Supabase, ordered by most recent first.
    """
    sb = get_supabase()
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