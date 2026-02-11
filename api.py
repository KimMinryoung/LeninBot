import asyncio
import json

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage

app = FastAPI(title="Cyber-Lenin API")

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    POST /chat — SSE 스트리밍 응답.
    LangGraph 전체 파이프라인(라우팅→검색→그레이딩→생성)을 실행한 뒤,
    최종 AI 응답을 토큰 단위로 SSE 스트리밍한다.
    """

    async def event_generator():
        inputs = {"messages": [HumanMessage(content=request.message)]}

        # LangGraph 워크플로우 전체 실행 (동기 함수를 스레드풀에서 실행)
        result = await asyncio.to_thread(get_graph().invoke, inputs)

        # 최종 AI 메시지 추출
        ai_message = result["messages"][-1]
        full_response = ai_message.content

        # 토큰 단위로 쪼개어 SSE 전송
        chunk_size = 4
        for i in range(0, len(full_response), chunk_size):
            token = full_response[i : i + chunk_size]
            yield {"data": json.dumps({"token": token}, ensure_ascii=False)}

        yield {
            "data": json.dumps(
                {"done": True, "full_response": full_response}, ensure_ascii=False
            )
        }

    return EventSourceResponse(event_generator())
