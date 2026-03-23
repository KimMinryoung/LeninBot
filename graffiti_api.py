"""graffiti_api.py — Graffiti system: local LLM leaves creative marks on the server.

Three types:
- dreams/  — surreal diary entries
- debates/ — LLM argues, Gemini rebuts, LLM counter-rebuts
- riddles/ — unanswerable riddles

Auth: X-Graffiti-Key header (GRAFFITI_API_KEY env var)
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# ── Auth ──────────────────────────────────────────────────────────────
_GRAFFITI_API_KEY = os.getenv("GRAFFITI_API_KEY", "")
_graffiti_key_header = APIKeyHeader(name="X-Graffiti-Key", auto_error=False)


async def require_graffiti_key(api_key: str = Security(_graffiti_key_header)):
    if not _GRAFFITI_API_KEY:
        raise HTTPException(status_code=503, detail="Graffiti API key not configured")
    if not api_key or api_key != _GRAFFITI_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid graffiti key")


router = APIRouter(prefix="/graffiti", tags=["graffiti"])

# ── Storage ───────────────────────────────────────────────────────────
GRAFFITI_DIR = Path(__file__).parent / "graffiti"
DREAMS_DIR = GRAFFITI_DIR / "dreams"
DEBATES_DIR = GRAFFITI_DIR / "debates"
RIDDLES_DIR = GRAFFITI_DIR / "riddles"

for d in (DREAMS_DIR, DEBATES_DIR, RIDDLES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")


def _save_json(directory: Path, data: dict) -> str:
    filename = f"{_timestamp()}.json"
    filepath = directory / filename
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Graffiti saved: %s", filepath)
    return filename


# ── Models ────────────────────────────────────────────────────────────
class DreamRequest(BaseModel):
    title: str
    content: str
    model: str = "unknown"


class DebateRespondRequest(BaseModel):
    topic: str
    local_claim: str


class DebateFinishRequest(BaseModel):
    topic: str
    local_claim: str
    server_rebuttal: str
    local_counter: str
    model: str = "unknown"


class RiddleRequest(BaseModel):
    riddle: str
    model: str = "unknown"


# ── Endpoints ─────────────────────────────────────────────────────────
@router.post("/dream")
async def save_dream(req: DreamRequest, _=Security(require_graffiti_key)):
    filename = _save_json(DREAMS_DIR, {
        "type": "dream",
        "title": req.title,
        "content": req.content,
        "model": req.model,
        "created_at": datetime.now(KST).isoformat(),
    })
    return {"status": "ok", "filename": filename}


@router.post("/debate/respond")
async def debate_respond(req: DebateRespondRequest, _=Security(require_graffiti_key)):
    """Generate a rebuttal using Gemini against the local LLM's claim."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from shared import MODEL_MAIN

    llm = ChatGoogleGenerativeAI(model=MODEL_MAIN, temperature=0.7)
    prompt = (
        f"주제: {req.topic}\n\n"
        f"상대방 주장: {req.local_claim}\n\n"
        "당신은 이 주장에 반대하는 입장입니다. "
        "날카롭고 논리적으로 반박하세요. 2-3문장으로 간결하게."
    )
    resp = llm.invoke(prompt)
    from shared import extract_text_content
    rebuttal = extract_text_content(resp.content).strip()
    return {"rebuttal": rebuttal}


@router.post("/debate/finish")
async def debate_finish(req: DebateFinishRequest, _=Security(require_graffiti_key)):
    filename = _save_json(DEBATES_DIR, {
        "type": "debate",
        "topic": req.topic,
        "rounds": [
            {"role": "local_llm", "content": req.local_claim},
            {"role": "server", "content": req.server_rebuttal},
            {"role": "local_llm", "content": req.local_counter},
        ],
        "model": req.model,
        "created_at": datetime.now(KST).isoformat(),
    })
    return {"status": "ok", "filename": filename}


@router.post("/riddle")
async def save_riddle(req: RiddleRequest, _=Security(require_graffiti_key)):
    filename = _save_json(RIDDLES_DIR, {
        "type": "riddle",
        "riddle": req.riddle,
        "model": req.model,
        "created_at": datetime.now(KST).isoformat(),
    })
    return {"status": "ok", "filename": filename}


@router.get("/list")
async def list_graffiti(type: str = "all", limit: int = 10):
    """List recent graffiti. type: all, dream, debate, riddle."""
    dirs = {
        "dream": DREAMS_DIR,
        "debate": DEBATES_DIR,
        "riddle": RIDDLES_DIR,
    }
    if type == "all":
        scan = list(dirs.values())
    elif type in dirs:
        scan = [dirs[type]]
    else:
        raise HTTPException(400, f"Invalid type: {type}")

    items = []
    for d in scan:
        for f in sorted(d.glob("*.json"), reverse=True)[:limit]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                data["_filename"] = f.name
                items.append(data)
            except Exception:
                continue

    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"items": items[:limit]}
