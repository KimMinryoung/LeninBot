"""Admin API for archival document translation.

Runs a translation spec from ``config/archival_translation/`` without the
sudo dance the CLI needs: this service already mounts ``deepseek_api_key``
from the credstore.

The run endpoint takes a spec **id**, never a path or URL. The spec is the
thing that pins which block ranges of which source get translated, so
requiring one keeps every run reproducible, cache-keyed and attributable —
there is no "point it at a page and translate whatever is there" surface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api_security import require_admin
from runtime_tools import archival_translation as at

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_admin)])

# One run at a time: concurrent runs of the same spec would interleave
# appends to the same JSONL cache.
_run_lock = threading.Lock()


class RunRequest(BaseModel):
    specId: str
    limitChunks: int = Field(default=0, ge=0, le=1000)
    concurrency: int = Field(default=5, ge=1, le=16)
    retries: int = Field(default=3, ge=1, le=5)
    model: str | None = None


def _options(req: RunRequest) -> at.Options:
    opts = at.Options(
        limit_chunks=req.limitChunks,
        concurrency=req.concurrency,
        retries=req.retries,
    )
    if req.model:
        opts.model = req.model
    return opts


@router.get("/admin/archival-translation/specs")
async def list_specs():
    """Registered specs, with output and cache state for each."""
    return {"specs": await asyncio.to_thread(at.list_specs)}


@router.post("/admin/archival-translation/plan")
async def plan_translation(req: RunRequest):
    """Slice, chunk and price a run without calling the model."""
    try:
        spec = await asyncio.to_thread(at.load_spec, req.specId)
        prepared = await asyncio.to_thread(at.plan, spec, _options(req))
    except at.SpecError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {k: v for k, v in prepared.items() if not k.startswith("_")}


@router.post("/admin/archival-translation/run")
async def run_translation(req: RunRequest):
    """Translate a spec, streaming progress as NDJSON.

    Each line is one event: plan, retry, chunk, done, or error. The response
    starts before the work finishes, so a multi-minute run stays observable
    instead of sitting on an open socket with nothing to show.
    """
    opts_preview = _options(req)
    try:
        spec = await asyncio.to_thread(at.load_spec, req.specId)
        # Check the credential before opening a stream, so a bad key is a 400
        # here instead of a stream of identical chunk failures.
        await asyncio.to_thread(at.preflight, opts_preview)
    except at.SpecError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not _run_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="another translation run is in progress")

    opts = opts_preview
    events: queue.Queue = queue.Queue()
    done = object()

    def worker() -> None:
        try:
            at.run(spec, opts, progress=events.put)
        except at.SpecError as e:
            events.put({"event": "error", "kind": "spec", "message": str(e)})
        except Exception as e:  # a failed chunk should reach the client, not just the log
            logger.exception("[archival-translation] %s failed", req.specId)
            events.put({"event": "error", "kind": "run", "message": str(e)})
        finally:
            events.put(done)

    async def stream():
        try:
            task = asyncio.get_running_loop().run_in_executor(None, worker)
            while True:
                event = await asyncio.to_thread(events.get)
                if event is done:
                    break
                yield json.dumps(event, ensure_ascii=False) + "\n"
            await task
        finally:
            _run_lock.release()

    return StreamingResponse(stream(), media_type="application/x-ndjson")
