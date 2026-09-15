"""Fixed localhost search/extract API. Keys, routing and budget live here."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tavily import AsyncTavilyClient

from content_fetch.url_security import UnsafeUrlError, validate_public_http_url
from tool_gateway.results import ToolFailure
from web_gateway import budget
from web_gateway.credentials import credential
from web_gateway.search import execute_web_search

@asynccontextmanager
async def lifespan(_app):
    budget.import_legacy_usage()
    yield


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Caller(StrictModel):
    service: str = Field(default="unknown", max_length=160)
    interface: str = Field(default="unknown", max_length=80)
    agent: str = Field(default="", max_length=160)
    task_id: str = Field(default="", max_length=160)
    scope_type: str = Field(default="", max_length=160)
    scope_id: str = Field(default="", max_length=160)
    request_id: str = Field(default="", max_length=160)


class SearchArguments(StrictModel):
    query: str = Field(min_length=1, max_length=1500)
    max_results: int = Field(default=5, ge=1, le=10)
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic"
    topic: Literal["general", "news", "finance"] = "general"
    time_range: Literal["day", "week", "month", "year"] | None = None
    use_cache: bool = True
    include_domains: list[str] | None = Field(default=None, max_length=10)
    exclude_domains: list[str] | None = Field(default=None, max_length=10)


class SearchRequest(StrictModel):
    arguments: SearchArguments
    caller: Caller = Field(default_factory=Caller)


class ExtractRequest(StrictModel):
    url: str = Field(min_length=1, max_length=4096)
    caller: Caller = Field(default_factory=Caller)


@app.middleware("http")
async def local_only(request: Request, call_next):
    if not request.client or request.client.host not in {"127.0.0.1", "::1"}:
        return JSONResponse({"error": "local access only"}, status_code=403)
    return await call_next(request)


async def read_request(request: Request, model):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 32768:
            raise HTTPException(413, "request too large")
    try:
        return model.model_validate(json.loads(body))
    except (ValueError, ValidationError):
        # Do not echo user-supplied secrets/URLs in validation errors.
        raise HTTPException(422, "invalid gateway request") from None


@app.get("/health")
def health():
    providers = [name for name, key in (("tavily", "TAVILY_API_KEY"), ("brave", "BRAVE_SEARCH_API_KEY")) if credential(key)]
    try:
        policy = budget.policy()
        # Probe the authoritative store at readiness, without a paid request.
        with budget._db(budget.STORE_PATH) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("SELECT COUNT(*) FROM web_usage").fetchone()
        ready = bool(providers) and all(policy[k] > 0 for k in ("tavily_credit_usd", "brave_search_usd"))
    except Exception:
        ready = False
    return JSONResponse({"status": "ok" if ready else "unavailable", "providers": providers}, status_code=200 if ready else 503)


@app.post("/search")
async def search(request: Request):
    payload = await read_request(request, SearchRequest)
    token = budget.usage_identity.set(payload.caller.model_dump())
    try:
        result = await execute_web_search(**payload.arguments.model_dump())
        return {"result": str(result), "error": isinstance(result, ToolFailure)}
    finally:
        budget.usage_identity.reset(token)


@app.post("/extract")
async def extract(request: Request):
    payload = await read_request(request, ExtractRequest)
    try:
        url = await asyncio.to_thread(validate_public_http_url, payload.url)
    except UnsafeUrlError:
        raise HTTPException(422, "unsafe or unresolved URL") from None
    key = credential("TAVILY_API_KEY")
    if not key:
        return {"error": True, "message": "Extraction provider unavailable"}
    token = budget.usage_identity.set(payload.caller.model_dump())
    try:
        with budget.paid_request("tavily", "extract", "basic") as charge:
            result = await AsyncTavilyClient(api_key=key).extract(urls=[url], extract_depth="basic", include_usage=True)
            charge.complete((result.get("usage") or {}).get("credits"))
        return {"results": [{"raw_content": str(item.get("raw_content") or item.get("content") or "")}
                            for item in result.get("results", []) if isinstance(item, dict)], "error": False}
    except budget.PaidWebBudgetError as exc:
        return {"error": True, "message": str(exc)}
    except Exception:
        return {"error": True, "message": "Extraction provider failed; usage reservation retained"}
    finally:
        budget.usage_identity.reset(token)


@app.get("/usage")
def usage(days: int = 7, by: Literal["service", "task"] = "service"):
    if not 1 <= days <= 366:
        raise HTTPException(422, "days must be 1..366")
    since = (datetime.now(timezone.utc).date() - timedelta(days=days - 1)).isoformat()
    rows = budget.report(since, by=by)
    return {"since_utc": since, "daily_budget_usd": budget.policy()["daily_budget_usd"] / 1_000_000,
            "accounted_usd": round(sum(row["accounted_usd"] for row in rows), 6),
            "note": "Local estimate, not invoice; unknown requests retain reservations.", "rows": rows}
