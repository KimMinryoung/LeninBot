"""Admin private report API routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api_security import require_admin
from runtime_tools.private_reports import (
    get_private_report_sync,
    list_private_reports_sync,
    publish_private_report_sync,
    save_private_report_sync,
)

router = APIRouter(dependencies=[Depends(require_admin)])


class PrivateReportRequest(BaseModel):
    title: str
    slug: str
    markdown_body: str
    source_task_id: int | None = None


class PublishPrivateReportRequest(BaseModel):
    body: str | None = None
    title: str | None = None


@router.get("/private-reports")
async def list_private_reports(
    limit: int = Query(default=20, ge=1, le=100),
    keyword: str | None = Query(default=None),
):
    """Admin-only private report list."""
    rows = await asyncio.to_thread(list_private_reports_sync, limit=limit, keyword=keyword)
    return {"reports": rows}


@router.get("/private-reports/{report_ref}")
async def get_private_report(report_ref: str):
    """Admin-only private report detail by id or slug."""
    if report_ref.isdigit():
        row = await asyncio.to_thread(get_private_report_sync, report_id=int(report_ref))
    else:
        row = await asyncio.to_thread(get_private_report_sync, slug=report_ref)
    if not row:
        return JSONResponse(status_code=404, content={"detail": "Private report not found"})
    return {"report": row}


@router.post("/private-reports")
async def save_private_report(request: PrivateReportRequest):
    """Create or update an admin-only private report."""
    try:
        row = await asyncio.to_thread(
            save_private_report_sync,
            title=request.title,
            slug=request.slug,
            markdown_body=request.markdown_body,
            source_task_id=request.source_task_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"report": row}


@router.post("/private-reports/{slug}/publish")
async def publish_private_report(slug: str, request: PublishPrivateReportRequest):
    """Publish a private report into public research_documents."""
    try:
        result = await asyncio.to_thread(
            publish_private_report_sync,
            slug=slug,
            body=request.body,
            title=request.title,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "private_report_id": result["private_report"]["id"],
        "research_document": result["research_document"],
        "is_overwrite": result["is_overwrite"],
        "public_url": result["public_url"],
    }
