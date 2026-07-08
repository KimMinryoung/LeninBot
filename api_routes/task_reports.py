"""Admin task report API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from api_security import require_admin
from db import query as db_query

router = APIRouter(dependencies=[Depends(require_admin)])


@router.get("/reports")
async def list_reports(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Completed task reports list (admin-only)."""
    rows = db_query(
        """SELECT id, content, result, created_at, completed_at
           FROM telegram_tasks
           WHERE status = 'done' AND result IS NOT NULL AND result != ''
             AND COALESCE(agent_type, '') != 'programmer'
           ORDER BY completed_at DESC
           LIMIT %s OFFSET %s""",
        (limit, offset),
    )
    count_rows = db_query(
        "SELECT COUNT(*) AS cnt FROM telegram_tasks WHERE status = 'done' AND result IS NOT NULL AND result != '' AND COALESCE(agent_type, '') != 'programmer'",
    )
    total = count_rows[0]["cnt"] if count_rows else 0
    return {"reports": rows, "total": total}


@router.get("/reports/{report_id}")
async def get_report(report_id: int):
    """Single task report (admin-only, full markdown)."""
    rows = db_query(
        """SELECT id, content, result, created_at, completed_at
           FROM telegram_tasks
           WHERE id = %s AND status = 'done' AND result IS NOT NULL
             AND COALESCE(agent_type, '') != 'programmer'""",
        (report_id,),
    )
    if not rows:
        return JSONResponse(status_code=404, content={"detail": "Report not found"})
    return {"report": rows[0]}
