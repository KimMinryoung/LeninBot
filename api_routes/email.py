"""Admin email bridge API routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api_security import require_admin
from email_bridge import (
    build_reply_prompt_input,
    deliver_inbound_email_to_internal_input,
    get_email_message,
    list_inbound_messages,
    list_messages_approved_for_internal_delivery,
    list_pending_email_approvals,
    mark_email_for_internal_delivery,
    queue_outbound_reply,
    reject_outbound_email,
    run_polling_cycle,
    send_outbound_email,
)

router = APIRouter(dependencies=[Depends(require_admin)])


class EmailDraftRequest(BaseModel):
    inbound_message_id: int
    draft_body: str
    subject: str | None = None
    to_emails: list[str] | None = None
    approver_user_id: int | None = None
    metadata: dict | None = None


class EmailApprovalRequest(BaseModel):
    action: str  # approve_send | save_draft | reject
    note: str = ""


class EmailInboundApprovalRequest(BaseModel):
    note: str = ""
    approved_by: int | None = None


@router.post("/email/poll")
async def email_poll(limit: int = Query(default=10, ge=1, le=50)):
    result = await asyncio.to_thread(run_polling_cycle, limit)
    return result


@router.get("/email/inbound")
async def email_inbound(
    limit: int = Query(default=20, ge=1, le=100),
    route: str | None = Query(default=None),
    status: str | None = Query(default=None),
):
    rows = await asyncio.to_thread(list_inbound_messages, limit, route=route, status=status)
    return {"messages": rows, "count": len(rows)}


@router.get("/email/pending")
async def email_pending(limit: int = Query(default=20, ge=1, le=100)):
    return {"pending": await asyncio.to_thread(list_pending_email_approvals, limit)}


@router.get("/email/messages/{message_id}")
async def email_message_detail(message_id: int):
    row = await asyncio.to_thread(get_email_message, message_id)
    if not row:
        raise HTTPException(status_code=404, detail="Email message not found")
    prompt_input = None
    if row.get("direction") == "inbound":
        prompt_input = build_reply_prompt_input(row)
    return {"message": row, "reply_prompt_input": prompt_input}


@router.post("/email/drafts")
async def create_email_draft(request: EmailDraftRequest):
    draft = await asyncio.to_thread(
        queue_outbound_reply,
        request.inbound_message_id,
        request.draft_body,
        approver_user_id=request.approver_user_id,
        subject=request.subject,
        to_emails=request.to_emails,
        metadata=request.metadata,
    )
    return {"draft": draft}


@router.post("/email/messages/{message_id}/approval")
async def approve_email_message(message_id: int, request: EmailApprovalRequest):
    action = request.action.strip().lower()
    if action == "approve_send":
        result = await asyncio.to_thread(send_outbound_email, message_id, True, request.note)
        return {"result": result}
    if action == "save_draft":
        result = await asyncio.to_thread(send_outbound_email, message_id, False, request.note)
        return {"result": result}
    if action == "reject":
        await asyncio.to_thread(reject_outbound_email, message_id, request.note)
        return {"result": {"status": "rejected", "message_id": message_id}}
    raise HTTPException(status_code=400, detail="Unsupported action")


@router.post("/email/messages/{message_id}/internal-approve")
async def approve_inbound_email_for_internal_input(message_id: int, request: EmailInboundApprovalRequest):
    try:
        result = await asyncio.to_thread(
            mark_email_for_internal_delivery,
            message_id,
            approved_by=request.approved_by,
            note=request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result}


@router.get("/email/internal-approved")
async def email_internal_approved(limit: int = Query(default=20, ge=1, le=100)):
    rows = await asyncio.to_thread(list_messages_approved_for_internal_delivery, limit)
    return {"messages": rows, "count": len(rows)}


@router.post("/email/messages/{message_id}/internal-deliver")
async def deliver_inbound_email_to_internal(message_id: int, request: EmailInboundApprovalRequest):
    try:
        result = await asyncio.to_thread(
            deliver_inbound_email_to_internal_input,
            message_id,
            delivered_by=f"api:{request.approved_by or 'admin'}",
            note=request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result}
