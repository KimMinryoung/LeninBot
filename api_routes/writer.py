import asyncio
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from api_security import require_writer_access

router = APIRouter()


class WriterProjectRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    premise: str = Field(default="", max_length=20000)
    style_notes: str = Field(default="", max_length=20000)


class WriterMessageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=30000)
    selection_start: int | None = Field(default=None, ge=0)
    selection_end: int | None = Field(default=None, ge=0)
    model: str | None = Field(default=None, max_length=64)
    critic: bool = Field(default=False)


class WriterSettingsRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=64)


class WriterPublishRequest(BaseModel):
    is_public: bool


class WriterDocumentRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    kind: str = Field(default="note", max_length=50)
    content: str = Field(default="", max_length=500_000)


class WriterManuscriptRequest(BaseModel):
    body: str = Field(default="", max_length=5_000_000)
    note: str = Field(default="", max_length=500)


class WriterManuscriptAppendRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500_000)
    note: str = Field(default="", max_length=500)


class WriterManuscriptReplaceRequest(BaseModel):
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    replacement: str = Field(default="", max_length=500_000)
    note: str = Field(default="", max_length=500)


@router.get("/writer")
async def writer_page():
    html_path = Path(__file__).resolve().parents[1] / "frontend" / "writer.html"
    return Response(
        content=html_path.read_text(encoding="utf-8"),
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow"},
    )


@router.get("/writer/projects", dependencies=[Depends(require_writer_access)])
async def list_writer_projects(
    limit: int = Query(default=100, ge=1, le=200),
    status: str = Query(default="active", pattern="^(active|deleted)$"),
):
    from creative_writer import (
        WRITER_INPUT_PRICE_PER_MTOK,
        WRITER_MODEL,
        WRITER_MODEL_DISPLAY,
        WRITER_OUTPUT_PRICE_PER_MTOK,
        get_selected_model_choice,
        list_projects,
        list_writer_models,
    )

    projects = await asyncio.to_thread(list_projects, limit, status)
    selected = await asyncio.to_thread(get_selected_model_choice)
    return {
        "projects": projects,
        "model": {
            "id": WRITER_MODEL,
            "display_name": WRITER_MODEL_DISPLAY,
            "input_price_per_mtok": WRITER_INPUT_PRICE_PER_MTOK,
            "output_price_per_mtok": WRITER_OUTPUT_PRICE_PER_MTOK,
        },
        "models": list_writer_models(),
        "selected_model": selected,
    }


@router.put("/writer/settings", dependencies=[Depends(require_writer_access)])
async def save_writer_settings(request: WriterSettingsRequest):
    from creative_writer import set_selected_model_choice

    try:
        saved = await asyncio.to_thread(set_selected_model_choice, request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"selected_model": saved}


@router.post("/writer/projects", dependencies=[Depends(require_writer_access)])
async def create_writer_project(request: WriterProjectRequest):
    from creative_writer import create_project

    project = await asyncio.to_thread(
        create_project,
        request.title,
        request.premise,
        request.style_notes,
    )
    return {"project": project}


@router.get("/writer/projects/{project_id}", dependencies=[Depends(require_writer_access)])
async def get_writer_project(project_id: int):
    from creative_writer import get_project_with_messages

    project = await asyncio.to_thread(get_project_with_messages, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="writer project not found")
    return project


@router.patch("/writer/projects/{project_id}", dependencies=[Depends(require_writer_access)])
async def update_writer_project(project_id: int, request: WriterProjectRequest):
    from creative_writer import get_project_with_messages, update_project

    project = await asyncio.to_thread(
        update_project,
        project_id,
        request.title,
        request.premise,
        request.style_notes,
    )
    if not project:
        raise HTTPException(status_code=404, detail="writer project not found")
    return await asyncio.to_thread(get_project_with_messages, project_id)


@router.delete("/writer/projects/{project_id}", dependencies=[Depends(require_writer_access)])
async def delete_writer_project(project_id: int, permanent: bool = Query(default=False)):
    from creative_writer import delete_project, trash_project

    if permanent:
        deleted = await asyncio.to_thread(delete_project, project_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="writer project not found")
        return {"deleted": True, "permanent": True}
    trashed = await asyncio.to_thread(trash_project, project_id)
    if not trashed:
        raise HTTPException(status_code=404, detail="writer project not found or already trashed")
    return {"deleted": True, "permanent": False}


@router.post("/writer/projects/{project_id}/publish", dependencies=[Depends(require_writer_access)])
async def publish_writer_project(project_id: int, request: WriterPublishRequest):
    from creative_writer import set_project_public

    project = await asyncio.to_thread(set_project_public, project_id, request.is_public)
    if not project:
        raise HTTPException(status_code=404, detail="writer project not found")
    return {"project": project}


_PUBLIC_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")


@router.get("/writer/public/{slug}")
async def get_public_novel(slug: str):
    """Anonymous manuscript read for the public site — no writer key required."""
    from creative_writer import get_public_manuscript

    if not _PUBLIC_SLUG_RE.match(slug):
        raise HTTPException(status_code=404, detail="novel not found")
    novel = await asyncio.to_thread(get_public_manuscript, slug)
    if not novel:
        raise HTTPException(status_code=404, detail="novel not found")
    return {"novel": novel}


@router.post("/writer/projects/{project_id}/restore", dependencies=[Depends(require_writer_access)])
async def restore_writer_project(project_id: int):
    from creative_writer import restore_project

    restored = await asyncio.to_thread(restore_project, project_id)
    if not restored:
        raise HTTPException(status_code=404, detail="writer project not found in trash")
    return {"restored": True}


@router.get("/writer/projects/{project_id}/manuscript", dependencies=[Depends(require_writer_access)])
async def get_writer_manuscript(project_id: int):
    from creative_writer import get_manuscript

    manuscript = await asyncio.to_thread(get_manuscript, project_id)
    if not manuscript:
        raise HTTPException(status_code=404, detail="writer project not found")
    return {"manuscript": manuscript}


@router.put("/writer/projects/{project_id}/manuscript", dependencies=[Depends(require_writer_access)])
async def save_writer_manuscript(project_id: int, request: WriterManuscriptRequest):
    from creative_writer import save_manuscript

    manuscript = await asyncio.to_thread(save_manuscript, project_id, request.body, request.note)
    if not manuscript:
        raise HTTPException(status_code=404, detail="writer project not found")
    return {"manuscript": manuscript}


@router.get("/writer/projects/{project_id}/manuscript/search", dependencies=[Depends(require_writer_access)])
async def search_writer_manuscript(
    project_id: int,
    q: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(default=20, ge=1, le=50),
):
    from creative_writer import get_project, search_manuscript

    project = await asyncio.to_thread(get_project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="writer project not found")
    results = await asyncio.to_thread(search_manuscript, project_id, q, limit)
    return {"results": results}


@router.post("/writer/projects/{project_id}/manuscript/append", dependencies=[Depends(require_writer_access)])
async def append_writer_manuscript(project_id: int, request: WriterManuscriptAppendRequest):
    from creative_writer import append_manuscript

    manuscript = await asyncio.to_thread(append_manuscript, project_id, request.text, request.note)
    if not manuscript:
        raise HTTPException(status_code=404, detail="writer project not found")
    return {"manuscript": manuscript}


@router.post("/writer/projects/{project_id}/manuscript/replace", dependencies=[Depends(require_writer_access)])
async def replace_writer_manuscript(project_id: int, request: WriterManuscriptReplaceRequest):
    from creative_writer import replace_manuscript_range

    try:
        manuscript = await asyncio.to_thread(
            replace_manuscript_range,
            project_id,
            request.start,
            request.end,
            request.replacement,
            request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not manuscript:
        raise HTTPException(status_code=404, detail="writer project not found")
    return {"manuscript": manuscript}


@router.get("/writer/projects/{project_id}/manuscript/revisions", dependencies=[Depends(require_writer_access)])
async def list_writer_manuscript_revisions(
    project_id: int,
    limit: int = Query(default=30, ge=1, le=100),
):
    from creative_writer import get_project, list_manuscript_revisions

    project = await asyncio.to_thread(get_project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="writer project not found")
    revisions = await asyncio.to_thread(list_manuscript_revisions, project_id, limit)
    return {"revisions": revisions}


@router.get("/writer/documents", dependencies=[Depends(require_writer_access)])
async def list_writer_shared_documents():
    """Shared background documents — visible to every writer project."""
    from creative_writer import list_shared_documents

    documents = await asyncio.to_thread(list_shared_documents)
    return {"documents": documents}


@router.post("/writer/documents", dependencies=[Depends(require_writer_access)])
async def save_writer_shared_document(request: WriterDocumentRequest):
    from creative_writer import save_shared_document

    document = await asyncio.to_thread(
        save_shared_document, request.title, request.content, request.kind
    )
    if not document:
        raise HTTPException(status_code=500, detail="failed to save shared document")
    return {"document": document}


@router.get("/writer/documents/{document_id}", dependencies=[Depends(require_writer_access)])
async def get_writer_shared_document(document_id: int):
    from creative_writer import get_shared_document

    document = await asyncio.to_thread(get_shared_document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="shared document not found")
    return {"document": document}


@router.put("/writer/documents/{document_id}", dependencies=[Depends(require_writer_access)])
async def update_writer_shared_document(document_id: int, request: WriterDocumentRequest):
    from creative_writer import update_document

    document = await asyncio.to_thread(
        update_document, None, document_id, request.title, request.kind, request.content
    )
    if not document:
        raise HTTPException(status_code=404, detail="shared document not found")
    return {"document": document}


@router.delete("/writer/documents/{document_id}", dependencies=[Depends(require_writer_access)])
async def delete_writer_shared_document(document_id: int):
    from creative_writer import delete_document

    deleted = await asyncio.to_thread(delete_document, None, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="shared document not found")
    return {"deleted": True}


@router.get("/writer/projects/{project_id}/documents", dependencies=[Depends(require_writer_access)])
async def list_writer_documents(project_id: int):
    from creative_writer import get_project, list_documents

    project = await asyncio.to_thread(get_project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="writer project not found")
    documents = await asyncio.to_thread(list_documents, project_id)
    return {"documents": documents}


@router.post("/writer/projects/{project_id}/documents", dependencies=[Depends(require_writer_access)])
async def save_writer_document(project_id: int, request: WriterDocumentRequest):
    from creative_writer import save_document

    document = await asyncio.to_thread(
        save_document, project_id, request.title, request.content, request.kind
    )
    if not document:
        raise HTTPException(status_code=404, detail="writer project not found")
    return {"document": document}


@router.get("/writer/projects/{project_id}/documents/{document_id}", dependencies=[Depends(require_writer_access)])
async def get_writer_document(project_id: int, document_id: int):
    from creative_writer import get_document

    document = await asyncio.to_thread(get_document, project_id, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="writer document not found")
    return {"document": document}


@router.put("/writer/projects/{project_id}/documents/{document_id}", dependencies=[Depends(require_writer_access)])
async def update_writer_document(project_id: int, document_id: int, request: WriterDocumentRequest):
    from creative_writer import update_document

    document = await asyncio.to_thread(
        update_document, project_id, document_id, request.title, request.kind, request.content
    )
    if not document:
        raise HTTPException(status_code=404, detail="writer document not found")
    return {"document": document}


@router.delete("/writer/projects/{project_id}/documents/{document_id}", dependencies=[Depends(require_writer_access)])
async def delete_writer_document(project_id: int, document_id: int):
    from creative_writer import delete_document

    deleted = await asyncio.to_thread(delete_document, project_id, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="writer document not found")
    return {"deleted": True}


@router.get("/writer/projects/{project_id}/stream", dependencies=[Depends(require_writer_access)])
async def writer_stream(project_id: int, http_req: Request):
    """Reattach to a live background writer run (page reload / dropped stream).
    Emits no_active_run immediately when there is nothing to attach to."""
    from creative_writer import stream_active_run

    return StreamingResponse(
        stream_active_run(
            project_id=project_id,
            client_disconnected=http_req.is_disconnected,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/writer/projects/{project_id}/messages", dependencies=[Depends(require_writer_access)])
async def writer_message(project_id: int, request: WriterMessageRequest, http_req: Request):
    from creative_writer import stream_writer_reply

    return StreamingResponse(
        stream_writer_reply(
            project_id=project_id,
            prompt=request.prompt,
            selection_start=request.selection_start,
            selection_end=request.selection_end,
            model_choice=request.model,
            critic=request.critic,
            client_disconnected=http_req.is_disconnected,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


