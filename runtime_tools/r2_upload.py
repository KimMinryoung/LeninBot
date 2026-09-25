"""upload_to_r2 tool: upload a local file to Cloudflare R2 and record it in file_registry."""
import asyncio
import logging
import os

logger = logging.getLogger(__name__)


UPLOAD_TO_R2_TOOL = {
    "name": "upload_to_r2",
    "description": (
        "Upload a local file to Cloudflare R2 and get a public URL. "
        "Automatically registers the file in the file_registry DB table so other agents can find it. "
        "Use for images, documents, or any file that needs a public URL (e.g. email attachments, web assets)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "local_path": {"type": "string", "description": "Absolute path to the local file."},
            "key": {"type": "string", "description": "Object key/path in R2 bucket (e.g. 'email-assets/logo.png'). Defaults to filename."},
            "description": {"type": "string", "description": "What this file is / what it's for."},
            "category": {
                "type": "string",
                "enum": ["email-asset", "image", "document", "research", "general"],
                "description": "File category for search. Default: general.",
            },
        },
        "required": ["local_path"],
    },
}


async def upload_to_r2_tool(
    local_path: str, key: str | None = None, description: str = "", category: str = "general",
) -> str:
    from shared import upload_to_r2
    from db import execute as db_execute, query as db_query
    import mimetypes

    path = os.path.abspath(local_path)
    if not os.path.isfile(path):
        return f"File not found: {local_path}"

    filename = os.path.basename(path)
    file_size = os.path.getsize(path)
    content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

    if key is None:
        key = f"{category}/{filename}" if category != "general" else filename

    # Check if already registered by local_path or R2 key
    existing = await asyncio.to_thread(
        db_query,
        "SELECT id, public_url FROM file_registry WHERE local_path = %s OR public_url LIKE %s LIMIT 1",
        (path, f"%/{key}"),
    )
    if existing:
        return f"Already registered: {existing[0]['public_url']}\n(file_registry id: {existing[0]['id']})"

    url = await asyncio.to_thread(upload_to_r2, path, key, content_type)
    if not url:
        return "R2 upload failed. Check R2 env config."

    # Get current task context for tracking
    task_id = None
    agent_type = None
    try:
        from llm.runtime_context import current_task_ctx
        ctx = current_task_ctx.get()
        task_id = ctx["task_id"] if ctx else None
    except Exception:
        pass

    # Register in file_registry
    registry_id = None
    try:
        reg_rows = await asyncio.to_thread(
            db_query,
            "INSERT INTO file_registry (local_path, public_url, filename, content_type, description, category, file_size, created_by_task_id, created_by_agent) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (path, url, filename, content_type, description or filename, category, file_size, task_id, agent_type),
        )
        registry_id = reg_rows[0]["id"] if reg_rows else None
    except Exception as e:
        logger.warning("file_registry insert failed: %s", e)

    reg_line = f"\nfile_registry id: {registry_id}" if registry_id else "\n(file_registry registration failed)"
    return f"Uploaded: {url}\nLocal: {path}\nSize: {file_size} bytes\nCategory: {category}{reg_line}"
