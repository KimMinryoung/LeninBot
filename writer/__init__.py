"""Personal fiction-writing workspace with selectable provider models.

Package layout mirrors the repo's other subsystems:
- config     — runtime constants
- matching   — whitespace/quote-tolerant passage matching
- store      — schema, projects, manuscripts, messages, settings (Postgres)
- documents  — background/reference documents
- models     — model catalog, persisted selection, provider resolution
- prompts    — system prompt, manuscript context, history budget, parsing
- tools      — tool schemas + per-project handlers (system.writer profile)
- runs       — in-process registry of live background runs
- stream     — SSE generation runs and reattachable streams

creative_writer.py is a compatibility shim re-exporting this API.
"""

from writer.documents import (
    delete_document,
    get_document,
    get_shared_document,
    list_documents,
    list_shared_documents,
    save_document,
    save_shared_document,
    search_documents,
    update_document,
)
from writer.models import (
    WRITER_DEFAULT_CHOICE,
    WRITER_INPUT_PRICE_PER_MTOK,
    WRITER_MODEL,
    WRITER_MODEL_CHOICES,
    WRITER_MODEL_DISPLAY,
    WRITER_OUTPUT_PRICE_PER_MTOK,
    get_selected_model_choice,
    list_writer_models,
    resolve_writer_model,
    set_selected_model_choice,
)
from writer.runs import WriterRun, get_active_run
from writer.store import (
    append_manuscript,
    create_project,
    delete_project,
    ensure_writer_tables,
    get_manuscript,
    get_project,
    get_project_with_messages,
    get_writer_setting,
    list_manuscript_revisions,
    list_projects,
    replace_manuscript_range,
    replace_manuscript_text,
    restore_project,
    save_manuscript,
    search_manuscript,
    set_writer_setting,
    trash_project,
    update_project,
)
from writer.stream import stream_active_run, stream_writer_reply
from writer.tools import build_critic_tools, build_writer_tools

__all__ = [
    "WRITER_DEFAULT_CHOICE",
    "WRITER_INPUT_PRICE_PER_MTOK",
    "WRITER_MODEL",
    "WRITER_MODEL_CHOICES",
    "WRITER_MODEL_DISPLAY",
    "WRITER_OUTPUT_PRICE_PER_MTOK",
    "WriterRun",
    "append_manuscript",
    "build_critic_tools",
    "build_writer_tools",
    "create_project",
    "delete_document",
    "delete_project",
    "ensure_writer_tables",
    "get_active_run",
    "get_document",
    "get_manuscript",
    "get_project",
    "get_project_with_messages",
    "get_selected_model_choice",
    "get_writer_setting",
    "get_shared_document",
    "list_documents",
    "list_shared_documents",
    "save_shared_document",
    "list_manuscript_revisions",
    "list_projects",
    "list_writer_models",
    "replace_manuscript_range",
    "replace_manuscript_text",
    "resolve_writer_model",
    "restore_project",
    "save_document",
    "save_manuscript",
    "search_documents",
    "search_manuscript",
    "set_selected_model_choice",
    "set_writer_setting",
    "stream_active_run",
    "stream_writer_reply",
    "trash_project",
    "update_document",
    "update_project",
]
