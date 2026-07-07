"""Background/reference documents: worldbuilding, character sheets, outlines,
and research notes stored separately from the manuscript.

Two scopes live in one table: project documents (project_id set) and SHARED
documents (project_id IS NULL) visible to every project. A project document
shadows a shared one with the same title for title-addressed reads, so a
project can override common reference material locally. Agent writes
(save_document) are always project-scoped; shared documents are managed by
the user through the shared API/UI surface.
"""

from __future__ import annotations

from db import writer_query as db_query
from db import writer_query_one as db_query_one

from writer.store import get_project

_DOCUMENT_COLUMNS = (
    "id, project_id, (project_id IS NULL) AS shared, title, kind, "
    "length(content)::int AS char_count, created_at, updated_at"
)


def list_documents(project_id: int) -> list[dict]:
    """Project documents plus shared documents, shared marked and listed after
    project docs at equal recency (the UI and prompt group them anyway)."""
    # manuscript_chars_at_update: how long the manuscript was when the document
    # was last saved (via the revision log). The prompt renders the gap as a
    # staleness nudge so the agent keeps its story bible current. NULL when the
    # document predates every manuscript revision — and always NULL for shared
    # documents (the correlated project_id is NULL), which is correct: shared
    # reference material has no per-project staleness.
    return db_query(
        f"""SELECT {_DOCUMENT_COLUMNS},
                   (SELECT length(r.after_body)::int
                      FROM writer_manuscript_revisions r
                     WHERE r.project_id = writer_documents.project_id
                       AND r.created_at <= writer_documents.updated_at
                     ORDER BY r.created_at DESC, r.id DESC
                     LIMIT 1) AS manuscript_chars_at_update
             FROM writer_documents
            WHERE project_id = %s OR project_id IS NULL
            ORDER BY (project_id IS NULL) ASC, updated_at DESC, id DESC""",
        (project_id,),
    )


def list_shared_documents() -> list[dict]:
    return db_query(
        f"""SELECT {_DOCUMENT_COLUMNS}
             FROM writer_documents
            WHERE project_id IS NULL
            ORDER BY updated_at DESC, id DESC""",
    )


def get_document(project_id: int, document_id: int | None = None, title: str | None = None) -> dict | None:
    """Fetch one document visible to this project (own scope or shared). By
    title, the project's own document shadows a shared one of the same name."""
    if document_id is not None:
        return db_query_one(
            f"""SELECT {_DOCUMENT_COLUMNS}, content
                 FROM writer_documents
                WHERE (project_id = %s OR project_id IS NULL) AND id = %s""",
            (project_id, document_id),
        )
    if title:
        return db_query_one(
            f"""SELECT {_DOCUMENT_COLUMNS}, content
                 FROM writer_documents
                WHERE (project_id = %s OR project_id IS NULL)
                  AND lower(title) = lower(%s)
                ORDER BY project_id NULLS LAST
                LIMIT 1""",
            (project_id, title.strip()),
        )
    return None


def get_shared_document(document_id: int) -> dict | None:
    return db_query_one(
        f"""SELECT {_DOCUMENT_COLUMNS}, content
             FROM writer_documents
            WHERE project_id IS NULL AND id = %s""",
        (document_id,),
    )


def save_document(project_id: int, title: str, content: str, kind: str = "note") -> dict | None:
    """Create or overwrite a PROJECT document, addressed by title (upsert).
    Saving a title that exists as a shared document creates a project-local
    override; the shared original is untouched."""
    if not get_project(project_id):
        return None
    return db_query_one(
        f"""INSERT INTO writer_documents (project_id, title, kind, content)
             VALUES (%s, %s, %s, %s)
        ON CONFLICT (project_id, title)
          DO UPDATE SET kind = EXCLUDED.kind,
                        content = EXCLUDED.content,
                        updated_at = NOW()
          RETURNING {_DOCUMENT_COLUMNS}""",
        (project_id, title.strip(), kind.strip() or "note", content),
    )


def save_shared_document(title: str, content: str, kind: str = "note") -> dict | None:
    """Create or overwrite a SHARED document (visible to all projects),
    addressed by title (upsert against the partial unique index)."""
    return db_query_one(
        f"""INSERT INTO writer_documents (project_id, title, kind, content)
             VALUES (NULL, %s, %s, %s)
        ON CONFLICT (title) WHERE project_id IS NULL
          DO UPDATE SET kind = EXCLUDED.kind,
                        content = EXCLUDED.content,
                        updated_at = NOW()
          RETURNING {_DOCUMENT_COLUMNS}""",
        (title.strip(), kind.strip() or "note", content),
    )


def update_document(project_id: int | None, document_id: int, title: str, kind: str, content: str) -> dict | None:
    """Update one document in an EXACT scope: pass a project id for a project
    document, None for a shared one (IS NOT DISTINCT FROM matches NULL)."""
    return db_query_one(
        f"""UPDATE writer_documents
               SET title = %s, kind = %s, content = %s, updated_at = NOW()
             WHERE project_id IS NOT DISTINCT FROM %s AND id = %s
         RETURNING {_DOCUMENT_COLUMNS}""",
        (title.strip(), kind.strip() or "note", content, project_id, document_id),
    )


def delete_document(project_id: int | None, document_id: int) -> bool:
    """Delete one document in an EXACT scope (None = shared)."""
    row = db_query_one(
        "DELETE FROM writer_documents WHERE project_id IS NOT DISTINCT FROM %s AND id = %s RETURNING id",
        (project_id, document_id),
    )
    return bool(row)


def search_documents(project_id: int, query: str, limit: int = 8) -> list[dict]:
    needle = query.strip()
    if not needle:
        return []
    rows = db_query(
        """SELECT id, title, kind, content, (project_id IS NULL) AS shared
             FROM writer_documents
            WHERE (project_id = %s OR project_id IS NULL)
              AND (title ILIKE %s OR content ILIKE %s)
            ORDER BY (project_id IS NULL) ASC, updated_at DESC
            LIMIT %s""",
        (project_id, f"%{needle}%", f"%{needle}%", limit),
    )
    results: list[dict] = []
    lowered = needle.lower()
    for row in rows:
        content = str(row.get("content") or "")
        found = content.lower().find(lowered)
        if found < 0:
            snippet = content[:400]
        else:
            snippet = content[max(0, found - 180):found + len(needle) + 220]
        results.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "kind": row.get("kind"),
                "shared": bool(row.get("shared")),
                "snippet": snippet,
            }
        )
    return results
