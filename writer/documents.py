"""Background/reference documents: per-project worldbuilding, character sheets,
outlines, and research notes stored separately from the manuscript."""

from __future__ import annotations

from db import writer_query as db_query
from db import writer_query_one as db_query_one

from writer.store import get_project

_DOCUMENT_COLUMNS = "id, project_id, title, kind, length(content)::int AS char_count, created_at, updated_at"


def list_documents(project_id: int) -> list[dict]:
    # manuscript_chars_at_update: how long the manuscript was when the document
    # was last saved (via the revision log). The prompt renders the gap as a
    # staleness nudge so the agent keeps its story bible current. NULL when the
    # document predates every manuscript revision.
    return db_query(
        f"""SELECT {_DOCUMENT_COLUMNS},
                   (SELECT length(r.after_body)::int
                      FROM writer_manuscript_revisions r
                     WHERE r.project_id = writer_documents.project_id
                       AND r.created_at <= writer_documents.updated_at
                     ORDER BY r.created_at DESC, r.id DESC
                     LIMIT 1) AS manuscript_chars_at_update
             FROM writer_documents
            WHERE project_id = %s
            ORDER BY updated_at DESC, id DESC""",
        (project_id,),
    )


def get_document(project_id: int, document_id: int | None = None, title: str | None = None) -> dict | None:
    if document_id is not None:
        return db_query_one(
            f"""SELECT {_DOCUMENT_COLUMNS}, content
                 FROM writer_documents
                WHERE project_id = %s AND id = %s""",
            (project_id, document_id),
        )
    if title:
        return db_query_one(
            f"""SELECT {_DOCUMENT_COLUMNS}, content
                 FROM writer_documents
                WHERE project_id = %s AND lower(title) = lower(%s)""",
            (project_id, title.strip()),
        )
    return None


def save_document(project_id: int, title: str, content: str, kind: str = "note") -> dict | None:
    """Create or overwrite a background document, addressed by title (upsert)."""
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


def update_document(project_id: int, document_id: int, title: str, kind: str, content: str) -> dict | None:
    return db_query_one(
        f"""UPDATE writer_documents
               SET title = %s, kind = %s, content = %s, updated_at = NOW()
             WHERE project_id = %s AND id = %s
         RETURNING {_DOCUMENT_COLUMNS}""",
        (title.strip(), kind.strip() or "note", content, project_id, document_id),
    )


def delete_document(project_id: int, document_id: int) -> bool:
    row = db_query_one(
        "DELETE FROM writer_documents WHERE project_id = %s AND id = %s RETURNING id",
        (project_id, document_id),
    )
    return bool(row)


def search_documents(project_id: int, query: str, limit: int = 8) -> list[dict]:
    needle = query.strip()
    if not needle:
        return []
    rows = db_query(
        """SELECT id, title, kind, content
             FROM writer_documents
            WHERE project_id = %s
              AND (title ILIKE %s OR content ILIKE %s)
            ORDER BY updated_at DESC
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
                "snippet": snippet,
            }
        )
    return results
