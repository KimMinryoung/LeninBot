"""Vector corpus search, chunking, ingestion, and source context helpers."""

import json
import logging
import os

from corpus.embeddings import _get_exp_embeddings

logger = logging.getLogger(__name__)

def similarity_search(
    query: str,
    k: int = 5,
    layer: str = None,
    rerank: bool = False,
    *,
    author: str | None = None,
    title: str | None = None,
    year: int | str | None = None,
    keywords: str | list[str] | None = None,
) -> list:
    """Search lenin_corpus via pgvector cosine similarity.

    Returns list of LangChain Document objects with page_content + metadata.
    When rerank=True, re-scores results with a cross-encoder for better relevance.

    Note on recall: when a layer filter is present, the default HNSW ef_search
    is too small to find enough candidates that also satisfy the layer filter —
    top-k by distance may all be on the wrong layer and get dropped, returning
    an empty result even when the DB has plenty of matches. We bump ef_search
    per-transaction to mitigate this (the SET clause on the SP itself is
    forbidden to non-superusers on managed Postgres).
    """
    from langchain_core.documents import Document
    from psycopg2.extras import RealDictCursor
    from db import get_conn

    emb = _get_exp_embeddings()
    vec = emb.embed_query(query)
    embedding_str = "[" + ",".join(str(v) for v in vec) + "]"
    fetch_k = k * 3 if rerank else k
    # Match SP's default threshold (0.4). Kept as a parameter pass-through so
    # the behaviour is visible to the reader without reading the SP.
    threshold = 0.4
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # SET LOCAL so the bump applies only within this transaction.
                # Integer literal is safe; no user input flows through this SET.
                cur.execute("SET LOCAL hnsw.ef_search = 200")
                if author or title or year or keywords:
                    clauses = ["1 = 1"]
                    params: list = []
                    if layer:
                        clauses.append("metadata->>'layer' = %s")
                        params.append(layer)
                    if author:
                        clauses.append("metadata->>'author' ILIKE %s")
                        params.append(f"%{author}%")
                    if title:
                        clauses.append(
                            "(metadata->>'title' ILIKE %s OR metadata->>'source' ILIKE %s)"
                        )
                        title_pattern = f"%{title}%"
                        params.extend([title_pattern, title_pattern])
                    if year:
                        clauses.append("metadata->>'year' = %s")
                        params.append(str(year))
                    kw_values = [keywords] if isinstance(keywords, str) else (keywords or [])
                    for kw in [str(v).strip() for v in kw_values if str(v).strip()]:
                        clauses.append("(content ILIKE %s OR metadata->>'title' ILIKE %s)")
                        kw_pattern = f"%{kw}%"
                        params.extend([kw_pattern, kw_pattern])
                    params.extend([embedding_str, fetch_k])
                    cur.execute(
                        f"""
                        SELECT content, metadata
                          FROM lenin_corpus
                         WHERE {' AND '.join(clauses)}
                         ORDER BY embedding <=> %s::vector
                         LIMIT %s
                        """,
                        params,
                    )
                else:
                    cur.execute(
                        "SELECT * FROM match_documents(%s::vector, %s, %s, %s)",
                        (embedding_str, threshold, fetch_k, layer),
                    )
                rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        error_msg = str(e)
        if "57014" in error_msg or "timeout" in error_msg.lower():
            logger.warning("[shared] similarity_search timeout")
        else:
            logger.warning("[shared] similarity_search error: %s", e)
        return []

    docs = [
        Document(page_content=row.get("content", ""), metadata=row.get("metadata", {}))
        for row in rows
        if row.get("content")
    ]

    if rerank and len(docs) > 2:
        try:
            ranked = emb.rerank(query, [d.page_content for d in docs], top_k=k)
            docs = [docs[idx] for idx, _score in ranked]
        except Exception as e:
            logger.warning("[shared] rerank failed, using original order: %s", e)
            docs = docs[:k]
    else:
        docs = docs[:k]

    return docs


# ── Corpus Ingestion (modern_analysis layer) ─────────────────────────
# Chunking/embedding pipeline used by both hub-curation auto-ingest and the
# manual literature-drop CLI. Keeps insert shape compatible with the existing
# match_documents() SP and the historical chunk size (~760 char avg, 1000 max).

_CORPUS_CHUNK_SIZE = 900
_CORPUS_CHUNK_OVERLAP = 120
_EMBED_BATCH_SIZE = int(os.getenv("CORPUS_EMBED_BATCH_SIZE", "32"))


def _chunk_text(text: str, size: int = _CORPUS_CHUNK_SIZE, overlap: int = _CORPUS_CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window chunker tuned to match the existing corpus distribution."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break on a newline or whitespace near the boundary for cleaner cuts.
        if end < len(text):
            window = text[max(start, end - 120):end]
            cut_rel = max(window.rfind("\n"), window.rfind(". "), window.rfind(" "))
            if cut_rel > 0:
                end = (end - 120 if end - 120 > start else start) + cut_rel + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def ingest_to_corpus(
    content: str,
    source: str,
    *,
    layer: str = "modern_analysis",
    author: str | None = None,
    year: int | None = None,
    extra_metadata: dict | None = None,
    skip_if_source_url_exists: bool = True,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """Chunk → embed → insert into lenin_corpus. Returns the number of chunks inserted.

    By default, skips ingest (returns 0) when extra_metadata['source_url'] already
    exists anywhere in the corpus — prevents duplicate embedding when two curations
    or drops point at the same article. Pass `skip_if_source_url_exists=False` to
    force-insert (e.g. reingest flows that delete the prior rows themselves).
    """
    from db import get_conn

    source_url = (extra_metadata or {}).get("source_url")
    if skip_if_source_url_exists and source_url:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM lenin_corpus WHERE metadata->>'source_url' = %s LIMIT 1",
                    (source_url,),
                )
                if cur.fetchone():
                    logger.info(
                        "[shared] ingest_to_corpus: skipping %r — source_url already in corpus",
                        source_url[:80],
                    )
                    return 0

    size = int(chunk_size or _CORPUS_CHUNK_SIZE)
    overlap = int(chunk_overlap if chunk_overlap is not None else _CORPUS_CHUNK_OVERLAP)
    chunks = _chunk_text(content, size=size, overlap=overlap)
    if not chunks:
        return 0

    emb = _get_exp_embeddings()
    batch_size = max(1, _EMBED_BATCH_SIZE)
    vectors = []
    for start in range(0, len(chunks), batch_size):
        vectors.extend(emb.embed_documents(chunks[start:start + batch_size]))
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
        )

    base_meta = {"layer": layer, "source": source}
    if author:
        base_meta["author"] = author
    if year:
        base_meta["year"] = int(year)
    if extra_metadata:
        base_meta.update(extra_metadata)

    rows = []
    chunk_count = len(chunks)
    for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
        emb_str = "[" + ",".join(str(v) for v in vec) + "]"
        chunk_meta = dict(base_meta)
        chunk_meta.update({
            "chunk_index": idx,
            "chunk_count": chunk_count,
            "chunk_size": size,
            "chunk_overlap": overlap,
        })
        rows.append((chunk, json.dumps(chunk_meta, ensure_ascii=False), emb_str))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO lenin_corpus(content, metadata, embedding) VALUES (%s, %s::jsonb, %s::vector)",
                rows,
            )
        conn.commit()

    return len(chunks)


def fetch_corpus_source_context(
    source: str,
    *,
    center_index: int = 0,
    window: int = 1,
    max_chars: int = 9000,
) -> str:
    """Fetch neighboring chunks from one corpus source for parent-context expansion."""
    from db import query as db_query

    source = (source or "").strip()
    if not source:
        return ""
    center_index = max(0, int(center_index or 0))
    window = max(0, int(window or 0))
    max_chars = max(1000, int(max_chars or 9000))
    lo = max(0, center_index - window)
    hi = center_index + window
    try:
        rows = db_query(
            """
            SELECT content, metadata
              FROM lenin_corpus
             WHERE metadata->>'source' = %s
               AND COALESCE((metadata->>'chunk_index')::int, 0) BETWEEN %s AND %s
             ORDER BY COALESCE((metadata->>'chunk_index')::int, 0)
            """,
            (source, lo, hi),
        )
    except Exception as e:
        logger.warning("[shared] fetch_corpus_source_context error: %s", e)
        return ""

    parts: list[str] = []
    total = 0
    for row in rows:
        text = str(row.get("content") or "").strip()
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "\n... (context truncated)"
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts)


def delete_corpus_source(source: str, layer: str | None = None) -> int:
    """Delete all chunks matching a given metadata.source (optionally filtered by layer).

    Use before re-ingesting the same document to avoid duplicate chunks."""
    from db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            if layer:
                cur.execute(
                    "DELETE FROM lenin_corpus WHERE metadata->>'source' = %s AND metadata->>'layer' = %s",
                    (source, layer),
                )
            else:
                cur.execute(
                    "DELETE FROM lenin_corpus WHERE metadata->>'source' = %s",
                    (source,),
                )
            deleted = cur.rowcount
        conn.commit()
    return deleted
