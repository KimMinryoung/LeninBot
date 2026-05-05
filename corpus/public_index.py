"""Public and self-produced analysis indexing helpers."""

import logging
from datetime import datetime, timezone

from corpus.store import delete_corpus_source, ingest_to_corpus

logger = logging.getLogger(__name__)

def save_self_produced_analysis(
    *,
    title: str,
    content: str,
    category: str = "insight",
    source_context: str = "",
) -> dict:
    """Persist the agent's own reusable analysis in a dedicated vector layer."""
    try:
        title = (title or "").strip()
        content = (content or "").strip()
        if not title:
            return {"ok": False, "error": "title is required"}
        if not content:
            return {"ok": False, "error": "content is required"}

        now = datetime.now(timezone.utc)
        source = f"self_analysis:{now.strftime('%Y%m%dT%H%M%SZ')}:{title[:80]}"
        chunks = ingest_to_corpus(
            content,
            source=source,
            layer="self_produced_analysis",
            author="Cyber-Lenin Orchestrator",
            year=now.year,
            extra_metadata={
                "title": title,
                "category": category or "insight",
                "source_context": source_context or "",
                "created_at": now.isoformat(),
                "source_type": "self_produced_analysis",
            },
            skip_if_source_url_exists=False,
        )
        logger.info("[shared] saved self_produced_analysis: %r (%d chunks)", title, chunks)
        return {"ok": True, "chunks": chunks, "source": source, "layer": "self_produced_analysis"}
    except Exception as e:
        logger.warning("[shared] save_self_produced_analysis error: %s", e)
        return {"ok": False, "error": str(e)}


def public_self_analysis_source(kind: str, slug: str) -> str:
    """Stable lenin_corpus metadata.source for public Cyber-Lenin outputs."""
    clean_kind = (kind or "public").strip().lower().replace(":", "_")
    clean_slug = (slug or "").strip().lower()
    return f"public_self_analysis:{clean_kind}:{clean_slug}"


def index_public_self_analysis(
    *,
    kind: str,
    slug: str,
    title: str,
    content: str,
    public_url: str,
    summary: str | None = None,
    content_sha256: str | None = None,
    extra_metadata: dict | None = None,
    chunk_size: int = 3200,
    chunk_overlap: int = 240,
) -> dict:
    """Index a public Cyber-Lenin output into self_produced_analysis.

    The source is stable per public artifact, so publish/edit paths can delete
    and reinsert the same document without creating duplicate chunks.
    """
    try:
        slug = (slug or "").strip()
        title = (title or "").strip()
        content = (content or "").strip()
        if not slug:
            return {"ok": False, "error": "slug is required"}
        if not title:
            return {"ok": False, "error": "title is required"}
        if not content:
            return {"ok": False, "error": "content is required"}

        source = public_self_analysis_source(kind, slug)
        deleted = delete_corpus_source(source, layer="self_produced_analysis")
        now = datetime.now(timezone.utc)
        metadata = {
            "title": title,
            "slug": slug,
            "kind": kind,
            "public_url": public_url,
        }
        if content_sha256:
            metadata["content_sha256"] = content_sha256
        if extra_metadata:
            metadata.update(extra_metadata)

        indexed_content = f"# {title}\n\n"
        if summary:
            indexed_content += f"Summary: {summary.strip()}\n\n"
        indexed_content += content
        chunks = ingest_to_corpus(
            indexed_content,
            source=source,
            layer="self_produced_analysis",
            author="Cyber-Lenin",
            year=now.year,
            extra_metadata=metadata,
            skip_if_source_url_exists=False,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info(
            "[shared] indexed public self analysis: %s/%s (%d chunks, %d deleted)",
            kind,
            slug,
            chunks,
            deleted,
        )
        return {
            "ok": True,
            "source": source,
            "layer": "self_produced_analysis",
            "chunks": chunks,
            "deleted": deleted,
        }
    except Exception as e:
        logger.warning("[shared] index_public_self_analysis error: %s", e)
        return {"ok": False, "error": str(e)}


def delete_public_self_analysis_index(kind: str, slug: str) -> dict:
    """Remove one public artifact from the self_produced_analysis vector layer."""
    try:
        source = public_self_analysis_source(kind, slug)
        deleted = delete_corpus_source(source, layer="self_produced_analysis")
        return {"ok": True, "source": source, "deleted": deleted}
    except Exception as e:
        logger.warning("[shared] delete_public_self_analysis_index error: %s", e)
        return {"ok": False, "error": str(e)}


