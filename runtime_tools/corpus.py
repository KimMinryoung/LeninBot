"""Bounded corpus passage retrieval by search-result UUID."""
import asyncio
from tool_gateway.results import ToolFailure, ToolRejection

READ_CORPUS_PASSAGE_TOOL = {
    "name": "read_corpus_passage",
    "description": "Read adjacent original corpus chunks by UUID from vector_search, within the same document/layer. Missing or ambiguous context is reported explicitly.",
    "input_schema": {"type": "object", "additionalProperties": False,
        "properties": {
            "chunk_id": {"type": "string", "format": "uuid"},
            "window": {"type": "integer", "minimum": 0, "maximum": 3, "default": 1},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": 20000, "default": 20000}},
        "required": ["chunk_id"]}}

async def read_corpus_passage(chunk_id: str, window: int = 1, max_chars: int = 20000) -> str:
    from corpus.store import fetch_corpus_source_context
    try:
        return await asyncio.to_thread(fetch_corpus_source_context, "", chunk_id=chunk_id,
                                       window=window, max_chars=max_chars)
    except ToolRejection:
        raise
    except Exception as exc:
        return ToolFailure(f"Corpus passage read failed: {exc}", {"failure_type": type(exc).__name__})
