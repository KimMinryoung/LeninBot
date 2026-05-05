"""Shared embedding client singleton."""

_exp_embeddings = None


def set_shared_embeddings(emb):
    """Allow modules that already have BGE-M3 loaded (e.g. embedding_server.py) to share it."""
    global _exp_embeddings
    _exp_embeddings = emb


def _get_exp_embeddings():
    global _exp_embeddings
    if _exp_embeddings is None:
        from llm.embedding_client import get_embedding_client
        _exp_embeddings = get_embedding_client()
    return _exp_embeddings


