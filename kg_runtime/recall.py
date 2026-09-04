"""Entity-gated KG recall for prompt injection.

Cheap by construction: the text is matched against the in-process alias
index (no embedding, no LLM); only when it names a known entity do we pull
that entity's top facts from Neo4j and render a small context block in the
same shape as ``memory_store.experiential.recall_experiences_block``.

Off by default — enable with ``KG_ENTITY_GATED_RECALL=1``. Returns "" when
disabled, when nothing matches, or on any failure.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.getenv("KG_ENTITY_GATED_RECALL", "0").strip().lower() in ("1", "true", "yes", "on")


def entity_gated_kg_block(text: str, provider: str = "claude", *, max_entities: int = 2,
                          max_facts: int = 6) -> str:
    if not enabled() or not text or not text.strip():
        return ""
    try:
        from kg_runtime.search import _alias_hits, _entity_neighborhood, _format_edge_line

        hits = _alias_hits(text[:2000], limit=max_entities, broad=False)
        if not hits:
            return ""
        lines: list[str] = []
        for hit in hits[:max_entities]:
            node, edges = _entity_neighborhood(hit.uuid, cap=max_facts * 3)
            if not node:
                continue
            active = [e for e in edges if not e.get("expired_at")] or edges
            # "Document X mentions it" edges are bookkeeping, not knowledge:
            # they neither qualify a node for recall nor appear in the block.
            active = [e for e in active if e.get("predicate") != "Reference"]
            if not active:
                continue
            summary = (node.get("summary") or "").strip()
            if len(summary) > 160:
                summary = summary[:160].rstrip() + "…"
            head = f"- {node.get('name')}"
            if summary:
                head += f": {summary}"
            lines.append(head)
            for e in active[:max_facts]:
                lines.append("  " + _format_edge_line(e))
        if not lines:
            return ""
        body = "\n".join(lines)
        injected = [ln[2:].split(":", 1)[0] for ln in lines if not ln.startswith("  ")]
        logger.info("[KG recall] injected %d entity(ies): %s", len(injected), ", ".join(injected))
        if (provider or "claude") == "claude":
            return (
                "<knowledge-graph>\n"
                f"{body}\n"
                "위 사실은 지식 그래프에서 이름이 일치해 자동 회수된 것이다. 근거로 쓰되, "
                "더 필요하면 knowledge_graph_search로 확인해라.\n"
                "</knowledge-graph>"
            )
        return (
            "### Knowledge Graph\n"
            f"{body}\n"
            "Auto-recalled by entity name match; verify with knowledge_graph_search when it matters."
        )
    except Exception as exc:
        logger.debug("[KG recall] skipped: %s", exc)
        return ""
