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


def _entity_gated_kg_block(text: str, provider: str = "claude", *, max_entities: int = 2,
                          max_facts: int = 6, _outcome=None) -> str:
    if not enabled() or not text or not text.strip():
        return ""
    try:
        from kg_runtime.search import _alias_hits, _entity_neighborhood, _format_edge_line

        hits = _alias_hits(text[:2000], limit=max_entities, broad=False, strict=True)
        if not hits:
            return ""
        lines: list[str] = []
        for hit in hits[:max_entities]:
            node, edges = _entity_neighborhood(hit.uuid, cap=max_facts * 3)
            if not node:
                continue
            active = [e for e in edges if not e.get("expired_at")]
            # "Document X mentions it" edges are bookkeeping, not knowledge:
            # they neither qualify a node for recall nor appear in the block.
            active = [e for e in active if e.get("predicate") != "Reference"]
            if not active:
                continue
            summary = (node.get("summary") or "").strip()
            if len(summary) > 160:
                summary = summary[:160].rstrip() + "…"
            head = f"- {node.get('name')}"
            if node.get("summary_source"):
                head += f" (summary src: {node['summary_source']})"
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
                "위 기록은 이름이 일치해 자동 회수된 저장 자료이며, 현재 사실의 보증이 아니다. "
                "원문의 출처·유효 기간과 자체 분석 여부를 구분하고, "
                "더 필요하면 knowledge_graph_search로 확인해라.\n"
                "</knowledge-graph>"
            )
        return (
            "### Knowledge Graph\n"
            f"{body}\n"
            "Stored records recalled by name match, not guaranteed current facts. "
            "Observe validity and source; internal analysis is not independent corroboration. "
            "Verify with knowledge_graph_search when it matters."
        )
    except Exception as exc:
        if _outcome is not None:
            _outcome["failed"] = True
        logger.debug("[KG recall] skipped: %s", exc)
        return ""


def entity_gated_kg_block(text: str, provider: str = 'claude', *, max_entities: int = 2,
                          max_facts: int = 6) -> str:
    if not enabled() or not text or not text.strip():
        return ''
    import time
    from kg_runtime.read_policy import ReadPolicy, _current
    token = _current.set(ReadPolicy())
    started = time.monotonic()
    outcome = {'empty': True, 'injected': False, 'failed': False}
    try:
        result = _entity_gated_kg_block(text, provider, max_entities=max_entities, max_facts=max_facts,
                                         _outcome=outcome)
        outcome.update(empty=not bool(result), injected=bool(result))
        return result
    finally:
        _current.reset(token)
        _audit_recall(outcome, round((time.monotonic() - started) * 1000))


def _audit_recall(outcome, latency_ms):
    from security_gateway.audit import audit
    from security_gateway.context import get_caller
    from security_gateway.gateway import Decision
    audit(get_caller(), 'kg_entity_recall', {},
          Decision(True, 'allow', 'read', '', 'observe', 'automatic_recall'),
          result_status='error' if outcome['failed'] else 'ok',
          latency_ms=latency_ms, result_metadata=outcome)
