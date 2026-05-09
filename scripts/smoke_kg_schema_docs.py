#!/usr/bin/env python3
"""Smoke check that KG schema docs mention every registered typed schema item."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from graph_memory.config import EDGE_TYPE_MAP
    from graph_memory.edges import EDGE_TYPES
    from graph_memory.entities import ENTITY_TYPES

    schema_doc = (ROOT / "dev_docs" / "knowledge_graph_schema.md").read_text(encoding="utf-8")
    design_doc = (ROOT / "dev_docs" / "knowledge_graph_design.md").read_text(encoding="utf-8")
    docs = schema_doc + "\n" + design_doc

    missing_entities = sorted(name for name in ENTITY_TYPES if f"`{name}`" not in docs)
    missing_edges = sorted(name for name in EDGE_TYPES if f"`{name}`" not in docs)
    assert not missing_entities, f"KG entity types missing from docs: {missing_entities}"
    assert not missing_edges, f"KG edge types missing from docs: {missing_edges}"

    mapped_edges = {
        edge
        for edge_names in EDGE_TYPE_MAP.values()
        for edge in edge_names
    }
    unknown_mapped_edges = sorted(mapped_edges - set(EDGE_TYPES))
    assert not unknown_mapped_edges, f"EDGE_TYPE_MAP references unknown edge types: {unknown_mapped_edges}"

    mentioned_fallback = "`Entity` fallback" in schema_doc and "`Statement`" in schema_doc and "`Causation`" in schema_doc
    assert mentioned_fallback, "KG schema docs must mention Statement/Causation fallback behavior"

    print("kg schema docs smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
