#!/usr/bin/env python3
"""Smoke test for provider-facing tool description compaction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_loop_common import compact_tool_definitions


def main() -> int:
    long_desc = " ".join(["description"] * 200)
    tools = [
        {
            "name": "sample_tool",
            "description": long_desc,
            "input_schema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["a", "b"],
                        "default": "a",
                        "description": long_desc,
                    },
                    "count": {"type": "integer", "description": "Short enough."},
                },
                "required": ["mode"],
            },
        }
    ]
    compacted = compact_tool_definitions(tools)
    original = tools[0]
    compact = compacted[0]

    assert original["description"] == long_desc
    assert compact["name"] == "sample_tool"
    assert compact["input_schema"]["type"] == "object"
    assert compact["input_schema"]["required"] == ["mode"]
    mode = compact["input_schema"]["properties"]["mode"]
    assert mode["type"] == "string"
    assert mode["enum"] == ["a", "b"]
    assert mode["default"] == "a"
    assert len(compact["description"]) < len(long_desc)
    assert len(mode["description"]) < len(long_desc)
    assert compact["input_schema"]["properties"]["count"]["description"] == "Short enough."
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
