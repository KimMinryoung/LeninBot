#!/usr/bin/env python3
"""Smoke checks for filesystem tool pagination behavior."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _run() -> None:
    from runtime_tools.filesystem import _exec_read_file

    converted_dir = ROOT / "data" / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = converted_dir / ".smoke_read_file_offset.md"
    text = "\n".join(f"line {i}" for i in range(1, 11)) + "\n" + ("tail " * 200)
    smoke_path.write_text(text, encoding="utf-8")
    try:
        line_page = await _exec_read_file(str(smoke_path), offset=4, limit=2)
        assert "lines 4-5" in line_page
        assert "line 4" in line_page
        assert "line 3" not in line_page

        char_page = await _exec_read_file(str(smoke_path), char_offset=15, char_limit=20)
        assert "chars 15-35" in char_page
        assert "(next: char_offset=35)" in char_page

        beyond_line_page = await _exec_read_file(str(smoke_path), offset=50, limit=2)
        assert "offset is a 1-indexed line number" in beyond_line_page
        assert "char_offset and char_limit explicitly" in beyond_line_page

        beyond_page = await _exec_read_file(str(smoke_path), char_offset=len(text) + 1, char_limit=20)
        assert "char_offset is beyond end of file" in beyond_page
    finally:
        smoke_path.unlink(missing_ok=True)


def main() -> int:
    asyncio.run(_run())
    print("filesystem tools smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
