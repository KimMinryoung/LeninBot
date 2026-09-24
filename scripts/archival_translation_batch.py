#!/usr/bin/env python3
"""Submit, inspect, or collect an archival Gemini translation batch.

Usage: venv/bin/python scripts/archival_translation_batch.py submit --spec <id>
       venv/bin/python scripts/archival_translation_batch.py status --spec <id>
       venv/bin/python scripts/archival_translation_batch.py collect --spec <id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_tools.archival_translation import Options, SpecError, load_spec
from runtime_tools.archival_translation import batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "status", "collect"))
    parser.add_argument("--spec", required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--max-chars", type=int, default=3500)
    parser.add_argument("--glossary-limit", type=int, default=60)
    parser.add_argument("--new-batch", action="store_true",
                        help="완료된 이전 배치를 보관하고 남은 청크를 새로 제출")
    parser.add_argument("--max-chunks", type=int,
                        help="submit에서 한 번에 제출할 미캐시 청크 수 제한")
    args = parser.parse_args()
    if args.new_batch and args.action != "submit":
        parser.error("--new-batch는 submit에만 쓸 수 있다")
    if args.max_chunks is not None and args.action != "submit":
        parser.error("--max-chunks는 submit에만 쓸 수 있다")
    opts = Options(cache_path=args.cache, max_chars=args.max_chars,
                   glossary_limit=args.glossary_limit)
    try:
        spec = load_spec(args.spec)
        result = (batch.submit(spec, opts, new_batch=args.new_batch, max_chunks=args.max_chunks) if args.action == "submit"
                  else getattr(batch, args.action)(spec, opts))
    except SpecError as exc:
        print(f"배치 오류: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(bool(result.get("failed") or result.get("error")))


if __name__ == "__main__":
    raise SystemExit(main())
