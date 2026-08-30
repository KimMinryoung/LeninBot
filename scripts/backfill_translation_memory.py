#!/usr/bin/env python3
"""Backfill the translation memory from existing archival chunk caches.

TM(runtime_tools/translation_memory.py)은 사료 파이프라인이 청크를 번역할 때
적재되지만, TM보다 먼저 번역된 문서들의 쌍은 청크 캐시(JSONL)에만 있다.
캐시 레코드는 해시 키와 출력 블록만 담고 원문을 담지 않으므로, 원문은 스펙을
다시 계획(plan)해서 복원한다 — run()의 완전 캐시 경로와 같은 일을 API 호출
없이 하는 셈이다. 자격증명이 필요 없고, 캐시에 없는 청크는 건너뛴다.

Usage:
    python scripts/backfill_translation_memory.py            # every spec
    python scripts/backfill_translation_memory.py --spec nine-commentaries
    python scripts/backfill_translation_memory.py --stats    # counts only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_tools import translation_memory
from runtime_tools.archival_translation import (
    Options,
    plan,
    load_spec,
    list_specs,
)
from runtime_tools.archival_translation.core import (
    Cache,
    _cache_path,
    _chunk_key,
    _chunk_prompt,
    apply_post_edits,
)


def backfill_spec(spec_id: str) -> dict:
    spec = load_spec(spec_id)
    prepared = plan(spec, Options())
    glossary, chunks, lang = prepared["_glossary"], prepared["_chunks"], prepared["_lang"]
    cache = Cache(_cache_path(spec, None))

    opts = Options()
    pairs: list[tuple[str, str]] = []
    block_ids: list[int] = []
    misses = 0
    for chunk in chunks:
        key = _chunk_key(_chunk_prompt(chunk, glossary, opts), opts, lang)
        rec = cache.get(key)
        if rec is None:
            misses += 1
            continue
        blocks = {int(k): v for k, v in rec["blocks"].items()}
        for idx, block in chunk:
            lines = blocks.get(idx)
            if not lines:
                continue
            # 캐시에는 모델 원출력이 있고, 사람이 postEdits로 고친 결과는 조립
            # 단계에만 있었다. 같은 치환을 적용해야 발행본과 같은 텍스트가
            # TM에 남는다.
            pairs.append(("\n".join(block["lines"]), "\n".join(apply_post_edits(lines, spec))))
            block_ids.append(idx)

    # frozen 스펙은 발행 전 사람이 통독한 문서다. 세그먼트 하나하나를 확인한
    # 것은 아니므로 reviewed가 아니라 published — 문서 단위 검수라는 사실
    # 그대로를 상태로 남긴다.
    status = "published" if spec.get("frozen") else "machine"
    inserted = translation_memory.record_segments(
        pairs, lang_pair=f"{lang.code}-ko", doc_id=spec_id, block_ids=block_ids, status=status
    )
    return {
        "spec": spec_id,
        "chunks": len(chunks),
        "cacheMisses": misses,
        "segments": len(pairs),
        "inserted": inserted,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Load archival chunk-cache pairs into the translation memory.")
    parser.add_argument("--spec", action="append", default=[], help="Spec id. Repeatable; default is every spec.")
    parser.add_argument("--stats", action="store_true", help="Print TM stats and exit.")
    args = parser.parse_args()

    if args.stats:
        print(json.dumps(translation_memory.stats(), ensure_ascii=False, indent=2))
        return 0

    spec_ids = args.spec or [entry["id"] for entry in list_specs()]
    failures = 0
    for spec_id in spec_ids:
        try:
            result = backfill_spec(spec_id)
        except Exception as exc:
            # 스펙 하나의 소스 드리프트(sha256 불일치 등)가 나머지 백필을
            # 막아서는 안 된다. 그 스펙은 어차피 재실행 전에 손봐야 한다.
            failures += 1
            print(f"failed {spec_id}: {exc}", file=sys.stderr)
            continue
        print(json.dumps(result, ensure_ascii=False))
    print(json.dumps({"tm": translation_memory.stats()}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
