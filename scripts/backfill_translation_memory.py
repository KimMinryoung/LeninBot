#!/usr/bin/env python3
"""Backfill the translation memory from existing archival chunk caches.

TM(runtime_tools/translation_memory.py)은 사료 파이프라인이 청크를 번역할 때
적재되지만, TM보다 먼저 번역된 문서들의 쌍은 청크 캐시(JSONL)에만 있다.

정렬은 캐시 키 재계산이 아니라 **블록 번호**로 한다. 처음 구현은 스펙을 다시
계획해 현재 프롬프트·옵션으로 키를 만들어 캐시를 찾았는데, 캐시 키에는 시스템
프롬프트 해시와 청크 옵션이 들어가므로 그중 하나라도 바뀌면 전부 miss가 된다
— 실제로 번역투 규칙을 프롬프트에 추가하자 첫 서버 실행이 0건으로 끝났다.
캐시 레코드의 blocks는 스펙 전역 블록 번호로 저장되므로, 번호로 원문과 직접
정렬하면 프롬프트·모델·청크 크기가 몇 번을 바뀌었든 복원된다. 파일은
append-only라 같은 블록이 여러 세대 있으면 뒤(최신) 레코드가 이긴다.
스펙의 sha256·경계 가드는 plan()이 그대로 집행하므로, 원문이 움직였으면
정렬 전에 요란하게 실패한다.

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
    _cache_path,
    apply_post_edits,
)


def align_cached_blocks(
    source_by_idx: dict[int, str], cache_lines: list[str], spec: dict
) -> tuple[list[tuple[str, str]], list[int]]:
    """캐시 JSONL 줄들을 블록 번호로 원문과 정렬해 (원문, 번역) 쌍을 만든다."""
    target_by_idx: dict[int, list[str]] = {}
    for line in cache_lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        for key, lines in (rec.get("blocks") or {}).items():
            if lines:
                target_by_idx[int(key)] = lines
    pairs: list[tuple[str, str]] = []
    block_ids: list[int] = []
    for idx, target_lines in sorted(target_by_idx.items()):
        source = source_by_idx.get(idx)
        if not source:
            continue
        # 캐시에는 모델 원출력이 있고, 사람이 postEdits로 고친 결과는 조립
        # 단계에만 있었다. 같은 치환을 적용해야 발행본과 같은 텍스트가 남는다.
        pairs.append((source, "\n".join(apply_post_edits(target_lines, spec))))
        block_ids.append(idx)
    return pairs, block_ids


def backfill_spec(spec_id: str) -> dict:
    spec = load_spec(spec_id)
    prepared = plan(spec, Options())
    docs, lang = prepared["_docs"], prepared["_lang"]

    source_by_idx: dict[int, str] = {}
    for doc in docs:
        for i, block in enumerate(doc["blocks"]):
            text = "\n".join(block["lines"])
            if text.strip():
                source_by_idx[doc["offset"] + i] = text

    cache_path = _cache_path(spec, None)
    cache_lines = (
        cache_path.read_text(encoding="utf-8").splitlines() if cache_path.is_file() else []
    )
    pairs, block_ids = align_cached_blocks(source_by_idx, cache_lines, spec)

    # frozen 스펙은 발행 전 사람이 통독한 문서다. 세그먼트 하나하나를 확인한
    # 것은 아니므로 reviewed가 아니라 published — 문서 단위 검수라는 사실
    # 그대로를 상태로 남긴다.
    status = "published" if spec.get("frozen") else "machine"
    inserted = translation_memory.record_segments(
        pairs, lang_pair=f"{lang.code}-ko", doc_id=spec_id, block_ids=block_ids, status=status
    )
    return {
        "spec": spec_id,
        "sourceBlocks": len(source_by_idx),
        "cachedBlocks": len(pairs),
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
