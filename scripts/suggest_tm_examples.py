#!/usr/bin/env python3
"""새 스펙에 고정할 참고 번역례를 TM에서 추천한다.

유사 세그먼트 예시 주입(인수인계 §2.3)의 추천 단계다. 예시를 실행 시점에
동적으로 뽑아 프롬프트에 넣으면 TM이 자랄 때마다 캐시 키가 바뀌어 postEdits
수정 하나에 문서 전체를 재번역하게 되므로, 예시는 스펙의 ``tmExamples``에
사람이 고정한다. 이 스크립트는 그 후보를 고르는 일만 한다: 검수 등급
(published/reviewed) 세그먼트 중 스펙 원문과 어휘가 가장 많이 겹치는 것을
점수순으로 보여주고, 붙여넣을 수 있는 JSON 조각을 출력한다.

유사도는 결정론적 어휘 겹침(자카드)이다 — 러시아어는 소문자화한 낱말 집합,
중국어는 한자 바이그램 집합. 임베딩이 아니라서 의미 유사는 못 잡지만, 이
용도(같은 관청 문체·같은 상투구의 선례 찾기)에는 표면 겹침이 오히려 신호다.

Usage:
    python scripts/suggest_tm_examples.py --spec new-spec-id
    python scripts/suggest_tm_examples.py --spec new-spec-id --limit 3 --min-score 0.3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_tools import translation_memory
from runtime_tools.archival_translation import Options, load_spec, plan

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]{3,}")
_HAN_RE = re.compile(r"[㐀-䶿一-鿿]")


def _tokens(text: str, lang_code: str) -> set[str]:
    if lang_code == "zh":
        chars = _HAN_RE.findall(text)
        return {a + b for a, b in zip(chars, chars[1:])} or set(chars)
    return set(w.lower() for w in _WORD_RE.findall(text))


def rank_examples(
    block_texts: list[str],
    segments: list[tuple[str, str, str]],
    lang_code: str,
    *,
    limit: int = 5,
    min_score: float = 0.25,
    min_len: int = 40,
    max_len: int = 400,
) -> list[dict]:
    """스펙 블록들과 가장 겹치는 TM 세그먼트 상위 limit개.

    완전 일치(점수 1.0의 동일 원문)는 뺀다 — 그 블록은 예시가 아니라
    프리필(_tm_prefill)의 몫이다. 너무 짧은 세그먼트는 상투구로서 정보가
    없고, 너무 긴 것은 프롬프트 예산을 잡아먹으므로 길이 창으로 거른다.
    """
    if lang_code == "zh":
        # 한자는 러시아어보다 글자당 정보가 조밀하다 — 같은 길이 창을 쓰면
        # 실질 한 문장짜리 세그먼트가 전부 길이 미달로 빠진다.
        min_len, max_len = min_len // 2, max_len // 2
    block_sets = [(_tokens(b, lang_code), b.strip()) for b in block_texts]
    block_sets = [(toks, text) for toks, text in block_sets if toks]
    scored: dict[str, dict] = {}
    for source, target, status in segments:
        if not (min_len <= len(source) <= max_len):
            continue
        stoks = _tokens(source, lang_code)
        if not stoks:
            continue
        best = 0.0
        for btoks, btext in block_sets:
            if source == btext:
                best = 0.0
                break  # 완전 일치 — 프리필 영역
            score = len(stoks & btoks) / len(stoks | btoks)
            best = max(best, score)
        if best >= min_score:
            prev = scored.get(source)
            if prev is None or best > prev["score"]:
                scored[source] = {"score": round(best, 3), "source": source,
                                  "target": target, "status": status}
    ranked = sorted(scored.values(), key=lambda x: -x["score"])
    return ranked[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Suggest tmExamples for a spec from reviewed TM segments.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.25)
    args = parser.parse_args()

    spec = load_spec(args.spec)
    prepared = plan(spec, Options())
    lang = prepared["_lang"]
    block_texts = ["\n".join(b["lines"]) for d in prepared["_docs"] for b in d["blocks"]
                   if any(ln.strip() for ln in b["lines"])]
    segments = translation_memory.list_segments(
        lang_pair=f"{lang.code}-ko", statuses=("published", "reviewed"))
    if not segments:
        print("검수 등급(published/reviewed) 세그먼트가 없다. 백필을 먼저 돌리거나 "
              "스펙을 frozen 처리한 뒤 다시 시도할 것.", file=sys.stderr)
        return 1

    ranked = rank_examples(block_texts, segments, lang.code,
                           limit=args.limit, min_score=args.min_score)
    if not ranked:
        print(f"점수 {args.min_score} 이상인 후보가 없다 (검수 세그먼트 {len(segments)}건 대조).")
        return 0

    print(f"# {args.spec} — 참고 번역례 후보 {len(ranked)}건 "
          f"(검수 세그먼트 {len(segments)}건 대조)\n")
    for c in ranked:
        print(f"  {c['score']:.2f} [{c['status']}] {c['source'][:70]}")
        print(f"        → {c['target'][:70]}\n")
    snippet = json.dumps(
        [{"source": c["source"], "target": c["target"]} for c in ranked],
        ensure_ascii=False, indent=2)
    print("채택할 항목만 남겨 스펙에 붙여넣을 것 (tmExamples 추가·수정은 해당 문서의")
    print("청크 캐시를 의도적으로 무효화한다 — frozen 스펙에는 넣지 말 것):\n")
    print(f'"tmExamples": {snippet}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
