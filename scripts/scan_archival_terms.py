#!/usr/bin/env python3
"""PREPARE scan: 번역 전에 스펙 원문에서 용어집 미등재 후보를 뽑는다.

새 문서의 낯선 인명·기관 약어는 지금까지 사람이 원문을 통독하며 스펙의
glossary.extra에 손으로 채웠다. 첫 등장 표기가 문서 전체의 일관성을 정하므로
이 단계를 빼먹으면 모델이 청크마다 다른 음차를 지어낸다(НКВД가
'인민내무위원부'로 나온 사례). 이 스크립트는 통독을 대신하지 않는다 — 후보와
등장 횟수, 첫 문맥을 추려 검토 목록으로 만들 뿐이고, 채택 여부와 한국어 표기는
사람이 정해서 glossary.extra에 넣는다.

러시아어 전용이다. 후보는 두 갈래로 뽑는다: 전부 대문자인 약어(НКВД, ЦК),
그리고 문장 중간에서 대문자로 시작하는 낱말(인명·지명 — 문장 첫 낱말은
대문자가 정보가 아니므로 제외). 중국어는 대소문자 신호가 없어 이 방법이
통하지 않는다 — NER 없이 돌리면 목록 전체가 소음이 되므로 거부한다.

Usage:
    python scripts/scan_archival_terms.py --spec new-spec-id
    python scripts/scan_archival_terms.py --spec a --spec b --min-count 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_tools.archival_translation import Options, load_spec, plan

# 전부 대문자인 약어. ЦК ВКП(б) 같은 괄호 꼬리는 뒤에서 별도로 붙지 않고
# 낱개 토큰(ЦК, ВКП)으로 잡혀도 검토 목록으로는 충분하다.
_ABBREV_RE = re.compile(r"(?<![А-Яа-яЁё])[А-ЯЁ]{2,10}(?![А-Яа-яЁё])")
# 대문자로 시작하는 낱말 (4자 이상 — И, Он 같은 짧은 것은 소음)
_CAPWORD_RE = re.compile(r"(?<![А-Яа-яЁё])[А-ЯЁ][а-яё]{3,}(?![А-Яа-яЁё])")
# 이 문자 뒤의 대문자는 문장 시작일 가능성이 높아 증거로 치지 않는다
_SENTENCE_END = ".!?…»\"”)("


def _mid_sentence(line: str, start: int) -> bool:
    head = line[:start].rstrip()
    if not head:
        return False
    return head[-1] not in _SENTENCE_END


def scan_candidates(docs: list[dict], glossary: list[dict],
                    *, min_count: int = 2) -> list[dict]:
    """용어집이 아직 모르는 약어·대문자 낱말 후보와 등장 횟수, 첫 문맥."""
    counts: Counter[str] = Counter()
    contexts: dict[str, str] = {}

    def _note(surface: str, line: str) -> None:
        counts[surface] += 1
        if surface not in contexts:
            contexts[surface] = line.strip()[:120]

    for doc in docs:
        for block in doc["blocks"]:
            for line in block["lines"]:
                for m in _ABBREV_RE.finditer(line):
                    _note(m.group(0), line)
                for m in _CAPWORD_RE.finditer(line):
                    # 문장 첫 낱말의 대문자는 고유명사 증거가 아니다. 문장
                    # 중간 등장이 한 번이라도 있어야 후보로 남는다 — 그래서
                    # 여기서는 문장 중간일 때만 센다.
                    if _mid_sentence(line, m.start()):
                        _note(m.group(0), line)

    out = []
    for surface, count in counts.most_common():
        if count < min_count:
            continue
        # 이미 용어집 패턴이 잡는 표면이면 등재된 항목이다 (곡용 변형 포함)
        if any(g["pattern"].search(surface) for g in glossary):
            continue
        out.append({"surface": surface, "count": count, "context": contexts[surface]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan spec sources for glossary candidates before translating.")
    parser.add_argument("--spec", action="append", required=True, help="Spec id. Repeatable.")
    parser.add_argument("--min-count", type=int, default=2,
                        help="이 횟수 미만 등장한 후보는 버린다 (기본 2 — 한 번뿐인 표면은 대개 소음)")
    args = parser.parse_args()

    failures = 0
    for spec_id in args.spec:
        try:
            spec = load_spec(spec_id)
            prepared = plan(spec, Options())
            lang = prepared["_lang"]
            if lang.code != "ru":
                print(f"skip {spec_id}: {lang.label}는 대소문자 신호가 없어 이 스캔이 "
                      "통하지 않는다 (NER 도입 전까지 수동 통독)", file=sys.stderr)
                continue
            candidates = scan_candidates(
                prepared["_docs"], prepared["_glossary"], min_count=args.min_count)
            print(f"\n# {spec_id} — 용어집 미등재 후보 {len(candidates)}건 "
                  f"(용어집 {prepared['glossaryEntries']}항목과 대조)")
            for c in candidates:
                print(f"  {c['count']:>3}× {c['surface']:<24} … {c['context']}")
            if candidates:
                print("  → 채택할 항목은 한국어 표기를 정해 스펙의 glossary.extra에 넣을 것.")
        except Exception as exc:
            failures += 1
            print(f"failed {spec_id}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
