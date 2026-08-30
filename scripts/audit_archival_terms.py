#!/usr/bin/env python3
"""사후 용어 감사: 번역이 끝난 사료 문서에서 표기 불일치·용어표 이탈을 LLM으로 찾는다.

번역 루프의 용어표 준수 검사는 표면 일치의 오탐이 재시도로 옳은 번역을
뒤집어서 제거했다(2026-08-30). 이 스크립트는 그 검사의 **보고 전용** 후신이다:
청크 캐시(모델 원출력)를 블록 번호로 원문과 정렬해 LLM에 (원문, 번역) 쌍을
보이고, 항목마다 번역문이 실제 쓴 표기를 뽑는다. 그런 다음 결정론으로 묶어

- 한 항목에 표기가 둘 이상인 것(불일치),
- 용어표 항목의 뜻인데 표기가 다른 것(이탈),
- 스펙 postEdits가 이미 덮는 것과 아직 아닌 것

을 마크다운 보고서로 남기고 postEdits 제안 조각을 붙인다. 재번역은 없고,
자동 수정도 없다 — 채택은 사람이 스펙 postEdits를 고쳐서 하며, 전 청크
캐시 적중이라 모델 호출 없이 재조립된다.

유료 호출(registry feature archival_term_extraction)이므로 --plan으로 견적을
먼저 본다. 추출 결과는 청크 단위로 캐시되어 두 번째 실행은 호출이 없다.

Usage:
    venv/bin/python scripts/audit_archival_terms.py --spec stalin-1925-xiv-congress --plan
    venv/bin/python scripts/audit_archival_terms.py --spec stalin-1925-xiv-congress [--out report.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_tools.archival_translation import Options, SpecError, load_spec
from runtime_tools.archival_translation import terms as at_terms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="스펙 id")
    ap.add_argument("--out", type=Path,
                    help="보고서 마크다운 경로 (기본 output/archival_translations/<id>.terms-audit.md)")
    ap.add_argument("--cache", type=Path, help="번역 청크 캐시 JSONL (기본: 스펙의 것)")
    ap.add_argument("--max-chars", type=int, default=3500, dest="max_chars",
                    help="청크 크기 — 번역 때와 같게 두면 청크 경계가 같다")
    ap.add_argument("--glossary-limit", type=int, default=60, dest="glossary_limit")
    ap.add_argument("--min-count", type=int, default=1, dest="min_count",
                    help="이 횟수 미만 등장한 항목은 보고에서 뺀다")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--plan", action="store_true", help="청크 수·예상 비용만 출력, 호출 없음")
    args = ap.parse_args()

    opts = Options(max_chars=args.max_chars, glossary_limit=args.glossary_limit,
                   cache_path=args.cache)
    try:
        spec = load_spec(args.spec)
        print(f"스펙     : {spec['id']} — {spec['title']}")
        if args.plan:
            est = at_terms.estimate(spec, opts, "post")
            print(f"청크     : {est['chunks']}개 ({est['chars']:,}자, 원문+번역 함께 전송)")
            print(f"캐시     : 추출 레코드 {est['cachedRecords']}개 (적중분은 호출 없음)")
            print(f"모델     : {est['provider']}/{est['model']}")
            print(f"예상 비용: 약 ${est['estimatedUsd']:.3f}")
            return 0

        def progress(event: dict) -> None:
            kind = event.get("event")
            if kind == "chunk" and (event["done"] % 10 == 0 or event["done"] == event["total"]):
                print(f"  {event['done']}/{event['total']} 청크", flush=True)
            elif kind in ("extractRetry", "extractFailed"):
                print(f"  {kind} {event['blocks'][0]}–{event['blocks'][1]}: {event['error']}",
                      flush=True)

        result = at_terms.post_audit(spec, opts, min_count=args.min_count,
                                     emit=progress, concurrency=args.concurrency)
        out = args.out or (at_terms.CACHE_DIR / f"{spec['id']}.terms-audit.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result["markdown"], encoding="utf-8")

        print(f"\n청크 {result['chunks']}개 (캐시 {result['cachedChunks']}, "
              f"실패 {len(result['failedChunks'])}), 번역 미발견 블록 {len(result['missingBlocks'])}개")
        print(f"표기 불일치 {len(result['inconsistent'])}건, 용어표 이탈 {len(result['deviations'])}건, "
              f"postEdits 제안 {len(result['postEditsSnippet'])}건")
        for item in result["inconsistent"][:30]:
            variants = ", ".join(
                f"{v['target']}×{len(v['blocks'])}" + ("(postEdits)" if v["covered"] and not v["remaining"] else "")
                for v in item["variants"])
            print(f"  {item['lemma']:<24} {item['majority']}×{len(item['majorityBlocks'])} vs {variants}")
        for d in result["deviations"][:30]:
            status = "postEdits" if d["covered"] and not d["remaining"] else "미처리"
            print(f"  이탈 {d['lemma']:<20} 기대 {d['expected']} / 실제 {d['target']} "
                  f"[{len(d['blocks'])}블록, {status}]")
        print(f"보고서: {out}")
        return 0
    except (SpecError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
