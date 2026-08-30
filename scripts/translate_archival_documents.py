#!/usr/bin/env python3
"""CLI for runtime_tools.archival_translation — archival documents → Korean.

Translates the block ranges a spec names (Soviet official documents
reproduced inside a saved archive page) and writes a CommuLingo reference
fragment. Scope rules and the source-drift guards live in the module.

Provider calls go through the LLM gateway proxy (llm_proxy, :8110), which
injects the real key, so no credential or sudo is needed:

  venv/bin/python scripts/translate_archival_documents.py --spec nkvd-1937-documents

The same run is available over the admin API:

  POST /admin/archival-translation/run  {"specId": "nkvd-1937-documents"}

Model, max_tokens and thinking come from the language pair's registry entry
(config/llm_call_sites.json: archival_document_translation_{ru,zh}); there is
no CLI override because the registry would ignore it. Use --compare to try
candidates side by side before changing the entry.

--plan prints the slicing, chunking and cost estimate without calling the
model. Chunks are cached by content hash, so re-running pays only for what
changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_tools import archival_translation as at


def _print_plan(prepared: dict) -> None:
    print(f"문서     : {len(prepared['documents'])}건")
    for d in prepared["documents"]:
        print(f"  - {d['id']}: 블록 {d['blocks']}개, {d['chars']:,}자 — {d['title']}")
    print(f"용어표   : {prepared['glossaryEntries']:,}항목")
    print(f"청크     : {prepared['chunks']}개")
    print(f"번역 대상: {prepared['chars']:,}자")
    think = "추론 on" if prepared.get("thinking") else "추론 off"
    print(f"예상 비용: 약 ${prepared['estimatedUsd']:.2f} ({prepared.get('model')}, {think} 기준)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True,
                    help=f"스펙 id ({at.SPEC_DIR} 안의 파일명, 확장자 제외)")
    ap.add_argument("--out", type=Path, help="출력 fragment 경로 (기본: 스펙의 output)")
    ap.add_argument("--cache", type=Path, help="청크 캐시 JSONL")
    ap.add_argument("--max-chars", type=int, default=3500, dest="max_chars")
    ap.add_argument("--glossary-limit", type=int, default=60, dest="glossary_limit")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit-chunks", type=int, default=0, dest="limit_chunks",
                    help="앞에서 N개 청크만 번역 (연기 테스트용)")
    ap.add_argument("--plan", "--dry-run", action="store_true", dest="plan_only",
                    help="모델을 호출하지 않고 계획만 출력")
    ap.add_argument("--probe", action="store_true",
                    help="provider를 직접 한 번 호출해 응답·finish_reason·usage를 그대로 출력")
    ap.add_argument("--compare",
                    help="같은 청크를 여러 모델로 번역해 비교. "
                         "쉼표 구분 'provider/model' (뒤에 +think, +effort=high 가능)")
    ap.add_argument("--compare-chunks", type=int, default=2, dest="compare_chunks")
    ap.add_argument("--compare-out", type=Path, dest="compare_out",
                    help="비교 결과 마크다운 경로")
    args = ap.parse_args()

    opts = at.Options(
        max_chars=args.max_chars,
        glossary_limit=args.glossary_limit, concurrency=args.concurrency,
        retries=args.retries, limit_chunks=args.limit_chunks,
        cache_path=args.cache, out_path=args.out,
    )

    try:
        spec = at.load_spec(args.spec)
        print(f"스펙     : {spec['id']} — {spec['title']}")
        if args.plan_only:
            _print_plan(at.plan(spec, opts))
            return 0

        if args.compare:
            variants = [v.strip() for v in args.compare.split(",") if v.strip()]
            # 비교 대상은 registry의 provider가 아니라 변형마다 다르다.
            for provider in dict.fromkeys(v.partition("+")[0].partition("/")[0] for v in variants):
                at.preflight(opts, at.language_for(spec), provider=provider)
            report = at.compare(spec, variants, opts, args.compare_chunks)
            out = args.compare_out or Path(f"/tmp/compare-{spec['id']}.md")
            lines = [f"# 번역 모델 비교 — {spec['id']}",
                     f"\n대상 {report['chars']:,}자\n"]
            for idx in sorted(report["source"]):
                src = " ".join(report["source"][idx]["lines"])
                lines.append(f"\n## 블록 {idx} (원문 {len(src)}자)\n\n**RU** {src}\n")
                for r in report["results"]:
                    if r.get("error"):
                        lines.append(f"\n**{r['variant']}** 실패: {r['error']}\n")
                        continue
                    got = " ".join(r["blocks"].get(idx, ["(응답 없음)"]))
                    lines.append(f"\n**{r['variant']}** ({len(got)}자)\n\n{got}\n")
            out.write_text("\n".join(lines), encoding="utf-8")

            print(f"\n{'변형':44} {'초':>6} {'검증 문제':>9}")
            for r in report["results"]:
                if r.get("error"):
                    print(f"  {r['variant']:42} 실패: {r['error'][:50]}")
                    continue
                label = f"{r['variant']} think={(r.get('thinking') or {}).get('type')}"
                print(f"  {label:42} {r['seconds']:6.1f} {len(r['problems']):9}")
            print(f"\n비교 문서: {out}")
            return 0

        if args.probe:
            at.preflight(opts, at.language_for(spec))
            for rec in at.probe(spec, opts):
                print(f"\n[{rec['case']}] {rec['provider']}/{rec['model']} "
                      f"max_tokens={rec['maxTokens']} thinking={rec['thinking']} "
                      f"({rec['seconds']}초)")
                if rec.get("error"):
                    print(f"  예외: {rec['error']}")
                    continue
                print(f"  ok={rec['ok']} content={rec['contentChars']}자")
                if rec["contentChars"]:
                    print(f"  미리보기: {rec['preview'][:120]}")
            return 0

        def progress(event: dict) -> None:
            kind = event.get("event")
            if kind == "plan":
                _print_plan(event)
                print(f"\n번역 시작 (동시 {opts.concurrency})")
            elif kind == "retry":
                print(f"    재시도 {event['attempt']}: {'; '.join(event['problems'])}", flush=True)
            elif kind == "chunk" and (event["done"] % 5 == 0 or event["done"] == event["total"]):
                print(f"  {event['done']}/{event['total']} 청크 완료", flush=True)
            elif kind == "chunkFailed":
                print(f"  실패 청크 {event['blocks'][0]}–{event['blocks'][1]}: {event['error']}",
                      flush=True)
            elif kind == "done":
                print(f"완료: {event['stats']} ({event['seconds']}초)")
                for f in event.get("failures", []):
                    print(f"  실패: 블록 {f['blocks'][0]}–{f['blocks'][1]} — {f['error']}")
                if event.get("output"):
                    print(f"fragment: {event['output']} ({event['bytes']:,} bytes)")
                    stray = event.get("strayCyrillic") or []
                    print(f"미번역 잔여: {', '.join(stray) if stray else '없음'}")
                elif event.get("note"):
                    print(f"({event['note']})")

        at.run(spec, opts, progress)
        return 0
    except at.SpecError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
