#!/usr/bin/env python3
"""Offline smoke test for runtime_tools.archival_translation.

Exercises everything except the API call: spec loading and its id guard,
source slicing and its drift guards, chunking, marker round-trip, the
validator's failure modes, and fragment assembly. Run before spending tokens.

  venv/bin/python scripts/smoke_archival_translation.py --spec nkvd-1937-documents
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

from runtime_tools import archival_translation as at

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else ' — ' + detail}")
    if not cond:
        failures.append(name)


def expect_spec_error(name: str, fn) -> None:
    try:
        fn()
        check(name, False, "SpecError가 나지 않았다")
    except at.SpecError:
        check(name, True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="nkvd-1937-documents")
    args = ap.parse_args()

    print("spec loading")
    spec = at.load_spec(args.spec)
    check("스펙을 id로 불러온다", spec["id"] == args.spec)
    check("스펙 목록에 나온다", any(s["id"] == args.spec for s in at.list_specs()))
    expect_spec_error("경로 주입을 막는다", lambda: at.spec_path("../../etc/passwd"))
    expect_spec_error("대문자·공백 id를 막는다", lambda: at.spec_path("Not A Slug"))
    expect_spec_error("없는 스펙을 막는다", lambda: at.load_spec("nope-does-not-exist"))

    print("adapters")
    check("militera 어댑터가 등록되어 있다", "militera" in at.ADAPTERS)
    check("기본 어댑터가 잡힌다", at.get_adapter(None) is at.ADAPTERS["militera"])
    try:
        at.get_adapter("mystery-format")
        check("모르는 포맷을 막는다", False, "KeyError가 나지 않았다")
    except KeyError:
        check("모르는 포맷을 막는다", True)

    print("source slicing")
    blocks = at.extract_blocks(spec)
    docs = at.slice_documents(blocks, spec)
    check("문서가 스펙 수만큼 잘린다", len(docs) == len(spec["documents"]))
    check("블록이 비지 않는다", all(d["blocks"] for d in docs))
    check("블록 범위가 겹치지 않는다",
          len({d["offset"] + i for d in docs for i in range(len(d["blocks"]))})
          == sum(len(d["blocks"]) for d in docs))

    drifted = json.loads(json.dumps(spec))
    drifted["documents"][0]["startsWith"] = "이 문자열은 원문에 없다"
    expect_spec_error("경계가 어긋나면 실패한다", lambda: at.slice_documents(blocks, drifted))
    tampered = json.loads(json.dumps(spec))
    tampered["source"]["sha256"] = "0" * 64
    expect_spec_error("원본이 바뀌면 실패한다", lambda: at.extract_blocks(tampered))

    print("chunking")
    chunks = [c for d in docs for c in at.chunk_document(d, 3500)]
    flat = [idx for c in chunks for idx, _ in c]
    check("모든 블록이 정확히 한 번씩 청크에 들어간다",
          len(flat) == len(set(flat)) == sum(len(d["blocks"]) for d in docs))
    check("청크가 상한을 넘지 않는다(단일 블록 초과분 제외)",
          all(sum(len(l) for _, b in c for l in b["lines"]) <= 3500 or len(c) == 1
              for c in chunks))
    check("제목이 청크 끝에 고아로 남지 않는다",
          all(c[-1][1]["tag"] not in ("h3", "h5") or len(c) == 1 for c in chunks))

    print("glossary")
    glossary = at.build_glossary(Path(spec["glossary"]["people"]),
                                 Path(spec["glossary"]["terms"]))
    by_ru = {g["ru"]: g for g in glossary}
    # Surname-prefix and derived-word collisions a substring match would make.
    for key, trap in [("Кулик", "Куликова"), ("Марков", "Марковский"),
                      ("Томский", "Томске"), ("Вознесенский", "Вознесенске"),
                      ("совет", "антисоветской")]:
        if key in by_ru:
            check(f"{key}가 {trap}에 걸리지 않는다", not by_ru[key]["pattern"].search(trap))
    if "Ежов" in by_ru:
        check("Ежов가 격변화형에 걸린다", bool(by_ru["Ежов"]["pattern"].search("приказ Ежова")))

    print("marker round-trip")
    sample = chunks[0]
    rendered = at.render_chunk(sample)
    check("마커 개수가 블록 수와 같다", len(at.MARKER_RE.findall(rendered)) == len(sample))

    # A well-formed stub response: same markers, same line counts, Korean text
    # roughly as long as the source line — a fixed-length stub would trip the
    # validator's short-translation rule, which is that rule working correctly.
    def _ko(line: str) -> str:
        return ("가나다라마바사아자차카타파하 " * (len(line) // 14 + 1))[: max(len(line), 8)]

    stub = "\n\n".join(
        f"[[{idx}|{b['tag']}]]\n" + "\n".join(_ko(l) for l in b["lines"])
        for idx, b in sample)
    got = at.parse_response(stub)
    check("응답 파싱이 모든 마커를 복원한다", set(got) == {idx for idx, _ in sample})
    check("정상 응답은 검증을 통과한다", not at.validate(sample, got),
          "; ".join(at.validate(sample, got)))

    print("validator catches bad output")
    dropped = {k: v for k, v in list(got.items())[:-1]}
    check("마커 누락을 잡는다", any("빠진" in p for p in at.validate(sample, dropped)))
    invented = {**got, 999999: ["없는 블록"]}
    check("없는 마커를 잡는다", any("없는 마커" in p for p in at.validate(sample, invented)))
    idx0 = sample[0][0]
    untranslated = {**got, idx0: ["Совершенно секретно, приказ народного комиссара"]}
    check("번역되지 않은 러시아어를 잡는다",
          any("러시아어" in p for p in at.validate(sample, untranslated)))
    big = max(sample, key=lambda x: sum(len(l) for l in x[1]["lines"]))
    if sum(len(l) for l in big[1]["lines"]) > 200:
        check("지나치게 짧은 번역을 잡는다",
              any("짧" in p for p in at.validate(sample, {**got, big[0]: ["짧음"]})))

    print("plan")
    prepared = at.plan(spec, at.Options())
    check("계획이 청크와 비용을 낸다",
          prepared["chunks"] > 0 and prepared["estimatedUsd"] > 0)
    check("계획 문서 수가 스펙과 같다", len(prepared["documents"]) == len(spec["documents"]))
    check("limit_chunks가 계획을 줄인다",
          at.plan(spec, at.Options(limit_chunks=2))["chunks"] == 2)

    print("assembly")
    stub_all = {
        d["offset"] + i: ["한국어 본문 " + ("문장 " * 3) for _ in b["lines"]]
        for d in docs for i, b in enumerate(d["blocks"])
    }
    html = at.assemble(spec, docs, stub_all)
    check("article로 감싼다", html.startswith("<article>") and html.rstrip().endswith("</article>"))
    check("첫 h1이 문서 제목이다", re.search(r"<article>\s*<h1>", html) is not None)
    check("문서마다 h1이 있다", html.count("<h1>") == 1 + len(docs))
    check("h5는 h2로 내려간다", "<h5" not in html and html.count("<h2>") > 0)
    check("인라인 스타일이 없다", "style=" not in html)
    check("금지 태그가 없다",
          not re.search(r"<(html|head|body|main|style|script)\b", html, re.I))
    check("blockquote 안은 p로 감싼다",
          all("<p>" in seg[:40] for seg in html.split("<blockquote>")[1:]))
    expect_spec_error(
        "블록이 비면 조립이 실패한다",
        lambda: at.assemble(spec, docs,
                            {k: v for k, v in stub_all.items() if k != next(iter(stub_all))}))

    print(f"\n{'모두 통과' if not failures else str(len(failures)) + '건 실패: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
