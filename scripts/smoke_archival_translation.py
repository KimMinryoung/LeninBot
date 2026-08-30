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
    # 스펙 수준 source는 선택이다. 문서마다 제 출처를 가진 스펙(조약과 의정서가
    # 서로 다른 페이지에 있는 경우)에서는 그것이 없으므로, 검사도 그때는
    # 문서별 출처를 본다.
    docs = at.slice_documents(spec)
    check("문서가 스펙 수만큼 잘린다", len(docs) == len(spec["documents"]))
    check("블록이 비지 않는다", all(d["blocks"] for d in docs))
    check("블록 범위가 겹치지 않는다",
          len({d["offset"] + i for d in docs for i in range(len(d["blocks"]))})
          == sum(len(d["blocks"]) for d in docs))

    drifted = json.loads(json.dumps(spec))
    drifted["documents"][0]["startsWith"] = "이 문자열은 원문에 없다"
    expect_spec_error("경계가 어긋나면 실패한다", lambda: at.slice_documents(drifted))
    tampered = json.loads(json.dumps(spec))
    if "source" in tampered:
        tampered["source"]["sha256"] = "0" * 64
    else:
        tampered["documents"][0]["source"]["sha256"] = "0" * 64
    expect_spec_error("원본이 바뀌면 실패한다", lambda: at.slice_documents(tampered))

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

    print("gloss dedupe / register")
    from runtime_tools.archival_translation import core as _core
    deduped = _core.dedupe_glosses([
        "필랴르(Пиляр)가 보고했다.",
        "이후 필랴르(Пиляр)는 체포되었다.",
        "네스테로프(Nesterov)와 네스테로프(Nesterov)가 함께",
        "전연방공산당(볼셰비키) 중앙위원회는 전연방공산당(볼셰비키) 소속이다.",
    ])
    check("두 번째 원문 병기를 지운다", deduped[1] == "이후 필랴르는 체포되었다.")
    check("같은 줄 안의 반복도 지운다", deduped[2].count("(Nesterov)") == 1)
    check("한글 괄호는 이름의 일부라 남긴다", deduped[3].count("(볼셰비키)") == 2)
    check("첫 병기는 보존한다", "(Пиляр)" in deduped[0])
    check("문서마다 문체가 지정되어 있다",
          all(d.get("register") for d in spec["documents"]))
    check("문체가 블록에 실린다", all(b.get("register") for d in docs for b in d["blocks"]))
    check("문체가 프롬프트에 들어간다",
          "문체:" in _core._chunk_prompt(chunks[0], glossary, at.Options()))

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

    print("translate loop (stub provider)")
    # _translate_chunk is the one path the other checks never enter, because it
    # is the one that calls the API. Stubbing the executor covers it offline —
    # without this a refactor can leave an undefined name in the loop and every
    # check still passes.
    from llm import call_registry

    from runtime_tools.archival_translation import core

    class _StubCache:
        def __init__(self):
            self.written = {}

        def get(self, key):
            return None

        def put(self, key, blocks, meta):
            self.written[key] = (blocks, meta)

    # _translate_chunk은 이제 용어집 준수까지 검증하므로, 스텁도 샘플에 등장하는
    # 확정 표기를 실어야 통과한다 — 실제 모델에게 요구하는 것과 같은 조건이다.
    def _ko_with_terms(line: str) -> str:
        needed = " ".join(t["ko"] for t in sample_terms if t["pattern"].search(line))
        return (_ko(line) + (" " + needed if needed else "")).rstrip()

    stub_ok = "\n\n".join(
        f"[[{idx}|{b['tag']}]]\n" + "\n".join(_ko_with_terms(l) for l in b["lines"])
        for idx, b in sample)

    original = call_registry.generate_sync
    call_registry.generate_sync = lambda *a, **k: stub_ok
    try:
        cache = _StubCache()
        stats = core.Stats()
        got_live = core._translate_chunk(sample, glossary, cache, at.Options(),
                                         stats, lambda e: None)
        check("번역 루프가 청크를 반환한다", set(got_live) == {i for i, _ in sample})
        check("번역 루프가 캐시에 기록한다", len(cache.written) == 1)
        check("성공이 stats에 반영된다", stats.translated == 1 and stats.failed == 0)

        call_registry.generate_sync = lambda *a, **k: ""  # provider가 빈 응답
        try:
            core._translate_chunk(sample, glossary, _StubCache(),
                                  at.Options(retries=1), core.Stats(), lambda e: None)
            check("빈 응답이면 실패로 올라온다", False, "예외가 나지 않았다")
        except RuntimeError as e:
            check("빈 응답이면 실패로 올라온다", "빈 응답" in str(e), str(e))
    finally:
        call_registry.generate_sync = original

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
    # 문서 제목 h1은 heading 설정을 따른다. 문서가 하나뿐인 스펙은 그 제목이
    # 곧 페이지 제목이라 heading:false로 끄고, 그때는 h1이 하나만 나와야 한다.
    # 페이지 제목 h1 뒤쪽만 본다. 문서가 하나인 스펙은 제목이 스펙 제목과
    # 같아서, 전체 문자열로 재면 페이지 제목이 문서 제목으로 잘못 잡힌다.
    after_title = html.split("</h1>", 1)[1]
    # 주석 문서는 h1이 아니라 주석 절의 h2를 제목으로 쓴다.
    titled = [d for d in docs if d.get("heading", True) and not d.get("notes")]
    check("문서 제목 h1이 heading 설정을 따른다",
          all(f"<h1>{d['titleKo']}</h1>" in after_title for d in titled)
          and not any(f"<h1>{d['titleKo']}</h1>" in after_title
                      for d in docs if not d.get("heading", True)))
    # 참고 문헌 16건이 쓰는 서두 틀 — 제목, byline 한 줄, 해제와 서지 목록을
    # 담은 엮은이 주 상자. 손으로 쓴 문서와 이 파이프라인의 출력이 같아야 한다.
    check("byline이 제목 바로 아래 온다",
          re.search(r"</h1>\s*<p class=\"doc-byline\"><strong>", html) is not None)
    check("엮은이 주 상자가 해제 뒤에 서지 목록을 담는다",
          re.search(r"doc-editorial-label\">[^<]+</p>(<p>.*?</p>)+<ul><li>", html)
          is not None)
    check("서두에 hr을 두지 않는다", "<hr" not in html)
    # 주석은 서고 전체가 한 양식을 쓴다: 본문의 [n]은 앵커이고 항목에는 돌아오는
    # 화살표가 달리며, 목록에는 크기·색이 걸린 notes-list 클래스가 붙는다.
    if any(d.get("notes") for d in docs):
        refs = set(re.findall(r'class="note-ref" id="ref-([^"]+)"', html))
        items = set(re.findall(r'<li id="note-([^"]+)"', html))
        backs = set(re.findall(r'back-link" href="#ref-([^"]+)"', html))
        check("주석 목록에 notes-list 클래스가 붙는다",
              '<ol class="notes-list">' in html
              and '<section class="notes"' in html)
        # 스텁 본문에는 [n]이 없으므로 개수가 아니라 정합성만 본다.
        check("본문 [n]이 주석 항목으로 연결된다",
              refs <= items, f"{sorted(refs - items)[:3]}")
        check("돌아오는 화살표가 끊기지 않는다",
              backs <= refs, f"{sorted(backs - refs)[:3]}")
    # 소제목이 있는 문서만 h2를 요구한다. 조항이 소제목 없이 이어지는 결의문
    # (21개 조건 같은)은 h2가 하나도 없는 것이 맞다.
    has_headings = any(b["tag"] in ("h2", "h3", "h4", "h5")
                       for d in docs for b in d["blocks"])
    check("h5는 h2로 내려간다",
          "<h5" not in html and (html.count("<h2>") > 0 or not has_headings))
    check("인라인 스타일이 없다", "style=" not in html)
    check("금지 태그가 없다",
          not re.search(r"<(html|head|body|main|style|script)\b", html, re.I))
    check("blockquote 안은 p로 감싼다",
          all("<p>" in seg[:40] for seg in html.split("<blockquote>")[1:]))
    expect_spec_error(
        "frozen 스펙은 재실행을 거부한다",
        lambda: at.run({**spec, "frozen": "테스트"}))
    expect_spec_error(
        "블록이 비면 조립이 실패한다",
        lambda: at.assemble(spec, docs,
                            {k: v for k, v in stub_all.items() if k != next(iter(stub_all))}))

    print(f"\n{'모두 통과' if not failures else str(len(failures)) + '건 실패: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
