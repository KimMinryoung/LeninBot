#!/usr/bin/env python3
"""Translate public research markdown documents into English.

This is intentionally separate from static_pages translation. Korean research
pages are stored as markdown under research/*.md; English translations live
under research/en/*.md and are loaded when the site language cookie is English.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secrets_loader import get_secret

RESEARCH_DIR = ROOT / "research"
OUTPUT_DIR = RESEARCH_DIR / "en"
# 모델·예산·타임아웃은 레지스트리 항목이 정한다 (config/llm_call_sites.json).
# 실키는 llm_proxy에만 있고, 호출은 게이트웨이를 지나 감사에 남는다.
FEATURE = "research_markdown_translation"

SYSTEM_PROMPT = """You are a meticulous Korean-to-English translation editor for political economy research.

Translate the user's Korean markdown document into polished, publication-quality English.

Requirements:
- Preserve markdown structure exactly: headings, lists, blockquotes, tables, code fences, links, footnotes, and horizontal rules.
- Translate prose and visible Korean text; keep URLs and markdown link destinations unchanged.
- Do not summarize, omit, expand, fact-check, or add commentary.
- Preserve the author's Marxist, anti-imperialist analytical stance without softening it.
- Use domain-aware terminology:
  - 대미 = toward / vis-a-vis the United States, or U.S.-linked depending on context.
  - 대중국 / 대중 when it means 對中國 = toward China / China-facing / China-dependent; never "popular" or "mass".
  - 민중 = the people / popular masses, depending on context.
  - 노동자 = workers.
  - 재벌 = chaebol.
  - 제국주의 = imperialism.
  - 종속 = dependency or subordination, depending on context.
  - 한반도 = Korean Peninsula.
  - 한국 = South Korea or Korean, depending on context.
- Keep proper names and organization names sensible. Do not invent Western politician names.

Return only the translated markdown. No code fence around the whole document.
"""


def _slug_to_path(slug_or_path: str) -> Path:
    raw = Path(slug_or_path)
    if raw.is_absolute():
        path = raw
    elif raw.suffix == ".md" or "/" in slug_or_path:
        path = ROOT / raw
    else:
        path = RESEARCH_DIR / f"{slug_or_path}.md"
    path = path.resolve()
    if RESEARCH_DIR.resolve() not in path.parents or path.suffix != ".md":
        raise ValueError(f"research markdown must be under {RESEARCH_DIR}: {slug_or_path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


from scripts._translation_common import hangul_ratio as _hangul_ratio


def _heading_signature(markdown: str) -> list[str]:
    sig = []
    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+", line)
        if match:
            sig.append(match.group(1))
    return sig


def _strip_outer_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:markdown|md)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip() + "\n"


# The label the model produces for '선행 보고서:' drifts between runs
# ("Preceding Reports:", "Preceding report:", "Preceding reports:"), so it is
# fixed here instead of hoping the prompt holds. Link destinations are left
# alone; the count check below catches a translation that mangles them.
_EN_PRECEDING_LABEL_RE = re.compile(
    r"(?mi)^(\s*)\*\*\s*preceding\s+reports?\s*:\s*\*\*"
)
_RESEARCH_REF_RE = re.compile(r"/reports/research/[a-z0-9-]+")


def normalize_translated_markdown(translated: str) -> str:
    """One canonical English wording for the fixed frame's labels."""
    return _EN_PRECEDING_LABEL_RE.sub(r"\1**Preceding reports:**", translated)


def _validate_translation(source: str, translated: str, *, max_hangul_ratio: float) -> None:
    if not translated.strip():
        raise ValueError("empty translation")
    if _heading_signature(source) != _heading_signature(translated):
        raise ValueError("translated heading depth sequence differs from source")
    ratio = _hangul_ratio(translated)
    if ratio > max_hangul_ratio:
        raise ValueError(f"translation still contains too much Hangul ({ratio:.1%}; max {max_hangul_ratio:.1%})")
    source_refs = _RESEARCH_REF_RE.findall(source)
    translated_refs = _RESEARCH_REF_RE.findall(translated)
    if sorted(source_refs) != sorted(translated_refs):
        raise ValueError(
            "translation changed the internal report links "
            f"({len(source_refs)} in source, {len(translated_refs)} in translation); "
            "link destinations must be copied verbatim"
        )


def _call_translator(markdown: str, *, correction: str = "") -> str:
    """게이트웨이를 지나는 원샷 호출.

    correction은 직전 시도의 검증 실패 사유다. 원문에 섞으면 모델이 그 문장까지
    번역할 수 있으므로 시스템 프롬프트 뒤에 붙인다.
    """
    from llm.call_registry import generate_sync

    text = generate_sync(FEATURE, markdown, system=SYSTEM_PROMPT + correction)
    if not text:
        # generate_sync는 실패 원인을 삼키고 None을 준다. HTTP 오류·정책 거부·
        # 빈 완성 어느 쪽인지는 llm_gateway.audit과 [llm-registry] 경고에 남는다.
        raise RuntimeError(
            f"{FEATURE}: 게이트웨이가 본문을 돌려주지 않았다 "
            f"(원인은 llm_gateway.audit / [llm-registry] 경고 참조)"
        )
    return _strip_outer_fence(text)


def translate_markdown_with_retry(source: str, *, max_hangul_ratio: float, attempts: int = 2) -> str:
    """호출→정규화→검증을 한 번에. 검증 실패는 사유를 다음 시도에 알려 재번역한다.

    예전에는 검증 실패가 곧 문서 실패였다: 제목 깊이 하나가 어긋나면 2만 토큰짜리
    완성본을 통째로 버리고 사람이 다시 돌렸다. 실패 사유는 이미 문장으로 나오므로,
    그 문장을 다음 호출에 붙여 한 번은 모델이 스스로 고치게 한다 — 사료 파이프라인의
    교정 재시도와 같은 패턴이다. 반복 자기수정 루프는 두지 않는다(인수인계 §3:
    reflection 1회).
    """
    correction = ""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        translated = normalize_translated_markdown(_call_translator(source, correction=correction))
        try:
            _validate_translation(source, translated, max_hangul_ratio=max_hangul_ratio)
            return translated
        except ValueError as exc:
            last_error = exc
            correction = (
                "\n\nThe previous attempt failed validation:\n"
                f"- {exc}\n"
                "Re-translate the full document, fixing exactly this problem and changing nothing else."
            )
            print(f"validation failed (attempt {attempt}/{attempts}): {exc}", file=sys.stderr)
    assert last_error is not None
    raise last_error


def translate_one(
    source_path: Path,
    *,
    output_dir: Path,
    max_hangul_ratio: float,
    force: bool,
    dry_run: bool,
) -> Path:
    output_path = output_dir / source_path.name
    if output_path.exists() and not force:
        print(f"skip: {output_path} exists")
        return output_path

    source = source_path.read_text(encoding="utf-8")
    print(f"translating: {source_path.name} ({len(source):,} chars) via {FEATURE}")
    translated = translate_markdown_with_retry(source, max_hangul_ratio=max_hangul_ratio)
    if dry_run:
        print(f"dry-run ok: {source_path.stem} ({len(translated):,} chars)")
        return output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(translated, encoding="utf-8")
    print(f"wrote: {output_path} ({len(translated):,} chars)")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate research/*.md documents with DeepSeek V4 Flash.")
    parser.add_argument("targets", nargs="+", help="Research slugs or paths, e.g. alt-economy-04")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max-hangul-ratio", type=float, default=0.03)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    failures = 0
    for target in args.targets:
        try:
            translate_one(
                _slug_to_path(target),
                output_dir=Path(args.output_dir),
                max_hangul_ratio=args.max_hangul_ratio,
                force=args.force,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"failed: {target}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
