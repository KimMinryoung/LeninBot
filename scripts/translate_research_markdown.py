#!/usr/bin/env python3
"""Translate public research markdown documents into English.

This is intentionally separate from static_pages translation. Korean research
pages are stored as markdown under research/*.md; English translations live
under research/en/*.md and are loaded when the site language cookie is English.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_runtime.storage import atomic_write, source_hash
from translation_runtime.structure import (markdown_problems, markdown_chunks,
    protect_markdown, restore_markdown, semantic_review)
from scripts._translation_common import generate_translation

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
    if not isinstance(translated, str):
        raise ValueError("translation must be text")
    if not translated.strip():
        raise ValueError("empty translation")
    if _heading_signature(source) != _heading_signature(translated):
        raise ValueError("translated heading depth sequence differs from source")
    problems = markdown_problems(source, translated)
    if problems:
        raise ValueError("; ".join(problems))
    # Protected code and destinations may legitimately contain Korean.
    visible, spans = protect_markdown(translated)
    for marker in spans:
        visible = visible.replace(marker, "")
    ratio = _hangul_ratio(visible)
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
    text = generate_translation(FEATURE, markdown, system=SYSTEM_PROMPT + correction)
    return _strip_outer_fence(text)




def _translate_segment(source: str, *, max_hangul_ratio: float, attempts: int,
                       cached=None, store=None) -> str:
    from translation_runtime import translate_validated
    masked, protected = protect_markdown(source)

    def parse(candidate):
        for marker in protected:
            if candidate.count(marker) != masked.count(marker):
                raise ValueError("protected placeholder count changed")
        return normalize_translated_markdown(restore_markdown(candidate, protected))

    def validate(candidate):
        try:
            _validate_translation(source, candidate, max_hangul_ratio=max_hangul_ratio)
        except ValueError as exc:
            return [str(exc)]
        return []

    return translate_validated(
        generate=lambda correction: _call_translator(masked, correction=(
            "\nPreserve every TRKEEP placeholder exactly, including its number of occurrences."
            + correction)), parse=parse, validate=validate, attempts=attempts,
        cached=cached, store=store)


def translate_markdown_with_retry(source: str, *, max_hangul_ratio: float, attempts: int = 2,
                                  cache_dir: Path | None = None, max_chars: int = 8000,
                                  review_path: Path | None = None) -> str:
    """Validated structural chunks; reuse successes after a failed document run."""
    from llm.call_registry import resolve
    if attempts < 1 or max_chars < 1:
        raise ValueError("attempts and max_chars must be positive")
    cache_dir = cache_dir or ROOT / "output" / "site_translation_cache"
    profile = dataclasses.asdict(resolve(FEATURE))
    profile.pop("note", None)
    chunks = markdown_chunks(source, max_chars)
    translated_chunks = []
    for source_chunk in chunks:
        fingerprint = json.dumps({"version": 1, "source": source_chunk, "profile": profile,
                                  "system": SYSTEM_PROMPT}, sort_keys=True, ensure_ascii=False)
        key = source_hash(fingerprint)
        path = cache_dir / f"{key}.json"
        cached = None
        if path.is_file():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))["target"]
            except (ValueError, KeyError, TypeError):
                pass
        translated = _translate_segment(source_chunk, max_hangul_ratio=max_hangul_ratio,
            attempts=attempts, cached=cached,
            store=lambda value: atomic_write(path, json.dumps({
                "sourceHash": source_hash(source_chunk), "target": value}, ensure_ascii=False)))
        translated_chunks.append(translated.strip())
    translated = "\n\n".join(translated_chunks) + "\n"
    _validate_translation(source, translated, max_hangul_ratio=max_hangul_ratio)
    if review_path:
        atomic_write(review_path, json.dumps({"sourceHash": source_hash(source),
                     "issues": semantic_review(source, translated)}, ensure_ascii=False, indent=2))
    return translated


def translate_one(
    source_path: Path,
    *,
    output_dir: Path,
    max_hangul_ratio: float,
    force: bool,
    dry_run: bool,
) -> Path:
    output_path = output_dir / source_path.name
    source = source_path.read_text(encoding="utf-8")
    metadata_path = output_path.with_suffix(".translation.json")
    metadata = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    if (output_path.exists() and not force
            and metadata.get("sourceHash") == source_hash(source)):
        try:
            _validate_translation(source, output_path.read_text(encoding="utf-8"),
                                  max_hangul_ratio=max_hangul_ratio)
        except ValueError as exc:
            # A hand-edited translation is never overwritten silently.
            raise RuntimeError(
                f"existing translation {output_path} fails validation ({exc}); "
                "fix it by hand or rerun with --force") from exc
        print(f"skip: {output_path} exists")
        return output_path

    print(f"translating: {source_path.name} ({len(source):,} chars) via {FEATURE}")
    translated = translate_markdown_with_retry(source, max_hangul_ratio=max_hangul_ratio,
        review_path=None if dry_run else output_path.with_suffix(".review.json"))
    if dry_run:
        print(f"dry-run ok: {source_path.stem} ({len(translated):,} chars)")
        return output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_path.read_text(encoding="utf-8") != source:
        raise RuntimeError("source changed during translation; rerun with the latest source")
    atomic_write(output_path, translated)
    atomic_write(metadata_path, json.dumps({"sourceHash": source_hash(source),
                                          "targetHash": source_hash(translated)}))
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
