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

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secrets_loader import get_secret

RESEARCH_DIR = ROOT / "research"
OUTPUT_DIR = RESEARCH_DIR / "en"
DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEFAULT_MODEL = os.getenv("RESEARCH_TRANSLATION_MODEL", "deepseek-v4-flash")
DEFAULT_MAX_TOKENS = int(os.getenv("RESEARCH_TRANSLATION_MAX_TOKENS", "20000"))
TIMEOUT_SECONDS = 240

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


def _hangul_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    hangul = sum(1 for ch in letters if "\uac00" <= ch <= "\ud7a3")
    return hangul / len(letters)


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


def _validate_translation(source: str, translated: str, *, max_hangul_ratio: float) -> None:
    if not translated.strip():
        raise ValueError("empty translation")
    if _heading_signature(source) != _heading_signature(translated):
        raise ValueError("translated heading depth sequence differs from source")
    ratio = _hangul_ratio(translated)
    if ratio > max_hangul_ratio:
        raise ValueError(f"translation still contains too much Hangul ({ratio:.1%}; max {max_hangul_ratio:.1%})")


def _call_deepseek(markdown: str, *, model: str, base_url: str, max_tokens: int) -> str:
    api_key = get_secret("DEEPSEEK_API_KEY", "") or ""
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": markdown},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(f"DeepSeek request failed: HTTP {response.status_code}: {response.text[:1000]}")
    data: dict[str, Any] = response.json()
    return _strip_outer_fence(data["choices"][0]["message"].get("content") or "")


def translate_one(
    source_path: Path,
    *,
    output_dir: Path,
    model: str,
    base_url: str,
    max_tokens: int,
    max_hangul_ratio: float,
    force: bool,
    dry_run: bool,
) -> Path:
    output_path = output_dir / source_path.name
    if output_path.exists() and not force:
        print(f"skip: {output_path} exists")
        return output_path

    source = source_path.read_text(encoding="utf-8")
    print(f"translating: {source_path.name} ({len(source):,} chars) with {model}")
    translated = _call_deepseek(source, model=model, base_url=base_url, max_tokens=max_tokens)
    _validate_translation(source, translated, max_hangul_ratio=max_hangul_ratio)
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
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
                model=args.model,
                base_url=args.base_url.rstrip("/"),
                max_tokens=args.max_tokens,
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
