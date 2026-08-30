"""Shared helpers for the site-content translation scripts.

translate_research_markdown, translate_research_documents, translate_db_content,
static_page_translation_pipeline grew independently and each carried its own
copy of fence stripping, JSON parsing, and Hangul detection. The copies drifted
(three fence strippers, two JSON parsers by 2026-08), so the shared parts live
here. Import as ``from scripts._translation_common import …`` — the repo root
is on sys.path in every script, and translate_research_documents already
imports its sibling through the ``scripts.`` package path, so this works both
for ``python scripts/x.py`` and ``python -m scripts.x``. The old bare
``from _translation_common import …`` only resolved because sys.path[0]
happened to be scripts/.
"""

from __future__ import annotations

import json
import re
from typing import Any


def hangul_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Hangul syllables."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    hangul = sum(1 for ch in letters if "가" <= ch <= "힣")
    return hangul / len(letters)


def strip_code_fences(text: str) -> str:
    """Remove one outer ``` fence if the model wrapped its whole answer in it."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model reply.

    Falls back to the outermost {...} slice when the reply carries prose
    around the object — the failure mode of a model that prefixes "Here is
    the JSON:" despite json_mode.
    """
    raw = strip_code_fences(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("translation output is not a JSON object")
    return data


def tag_sequence(html: str) -> list[str]:
    """Lowercased HTML tag-name sequence, comments stripped.

    Two documents whose visible text differs but whose tag sequences match
    have preserved their markup structure — the deterministic check behind
    the "preserve HTML tags" prompt promise.
    """
    without_comments = re.sub(r"<!--.*?-->", "", html or "", flags=re.DOTALL)
    tags = re.findall(r"<\s*/?\s*([a-zA-Z][a-zA-Z0-9:-]*)\b", without_comments)
    return [tag.lower() for tag in tags]


_URL_RE = re.compile(r"https?://[^\s)\"'<>\]]+")


def url_multiset(text: str) -> dict[str, int]:
    """URLs and their counts — a translation must carry destinations verbatim."""
    counts: dict[str, int] = {}
    for url in _URL_RE.findall(text or ""):
        url = url.rstrip(".,;:")
        counts[url] = counts.get(url, 0) + 1
    return counts


def field_translation_problems(
    source: str,
    target: str,
    *,
    label: str,
    max_hangul_ratio: float = 0.05,
    long_text_min: int = 80,
) -> list[str]:
    """Deterministic checks for one KO→EN translated field (인수인계 §2.5).

    Three families of check: residual Hangul (only for long fields whose
    source is actually Korean — a short English title quoting 조선일보 is
    legitimate, so short targets are only checked for verbatim echo), HTML
    tag-sequence preservation, and URL preservation. Problem strings are in
    English because they are fed back to an English-prompted model as the
    correction message.
    """
    problems: list[str] = []
    source = (source or "").strip()
    target = (target or "").strip()
    if not source or not target:
        return problems
    if hangul_ratio(source) >= 0.3:
        if len(target) >= long_text_min:
            ratio = hangul_ratio(target)
            if ratio > max_hangul_ratio:
                problems.append(
                    f"{label}: too much Hangul remains in the translation "
                    f"({ratio:.1%} > {max_hangul_ratio:.1%})"
                )
        elif target == source:
            problems.append(f"{label}: the source text was returned untranslated")
    if tag_sequence(source) != tag_sequence(target):
        problems.append(f"{label}: HTML tag sequence differs from the source")
    if url_multiset(source) != url_multiset(target):
        problems.append(f"{label}: URLs differ from the source; copy link destinations verbatim")
    return problems


class TranslationCallError(RuntimeError):
    """제공자가 실제로 무엇을 돌려줬는지 함께 들고 다니는 오류.

    예전에는 빈 응답이 json.loads에서 "Expecting value: line 1 column 1"으로
    터졌다. 그 문장만 보고는 모델이 거부한 것인지, 응답이 잘린 것인지, 아예
    비어서 온 것인지 구분할 수 없어서 로그를 봐도 손을 못 댔다.
    """
