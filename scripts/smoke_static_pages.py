#!/usr/bin/env python3
"""Smoke checks for localized static page selection.

Rendering is deliberately not covered here any more. 4275c42 (2026-04-29)
moved static pages into the database and the HTML is now produced by the
frontend repo (frontend/routes/pages.js serves /p/:slug and sanitizes the body
client-side), so api._render_static_page_html no longer exists. This file kept
importing it and has failed at import since — three months of a green-looking
suite that never ran. What stays here is the part this repo still owns: which
language variant localize_static_page hands out, and what it falls back to.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from site_publishing import localize_static_page


def main() -> int:
    page = {
        "slug": "sample",
        "title": "한국어 제목",
        "summary": "한국어 요약",
        "html_body": "<article><h2>한국어 본문</h2><p>원문</p></article>",
        "title_en": "English Title",
        "summary_en": "English summary",
        "html_body_en": "<article><h2>English Body</h2><p>Translation</p></article>",
    }

    ko = localize_static_page(page, "ko")
    assert ko["language"] == "ko"
    assert ko["title"] == "한국어 제목"
    assert ko["available_languages"] == ["ko", "en"]

    en = localize_static_page(page, "en")
    assert en["language"] == "en"
    assert en["title"] == "English Title"
    assert "English Body" in en["html_body"]

    fallback = localize_static_page({k: v for k, v in page.items() if not k.endswith("_en")}, "en")
    assert fallback["requested_language"] == "en"
    assert fallback["language"] == "ko"
    assert fallback["available_languages"] == ["ko"]

    # The two rendering assertions that used to live here are gone with the
    # function they called. Their surviving intent — a language variant must not
    # leak the other language's body — belongs on the selector, so it is checked
    # here instead.
    assert "한국어 본문" not in en["html_body"]
    assert "English Body" not in ko["html_body"]

    # An unknown or missing language falls back to Korean rather than erroring.
    for lang in (None, "", "fr"):
        picked = localize_static_page(page, lang)
        assert picked["language"] == "ko", f"{lang!r} should fall back to ko"

    print("static page localization smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
