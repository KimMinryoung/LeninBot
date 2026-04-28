#!/usr/bin/env python3
"""Smoke checks for localized static page publishing and rendering."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import _render_static_page_html
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

    rendered_en = _render_static_page_html(page, "en")
    assert '<html lang="en">' in rendered_en
    assert "English Body" in rendered_en
    assert "Cyber-Lenin Reports" in rendered_en
    assert "한국어 본문" not in rendered_en

    rendered_ko = _render_static_page_html(page, "ko")
    assert '<html lang="ko">' in rendered_ko
    assert "한국어 본문" in rendered_ko

    print("static page localization smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
