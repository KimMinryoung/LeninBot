#!/usr/bin/env python3
"""Compose a 4-panel political comic page for cyber-lenin.com.

Layout: 4 panels stacked in a single column (one panel per row) so each panel
can be large enough to read on mobile. Each panel is 960x320.

Content rule (see feedback_comic_panel_content.md and
feedback_comic_visual_vocabulary.md in the frontend memory):
  - The agent authors `scene_svg` per panel using named-object icons (tv_news,
    vault, goldbar_stack, etc. — see assets/comic_icons/).
  - Panels contain only the scene + a single speech balloon. No captions,
    headings, subtext, transcripts, or analysis sections on the page.

Payload schema:
{
  "slug": "<slug-a-z0-9->",
  "title": "...",
  "summary": "...",
  "panels": [
    { "scene_svg": "<g>...</g> or raw SVG children", "speech": "..." },
    ... (exactly 4) ...
  ]
}
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from site_publishing import _exec_publish_static_page  # type: ignore

_ALLOWED_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")

# Panel geometry (per-panel viewBox)
PANEL_WIDTH = 960
PANEL_HEIGHT = 320

# Speech balloon geometry within a panel
BALLOON_X = 40
BALLOON_Y = 28
BALLOON_W = 380
BALLOON_H = 108
BALLOON_TEXT_X = 62
BALLOON_TEXT_Y = 62
BALLOON_LINE_HEIGHT = 30
BALLOON_MAX_CHARS = 22
BALLOON_MAX_LINES = 3

# Scene-SVG sanitization: strip dangerous tags, event handlers, and
# javascript:/data: href schemes. DOMPurify on the frontend is the second
# line of defense; this prevents dangerous content from ever landing in the
# persisted page JSON.
_FORBIDDEN_TAGS_BLOCK = re.compile(
    r"<(?:script|foreignObject|iframe|object|embed|style)\b[^>]*>.*?</(?:script|foreignObject|iframe|object|embed|style)>",
    re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN_TAGS_SELF = re.compile(
    r"<(?:script|foreignObject|iframe|object|embed|style)\b[^>]*/?>",
    re.IGNORECASE,
)
_EVENT_ATTR = re.compile(r"""\s+on[a-z]+\s*=\s*["'][^"']*["']""", re.IGNORECASE)
_DANGEROUS_HREF = re.compile(
    r"""\s+(?:xlink:)?href\s*=\s*["']\s*(?:javascript:|data:)[^"']*["']""",
    re.IGNORECASE,
)


def _escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    """Greedy whitespace-based wrap. Korean without spaces will fall back to a
    single line — acceptable given BALLOON_MAX_CHARS is a soft budget.
    """
    words = [w for w in _norm_text(text).split() if w]
    if not words:
        return []
    # If one "word" is itself longer than budget, accept it as one line.
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _sanitize_scene_svg(fragment: str) -> str:
    s = _FORBIDDEN_TAGS_BLOCK.sub("", fragment)
    s = _FORBIDDEN_TAGS_SELF.sub("", s)
    s = _EVENT_ATTR.sub("", s)
    s = _DANGEROUS_HREF.sub("", s)
    return s


def _speech_block(text: str) -> str:
    lines = _wrap_text(text, max_chars=BALLOON_MAX_CHARS)
    if not lines:
        return ""
    tspans = []
    for idx, line in enumerate(lines[:BALLOON_MAX_LINES]):
        dy = 0 if idx == 0 else BALLOON_LINE_HEIGHT
        tspans.append(
            f'<tspan x="{BALLOON_TEXT_X}" dy="{dy}">{_escape(line)}</tspan>'
        )
    text_el = (
        f'<text x="{BALLOON_TEXT_X}" y="{BALLOON_TEXT_Y}" '
        f'class="comic-speech">{"".join(tspans)}</text>'
    )
    # Tail: triangle dropping from balloon bottom, pointing down-left into the scene
    tail_apex_x = BALLOON_X + 96
    tail_apex_y = BALLOON_Y + BALLOON_H
    tail_foot_x = tail_apex_x - 28
    tail_foot_y = tail_apex_y + 44
    tail_back_x = tail_apex_x + 44
    tail_back_y = tail_apex_y + 6
    return (
        '\n      <g class="comic-balloon">'
        f'\n        <rect x="{BALLOON_X}" y="{BALLOON_Y}" '
        f'width="{BALLOON_W}" height="{BALLOON_H}" rx="22" />'
        f'\n        <path d="M {tail_apex_x} {tail_apex_y} '
        f'L {tail_foot_x} {tail_foot_y} L {tail_back_x} {tail_back_y} Z" />'
        f'\n        {text_el}'
        '\n      </g>'
    )


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    slug = _norm_text(payload.get("slug")).lower()
    title = _norm_text(payload.get("title"))
    summary = _norm_text(payload.get("summary"))

    if not slug or not _ALLOWED_SLUG_RE.match(slug):
        raise ValueError("slug must match ^[a-z0-9][a-z0-9-]{0,79}$")
    if not title:
        raise ValueError("title is required")
    if not summary:
        raise ValueError("summary is required")

    panels = payload.get("panels")
    if not isinstance(panels, list) or len(panels) != 4:
        raise ValueError("panels must be a list of exactly 4 items")

    cleaned_panels: list[dict[str, str]] = []
    for index, panel in enumerate(panels, start=1):
        if not isinstance(panel, dict):
            raise ValueError(f"panel {index} must be an object")
        scene_svg = _norm_text(panel.get("scene_svg"))
        speech = _norm_text(panel.get("speech"))
        if not scene_svg:
            raise ValueError(f"panel {index} scene_svg is required")
        if speech and len(_wrap_text(speech, max_chars=BALLOON_MAX_CHARS)) > BALLOON_MAX_LINES:
            raise ValueError(
                f"panel {index} speech wraps to more than {BALLOON_MAX_LINES} lines "
                f"at {BALLOON_MAX_CHARS}-char width; shorten it"
            )
        cleaned_panels.append({"scene_svg": scene_svg, "speech": speech})

    return {
        "slug": slug,
        "title": title,
        "summary": summary,
        "panels": cleaned_panels,
    }


def build_html_body(payload: dict[str, Any]) -> str:
    data = validate_payload(payload)
    panel_blocks: list[str] = []
    for idx, panel in enumerate(data["panels"], start=1):
        scene = _sanitize_scene_svg(panel["scene_svg"])
        balloon = _speech_block(panel["speech"])
        panel_blocks.append(
            f'<figure class="comic-panel" aria-label="Panel {idx}">\n'
            f'    <svg class="comic-panel-svg" viewBox="0 0 {PANEL_WIDTH} {PANEL_HEIGHT}" '
            f'xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f'      <rect x="3" y="3" width="{PANEL_WIDTH - 6}" height="{PANEL_HEIGHT - 6}" '
            f'rx="22" class="comic-panel-box" />\n'
            f'      <g class="comic-scene">{scene}</g>{balloon}\n'
            f'    </svg>\n'
            f'  </figure>'
        )
    panels_html = "\n  ".join(panel_blocks)

    return (
        '<article class="comic-page">\n'
        '  <style>\n'
        '    .comic-page { max-width: 960px; margin: 0 auto; padding: 8px 0 40px; color: #2d3436; }\n'
        '    .comic-panel { margin: 0 0 20px; }\n'
        '    .comic-panel:last-child { margin-bottom: 0; }\n'
        '    .comic-panel-svg { width: 100%; height: auto; display: block; }\n'
        '    .comic-panel-box { fill: #fffdf5; stroke: #2d3436; stroke-width: 3; }\n'
        '    .comic-balloon rect { fill: #ffffff; stroke: #2d3436; stroke-width: 3; }\n'
        '    .comic-balloon path { fill: #ffffff; stroke: #2d3436; stroke-width: 3; stroke-linejoin: round; }\n'
        '    .comic-speech { font-family: "Helvetica Neue", "Noto Sans KR", system-ui, sans-serif; '
        'font-size: 26px; font-weight: 800; fill: #2d3436; letter-spacing: 0.02em; }\n'
        '  </style>\n'
        f'  {panels_html}\n'
        '</article>\n'
    )


def build_page_payload(payload: dict[str, Any]) -> dict[str, str]:
    data = validate_payload(payload)
    return {
        "slug": data["slug"],
        "title": data["title"],
        "summary": data["summary"],
        "html_body": build_html_body(data),
    }


async def publish_payload(payload: dict[str, Any]) -> str:
    page = build_page_payload(payload)
    return await _exec_publish_static_page(**page)


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or publish a 4-panel political comic static page "
        "from agent-authored scene SVGs.",
    )
    parser.add_argument("payload", help="Path to JSON payload.")
    parser.add_argument(
        "--print-html",
        action="store_true",
        help="Print html_body instead of the JSON page payload.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish directly into static_pages via site_publishing.",
    )
    args = parser.parse_args()

    payload = load_payload(Path(args.payload))
    if args.publish:
        import asyncio
        print(asyncio.run(publish_payload(payload)))
        return

    page_payload = build_page_payload(payload)
    if args.print_html:
        print(page_payload["html_body"])
    else:
        print(json.dumps(page_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
