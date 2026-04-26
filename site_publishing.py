"""site_publishing.py — Publishing tools for cyber-lenin.com sections.

Two tools:
- `publish_hub_curation` — insert a structured curation row into `hub_curations`.
  Served at https://cyber-lenin.com/hub/{slug}. Format is STRUCTURED (source link +
  selection rationale + context) rather than freeform markdown, because curation
  quality depends on consistent per-field discipline.
- `publish_static_page` — write an HTML page to a sandboxed directory under
  `/home/grass/leninbot/static_pages/`. Served at https://cyber-lenin.com/p/{slug}.
  The slug is validated (alphanumeric + dash only) so the path cannot escape the
  sandbox. The HTML body is inserted inside the site's common layout (nav, footer,
  CSS) by the frontend — agents only write page content, not a full HTML document.

Both tools are safe for autonomous use: they can only write to their own
namespaces, not arbitrary paths. `write_file` stays excluded from the autonomous
agent's toolset.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from db import execute as db_execute, query as db_query, query_one as db_query_one
from shared import KST
from telegram.channel_broadcast import maybe_broadcast_autonomous_publication

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_PAGES_DIR = Path(PROJECT_ROOT) / "static_pages"

# slug must start with alphanum, then alphanum or '-' only, 1..80 chars.
# This deliberately forbids '/', '..', '\\', '.' to prevent path traversal.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")

_HUB_TABLE_ENSURED = False


def _ensure_hub_table() -> None:
    global _HUB_TABLE_ENSURED
    if _HUB_TABLE_ENSURED:
        return
    db_execute("""
        CREATE TABLE IF NOT EXISTS hub_curations (
            id                   SERIAL PRIMARY KEY,
            slug                 VARCHAR(100) UNIQUE NOT NULL,
            title                TEXT NOT NULL,
            source_url           TEXT NOT NULL,
            source_title         TEXT,
            source_author        TEXT,
            source_publication   TEXT,
            source_published_at  DATE,
            selection_rationale  TEXT NOT NULL,
            context              TEXT NOT NULL,
            tags                 JSONB NOT NULL DEFAULT '[]'::jsonb,
            published_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    db_execute("""
        CREATE INDEX IF NOT EXISTS hub_curations_published_at_idx
        ON hub_curations(published_at DESC)
    """)
    _HUB_TABLE_ENSURED = True


def _slugify(text: str, max_len: int = 60) -> str:
    """ASCII-safe slug from free text. Korean / non-ASCII chars are stripped; if the
    result is empty, returns an empty string and the caller must provide a fallback.
    """
    s = unicodedata.normalize("NFKD", text or "")
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len]


def _unique_slug(base: str, table: str) -> str:
    """Append -2, -3, … if base collides on `table.slug`. Caller must have already
    validated `base` matches `_SLUG_RE` (stripped, non-empty).
    """
    row = db_query_one(f"SELECT 1 FROM {table} WHERE slug = %s", (base,))
    if not row:
        return base
    for n in range(2, 100):
        cand = f"{base}-{n}"[:100]
        row = db_query_one(f"SELECT 1 FROM {table} WHERE slug = %s", (cand,))
        if not row:
            return cand
    # Extremely unlikely; fall through with a timestamp suffix
    return f"{base}-{int(datetime.now().timestamp())}"[:100]


# ══════════════════════════════════════════════════════════════════════
# Tool 1: publish_hub_curation
# ══════════════════════════════════════════════════════════════════════

PUBLISH_HUB_CURATION_TOOL = {
    "name": "publish_hub_curation",
    "description": (
        "Publish a curation entry to the cyber-lenin.com /hub section. Hub entries "
        "link to external Korean-language progressive writing that the agent judges "
        "as excellent, with a selection rationale and context tying it into other "
        "debates. Served at https://cyber-lenin.com/hub/{slug}. Use ONLY for pieces "
        "that meet the curation criteria (theoretical depth, on-the-ground specifics, "
        "real-world fit) — do not flood with marginal material. Korean-language "
        "sources only at this time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Your framing title for this curation entry (Korean). E.g. '왜 이 글이 지금 중요한가: …'",
            },
            "source_url": {
                "type": "string",
                "description": "The external URL being curated. Must be a direct link to the source piece.",
            },
            "source_title": {
                "type": "string",
                "description": "The original piece's title, verbatim.",
            },
            "source_author": {
                "type": "string",
                "description": "Original author/byline if known. Optional.",
            },
            "source_publication": {
                "type": "string",
                "description": "Publication/site name (e.g. '참세상', '민중의소리'). Optional.",
            },
            "source_published_at": {
                "type": "string",
                "description": "Original publication date in YYYY-MM-DD. Optional.",
            },
            "selection_rationale": {
                "type": "string",
                "description": (
                    "Why this piece was selected — tie to the curation criteria: theoretical "
                    "sophistication, reality-fit, hard-to-find on-the-ground specifics."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Brief contextual framing — how this piece connects to other debates, "
                    "what the reader should know before clicking through."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional short tags for browsing/filtering (e.g. ['이론', '제국주의']).",
            },
            "slug": {
                "type": "string",
                "description": (
                    "Optional URL slug (lowercase a-z0-9 and dashes only, 1–80 chars). "
                    "Auto-generated from title if omitted; non-ASCII titles require an explicit slug."
                ),
            },
        },
        "required": ["title", "source_url", "selection_rationale", "context"],
    },
}


async def _exec_publish_hub_curation(
    title: str,
    source_url: str,
    selection_rationale: str,
    context: str,
    source_title: str | None = None,
    source_author: str | None = None,
    source_publication: str | None = None,
    source_published_at: str | None = None,
    tags: list | None = None,
    slug: str | None = None,
) -> str:
    # ── Validation ──
    title = (title or "").strip()
    source_url = (source_url or "").strip()
    selection_rationale = (selection_rationale or "").strip()
    context = (context or "").strip()
    if not title:
        return "Error: title is required."
    if not source_url or not source_url.startswith(("http://", "https://")):
        return "Error: source_url must be an http(s) URL."
    if not selection_rationale:
        return "Error: selection_rationale is required."
    if not context:
        return "Error: context is required."

    _ensure_hub_table()

    # ── Slug ──
    if slug:
        slug = slug.strip().lower()
        if not _SLUG_RE.match(slug):
            return "Error: slug must match ^[a-z0-9][a-z0-9-]{0,79}$ (lowercase alphanumeric + dashes)."
    else:
        base = _slugify(title)
        if not base:
            base = _slugify(source_title or "") or "curation"
        slug = base
    slug = _unique_slug(slug, "hub_curations")

    tag_list = [str(t)[:50] for t in (tags or [])][:20]

    row = db_query_one(
        """
        INSERT INTO hub_curations(
            slug, title, source_url, source_title, source_author,
            source_publication, source_published_at,
            selection_rationale, context, tags
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, published_at
        """,
        (
            slug, title, source_url, source_title, source_author,
            source_publication, source_published_at,
            selection_rationale, context, json.dumps(tag_list),
        ),
    )
    # Fetch the source article body and stash it on the row so the nightly ingest
    # job can chunk + embed it into lenin_corpus.modern_analysis later. Fetch here
    # (cheap, seconds) rather than embedding here (tens of seconds per long doc).
    # Failures are non-blocking — the curation is already committed and the job
    # will keep re-trying until source_content is populated.
    fetch_note = ""
    try:
        content = await _wrap_external_fetch(source_url)
        if content:
            db_execute(
                "UPDATE hub_curations SET source_content = %s WHERE id = %s",
                (content, row["id"]),
            )
            fetch_note = f"\nQueued for corpus ingest ({len(content):,} chars, embeds nightly)"
        else:
            fetch_note = "\nSource fetch returned empty — can be retried later"
    except Exception as e:
        logger.warning("hub curation source fetch failed for %s: %s", slug, e)
        fetch_note = f"\nSource fetch failed: {e} (can be retried later)"

    public_url = f"https://cyber-lenin.com/hub/{slug}"
    broadcast_note = ""
    try:
        br = await maybe_broadcast_autonomous_publication(
            title=title,
            url=public_url,
            body=f"{selection_rationale}\n\n{context}",
            source="cyber-lenin.com hub",
        )
        if br.ok:
            broadcast_note = f"\nTelegram channel broadcast: sent ({br.sent_count})"
    except Exception as e:
        logger.warning("hub curation channel broadcast failed for %s: %s", slug, e)
        broadcast_note = f"\nTelegram channel broadcast failed: {e}"
    return (
        f"Published hub curation #{row['id']}\n"
        f"Slug: {slug}\n"
        f"Public URL: {public_url}\n"
        f"Published at: {row['published_at']}"
        f"{fetch_note}"
        f"{broadcast_note}"
    )


async def _wrap_external_fetch(url: str) -> str | None:
    """Fetch the full article body for later corpus ingest. Max 500k chars."""
    from shared import fetch_url_content_async

    content = await fetch_url_content_async(url, max_chars=500_000)
    if not content or len(content.strip()) < 200:
        return None
    return content


# ══════════════════════════════════════════════════════════════════════
# Tool 2: publish_static_page
# ══════════════════════════════════════════════════════════════════════

PUBLISH_STATIC_PAGE_TOOL = {
    "name": "publish_static_page",
    "description": (
        "Write a standalone HTML page to cyber-lenin.com under the /p/{slug} route. "
        "The frontend wraps your HTML body inside the site's common layout (nav, "
        "footer, CSS) — you only provide the inner content. Use for pages that "
        "need custom formatting beyond markdown (layouts, visual structure, embedded "
        "media). Slug must be alphanumeric + dashes (lowercase). Path is sandboxed "
        "— you cannot write outside /home/grass/leninbot/static_pages/. Overwrites "
        "existing pages with the same slug (useful for iterating a draft)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "URL path slug (lowercase a-z0-9 and dashes, 1–80 chars). This becomes /p/{slug}.",
            },
            "title": {
                "type": "string",
                "description": "Page title (used in <title>, meta tags, and page header).",
            },
            "html_body": {
                "type": "string",
                "description": (
                    "HTML inner body. Do NOT include <html>, <head>, <body>, or <nav> — "
                    "the site layout wraps these. Use semantic tags (<article>, <section>, "
                    "<h2>, <p>, <figure>, etc.)."
                ),
            },
            "summary": {
                "type": "string",
                "description": "Optional short description for meta tags and the page-list index.",
            },
        },
        "required": ["slug", "title", "html_body"],
    },
}


async def _exec_publish_static_page(
    slug: str,
    title: str,
    html_body: str,
    summary: str | None = None,
) -> str:
    slug = (slug or "").strip().lower()
    title = (title or "").strip()
    html_body = (html_body or "").strip()
    summary = (summary or "").strip() or None

    if not slug:
        return "Error: slug is required."
    if not _SLUG_RE.match(slug):
        return "Error: slug must match ^[a-z0-9][a-z0-9-]{0,79}$ (lowercase alphanumeric + dashes only — ASCII)."
    if not title:
        return "Error: title is required."
    if not html_body:
        return "Error: html_body is required."

    # Forbid full-document HTML — the frontend provides the layout wrapper.
    # Use regex so semantic tags like <header> do not false-positive on '<head'.
    lower = html_body.lower()
    for forbidden_re, label in (
        (r"<html(?:\s|>)", "<html"),
        (r"<body(?:\s|>)", "<body"),
        (r"<head(?:\s|>)", "<head"),
        (r"</html\s*>", "</html"),
        (r"</body\s*>", "</body"),
        (r"</head\s*>", "</head"),
    ):
        if re.search(forbidden_re, lower):
            return (
                f"Error: html_body must be inner content only — found {label!r}. "
                "Provide semantic body content (<article>, <section>, etc.); the site "
                "wraps it with nav/layout/css."
            )

    STATIC_PAGES_DIR.mkdir(exist_ok=True)
    # Final safety: resolve the target inside the sandbox and verify containment.
    target = (STATIC_PAGES_DIR / f"{slug}.json").resolve()
    sandbox = STATIC_PAGES_DIR.resolve()
    if sandbox not in target.parents:
        return "Error: refusing to write outside static_pages sandbox."

    payload = {
        "slug": slug,
        "title": title,
        "summary": summary,
        "html_body": html_body,
        "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
    }
    try:
        await asyncio.to_thread(
            target.write_text,
            json.dumps(payload, ensure_ascii=False, indent=2),
            "utf-8",
        )
    except Exception as e:
        logger.error("publish_static_page write error: %s", e)
        return f"Failed to write static page: {e}"

    public_url = f"https://cyber-lenin.com/p/{slug}"
    broadcast_note = ""
    try:
        plain_excerpt = re.sub(r"<[^>]+>", " ", html_body)
        br = await maybe_broadcast_autonomous_publication(
            title=title,
            url=public_url,
            body=summary or plain_excerpt,
            source="cyber-lenin.com page",
        )
        if br.ok:
            broadcast_note = f"\nTelegram channel broadcast: sent ({br.sent_count})"
    except Exception as e:
        logger.warning("static page channel broadcast failed for %s: %s", slug, e)
        broadcast_note = f"\nTelegram channel broadcast failed: {e}"
    return (
        f"Published static page: {slug}\n"
        f"Local path: {target}\n"
        f"Public URL: {public_url}\n"
        f"Body size: {len(html_body)} chars"
        f"{broadcast_note}"
    )


# ══════════════════════════════════════════════════════════════════════
# Tool 3: publish_comic
# ══════════════════════════════════════════════════════════════════════

PUBLISH_COMIC_TOOL = {
    "name": "publish_comic",
    "description": (
        "Publish a 4-panel political comic to cyber-lenin.com under /p/{slug}. "
        "You author the scene SVG for each panel; the composer renders the panel "
        "frame and the speech balloon so those stay visually consistent. "
        "Panel viewBox is 960×320 (landscape, 4 panels stack vertically). The speech "
        "balloon occupies the rectangle (40, 28)–(420, 136) inside each panel — keep "
        "your scene content clear of that area. "
        "Visual vocabulary: reuse named-object templates from `assets/comic_icons/` "
        "(tv_news, missile_alert, chart_up/down, vault, goldbar_stack, dollar_bill, "
        "sanctions_stamp, torn_paper, speaker_head). Each icon is a 100×100 viewBox; "
        "copy its inner children and wrap in `<g transform=\"translate(x,y) scale(s)\">`. "
        "Recolor/relabel as needed. "
        "Content rule: each panel contains ONLY imagery and one short speech line. No "
        "captions, headings, subtext, transcripts, or analysis. Visual elements must be "
        "recognizable named objects — abstract rectangles/triangles/dashed circles "
        "without meaning are banned. A reader must parse each panel in ≤2 seconds. "
        "scene_svg is sanitized server-side (<script>, <style>, <iframe>, "
        "<foreignObject>, on* event handlers, javascript:/data: hrefs stripped); don't "
        "rely on them. Overwrites existing pages with the same slug — useful for iteration."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "URL slug (lowercase a–z0–9 and dashes, 1–80 chars). Becomes /p/{slug}.",
            },
            "title": {
                "type": "string",
                "description": "Comic title shown in the page <title>, meta tags, and list indices.",
            },
            "panels": {
                "type": "array",
                "description": "Exactly 4 panels, rendered top-to-bottom in a single column.",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "scene_svg": {
                            "type": "string",
                            "description": (
                                "Raw SVG children for the panel interior (no outer <svg> tag). "
                                "Typically a few `<g transform=\"translate(x, y) scale(s)\">…icon children…</g>` "
                                "groups arranging named-object icons. Avoid the balloon area at top-left."
                            ),
                        },
                        "speech": {
                            "type": "string",
                            "description": (
                                "One short Korean line that appears inside the speech balloon. Wraps to at "
                                "most 3 lines at ~22 chars per line — keep punchy. Empty string renders no balloon."
                            ),
                        },
                    },
                    "required": ["scene_svg", "speech"],
                },
            },
            "summary": {
                "type": "string",
                "description": "Optional short description for meta tags and the /reports research-tab listing.",
            },
        },
        "required": ["slug", "title", "panels"],
    },
}


async def _exec_publish_comic(
    slug: str,
    title: str,
    panels: list[dict],
    summary: str | None = None,
) -> str:
    # Lazy import to avoid a circular dependency: comic_composer imports
    # _exec_publish_static_page from this module for its own --publish CLI path.
    from scripts.comic_composer import build_page_payload

    slug = (slug or "").strip().lower()
    title = (title or "").strip()
    summary_in = (summary or "").strip() or title

    try:
        page = build_page_payload({
            "slug": slug,
            "title": title,
            "summary": summary_in,
            "panels": panels or [],
        })
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.error("publish_comic build error: %s", e)
        return f"Failed to build comic: {e}"

    return await _exec_publish_static_page(
        slug=page["slug"],
        title=page["title"],
        html_body=page["html_body"],
        summary=page.get("summary") or summary_in,
    )


# ══════════════════════════════════════════════════════════════════════
# Read helpers used by api.py (not agent tools)
# ══════════════════════════════════════════════════════════════════════

def list_hub_curations(limit: int = 20, offset: int = 0) -> list[dict]:
    _ensure_hub_table()
    rows = db_query(
        """
        SELECT id, slug, title, source_url, source_title, source_author,
               source_publication, source_published_at,
               selection_rationale, context, tags, published_at
          FROM hub_curations
         ORDER BY published_at DESC
         LIMIT %s OFFSET %s
        """,
        (int(limit), int(offset)),
    )
    return rows


def count_hub_curations() -> int:
    _ensure_hub_table()
    row = db_query_one("SELECT COUNT(*)::int AS n FROM hub_curations")
    return int(row["n"]) if row else 0


def get_hub_curation(slug: str) -> dict | None:
    _ensure_hub_table()
    if not _SLUG_RE.match(slug or ""):
        return None
    return db_query_one(
        """
        SELECT id, slug, title, source_url, source_title, source_author,
               source_publication, source_published_at,
               selection_rationale, context, tags, published_at
          FROM hub_curations
         WHERE slug = %s
        """,
        (slug,),
    )


def list_static_pages() -> list[dict]:
    STATIC_PAGES_DIR.mkdir(exist_ok=True)
    out: list[dict] = []
    for p in sorted(STATIC_PAGES_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "slug": data.get("slug") or p.stem,
                "title": data.get("title") or p.stem,
                "summary": data.get("summary"),
                "updated_at": data.get("updated_at"),
            })
        except Exception:
            continue
    return out


def get_static_page(slug: str) -> dict | None:
    if not _SLUG_RE.match(slug or ""):
        return None
    STATIC_PAGES_DIR.mkdir(exist_ok=True)
    target = (STATIC_PAGES_DIR / f"{slug}.json").resolve()
    sandbox = STATIC_PAGES_DIR.resolve()
    if sandbox not in target.parents or not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# Registration (imported by telegram_tools.py)
# ══════════════════════════════════════════════════════════════════════

SITE_PUBLISHING_TOOLS = [
    PUBLISH_HUB_CURATION_TOOL,
    PUBLISH_STATIC_PAGE_TOOL,
    PUBLISH_COMIC_TOOL,
]
SITE_PUBLISHING_TOOL_HANDLERS = {
    "publish_hub_curation": _exec_publish_hub_curation,
    "publish_static_page": _exec_publish_static_page,
    "publish_comic": _exec_publish_comic,
}
