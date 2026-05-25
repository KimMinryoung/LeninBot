"""runtime_tools.research — Publish, edit, and unpublish public research documents.

Public research documents are stored as Markdown rows in Supabase/Postgres and
served at https://cyber-lenin.com/reports/research/{slug}, where slug is the
filename without its .md extension. Legacy files under research/ and
output/research/ remain readable only as fallback/import sources.

This module backs the unified `research_document` runtime tool. Public
publication, edits, private saves, and visibility changes all flow through that
action-based interface so older tool names cannot be invoked directly.

`unpublish_public` is intentionally non-destructive: DB rows are marked private.
If a document only exists as a legacy fallback file, that file is relocated out
of the public-listing scope.

Mirrors the runtime_tools.post_edit pattern (UPDATE + cache purge in one step) for
DB-backed public content.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from telegram.channel_broadcast import maybe_broadcast_autonomous_publication
from autonomous_publication_controls import (
    check_autonomous_publication_allowed,
    is_autonomous_publication_context,
    record_autonomous_publication,
    record_autonomous_staged_draft,
    review_autonomous_publication,
    validate_autonomous_research_publication,
)
import research_store

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = _PROJECT_ROOT / "research"
LEGACY_RESEARCH_DIR = _PROJECT_ROOT / "output" / "research"
PRIVATE_RESEARCH_DIR = RESEARCH_DIR / "private"
PUBLICATION_DRAFT_DIR = _PROJECT_ROOT / "data" / "publication_drafts" / "research"
FRONTEND_DIR = os.getenv("FRONTEND_DIR", "/home/grass/frontend")
CF_PURGE_SCRIPT = os.getenv(
    "CF_PURGE_SCRIPT",
    os.path.join(FRONTEND_DIR, "scripts", "cloudflare-purge.js"),
)

KST = timezone(timedelta(hours=9))


# ── Helpers ──────────────────────────────────────────────────────────

# Filename allowlist: ASCII alnum + . _ - , must end in .md, no leading dot.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.md$")


def _validate_filename(filename: str) -> str:
    """Normalize and validate a research filename. Raises ValueError on bad input."""
    if not filename:
        raise ValueError("filename is required")
    fname = filename.strip()
    if not fname:
        raise ValueError("filename is required")
    if not fname.endswith(".md"):
        fname += ".md"
    if "/" in fname or "\\" in fname or ".." in fname:
        raise ValueError("filename must not contain path separators or '..'")
    if not _FILENAME_RE.fullmatch(fname):
        raise ValueError(
            "filename must be ASCII letters/digits with '.', '_', '-' only "
            "(no spaces, no leading dot)"
        )
    return fname


def _slug_from_title(title: str) -> str:
    """Build an ASCII slug from a title for filename use."""
    normalized = unicodedata.normalize("NFKD", title)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_only).strip("_")
    return slug[:80] if slug else "research"


def _resolve_existing(filename: str) -> Path | None:
    """Return existing file path (preferring research/ over legacy output/research/), else None."""
    primary = RESEARCH_DIR / filename
    if primary.is_file():
        return primary
    legacy = LEGACY_RESEARCH_DIR / filename
    if legacy.is_file():
        return legacy
    return None


def _atomic_write(path: Path, content: str) -> None:
    """Write content atomically: stage to .tmp then os.replace.

    Prevents a half-written file from appearing in the public listing if the
    write is interrupted (signal, OOM, disk-full).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _extract_h1(path: Path) -> str | None:
    """Return the first markdown H1 in the file, or None."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(40):
                line = fh.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    return stripped[2:].strip() or None
    except Exception:
        return None
    return None


_PUBLISH_DATE_RE = re.compile(
    r"^\s*(?:\*\*)?작성일(?::)?(?:\*\*)?\s*:?\s*(\d{4}-\d{2}-\d{2})"
)
_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$")
_LEADING_SCAFFOLD_META_RE = re.compile(
    r"^\s*(?:\*\*)?(?:작성자|작성일|Author|Date)(?::)?(?:\*\*)?\s*:?.*$",
    re.IGNORECASE,
)
_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$")
_BARE_NUMERIC_CITATION_RE = re.compile(r"(?<!\!)\[(\d+)\](?!\()")
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _extract_h1_from_markdown(markdown: str) -> str | None:
    for line in (markdown or "").splitlines()[:80]:
        m = _H1_RE.match(line)
        if m:
            title = m.group(1).strip()
            if title:
                return title
    return None


def _strip_leading_research_scaffold(markdown: str) -> str:
    """Accept either body markdown or a complete research document.

    research_document edit_public asks for body markdown, but agents often pass the
    full document returned by read_self. Strip only the canonical leading H1
    plus author/date/horizontal-rule scaffold so the tool does not duplicate
    the public header on every edit.
    """
    lines = (markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    idx = 0

    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx < len(lines) and _H1_RE.match(lines[idx]):
        idx += 1

    while True:
        start = idx
        while idx < len(lines) and not lines[idx].strip():
            idx += 1

        meta_count = 0
        while idx < len(lines) and _LEADING_SCAFFOLD_META_RE.match(lines[idx].strip()):
            idx += 1
            meta_count += 1
            while idx < len(lines) and not lines[idx].strip():
                idx += 1

        if meta_count and idx < len(lines) and _HR_RE.match(lines[idx].strip()):
            idx += 1
            continue

        if meta_count:
            break

        if idx == start:
            break

    return "\n".join(lines[idx:]).strip()


def _extract_publish_date(path: Path) -> str | None:
    """Find the existing `**작성일:** YYYY-MM-DD` line. Lets edit preserve original date."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(20):
                line = fh.readline()
                if not line:
                    break
                m = _PUBLISH_DATE_RE.match(line.strip())
                if m:
                    return m.group(1)
    except Exception:
        return None
    return None


def _build_document(title: str, content: str, publish_date: str) -> str:
    """Compose the canonical research-document layout."""
    return (
        f"# {title}\n"
        f"**작성자:** Cyber-Lenin (사이버-레닌)\n"
        f"**작성일:** {publish_date}\n\n"
        f"---\n\n"
        f"{content.strip()}\n"
    )


def _validate_public_citation_format(content: str) -> str | None:
    """Reject citation syntaxes that the public research renderer should not rely on."""
    text = _FENCED_CODE_RE.sub("", content or "")
    text = _INLINE_CODE_RE.sub("", text)
    if _BARE_NUMERIC_CITATION_RE.search(text):
        return (
            "Unsupported citation syntax: do not use bare numeric references like `[1]`. "
            "Use Markdown footnotes in body text as `[^1]`, `[^2]`, etc. and define them "
            "at the end as `[^1]: Publisher, title, date. https://example.com`."
        )
    return None


def _save_publication_draft(
    *,
    filename: str,
    title: str,
    document: str,
    fact_check_passed: bool,
    fact_check_notes: str | None,
) -> Path:
    """Persist the exact pre-publication document for later audit/recovery."""
    PUBLICATION_DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = PUBLICATION_DRAFT_DIR / f"{Path(filename).stem}.{ts}.{digest[:12]}.json"
    payload = {
        "kind": "research",
        "filename": filename,
        "title": title,
        "public_url": _public_url(filename),
        "content_sha256": digest,
        "fact_check_passed": bool(fact_check_passed),
        "fact_check_notes": (fact_check_notes or "").strip() or None,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "document": document,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _validate_fact_check_notes(notes: str | None) -> str | None:
    text = (notes or "").strip()
    if len(text) < 120:
        return (
            "fact_check_notes must be at least 120 characters and summarize the "
            "claims checked, sources consulted, and any corrections made"
        )
    source_markers = (
        "http://",
        "https://",
        "KG:",
        "knowledge_graph",
        "vector_search",
        "web_search",
        "fetch_url",
    )
    if not any(marker in text for marker in source_markers):
        return (
            "fact_check_notes must cite at least one verifiable source marker "
            "(URL, KG:, knowledge_graph, vector_search, web_search, or fetch_url)"
        )
    return None


def _format_draft_revision_guidance(*, filename: str, draft_path: Path, blocker: str | None = None) -> str:
    lines = []
    if blocker:
        lines.extend([
            "Publication metadata must be corrected before publication.",
            f"Tool requirement: {blocker}",
        ])
    else:
        lines.append("If your independent fact-check finds errors in this draft:")
    lines.extend([
        f"1. Review the saved draft backup if needed: {draft_path}",
        "2. Revise the draft text yourself before publishing. Correct the `content` argument in "
        "your next research_document(action=\"publish_public\") call and preserve the same "
        f"`slug` (`{filename}`) so the revised draft replaces this publication candidate.",
        "3. Re-check every affected proper noun, date, figure, current office, vote/seat count, "
        "quotation, and source attribution after editing.",
        "   For autonomous reports, also revise any stale or low-utility material so the report "
        "is useful at the 2026 current moment.",
        "4. Publish only after the corrected draft passes verification: call "
        "research_document(action=\"publish_public\") again with the revised `content`, "
        "the same `slug`, and `fact_check_notes` that cites sources, explicitly names "
        "corrections made, and states what was updated or retained for current usefulness.",
        "If the document is already public, use research_document(action=\"edit_public\", "
        "slug=..., content=...) instead of creating a duplicate publication.",
    ])
    return "\n".join(lines)


def _cache_safe_key(filename: str) -> str:
    """Mirror the frontend's safe-key transform: non-[A-Za-z0-9._-] → '_'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", filename)


def _invalidate_cache_sync(filename: str) -> dict[str, Any]:
    """Drop the per-file and list caches in Redis. Returns {ok, deleted, reason?}."""
    from redis_state import get_redis

    r = get_redis()
    if r is None:
        return {"ok": False, "deleted": 0, "reason": "redis_unavailable"}
    deleted = 0
    try:
        safe = _cache_safe_key(filename)
        deleted += int(r.delete(
            f"research:{safe}",
            f"research:{safe}:ko",
            f"research:{safe}:en",
            f"research:v3:{safe}:ko",
            f"research:v3:{safe}:en",
            "report:research_list",
            "report:research_list:ko",
            "report:research_list:en",
            "report:research_list:v3:ko",
            "report:research_list:v3:en",
        ) or 0)
    except Exception as e:
        logger.warning("research cache invalidation failed for %s: %s", filename, e)
        return {"ok": False, "deleted": deleted, "reason": f"{type(e).__name__}: {e}"}
    return {"ok": True, "deleted": deleted}


def _public_slug(filename: str) -> str:
    return filename[:-3] if filename.endswith(".md") else filename


def _public_url(filename: str) -> str:
    return f"https://cyber-lenin.com/reports/research/{_public_slug(filename)}"


def _cloudflare_purge_paths(filename: str) -> list[str]:
    slug = _public_slug(filename)
    return [
        f"/reports/research/{slug}",
        "/reports/research",
        "/reports",
        "/rss.xml",
        "/atom.xml",
        "/sitemap.xml",
    ]


def _purge_cloudflare_sync(filename: str) -> dict[str, Any]:
    paths = list(dict.fromkeys(_cloudflare_purge_paths(filename)))
    if not os.path.isfile(CF_PURGE_SCRIPT):
        return {
            "ok": False,
            "purged": 0,
            "urls": paths,
            "reason": f"script_missing: {CF_PURGE_SCRIPT}",
        }
    try:
        proc = subprocess.run(
            ["node", CF_PURGE_SCRIPT, *paths],
            cwd=FRONTEND_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except Exception as e:
        logger.warning("research Cloudflare purge failed before execution for %s: %s", filename, e)
        return {
            "ok": False,
            "purged": 0,
            "urls": paths,
            "reason": f"{type(e).__name__}: {e}",
        }

    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    if proc.returncode != 0:
        logger.warning("research Cloudflare purge failed for %s: %s", filename, output)
        return {
            "ok": False,
            "purged": 0,
            "urls": paths,
            "reason": output or f"exit_{proc.returncode}",
        }
    return {"ok": True, "purged": len(paths), "urls": paths, "output": output}


def _format_cache_note(cache: dict[str, Any], filename: str, *, missing_msg: str | None = None) -> str:
    if cache["ok"]:
        return f"cache invalidated ({cache['deleted']} key(s))"
    base = f"CACHE INVALIDATION FAILED ({cache['reason']})"
    safe = _cache_safe_key(filename)
    manual = (
        f"redis-cli DEL research:{safe} research:{safe}:ko research:{safe}:en "
        f"research:v3:{safe}:ko research:v3:{safe}:en "
        "report:research_list report:research_list:ko report:research_list:en "
        "report:research_list:v3:ko report:research_list:v3:en"
    )
    tail = f" — {missing_msg} Manually run: {manual}" if missing_msg else f" — readers may see stale data. Manually run: {manual}"
    return base + tail


def _format_invalidation_note(
    cache: dict[str, Any],
    cloudflare: dict[str, Any],
    filename: str,
    *,
    missing_msg: str | None = None,
) -> str:
    cache_note = _format_cache_note(cache, filename, missing_msg=missing_msg)
    if cloudflare["ok"]:
        cf_note = f"Cloudflare purged ({cloudflare['purged']} URL(s))"
    else:
        urls = " ".join(cloudflare.get("urls") or [])
        cf_note = (
            f"CLOUDFLARE PURGE FAILED ({cloudflare.get('reason', 'unknown')}) — "
            f"run `cd {FRONTEND_DIR} && node scripts/cloudflare-purge.js {urls}` manually"
        )
    return f"{cache_note}; {cf_note}"


# ── Public research document publication ───────────────────────────────

async def _exec_research_document_publish_public(
    title: str,
    content: str,
    filename: str | None = None,
    fact_check_passed: bool = False,
    fact_check_notes: str | None = None,
    source_task_id: int | None = None,
    broadcast: bool = True,
) -> str:
    if not title or not title.strip():
        return "Error: title is required."
    if not content or not content.strip():
        return "Error: content is required."

    title = title.strip()
    citation_error = _validate_public_citation_format(content)
    if citation_error:
        return f"Error: {citation_error}"

    now = datetime.now(KST)

    if filename:
        try:
            fname = _validate_filename(filename)
        except ValueError as e:
            return f"Error: {e}."
    elif is_autonomous_publication_context():
        return (
            "Error: autonomous research_document public actions require an explicit stable slug. "
            "Pass the same slug to stage_public and publish_public so later ticks update the same report."
        )
    else:
        fname = f"{now.strftime('%Y%m%d')}_{_slug_from_title(title)}.md"

    document = _build_document(title, content, now.strftime("%Y-%m-%d"))
    try:
        draft_path = await asyncio.to_thread(
            _save_publication_draft,
            filename=fname,
            title=title,
            document=document,
            fact_check_passed=fact_check_passed is True,
            fact_check_notes=fact_check_notes,
        )
    except Exception as e:
        logger.error("publication draft backup failed for %s: %s", fname, e)
        return f"Error: failed to back up draft before publication: {type(e).__name__}: {e}"

    if fact_check_passed is not True:
        existing_before = await asyncio.to_thread(
            research_store.get_document, fname, include_private=True
        )
        if existing_before and existing_before.get("status") == "public":
            return (
                "Error: staged draft would overwrite an already-public research document.\n"
                f"Draft backup: {draft_path}\n"
                f"Existing public document: {fname}\n"
                "Use action='edit_public' for public-document revisions, or choose a new stable slug "
                "for a new research document."
            )
        try:
            row, _ = await asyncio.to_thread(
                research_store.upsert_document,
                filename=fname,
                title=title,
                markdown=document,
                summary=research_store.extract_excerpt(document),
                status="staged",
                source_task_id=source_task_id,
            )
        except Exception as e:
            logger.error("research_document stage_public DB write error for %s: %s", fname, e)
            return (
                "Error: failed to store staged draft after backup.\n"
                f"Draft backup: {draft_path}\n"
                f"Storage error: {type(e).__name__}: {e}"
            )
        draft_meta = {"filename": fname, "research_document_id": row["id"], "status": "staged"}
        if source_task_id is not None:
            draft_meta["source_task_id"] = source_task_id
        record_autonomous_staged_draft(
            publication_kind="research",
            title=title,
            public_url=_public_url(fname),
            meta=draft_meta,
        )
        return (
            "Draft saved, not published.\n"
            f"Draft backup: {draft_path}\n"
            f"Storage: research_documents id={row['id']} status=staged sha256={row['content_sha256'][:12]}\n"
            f"Candidate filename: {fname}\n"
            f"Candidate public URL: {_public_url(fname)}\n"
            "Before publishing, fact-check proper nouns, dates, numerical claims, seat/vote counts, "
            "current offices, quotations, source attributions, and whether any claims or framing are "
            "stale at the 2026 current moment. Then call research_document publish_public again "
            "with fact_check_passed=true and fact_check_notes listing the checked claims, sources, "
            "corrections made, and current-usefulness revisions.\n\n"
            f"{_format_draft_revision_guidance(filename=fname, draft_path=draft_path)}"
        )

    fact_check_error = _validate_fact_check_notes(fact_check_notes)
    if fact_check_error:
        return (
            "Error: publication blocked after draft backup.\n"
            f"Draft backup: {draft_path}\n"
            f"{_format_draft_revision_guidance(filename=fname, draft_path=draft_path, blocker=fact_check_error)}"
        )

    public_url = _public_url(fname)
    if is_autonomous_publication_context():
        gate_error = validate_autonomous_research_publication(
            title=title,
            content=content,
            identifier=fname,
            fact_check_notes=fact_check_notes,
        )
        if gate_error:
            return (
                "Publication blocked after draft backup.\n"
                f"Draft backup: {draft_path}\n"
                f"{gate_error}\n\n"
                f"{_format_draft_revision_guidance(filename=fname, draft_path=draft_path, blocker=gate_error)}"
            )

    allowed, reason = check_autonomous_publication_allowed("research")
    if not allowed:
        return (
            "Publication blocked after draft backup.\n"
            f"Draft backup: {draft_path}\n"
            f"{reason}"
        )
    review_note = await review_autonomous_publication(
        publication_kind="research",
        title=title,
        content=content,
        public_url=public_url,
    )

    existing_before = await asyncio.to_thread(
        research_store.get_document, fname, include_private=True
    )
    was_already_public = bool(existing_before and existing_before.get("status") == "public")

    try:
        row, is_overwrite = await asyncio.to_thread(
            research_store.upsert_document,
            filename=fname,
            title=title,
            markdown=document,
            summary=research_store.extract_excerpt(document),
            status="public",
            source_task_id=source_task_id,
        )
    except Exception as e:
        logger.error("research_document publish_public DB write error for %s: %s", fname, e)
        return f"Error: failed to store {fname}: {type(e).__name__}: {e}"

    cache = await asyncio.to_thread(_invalidate_cache_sync, fname)
    cloudflare = await asyncio.to_thread(_purge_cloudflare_sync, fname)
    status = "Updated public document" if was_already_public else "Published"
    broadcast_note = ""
    if broadcast:
        try:
            br = await maybe_broadcast_autonomous_publication(
                title=title,
                url=public_url,
                body=content,
                source="cyber-lenin.com research",
            )
            if br.ok:
                broadcast_note = f"\nTelegram channel broadcast: sent ({br.sent_count})"
                if getattr(br, "message_ids", None):
                    try:
                        from publication_records import record_publication_broadcast_sync

                        await asyncio.to_thread(
                            record_publication_broadcast_sync,
                            slug=row["slug"],
                            public_url=public_url,
                            channel_message_ids=br.message_ids,
                            source="research_document_publish_public",
                        )
                        broadcast_note += f"; tracked {len(br.message_ids or [])} message id(s)"
                    except Exception as e:
                        logger.warning("publication broadcast record failed for %s: %s", fname, e)
                        broadcast_note += f"; message-id tracking failed ({e})"
        except Exception as e:
            logger.warning("research channel broadcast failed for %s: %s", fname, e)
            broadcast_note = f"\nTelegram channel broadcast failed: {e}"
    else:
        broadcast_note = "\nTelegram channel broadcast: skipped by request"
    if not was_already_public:
        meta = {"filename": fname, "research_document_id": row["id"]}
        if source_task_id is not None:
            meta["source_task_id"] = source_task_id
        record_autonomous_publication(
            publication_kind="research",
            title=title,
            public_url=public_url,
            meta=meta,
        )
    return (
        f"{status}: {fname}\n"
        f"Storage: research_documents id={row['id']} sha256={row['content_sha256'][:12]}\n"
        f"Public URL: {public_url}\n"
        f"{review_note}\n"
        f"Size: {len(document)} chars; {_format_invalidation_note(cache, cloudflare, fname)}"
        f"{broadcast_note}"
    )


# ── Public research document edits/visibility ──────────────────────────

def _unpublish_sync(existing: Path) -> Path:
    """Move file to research/private/ with a collision-safe destination. Returns new path."""
    PRIVATE_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    dest = PRIVATE_RESEARCH_DIR / existing.name
    if dest.exists():
        ts = datetime.now(KST).strftime("%Y%m%dT%H%M%S")
        dest = PRIVATE_RESEARCH_DIR / f"{existing.stem}.{ts}{existing.suffix}"
    os.replace(existing, dest)
    return dest


def _extract_publish_date_from_markdown(markdown: str) -> str | None:
    for line in markdown.splitlines()[:20]:
        m = _PUBLISH_DATE_RE.match(line.strip())
        if m:
            return m.group(1)
    return None


async def _exec_research_document_edit_public(
    operation: str,
    filename: str,
    title: str | None = None,
    content: str | None = None,
    broadcast: bool = True,
    fact_check_notes: str | None = None,
) -> str:
    op = (operation or "").strip().lower()
    if op not in {"edit", "unpublish", "publish"}:
        return "Error: operation must be 'edit', 'unpublish', or 'publish'."
    try:
        fname = _validate_filename(filename)
    except ValueError as e:
        return f"Error: {e}."

    existing_doc = await asyncio.to_thread(research_store.get_document, fname, include_private=True)
    existing = None if existing_doc else _resolve_existing(fname)
    if existing_doc is None and existing is None:
        return f"Error: no research document named '{fname}' in DB or legacy fallback files."

    if op == "edit":
        if not content or not content.strip():
            return "Error: content is required for operation='edit'."
        existing_markdown = existing_doc["markdown"] if existing_doc else existing.read_text(encoding="utf-8")
        existing_title = (
            existing_doc.get("title") if existing_doc else None
        ) or research_store.extract_title(existing_markdown, "") or (None if existing is None else _extract_h1(existing))
        content_title = _extract_h1_from_markdown(content)
        new_title = (title or "").strip() or content_title or existing_title
        if not new_title:
            return "Error: existing document has no title/H1 — pass `title` explicitly."
        body = _strip_leading_research_scaffold(content)
        if not body:
            return "Error: content has no body after removing the research-document header."
        citation_error = _validate_public_citation_format(body)
        if citation_error:
            return f"Error: {citation_error}"
        draft_path = None
        review_note = ""
        if is_autonomous_publication_context():
            fact_check_error = _validate_fact_check_notes(fact_check_notes)
            if fact_check_error:
                return f"Error: autonomous edit_public requires fact_check_notes. {fact_check_error}."
            gate_error = validate_autonomous_research_publication(
                title=new_title,
                content=body,
                identifier=fname,
                fact_check_notes=fact_check_notes,
            )
            if gate_error:
                return gate_error
        publish_date = (
            _extract_publish_date_from_markdown(existing_markdown)
            or (None if existing is None else _extract_publish_date(existing))
            or datetime.now(KST).strftime("%Y-%m-%d")
        )
        document = _build_document(new_title, body, publish_date)
        if is_autonomous_publication_context():
            try:
                draft_path = await asyncio.to_thread(
                    _save_publication_draft,
                    filename=fname,
                    title=new_title,
                    document=document,
                    fact_check_passed=True,
                    fact_check_notes=fact_check_notes,
                )
            except Exception as e:
                logger.error("publication edit draft backup failed for %s: %s", fname, e)
                return f"Error: failed to back up edited draft before publication: {type(e).__name__}: {e}"
            review_note = await review_autonomous_publication(
                publication_kind="research_edit",
                title=new_title,
                content=body,
                public_url=_public_url(fname),
            )
        try:
            row, _ = await asyncio.to_thread(
                research_store.upsert_document,
                filename=fname,
                title=new_title,
                markdown=document,
                summary=research_store.extract_excerpt(document),
                status="public",
            )
        except Exception as e:
            logger.error("research_document edit_public DB write error for %s: %s", fname, e)
            return f"Error: failed to rewrite {fname}: {type(e).__name__}: {e}"

        cache = await asyncio.to_thread(_invalidate_cache_sync, fname)
        cloudflare = await asyncio.to_thread(_purge_cloudflare_sync, fname)
        draft_note = f"Draft backup: {draft_path}\n" if draft_path else ""
        review_line = f"{review_note}\n" if review_note else ""
        return (
            f"Edited: {fname}\n"
            f"{draft_note}"
            f"Storage: research_documents id={row['id']} sha256={row['content_sha256'][:12]}\n"
            f"Public URL: {_public_url(fname)}\n"
            f"{review_line}"
            f"Title: {new_title}; size: {len(document)} chars; {_format_invalidation_note(cache, cloudflare, fname)}"
        )

    if op == "publish":
        if not existing_doc:
            return f"Error: cannot publish legacy fallback file '{fname}' because no DB row exists. Import it into research_documents first."
        if existing_doc.get("status") == "public":
            return f"Already public: {fname}\nPublic URL: {_public_url(fname)}"
        if is_autonomous_publication_context():
            body = _strip_leading_research_scaffold(existing_doc.get("markdown") or "")
            title_for_gate = existing_doc.get("title") or research_store.extract_title(existing_doc.get("markdown") or "", "")
            return await _exec_research_document_publish_public(
                title=title_for_gate or "",
                content=body,
                filename=fname,
                fact_check_passed=True,
                fact_check_notes=fact_check_notes,
                source_task_id=existing_doc.get("source_task_id"),
                broadcast=broadcast,
            )
        try:
            row = await asyncio.to_thread(research_store.set_status, fname, "public")
            if not row:
                return f"Error: no private research document named '{fname}' in DB."
        except Exception as e:
            logger.error("research_document republish_public DB error for %s: %s", fname, e)
            return f"Error: failed to mark {fname} public: {type(e).__name__}: {e}"

        cache = await asyncio.to_thread(_invalidate_cache_sync, fname)
        cloudflare = await asyncio.to_thread(_purge_cloudflare_sync, fname)
        public_url = _public_url(fname)
        broadcast_note = ""
        if broadcast:
            try:
                br = await maybe_broadcast_autonomous_publication(
                    title=row["title"],
                    url=public_url,
                    body=row.get("markdown") or "",
                    source="cyber-lenin.com research visibility change",
                )
                if br.ok:
                    broadcast_note = f"\nTelegram channel broadcast: sent ({br.sent_count})"
                    if getattr(br, "message_ids", None):
                        try:
                            from publication_records import record_publication_broadcast_sync

                            await asyncio.to_thread(
                                record_publication_broadcast_sync,
                                slug=row["slug"],
                                public_url=public_url,
                                channel_message_ids=br.message_ids,
                                source="research_document_republish_public",
                            )
                            broadcast_note += f"; tracked {len(br.message_ids or [])} message id(s)"
                        except Exception as e:
                            logger.warning("publication broadcast record failed for %s: %s", fname, e)
                            broadcast_note += f"; message-id tracking failed ({e})"
                else:
                    broadcast_note = f"\nTelegram channel broadcast skipped/failed: {br.message}"
            except Exception as e:
                logger.warning("research publish channel broadcast failed for %s: %s", fname, e)
                broadcast_note = f"\nTelegram channel broadcast failed: {e}"
        return (
            f"Published existing private research document: {fname}\n"
            f"Storage: research_documents id={row['id']} sha256={row['content_sha256'][:12]}\n"
            f"Public URL: {public_url}\n"
            f"{_format_invalidation_note(cache, cloudflare, fname)}"
            f"{broadcast_note}"
        )

    # operation == "unpublish"
    backup_note = ""
    if existing_doc:
        try:
            row = await asyncio.to_thread(research_store.set_status, fname, "private")
            backup_note = f"Storage: research_documents id={row['id']} status=private"
        except Exception as e:
            logger.error("research_document unpublish_public DB error for %s: %s", fname, e)
            return f"Error: failed to mark {fname} private: {type(e).__name__}: {e}"
    else:
        try:
            new_path = await asyncio.to_thread(_unpublish_sync, existing)
            backup_note = f"Backup path: {new_path}"
        except Exception as e:
            logger.error("research_document unpublish_public error for %s: %s", fname, e)
            return f"Error: failed to move {fname} to private/: {type(e).__name__}: {e}"

    cache = await asyncio.to_thread(_invalidate_cache_sync, fname)
    cloudflare = await asyncio.to_thread(_purge_cloudflare_sync, fname)
    cache_note = _format_invalidation_note(
        cache,
        cloudflare,
        fname,
        missing_msg="document was unpublished but the cached copy may still be served." if not cache["ok"] else None,
    )
    delete_note = ""
    try:
        from publication_records import delete_broadcasts_for_slug

        delete_result = await delete_broadcasts_for_slug(_public_slug(fname))
        delete_note = (
            "\nTelegram channel cleanup: "
            f"attempted={delete_result.get('attempted', 0)} "
            f"deleted={delete_result.get('deleted', 0)} "
            f"failed={delete_result.get('failed', 0)} "
            f"({delete_result.get('message')})"
        )
    except Exception as e:
        logger.warning("research unpublish channel cleanup failed for %s: %s", fname, e)
        delete_note = f"\nTelegram channel cleanup failed: {e}"
    return (
        f"Unpublished: {fname}\n"
        f"{backup_note}\n"
        f"Public URL (now 404): {_public_url(fname)}\n"
        f"{cache_note}"
        f"{delete_note}"
    )


# ── Unified research_document tool ───────────────────────────────────

RESEARCH_DOCUMENT_TOOL = {
    "name": "research_document",
    "description": (
        "Create, publish, edit, unpublish, republish, or privately save a markdown "
        "research document. Public and private documents are the same content family; "
        "private documents are simply unpublished research documents. The public "
        "publishing flow is two-step: action='stage_public' saves an exact draft and "
        "does not publish; action='publish_public' requires fact_check_notes and publishes "
        "the checked version. Use this tool for research documents only. Use edit_content "
        "for diary, task report, blog post, and hub curation edits. Citation format is fixed "
        "for website rendering: cite sources in body text only as Markdown footnotes `[^1]`, "
        "`[^2]`, etc.; end the document with matching footnote definitions that contain URLs, "
        "e.g. `[^1]: Publisher, title, date. https://example.com`. Do not invent other "
        "citation formats such as bare `[1]`, numbered source lists, parenthetical source "
        "notes, or raw body URLs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "stage_public",
                    "publish_public",
                    "edit_public",
                    "unpublish_public",
                    "republish_public",
                    "save_private",
                    "publish_private",
                ],
                "description": "Research-document mutation to perform. Autonomous public-bound actions require an explicit stable slug.",
            },
            "title": {"type": "string", "description": "Document title or optional replacement title."},
            "content": {
                "type": "string",
                "description": (
                    "Markdown body/content for public stage/publish/edit. Use only `[^n]` "
                    "body citations and final `[^n]: description URL` definitions; do not use bare `[n]`."
                ),
            },
            "slug": {"type": "string", "description": "Stable research document slug. '.md' is optional. Required for autonomous public-bound actions."},
            "id": {"type": "integer", "description": "Optional private research document id for compatibility reads/edits."},
            "markdown_body": {"type": "string", "description": "Markdown body for action='save_private'."},
            "body": {"type": "string", "description": "Optional replacement Markdown body for action='publish_private'."},
            "source_task_id": {"type": "integer", "description": "Optional originating task id for private saves."},
            "fact_check_notes": {
                "type": "string",
                "description": "Required for autonomous public-bound actions: publish_public, edit_public, republish_public, and publish_private. Summarize checked claims, sources, and corrections.",
            },
            "broadcast": {
                "type": "boolean",
                "description": "For publish_private/republish_public and manual visibility changes: whether to broadcast to Telegram. Default true.",
                "default": True,
            },
        },
        "required": ["action"],
    },
}


def _filename_from_slug(slug: str | None) -> str | None:
    if not slug:
        return None
    return _validate_filename(slug)


async def _exec_research_document(
    action: str,
    title: str | None = None,
    content: str | None = None,
    slug: str | None = None,
    id: int | None = None,
    markdown_body: str | None = None,
    body: str | None = None,
    source_task_id: int | None = None,
    fact_check_notes: str | None = None,
    broadcast: bool = True,
) -> str:
    op = (action or "").strip().lower()
    if op in {"stage_public", "publish_public"}:
        return await _exec_research_document_publish_public(
            title=title or "",
            content=content or "",
            filename=slug,
            fact_check_passed=(op == "publish_public"),
            fact_check_notes=fact_check_notes,
            source_task_id=source_task_id,
            broadcast=broadcast,
        )
    if op == "edit_public":
        try:
            filename = _filename_from_slug(slug)
        except ValueError as e:
            return f"Error: {e}."
        return await _exec_research_document_edit_public(
            operation="edit",
            filename=filename or "",
            title=title,
            content=content,
            broadcast=broadcast,
            fact_check_notes=fact_check_notes,
        )
    if op == "unpublish_public":
        try:
            filename = _filename_from_slug(slug)
        except ValueError as e:
            return f"Error: {e}."
        return await _exec_research_document_edit_public(operation="unpublish", filename=filename or "", broadcast=broadcast)
    if op == "republish_public":
        try:
            filename = _filename_from_slug(slug)
        except ValueError as e:
            return f"Error: {e}."
        return await _exec_research_document_edit_public(
            operation="publish",
            filename=filename or "",
            broadcast=broadcast,
            fact_check_notes=fact_check_notes,
        )
    if op == "save_private":
        if not title or not slug or not markdown_body:
            return "Error: action='save_private' requires title, slug, and markdown_body."
        from runtime_tools.private_reports import _exec_save_private_report

        return await _exec_save_private_report(
            title=title,
            slug=slug,
            markdown_body=markdown_body,
            source_task_id=source_task_id,
        )
    if op == "publish_private":
        if not slug:
            return "Error: action='publish_private' requires slug."
        if is_autonomous_publication_context():
            from runtime_tools.private_reports import get_private_report_sync

            clean_slug = slug[:-3] if slug.endswith(".md") else slug
            try:
                private = await asyncio.to_thread(get_private_report_sync, slug=clean_slug)
            except Exception as e:
                return f"Error: failed to read private research document before autonomous publication: {type(e).__name__}: {e}"
            if not private:
                return f"Error: no private research document found for slug={clean_slug!r}."
            markdown_source = body if body is not None and body.strip() else (content if content is not None and content.strip() else private.get("markdown") or "")
            public_title = (title or "").strip() or research_store.extract_title(markdown_source, private.get("title") or "")
            public_body = _strip_leading_research_scaffold(markdown_source)
            return await _exec_research_document_publish_public(
                title=public_title or "",
                content=public_body,
                filename=f"{clean_slug}.md",
                fact_check_passed=True,
                fact_check_notes=fact_check_notes,
                source_task_id=private.get("source_task_id"),
                broadcast=broadcast,
            )
        from runtime_tools.private_reports import _exec_publish_private_report

        return await _exec_publish_private_report(
            slug=slug,
            body=body if body is not None else content,
            title=title,
            broadcast=broadcast,
        )
    return (
        "Error: action must be one of stage_public, publish_public, edit_public, "
        "unpublish_public, republish_public, save_private, publish_private."
    )


# ── Registry exports ─────────────────────────────────────────────────

RESEARCH_TOOLS = [RESEARCH_DOCUMENT_TOOL]
RESEARCH_TOOL_HANDLERS = {
    "research_document": _exec_research_document,
}
