#!/usr/bin/env python3
"""English translation pipeline for published static pages.

The default export/import flow supports manual review. The DeepL command can
batch-translate pages directly with the low-cost DeepL Translation API, then
reuses the same HTML and residual-Korean validation before writing fields into
static_pages/{slug}.json.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secrets_loader import get_secret
from site_publishing import STATIC_PAGES_DIR, _SLUG_RE, _validate_inner_html

TRANSLATION_DIR = ROOT / "output" / "static_page_translations"
DEEPL_MAX_REQUEST_BYTES = 120_000
DEEPL_TIMEOUT_SECONDS = 90
_TAG_OR_COMMENT_RE = re.compile(r"(<!--.*?-->|<[^>]+>)", re.DOTALL)
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")

PROMPT = """You are a meticulous Korean-to-English translation editor.

Translate this Cyber-Lenin research page into polished, publication-quality
English. Preserve the author's analytical stance and terminology. Do not
summarize, omit, soften, expand, or fact-check the argument.

Critical HTML rules:
- Return the same inner HTML structure as the source html_body.
- Translate human-readable Korean text only.
- Preserve all tags, attributes, URLs, numbers, table structure, inline styles,
  emojis, and relative links exactly unless the visible Korean text itself is
  inside the attribute value.
- Do not add <html>, <head>, or <body>.

Return only a JSON object with these string keys:
title_en, summary_en, html_body_en
"""


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _page_paths(slugs: list[str]) -> list[Path]:
    return [_page_path(slug) for slug in slugs] if slugs else sorted(STATIC_PAGES_DIR.glob("*.json"))


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = _strip_fences(text)
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


def _tag_sequence(html: str) -> list[str]:
    without_comments = re.sub(r"<!--.*?-->", "", html or "", flags=re.DOTALL)
    tags = re.findall(r"<\s*/?\s*([a-zA-Z][a-zA-Z0-9:-]*)\b", without_comments)
    return [tag.lower() for tag in tags]


def _hangul_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    hangul = sum(1 for ch in letters if "\uac00" <= ch <= "\ud7a3")
    return hangul / len(letters)


def _split_edge_whitespace(text: str) -> tuple[str, str, str]:
    prefix_len = len(text) - len(text.lstrip())
    suffix_len = len(text) - len(text.rstrip())
    if suffix_len:
        return text[:prefix_len], text[prefix_len:-suffix_len], text[-suffix_len:]
    return text[:prefix_len], text[prefix_len:], ""


def _deepl_translate_texts(
    texts: list[str],
    *,
    api_key: str,
    api_base: str,
    target_lang: str,
    source_lang: str,
    tag_handling: str | None = None,
) -> list[str]:
    import requests

    if not texts:
        return []
    payload: dict[str, Any] = {
        "text": texts,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "preserve_formatting": True,
    }
    if tag_handling:
        payload["tag_handling"] = tag_handling

    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(encoded) > DEEPL_MAX_REQUEST_BYTES:
        raise ValueError(
            f"DeepL request too large ({len(encoded):,} bytes). "
            "Use --html-mode=segments or split this page before translating."
        )

    response = requests.post(
        f"{api_base}/v2/translate",
        headers=_deepl_headers(api_key),
        data=encoded,
        timeout=DEEPL_TIMEOUT_SECONDS,
    )
    if response.status_code == 456:
        raise RuntimeError("DeepL quota exceeded.")
    if not response.ok:
        raise RuntimeError(f"DeepL translate failed: HTTP {response.status_code}: {response.text[:500]}")
    data = response.json()
    translations = data.get("translations") or []
    if len(translations) != len(texts):
        raise RuntimeError(f"DeepL returned {len(translations)} translations for {len(texts)} fields")
    return [str(item.get("text") or "").strip() for item in translations]


def _translate_html_segments(
    html_body: str,
    *,
    api_key: str,
    api_base: str,
    target_lang: str,
    source_lang: str,
) -> str:
    """Translate only text nodes while preserving tag bytes exactly.

    DeepL's native HTML mode can legally rearrange inline tags for target-language
    word order. Our stored pages are layout-heavy, so exact tag preservation is
    more important than letting the translator move markup around.
    """
    parts = _TAG_OR_COMMENT_RE.split(html_body or "")
    text_indexes: list[int] = []
    cores: list[str] = []
    edges: dict[int, tuple[str, str]] = {}
    for idx, part in enumerate(parts):
        if not part or part.startswith("<"):
            continue
        if not _HANGUL_RE.search(part):
            continue
        prefix, core, suffix = _split_edge_whitespace(part)
        if not core.strip():
            continue
        text_indexes.append(idx)
        cores.append(html.unescape(core))
        edges[idx] = (prefix, suffix)

    translated = _deepl_translate_texts(
        cores,
        api_key=api_key,
        api_base=api_base,
        target_lang=target_lang,
        source_lang=source_lang,
    )
    for idx, value in zip(text_indexes, translated):
        prefix, suffix = edges[idx]
        parts[idx] = prefix + html.escape(value, quote=False) + suffix
    return "".join(parts)


def _validate_translation(source: dict[str, Any], translated: dict[str, Any], *, max_hangul_ratio: float) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key in ("title_en", "summary_en", "html_body_en"):
        value = translated.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"translation missing non-empty {key}")
        cleaned[key] = value.strip()

    validation_error = _validate_inner_html(cleaned["html_body_en"], "html_body_en")
    if validation_error:
        raise ValueError(validation_error)

    source_tags = _tag_sequence(str(source.get("html_body") or ""))
    translated_tags = _tag_sequence(cleaned["html_body_en"])
    if source_tags != translated_tags:
        raise ValueError(
            "translated HTML tag sequence differs from source "
            f"(source={len(source_tags)} tags, translated={len(translated_tags)} tags)"
        )

    ratio = _hangul_ratio(" ".join(cleaned.values()))
    if ratio > max_hangul_ratio:
        raise ValueError(f"translation still contains too much Hangul ({ratio:.1%}; max {max_hangul_ratio:.1%})")

    return cleaned


def _page_path(slug: str) -> Path:
    normalized = slug.strip().lower()
    if not _SLUG_RE.match(normalized):
        raise ValueError(f"invalid slug: {slug}")
    return STATIC_PAGES_DIR / f"{normalized}.json"


def _source_payload(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "slug": page.get("slug"),
        "title": page.get("title"),
        "summary": page.get("summary") or "",
        "html_body": page.get("html_body") or "",
    }


def _export(slugs: list[str], *, force: bool) -> int:
    paths = _page_paths(slugs)
    exported = 0
    for path in paths:
        if not path.is_file():
            print(f"missing: {path}", file=sys.stderr)
            continue
        page = _load_json(path)
        if not force and str(page.get("html_body_en") or "").strip():
            continue

        slug = str(page.get("slug") or path.stem)
        payload = _source_payload(page)
        source_path = TRANSLATION_DIR / f"{slug}.source.json"
        prompt_path = TRANSLATION_DIR / f"{slug}.prompt.md"
        output_path = TRANSLATION_DIR / f"{slug}.translation.json"

        _write_json(source_path, payload)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            PROMPT
            + "\n\nPaste/attach this source JSON, then return the translated JSON only:\n\n"
            + "```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
        print(f"exported: {slug}")
        print(f"  prompt:      {prompt_path}")
        print(f"  source JSON: {source_path}")
        print(f"  import path: {output_path}")
        exported += 1
    if exported == 0:
        print("No static pages need translation.")
    return 0


def _import_one(slug: str, translation_path: Path, *, max_hangul_ratio: float, dry_run: bool) -> None:
    page_path = _page_path(slug)
    if not page_path.is_file():
        raise FileNotFoundError(page_path)
    page = _load_json(page_path)
    translated = _parse_json_object(translation_path.read_text(encoding="utf-8"))
    cleaned = _validate_translation(page, translated, max_hangul_ratio=max_hangul_ratio)
    page.update(cleaned)
    if dry_run:
        print(f"dry-run ok: {slug} ({len(cleaned['html_body_en']):,} English HTML chars)")
        return
    _write_json(page_path, page)
    print(f"updated: {page_path} ({len(cleaned['html_body_en']):,} English HTML chars)")


def _import(slugs: list[str], *, max_hangul_ratio: float, dry_run: bool) -> int:
    if slugs:
        targets = [(slug, TRANSLATION_DIR / f"{slug}.translation.json") for slug in slugs]
    else:
        targets = [(path.name.removesuffix(".translation.json"), path) for path in sorted(TRANSLATION_DIR.glob("*.translation.json"))]

    if not targets:
        print(f"No translation JSON files found in {TRANSLATION_DIR}.")
        return 0

    failures = 0
    for slug, translation_path in targets:
        if not translation_path.is_file():
            print(f"missing translation: {translation_path}", file=sys.stderr)
            failures += 1
            continue
        try:
            _import_one(slug, translation_path, max_hangul_ratio=max_hangul_ratio, dry_run=dry_run)
        except Exception as exc:
            print(f"failed: {slug}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def _deepl_api_base(api_key: str, explicit_base: str | None = None) -> str:
    if explicit_base:
        return explicit_base.rstrip("/")
    if api_key.endswith(":fx"):
        return "https://api-free.deepl.com"
    return "https://api.deepl.com"


def _deepl_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"DeepL-Auth-Key {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "leninbot-static-page-translation/1.0",
    }


def _deepl_translate_fields(
    page: dict[str, Any],
    *,
    api_key: str,
    api_base: str,
    target_lang: str,
    source_lang: str,
    html_mode: str,
) -> dict[str, Any]:
    title = str(page.get("title") or "")
    summary = str(page.get("summary") or "")
    html_body = str(page.get("html_body") or "")
    translated_meta = _deepl_translate_texts(
        [title, summary or title],
        api_key=api_key,
        api_base=api_base,
        target_lang=target_lang,
        source_lang=source_lang,
    )
    out = {
        "title_en": translated_meta[0] if translated_meta else "",
        "summary_en": translated_meta[1] if len(translated_meta) > 1 else "",
        "html_body_en": "",
    }
    if html_mode == "segments":
        out["html_body_en"] = _translate_html_segments(
            html_body,
            api_key=api_key,
            api_base=api_base,
            target_lang=target_lang,
            source_lang=source_lang,
        )
    else:
        html_translated = _deepl_translate_texts(
            [html_body],
            api_key=api_key,
            api_base=api_base,
            target_lang=target_lang,
            source_lang=source_lang,
            tag_handling="html",
        )
        out["html_body_en"] = html_translated[0] if html_translated else ""
    if not out["summary_en"]:
        out["summary_en"] = out["title_en"]
    return out


def _deepl_usage(*, api_key: str, api_base: str) -> dict[str, Any]:
    import requests

    response = requests.get(
        f"{api_base}/v2/usage",
        headers=_deepl_headers(api_key),
        timeout=DEEPL_TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(f"DeepL usage check failed: HTTP {response.status_code}: {response.text[:500]}")
    return response.json()


def _translate_deepl(
    slugs: list[str],
    *,
    force: bool,
    dry_run: bool,
    max_hangul_ratio: float,
    max_chars: int,
    target_lang: str,
    source_lang: str,
    api_base: str | None,
    sleep_seconds: float,
    html_mode: str,
) -> int:
    api_key = get_secret("DEEPL_API_KEY", "") or ""
    if not api_key:
        print("DEEPL_API_KEY is required.", file=sys.stderr)
        return 2
    resolved_base = _deepl_api_base(api_key, api_base)

    try:
        usage = _deepl_usage(api_key=api_key, api_base=resolved_base)
        used = int(usage.get("character_count") or usage.get("api_key_character_count") or 0)
        limit = int(usage.get("character_limit") or usage.get("api_key_character_limit") or 0)
        if limit:
            print(f"DeepL usage before run: {used:,}/{limit:,} chars")
        else:
            print(f"DeepL usage before run: {used:,} chars")
    except Exception as exc:
        print(f"warning: could not check DeepL usage: {exc}", file=sys.stderr)

    translated_chars = 0
    failures = 0
    for path in _page_paths(slugs):
        if not path.is_file():
            print(f"missing: {path}", file=sys.stderr)
            failures += 1
            continue
        page = _load_json(path)
        slug = str(page.get("slug") or path.stem)
        if not force and str(page.get("html_body_en") or "").strip():
            print(f"skip: {slug} already has html_body_en")
            continue

        source_chars = sum(len(str(page.get(key) or "")) for key in ("title", "summary", "html_body"))
        if max_chars > 0 and translated_chars + source_chars > max_chars:
            print(
                f"stopping before {slug}: max char budget would be exceeded "
                f"({translated_chars:,}+{source_chars:,}>{max_chars:,})"
            )
            break

        print(f"translating: {slug} ({source_chars:,} source chars)")
        try:
            mode = "segments" if html_mode == "segments" else "html"
            translated = _deepl_translate_fields(
                page,
                api_key=api_key,
                api_base=resolved_base,
                target_lang=target_lang,
                source_lang=source_lang,
                html_mode=mode,
            )
            try:
                cleaned = _validate_translation(page, translated, max_hangul_ratio=max_hangul_ratio)
            except ValueError as exc:
                if html_mode != "auto" or "tag sequence differs" not in str(exc):
                    raise
                print(f"  DeepL HTML mode changed tags; retrying {slug} with segment-preserving mode")
                translated = _deepl_translate_fields(
                    page,
                    api_key=api_key,
                    api_base=resolved_base,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    html_mode="segments",
                )
                cleaned = _validate_translation(page, translated, max_hangul_ratio=max_hangul_ratio)
        except Exception as exc:
            print(f"failed: {slug}: {exc}", file=sys.stderr)
            failures += 1
            continue

        page.update(cleaned)
        page["translation_provider"] = "deepl"
        page["translation_source_lang"] = source_lang
        page["translation_target_lang"] = target_lang
        page["translation_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if dry_run:
            print(f"dry-run ok: {slug} ({len(cleaned['html_body_en']):,} English HTML chars)")
        else:
            _write_json(path, page)
            print(f"updated: {path} ({len(cleaned['html_body_en']):,} English HTML chars)")
        translated_chars += source_chars
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return 1 if failures else 0


def _usage_deepl(api_base: str | None) -> int:
    api_key = get_secret("DEEPL_API_KEY", "") or ""
    if not api_key:
        print("DEEPL_API_KEY is required.", file=sys.stderr)
        return 2
    usage = _deepl_usage(api_key=api_key, api_base=_deepl_api_base(api_key, api_base))
    print(json.dumps(usage, ensure_ascii=False, indent=2))
    return 0


def _status() -> int:
    for path in sorted(STATIC_PAGES_DIR.glob("*.json")):
        page = _load_json(path)
        slug = str(page.get("slug") or path.stem)
        has_en = bool(str(page.get("html_body_en") or "").strip())
        marker = "ok" if has_en else "missing"
        print(
            f"{marker:7} {slug:42} "
            f"ko={len(str(page.get('html_body') or '')):6,} "
            f"en={len(str(page.get('html_body_en') or '')):6,}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Static page English translation pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser("export", help="Create ChatGPT prompt/source files for pages missing English.")
    export_p.add_argument("slugs", nargs="*", help="Specific static page slugs. Defaults to all missing translations.")
    export_p.add_argument("--force", action="store_true", help="Export even when a page already has html_body_en.")

    import_p = sub.add_parser("import", help="Validate and attach returned translation JSON.")
    import_p.add_argument("slugs", nargs="*", help="Specific slugs. Defaults to all *.translation.json files.")
    import_p.add_argument("--dry-run", action="store_true", help="Validate without writing static_pages files.")
    import_p.add_argument("--max-hangul-ratio", type=float, default=0.08)

    deepl_p = sub.add_parser("translate-deepl", help="Batch translate pages through the DeepL Translation API.")
    deepl_p.add_argument("slugs", nargs="*", help="Specific static page slugs. Defaults to pages missing English.")
    deepl_p.add_argument("--force", action="store_true", help="Regenerate translations even when html_body_en exists.")
    deepl_p.add_argument("--dry-run", action="store_true", help="Translate and validate without writing files.")
    deepl_p.add_argument("--max-hangul-ratio", type=float, default=0.08)
    deepl_p.add_argument("--max-chars", type=int, default=500_000, help="Stop before exceeding this source-char budget. Use 0 for no script-side cap.")
    deepl_p.add_argument("--target-lang", default="EN-US")
    deepl_p.add_argument("--source-lang", default="KO")
    deepl_p.add_argument("--api-base", default=None, help="Override DeepL endpoint. Defaults to api-free for :fx keys, otherwise api.")
    deepl_p.add_argument("--sleep", type=float, default=0.2, help="Delay between page translations.")
    deepl_p.add_argument(
        "--html-mode",
        choices=["auto", "html", "segments"],
        default="auto",
        help="DeepL HTML handling. auto retries with exact tag-preserving text segments if DeepL changes markup.",
    )

    usage_p = sub.add_parser("deepl-usage", help="Print DeepL API usage/quota.")
    usage_p.add_argument("--api-base", default=None, help="Override DeepL endpoint. Defaults to api-free for :fx keys, otherwise api.")

    sub.add_parser("status", help="Show which static pages have English translations.")

    args = parser.parse_args()
    if args.command == "export":
        return _export(args.slugs, force=args.force)
    if args.command == "import":
        return _import(args.slugs, max_hangul_ratio=args.max_hangul_ratio, dry_run=args.dry_run)
    if args.command == "translate-deepl":
        return _translate_deepl(
            args.slugs,
            force=args.force,
            dry_run=args.dry_run,
            max_hangul_ratio=args.max_hangul_ratio,
            max_chars=args.max_chars,
            target_lang=args.target_lang,
            source_lang=args.source_lang,
            api_base=args.api_base,
            sleep_seconds=args.sleep,
            html_mode=args.html_mode,
        )
    if args.command == "deepl-usage":
        return _usage_deepl(args.api_base)
    if args.command == "status":
        return _status()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
