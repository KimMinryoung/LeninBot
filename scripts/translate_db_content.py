#!/usr/bin/env python3
"""Translate frontend DB posts/diaries/hub curations into English columns.

Korean originals stay in title/content. English translations are written to
title_en/content_en only when missing, so publishing can remain Korean-first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
import redis
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secrets_loader import get_secret

FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", ROOT.parent / "frontend")).resolve()
FRONTEND_ENV = FRONTEND_DIR / ".env"
DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEFAULT_MODEL = os.getenv("DB_CONTENT_TRANSLATION_MODEL", "deepseek-v4-flash")
DEFAULT_MAX_TOKENS = int(os.getenv("DB_CONTENT_TRANSLATION_MAX_TOKENS", "12000"))
TIMEOUT_SECONDS = 180

TARGETS = {
    "posts": {
        "table": "posts",
        "cache_pattern": "post:*",
        "label": "Bichon blog post",
    },
    "diary": {
        "table": "ai_diary",
        "cache_pattern": "diary:*",
        "label": "Cyber-Lenin diary entry",
    },
    "curation": {
        "table": "hub_curations",
        "cache_pattern": "hub:*",
        "label": "Cyber-Lenin curation entry",
    },
}

SYSTEM_PROMPT = """You are a careful Korean-to-English translation editor.

Translate the supplied Korean title and body into polished, natural English.

Requirements:
- Preserve HTML tags, links, URLs, line breaks, markdown-like bullets, and inline code.
- Translate visible Korean text only; do not summarize, omit, expand, fact-check, or add commentary.
- Keep the writer's tone: casual blog posts may stay casual, Cyber-Lenin diary entries may stay analytical and political.
- For curation entries, preserve the distinction between source title, selection rationale, and context.
- Use South Korea/Korean for 한국 when that is the meaning.
- Return strict JSON only, with exactly these keys: "title_en", "content_en".
"""


def _load_frontend_env() -> dict[str, str]:
    values = {k: v for k, v in dotenv_values(FRONTEND_ENV).items() if v is not None}
    merged = {**values, **os.environ}
    return merged


def _connect_db(env: dict[str, str]):
    return psycopg2.connect(
        host=env.get("DB_HOST"),
        port=int(env.get("DB_PORT") or 5432),
        user=env.get("DB_USER"),
        password=env.get("DB_PASSWORD"),
        dbname=env.get("DB_NAME"),
        sslmode="require" if env.get("DB_SSL") == "true" else "prefer",
    )


def _ensure_curation_columns(conn) -> None:
    with conn.cursor() as cur:
        for ddl in (
            "ALTER TABLE hub_curations ADD COLUMN IF NOT EXISTS title_en TEXT",
            "ALTER TABLE hub_curations ADD COLUMN IF NOT EXISTS source_title_en TEXT",
            "ALTER TABLE hub_curations ADD COLUMN IF NOT EXISTS selection_rationale_en TEXT",
            "ALTER TABLE hub_curations ADD COLUMN IF NOT EXISTS context_en TEXT",
        ):
            cur.execute(ddl)
    conn.commit()


def _select_rows(conn, target_name: str, table: str, *, ids: list[int], limit: int, force: bool) -> list[dict[str, Any]]:
    where = ""
    params: list[Any] = []
    if ids:
        where = "WHERE id = ANY(%s)"
        params.append(ids)
    elif not force:
        if target_name == "curation":
            where = (
                "WHERE NULLIF(BTRIM(COALESCE(title_en, '')), '') IS NULL "
                "OR NULLIF(BTRIM(COALESCE(selection_rationale_en, '')), '') IS NULL "
                "OR NULLIF(BTRIM(COALESCE(context_en, '')), '') IS NULL"
            )
        else:
            where = "WHERE NULLIF(BTRIM(COALESCE(title_en, '')), '') IS NULL OR NULLIF(BTRIM(COALESCE(content_en, '')), '') IS NULL"
    order_column = "published_at" if target_name == "curation" else "created_at"
    order_limit = f"ORDER BY {order_column} DESC"
    if limit > 0:
        order_limit += " LIMIT %s"
        params.append(limit)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if target_name == "curation":
            cur.execute(
                f"""
                SELECT id, title, source_title, selection_rationale, context,
                       title_en, source_title_en, selection_rationale_en, context_en,
                       published_at
                FROM {table}
                {where}
                {order_limit}
                """,
                params,
            )
        else:
            cur.execute(
                f"""
                SELECT id, title, content, title_en, content_en, created_at
                FROM {table}
                {where}
                {order_limit}
                """,
                params,
            )
        return [dict(row) for row in cur.fetchall()]


def _parse_json_response(text: str) -> dict[str, str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    data = json.loads(stripped)
    title = (data.get("title_en") or "").strip()
    content = (data.get("content_en") or "").strip()
    if not title or not content:
        raise ValueError("translation JSON is missing title_en or content_en")
    return {"title_en": title, "content_en": content}


def _parse_curation_json_response(text: str) -> dict[str, str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    data = json.loads(stripped)
    out = {
        "title_en": (data.get("title_en") or "").strip(),
        "source_title_en": (data.get("source_title_en") or "").strip(),
        "selection_rationale_en": (data.get("selection_rationale_en") or "").strip(),
        "context_en": (data.get("context_en") or "").strip(),
    }
    if not out["title_en"] or not out["selection_rationale_en"] or not out["context_en"]:
        raise ValueError("curation translation JSON is missing title_en, selection_rationale_en, or context_en")
    return out


def _call_deepseek(row: dict[str, Any], *, label: str, model: str, base_url: str, max_tokens: int) -> dict[str, str]:
    api_key = get_secret("DEEPSEEK_API_KEY", "") or ""
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    if "selection_rationale" in row:
        system_prompt = SYSTEM_PROMPT + '\nFor curation entries, return strict JSON only, with exactly these keys: "title_en", "source_title_en", "selection_rationale_en", "context_en".'
        payload = {
            "kind": label,
            "id": row["id"],
            "title": row.get("title") or "",
            "source_title": row.get("source_title") or "",
            "selection_rationale": row.get("selection_rationale") or "",
            "context": row.get("context") or "",
        }
    else:
        system_prompt = SYSTEM_PROMPT
        payload = {
            "kind": label,
            "id": row["id"],
            "title": row.get("title") or "",
            "content": row.get("content") or "",
        }
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        },
        timeout=TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(f"DeepSeek request failed: HTTP {response.status_code}: {response.text[:1000]}")
    data = response.json()
    content = data["choices"][0]["message"].get("content") or ""
    if "selection_rationale" in row:
        return _parse_curation_json_response(content)
    return _parse_json_response(content)


def _update_row(conn, target_name: str, table: str, row_id: int, translated: dict[str, str], row: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        if target_name == "curation":
            cur.execute(
                f"""
                UPDATE {table}
                   SET title_en = %s,
                       source_title_en = COALESCE(NULLIF(%s, ''), source_title_en),
                       selection_rationale_en = %s,
                       context_en = %s
                 WHERE id = %s
                """,
                [
                    translated["title_en"],
                    translated.get("source_title_en") or "",
                    translated["selection_rationale_en"],
                    translated["context_en"],
                    row_id,
                ],
            )
        else:
            cur.execute(
                f"UPDATE {table} SET title_en = %s, content_en = %s WHERE id = %s",
                [translated["title_en"], translated["content_en"], row_id],
            )
    conn.commit()


def _clear_cache(patterns: set[str], env: dict[str, str]) -> None:
    redis_url = env.get("REDIS_URL") or "redis://127.0.0.1:6379"
    client = redis.Redis.from_url(redis_url)
    deleted = 0
    for pattern in patterns:
        keys = list(client.scan_iter(match=pattern))
        if keys:
            deleted += client.delete(*keys)
    print(f"cleared redis cache keys: {deleted}")


def translate_target(
    target_name: str,
    *,
    ids: list[int],
    limit: int,
    force: bool,
    dry_run: bool,
    select_only: bool,
    model: str,
    base_url: str,
    max_tokens: int,
) -> tuple[int, str]:
    target = TARGETS[target_name]
    env = _load_frontend_env()
    conn = _connect_db(env)
    changed = 0
    try:
        if target_name == "curation":
            _ensure_curation_columns(conn)
        rows = _select_rows(conn, target_name, target["table"], ids=ids, limit=limit, force=force)
        print(f"{target_name}: selected {len(rows)} row(s)")
        for row in rows:
            print(f"translating {target_name}#{row['id']}: {row.get('title') or ''}")
            if select_only:
                continue
            translated = _call_deepseek(
                row,
                label=target["label"],
                model=model,
                base_url=base_url,
                max_tokens=max_tokens,
            )
            if dry_run:
                print(f"dry-run ok {target_name}#{row['id']}: {translated['title_en']}")
                continue
            _update_row(conn, target_name, target["table"], int(row["id"]), translated, row)
            changed += 1
            print(f"updated {target_name}#{row['id']}: {translated['title_en']}")
    finally:
        conn.close()
    return changed, target["cache_pattern"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate posts/ai_diary/hub_curations rows into *_en columns.")
    parser.add_argument("--kind", choices=["posts", "diary", "curation", "all"], default="all")
    parser.add_argument("--id", dest="ids", type=int, action="append", default=[], help="Translate a specific row id. Repeatable.")
    parser.add_argument("--limit", type=int, default=10, help="Rows per selected kind. Use 0 for no limit.")
    parser.add_argument("--force", action="store_true", help="Retranslate even when *_en columns already exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--select-only", action="store_true", help="Only list selected rows; do not call the translation API.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    names = ["posts", "diary", "curation"] if args.kind == "all" else [args.kind]
    changed_total = 0
    cache_patterns: set[str] = set()
    failures = 0
    for name in names:
        try:
            changed, pattern = translate_target(
                name,
                ids=args.ids,
                limit=args.limit,
                force=args.force,
                dry_run=args.dry_run,
                select_only=args.select_only,
                model=args.model,
                base_url=args.base_url.rstrip("/"),
                max_tokens=args.max_tokens,
            )
            changed_total += changed
            if changed:
                cache_patterns.add(pattern)
        except Exception as exc:
            print(f"failed {name}: {exc}", file=sys.stderr)
            failures += 1

    if cache_patterns and not args.dry_run:
        _clear_cache(cache_patterns, _load_frontend_env())
    print(f"done: updated {changed_total} row(s), failures {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
