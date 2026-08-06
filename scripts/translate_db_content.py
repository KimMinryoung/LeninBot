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
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secrets_loader import get_secret

FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", ROOT.parent / "frontend")).resolve()
FRONTEND_ENV = FRONTEND_DIR / ".env"
# 프로바이더·모델·예산·타임아웃·thinking은 전부 레지스트리 항목이 정한다
# (config/llm_call_sites.json). 여기에 base_url이나 키가 없는 것이 정상이다 —
# 실키는 llm_proxy에만 있고 호출은 게이트웨이를 지난다.
FEATURE = "db_content_translation"

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


class TranslationCallError(RuntimeError):
    """제공자가 실제로 무엇을 돌려줬는지 함께 들고 다니는 오류.

    예전에는 빈 응답이 json.loads에서 "Expecting value: line 1 column 1"으로
    터졌다. 그 문장만 보고는 모델이 거부한 것인지, 응답이 잘린 것인지, 아예
    비어서 온 것인지 구분할 수 없어서 로그를 봐도 손을 못 댔다.
    """


def _generate(system_prompt: str, payload: dict[str, Any]) -> str:
    """게이트웨이를 지나는 원샷 호출. 모델·예산·thinking은 레지스트리가 정한다."""
    from llm.call_registry import generate_sync

    return generate_sync(
        FEATURE,
        json.dumps(payload, ensure_ascii=False),
        system=system_prompt,
    ) or ""


def _call_translator(row: dict[str, Any], *, label: str) -> dict[str, str]:
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
    content = _generate(system_prompt, payload)
    if not content:
        # generate_sync는 어떤 실패든 None으로 삼킨다. 원인(HTTP 오류, 정책
        # 거부, 빈 완성)은 llm_gateway.audit 로그와 journald의 registry 경고에
        # 남으므로, 여기서는 그쪽을 보라고 가리킨다.
        raise TranslationCallError(
            f"{FEATURE}: 게이트웨이가 본문을 돌려주지 않았다 "
            f"(원인은 llm_gateway.audit / [llm-registry] 경고 참조)"
        )
    try:
        if "selection_rationale" in row:
            return _parse_curation_json_response(content)
        return _parse_json_response(content)
    except (json.JSONDecodeError, ValueError) as exc:
        # 잘린 JSON과 안내문으로 시작하는 JSON은 증상이 같다. 앞뒤를 같이
        # 남겨야 다음 사람이 로그만 보고 어느 쪽인지 안다.
        raise TranslationCallError(
            f"could not parse the translation JSON ({exc}); "
            f"{len(content)} chars, head={content[:200]!r}, tail={content[-200:]!r}"
        ) from exc


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
) -> tuple[int, str, list[str]]:
    target = TARGETS[target_name]
    env = _load_frontend_env()
    conn = _connect_db(env)
    changed = 0
    failures: list[str] = []
    try:
        if target_name == "curation":
            _ensure_curation_columns(conn)
        rows = _select_rows(conn, target_name, target["table"], ids=ids, limit=limit, force=force)
        print(f"{target_name}: selected {len(rows)} row(s)")
        for row in rows:
            print(f"translating {target_name}#{row['id']}: {row.get('title') or ''}")
            if select_only:
                continue
            # 한 줄이 실패해도 나머지는 계속한다. 예전에는 여기서 예외가 그대로
            # 올라가 그 종류의 남은 줄을 전부 건너뛰었다. 2026-08-06에 일기
            # #418 하나가 빈 응답으로 실패하자 뒤에 있던 #417이 시도조차 되지
            # 않았고, 매일 밤 같은 #418을 먼저 집어 같은 자리에서 죽는 바람에
            # 사흘치 일기가 통째로 번역되지 않았다. 한 편이 안 되는 것과 전부
            # 멈추는 것은 다른 사고다.
            try:
                translated = _call_translator(row, label=target["label"])
            except Exception as exc:
                failures.append(f"{target_name}#{row['id']}: {exc}")
                print(f"failed {target_name}#{row['id']}: {exc}", file=sys.stderr)
                continue
            if dry_run:
                print(f"dry-run ok {target_name}#{row['id']}: {translated['title_en']}")
                continue
            _update_row(conn, target_name, target["table"], int(row["id"]), translated, row)
            changed += 1
            print(f"updated {target_name}#{row['id']}: {translated['title_en']}")
    finally:
        conn.close()
    return changed, target["cache_pattern"], failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate posts/ai_diary/hub_curations rows into *_en columns.")
    parser.add_argument("--kind", choices=["posts", "diary", "curation", "all"], default="all")
    parser.add_argument("--id", dest="ids", type=int, action="append", default=[], help="Translate a specific row id. Repeatable.")
    parser.add_argument("--limit", type=int, default=10, help="Rows per selected kind. Use 0 for no limit.")
    parser.add_argument("--force", action="store_true", help="Retranslate even when *_en columns already exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--select-only", action="store_true", help="Only list selected rows; do not call the translation API.")
    # 모델을 바꿔 보려면 레지스트리 항목을 고치거나 환경변수
    # LLM_SITE_DB_CONTENT_TRANSLATION_MODEL을 쓴다. 예전의
    # --model/--base-url/--max-tokens는 레지스트리가 값을 쥔 뒤로 아무 효과가
    # 없어서 없앴다 — 먹지 않는 플래그를 남겨 두는 편이 더 나쁘다.
    args = parser.parse_args()

    names = ["posts", "diary", "curation"] if args.kind == "all" else [args.kind]
    changed_total = 0
    cache_patterns: set[str] = set()
    failures: list[str] = []
    for name in names:
        try:
            changed, pattern, row_failures = translate_target(
                name,
                ids=args.ids,
                limit=args.limit,
                force=args.force,
                dry_run=args.dry_run,
                select_only=args.select_only,
            )
            changed_total += changed
            failures.extend(row_failures)
            if changed:
                cache_patterns.add(pattern)
        except Exception as exc:
            # 여기까지 올라오는 것은 이제 그 종류 전체가 못 도는 사고다
            # (DB 연결 실패 등). 개별 줄의 실패는 translate_target 안에서
            # 잡혀 row_failures로 돌아온다.
            print(f"failed {name}: {exc}", file=sys.stderr)
            failures.append(f"{name}: {exc}")

    # 한 줄이라도 번역됐으면 캐시를 비운다. 실패가 섞여 있어도 성공한 것은
    # 바로 보여야 한다.
    if cache_patterns and not args.dry_run:
        _clear_cache(cache_patterns, _load_frontend_env())
    print(f"done: updated {changed_total} row(s), failures {len(failures)}")
    for detail in failures:
        print(f"  - {detail}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
