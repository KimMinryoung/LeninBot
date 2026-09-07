#!/usr/bin/env python3
"""Translate frontend DB posts/diaries/hub curations into English columns.

Korean originals stay in title/content. English translations are written to
title_en/content_en when missing or when the observed source changes.
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
from translation_runtime import TranslationProviderError
from translation_runtime.batch_state import BatchState
from scripts._translation_common import (
    TranslationCallError,
    field_translation_problems,
    parse_json_object,
)

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
        where = f"WHERE ({_missing_translation_sql(target_name)}) OR translation_source_sha256 IS DISTINCT FROM {_source_hash_sql(target_name)}"
    order_column = "published_at" if target_name == "curation" else "created_at"
    order_limit = f"ORDER BY {order_column} DESC, id DESC"
    if limit > 0:
        order_limit += " LIMIT %s"
        params.append(limit)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if target_name == "curation":
            cur.execute(
                f"""
                SELECT id, title, source_title, selection_rationale, context,
                       title_en, source_title_en, selection_rationale_en, context_en,
                       published_at, {_source_hash_sql(target_name)} AS translation_current_sha256
                FROM {table}
                {where}
                {order_limit}
                """,
                params,
            )
        else:
            cur.execute(
                f"""
                SELECT id, title, content, title_en, content_en, created_at,
                       {_source_hash_sql(target_name)} AS translation_current_sha256
                FROM {table}
                {where}
                {order_limit}
                """,
                params,
            )
        rows = [dict(row) for row in cur.fetchall()]
    return rows


def _source_fields(target_name):
    return ('title', 'source_title', 'selection_rationale', 'context') if target_name == 'curation' else ('title', 'content')


def _source_hash_sql(target_name):
    fields = ', '.join(_source_fields(target_name))
    return f"encode(sha256(convert_to(jsonb_build_array({fields})::text, 'UTF8')), 'hex')"


def _missing_translation_sql(target_name):
    fields = ('title_en', 'selection_rationale_en', 'context_en') if target_name == 'curation' else ('title_en', 'content_en')
    return ' OR '.join(f"NULLIF(BTRIM(COALESCE({field}, '')), '') IS NULL" for field in fields)


def translation_freshness_migration_sql():
    """Initialize complete legacy translations once; reruns never bless stale rows.

    Existing translations are preserved as a baseline, not claimed to be checked.
    The column and its baseline are created in the caller's same transaction.
    """
    statements = []
    for kind, target in TARGETS.items():
        table = target['table']
        statements.append(f"""DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_attribute
                           WHERE attrelid = '{table}'::regclass
                             AND attname = 'translation_source_sha256' AND NOT attisdropped) THEN
                ALTER TABLE {table} ADD COLUMN translation_source_sha256 TEXT;
                UPDATE {table} SET translation_source_sha256 = {_source_hash_sql(kind)}
                 WHERE NOT ({_missing_translation_sql(kind)});
            END IF;
        END $$""")
    return statements


def _parse_json_response(text: str) -> dict[str, str]:
    data = parse_json_object(text, keys=["title_en", "content_en"])
    if any(not isinstance(value, str) for value in data.values()):
        raise ValueError("translation JSON values must be strings")
    title = (data.get("title_en") or "").strip()
    content = (data.get("content_en") or "").strip()
    if not title or not content:
        raise ValueError("translation JSON is missing title_en or content_en")
    return {"title_en": title, "content_en": content}


def _parse_curation_json_response(text: str) -> dict[str, str]:
    data = parse_json_object(text, keys=["title_en", "source_title_en",
                                         "selection_rationale_en", "context_en"])
    if any(not isinstance(value, str) for value in data.values()):
        raise ValueError("translation JSON values must be strings")
    out = {
        "title_en": (data.get("title_en") or "").strip(),
        "source_title_en": (data.get("source_title_en") or "").strip(),
        "selection_rationale_en": (data.get("selection_rationale_en") or "").strip(),
        "context_en": (data.get("context_en") or "").strip(),
    }
    if not out["title_en"] or not out["selection_rationale_en"] or not out["context_en"]:
        raise ValueError("curation translation JSON is missing title_en, selection_rationale_en, or context_en")
    return out


# 원문 컬럼 → 번역 컬럼. 검증과 TM 적재가 같은 대응을 쓴다.
_FIELD_MAP = {
    "title": "title_en",
    "content": "content_en",
    "source_title": "source_title_en",
    "selection_rationale": "selection_rationale_en",
    "context": "context_en",
}


def _validate_translated_fields(row: dict[str, Any], translated: dict[str, str]) -> list[str]:
    """결정론적 검증 (인수인계 §2.5). 문제 목록을 돌려주고, 비어 있으면 통과다.

    예전에는 JSON 키가 채워져 있기만 하면 그대로 DB에 썼다 — 이 스크립트만
    출력이 절반쯤 한국어로 남아도 아무도 모르는 상태였다. 프롬프트가 약속하는
    것(HTML 태그·링크 보존, 한국어 제거)을 여기서 확인한다.
    """
    problems: list[str] = []
    for src_key, en_key in _FIELD_MAP.items():
        if src_key not in row:
            continue
        problems.extend(
            field_translation_problems(
                row.get(src_key) or "", translated.get(en_key) or "", label=en_key,
                # source_title_en may stay empty; _update_row keeps the old value then.
                optional=(src_key == "source_title"),
            )
        )
    return problems


def _generate(system_prompt: str, payload: dict[str, Any]) -> str:
    """게이트웨이를 지나는 원샷 호출. 모델·예산·thinking은 레지스트리가 정한다."""
    from scripts._translation_common import generate_translation
    return generate_translation(FEATURE, json.dumps(payload, ensure_ascii=False), system=system_prompt)


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
    from translation_runtime import translate_validated
    parser = _parse_curation_json_response if "selection_rationale" in row else _parse_json_response
    return translate_validated(
        generate=lambda correction: _generate(system_prompt + correction, payload),
        parse=parser, validate=lambda value: _validate_translated_fields(row, value), attempts=2)


def _update_row(conn, target_name: str, table: str, row_id: int, translated: dict[str, str], row: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        if target_name == "curation":
            cur.execute(
                f"""
                UPDATE {table}
                   SET title_en = %s,
                       source_title_en = COALESCE(NULLIF(%s, ''), source_title_en),
                       selection_rationale_en = %s,
                       context_en = %s,
                       translation_source_sha256 = {_source_hash_sql(target_name)}
                 WHERE id = %s AND title IS NOT DISTINCT FROM %s
                   AND source_title IS NOT DISTINCT FROM %s
                   AND selection_rationale IS NOT DISTINCT FROM %s
                   AND context IS NOT DISTINCT FROM %s
                """,
                [
                    translated["title_en"],
                    translated.get("source_title_en") or "",
                    translated["selection_rationale_en"],
                    translated["context_en"],
                    row_id, row.get("title"), row.get("source_title"),
                    row.get("selection_rationale"), row.get("context"),
                ],
            )
        else:
            cur.execute(
                f"UPDATE {table} SET title_en = %s, content_en = %s, translation_source_sha256 = {_source_hash_sql(target_name)} WHERE id = %s AND title IS NOT DISTINCT FROM %s AND content IS NOT DISTINCT FROM %s",
                [translated["title_en"], translated["content_en"], row_id, row.get("title"), row.get("content")],
            )
        if cur.rowcount != 1:
            conn.rollback()
            raise RuntimeError("source changed during translation; retry latest row")
    conn.commit()


def _record_tm(target_name: str, row: dict[str, Any], translated: dict[str, str]) -> None:
    """짧은 필드의 (원문, 번역) 쌍을 코퍼스 단위 번역 메모리에 적재한다.

    content는 문서 통짜라 세그먼트로서 재사용 가치가 낮아 제외한다. TM 실패가
    번역 저장을 깨서는 안 되므로 예외는 경고로만 남긴다.
    """
    try:
        from runtime_tools import translation_memory

        pairs = [
            (row.get(src) or "", translated.get(en) or "")
            for src, en in _FIELD_MAP.items()
            if src != "content" and src in row
        ]
        translation_memory.record_segments(
            pairs, lang_pair="ko-en", doc_id=f"{target_name}#{row['id']}"
        )
    except Exception as exc:
        print(f"warning: tm record skipped for {target_name}#{row['id']}: {exc}", file=sys.stderr)


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
    retry_failed: bool = False,
) -> tuple[int, str, list[str]]:
    target = TARGETS[target_name]
    env = _load_frontend_env()
    conn = _connect_db(env)
    state = BatchState()
    changed = 0
    failures: list[str] = []
    try:
        if target_name == "curation" and not select_only and not dry_run:
            _ensure_curation_columns(conn)
        rows = _select_rows(conn, target_name, target["table"], ids=ids, limit=0, force=force)
        attempted = 0
        print(f"{target_name}: selected {len(rows)} row(s)")
        for row in rows:
            key, fingerprint = f"{target_name}:{row['id']}", row["translation_current_sha256"]
            if not (retry_failed or force or ids or select_only or dry_run) and state.deferred(key, fingerprint):
                failures.append(f"{key}: validation cooldown")
                print(f"deferred {key}")
                continue
            if limit > 0 and attempted >= limit:
                break
            attempted += 1
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
                if not dry_run and not select_only:
                    state.failed(key, fingerprint, exc)
                failures.append(f"{target_name}#{row['id']}: {exc}")
                print(f"failed {target_name}#{row['id']}: {exc}", file=sys.stderr)
                if isinstance(exc, TranslationProviderError) and exc.result.error_kind in {
                        "authentication", "quota", "policy", "configuration"}:
                    exc.partial_changed = changed
                    exc.partial_failures = failures[:-1]  # the last entry is this error
                    exc.cache_pattern = target["cache_pattern"]
                    raise
                continue
            if dry_run:
                print(f"dry-run ok {target_name}#{row['id']}: {translated['title_en']}")
                continue
            try:
                _update_row(conn, target_name, target["table"], int(row["id"]), translated, row)
            except Exception as exc:
                conn.rollback()
                if not dry_run and not select_only:
                    state.failed(key, fingerprint, exc)
                failures.append(f"{target_name}#{row['id']}: {exc}")
                continue
            changed += 1
            try:
                state.succeeded(key, fingerprint)
            except Exception as exc:
                failures.append(f"{key}: translation saved but cooldown cleanup failed: {exc}")
            _record_tm(target_name, row, translated)
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
    parser.add_argument("--retry-failed", action="store_true", help="Bypass saved retry delays and validation quarantine.")
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
                retry_failed=args.retry_failed,
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
            failures.extend(getattr(exc, "partial_failures", []))
            failures.append(f"{name}: {exc}")
            if isinstance(exc, TranslationProviderError):
                changed_total += getattr(exc, "partial_changed", 0)
                if getattr(exc, "partial_changed", 0):
                    cache_patterns.add(exc.cache_pattern)
                break

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
