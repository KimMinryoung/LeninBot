"""Corpus-level translation memory (TM).

The archival pipeline's JSONL chunk cache answers one question — "did this
exact prompt, under this exact model and system prompt, already run?" — and
its records store only the output blocks, keyed by hash. Nothing in the
repository stores *aligned source/target pairs*, so nothing can answer "how
did we translate this sentence last month, in any document?" That gap is what
this module fills: an append-mostly SQLite store of aligned segments, fed by
the pipelines as they translate.

v1 is deliberately write-first (번역기 인수인계 §5-1: "기존 스크립트의 청크
결과를 여기 적재하는 것부터"). Reads are exact-match only; fuzzy retrieval
and prompt-example injection can build on the same table later without a
migration. Recording must never break a translation run — callers wrap it in
try/except and report, they do not propagate.

Statuses (lookup precedence: machine < published < reviewed):
    machine    pipeline output, unreviewed (the default)
    published  the pair comes from a frozen (published) archival spec — the
               document as a whole passed a human read-through before
               publication, but no one confirmed this segment individually
    reviewed   a human confirmed this specific pair
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "output" / "translation_memory.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id         INTEGER PRIMARY KEY,
    lang_pair  TEXT NOT NULL,
    source     TEXT NOT NULL,
    target     TEXT NOT NULL,
    doc_id     TEXT NOT NULL,
    block_id   TEXT,
    status     TEXT NOT NULL DEFAULT 'machine',
    provider   TEXT,
    model      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (lang_pair, doc_id, source, target)
);
CREATE INDEX IF NOT EXISTS idx_segments_lookup ON segments (lang_pair, source);
CREATE INDEX IF NOT EXISTS idx_segments_doc ON segments (lang_pair, doc_id);
"""


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def record_segments(
    pairs: Iterable[tuple[str, str]],
    *,
    lang_pair: str,
    doc_id: str,
    block_ids: Sequence[object] | None = None,
    status: str = "machine",
    provider: str | None = None,
    model: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Store aligned (source, target) pairs. Returns rows inserted or upgraded.

    Idempotent: re-recording an existing pair changes nothing, except that a
    higher status wins — a pair first stored as machine at translation time
    and later backfilled from a frozen spec is upgraded to published instead
    of being silently ignored. A status never goes down this way.

    Empty sides are skipped rather than rejected: an archival block whose
    translation is legitimately empty (a spacer) is not a segment.
    """
    rows = []
    for i, (source, target) in enumerate(pairs):
        source = (source or "").strip()
        target = (target or "").strip()
        if not source or not target:
            continue
        block_id = str(block_ids[i]) if block_ids is not None else None
        rows.append((lang_pair, source, target, doc_id, block_id, status, provider, model))
    if not rows:
        return 0
    rank = "CASE {} WHEN 'reviewed' THEN 2 WHEN 'published' THEN 1 ELSE 0 END"
    with _connect(db_path) as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT INTO segments"
            " (lang_pair, source, target, doc_id, block_id, status, provider, model)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (lang_pair, doc_id, source, target) DO UPDATE SET"
            "   status = excluded.status"
            f" WHERE {rank.format('excluded.status')} > {rank.format('segments.status')}",
            rows,
        )
        return conn.total_changes - before


def exact_matches(
    sources: Iterable[str],
    *,
    lang_pair: str,
    statuses: Sequence[str] | None = None,
    db_path: Path | str | None = None,
) -> dict[str, str]:
    """source text → target for exact matches.

    reviewed > published > machine 순서로 이기고, 같은 상태에서는 새 행이
    이긴다 — 정렬을 그 순서로 두고 dict를 덮어쓰게 해서, 마지막에 남는 행이
    승자다. statuses를 주면 그 상태의 행만 본다(자동 재사용은 검수 등급만
    쓰는 식으로).
    """
    out: dict[str, str] = {}
    cleaned = [s.strip() for s in sources if (s or "").strip()]
    if not cleaned:
        return out
    status_sql = ""
    status_params: list[str] = []
    if statuses:
        status_sql = f" AND status IN ({','.join('?' for _ in statuses)})"
        status_params = list(statuses)
    with _connect(db_path) as conn:
        for start in range(0, len(cleaned), 500):
            batch = cleaned[start : start + 500]
            marks = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT source, target FROM segments"
                f" WHERE lang_pair = ? AND source IN ({marks})" + status_sql +
                " ORDER BY CASE status WHEN 'reviewed' THEN 2 WHEN 'published' THEN 1 ELSE 0 END, id",
                [lang_pair, *batch, *status_params],
            ).fetchall()
            for source, target in rows:
                out[source] = target
    return out


def stats(db_path: Path | str | None = None) -> dict:
    with _connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        by_pair = dict(
            conn.execute("SELECT lang_pair, COUNT(*) FROM segments GROUP BY lang_pair").fetchall()
        )
        by_status = dict(
            conn.execute("SELECT status, COUNT(*) FROM segments GROUP BY status").fetchall()
        )
    return {"total": total, "byLangPair": by_pair, "byStatus": by_status}
