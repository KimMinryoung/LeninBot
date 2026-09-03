"""Documents → knowledge graph (research, archival manifest, synthesis notes).

Runs from ``jobs.kg_sync --source documents``. Deterministic layer always;
the LLM extraction layer only when ``KG_DOC_EXTRACT_LLM=1`` (registry call
site ``kg_document_extraction``). ``--limit`` caps documents per run so a
backfill spreads over nights; unchanged documents (same content hash on the
Document node) are skipped and do not count against the limit.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from db import query as db_query
from kg_runtime import doc_extract as dx

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", "/home/grass/frontend"))
MANIFEST_PATH = FRONTEND_DIR / "data" / "commulingo" / "docs" / "manifest.json"
ORDER = ("research", "archival", "autonote")  # research first: closes the webchat content gap


def _commulingo_names() -> dict[str, dict[str, str]]:
    names = {"person": {}, "term": {}, "event": {}}
    try:
        for r in db_query("SELECT id, name_ko, name_en FROM commulingo_people"):
            names["person"][r["id"]] = (r.get("name_ko") or r.get("name_en") or r["id"]).strip()
        for r in db_query("SELECT id, term_ko, term_en FROM commulingo_terms"):
            names["term"][r["id"]] = (r.get("term_ko") or r.get("term_en") or r["id"]).strip()
        for r in db_query("SELECT id, title_ko, title_en FROM commulingo_history_events"):
            names["event"][r["id"]] = (r.get("title_ko") or r.get("title_en") or r["id"]).strip()
    except Exception as exc:
        logger.warning("[kg-sync documents] CommuLingo name lookup failed: %s", exc)
    return names


def load_records(kinds=ORDER, *, since: datetime | None = None) -> list[dx.DocRecord]:
    recs: list[dx.DocRecord] = []
    if "research" in kinds:
        sql = "SELECT * FROM research_documents WHERE status = 'public'"
        params: tuple = ()
        if since is not None:
            sql += " AND (updated_at > %s OR published_at > %s)"
            params = (since, since)
        for row in db_query(sql + " ORDER BY published_at DESC", params):
            recs.append(dx.research_record(row))
    if "archival" in kinds:
        if MANIFEST_PATH.exists():
            try:
                manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                docs = manifest.get("docs") if isinstance(manifest, dict) else manifest
                for doc in docs or []:
                    html_path = MANIFEST_PATH.parent / str(doc.get("file") or "")
                    html = html_path.read_text(encoding="utf-8") if html_path.is_file() else None
                    recs.append(dx.archival_record(doc, html))
            except Exception as exc:
                logger.warning("[kg-sync documents] manifest unreadable: %s", exc)
        else:
            logger.info("[kg-sync documents] archival manifest not found at %s — skipped", MANIFEST_PATH)
    if "autonote" in kinds:
        sql = "SELECT id, project_id, turn, text, sources, created_at, kind FROM autonomous_project_notes WHERE kind = 'synthesis'"
        params = ()
        if since is not None:
            sql += " AND created_at > %s"
            params = (since,)
        for row in db_query(sql + " ORDER BY created_at DESC", params):
            recs.append(dx.autonote_record(row))
    return recs


def run(*, since: datetime | None = None, full: bool = False, limit: int | None = None,
        dry_run: bool = False, kinds=ORDER, use_llm: bool | None = None) -> dict:
    use_llm = dx.llm_enabled() if use_llm is None else use_llm
    recs = load_records(kinds, since=None if full else since)
    stats: dict = {"documents": len(recs), "by_kind": {}, "llm": use_llm, "processed": 0,
                   "unchanged": 0, "written": 0, "rejected": 0, "expired": 0, "errors": [], "items": []}
    for r in recs:
        stats["by_kind"][r["kind"]] = stats["by_kind"].get(r["kind"], 0) + 1
    if not recs:
        return stats

    if dry_run:
        from kg_runtime.identity import AliasIndex
        idx = AliasIndex()
        try:
            idx.refresh_from_neo4j()
        except Exception:
            idx = None
        names = _commulingo_names()
        sample = []
        for rec in recs[: (limit or 3)]:
            facts = dx.build_document_facts(rec, names=names, alias_index=idx, use_llm=False)
            sample.append({"ref": rec.ref, "title": rec["title"], "deterministic_facts": len(facts),
                           "links": {k: len(v) for k, v in rec["links"].items()},
                           "examples": [f"{f['subject_name']} —{f['predicate']}({f['attributes'].get('reference_type')})→ {f['object_name']}" for f in facts[:6]]})
        stats["sample"] = sample
        return stats

    existing: dict[str, str] = {}
    for prefix in ("research:", "archival:", "autonote:"):
        try:
            existing.update(dx.existing_document_hashes(prefix))
        except Exception as exc:
            logger.warning("[kg-sync documents] hash lookup failed (%s): %s", prefix, exc)
    names = _commulingo_names()
    from kg_runtime.identity import get_alias_index
    idx = get_alias_index()
    idx.ensure_loaded()

    for rec in recs:
        if limit is not None and stats["processed"] >= limit:
            break
        try:
            # ``full`` widens the candidate set to every document; it never
            # forces re-extraction — unchanged hashes are still skipped, so a
            # weekly full pass costs nothing (and no LLM spend) for stable docs.
            res = dx.extract_document(rec, names=names, alias_index=idx, use_llm=use_llm,
                                      force=False, existing_sha=existing.get(rec.ref))
        except Exception as exc:
            logger.exception("[kg-sync documents] %s failed", rec.ref)
            stats["errors"].append(f"{rec.ref}: {exc}")
            continue
        if res.get("status") == "unchanged":
            stats["unchanged"] += 1
            continue
        stats["processed"] += 1
        stats["written"] += res.get("written", 0)
        stats["rejected"] += res.get("rejected", 0)
        stats["expired"] += res.get("expired", 0)
        if res.get("status") == "error":
            stats["errors"].append(f"{rec.ref}: {res.get('message')}")
        stats["items"].append({k: res.get(k) for k in ("ref", "status", "facts", "written", "rejected", "expired")})
        try:
            idx.refresh_from_neo4j()  # new entities become matchable for the next document
        except Exception:
            pass
    if stats["errors"]:
        stats["error"] = f"{len(stats['errors'])} document(s) failed: {stats['errors'][0][:200]}"
    stats["items"] = stats["items"][:50]
    return stats
