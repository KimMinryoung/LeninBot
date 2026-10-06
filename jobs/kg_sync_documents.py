"""Documents → knowledge graph (research, archival manifest, synthesis notes).

Runs from ``jobs.kg_sync --source documents``. Deterministic layer always;
the LLM extraction layer only when ``KG_DOC_EXTRACT_LLM=1`` (registry call
site ``kg_document_extraction``). ``--limit`` caps documents per run so a
backfill spreads over nights; unchanged documents (same content hash on the
Document node) are skipped and do not count against the limit.
"""

from __future__ import annotations

import logging
from datetime import datetime

from db import query as db_query
from kg_runtime import doc_extract as dx

logger = logging.getLogger(__name__)

ORDER = ("research", "archival", "autonote")  # research first: closes the webchat content gap


def _commulingo_names() -> dict[str, dict[str, str]]:
    names = {"person": {}, "term": {}, "event": {}}
    try:
        from commulingo.reads import reads
        for r in reads.dataset("people"):
            names["person"][r["id"]] = (r.get("name_ko") or r.get("name_en") or r["id"]).strip()
        for r in reads.dataset("terms"):
            names["term"][r["id"]] = (r.get("term_ko") or r.get("term_en") or r["id"]).strip()
        for r in reads.dataset("events"):
            names["event"][r["id"]] = (r.get("title_ko") or r.get("title_en") or r["id"]).strip()
    except Exception as exc:
        raise RuntimeError("CommuLingo name lookup failed") from exc
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
        # CommuLingo reference documents, through its admin MCP (docs_list, doc_get).
        try:
            from commulingo.mcp_client import call_tool
            ids, offset = [], 0
            while True:
                page = call_tool("docs_list", {"limit": 100, "offset": offset})
                ids += [item["id"] for item in page["items"]]
                offset += len(page["items"])
                if not page["items"] or offset >= page["total"]:
                    break
            for doc_id in ids:
                found = call_tool("doc_get", {"id": doc_id})
                doc, html = found["doc"], found["html"]
                if doc.get("file") and html is None:
                    raise FileNotFoundError(f"archival body missing: {doc_id}")
                rec = dx.archival_record(doc, html)
                rec['source_updated_at'] = doc.get("modifiedAt")
                recs.append(rec)
        except Exception as exc:
            raise RuntimeError(f"archival documents unreadable: {exc}") from exc
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
        dry_run: bool = False, kinds=ORDER, use_llm: bool | None = None, force: bool = False) -> dict:
    use_llm = dx.llm_enabled() if use_llm is None else use_llm
    recs = load_records(kinds)  # Daily deterministic reconciliation; LLM remains hash-gated.
    stats: dict = {"documents": len(recs), "by_kind": {}, "llm": use_llm, "processed": 0,
                   "unchanged": 0, "written": 0, "rejected": 0, "expired": 0, "errors": [], "items": [], "remaining": 0, "failed": 0, "complete": True}
    for r in recs:
        stats["by_kind"][r["kind"]] = stats["by_kind"].get(r["kind"], 0) + 1

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

    states = dx.existing_document_states()
    states = {ref: st for ref, st in states.items() if ref.split(':', 1)[0] in kinds}
    existing = {ref: st['sha'] for ref, st in states.items()}
    names = _commulingo_names()
    from kg_runtime.identity import get_alias_index
    idx = get_alias_index()
    if not idx.ensure_loaded():
        raise RuntimeError("Document alias index unavailable; refusing incomplete reconciliation"
                           + (f" ({idx.last_error})" if idx.last_error else ""))

    stats["links_repaired"] = 0
    attempted = 0
    for rec in recs:
        needs_extraction = force or existing.get(rec.ref) != rec['sha']
        allow_extract = limit is None or attempted < limit
        if needs_extraction and not allow_extract:
            stats['remaining'] += 1
            continue
        if needs_extraction:
            attempted += 1
        try:
            res = dx.reconcile_document(rec, names=names, alias_index=idx, use_llm=use_llm,
                                        force=force, allow_extract=allow_extract if needs_extraction else False)
        except Exception as exc:
            logger.exception("[kg-sync documents] %s failed", rec.ref)
            stats['errors'].append(f"{rec.ref}: {exc}")
            continue
        if res['status'] == 'pending':
            stats['remaining'] += 1
            continue
        if res['status'] == 'unchanged':
            stats['unchanged'] += 1
        else:
            stats['processed'] += 1
        stats['written'] += res.get('written', 0)
        stats['rejected'] += res.get('rejected', 0)
        stats['expired'] += res.get('expired', 0)
        if res['status'] == 'metadata':
            stats['links_repaired'] += res.get('written', 0)
        if res['status'] == 'error':
            stats['errors'].append(f"{rec.ref}: {res.get('message')}")
        stats['items'].append(res)
        if res['status'] == 'ok':
            idx.refresh_from_neo4j()
    stats['reconciliation'] = {'unresolved': len(stats['errors']), 'differences': stats['errors'][:100]}
    stats["failed"] = len(stats["errors"])
    stats["remaining"] += stats["failed"]
    stats["complete"] = stats["remaining"] == 0
    if stats["complete"]:
        live_refs = {r.ref for r in recs}
        for ref in existing.keys() - live_refs:
            dx.withdraw_document(ref)
    if stats["errors"]:
        stats["error"] = f"{len(stats['errors'])} document(s) failed: {stats['errors'][0][:200]}"
    stats["items"] = stats["items"][:50]
    return stats
