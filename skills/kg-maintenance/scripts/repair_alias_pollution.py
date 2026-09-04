#!/usr/bin/env python3
"""Repair the 2026-09-03 alias pollution (dry-run by default).

What went wrong: the first document LLM extraction resolved entities through
the aliases the model supplied and then unioned name + aliases into whatever
node it landed on. United States absorbed 소련/중국/프랑스/스탈린/CIA/연준 …
(60 aliases), SpaceX absorbed IRGC/파르스통신, and common nouns (정권, 노조,
회사, 선언 …) became strong aliases of specific organizations. Every later
document mention of '소련' then attached to the United States node.

Steps (each reported; nothing is written without --execute):
  1. backup (text-only) into data/kg_backups/
  2. strip aliases: explicit keep-lists for the polluted hubs, generic nouns
     everywhere (identity.GENERIC_ENTITY_NAMES), a few hand-picked wrong ones;
     alias_keys / weak_keys / alias_text are rebuilt from what remains
  3. delete every document-derived edge (r.doc_ref set) — LLM facts and
     alias-index mentions alike — then orphaned `documents`-group nodes.
     `python -m jobs.kg_sync --source documents --full --force` rebuilds them
     with the trusted-source rules (run it after this script).
  4. repair fact text mangled by the case-insensitive abbreviation map
     (who → World Health Organization, us → United States)

Usage:
  scripts/psql-main is not needed; Neo4j only. From the repo root:
    set -a && source <(grep -vE '^\\s*#|^\\s*$' .env) && set +a
    export CREDENTIALS_DIRECTORY=/run/credentials/leninbot-api.service
    ./venv/bin/python skills/kg-maintenance/scripts/repair_alias_pollution.py [--execute]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "skills", "kg-maintenance", "scripts"))

from kg_runtime.identity import (  # noqa: E402
    GENERIC_ENTITY_NAMES, normalize_alias_key, split_alias_keys,
)
from kg_runtime.search import _get_neo4j_sync_driver  # noqa: E402

# Hubs whose alias lists were poisoned: keep ONLY these (case/spacing exact).
KEEP = {
    "United States": {"미국", "美", "US", "USA", "U.S.", "미합중국", "북아메리카 합중국", "아메리카",
                      "미 정부", "미국 정부", "Trump administration", "트럼프 행정부", "트럼프 2기",
                      "바이든 정부", "바이든 행정부", "미 제국주의", "미국 제국주의 세력", "미"},
    "Soviet Union": {"소련", "소비에트", "소비에트 연방", "소비에트 사회주의 공화국 연방", "USSR", "CCCP", "СССР"},
    "SpaceX": {"스페이스X", "스페이스엑스"},
    "Iran": {"이란", "이란 정부", "테헤란", "이란 이슬람공화국", "Islamic Republic of Iran"},
    "Islamic Revolutionary Guard Corps": {"IRGC", "이란 혁명수비대", "이란 이슬람혁명수비대", "혁명수비대"},
}
# (node name, alias) pairs that are simply wrong
STRIP = {
    ("Bab-el-Mandeb Strait", "홍해"), ("박재현", "재현"), ("김세연", "세연"), ("오달수", "달수"), ("최수빈", "수빈"),
    ("여수산단", "여수"), ("Circle", "서클"), ("Brent Crude", "유가"), ("Interest Rate", "금리"),
    ("Government Bond", "국채"), ("원/달러 환율", "환율"), ("Crude oil", "원유"), ("자영업자 대출", "대출"),
    ("한국 수출", "수출"), ("고용 지표", "고용"), ("임금노동과 자본", "강연"), ("국민의힘", "여당"),
    ("웹진 반란", "반란"), ("폴란드통일노동자당", "정권"), ("삼성바이오로직스 상생노조", "노조"),
    ("Samsung Biologics", "회사"), ("반소비에트 음모 활동", "음모"), ("공산당 선언", "선언"),
}
# fact-text repairs: edge uuid → (old fragment, new fragment) decided by hand
# Only verbs that take a pronoun object: "of/for/with/to United States imports"
# came from a legitimate "US imports" and must stay.
_PRONOUN_VERBS = r"(let|lets|tell|tells|told|telling|give|gives|gave|giving|help|helps|helped|allow|allows|allowed|among|join|joins|joined|remind|reminds|reminded|leave|leaves|left|keep|keeps|kept|bring|brings|brought|teach|taught)"
US_PRONOUN_RE = re.compile(r"\b" + _PRONOUN_VERBS + r" United States\b(?! [A-Z0-9])")
_WHO_STOP_PREV = {"and", "the", "of", "by", "to", "with", "from", "at", "or", "a", "an", "in", "on", "for", "under", "via"}


def fix_who(text: str) -> str:
    """'…, World Health Organization said' / 'employee World Health Organization asked' → 'who'
    when the previous token is a lowercase word or comma (not a preposition/
    article) and the next token is lowercase — i.e. a relative pronoun."""
    def repl(m):
        prev, nxt = m.group(1), m.group(2)
        word = prev.rstrip(",;")
        if word in _WHO_STOP_PREV or not (word.islower() or prev.endswith(",")):
            return m.group(0)
        return f"{prev} who {nxt}"
    return re.sub(r"(\S+) World Health Organization (\S+)", repl, text)


def rows(session, cypher, **p):
    return [dict(r) for r in session.run(cypher, **p)]


def step_backup(execute: bool) -> str | None:
    if not execute:
        print("[1] backup: skipped in dry-run")
        return None
    from secrets_loader import get_secret
    os.environ.setdefault("NEO4J_PASSWORD", get_secret("NEO4J_PASSWORD") or "")
    from backup_kg import backup
    ts = backup(include_embeddings=False)
    print(f"[1] backup written: data/kg_backups/*_{ts}.json")
    return ts


def step_aliases(session, execute: bool) -> list[dict]:
    plan = []
    nodes = rows(session, """
        MATCH (n:Entity) WHERE size(coalesce(n.aliases, [])) > 0
        RETURN n.uuid AS uuid, n.name AS name, n.aliases AS aliases, n.name_ko AS name_ko, n.name_en AS name_en,
               coalesce(n.external_ids, []) AS ext""")
    for n in nodes:
        name, aliases = n["name"], list(n["aliases"] or [])
        keep = KEEP.get(name)
        removed = []
        kept = []
        for a in aliases:
            k = normalize_alias_key(a)
            bad = False
            if keep is not None and a not in keep:
                bad = True
            elif k in GENERIC_ENTITY_NAMES:
                bad = True
            elif (name, a) in STRIP:
                bad = True
            (removed if bad else kept).append(a)
        if not removed:
            continue
        strong, weak = split_alias_keys(name, kept, n["name_ko"], n["name_en"])
        plan.append({"uuid": n["uuid"], "name": name, "removed": removed, "kept": kept,
                     "alias_keys": strong, "weak_keys": weak})
    print(f"[2] alias strip: {len(plan)} node(s), {sum(len(p['removed']) for p in plan)} alias(es)")
    for p in plan:
        print(f"    - {p['name']}: -{p['removed']}")
    if execute:
        for p in plan:
            session.run("""
                MATCH (n:Entity {uuid: $uuid})
                SET n.aliases = $aliases, n.alias_keys = $keys, n.weak_keys = $weak,
                    n.alias_text = $text""", uuid=p["uuid"], aliases=p["kept"], keys=p["alias_keys"],
                        weak=p["weak_keys"], text=" / ".join(p["kept"])).consume()
        print("    applied")
    return plan


def step_doc_edges(session, execute: bool) -> dict:
    counts = rows(session, """
        MATCH ()-[r:RELATES_TO]->() WHERE r.doc_ref IS NOT NULL
        RETURN coalesce(r.extraction, CASE WHEN r.reference_type IS NOT NULL THEN 'reference:' + r.reference_type ELSE 'other' END) AS kind,
               count(*) AS c ORDER BY c DESC""")
    total = sum(c["c"] for c in counts)
    print(f"[3] document-derived edges to delete: {total}")
    for c in counts:
        print(f"    - {c['kind']}: {c['c']}")
    orphans = rows(session, """
        MATCH (n:Entity) WHERE n.group_id = 'documents' AND NOT n:Document
        OPTIONAL MATCH (n)-[r:RELATES_TO]-() WHERE r.doc_ref IS NULL
        WITH n, count(r) AS other
        WHERE other = 0
        RETURN n.uuid AS uuid, n.name AS name""")
    print(f"    documents-group nodes that become orphans (deleted, rebuilt by re-sync): {len(orphans)}")
    print("    e.g. " + ", ".join(o["name"] for o in orphans[:40]))
    if execute:
        deleted = 0
        while True:
            rec = session.run("""
                MATCH ()-[r:RELATES_TO]->() WHERE r.doc_ref IS NOT NULL
                WITH r LIMIT 2000 DELETE r RETURN count(*) AS c""").single()
            deleted += rec["c"]
            if rec["c"] == 0:
                break
        gone = session.run("""
            MATCH (n:Entity) WHERE n.group_id = 'documents' AND NOT n:Document AND NOT (n)-[:RELATES_TO]-()
            DETACH DELETE n RETURN count(*) AS c""").single()["c"]
        # Document nodes keep their sha; --force ignores it.
        print(f"    deleted {deleted} edge(s), {gone} orphan node(s)")
    return {"edges": total, "orphans": len(orphans)}


def step_fact_text(session, execute: bool) -> list[dict]:
    fixes = []
    for r in rows(session, "MATCH ()-[r:RELATES_TO]->() WHERE r.fact CONTAINS 'World Health Organization' RETURN r.uuid AS uuid, r.fact AS fact"):
        new = fix_who(r["fact"])
        if new != r["fact"]:
            fixes.append({"uuid": r["uuid"], "old": r["fact"], "new": new})
    for r in rows(session, "MATCH ()-[r:RELATES_TO]->() WHERE r.fact CONTAINS ' United States' RETURN r.uuid AS uuid, r.fact AS fact"):
        new = US_PRONOUN_RE.sub(lambda m: m.group(1) + " us", r["fact"])
        if new != r["fact"]:
            fixes.append({"uuid": r["uuid"], "old": r["fact"], "new": new})
    print(f"[4] fact-text repairs: {len(fixes)}")
    for f in fixes:
        print(f"    - {f['old'][:150]}\n      → {f['new'][:150]}")
    if execute:
        for f in fixes:
            session.run("MATCH ()-[r:RELATES_TO]->() WHERE r.uuid = $u SET r.fact = $f", u=f["uuid"], f=f["new"]).consume()
        print("    applied")
    return fixes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--skip-doc-edges", action="store_true", help="only aliases + fact text")
    args = ap.parse_args()
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"== repair_alias_pollution [{mode}] {datetime.now():%Y-%m-%d %H:%M:%S}")
    report = {"mode": mode}
    report["backup"] = step_backup(args.execute)
    with _get_neo4j_sync_driver() as (drv, db):
        with drv.session(database=db) as s:
            report["aliases"] = step_aliases(s, args.execute)
            if not args.skip_doc_edges:
                report["doc_edges"] = step_doc_edges(s, args.execute)
            report["fact_text"] = step_fact_text(s, args.execute)
    out = os.path.join(ROOT, "data", "kg_backups", f"repair_alias_pollution_{mode.lower()}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
