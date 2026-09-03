#!/usr/bin/env python3
"""Smoke: CommuLingo → KG mirror end-to-end against the live Neo4j, isolated.

Uses a scratch namespace (external ids / sync keys ``smoketest:*``) and a
scratch ``group_id`` so nothing touches real ``commulingo:*`` data, then
deletes everything it created. Exercises the identity layer (external-id
resolution, alias props), sync-key idempotency, changed-fact expiry, id
redirects and the merge path.

Needs Neo4j + the LLM proxy (embeddings). Ad-hoc:
    NEO4J_PASSWORD=$(cat /run/credentials/leninbot-api.service/neo4j_password) \\
    venv/bin/python scripts/smoke_kg_sync.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

NAMESPACE = "smoketest"
GROUP = "smoke_sync"
PASSED = 0
FAILED = 0


def check(cond, label, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        FAILED += 1
        print(f"  FAIL {label} {detail}")


def cypher(q, **params):
    from kg_runtime.search import _get_neo4j_sync_driver
    with _get_neo4j_sync_driver() as (drv, db):
        with drv.session(database=db) as s:
            return [dict(r) for r in s.run(q, **params)]


def cleanup():
    n = cypher("MATCH (n) WHERE n.group_id = $g DETACH DELETE n RETURN count(*) AS c", g=GROUP)
    left = cypher("MATCH (n:Entity) WHERE any(x IN coalesce(n.external_ids, []) WHERE x STARTS WITH $p) "
                  "RETURN count(n) AS c", p=NAMESPACE + ":")
    return n[0]["c"] if n else 0, left[0]["c"] if left else 0


def fixture(sync):
    return sync.Source(
        people=[
            {"id": "smoke-a", "group_id": "smoke-era", "name_ko": "스모크 인물 알파", "name_en": "Smoke Person Alpha",
             "cyrillic": "Смоук Альфа", "years_label": "1900–1950", "bio_ko": "스모크 테스트용 인물."},
            {"id": "smoke-b", "group_id": "smoke-era", "name_ko": "스모크 인물 베타", "name_en": "Smoke Person Beta"},
            {"id": "smoke-b-old", "group_id": "smoke-era", "name_ko": "스모크 인물 베타 구버전", "name_en": "Smoke Person Beta Old"},
        ],
        person_aliases=[{"person_id": "smoke-a", "lang": "en", "alias": "S. Alpha"}],
        career=[{"person_id": "smoke-a", "sort_order": 0, "period_label": "1930–35", "role_ko": "스모크 위원"}],
        person_roles=[{"person_id": "smoke-a", "office_id": "smoke-office", "label_ko": "스모크 수장", "category_id": None},
                      {"person_id": "smoke-b", "office_id": None, "label_ko": "", "category_id": None},
                      {"person_id": "smoke-b-old", "office_id": None, "label_ko": "", "category_id": None}],
        role_categories=[],
        people_groups=[{"id": "smoke-era", "range_label": "1900–1950", "title_ko": "스모크 시대의 사람들", "title_en": "Smoke era"}],
        offices=[{"id": "smoke-office", "title_ko": "스모크 기관", "title_en": "Smoke office", "blurb_ko": "테스트 기관"}],
        office_rows=[{"id": 990001, "office_id": "smoke-office", "period_label": "1930–1935", "start_year": 1930,
                      "start_month": None, "end_year": 1935, "end_month": 6, "body_ko": "스모크 수장", "person_id": "smoke-a"}],
        events=[{"id": "smoke-event", "period_label": "1931", "title_ko": "스모크 사건", "title_en": "Smoke Incident",
                 "summary_ko": "스모크 테스트 사건.", "locations": [{"kind": "main", "label": {"en": "Smoke City", "ko": "스모크 시"}}]}],
        event_people=[{"event_id": "smoke-event", "person_id": "smoke-a", "relation_ko": "주도자", "relation_kind": "leader",
                       "note_ko": "스모크 사건을 주도했다."}],
        terms=[{"id": "smoke-term", "term_ko": "스모크주의", "term_en": "Smokeism", "original": "Смоукизм",
                "definition_ko": "스모크 테스트 용어.", "category": None, "parent_id": None}],
        term_aliases=[{"term_id": "smoke-term", "lang": "ko", "alias": "스모크 이론"}],
        term_categories=[],
        term_relations=[],
        term_people=[{"term_id": "smoke-term", "person_id": "smoke-a"}],
        term_events=[{"term_id": "smoke-term", "event_id": "smoke-event", "same_subject": True}],
        redirects=[],
    )


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    from jobs import kg_sync_commulingo as sync

    sync.NAMESPACE = NAMESPACE
    prefix = NAMESPACE + ":"

    print("[0] pre-clean scratch group")
    cleanup()

    src = fixture(sync)
    facts = sync.build_facts(src)
    print(f"[1] built {len(facts)} facts; writing to group '{GROUP}'")
    res = sync.write_facts(facts, group_id=GROUP)
    print("   ", {k: v for k, v in res.items() if k != "errors"}, res["errors"][:1])
    check(not res["errors"], "first write has no errors")
    check(res["written"] == len(facts), "all facts written", f"{res['written']}/{len(facts)}")

    print("[2] identity props on nodes")
    rows = cypher(
        "MATCH (n:Entity) WHERE $eid IN coalesce(n.external_ids, []) "
        "RETURN n.name AS name, n.aliases AS aliases, n.alias_keys AS keys, n.name_en AS name_en, "
        "n.summary AS summary, labels(n) AS labels, n.group_id AS g", eid=f"{prefix}person:smoke-a")
    check(len(rows) == 1, "one node per external id", str(len(rows)))
    if rows:
        r = rows[0]
        check(r["name"] == "스모크 인물 알파", "canonical name is name_ko", r["name"])
        check("Smoke Person Alpha" in (r["aliases"] or []) and "S. Alpha" in (r["aliases"] or []), "aliases stored", str(r["aliases"]))
        check("smoke person alpha" in (r["keys"] or []) and "s alpha" in (r["keys"] or []), "alias keys normalized", str(r["keys"]))
        check(r["name_en"] == "Smoke Person Alpha", "name_en stored")
        check("스모크 위원" in (r["summary"] or ""), "summary carries career", (r["summary"] or "")[:80])
        check("Person" in r["labels"] and r["g"] == GROUP, "label + scratch group")

    print("[3] edges: sync_key, attributes, dates")
    edges = cypher(
        "MATCH (a)-[r:RELATES_TO]->(b) WHERE r.sync_key STARTS WITH $p "
        "RETURN a.name AS s, r.name AS pred, b.name AS o, r.sync_key AS key, r.valid_at AS va, "
        "r.invalid_at AS ia, r.position AS position, r.reference_type AS rt, r.role_in_incident AS role", p=prefix)
    check(len(edges) == len(facts), "edge count equals facts", f"{len(edges)}/{len(facts)}")
    by_key = {e["key"]: e for e in edges}
    row = by_key.get(f"{prefix}office_row:990001")
    check(row is not None and row["va"] is not None and row["ia"] is not None, "office row has valid_at/invalid_at", str(row))
    check(row is not None and row["position"] == "스모크 수장", "edge attribute persisted (position)")
    inv = by_key.get(f"{prefix}event_person:smoke-event:smoke-a")
    check(inv is not None and inv["role"] == "leader" and inv["pred"] == "Involvement", "involvement attrs", str(inv))
    tp = by_key.get(f"{prefix}term_person:smoke-term:smoke-a")
    check(tp is not None and tp["rt"] == "person_term" and tp["pred"] == "Reference", "Reference edge with reference_type", str(tp))

    print("[4] idempotency: second diff finds nothing new")
    existing = sync.existing_sync_edges(prefix)
    missing = [f["attributes"]["sync_key"] for f in facts if f["attributes"]["sync_key"] not in existing]
    changed = [f for f in facts if f["attributes"]["sync_key"] in existing and existing[f["attributes"]["sync_key"]]["fact"] != f["fact"]]
    check(not missing and not changed, "no new/changed facts on re-run", f"missing={missing[:2]} changed={len(changed)}")

    print("[5] changed fact → expire old edge, write new, entity reused (no duplicate node)")
    src.events["smoke-event"]["summary_ko"] = "스모크 테스트 사건 (개정)."
    src.event_people[0]["note_ko"] = "스모크 사건을 새로 주도했다."
    facts2 = sync.build_facts(src, changed={"person": set(), "event": {"smoke-event"}, "term": set(), "office": set()})
    existing = sync.existing_sync_edges(prefix)
    to_write, to_expire = [], []
    for f in facts2:
        old = existing.get(f["attributes"]["sync_key"])
        if old is None:
            to_write.append(f)
        elif old["fact"] != f["fact"]:
            to_expire.append(old["uuid"]); to_write.append(f)
    check(len(to_expire) == 1 and len(to_write) == 1, "exactly one changed fact detected", f"{len(to_expire)}/{len(to_write)}")
    expired = sync.expire_edges(to_expire)
    res2 = sync.write_facts(to_write, group_id=GROUP)
    check(expired == 1 and res2["written"] == 1 and res2["new_entities"] == 0, "expired 1, wrote 1, reused entities", str((expired, res2["written"], res2["new_entities"])))
    n_person = cypher("MATCH (n:Entity) WHERE $eid IN coalesce(n.external_ids, []) RETURN count(n) AS c", eid=f"{prefix}person:smoke-a")[0]["c"]
    check(n_person == 1, "still one node for the person", str(n_person))
    active = cypher("MATCH ()-[r:RELATES_TO]->() WHERE r.sync_key = $k AND r.expired_at IS NULL RETURN count(r) AS c",
                    k=f"{prefix}event_person:smoke-event:smoke-a")[0]["c"]
    check(active == 1, "one active edge for the changed key", str(active))

    print("[6] id redirect merges the old node into the canonical one")
    stats = sync.apply_redirects([{"entity_type": "person", "from_id": "smoke-b-old", "to_id": "smoke-b"}])
    check(stats["merged"] == 1, "redirect merged", str(stats))
    b = cypher("MATCH (n:Entity) WHERE $new IN coalesce(n.external_ids, []) RETURN n.external_ids AS ids, n.aliases AS aliases, "
               "size([(n)-[r:RELATES_TO]-() | r]) AS deg", new=f"{prefix}person:smoke-b")
    check(len(b) == 1 and f"{prefix}person:smoke-b-old" in b[0]["ids"], "canonical carries both external ids", str(b))
    check(b and "스모크 인물 베타 구버전" in (b[0]["aliases"] or []), "old name kept as alias", str(b[0]["aliases"] if b else None))
    gone = cypher("MATCH (n:Entity {name: '스모크 인물 베타 구버전'}) RETURN count(n) AS c")[0]["c"]
    check(gone == 0, "old node deleted")

    print("[7] search fallback sees scratch data by alias text")
    from kg_runtime.identity import AliasIndex
    idx = AliasIndex(); idx.refresh_from_neo4j()
    hits = idx.match("스모크 인물 알파가 스모크 사건을 주도했다")
    names = {h.name for h in hits}
    check("스모크 인물 알파" in names and "스모크 사건" in names, "alias index matches scratch entities", str(names))

    print("[8] cleanup")
    deleted, left = cleanup()
    check(left == 0, "no scratch entities remain", f"deleted={deleted} left={left}")

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        try:
            cleanup()
        except Exception:
            pass
