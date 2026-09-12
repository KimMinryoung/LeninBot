"""Rebuild derived alias keys and detach document IDs from non-Document nodes.

Read-only by default. Back up KG before --execute; the next documents full sync
creates proper Document nodes and replaces their source-owned relations.
No entity or relation is deleted or merged.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
from kg_runtime.identity import split_alias_keys, normalize_alias_key
from kg_runtime.search import _get_neo4j_sync_driver


def plan_row(row, terms=None):
    row = dict(row)
    detached_terms = []
    if terms:
        live = [eid for eid in row['external_ids'] if eid in terms]
        if len(live) > 1:
            keep = [eid for eid in live if normalize_alias_key(terms[eid]['name']) == normalize_alias_key(row['name'])]
            if len(keep) != 1:
                raise RuntimeError(f"Ambiguous live source IDs on {row['uuid']}; no automatic split")
            survivor = terms[keep[0]]
            detached_terms = [eid for eid in live if eid != keep[0]]
            def keys(side):
                return {normalize_alias_key(a) for a in [side['name'], *side['aliases']]}
            removed_keys = set().union(*(keys(terms[eid]) for eid in detached_terms)) - keys(survivor)
            row['aliases'] = [a for a in row['aliases'] if normalize_alias_key(a) not in removed_keys]
            row['name_ko'], row['name_en'] = survivor['name_ko'], survivor['name_en']
    strong, weak = split_alias_keys(row['name'], row['aliases'], row.get('name_ko'), row.get('name_en'))
    detached = [eid for eid in row['external_ids'] if eid.startswith(('archival:', 'research:', 'autonote:'))
                and 'Document' not in row['labels']] + detached_terms
    if set(strong) == set(row['alias_keys']) and set(weak) == set(row['weak_keys']) and not detached:
        return None
    return {'uuid': row['uuid'], 'alias_keys': strong, 'weak_keys': weak, 'detached': detached,
            'external_ids': [eid for eid in row['external_ids'] if eid not in detached],
            'aliases': row['aliases'], 'name_ko': row.get('name_ko'), 'name_en': row.get('name_en')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--split-live-terms', action='store_true', help='Separate distinct active term IDs; preserve redirect IDs')
    args = parser.parse_args()
    terms = None
    if args.split_live_terms:
        from jobs.kg_sync_commulingo import load_source
        source = load_source()
        terms = {'commulingo:term:' + key: source.term(key) for key in source.terms}
    with _get_neo4j_sync_driver() as (driver, database):
        with driver.session(database=database) as session:
            rows = session.run('''MATCH (n:Entity) RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels,
                coalesce(n.aliases, []) AS aliases, coalesce(n.alias_keys, []) AS alias_keys,
                coalesce(n.weak_keys, []) AS weak_keys, coalesce(n.external_ids, []) AS external_ids,
                n.name_ko AS name_ko, n.name_en AS name_en''')
            changes = [change for r in rows if (change := plan_row(dict(r), terms))]
            print(json.dumps({'execute': args.execute, 'changed_nodes': len(changes),
                              'detached_external_ids': [eid for c in changes for eid in c['detached']]}, ensure_ascii=False))
            if args.execute:
                session.execute_write(lambda tx: tx.run('''UNWIND $rows AS row
                    MATCH (n:Entity {uuid: row.uuid})
                    SET n.alias_keys = row.alias_keys, n.weak_keys = row.weak_keys,
                        n.external_ids = row.external_ids, n.aliases = row.aliases,
                        n.name_ko = row.name_ko, n.name_en = row.name_en,
                        n.alias_text = reduce(s = '', a IN row.aliases | s + ' / ' + a)''', rows=changes).consume())


if __name__ == '__main__':
    main()
