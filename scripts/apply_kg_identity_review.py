"""Apply the individually reviewed KG identities. Back up KG before --execute.

Default is read-only. Applies all pending changes in one transaction, verifies
relation properties and source IDs, and refreshes embeddings for renamed nodes.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kg_runtime.identity_review import CONFIG
from kg_runtime.identity import build_identity_props, merge_entity_nodes_sync
from kg_runtime.search import _get_neo4j_sync_driver


def inspect(session, entries):
    ids = {e['uuid'] for e in entries} | {e['target_uuid'] for e in entries}
    rows = {r['uuid']: dict(r) for r in session.run('''MATCH (n:Entity) WHERE n.uuid IN $ids
        RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels,
        coalesce(n.external_ids, []) AS external_ids, coalesce(n.aliases, []) AS aliases,
        n.name_ko AS name_ko, n.name_en AS name_en, n.reviewed_name AS reviewed_name''', ids=list(ids))}
    for entry in entries:
        target = rows.get(entry['target_uuid'])
        if not target or not set(entry['target_types']).issubset(target['labels']):
            raise RuntimeError(f"Missing or changed target: {entry['target_uuid']}")
        row = rows.get(entry['uuid'])
        if row is None and entry['action'] == 'merge':
            continue
        if (not row or row['name'] not in (entry['original_name'], entry['name'])
                or set(row['external_ids']) != set(entry['external_ids'])):
            raise RuntimeError(f"Review is stale: {entry['uuid']}")
    return rows


def relation_snapshot(tx, ids):
    return {r['uuid']: dict(r) for r in tx.run('''MATCH (n:Entity)-[r:RELATES_TO]-()
        WHERE n.uuid IN $ids RETURN DISTINCT r.uuid AS uuid, properties(r) AS props,
        startNode(r).uuid AS source, endNode(r).uuid AS target''', ids=list(ids))}


def apply(tx, entries, embeddings):
    rows = inspect(tx, entries)
    before = relation_snapshot(tx, rows)
    redirects = {e['uuid']: e['target_uuid'] for e in entries if e['action'] == 'merge'}
    merged, renamed = [], []
    for entry in entries:
        if entry['action'] == 'merge' and entry['uuid'] in rows:
            merged.append(merge_entity_nodes_sync(tx, entry['target_uuid'], [entry['uuid']]))
        elif entry['action'] == 'rename':
            row = rows[entry['uuid']]
            aliases = list(dict.fromkeys([*row['aliases'], entry['original_name']]))
            props = build_identity_props(entry['name'], aliases=aliases,
                                         external_ids=row['external_ids'],
                                         name_ko=entry['name'], name_en=row['name_en'])
            tx.run('''MATCH (n:Entity {uuid:$uuid})
                SET n.name=$name, n.reviewed_name=$name, n.curated_name=$name,
                    n.name_ko=$name, n.aliases=$aliases, n.alias_keys=$keys,
                    n.weak_keys=$weak, n.alias_text=$text, n.name_embedding=$embedding,
                    n.identity_review_reason=$reason, n.identity_reviewed_at=datetime()''',
                uuid=entry['uuid'], name=entry['name'], aliases=props['aliases'], keys=props['alias_keys'],
                weak=props['weak_keys'], text=props['alias_text'], embedding=embeddings[entry['uuid']],
                reason=entry['reason']).consume()
            renamed.append(entry['uuid'])
    after = relation_snapshot(tx, {redirects.get(i, i) for i in rows})
    expected = {}
    removed = []
    for uuid, edge in before.items():
        edge['source'] = redirects.get(edge['source'], edge['source'])
        edge['target'] = redirects.get(edge['target'], edge['target'])
        if edge['source'] == edge['target']:
            removed.append(uuid)
        else:
            expected[uuid] = edge
    if expected != after:
        raise RuntimeError('Relation content/provenance changed; rolling back review')
    return {'merged': merged, 'renamed': renamed, 'self_loop_edges_removed': removed,
            'preserved_relations': len(after)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    entries = json.loads(CONFIG.read_text())['entries']
    with _get_neo4j_sync_driver() as (driver, database):
        with driver.session(database=database) as session:
            rows = inspect(session, entries)
            pending = [e for e in entries if (e['action'] == 'merge' and e['uuid'] in rows)
                       or (e['action'] == 'rename' and rows[e['uuid']]['reviewed_name'] != e['name'])]
    if not args.execute:
        print(json.dumps({'pending': pending}, ensure_ascii=False, indent=2))
        return
    embeddings = {}
    names = [e for e in pending if e['action'] == 'rename']
    if names:
        from kg_runtime.service_runtime import get_kg_service, run_kg_task
        service = get_kg_service()
        if service is None:
            raise RuntimeError('KG embedding service unavailable')
        async def embed():
            from graphiti_core.nodes import EntityNode, create_entity_node_embeddings
            nodes = [EntityNode(uuid=e['uuid'], name=e['name'], group_id='documents') for e in names]
            await create_entity_node_embeddings(service._graphiti.embedder, nodes)
            return {node.uuid: node.name_embedding for node in nodes}
        embeddings = run_kg_task(embed)
        if any(not v for v in embeddings.values()):
            raise RuntimeError('Missing reviewed name embedding')
    from kg_runtime.locks import kg_write_lock
    with kg_write_lock('sync:commulingo'), kg_write_lock('sync:documents'):
        with _get_neo4j_sync_driver() as (driver, database):
            with driver.session(database=database) as session:
                result = session.execute_write(apply, pending, embeddings)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'data' / 'kg_backups' / f'identity_review_{stamp}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('Report:', output)


if __name__ == '__main__':
    main()
