"""Read-only service-context KG retrieval checks, including tool audit delivery.

Run in leninbot-kg-verify.service to exercise systemd credentials and the public
tool handler. Only the normal tool audit records are written; no KG mutations,
notifications or generative extraction calls are requested.
"""
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')


async def verify():
    from runtime_tools.registry import TOOLS, TOOL_HANDLERS
    from tool_gateway.dispatcher import execute_tool
    from security_gateway import CallerContext, caller_scope
    from security_gateway.audit import _DB_QUEUE
    from kg_runtime.search import _run_rows
    cases = [
        ({'entity': '디아마트 (DiaMat)', 'mode': 'entity', 'num_results': 3}, 'entity', '디아마트 (DiaMat) [Organization]'),
        ({'query': 'DiaMat', 'num_results': 5}, 'ambiguous', 'commulingo:term:dialectical-materialism'),
        ({'entity': '레닌', 'mode': 'entity', 'num_results': 3}, 'entity', 'summary src: commulingo:person:lenin'),
        ({'query': 'Soviet economic reform', 'mode': 'semantic', 'num_results': 5}, 'semantic', 'Facts/Relations'),
        ({'entity': 'kg-verification-nonexistent-7fda001', 'mode': 'entity', 'num_results': 3}, 'entity', 'No knowledge graph results'),
    ]
    name = 'knowledge_graph_search'
    schema = next(t['input_schema'] for t in TOOLS if t['name'] == name)
    ctx = CallerContext(interface='autonomous', agent_name='kg_verification', is_owner=True,
                        task_id='kg-runtime-verification')
    outputs = []
    with caller_scope(ctx):
        for args, path, expected in cases:
            result, error = await execute_tool(name, args, TOOL_HANDLERS, tool_schema=schema)
            metadata = getattr(result, 'result_metadata', None) or {}
            ok = not error and metadata.get('path') == path and expected in result and not metadata.get('fallback')
            ok = ok and metadata.get('result_count', 999) <= args['num_results']
            outputs.append({'args': args, 'ok': ok, 'metadata': metadata})
    await asyncio.to_thread(_DB_QUEUE.join)
    # Document/incident separation must survive source sync.
    docs = _run_rows("MATCH (n:Entity) WHERE 'archival:helsinki-final-act-1975' IN coalesce(n.external_ids, []) RETURN labels(n) AS labels")
    document_ok = len(docs) == 1 and 'Document' in docs[0]['labels']
    print(json.dumps({'checks': outputs, 'document_identity_ok': document_ok}, ensure_ascii=False, indent=2))
    return all(o['ok'] for o in outputs) and document_ok


if __name__ == '__main__':
    raise SystemExit(0 if asyncio.run(verify()) else 1)
