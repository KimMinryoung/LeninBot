"""Policy checks without invoking an LLM or loading production credentials."""
import ast
import os
from pathlib import Path
import unittest

ROOT = Path(os.environ.get('COMMULINGO_TEST_SOURCE', '/home/grass/leninbot'))
source = ROOT / 'scripts/commulingo_people_maintainer.py'
module = ast.parse(source.read_text())
selected = [n for n in module.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in {'select_sparse_person', 'enrich_step', 'build_no_edit_handler', '_call_curator_stage'}]
queries = []
namespace = {'db_query': lambda sql, params: queries.append((sql, params)) or [], 'json': __import__('json'), 'MAX_SECTIONS': 12}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)

class EditorialSelection(unittest.IsolatedAsyncioTestCase):
    def test_selection_query_and_step_contract(self):
        namespace['select_sparse_person'](30, exclude_ids=['excluded-person'])
        sql, params = queries[-1]
        self.assertIn("q.status='pending'", sql)
        self.assertIn('e.review_after>NOW()', sql)
        self.assertIn("editorial_states ? 'events'", sql)
        self.assertIn("editorial_states ? 'sections'", sql)
        self.assertEqual(params['excluded'], ['excluded-person'])
        self.assertEqual(namespace['enrich_step']({'editorial_step': 7}), 7)
        self.assertEqual(namespace['enrich_step']({'editorial_step': 8}), 8)
    async def test_pending_tool_result_is_terminal_without_applied_edit(self):
        from types import SimpleNamespace
        async def pending(**kwargs): return 'OK — pending: review required'
        class Memory:
            def __init__(self, key): pass
            async def chat(self, *args, **kwargs):
                await kwargs['tool_handlers']['write']()
                return 'A model summary that does not mention pending'
        namespace.update(ResearchMemory=Memory, NARROW_WRITE_TOOLS={'write'},
            resolve_agent_tool_loop=lambda spec, policy: SimpleNamespace(chat=None, client=None, model='test', render_provider='test', reasoning={}),
            completed_run_count=lambda: 10)
        policy = SimpleNamespace(max_output_continuations=0, max_rounds=1, max_output_tokens=100, max_input_tokens=100, budget_usd=0)
        result, _, _ = await namespace['_call_curator_stage'](task='test',
            spec=SimpleNamespace(name='test', render_prompt=lambda **kw: ''), tools=[], handlers={'write': pending},
            policy=policy, stage='enrich', expect_edit=True, before_count=10, finalization_tools=['write'], terminal_tools=['write'])
        self.assertTrue(result.startswith('OK — pending:'))
    async def test_completion_is_explicit_and_validated(self):
        box = {}
        handler = namespace['build_no_edit_handler'](box)
        await handler(reason='Reviewed the archive', status='not_applicable', sources=['Archive index'])
        self.assertEqual(box['status'], 'not_applicable')
        with self.assertRaises(ValueError): await handler(reason='yes', status='invented')
        with self.assertRaises(ValueError): await handler(reason='')

if __name__ == '__main__': unittest.main()
