"""Offline editor boundaries shared by unittest and pytest.

Unit cases execute already-mocked synchronous dependencies inline. Real
executor shutdown is an integration concern; restricted sandboxes can lose
asyncio's cross-thread wakeup even for a lambda. Set COMMULINGO_TEST_REAL_THREADS
only when explicitly checking that boundary outside such a sandbox.
"""
from contextlib import ExitStack, contextmanager
import os
import unittest
from unittest.mock import patch

os.environ['LENINBOT_LLM_AUDIT_DB'] = '0'


@contextmanager
def no_external_io():
    """Fail even if production error handling swallows an unexpected IO attempt."""
    attempted = []
    def blocked(boundary):
        def fail(*args, **kwargs):
            attempted.append(boundary)
            raise AssertionError(f'unmocked external IO: {boundary}')
        return fail
    with ExitStack() as stack:
        for boundary in ('psycopg2.connect', 'subprocess.run', 'subprocess.Popen',
                         'httpx.Client.send', 'httpx.AsyncClient.send',
                         'requests.sessions.Session.send', 'socket.socket.connect',
                         'socket.socket.connect_ex'):
            stack.enter_context(patch(boundary, side_effect=blocked(boundary)))
        try:
            yield
        finally:
            if attempted:
                raise AssertionError('unexpected external IO: ' + ', '.join(attempted))


class HermeticAsyncCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(no_external_io())
        if os.getenv('COMMULINGO_TEST_REAL_THREADS') != '1':
            async def inline(call, *args, **kwargs):
                return call(*args, **kwargs)
            self.enterContext(patch('asyncio.to_thread', side_effect=inline))


def citation_result(feature, state, questions, *, support='supports', **defaults):
    """Fake only the provider reply; real passage/gate/cache/accounting code runs."""
    from llm.call_registry import Decision, DecisionResult
    if feature not in {'commulingo_citation_support', 'commulingo_review_citation_support'}:
        raise AssertionError(f'unmocked classification: {feature}')
    answers = {}
    for cid in state['items']:
        answers.update({
            f'{cid}_support': {'choice': support, 'confidence': .99},
            f'{cid}_specific': {'noul': .99},
            f'{cid}_boilerplate': {'noul': 0},
        })
    if set(answers) != set(questions):
        raise AssertionError('unexpected citation questions')
    return DecisionResult(decision=Decision(answers=answers, model='fixture/jev', cost_usd=.0001))


class EditorCase(HermeticAsyncCase):
    def setUp(self):
        super().setUp()
        self.jev = self.enterContext(patch('llm.call_registry.decide_detailed', side_effect=citation_result))
        self.enterContext(patch('commulingo.pipeline.citation_gate.settings', return_value={
            'enabled': True, 'enforce': True, 'thresholds': {'reject': .85, 'boilerplate': .9}}))
