import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm.call_registry import Decision, DecisionResult
from runtime_tools import roleplay_review as review, roleplay_turn as turn
from runtime_tools.roleplay_memory import STATE_DEFAULTS
from runtime_tools.roleplay_dynamics import with_defaults


class ReviewScreenTests(unittest.TestCase):
    def test_only_confident_clean_decisions_skip_generation(self):
        profile = SimpleNamespace(extra={'enabled': True, 'thresholds': {'accept': .9}})
        cases = [('consistent', .95, True), ('consistent', .89, False),
                 ('contradiction', .99, False), ('uncertain', .99, False),
                 ('invented', 1, False), ('consistent', None, False),
                 ('consistent', float('nan'), False), ('consistent', 2, False)]
        for label, confidence, clean in cases:
            decision = Decision(model='test', cost_usd=.0001, answers={
                'consistency': {'choice': label, 'confidence': confidence}})
            with self.subTest(label=label, confidence=confidence), \
                 patch.object(review, 'resolve', return_value=profile), \
                 patch.object(review, 'decide_detailed', return_value=DecisionResult(decision=decision)) as call:
                result = review.screen_reply({'draft': 'scene'})
                self.assertEqual(result['clean'], clean)
                self.assertEqual(result['cost_usd'], .0001)
                self.assertEqual(call.call_count, 1)

    def test_disabled_and_unavailable_require_detailed_review(self):
        for enabled in (False, True):
            with patch.object(review, 'resolve', return_value=SimpleNamespace(extra={'enabled': enabled})), \
                 patch.object(review, 'decide_detailed', return_value=DecisionResult(error_kind='transport')) as call:
                result = review.screen_reply({'draft': 'scene'})
                self.assertFalse(result['clean'])
                self.assertEqual(call.call_count, int(enabled))

    def test_screen_controls_fallback_and_preserves_minimal_payload(self):
        import bot_config
        profile = SimpleNamespace(provider='deepseek_anthropic', extra={'enabled': True, 'jev_precheck': True})
        connection = SimpleNamespace(base_url=bot_config.DEEPSEEK_ANTHROPIC_BASE_URL)
        state = with_defaults(dict(STATE_DEFAULTS))
        stage = {'baseline': {'notes': [('old', 'private')]},
                 'records': {'notes': [('old', 'private'), ('new', 'changed')]}}
        for clean in (True, False):
            with self.subTest(clean=clean), patch.object(turn, 'resolve', return_value=profile), \
                 patch.object(turn, 'resolve_provider_connection', return_value=connection), \
                 patch.object(turn, 'screen_reply', return_value={'clean': clean}) as screen, \
                 patch.object(turn, 'generate_detailed', return_value=SimpleNamespace(
                     text='{"approved":false,"issues":["location conflict"]}', error_kind=None, truncated=False)) as generate:
                result = turn.review_reply('draft', state, state, stage, applied={'events': ['water'], 'minutes': 45, 'intensity': 'mild'})
                self.assertEqual(generate.call_count, int(not clean))
                self.assertEqual(result['approved'], clean)
                self.assertEqual(result['screen'], {'clean': clean})
                payload = screen.call_args.args[0]
                self.assertEqual(payload['new_records']['notes'], [('new', 'changed')])
                self.assertNotIn('resolve', payload['settled_state'])
                self.assertEqual(payload['settled_events'], ['물'])
                self.assertNotIn('minutes', payload)
                if not clean:
                    self.assertEqual(json.loads(generate.call_args.args[1])['settled_events'], ['물'])
                if not clean:
                    self.assertEqual(result['issues'], ['location conflict'])
