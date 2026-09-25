"""Non-scene operations must not depend on scene model availability."""
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from roleplay import turn, jev
from roleplay.timing import validate_appointment
from test_roleplay_jev import initial, verdict


def authorization(mode, text='현재 요청', **values):
    return {'user_text': text, 'labels': {'mode': mode, 'transition': 'current',
            'span': 'brief', 'time_scope': 'none'}, 'duration_minutes': 0,
            'corrections': {}, 'appointment': None, 'model': 'input-fixture', **values}


class ModeRouteTests(TestCase):
    def test_non_scene_modes_need_no_event_duration_or_review_calls(self):
        cases = [authorization('discussion'), authorization('correction', corrections={'hunger': 20}),
                 authorization('reset'), authorization('plan', duration_minutes=20, appointment={'title': '방문'})]
        for auth in cases:
            with self.subTest(mode=auth['labels']['mode']), \
                 patch.object(jev, 'classify', side_effect=AssertionError('no scene call')) as classify, \
                 patch.object(jev, 'estimate_duration', side_effect=AssertionError('no duration call')) as duration, \
                 patch.object(turn, 'review_reply', side_effect=AssertionError('no narrative screen/review')) as review:
                before = initial()
                saved = deepcopy(before)
                # Even a stale supplied scene verdict must not leak events into these modes.
                out = turn.prepare(auth['user_text'], before, [], [], 'scope', '답변', auth,
                                   verdict=verdict(event='meal'))
                classify.assert_not_called()
                duration.assert_not_called()
                review.assert_not_called()
                self.assertEqual(before, saved)
                self.assertEqual(out['verdict']['classification_skipped'], 'non_scene')
                self.assertEqual(out['verdict']['calls'], {})
                self.assertEqual(out['verdict']['final_review']['status'], 'skipped')
                self.assertIsNone(out['verdict']['final_review']['approved'])
                mode = auth['labels']['mode']
                if mode == 'discussion':
                    self.assertEqual(out['state'], before)
                elif mode == 'correction':
                    self.assertEqual(out['state']['hunger'], 20)
                    self.assertEqual(out['state']['scene_minute'], before['scene_minute'])
                elif mode == 'reset':
                    self.assertIsNone(out['state']['hunger'])
                else:
                    event = out['state']['story_events'][0]
                    self.assertEqual((event['title'], event['due_minute']), ('방문', 20))
                    self.assertEqual(out['state']['scene_minute'], before['scene_minute'])

    def test_scene_retains_event_effects_and_review(self):
        auth = authorization('scene', '식사를 먹었다')
        with patch.object(jev, 'classify', return_value=verdict(event='meal')) as classify, \
             patch.object(jev, 'estimate_duration', return_value={'elapsed_minutes': 3}), \
             patch.object(turn, 'review_reply', return_value={'approved': True, 'issues': []}) as review:
            out = turn.prepare(auth['user_text'], initial(), [], [], 'scope', '식사를 마쳤다', auth)
        classify.assert_called_once()
        review.assert_called_once()
        self.assertEqual(out['applied']['events'], ['meal'])
        self.assertLess(out['state']['hunger'], initial()['hunger'])

    def test_plan_does_not_invent_missing_event_or_delay(self):
        for appointment, minutes in [(None, 20), ({'title': ''}, 20), ({'title': '방문'}, 0),
                                     ({'title': '방문'}, True), ({'title': '방문'}, 10081),
                                     ({'title': '방문', 'action': 'complete'}, 20)]:
            with self.subTest(appointment=appointment, minutes=minutes), self.assertRaises(ValueError):
                auth = authorization('plan', duration_minutes=minutes, appointment=appointment)
                turn.prepare(auth['user_text'], initial(), [], [], 'scope', '답변', auth)
        with self.assertRaises(ValueError):
            validate_appointment({'title': '방문'}, 'discussion', 20)

    def test_stale_input_or_unknown_mode_is_rejected(self):
        for auth in (authorization('discussion', '이전 요청'), authorization('invented')):
            with self.assertRaises(ValueError):
                turn.prepare('현재 요청', initial(), [], [], 'scope', '답변', auth)
