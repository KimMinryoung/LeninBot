"""Bargains, prison routine and the optional historical track."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from runtime_tools import roleplay_memory as memory, roleplay_jev as jev, roleplay_turn as turn, roleplay_track as track
from runtime_tools.roleplay_actor import actor_state_view
from runtime_tools.roleplay_dynamics import with_defaults, routine_occurrences
from runtime_tools.roleplay_pacing import policy_for, turn_time_scope
from tool_gateway.security import caller_scope, new_run_context


def initial(**values):
    return with_defaults({**memory.STATE_DEFAULTS, 'hunger': 40, 'fatigue': 40, 'pain': 20, 'tension': 50,
                          'resolve': 20, 'clarity': 55, 'humiliation': 70, 'conditions_initialized': True,
                          'activity': 'light', 'threat': 'threatening', 'participants': ['rodos'],
                          'clock': {'date': '1939-04-28', 'time': '20:00', 'daypart': 'evening', 'certainty': 'explicit'}, **values})


def bargain(request, price, **extra):
    return {'id': 'b1', 'request': request, 'price': price, 'status': 'open', 'paid': False, 'struck_minute': 0, **extra}


def project(text, state, time_scope='auto', **labels):
    verdict = {'status': 'classified', 'labels': {'mode': 'scene', 'elapsed': '0', 'event': 'none', **labels},
               'uncertain': [], 'model': 'jev-test', 'answers': {}}
    with turn_time_scope(policy_for(text, mode=labels.get('mode', 'scene'), time_scope=time_scope)):
        return jev.project(state, text, [{'person_id': 'rodos', 'name': '로도스'}], verdict, 'scope')


class BargainTests(unittest.TestCase):
    def test_paid_kept_and_broken_settle_once_with_fixed_effects(self):
        before = initial(bargains=[bargain('의사의 처치', '골로쇼킨 기입')])
        self.assertEqual(set(jev.build_questions(before, [])['bargain_0']['criteria']), {'keep', 'paid', 'kept', 'broken'})
        paid, applied = project('골로쇼킨을 적었다', before, event='implicating_others', intensity='moderate', bargain_0='paid')
        self.assertEqual(applied['bargains'], {'의사의 처치': 'paid'})
        self.assertTrue(paid['bargains'][0]['paid'])
        self.assertEqual(paid['bargains'][0]['status'], 'open')
        self.assertIn('값은 이미 치렀다', jev.build_questions(paid, [])['bargain_0']['instructions'])
        kept, applied = project('의사가 와서 처치했다', paid, event='treatment', bargain_0='kept')
        self.assertEqual(kept['bargains'][0]['status'], 'kept')
        self.assertEqual(kept['resolve_events'][-1]['kind'], 'bargain_kept')
        self.assertGreater(kept['resolve'], paid['resolve'])
        self.assertEqual(kept['humiliation'], paid['humiliation'] - 3)
        self.assertEqual(kept['tension'], paid['tension'] - 3)
        again, applied_again = project('다시', kept, event='none', bargain_0='kept')
        self.assertNotIn('bargains', applied_again)
        broken, applied = project('처치는 없다고 통보했다', paid, event='none', bargain_0='broken')
        self.assertEqual(broken['bargains'][0]['status'], 'broken')
        self.assertEqual(broken['resolve_events'][-1]['kind'], 'betrayal')
        self.assertLess(broken['resolve'], paid['resolve'])
        self.assertGreater(broken['tension'], paid['tension'])
        self.assertNotIn('bargain_kept', jev.build_questions(before, [])['event_relief']['criteria'])

    def test_actor_records_a_deal_but_cannot_settle_it(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(memory, 'MEMORY_PATH', Path(tmp) / 'm.sqlite3'):
            with memory._connection() as conn:
                conn.execute('INSERT INTO character_state VALUES (?,?)', ('1', json.dumps(initial(revision=1))))
            with caller_scope(new_run_context(interface='telegram', agent_name='roleplay', user_id='1', is_owner=True, scope_type='telegram_message', scope_id='s')):
                result = json.loads(memory.roleplay_state('update', changes={'bargain': {'request': '나탈리아 소식', 'price': '서명'}}, reason='거래 성립', expected_revision=1))
                self.assertEqual(result['bargains']['열린 거래'], ['나탈리아 소식 ← 서명 (값 미지불)'])
                same = json.loads(memory.roleplay_state('update', changes={'bargain': {'request': '나탈리아 소식', 'price': '서명'}}, reason='재확인', expected_revision=2))
                self.assertEqual(len(same['bargains']['열린 거래']), 1)
                with self.assertRaises(ValueError):
                    memory.roleplay_state('update', changes={'bargain': {'request': '', 'price': '서명'}}, reason='빈 요구', expected_revision=3)
                for i in range(2):
                    memory.roleplay_state('update', changes={'bargain': {'request': f'요구{i}', 'price': '값'}}, reason='추가', expected_revision=3 + i)
                with self.assertRaises(ValueError):
                    memory.roleplay_state('update', changes={'bargain': {'request': '넷째', 'price': '값'}}, reason='초과', expected_revision=5)
            stored = memory.load_state('1')
            self.assertEqual([b['status'] for b in stored['bargains']], ['open'] * 3)
            view = actor_state_view(stored)
            self.assertNotIn('struck_minute', json.dumps(view, ensure_ascii=False))


class EventAxesTests(unittest.TestCase):
    """Material events (intake/care/harm) and impact events (pressure/relief) are independent axes."""

    def test_meal_and_kindness_apply_together(self):
        before = initial(hunger=60, resolve=30, humiliation=50)
        state, applied = project('간수가 따뜻한 죽을 건넸고 그걸 먹었다', before, event_intake='meal', event_relief='kindness', intensity='moderate')
        self.assertEqual(applied['events'], ['kindness', 'meal'])
        self.assertEqual(applied['event'], 'kindness')
        self.assertEqual(state['hunger'], 25)
        self.assertEqual(state['resolve'], 33)
        self.assertEqual(state['humiliation'], 47)
        self.assertEqual([e['kind'] for e in state['resolve_events']], ['kindness'])
        self.assertEqual(turn.feedback_line({'status': 'applied', 'applied': applied}, state).split(' · ')[0], '⚙ 확정: 배려·양보 + 식사')
        from runtime_tools.roleplay_actor import actor_outcome_view
        self.assertEqual(actor_outcome_view({'status': 'applied', 'applied': applied})['confirmed_event'], '배려·양보')

    def test_unsettled_family_becomes_the_players_choice_and_a_pick_settles_the_rest(self):
        labels = {'mode': 'scene', 'elapsed': '0', 'intensity': 'moderate', 'event_intake': 'meal', 'event_harm': 'none', 'event_care': 'none', 'event_pressure': 'none'}
        uncertain = ['event_relief']
        answers = {'event_relief': {'choice': 'kindness', 'confidence': .5, 'probabilities': {'kindness': .5, 'none': .3, 'recognition': .2}}}
        jev.settle_event_families(labels, uncertain, answers)
        self.assertEqual(uncertain, ['event_relief', 'event'])
        verdict = {'labels': labels, 'answers': answers}
        with turn_time_scope(policy_for('죽')), self.assertRaises(jev.PendingChoice) as caught:
            jev.project(initial(), '죽', [], verdict, 'scope')
        self.assertEqual([k for k, _ in caught.exception.candidates], ['kindness', 'recognition', 'none'])
        picked = {**labels, 'event': 'kindness'}
        self.assertEqual(jev.resolve_events(picked), (['kindness', 'meal'], False))
        self.assertEqual(jev.resolve_events({**labels, 'event': 'none'}), (['meal'], False))
        # Leaning to none settles silently.
        lean = {'mode': 'scene', 'elapsed': '0'}
        unsure = list(jev.FAMILY_KEYS)
        jev.settle_event_families(lean, unsure, {k: {'choice': 'none', 'confidence': .4, 'probabilities': {'none': .6}} for k in jev.FAMILY_KEYS})
        self.assertEqual((lean['event'], lean['events'], unsure), ('none', [], []))


class RoutineTests(unittest.TestCase):
    def routine(self):
        return memory.set_routine(initial(), [{'time': '06:00', 'title': '아침 배식'}, {'time': '18:00', 'title': '저녁 배식'}, {'time': '06:00', 'title': '아침 배식'}])

    def test_occurrences_and_validation(self):
        state = self.routine()
        self.assertEqual([i['time'] for i in state['routine']], ['06:00', '18:00'])
        self.assertEqual([(at, i['title']) for at, i in routine_occurrences(state, 1440)], [(600, '아침 배식'), (1320, '저녁 배식')])
        self.assertEqual(routine_occurrences(state, 599), [])
        self.assertEqual(routine_occurrences({**state, 'clock': {'time': None}}, 1440), [])
        with self.assertRaises(ValueError):
            memory.set_routine(initial(), [{'time': '6:00', 'title': 'x'}])
        with self.assertRaises(ValueError):
            memory.set_routine(initial(), [{'time': f'{h:02d}:00', 'title': 't'} for h in range(9)])

    def test_explicit_passage_stops_at_routine_and_actor_is_told(self):
        state = self.routine()
        auth = {'user_text': '밤을 넘겨 12시간 쉬어', 'labels': {'mode': 'scene', 'transition': 'current', 'span': 'brief', 'time_scope': 'explicit'}}
        text = '12시간 쉬어'
        auth['user_text'] = text
        self.assertEqual(turn.expected_stop(auth, state), {'title': '06:00 아침 배식', 'minutes': 600})
        self.assertIn('06:00 아침 배식(600분 뒤)에서 멈춘다', turn.direction(auth, state))
        self.assertIsNone(turn.expected_stop({**auth, 'user_text': '감방으로 보내'}, state))
        after, applied = project(text, {**state, 'participants': []}, elapsed='explicit', activity='rest')
        self.assertTrue(applied['interrupted'])
        self.assertEqual(applied['stopped_at'], ['06:00 아침 배식'])
        self.assertEqual(applied['stopped_kinds'], ['routine'])
        self.assertEqual(after['scene_minute'], 600)
        self.assertEqual(after['clock']['time'], '06:00')
        ready = [e for e in after['story_events'] if e['status'] == 'ready']
        self.assertEqual([e['kind'] for e in ready], ['routine'])
        self.assertIn('일과', jev.build_questions(after, [])['story_0']['instructions'])
        line = turn.feedback_line({'status': 'applied', 'applied': applied}, after)
        self.assertEqual(line, '⚙ 확정: 도래 사건까지 진행 · 600분 · 06:00 · 06:00 아침 배식에서 멈춤')
        done, _ = project('간수가 죽을 넣어 줬다', after, event='none', story_0='complete')
        self.assertEqual(done['story_events'][0]['status'], 'completed')
        self.assertEqual(done['hunger'], after['hunger'])  # arrival is not eating

    def test_prepare_accepts_an_announced_stop_only_for_explicit_passages(self):
        state = self.routine()
        state['participants'] = []
        auth = {'user_text': '12시간 쉬어', 'labels': {'mode': 'scene', 'transition': 'current', 'span': 'brief', 'time_scope': 'explicit'}}
        verdict = {'status': 'classified', 'labels': {'mode': 'scene', 'event': 'none', 'activity': 'rest', 'location': 'keep'}, 'uncertain': [], 'answers': {}, 'model': 't', 'draft': '초안'}
        with patch.object(turn, 'review_reply', return_value={'approved': True, 'issues': []}):
            prepared = turn.prepare('12시간 쉬어', state, [], [], '9', '초안', auth, None, verdict)
        self.assertTrue(prepared['applied']['interrupted'])
        self.assertEqual(prepared['state']['scene_minute'], 600)
        brief = {**verdict, 'duration_estimate': {'elapsed_minutes': 5}}
        with_event = jev.apply_story_updates(state, [{'op': 'schedule', 'id': 'knock', 'title': '문 두드림', 'source': '약속', 'due_minute': 2}])
        with patch.object(turn, 'review_reply', return_value={'approved': True, 'issues': []}), self.assertRaises(ValueError):
            turn.prepare('물을 마셔', with_event, [], [], '9', '초안', {**auth, 'user_text': '물을 마셔'}, None, brief)

    def test_routine_survives_reset_and_far_events_do_not_block_next_day(self):
        state = self.routine()
        text = '새 장면. 초기화'
        with turn_time_scope(policy_for(text, mode='reset')):
            fresh, _ = jev.project(state, text, [], {'labels': {'mode': 'reset'}}, 'r')
        self.assertEqual(fresh['routine'], state['routine'])
        self.assertEqual(fresh['bargains'], [])
        far = jev.apply_story_updates(initial(participants=[]), [{'op': 'schedule', 'id': 'far', 'title': '먼 사건', 'source': 's', 'due_minute': 5000}])
        skipped, applied = project('다음 날 아침이 되었다', far, time_scope='day_skip', elapsed='explicit', activity='rest')
        self.assertEqual(skipped['clock']['date'], '1939-04-29')
        near = jev.apply_story_updates(initial(participants=[]), [{'op': 'schedule', 'id': 'near', 'title': '가까운 사건', 'source': 's', 'due_minute': 300}])
        with self.assertRaises(ValueError):
            project('다음 날 아침이 되었다', near, time_scope='day_skip', elapsed='explicit', activity='rest')


class TrackTests(unittest.TestCase):
    def test_enable_schedules_only_future_milestones_and_summary_counts(self):
        state = track.enable(initial(scene_minute=100))
        events = [e for e in state['story_events'] if e['kind'] == 'track']
        self.assertEqual([e['date'] for e in events][:2], ['1939-04-30', '1939-06-10'])
        self.assertEqual(len(events), len(track.MILESTONES))
        self.assertEqual(events[0]['due_minute'], 100 + 2 * 1440 - 20 * 60 + 6 * 60)
        self.assertTrue(state['track']['enabled'])
        summary = track.summary(state)
        self.assertEqual((summary['matched'], summary['departed'], summary['remaining']), (0, 0, 7))
        self.assertEqual(summary['next']['days'], 1)
        with self.assertRaises(ValueError):
            track.enable(initial(clock={'date': None, 'time': None}))
        with self.assertRaises(ValueError):
            track.enable(initial(clock={'date': '1940-03-01', 'time': '08:00'}))
        with self.assertRaises(ValueError):
            track.enable(state)  # already registered
        # The played story diverges from or matches the record through ordinary Jev story labels.
        ready = {**state, 'scene_minute': events[0]['due_minute']}
        from runtime_tools.roleplay_story import refresh_events
        refresh_events(ready)
        index = [i for i, e in enumerate(ready['story_events']) if e['id'] == events[0]['id']][0]
        self.assertIn('실존 연표', jev.build_questions(ready, [])[f'story_{index}']['instructions'])
        diverged, _ = project('66명 대신 침묵을 지켰다', ready, event='none', **{f'story_{index}': 'cancel'})
        self.assertEqual(track.summary(diverged)['departed'], 1)
        released = track.disable(diverged)
        self.assertFalse(released['track']['enabled'])
        self.assertEqual(track.summary(released)['remaining'], 0)
        self.assertEqual(track.summary(released)['departed'], 1)  # a release is not a departure

    def test_far_milestones_do_not_block_next_day_and_bot_commands(self):
        state = track.enable(initial(participants=[]))
        skipped, _ = project('다음 날 아침이 되었다', state, time_scope='day_skip', elapsed='explicit', activity='rest')
        self.assertEqual(skipped['clock']['date'], '1939-04-29')
        from telegram import roleplay_bot as bot
        self.assertIn('다음: 1939-04-30 조서: 음모 가담자 66명 지목 (1일 뒤)', bot._track_display(state))
        self.assertEqual(bot._track_display(initial()), '')


class BotWorldCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_routine_and_track_commands_mutate_one_audited_revision(self):
        from telegram import roleplay_bot as bot
        async def inline_thread(func, *args, **kwargs):
            return func(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(memory, 'MEMORY_PATH', Path(tmp) / 'm.sqlite3'), patch.object(bot.asyncio, 'to_thread', side_effect=inline_thread):
            with memory._connection() as conn:
                conn.execute('INSERT INTO character_state VALUES (?,?)', ('1', json.dumps(initial(revision=4))))
            def msg(text):
                return SimpleNamespace(from_user=SimpleNamespace(id=1), text=text, answer=AsyncMock())
            m = msg('/routine 추가 06:00 아침 배식'); await bot.cmd_routine(m)
            self.assertIn('1. 06:00 아침 배식', m.answer.call_args.args[0])
            self.assertIn('아침 배식 600분 뒤', m.answer.call_args.args[0])
            m = msg('/routine 추가 25:00 x'); await bot.cmd_routine(m)
            self.assertIn('바꾸지 못했어', m.answer.call_args.args[0])
            m = msg('/track 켜기'); await bot.cmd_track(m)
            self.assertIn('실존 궤도: 켜짐', m.answer.call_args.args[0])
            state = memory.load_state('1')
            self.assertEqual(state['revision'], 6)
            self.assertEqual(len(state['routine']), 1)
            m = msg('/routine 삭제 1'); await bot.cmd_routine(m)
            self.assertIn('등록된 일과가 없어', m.answer.call_args.args[0])
            m = msg('/track 끄기'); await bot.cmd_track(m)
            self.assertIn('꺼짐', m.answer.call_args.args[0])
            with memory._connection() as conn:
                actions = [json.loads(r[0])['action'] for r in conn.execute('SELECT payload FROM state_history WHERE user_id=? ORDER BY id', ('1',))]
            self.assertEqual(actions, ['command'] * 4)


if __name__ == '__main__':
    unittest.main()
