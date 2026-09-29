import asyncio
import json
import unittest
from contextlib import contextmanager

from commulingo.pipeline import event_links
from commulingo.pipeline.event_links import EventLinker, candidate_events, card_text, parse, screen
from commulingo.pipeline.store import BudgetUnavailable

EVENTS = [
    {'id': 'world-war-i', 'period_label': '1914–1918', 'title_ko': '1차 세계대전', 'title_en': 'World War I'},
    {'id': 'nazi-soviet-pact', 'period_label': '1939.08–1940.08', 'title_ko': '독소 불가침조약',
     'title_en': 'Nazi–Soviet Pact'},
    {'id': 'perestroika', 'period_label': '1985–1991', 'title_ko': '페레스트로이카', 'title_en': 'Perestroika'},
]
PERSON = {'id': 'joachim-von-ribbentrop', 'name_ko': '요아힘 폰 리벤트로프', 'name_en': 'Joachim von Ribbentrop',
          'years_label': '1893–1946', 'birth_year': 1893, 'death_year': 1946,
          'bio_ko': '나치 독일의 외무장관이다. 1939년 8월 모스크바에서 몰로토프와 불가침조약에 서명했다.',
          'bio_en': 'Nazi Germany\'s foreign minister. In August 1939 he signed the non-aggression pact with Molotov in Moscow.'}
LINK = {'event_id': 'nazi-soviet-pact', 'kind': 'leader', 'relation_ko': '조약에 서명한 외무장관',
        'relation_en': 'Foreign minister who signed the pact',
        'note_ko': '1939년 8월 모스크바에서 몰로토프와 조약에 서명했다.',
        'note_en': 'He signed the pact with Molotov in Moscow in August 1939.',
        'basis': '1939년 8월 모스크바에서 몰로토프와 불가침조약에 서명했다.'}


def ok(entry):
    return ''


class PureTests(unittest.TestCase):
    def test_candidate_events_follow_the_adult_life(self):
        self.assertEqual([e['id'] for e in candidate_events(PERSON, EVENTS)], ['world-war-i', 'nazi-soviet-pact'])
        unknown = {'id': 'x', 'years_label': ''}
        self.assertEqual(len(candidate_events(unknown, EVENTS)), 3)
        from_label = {'id': 'y', 'years_label': '1950–2010'}
        late = candidate_events(from_label, EVENTS)
        self.assertEqual([(e['id'], e['before_life']) for e in late],
                         [('world-war-i', True), ('nazi-soviet-pact', True), ('perestroika', False)])
        early = {'id': 'z', 'years_label': '1850–1900'}
        self.assertEqual(candidate_events(early, EVENTS), [])

    def test_an_event_before_the_life_takes_only_a_historian(self):
        historian_card = {**PERSON, 'id': 'h', 'years_label': '1961–', 'birth_year': 1961, 'death_year': None}
        events = candidate_events(historian_card, EVENTS)
        card = card_text(PERSON, [], [])
        kept, rejected = screen([LINK], card, events, ok)
        self.assertEqual((kept, rejected[0]['problem']),
                         ([], "an event before the person's adult life can only take a historian link"))
        kept, rejected = screen([{**LINK, 'kind': 'historian'}], card, events, ok)
        self.assertEqual(([k['event_id'] for k in kept], rejected), (['nazi-soviet-pact'], []))

    def test_screen_requires_a_verbatim_card_quote_and_known_event(self):
        card = card_text(PERSON, [], [])
        spaced = {**LINK, 'basis': '1939년 8월   모스크바에서\n몰로토프와 불가침조약에 서명했다.'}
        invented = {**LINK, 'basis': '리벤트로프는 조약 비밀의정서를 기초했다.'}
        foreign = {**LINK, 'event_id': 'perestroika'}
        kept, rejected = screen([spaced, invented, foreign, dict(LINK)], card, EVENTS[:2], ok)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['basis'], '1939년 8월 모스크바에서 몰로토프와 불가침조약에 서명했다.')
        self.assertEqual([r['problem'] for r in rejected], ['basis is not a quote of the card',
                                                          'unknown or out-of-period event',
                                                          'the same quote is already used for another event'])

    def test_screen_applies_the_writer_checks(self):
        kept, rejected = screen([LINK], card_text(PERSON, [], []), EVENTS, lambda entry: 'em dash in note_ko')
        self.assertEqual((kept, rejected[0]['problem']), ([], 'em dash in note_ko'))

    def test_prompt_ceilings_match_the_writer(self):
        from scripts.commulingo_gap_event_links import acceptable
        at_cap = {**LINK, **{k: '가' * cap for k, cap in event_links.TEXT_CAPS.items()}}
        over = {**at_cap, 'relation_ko': '가' * 65}
        self.assertEqual(acceptable(at_cap), '')
        self.assertNotEqual(acceptable(over), '')

    def test_parse_accepts_fenced_json(self):
        links, reason = parse('```json\n' + json.dumps({'links': [LINK], 'reason': 'r'}) + '\n```')
        self.assertEqual((links[0]['event_id'], reason), ('nazi-soviet-pact', 'r'))
        with self.assertRaises(ValueError):
            parse('no links here')


class FakeLinker(EventLinker):
    def __init__(self, reply, belongs=True, budget=True, kind=None):
        """``belongs``: the check's verdict for every link, or a raw check reply string."""
        self.calls = {'write': [], 'mark': [], 'settle': []}
        store = type('S', (), {})()
        store.reserve = lambda *a, **k: self._reserve(budget)
        store.settle = lambda token, cost: self.calls['settle'].append(cost)

        replies = list(reply) if isinstance(reply, list) else [reply]
        self.requests, self.checks = [], []

        async def generate(system, text, feature):
            if feature == event_links.CHECK_FEATURE:
                self.checks.append(json.loads(text))
                if isinstance(belongs, str):
                    return belongs, 0.001
                return json.dumps({'verdicts': [{'event_id': l['event_id'], 'belongs': belongs,
                                                  'kind': kind or l['proposed_kind'], 'reason': 'r'}
                                                 for l in self.checks[-1]['links']]}), 0.001
            self.requests.append(text)
            if len(self.requests) == 1:
                self.prompt, self.system = json.loads(text), system
            return replies[min(len(self.requests), len(replies)) - 1], 0.004

        super().__init__(store, cap=4, review_fraction=0.3, generate=generate,
                         write=lambda pid, entry: self.calls['write'].append(entry['event_id']),
                         mark=lambda pid, status, reason: self.calls['mark'].append(status),
                         acceptable=ok)

    def _reserve(self, budget):
        if not budget:
            raise BudgetUnavailable('daily budget reserved or spent')
        return 'token'

    def load(self, person_id):
        return PERSON, [], [], EVENTS

    def people(self, limit, person_ids=None):
        return ['joachim-von-ribbentrop']

    def linked(self, person_id):
        return getattr(self, 'existing', [])


class LinkerTests(unittest.TestCase):
    def test_writes_checked_link_and_marks_complete(self):
        linker = FakeLinker(json.dumps({'links': [LINK], 'reason': 'pact'}))
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(linker.calls, {'write': ['nazi-soviet-pact'], 'mark': ['complete'], 'settle': [0.004, 0.001]})
        self.assertEqual(linker.prompt['candidates'], ['world-war-i', 'nazi-soviet-pact'])
        check = linker.checks[0]['links'][0]
        self.assertEqual((check['event_id'], check['quote'], check['event']['title_en']),
                         ('nazi-soviet-pact', LINK['basis'], 'Nazi–Soviet Pact'))

    def test_check_refusal_is_resubmitted_then_queued(self):
        linker = FakeLinker(json.dumps({'links': [LINK]}), belongs=False)
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual((result['status'], result['attempts'], linker.calls['write'], linker.calls['mark']),
                         ('open', 3, [], ['open']))
        self.assertEqual(linker.calls['settle'], [0.004, 0.001] * 3)
        self.assertIn('not part of this event', linker.requests[1])
        self.assertIn('Your previous reply', linker.requests[1])

    def test_a_linked_person_is_asked_only_for_further_events(self):
        linker = FakeLinker(json.dumps({'links': []}))
        linker.existing = [{'event_id': 'world-war-i', 'relation_kind': 'participant', 'title_en': 'World War I'}]
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual(linker.prompt['candidates'], ['nazi-soviet-pact'])
        self.assertEqual(linker.prompt['already_linked'][0]['event_id'], 'world-war-i')
        self.assertEqual((result['status'], linker.calls['mark']), ('complete', ['complete']))
        self.assertTrue(result['reason'].startswith('no further event'))

    def test_a_linked_person_gains_a_further_link(self):
        linker = FakeLinker(json.dumps({'links': [LINK]}))
        linker.existing = [{'event_id': 'world-war-i', 'relation_kind': 'participant', 'title_en': 'World War I'}]
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual((result['status'], linker.calls['write']), ('complete', ['nazi-soviet-pact']))

    def test_the_event_focus_reaches_both_calls(self):
        linker = FakeLinker(json.dumps({'links': [LINK]}))
        focused = [dict(e, focus={'ko': '소련', 'en': 'The Soviet Union'}) if e['id'] == 'nazi-soviet-pact' else e
                   for e in EVENTS]
        linker.load = lambda person_id: (PERSON, [], [], focused)
        asyncio.run(linker.link('joachim-von-ribbentrop'))
        catalogue = json.loads(linker.system.split('Event catalogue:\n', 1)[1])
        by_id = {e['event_id']: e['focus'] for e in catalogue}
        self.assertEqual(by_id['nazi-soviet-pact'], 'The Soviet Union')
        self.assertTrue(by_id['world-war-i'].startswith('none'))
        self.assertEqual(linker.checks[0]['links'][0]['event']['focus'], 'The Soviet Union')

    def test_the_system_prompt_is_the_same_for_every_person(self):
        linker = FakeLinker(json.dumps({'links': []}))
        asyncio.run(linker.link('joachim-von-ribbentrop'))
        first = linker.system
        other = {**PERSON, 'id': 'someone-else', 'years_label': '1950–2010', 'birth_year': 1950, 'death_year': 2010}
        linker.load = lambda person_id: (other, [], [], EVENTS)
        linker.requests = []
        asyncio.run(linker.link('someone-else'))
        self.assertEqual(linker.system, first)
        self.assertEqual(linker.prompt['before_life'], ['world-war-i', 'nazi-soviet-pact'])

    def test_a_kind_the_check_disputes_is_resubmitted(self):
        fixed = {**LINK, 'kind': 'executor'}
        linker = FakeLinker([json.dumps({'links': [LINK]}), json.dumps({'links': [fixed]})], kind='executor')
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual((result['attempts'], linker.calls['write']), (2, ['nazi-soviet-pact']))
        self.assertIn('kind should be executor, not leader', linker.requests[1])

    def test_a_link_without_a_verdict_is_refused(self):
        linker = FakeLinker(json.dumps({'links': [LINK]}), belongs=json.dumps({'verdicts': []}))
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual((result['status'], linker.calls['write']), ('open', []))
        self.assertIn('no verdict', result['rejected'][0]['problem'])

    def test_nothing_proposed_needs_no_check(self):
        linker = FakeLinker(json.dumps({'links': []}))
        asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual((linker.checks, linker.calls['settle']), ([], [0.004]))

    def test_one_quote_cannot_ground_two_events(self):
        reused = {**LINK, 'event_id': 'world-war-i'}
        kept, rejected = screen([LINK, reused], card_text(PERSON, [], []), EVENTS, ok)
        self.assertEqual(([k['event_id'] for k in kept], rejected[0]['problem']),
                         (['nazi-soviet-pact'], 'the same quote is already used for another event'))

    def test_dry_run_neither_writes_nor_marks(self):
        linker = FakeLinker(json.dumps({'links': [LINK]}))
        result = asyncio.run(linker.link('joachim-von-ribbentrop', apply=False))
        self.assertEqual((len(result['links']), linker.calls['write'], linker.calls['mark']), (1, [], []))

    def test_malformed_reply_is_resubmitted_in_the_same_call(self):
        linker = FakeLinker(['I cannot answer', json.dumps({'links': [LINK]})])
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual((result['status'], result['attempts'], linker.calls['write']),
                         ('complete', 2, ['nazi-soviet-pact']))
        self.assertIn('not the requested JSON object', linker.requests[1])

    def test_resubmission_keeps_saved_links_and_fixes_the_rest(self):
        bad = {**LINK, 'event_id': 'world-war-i', 'basis': '카드에 없는 문장이다, 전혀 없다.'}
        fixed = {**bad, 'basis': '나치 독일의 외무장관이다.'}
        linker = FakeLinker([json.dumps({'links': [LINK, bad]}), json.dumps({'links': [LINK, fixed]})])
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual((result['attempts'], linker.calls['write']), (2, ['nazi-soviet-pact', 'world-war-i']))
        self.assertIn('Already saved (do not repeat): nazi-soviet-pact', linker.requests[1])

    def test_only_an_empty_proposal_parks_the_person(self):
        linker = FakeLinker(json.dumps({'links': [], 'reason': 'card names no event'}))
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual((result['status'], linker.calls['mark']), ('not_applicable', ['not_applicable']))

    def test_an_unusable_check_is_an_error_queued_for_retry(self):
        linker = FakeLinker(json.dumps({'links': [LINK]}), belongs='no json')
        results = asyncio.run(linker.run(limit=5))
        self.assertEqual(([r['status'] for r in results], linker.calls['mark']), (['error'], ['open']))

    def test_budget_exhaustion_stops_the_batch(self):
        linker = FakeLinker('{}', budget=False)
        results = asyncio.run(linker.run(limit=5))
        self.assertEqual([r['status'] for r in results], ['budget_wait'])
        self.assertEqual(linker.calls['mark'], [])

    def test_an_event_with_sides_takes_a_side_and_no_opponent(self):
        sided = [dict(e, sides=[{'id': 'germany', 'label': {'ko': '독일', 'en': 'Germany'}},
                                {'id': 'soviet', 'label': {'ko': '소련', 'en': 'Soviet Union'}}])
                 if e['id'] == 'nazi-soviet-pact' else e for e in EVENTS]
        card = card_text(PERSON, [], [])
        kept, rejected = screen([{**LINK, 'side': 'germany'}], card, sided, ok)
        self.assertEqual((kept[0]['side'], rejected), ('germany', []))
        kept, rejected = screen([{**LINK, 'side': 'japan'}], card, sided, ok)
        self.assertIn('side must be one of germany, soviet', rejected[0]['problem'])
        kept, rejected = screen([{**LINK, 'kind': 'opponent', 'side': 'germany'}], card, sided, ok)
        self.assertIn('instead of opponent', rejected[0]['problem'])
        kept, _ = screen([{**LINK, 'side': 'germany'}], card, EVENTS, ok)
        self.assertNotIn('side', kept[0], 'no side is carried for an event without sides')

    def test_sides_reach_both_calls_and_the_check_can_correct_the_side(self):
        sided = [dict(e, sides=[{'id': 'germany', 'label': {'ko': '독일', 'en': 'Germany'}},
                                {'id': 'soviet', 'label': {'ko': '소련', 'en': 'Soviet Union'}}])
                 if e['id'] == 'nazi-soviet-pact' else e for e in EVENTS]
        first, fixed = {**LINK, 'side': 'soviet'}, {**LINK, 'side': 'germany'}
        linker = FakeLinker([json.dumps({'links': [first]}), json.dumps({'links': [fixed]})])
        linker.load = lambda person_id: (PERSON, [], [], sided)
        written = []
        linker.write = lambda pid, entry: written.append((entry['event_id'], entry.get('side')))
        verdict_sides = iter(['germany', 'germany'])

        async def generate(system, text, feature, inner=linker.generate):
            if feature == event_links.CHECK_FEATURE:
                request = json.loads(text)
                linker.checks.append(request)
                side = next(verdict_sides)
                return json.dumps({'verdicts': [{'event_id': l['event_id'], 'belongs': True,
                                                 'kind': l['proposed_kind'], 'side': side, 'reason': 'r'}
                                                for l in request['links']]}), 0.001
            return await inner(system, text, feature)
        linker.generate = generate
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        catalogue = {e['event_id']: e for e in json.loads(linker.system.split('Event catalogue:\n', 1)[1])}
        self.assertEqual([s['id'] for s in catalogue['nazi-soviet-pact']['sides']], ['germany', 'soviet'])
        self.assertNotIn('sides', catalogue['world-war-i'])
        self.assertIn('names its sides', catalogue['nazi-soviet-pact']['focus'])
        self.assertEqual(linker.checks[0]['links'][0]['proposed_side'], 'soviet')
        self.assertIn('side should be germany, not soviet', linker.requests[1])
        self.assertEqual((result['attempts'], written), (2, [('nazi-soviet-pact', 'germany')]))

    def test_kinds_match_the_writer(self):
        from commulingo.people import _HISTORY_RELATION_KINDS
        self.assertEqual(tuple(_HISTORY_RELATION_KINDS), event_links.KINDS)


if __name__ == '__main__':
    unittest.main()
