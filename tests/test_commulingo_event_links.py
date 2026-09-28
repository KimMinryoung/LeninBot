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
        self.assertEqual([e['id'] for e in candidate_events(from_label, EVENTS)], ['perestroika'])

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
    def __init__(self, reply, check=None, budget=True):
        self.calls = {'write': [], 'mark': [], 'settle': []}
        store = type('S', (), {})()
        store.reserve = lambda *a, **k: self._reserve(budget)
        store.settle = lambda token, cost: self.calls['settle'].append(cost)

        replies = list(reply) if isinstance(reply, list) else [reply]
        self.requests = []

        async def generate(system, text):
            self.requests.append(text)
            if len(self.requests) == 1:
                self.prompt = json.loads(text)
            return replies[min(len(self.requests), len(replies)) - 1], 0.004

        async def gate(claim, quote, person_id):
            return check or {'support': 'supports', 'confidence': 0.97}

        super().__init__(store, cap=4, review_fraction=0.3, generate=generate, gate=gate,
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


class LinkerTests(unittest.TestCase):
    def test_writes_gated_link_and_marks_complete(self):
        linker = FakeLinker(json.dumps({'links': [LINK], 'reason': 'pact'}))
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(linker.calls, {'write': ['nazi-soviet-pact'], 'mark': ['complete'], 'settle': [0.004]})
        self.assertEqual([e['event_id'] for e in linker.prompt['events']], ['world-war-i', 'nazi-soviet-pact'])

    def test_gate_rejection_is_resubmitted_then_queued(self):
        linker = FakeLinker(json.dumps({'links': [LINK]}), check={'support': 'unrelated', 'confidence': 0.6})
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual((result['status'], result['attempts'], linker.calls['write'], linker.calls['mark']),
                         ('open', 3, [], ['open']))
        self.assertEqual(linker.calls['settle'], [0.004] * 3)
        self.assertIn('does not support this link', linker.requests[1])
        self.assertIn('Your previous reply', linker.requests[1])

    def test_partial_support_is_not_enough(self):
        linker = FakeLinker(json.dumps({'links': [LINK]}), check={'support': 'partially_supports', 'confidence': 0.9})
        result = asyncio.run(linker.link('joachim-von-ribbentrop'))
        self.assertEqual((result['status'], linker.calls['write']), ('open', []))

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

    def test_errors_queue_a_retry(self):
        linker = FakeLinker(json.dumps({'links': [LINK]}), check=None)
        async def unavailable(*args):
            return None
        linker.gate = unavailable
        results = asyncio.run(linker.run(limit=5))
        self.assertEqual(([r['status'] for r in results], linker.calls['mark']), (['error'], ['open']))

    def test_budget_exhaustion_stops_the_batch(self):
        linker = FakeLinker('{}', budget=False)
        results = asyncio.run(linker.run(limit=5))
        self.assertEqual([r['status'] for r in results], ['budget_wait'])
        self.assertEqual(linker.calls['mark'], [])

    def test_kinds_match_the_writer(self):
        from commulingo.people import _HISTORY_RELATION_KINDS
        self.assertEqual(tuple(_HISTORY_RELATION_KINDS), event_links.KINDS)


if __name__ == '__main__':
    unittest.main()
