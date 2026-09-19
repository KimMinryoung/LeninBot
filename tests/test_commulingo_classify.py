"""Person classification at registration: the runner assigns group/role from the drafted card."""
import unittest
from unittest.mock import patch

from llm.call_registry import CallSiteProfile, Decision, DecisionResult
from runtime_tools import commulingo_classify as cc

GROUPS = [{'id': 'thaw', 'title_en': 'Thaw', 'range_label': '1953–1985'},
          {'id': 'international-revolutionary', 'title_en': 'Non-Soviet', 'range_label': ''}]
OFFICES = [{'id': 'nationalities-federal', 'title_en': 'Nationalities', 'range_label': '1917–1991'}]
CATS = [{'id': 'socialist-bloc-leader', 'label_en': 'Bloc leader', 'label_ko': '사회주의권 지도자'},
        {'id': 'scholar', 'label_en': 'Scholar', 'label_ko': '연구자'}]
CATALOGS = (GROUPS, OFFICES, CATS)
PROFILE = CallSiteProfile(feature=cc.FEATURE, provider='openrouter', model='typesafe/jev-1.13',
                          extra={'thresholds': {'accept': 0.7}})
FIELDS = {'givenName': {'ko': '야노시', 'en': 'János'}, 'familyName': {'ko': '카다르', 'en': 'Kádár'}, 'years': '1912–1989',
          'epithet': {'ko': '헝가리 지도자', 'en': 'Hungarian leader'}, 'citizenship': {'code': 'hungary', 'label': {}},
          'career': [{'y': '1956–1988', 'r': {'ko': '헝가리 사회주의노동자당 제1서기', 'en': 'First Secretary'}}],
          'bio': {'ko': ['첫 문장.', '둘째 문장.'], 'en': ['First.', 'Second.']}}


def result(group, role, gconf=0.95, rconf=0.9):
    return DecisionResult(decision=Decision(answers={
        'group': {'choice': group, 'confidence': gconf, 'probabilities': {group: gconf}},
        'role': {'choice': role, 'confidence': rconf, 'probabilities': {role: rconf}}}, model='typesafe/jev-test'))


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        p = patch('llm.call_registry.resolve', return_value=PROFILE); p.start(); self.addCleanup(p.stop)

    def test_state_joins_sentences_and_names(self):
        state = cc.state_from_fields(FIELDS)
        self.assertEqual(state['name'], '야노시 카다르')
        self.assertEqual(state['bio_ko'], '첫 문장. 둘째 문장.')
        self.assertEqual(state['career'], ['헝가리 사회주의노동자당 제1서기 (1956–1988)'])
        self.assertEqual(state['citizenship'], 'hungary')

    def test_non_soviet_person_gets_categories_only_and_fill_replaces_writer_values(self):
        seen = {}
        def decide(feature, state, questions, label=None):
            seen.update(feature=feature, roles=set(questions['role']['criteria']))
            return result('international-revolutionary', 'socialist-bloc-leader')
        out = cc.classify_person(FIELDS, catalogs=CATALOGS, decide=decide)
        self.assertEqual(seen['feature'], cc.FEATURE)
        self.assertEqual(seen['roles'], {'socialist-bloc-leader', 'scholar'})
        self.assertEqual(out['groupId'], 'international-revolutionary')
        self.assertEqual(out['role'], {'category': 'socialist-bloc-leader'})
        self.assertFalse(out['low_confidence'])
        filled = cc.fill_classification({**FIELDS, 'groupId': 'thaw', 'role': {'category': 'scholar'}}, out)
        self.assertEqual((filled['groupId'], filled['role']), ('international-revolutionary', {'category': 'socialist-bloc-leader'}))

    def test_soviet_and_successor_citizens_may_receive_an_office(self):
        def decide(feature, state, questions, label=None):
            self.assertIn('nationalities-federal', questions['role']['criteria'])
            self.assertIn('Union-republic first secretaries', questions['role']['criteria']['nationalities-federal'])
            return result('thaw', 'nationalities-federal', rconf=0.6)
        for code in ('soviet', 'azerbaijan'):
            out = cc.classify_person({**FIELDS, 'citizenship': {'code': code}}, catalogs=CATALOGS, decide=decide)
            self.assertEqual(out['role'], {'officeId': 'nationalities-federal'})
            self.assertTrue(out['low_confidence'])

    def test_unavailable_or_unknown_choice_returns_none_and_keeps_writer_values(self):
        def down(feature, state, questions, label=None):
            return DecisionResult(error_kind='transport', error='down')
        self.assertIsNone(cc.classify_person(FIELDS, catalogs=CATALOGS, decide=down))
        def odd(feature, state, questions, label=None):
            return result('no-such-group', 'scholar')
        self.assertIsNone(cc.classify_person(FIELDS, catalogs=CATALOGS, decide=odd))
        writer = {**FIELDS, 'groupId': 'thaw', 'role': {'category': 'scholar'}}
        self.assertEqual(cc.fill_classification(writer, None), writer)

    def test_disabled_entry_makes_no_call(self):
        with patch('llm.call_registry.resolve', return_value=CallSiteProfile(
                feature=cc.FEATURE, provider='openrouter', model='m', extra={'enabled': False})):
            def boom(*a, **k): raise AssertionError('must not be called')
            self.assertIsNone(cc.classify_person(FIELDS, catalogs=CATALOGS, decide=boom))


if __name__ == '__main__':
    unittest.main()


TERM_CATS = [{'id': 'theory', 'label_ko': '이념·이론', 'label_en': 'Ideology and theory'},
             {'id': 'economy', 'label_ko': '경제·계획', 'label_en': 'Economy and planning'}]
TERM_PROFILE = CallSiteProfile(feature=cc.TERM_FEATURE, provider='openrouter', model='typesafe/jev-1.13',
                               extra={'thresholds': {'accept': 0.7}})
TERM = {'term': {'ko': '전시 공산주의', 'en': 'War communism'}, 'period': {'ko': '1918–1921', 'en': '1918–1921'},
        'definition': {'ko': ['내전기 경제 체제.'], 'en': ['Civil-war economic system.']}, 'body': {'ko': '본문', 'en': 'Body'}}


def term_result(category, conf):
    return DecisionResult(decision=Decision(answers={
        'category': {'choice': category, 'confidence': conf, 'probabilities': {category: conf}}}, model='typesafe/jev-test'))


class ClassifyTermTests(unittest.TestCase):
    def setUp(self):
        p = patch('llm.call_registry.resolve', return_value=TERM_PROFILE); p.start(); self.addCleanup(p.stop)

    def test_term_state_and_rules_in_criteria(self):
        state = cc.term_state(TERM)
        self.assertEqual(state['definition']['ko'], '내전기 경제 체제.')
        self.assertEqual(state['period'], '1918–1921')
        criteria = cc.term_questions(TERM_CATS)['category']['criteria']
        self.assertIn('Soviet planning', criteria['economy'])

    def test_confident_category_replaces_writer_value_and_unsure_keeps_it(self):
        def decide(feature, state, questions, label=None):
            self.assertEqual(feature, cc.TERM_FEATURE)
            return term_result('economy', 0.93)
        out = cc.classify_term(TERM, categories=TERM_CATS, decide=decide)
        self.assertEqual((out['category'], out['low_confidence']), ('economy', False))
        self.assertEqual(cc.fill_term_category({**TERM, 'category': 'theory'}, out)['category'], 'economy')
        unsure = cc.classify_term(TERM, categories=TERM_CATS, decide=lambda *a, **k: term_result('economy', 0.4))
        self.assertTrue(unsure['low_confidence'])
        self.assertEqual(cc.fill_term_category({**TERM, 'category': 'theory'}, unsure)['category'], 'theory')
        self.assertEqual(cc.fill_term_category(TERM, unsure)['category'], 'economy')

    def test_unavailable_or_unknown_category_is_none(self):
        self.assertIsNone(cc.classify_term(TERM, categories=TERM_CATS,
                                           decide=lambda *a, **k: DecisionResult(error_kind='server', error='503')))
        self.assertIsNone(cc.classify_term(TERM, categories=TERM_CATS, decide=lambda *a, **k: term_result('nope', 0.9)))
        self.assertEqual(cc.fill_term_category({**TERM, 'category': 'theory'}, None)['category'], 'theory')
