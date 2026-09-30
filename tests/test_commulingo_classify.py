"""Person classification at registration: the runner assigns group and primary activity from the drafted card."""
import unittest
from unittest.mock import patch

from llm.call_registry import CallSiteProfile, Decision, DecisionResult
from commulingo import classify as cc

GROUPS = [{'id': 'thaw', 'title_en': 'Thaw', 'range_label': '1953–1985'},
          {'id': 'international-revolutionary', 'title_en': 'Non-Soviet', 'range_label': ''}]
OFFICES = [{'id': 'nationalities-federal', 'title_en': 'Nationalities', 'range_label': '1917–1991'}]
CATALOGS = (GROUPS, OFFICES)
PROFILE = CallSiteProfile(feature=cc.FEATURE, provider='openrouter', model='typesafe/jev-1.13',
                          extra={'thresholds': {'accept': 0.7}})
FIELDS = {'givenName': {'ko': '야노시', 'en': 'János'}, 'familyName': {'ko': '카다르', 'en': 'Kádár'}, 'years': '1912–1989',
          'epithet': {'ko': '헝가리 지도자', 'en': 'Hungarian leader'}, 'citizenship': {'code': 'hungary', 'label': {}},
          'career': [{'y': '1956–1988', 'r': {'ko': '헝가리 사회주의노동자당 제1서기', 'en': 'First Secretary'}}],
          'bio': {'ko': ['첫 문장.', '둘째 문장.'], 'en': ['First.', 'Second.']}}
EVIDENCE = [{'field': 'career', 'source': 'https://example.org/kadar', 'locator': 'Career',
             'claim': 'First Secretary of the Hungarian party 1956–1988',
             'excerpt': 'Kádár led the Hungarian Socialist Workers\' Party from 1956 to 1988.'}]


def answers(**choices):
    return DecisionResult(decision=Decision(answers={
        k: {'choice': c, 'confidence': p, 'probabilities': {c: p}} for k, (c, p) in choices.items()}, model='typesafe/jev-test'))


def activity_decide(seen=None, **card):
    """Answers the three dependent activity requests; ``card`` adds the first request's answers."""
    def decide(feature, state, questions, label=None):
        if seen is not None:
            seen.setdefault('labels', []).append(label)
            seen.setdefault('questions', {})[label] = questions
            seen.setdefault('states', {})[label] = dict(state)
        if label == 'person-activity-basis':
            return answers(activity_basis=('0', 0.95))
        if label == 'person-activity-affiliation':
            return answers(activity_affiliation=('unresolved', 0.9))
        return answers(group=('international-revolutionary', 0.95), activity_function=('political-leadership', 0.92), **card)
    return decide


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        p = patch('llm.call_registry.resolve', return_value=PROFILE); p.start(); self.addCleanup(p.stop)

    def test_state_joins_sentences_and_names(self):
        state = cc.state_from_fields(FIELDS)
        self.assertEqual(state['name'], '야노시 카다르')
        self.assertEqual(state['bio_ko'], '첫 문장. 둘째 문장.')
        self.assertEqual(state['career'], ['헝가리 사회주의노동자당 제1서기 (1956–1988)'])
        self.assertEqual(state['citizenship'], 'hungary')
        # Labels only (the writer's schema has no codes): the state still names the nation and carries moment/fate.
        labels = cc.state_from_fields({**FIELDS, 'citizenship': {'label': {'ko': '헝가리', 'en': 'Hungary'}},
                                       'nationalOrigin': {'label': {'ko': '헝가리인', 'en': 'Hungarian'}},
                                       'moment': {'ko': ['1956년 소련군 진주 뒤 집권.'], 'en': ['Took power in 1956.']},
                                       'fate': {'label': {'ko': '자연사', 'en': 'Natural causes'}}})
        self.assertEqual((labels['citizenship'], labels['national_origin']), ('Hungary', 'Hungarian'))
        self.assertEqual(labels['moment_ko'], '1956년 소련군 진주 뒤 집권.')
        self.assertEqual(labels['fate'], ' · 자연사')

    def test_group_and_activity_replace_writer_values_and_no_role_is_asked_or_filled(self):
        seen = {}
        out = cc.classify_person({**FIELDS, 'evidence': EVIDENCE}, catalogs=CATALOGS, decide=activity_decide(seen))
        first = seen['questions']['person-classification']
        self.assertEqual(set(first), {'group', 'activity_function'})
        self.assertEqual(out['groupId'], 'international-revolutionary')
        self.assertNotIn('role', out)
        self.assertEqual(out['activities'][0]['functionId'], 'political-leadership')
        self.assertTrue(out['activities'][0]['primary'])
        self.assertFalse(out['low_confidence'])
        filled = cc.fill_classification({**FIELDS, 'groupId': 'thaw', 'role': {'category': 'scholar'}}, out)
        self.assertEqual(filled['groupId'], 'international-revolutionary')
        self.assertNotIn('role', filled)
        self.assertEqual(filled['activities'], out['activities'])

    def test_research_excerpts_for_bio_and_career_join_the_state(self):
        seen = {}
        claims = {'career': [{'claim': 'First Secretary 1956–1988', 'excerpt': 'x' * 2000}],
                  'bio': [{'claim': 'b', 'excerpt': 'short'}], 'citizenship': [{'claim': 'c', 'excerpt': 'ignored here'}]}
        cc.classify_person({**FIELDS, 'evidence': EVIDENCE}, catalogs=CATALOGS, claims=claims, decide=activity_decide(seen))
        ex = seen['states']['person-classification']['research_excerpts']
        self.assertEqual([e['field'] for e in ex], ['bio', 'career'])
        self.assertEqual(len(ex[1]['excerpt']), cc.EVIDENCE_CHARS)
        seen.clear()
        cc.classify_person({**FIELDS, 'evidence': EVIDENCE}, catalogs=CATALOGS, decide=activity_decide(seen))
        self.assertNotIn('research_excerpts', seen['states']['person-classification'])

    def test_unavailable_or_unknown_choice_returns_none_and_keeps_writer_values(self):
        def down(feature, state, questions, label=None):
            return DecisionResult(error_kind='transport', error='down')
        self.assertIsNone(cc.classify_person({**FIELDS, 'evidence': EVIDENCE}, catalogs=CATALOGS, decide=down))
        def odd(feature, state, questions, label=None):
            if label == 'person-activity-basis':
                return answers(activity_basis=('0', 0.95))
            if label == 'person-activity-affiliation':
                return answers(activity_affiliation=('unresolved', 0.9))
            return answers(group=('no-such-group', 0.9), activity_function=('political-leadership', 0.9))
        self.assertIsNone(cc.classify_person({**FIELDS, 'evidence': EVIDENCE}, catalogs=CATALOGS, decide=odd))
        writer = {**FIELDS, 'groupId': 'thaw'}
        self.assertEqual(cc.fill_classification(writer, None), writer)

    def test_disabled_entry_makes_no_call(self):
        with patch('llm.call_registry.resolve', return_value=CallSiteProfile(
                feature=cc.FEATURE, provider='openrouter', model='m', extra={'enabled': False})):
            def boom(*a, **k): raise AssertionError('must not be called')
            self.assertIsNone(cc.classify_person({**FIELDS, 'evidence': EVIDENCE}, catalogs=CATALOGS, decide=boom))


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
        state = cc.term_state({**TERM, 'aliases': {'ko': ['군사공산주의'], 'en': ['military communism']}, 'parentId': 'nep'})
        self.assertEqual(state['definition']['ko'], '내전기 경제 체제.')
        self.assertEqual(state['aliases'], ['군사공산주의', 'military communism'])
        self.assertEqual(state['parent_term'], 'nep')
        self.assertEqual(state['period'], '1918–1921')
        criteria = cc.term_questions(TERM_CATS)['category']['criteria']
        self.assertIn('Soviet planning', criteria['economy'])

    def test_category_always_comes_from_the_classifier(self):
        def decide(feature, state, questions, label=None):
            self.assertEqual(feature, cc.TERM_FEATURE)
            return term_result('economy', 0.93)
        out = cc.classify_term(TERM, categories=TERM_CATS, decide=decide)
        self.assertEqual((out['category'], out['low_confidence']), ('economy', False))
        self.assertEqual(cc.fill_term_category({**TERM, 'category': 'theory'}, out)['category'], 'economy')
        unsure = cc.classify_term(TERM, categories=TERM_CATS, decide=lambda *a, **k: term_result('economy', 0.4))
        self.assertTrue(unsure['low_confidence'])
        self.assertEqual(cc.fill_term_category(TERM, unsure)['category'], 'economy')

    def test_unavailable_or_unknown_category_is_none(self):
        self.assertIsNone(cc.classify_term(TERM, categories=TERM_CATS,
                                           decide=lambda *a, **k: DecisionResult(error_kind='server', error='503')))
        self.assertIsNone(cc.classify_term(TERM, categories=TERM_CATS, decide=lambda *a, **k: term_result('nope', 0.9)))
        self.assertEqual(cc.fill_term_category({**TERM, 'category': 'theory'}, None)['category'], 'theory')


CODES_PROFILE = CallSiteProfile(feature=cc.CODES_FEATURE, provider='openrouter', model='typesafe/jev-1.13',
                                extra={'thresholds': {'accept': 0.7}})
CARD = {'name': {'ko': '류시코프', 'en': 'Lyushkov'}, 'years': '1900–1945',
        'citizenship': {'label': {'ko': '소련', 'en': 'Soviet Union'}},
        'fate': {'label': {'ko': '다롄에서 사망, 정황 미확인', 'en': 'Dalian, 1945; circumstances unconfirmed'}}}


def codes_result(**answers):
    return DecisionResult(decision=Decision(answers={
        k: {'choice': c, 'confidence': p, 'probabilities': {c: p}} for k, (c, p) in answers.items()}, model='typesafe/jev-test'))


class ClassifyCardTests(unittest.TestCase):
    def setUp(self):
        p = patch('llm.call_registry.resolve', return_value=PROFILE); p.start(); self.addCleanup(p.stop)

    def test_one_card_request_asks_codes_group_and_activity_function(self):
        card = {k: v for k, v in FIELDS.items() if k != 'citizenship'}
        card.update(citizenship={'label': {'ko': '소련', 'en': 'Soviet'}}, fate={'label': {'ko': '자연사', 'en': 'Natural'}},
                    evidence=EVIDENCE)
        seen = {}
        decide = activity_decide(seen, citizenship=('soviet', 0.98), fate=('natural', 0.9))
        out = cc.classify_person_card(card, catalogs=CATALOGS, claims={'fate': [{'claim': 'died at home', 'excerpt': 'умер'}]},
                                      decide=decide)
        self.assertEqual(seen['labels'], ['person-card', 'person-activity-basis', 'person-activity-affiliation'])
        self.assertEqual(set(seen['questions']['person-card']), {'citizenship', 'fate', 'group', 'activity_function'})
        state = seen['states']['person-card']
        self.assertEqual(state['fate_claims'][0]['excerpt'], 'умер')
        self.assertIn('bio_ko', state); self.assertIn('career', state)
        self.assertEqual(out['codes']['citizenship']['code'], 'soviet')
        self.assertEqual(out['codes']['fate']['kind'], 'natural')
        self.assertEqual(out['person']['groupId'], 'international-revolutionary')
        self.assertNotIn('role', out['person'])
        self.assertIsNone(cc.classify_person_card(card, catalogs=CATALOGS,
                                                  decide=lambda *a, **k: DecisionResult(error_kind='server', error='503')))


class ClassifyCodesTests(unittest.TestCase):
    def setUp(self):
        p = patch('llm.call_registry.resolve', return_value=CODES_PROFILE); p.start(); self.addCleanup(p.stop)

    def test_labels_and_claims_feed_one_request_and_fill_both_codes(self):
        seen = {}
        def decide(feature, state, questions, label=None):
            seen.update(feature=feature, state=state, questions=set(questions))
            return codes_result(citizenship=('soviet', 0.99), fate=('murdered', 0.86))
        claims = {'fate': [{'claim': 'Killed by Soviet fire in Dalian', 'excerpt': 'погиб в Дайрэне'}]}
        codes = cc.classify_person_codes(CARD, claims=claims, decide=decide)
        self.assertEqual(seen['feature'], cc.CODES_FEATURE)
        self.assertEqual(seen['questions'], {'citizenship', 'fate'})
        self.assertEqual(seen['state']['fate_claims'][0]['excerpt'], 'погиб в Дайрэне')
        self.assertIn('bio_ko', seen['state'])
        filled = cc.fill_person_codes(CARD, codes)
        self.assertEqual(filled['citizenship']['code'], 'soviet')
        self.assertEqual(filled['fate']['kind'], 'murdered')
        self.assertEqual(cc.missing_person_codes(filled), [])
        self.assertEqual(cc.missing_person_codes(CARD), ['citizenship.code', 'fate.kind'])
        self.assertEqual(seen['questions'], {'citizenship', 'fate'})  # no nationalOrigin object on this card

    def test_unconfirmed_maps_to_empty_kind_and_low_confidence_is_reported_not_deferred(self):
        def decide(feature, state, questions, label=None):
            return codes_result(citizenship=('russia', 0.4), fate=('unconfirmed', 0.9))
        codes = cc.classify_person_codes(CARD, decide=decide)
        self.assertEqual(codes['fate']['kind'], '')
        self.assertTrue(codes['citizenship']['low_confidence'])
        filled = cc.fill_person_codes(CARD, codes)
        self.assertEqual((filled['citizenship']['code'], filled['fate']['kind']), ('russia', ''))

    def test_living_person_gets_empty_fate_without_asking_about_it(self):
        def decide(feature, state, questions, label=None):
            self.assertEqual(set(questions), {'citizenship'})
            return codes_result(citizenship=('hungary', 0.95))
        codes = cc.classify_person_codes({**CARD, 'years': '1963–'}, decide=decide)
        self.assertEqual(codes['fate'], {'kind': '', 'confidence': 1.0, 'low_confidence': False})
        self.assertEqual(codes['citizenship']['code'], 'hungary')

    def test_unavailable_model_leaves_codes_missing(self):
        codes = cc.classify_person_codes(CARD, decide=lambda *a, **k: DecisionResult(error_kind='server', error='503'))
        self.assertIsNone(codes)
        self.assertEqual(cc.missing_person_codes(cc.fill_person_codes(CARD, codes)), ['citizenship.code', 'fate.kind'])

    def test_create_tool_schema_no_longer_requires_the_codes(self):
        from commulingo.people import COMMULINGO_PERSON_CREATE_TOOL, _NATIONALITY_SCHEMA
        props = COMMULINGO_PERSON_CREATE_TOOL['input_schema']['properties']['fields']['properties']
        for key in ('citizenship', 'nationalOrigin', 'fate'):
            self.assertEqual(props[key]['required'], ['label'])
            self.assertEqual(set(props[key]['properties']), {'label'})
        self.assertNotIn('groupId', props); self.assertNotIn('role', props)
        self.assertEqual(_NATIONALITY_SCHEMA['required'], ['code', 'label'])  # shared object untouched

    def test_origin_code_follows_the_background_rules(self):
        seen = {}
        def decide(feature, state, questions, label=None):
            seen.update(state=state, instr=questions['nationalOrigin']['instructions'])
            return codes_result(nationalOrigin=('poland', 0.96), citizenship=('soviet', 0.99), fate=('natural', 0.9))
        card = {**CARD, 'nationalOrigin': {'label': {'ko': '폴란드계 유대인', 'en': 'Polish-Jewish'}}}
        codes = cc.classify_person_codes(card, claims={'nationalOrigin': [{'claim': 'born to a Jewish family in Lwów', 'excerpt': 'x'}]}, decide=decide)
        self.assertIn('NEVER israel', seen['instr'])
        self.assertEqual(seen['state']['origin_claims'][0]['claim'], 'born to a Jewish family in Lwów')
        self.assertEqual(cc.fill_person_codes(card, codes)['nationalOrigin']['code'], 'poland')
        self.assertIn('nationalOrigin.code', cc.missing_person_codes(card))


class ReviewRiskTests(unittest.TestCase):
    def test_unsure_classification_still_lands_and_is_left_to_the_reviewer(self):
        activity = {'functionId': 'scholarship', 'affiliationId': None, 'affiliationStatus': 'unresolved',
                    'relation': 'unresolved', 'primary': True, 'evidence': EVIDENCE}
        unsure = {'groupId': 'thaw', 'activities': [activity], 'confidence': {'group': 0.5}, 'low_confidence': True}
        filled = cc.fill_classification({**FIELDS, 'groupId': 'stale'}, unsure)
        self.assertEqual((filled['groupId'], filled['activities']), ('thaw', [activity]))


class GroupEraTests(unittest.TestCase):
    """Stage one of the group decision: life years rule eras out before any model call."""
    GROUPS = [{"id": g} for g in ("old-regime", "bolshevik", "stalin-era", "thaw", "perestroika", "china-reform",
                                   "france-revolution", "international-revolutionary",
                                   "foreign-statesmen", "international-counterrevolutionary", "scholar", "unlisted")]

    def ids(self, years):
        from commulingo.classify import groups_for_years
        return {g["id"] for g in groups_for_years(self.GROUPS, years, today=2026)}

    def test_revolutionary_of_1794_sees_only_french_eras(self):
        self.assertEqual(self.ids("1758–1794"), {"old-regime", "france-revolution", "unlisted"})

    def test_modern_revolutionary_never_sees_the_french_eras(self):
        ids = self.ids("1842–1911")
        self.assertNotIn("france-revolution", ids)
        self.assertIn("international-revolutionary", ids)

    def test_bridging_lives_keep_both_sides_for_the_model(self):
        ids = self.ids("1788–1856")
        self.assertIn("france-revolution", ids)
        self.assertIn("international-revolutionary", ids)

    def test_living_and_open_labels(self):
        from commulingo.classify import active_span
        self.assertEqual(active_span("1950–", today=2026), (1966, 2026))
        self.assertEqual(active_span("1900–1950 이후", today=2026), (1916, 2026))
        self.assertEqual(active_span("c. 1729/1730–1800", today=2026), (1745, 1800))
        self.assertEqual(active_span("?–1794", today=2026), (1744, 1794))
        self.assertEqual(active_span("1900–?", today=2026), (1916, 1990))
        self.assertIsNone(active_span("", today=2026))
        self.assertEqual(len(self.ids("")), len(self.GROUPS))

    def test_card_classifier_offers_only_the_era_groups(self):
        from commulingo.classify import person_card_questions, groups_for_years
        groups = [{"id": g["id"], "title_en": g["id"], "range_label": ""} for g in self.GROUPS]
        kept = groups_for_years(groups, "1758–1794", today=2026)
        q = person_card_questions({"years": "1758–1794", "citizenship": {"code": "france"}}, kept, ["france"], [], codes=False)
        self.assertNotIn("international-revolutionary", q["group"]["criteria"])
