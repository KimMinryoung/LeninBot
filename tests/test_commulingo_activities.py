import unittest
from unittest.mock import patch
from llm.call_registry import Decision, DecisionResult
from commulingo.activities import activity_questions, activity_person_from, load_catalog
from commulingo import classify as c
from tests.test_commulingo_classify import PROFILE, CATALOGS, FIELDS

EVIDENCE=[{'field':'career','source':'https://example.org/kang','locator':'Career','claim':'Party intelligence work','excerpt':'Kang Sheng directed the party intelligence apparatus.'}]

def verdict(**overrides):
    choices={'group':'international-revolutionary','activity_function':'security','activity_affiliation':'china-ccp','activity_basis':'0'}|overrides
    return Decision(answers={k:{'choice':v,'confidence':.98,'probabilities':{v:.98}} for k,v in choices.items()},model='test')

class ActivitiesTests(unittest.TestCase):
    def test_missing_source_evidence_makes_no_legacy_guess(self):
        with patch('llm.call_registry.resolve',return_value=PROFILE):
            self.assertIsNone(c.classify_person_card(FIELDS,catalogs=CATALOGS,
                decide=lambda *a,**kw:self.fail('no request without cited evidence')))

    def test_scopes_share_one_catalog_and_do_not_use_citizenship(self):
        q=activity_questions(load_catalog(),EVIDENCE)
        self.assertIn('china-ccp',q['activity_affiliation']['criteria'])
        self.assertIn('comintern',q['activity_affiliation']['criteria'])
        self.assertIn('unresolved',q['activity_affiliation']['criteria'])
        self.assertNotIn('role_china',q)

    def test_party_state_organs_are_offered_only_as_the_ruling_party(self):
        q=activity_questions(load_catalog(),EVIDENCE)['activity_affiliation']
        for retired in ('china-prc','state-soviet','state-east-germany','state-north-korea'):
            self.assertNotIn(retired,q['criteria'])
        self.assertIn('Party-state rule: 1949 onward',q['criteria']['china-ccp'])
        self.assertIn('party-polish-pzpr',q['criteria']['state-poland'])
        self.assertIn('party-state is one affiliation',q['instructions'])

    def test_unknown_affiliation_stays_unknown_and_unsupported_is_rejected(self):
        out=activity_person_from(verdict(activity_affiliation='unresolved'),load_catalog(),EVIDENCE,{'international-revolutionary'})
        self.assertIsNone(out['activities'][0]['affiliationId'])
        self.assertEqual(out['activities'][0]['affiliationStatus'],'unresolved')
        for overrides in ({'activity_basis':'unsupported'},{'activity_basis':'19'},{'activity_affiliation':'not-in-catalog'}):
            self.assertIsNone(activity_person_from(verdict(**overrides),load_catalog(),EVIDENCE,{'international-revolutionary'}))

    def test_evidence_driven_card_uses_activity_questions_and_preserves_citation(self):
        stages = []
        def decide(feature,state,questions,label=None):
            stages.append(label)
            self.assertFalse(any(k.startswith('role') for k in questions))
            self.assertEqual(state['cited_activity_evidence'][0]['source'], EVIDENCE[0]['source'])
            if label == 'person-classification':
                self.assertIn('activity_function', questions)
                self.assertNotIn('activity_affiliation', questions)
                self.assertNotIn('activity_basis', questions)
            elif label == 'person-activity-basis':
                self.assertEqual(state['selected_activity_function'], 'security')
                self.assertEqual(set(questions), {'activity_basis'})
            else:
                self.assertEqual(state['selected_activity_evidence']['source'], EVIDENCE[0]['source'])
                self.assertEqual(set(questions), {'activity_affiliation'})
            return DecisionResult(decision=verdict())
        with patch('llm.call_registry.resolve',return_value=PROFILE):
            out=c.classify_person_card({**FIELDS,'evidence':EVIDENCE},catalogs=CATALOGS,decide=decide,codes=False)
        self.assertEqual(stages, ['person-classification', 'person-activity-basis', 'person-activity-affiliation'])
        person=out['person'];filled=c.fill_classification(FIELDS,person)
        self.assertEqual(filled['activities'][0]['affiliationId'],'china-ccp')
        self.assertEqual(filled['activities'][0]['evidence'][0]['source'],EVIDENCE[0]['source'])
        self.assertIsNone(filled['activities'][0]['startYear'])

    def test_search_filters_expand_children_and_reject_unknown_ids(self):
        from commulingo.activities import activity_search_params
        params = activity_search_params('military', 'china-ccp')
        self.assertIn('china-ccp', params['descendants'])
        self.assertIn('china-pla', params['descendants'])
        self.assertNotIn('china-kmt', params['descendants'])
        for function, affiliation in [('unknown', ''), ('', 'unknown')]:
            with self.assertRaises(ValueError):
                activity_search_params(function, affiliation)

    def test_unavailable_or_unsupported_grounding_never_returns_an_activity(self):
        for mode in ('unavailable', 'unsupported'):
            def decide(feature, state, questions, label=None):
                if label == 'person-activity-basis':
                    return DecisionResult(error='offline') if mode == 'unavailable' else DecisionResult(decision=verdict(activity_basis='unsupported'))
                return DecisionResult(decision=verdict())
            with patch('llm.call_registry.resolve', return_value=PROFILE):
                out=c.classify_person_card({**FIELDS,'evidence':EVIDENCE},catalogs=CATALOGS,decide=decide,codes=False)
            self.assertTrue(out is None or out['person'] is None)


class AffiliationPeriodTests(unittest.TestCase):
    """The chosen excerpt's years decide which organizations existed and are offered."""

    def test_excerpt_years_ignore_footnotes_and_dates_outside_the_life(self):
        from commulingo.activities import excerpt_years
        text = 'On July 25, 1830, the king signed the Ordinances.[193] He died in 1834; a 1989 study.'
        self.assertEqual(excerpt_years(text, (1773, 1834)), [1830, 1834])
        self.assertEqual(excerpt_years(text), [1830, 1834, 1989])

    def test_offered_affiliations_follow_the_window(self):
        catalog = load_catalog()
        in_1830 = set(activity_questions(catalog, EVIDENCE, (1830, 1830))['activity_affiliation']['criteria'])
        self.assertNotIn('french-first-republic', in_1830)
        self.assertNotIn('french-jacobins', in_1830)
        self.assertIn('state-france', in_1830)
        self.assertIn('state-usa', in_1830)          # no periods: always offered
        self.assertIn('unresolved', in_1830)
        in_1794 = set(activity_questions(catalog, EVIDENCE, (1794, 1794))['activity_affiliation']['criteria'])
        self.assertIn('french-first-republic', in_1794)
        self.assertNotIn('state-france', in_1794)
        self.assertNotIn('french-monarchy', in_1794)

    def test_card_classifier_filters_affiliations_by_the_chosen_excerpt(self):
        evidence = [{'field':'career','source':'https://example.org/laf','locator':'1830','claim':'July Revolution',
                     'excerpt':'In July 1830 Lafayette went to the barricades and was made head of the National Guard.'}]
        offered = {}
        def decide(feature, state, questions, label=None):
            if label == 'person-activity-affiliation':
                offered.update(questions['activity_affiliation']['criteria'])
                self.assertEqual(state['selected_activity_years'], [1830, 1830])
            return DecisionResult(decision=verdict(activity_function='military', activity_affiliation='state-france'))
        with patch('llm.call_registry.resolve', return_value=PROFILE):
            c.classify_person_card({**FIELDS, 'years': '1757–1834', 'evidence': evidence}, catalogs=CATALOGS, decide=decide, codes=False)
        self.assertIn('state-france', offered)
        self.assertNotIn('french-first-republic', offered)


class FrenchRevolutionBasisTests(unittest.TestCase):
    """Only French Revolution figures are asked to prefer their role in the Revolution."""

    def basis_instructions(self, group):
        seen = {}
        def decide(feature, state, questions, label=None):
            if label == 'person-activity-basis':
                seen['text'] = questions['activity_basis']['instructions']
            return DecisionResult(decision=verdict(group=group))
        with patch('llm.call_registry.resolve', return_value=PROFILE):
            c.classify_person_card({**FIELDS, 'evidence': EVIDENCE}, catalogs=CATALOGS, decide=decide, codes=False)
        return seen['text']

    def test_french_revolution_group_prefers_the_revolutionary_role(self):
        self.assertIn('role in the Revolution', self.basis_instructions('france-revolution'))

    def test_other_groups_keep_the_plain_rule(self):
        self.assertNotIn('role in the Revolution', self.basis_instructions('international-revolutionary'))
