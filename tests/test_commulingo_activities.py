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
            elif label == 'person-activity-affiliation':
                self.assertEqual(state['selected_activity_function'], 'security')
                self.assertEqual(set(questions), {'activity_affiliation'})
            else:
                self.assertEqual(state['selected_activity_affiliation'], 'china-ccp')
                self.assertEqual(set(questions), {'activity_basis'})
            return DecisionResult(decision=verdict())
        with patch('llm.call_registry.resolve',return_value=PROFILE):
            out=c.classify_person_card({**FIELDS,'evidence':EVIDENCE},catalogs=CATALOGS,decide=decide,codes=False)
        self.assertEqual(stages, ['person-classification', 'person-activity-affiliation', 'person-activity-basis'])
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
