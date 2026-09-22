import unittest
from unittest.mock import patch
from llm.call_registry import Decision, DecisionResult
from runtime_tools.commulingo_activities import activity_questions, activity_person_from, load_catalog
from runtime_tools import commulingo_classify as c
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
        def decide(feature,state,questions,label=None):
            self.assertIn('activity_function',questions)
            self.assertFalse(any(k.startswith('role') for k in questions))
            return DecisionResult(decision=verdict())
        with patch('llm.call_registry.resolve',return_value=PROFILE):
            out=c.classify_person_card({**FIELDS,'evidence':EVIDENCE},catalogs=CATALOGS,decide=decide,codes=False)
        person=out['person'];filled=c.fill_classification(FIELDS,person)
        self.assertEqual(filled['activities'][0]['affiliationId'],'china-ccp')
        self.assertEqual(filled['activities'][0]['evidence'][0]['source'],EVIDENCE[0]['source'])
        self.assertIsNone(filled['activities'][0]['startYear'])
