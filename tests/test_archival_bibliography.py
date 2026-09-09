import unittest
from runtime_tools.archival_translation import core
class BibliographyTests(unittest.TestCase):
 def test_original_reference_requires_explicit_policy(self):
  b={'tag':'p','lines':['1. Raanan Rein, Untold Stories of the Spanish Civil War (Routledge, 2024).']}
  self.assertTrue(core.validate([(0,b)],{0:b['lines']},core.ENGLISH))
  self.assertEqual(core.validate([(0,{**b,'preserveBibliography':True})],{0:b['lines']},core.ENGLISH),[])
 def test_policy_does_not_allow_missing_markers_or_truncated_content(self):
  b={'tag':'p','lines':['A reference and a long explanation. '*20],'preserveBibliography':True}
  self.assertTrue(core.validate([(0,b)],{},core.ENGLISH))
  self.assertTrue(core.validate([(0,b)],{0:['설명.']},core.ENGLISH))
if __name__=='__main__':unittest.main()
