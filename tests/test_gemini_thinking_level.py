import sys,unittest
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
from llm import call_registry as cr
from runtime_tools.archival_translation import core
class ThinkingTests(unittest.TestCase):
 def profile(self,extra):return cr.CallSiteProfile(feature='t',provider='gemini',model='gemini-3.1-pro-preview',extra=extra)
 def test_actual_sdk_request_has_low_and_default_is_unchanged(self):
  for extra,want in [({'thinking_level':'low'},'LOW'),({},None)]:
   client=MagicMock();client.models.generate_content.return_value=SimpleNamespace(text='번역문',usage_metadata=None,candidates=[])
   with patch('google.genai.Client',return_value=client),patch.object(cr,'resolve_provider_connection',return_value=SimpleNamespace(api_key='test',base_url='http://localhost:8110/gemini')):
    cr._generate_gemini(self.profile(extra),'원문','번역')
   config=client.models.generate_content.call_args.kwargs['config']
   self.assertEqual(str(config.thinking_config.thinking_level).split('.')[-1] if config.thinking_config else None,want)
 def test_cache_distinguishes_level(self):
  keys=[]
  for extra in [{},{'thinking_level':'low'},{'thinking_level':'high'}]:
   with patch.object(cr,'resolve',return_value=self.profile(extra)):keys.append(core._chunk_key('same source',core.Options()))
  self.assertEqual(len(set(keys)),3)
if __name__ == "__main__":
 unittest.main()
