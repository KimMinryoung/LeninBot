from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch

from runtime_tools.commulingo_section_slug import generate_section_slug


class SectionSlugTests(TestCase):
    def test_registry_generates_distinct_topic_slug_and_tracks_cost(self):
        usage = SimpleNamespace(tracker={})
        profile = SimpleNamespace(provider='openai', model='gpt-6-luna')
        result = SimpleNamespace(text='military-career', usage={'tokens_in': 100, 'tokens_out': 12}, error=None)
        with patch('runtime_tools.commulingo_section_slug.resolve', return_value=profile), \
             patch('runtime_tools.commulingo_section_slug.generate_detailed', return_value=result) as call, \
             patch('runtime_tools.commulingo_section_slug.estimate_cost_usd', return_value=.0001):
            slug = generate_section_slug('nikolai-pukhov',
                {'ko':'군 경력', 'en':'Military career'}, {'en':'His commands.'},
                [{'slug':'early-life','heading':{'en':'Early life'}}], usage=usage)
        self.assertEqual(slug, 'military-career')
        self.assertEqual(usage.tracker['auxiliary_cost_usd'], .0001)
        self.assertEqual(usage.tracker['section_slug_calls'], 1)
        self.assertEqual(call.call_args.args[0], 'commulingo_section_slug')

    def test_unusable_or_failed_model_slug_falls_back_to_heading(self):
        profile = SimpleNamespace(provider='openai', model='gpt-6-luna')
        existing = [{'slug':'early-life'}, {'slug':'military-career'}]
        with patch('runtime_tools.commulingo_section_slug.resolve', return_value=profile), \
             patch('runtime_tools.commulingo_section_slug.generate_detailed') as call:
            for text in ('nikolai-pukhov', 'early-life', 'not a slug!' * 10, None):
                with self.subTest(text=text):
                    call.return_value = SimpleNamespace(text=text, usage={}, error=None)
                    usage = SimpleNamespace(tracker={})
                    slug = generate_section_slug('nikolai-pukhov', {'en':'Military career'}, {},
                                                 existing, usage=usage)
                    self.assertEqual(slug, 'military-career-2')
                    self.assertEqual(usage.tracker['section_slug_fallbacks'], 1)

    def test_model_slug_is_folded_before_validation(self):
        profile = SimpleNamespace(provider='openai', model='gpt-6-luna')
        result = SimpleNamespace(text='`Military Career`', usage={}, error=None)
        with patch('runtime_tools.commulingo_section_slug.resolve', return_value=profile), \
             patch('runtime_tools.commulingo_section_slug.generate_detailed', return_value=result):
            self.assertEqual(generate_section_slug('p', {'en':'Military career'}, {}), 'military-career')

    def test_fallback_trims_long_and_korean_only_headings(self):
        profile = SimpleNamespace(provider='openai', model='gpt-6-luna')
        failed = SimpleNamespace(text=None, usage={}, error='timeout')
        with patch('runtime_tools.commulingo_section_slug.resolve', return_value=profile), \
             patch('runtime_tools.commulingo_section_slug.generate_detailed', return_value=failed):
            long = generate_section_slug('p', {'en':'SADCC and Regional Economic Liberation: A Common Front Against Dependence'}, {})
            self.assertEqual(long, 'sadcc-and-regional-economic-liberation-a-common')
            self.assertEqual(generate_section_slug('p', {'ko':'군 경력'}, {}), 'section')

    def test_existing_heading_never_calls_model(self):
        with patch('runtime_tools.commulingo_section_slug.generate_detailed') as call:
            with self.assertRaisesRegex(ValueError, "'early-life' already covers"):
                generate_section_slug('person', {'en':'EARLY LIFE'}, {},
                    [{'slug':'early-life','heading':{'en':'Early life'}}])
            call.assert_not_called()

    def test_missing_heading_never_calls_model(self):
        with patch('runtime_tools.commulingo_section_slug.generate_detailed') as call:
            with self.assertRaisesRegex(ValueError, 'heading is required'):
                generate_section_slug('person', {}, {})
            call.assert_not_called()


class SectionApiTests(IsolatedAsyncioTestCase):
    async def test_create_generates_slug_after_read_and_update_keeps_existing_slug(self):
        from runtime_tools.commulingo_people import _exec_commulingo_section_save
        current = {'sections':[{'slug':'early-life'}]}
        with patch('runtime_tools.commulingo_people.call_person_service', return_value=current) as read, \
             patch('runtime_tools.commulingo_section_slug.generate_section_slug', return_value='military-career') as slugger, \
             patch('runtime_tools.commulingo_people._exec_commulingo_write', return_value='saved') as write:
            result = await _exec_commulingo_section_save('create','nikolai-pukhov',
                {'en':'Military career'}, {'en':'His commands.'}, [], expected_revision='v1')
            self.assertEqual(result, 'saved')
            self.assertEqual(write.call_args.args[4]['slug'], 'military-career')
            self.assertEqual(slugger.call_args.args[3], current['sections'])
            read.assert_called_once()
            result = await _exec_commulingo_section_save('update','nikolai-pukhov',
                {'en':'Early life'}, {'en':'A revision.'}, [], slug='early-life', expected_revision='v2')
            self.assertEqual(result, 'saved')
            self.assertEqual(write.call_args.args[4]['slug'], 'early-life')
            slugger.assert_called_once()
            read.assert_called_once()

    async def test_update_without_slug_is_rejected_before_write(self):
        from runtime_tools.commulingo_people import _exec_commulingo_section_save
        with patch('runtime_tools.commulingo_people._exec_commulingo_write') as write:
            result = await _exec_commulingo_section_save('update','nikolai-pukhov',
                {'en':'Early life'}, {'en':'A revision.'}, [])
            self.assertIn('slug is required', result)
            write.assert_not_called()

    async def test_duplicate_heading_on_create_returns_error_without_write(self):
        from runtime_tools.commulingo_people import _exec_commulingo_section_save
        current = {'sections':[{'slug':'early-life','heading':{'en':'Early life'}}]}
        with patch('runtime_tools.commulingo_people.call_person_service', return_value=current), \
             patch('runtime_tools.commulingo_section_slug.generate_detailed') as call, \
             patch('runtime_tools.commulingo_people._exec_commulingo_write') as write:
            result = await _exec_commulingo_section_save('create','nikolai-pukhov',
                {'en':'Early Life'}, {'en':'Childhood.'}, [], expected_revision='v1')
        self.assertIn("'early-life' already covers", result)
        call.assert_not_called()
        write.assert_not_called()
