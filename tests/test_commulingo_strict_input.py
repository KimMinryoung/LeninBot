import unittest
from unittest.mock import AsyncMock
from tool_gateway.results import ToolRejection

from jsonschema import Draft202012Validator
from commulingo.pipeline.strict_input import strict_tool
from commulingo.pipeline.author_draft import AuthorDraft, obj
from llm.openai_tool_loop import _convert_tool_anthropic_to_openai


class StrictInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_languages_null_values_and_withdrawal_are_distinct(self):
        fields = obj({'body':obj({'ko':{'type':'string'},'en':{'type':'string'}}, ['ko','en']),
                      'year':{'type':['integer','null']}})
        draft = AuthorDraft(fields, [])
        draft.save({'status':'ready','reason':'The source supports this edit.', 'issue_results':[],
                    'claims':[], 'fields':{'body':{'ko':'이전 문장','en':'Previous text'},'year':1950}})
        handler = AsyncMock(side_effect=lambda **value: draft.submission(value))
        tool, accept = strict_tool(draft.submit_tool, handler)
        converted = _convert_tool_anthropic_to_openai(tool)
        self.assertTrue(converted['function']['strict'])
        Draft202012Validator.check_schema(tool['input_schema'])
        value = dict.fromkeys(tool['input_schema']['properties'])
        value['fields'] = {'body':{'ko':None,'en':'Updated text'},'year':None}
        updated = await accept(**value)
        self.assertEqual(updated['fields'], {'body':{'ko':'이전 문장','en':'Updated text'},'year':1950})
        value['fields']['year'] = {'value':None}
        cleared = await accept(**value)
        self.assertIsNone(cleared['fields']['year'])
        value['fields'] = None
        value['remove_fields'] = ['year']
        removed = await accept(**value)
        self.assertNotIn('year', removed['fields'])
        value['remove_fields'] = ['year','year']
        with self.assertRaises(ToolRejection):
            await accept(**value)
        self.assertEqual(draft.draft['args']['fields']['year'], 1950)

    async def test_wire_rejects_unknown_keys_and_required_null(self):
        draft = AuthorDraft(obj({'body':{'type':'string'}}), [{'id':'missing:body'}])
        handler = AsyncMock()
        tool, accept = strict_tool(draft.no_edit_tool, handler)
        value = {'status':'complete','reason':'Already sufficiently documented.',
                 'issues':{'missing:body':{'status':'resolved','reason':'Already documented.'}}, 'notes':None}
        await accept(**value)
        self.assertNotIn('notes', handler.call_args.kwargs)
        for bad in ({**value,'reason':None}, {**value,'unexpected':1}):
            with self.assertRaises(ToolRejection):
                await accept(**bad)
        self.assertEqual(handler.await_count, 1)
