import unittest
from unittest.mock import patch

from commulingo_pipeline.draft_repair import DraftRepair
from scripts.commulingo_write_session import draft_id
from tool_gateway.dispatcher import execute_tool
from tool_gateway.security import caller_scope, new_run_context


class DraftRepairTests(unittest.IsolatedAsyncioTestCase):
    def session(self):
        return DraftRepair({'name':'commulingo_pipeline_result','input_schema':{
            'type':'object','additionalProperties':False,'required':['fields'],
            'properties':{'fields':{'type':'object','additionalProperties':False,
                'properties':{'bio':{'type':'object','additionalProperties':False,
                    'properties':{'en':{'type':'string','maxLength':20}}},
                    'groupId':{'type':'string','enum':['valid-group']}}}}}})

    async def test_actual_dispatcher_keeps_rejected_draft_and_accepts_only_valid_repair(self):
        session=self.session()
        completed=[]
        async def finish(**value):
            from tool_gateway.results import ToolRejection
            try:
                prepared=session.prepare(value)
            except ValueError as exc:
                raise ToolRejection(str(exc)) from exc
            completed.append(prepared)
            return 'saved'
        context=new_run_context(interface='autonomous',agent_name='commulingo_curator',is_owner=True,
            scope_type='maintenance_job',scope_id='commulingo_pipeline:repair-test')
        with caller_scope(context), patch('tool_gateway.security.audit'):
            result,failed=await execute_tool(session.name,{'fields':{'bio':{'en':'Long prose that exceeds the original limit'},
                'groupId':'valid-group'}},{session.name:finish},tool_schema=session.tool)
            self.assertTrue(failed)
            self.assertIn('draft_id=',result)
            self.assertEqual(completed,[])
            old_id=draft_id(session.draft)
            result,failed=await execute_tool(session.name,{'draft_id':old_id,'repairs':[
                {'op':'set','path':'/fields/bio/en','value':'Still much longer than the hard limit'}]},
                {session.name:finish},tool_schema=session.tool)
            self.assertTrue(failed)
            self.assertNotEqual(draft_id(session.draft),old_id)
            result,failed=await execute_tool(session.name,{'draft_id':draft_id(session.draft),'repairs':[
                {'op':'set','path':'/fields/bio/en','value':'Concise biography'}]},
                {session.name:finish},tool_schema=session.tool)
            self.assertFalse(failed,result)
        self.assertEqual(completed,[{'fields':{'bio':{'en':'Concise biography'},'groupId':'valid-group'}}])

    def test_repairs_target_the_current_draft_and_cannot_change_revision_or_shape(self):
        session=self.session()
        with self.assertRaises(ValueError):
            session.prepare({'draft_id':'nothing-yet','repairs':[{'op':'set','path':'/fields/bio/en','value':'Short'}]})
        with self.assertRaises(ValueError):
            session.prepare({'fields':{'bio':{'en':'Long prose that exceeds the original limit'}}})
        current=draft_id(session.draft)
        for value in (
            {'draft_id':current,'repairs':[{'op':'set','path':'/fields/expectedRevision','value':'new'}]},
            {'repairs':[{'op':'set','path':'/fields/bio/en','value':'Short'}],'fields':{}},
        ):
            with self.assertRaises(ValueError):
                session.prepare(value)
        # A stale or missing ID still repairs the one draft this call holds.
        self.assertEqual(session.prepare({'draft_id':'stale','repairs':[{'op':'set','path':'/fields/bio/en','value':'Short'}]}),
                         {'fields':{'bio':{'en':'Short'}}})
        self.assertEqual(session.prepare({'repairs':[{'op':'set','path':'/fields/bio/en','value':'Shorter'}]}),
                         {'fields':{'bio':{'en':'Shorter'}}})
        with self.assertRaises(ValueError):
            session.prepare({'repairs':[{'op':'set','path':'/fields','value':{'unexpected':1}}]})


if __name__=='__main__':unittest.main()
