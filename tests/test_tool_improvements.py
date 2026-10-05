"""Regression tests for download, audit, corpus and proposal-specific boundaries."""
import json
import socket
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from tool_gateway.results import ToolFailure, ToolResult, ToolRejection, ToolContinue


class Response:
    def __init__(self, chunks=(), headers=None, status=200):
        self.chunks = chunks
        self.headers = headers or {'Content-Type': 'image/png'}
        self.status_code = status
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_content(self, **kwargs):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class DownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_initial_and_redirect_never_contact_private_target(self):
        from runtime_tools.fetch import _exec_download_file, _exec_download_image
        from content_fetch.url_security import safe_requests_get
        resolver = lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))]
        for handler in (_exec_download_file, _exec_download_image):
            request = Mock()
            with patch('requests.get', request):
                self.assertIsInstance(await handler('http://127.0.0.1/a'), ToolFailure)
                request.assert_not_called()
            redirect = Response(headers={'Location': 'http://169.254.169.254/a'}, status=302)
            request = Mock(return_value=redirect)
            def safe(url, **kw):
                return safe_requests_get(url, request_get=request, resolver=resolver, **kw)
            with patch('content_fetch.url_security.safe_requests_get', safe):
                self.assertIsInstance(await handler('https://example.org/a'), ToolFailure)
            self.assertEqual(request.call_count, 1)
            self.assertTrue(redirect.closed)

    async def test_stream_limits_cleanup_and_atomic_replacement(self):
        from runtime_tools.fetch import _exec_download_file, _exec_download_image
        for handler, directory, mb in ((_exec_download_file, 'downloads', 100), (_exec_download_image, 'reference_images', 20)):
            with tempfile.TemporaryDirectory() as tmp, patch('runtime_tools.fetch._project_root', return_value=tmp):
                target = Path(tmp) / 'data' / directory / 'keep.png'
                target.parent.mkdir(parents=True)
                target.write_bytes(b'old complete')
                cases = [Response([b'x'], {'Content-Type':'image/png','Content-Length':str((mb+1)*1024**2)}),
                         Response([b'x' * 1024**2] * (mb+1)),
                         Response([b'partial', OSError('broken stream')]), Response([])]
                for response in cases:
                    with patch('content_fetch.url_security.safe_requests_get', return_value=response) as get:
                        result = await handler('https://example.org/a', 'keep.png')
                    self.assertIsInstance(result, ToolFailure)
                    self.assertTrue(get.call_args.kwargs['stream'])
                    self.assertTrue(response.closed)
                    self.assertEqual(target.read_bytes(), b'old complete')
                    self.assertEqual(list(target.parent.iterdir()), [target])
                with patch('content_fetch.url_security.safe_requests_get', return_value=Response([b'complete'])):
                    result = await handler('https://example.org/a', 'keep.png')
                self.assertNotIsInstance(result, ToolFailure)
                self.assertEqual(target.read_bytes(), b'complete')

    async def test_extraction_failures_offsets_and_diagnostics(self):
        from runtime_tools.fetch import _exec_fetch_url, _exec_convert_document
        with self.assertRaises(ToolRejection):
            await _exec_fetch_url('https://example.org', offset=-1)
        with patch('content_fetch.urls.fetch_url_content_async', AsyncMock(return_value='short')):
            with self.assertRaises(ToolRejection):
                await _exec_fetch_url('https://example.org', offset=9)
        with patch('content_fetch.urls.fetch_url_content_async', AsyncMock(return_value='')), patch('content_fetch.urls.diagnose_url_fetch_failure', return_value='offline'):
            result = await _exec_fetch_url('https://example.org')
            self.assertIsInstance(result, ToolFailure)
            self.assertTrue(result.result_metadata['empty'])
        with patch('content_fetch.urls.fetch_url_content_async', AsyncMock(return_value=ToolResult('text', {'path':'http','cache_hit':False}))):
            result = await _exec_fetch_url('https://example.org')
            self.assertEqual(result.result_metadata['extracted_chars'], 4)
            self.assertEqual(result.result_metadata['path'], 'http')
        self.assertIsInstance(await _exec_convert_document('/nonexistent/file.pdf'), ToolFailure)
        with tempfile.NamedTemporaryFile() as file, patch('content_fetch.documents.convert_document', return_value=''):
            self.assertIsInstance(await _exec_convert_document(file.name), ToolFailure)


class CorpusTests(unittest.IsolatedAsyncioTestCase):
    async def test_uuid_context_isolation_limits_missing_and_ambiguous(self):
        from runtime_tools.corpus import read_corpus_passage
        uid = str(uuid4())
        center = {'id': uid, 'content': 'center', 'metadata': {'source':'doc','layer':'core_theory','chunk_index':4}}
        chunks = [{'id': uid if i == 4 else str(uuid4()), 'content': 'x'*10000, 'metadata': {'chunk_index':i}} for i in (3,4,5)]
        with patch('db.query', side_effect=[[center], chunks]) as query:
            result = await read_corpus_passage(uid)
        self.assertEqual(len(result), 20000)
        self.assertIn(uid, result)
        self.assertIn('truncated', result)
        sql, params = query.call_args.args
        self.assertIn("metadata->>'layer' = %s", sql)
        self.assertEqual(params, ('doc','core_theory',3,5))
        with patch('db.query', return_value=[]):
            self.assertIn('deleted', await read_corpus_passage(uid))
        with patch('db.query', return_value=[{**center, 'metadata':{'source':'doc'}}]):
            self.assertIn('Insufficient', await read_corpus_passage(uid))
        with patch('db.query', side_effect=[[center], [chunks[1], chunks[1]]]):
            self.assertIn('Ambiguous', await read_corpus_passage(uid))
        with self.assertRaises(ToolRejection):
            await read_corpus_passage('made-up-id')
        with self.assertRaises(ToolRejection):
            await read_corpus_passage(uid, window=4)

    async def test_vector_empty_success_uuid_and_backend_failure(self):
        from runtime_tools.vector_search import exec_vector_search, search_corpus_multilingual
        with patch('runtime_tools.vector_search.search_corpus_multilingual', AsyncMock(return_value=[])):
            result = await exec_vector_search('q')
            self.assertNotIsInstance(result, ToolFailure)
            self.assertEqual(result.result_metadata['result_count'], 0)
        uid = str(uuid4())
        doc = SimpleNamespace(metadata={'chunk_id':uid}, page_content='source text')
        with patch('runtime_tools.vector_search.search_corpus_multilingual', AsyncMock(return_value=[doc])):
            self.assertIn(uid, await exec_vector_search('q'))
        with patch('corpus.store.similarity_search', side_effect=RuntimeError('offline')), patch('runtime_tools.vector_search.llm_translate_search_query', AsyncMock(return_value='query')):
            with self.assertRaises(RuntimeError):
                await search_corpus_multilingual('검색', 5, 'core_theory')


class AuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_report_periods_test_partition_and_unknown_history(self):
        from mcp_gateway.audit_tools import tool_usage_report
        rows = [dict(tool_name='fetch_url', interface='telegram', agent_name=None,
                     execution_kind=kind, period=period, calls=count, latency_p50_ms=10,
                     latency_p95_ms=20, result_statuses={'ok':count})
                for kind,period,count in [('runtime','current',8),('runtime','previous',4),('test','current',2),('unknown','previous',3)]]
        with patch('mcp_gateway.audit_tools._query', return_value=rows) as query:
            data = json.loads(await tool_usage_report(profile='operator'))
        self.assertEqual(data['groups']['runtime'][0]['calls_change_pct'], 100)
        self.assertEqual(data['groups']['test'][0]['current']['calls'], 2)
        self.assertEqual(data['groups']['unknown'][0]['calls_delta'], -3)
        args = query.call_args.args[1]
        self.assertEqual(args[2] - args[0], timedelta(days=7))
        self.assertEqual(args[0] - args[1], timedelta(days=7))
        for days in (0,91,True):
            with self.assertRaises(ToolRejection):
                await tool_usage_report(days=days, profile='operator')
        with self.assertRaises(ToolRejection):
            await tool_usage_report(profile='inspect')

    async def test_trace_pagination_relationships_retries_no_arguments(self):
        from mcp_gateway.audit_tools import trace_tool_run
        rows = [dict(id=i, task_id='12', request_id='child', parent_request_id='parent',
                     result_status='ok', same_call_ordinal=i+1) for i in range(201)]
        with patch('mcp_gateway.audit_tools._query', side_effect=[rows,[{'id':12,'status':'done'}]]) as query:
            data = json.loads(await trace_tool_run(task_id=12, profile='operator'))
        self.assertEqual(len(data['calls']), 200)
        self.assertEqual(data['next_offset'], 200)
        self.assertEqual(data['calls'][1]['retry_evidence'], 'possible_repeat')
        self.assertEqual(data['tasks'][0]['status'], 'done')
        sql = query.call_args_list[0].args[0]
        self.assertNotIn('args_summary', data['calls'][0])
        self.assertNotIn('error_excerpt', data['calls'][0])
        self.assertIn('UNION', sql)
        for args in ({}, {'request_id':'a','task_id':1}, {'request_id':'a','limit':201}, {'request_id':'a','offset':-1}):
            with self.assertRaises(ToolRejection):
                await trace_tool_run(profile='operator', **args)
        with patch('mcp_gateway.audit_tools._query', return_value=[]):
            data = json.loads(await trace_tool_run(request_id='absent',profile='operator'))
            self.assertEqual(data['calls'], [])
            self.assertIsNone(data['next_offset'])

    async def test_mcp_failures_rejections_and_continuations_audited(self):
        from mcp_gateway.server import _call_tool
        for result,status in [(ToolFailure('empty', {'empty':True}),'error'), (ToolContinue('saved'),'continued'), (ToolResult('no hits',{'result_count':0}),'ok')]:
            with patch('mcp_gateway.server.build_handlers', return_value={'fetch_url':AsyncMock(return_value=result)}), patch('security_gateway.audit.audit') as audit:
                response = await _call_tool('fetch_url', {'url': 'https://example.org'}, 'inspect')
            self.assertEqual(response['isError'], status!='ok')
            self.assertEqual(audit.call_args.kwargs['result_status'],status)
            self.assertEqual(audit.call_args.kwargs['result_metadata'],result.result_metadata)
        with patch('mcp_gateway.server.build_handlers',return_value={'fetch_url':AsyncMock(side_effect=ToolRejection('offset'))}), patch('security_gateway.audit.audit') as audit:
            await _call_tool('fetch_url',{'url': 'https://example.org'},'inspect')
        self.assertEqual(audit.call_args.kwargs['result_status'],'rejected')

    async def test_dispatcher_preserves_diagnostics_and_allows_failure_retry(self):
        from tool_gateway.dispatcher import execute_tool
        from security_gateway.context import CallerContext, caller_scope
        with caller_scope(CallerContext(interface="system", is_owner=True, execution_kind="test")):
            handler = AsyncMock(side_effect=[ToolFailure("failed", {"failure_type":"provider_failure"}),
                                             ToolResult("no results", {"result_count":0,"empty":True})])
            cache = {}
            with patch('tool_gateway.security.audit') as audit:
                first = await execute_tool('web_search', {'query':'q'}, {'web_search':handler}, idempotency_cache=cache)
                second = await execute_tool('web_search', {'query':'q'}, {'web_search':handler}, idempotency_cache=cache)
            self.assertTrue(first[1])
            self.assertFalse(second[1])
            self.assertEqual(handler.await_count, 2)
            self.assertEqual([call.kwargs['result_status'] for call in audit.call_args_list], ['error','ok'])
            with patch('tool_gateway.security.audit') as audit:
                await execute_tool('web_search', {'query':'q'}, {'web_search':AsyncMock(return_value=ToolContinue('saved'*11000, {'returned_chars':55000}))})
            self.assertEqual(audit.call_args.kwargs['result_status'], 'continued')
            self.assertEqual(audit.call_args.kwargs['result_metadata'], {'returned_chars':55000})

    def test_execution_kind_and_bounded_metadata(self):
        from security_gateway.audit import audit
        from security_gateway.context import CallerContext
        from security_gateway.gateway import Decision
        from ops.audit_sink import normalize_row
        with patch('security_gateway.audit._WRITER.enqueue') as enqueue, patch.dict('os.environ', {'LENINBOT_EXECUTION_KIND':'test','LENINBOT_TOOL_AUDIT_DB':'1'}):
            audit(CallerContext(), 'fetch_url', {}, Decision(True,'allow','fetch','','',''), result_status='ok', result_metadata={'empty':False,'extracted_chars':10})
        row = enqueue.call_args.args[0]
        self.assertEqual(json.loads(normalize_row('tool',row)['result_metadata'])['execution_kind'],'test')
        self.assertIsNone(normalize_row('tool',{'tool_name':'x','decision':'allow'})['result_metadata'])
        for metadata in ({'body':'secret'}, {'path':'https://secret.example/path'}, {'result_count':'body text'}, {'empty':[]}, {'execution_kind':'invented'}):
            with self.assertRaises(ValueError):
                normalize_row('tool',{'tool_name':'x','decision':'allow','result_metadata':metadata})

    def test_kg_recall_metadata_round_trip(self):
        from ops.audit_sink import normalize_row
        for empty, injected, failed in ((False, True, False), (True, False, False), (True, False, True)):
            metadata = {'empty': empty, 'injected': injected, 'failed': failed}
            row = {'tool_name': 'kg_recall', 'decision': 'allow', 'result_metadata': metadata}
            normalized = normalize_row('tool', row)
            self.assertEqual(json.loads(normalized['result_metadata']), metadata)
            self.assertEqual(normalize_row('tool', normalized), normalized)
        for key in ('failed', 'injected'):
            with self.assertRaises(ValueError):
                normalize_row('tool', {'tool_name':'kg_recall','decision':'allow', 'result_metadata':{key:'raw text'}})

    def test_profiles_and_retired_implementations(self):
        from mcp_gateway.tools import build_handlers
        from runtime_tools.registry import TOOLS, TOOL_HANDLERS, RETIRED_TOOL_NAMES
        from runtime_tools.db import DB_TOOL_HANDLERS
        from tool_gateway.inference import REPLAY_SAFE_TOOLS
        from tool_gateway.profiles import WEB_CYBER_LENIN_TOOLS, A2A_GENERAL_TOOLS
        from security_gateway.gateway import authorize
        from security_gateway.context import CallerContext
        self.assertFalse(RETIRED_TOOL_NAMES & set(TOOL_HANDLERS))
        self.assertFalse(RETIRED_TOOL_NAMES & {t['name'] for t in TOOLS})
        self.assertIn('query_db', DB_TOOL_HANDLERS)
        self.assertNotIn('query_db', REPLAY_SAFE_TOOLS)
        self.assertIn('read_corpus_passage', build_handlers('inspect'))
        self.assertNotIn('tool_usage_report', build_handlers('inspect'))
        self.assertIn('tool_usage_report', build_handlers('operator'))
        self.assertNotIn('read_corpus_passage', WEB_CYBER_LENIN_TOOLS | A2A_GENERAL_TOOLS)
        for name in ('tool_usage_report','trace_tool_run','read_corpus_passage'):
            self.assertTrue(authorize(CallerContext(interface='webchat',is_owner=True),name).denied)


class ReviewAndDiaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_proposal_choices_and_cache_bad_ids_do_not_get_replaced(self):
        from commulingo.review_policy import decision_tool, resolve_review_checks
        from commulingo.pipeline.evidence import Passages
        from commulingo.pipeline.source_session import Sources
        proposal = {'source_refs':['https://example.org A','https://example.org B']}
        properties = decision_tool(proposal)['input_schema']['properties']['checks']['items']['properties']
        self.assertEqual(properties['citation_id']['enum'],['S1','S2'])
        with self.assertRaisesRegex(ValueError, 'S1, S2'):
            resolve_review_checks({'checks':[{'citation_id':'S99'}]},proposal,{},Passages())
        now = datetime.now(timezone.utc)
        source = dict(id='correct',body='A passage',url='https://example.org',fetched_at=now,expires_at=now+timedelta(days=1))
        session = Sources(None,{},SimpleNamespace(tracker={}),{'correct':source})
        read = session.cached_tool()[1]
        listing = await read()
        self.assertTrue(listing.result_metadata['cache_hit'])
        self.assertEqual(json.loads(listing)['available_pages'][0]['source_id'],'correct')
        page = await read(source_id='correct')
        self.assertEqual(page.result_metadata['extracted_chars'],len(source['body']))
        with self.assertRaisesRegex(ToolRejection,'correct'):
            await read(source_id='wrong')
        with self.assertRaises(ToolRejection):
            await read(source_id='correct',passages=['P999'])
        with self.assertRaisesRegex(ToolRejection,'available labels'):
            await read(passages=['P999'])

    async def test_target_fields_and_diary_modes(self):
        from commulingo.pipeline.review_context import context_tool
        from telegram.diary_mode import filter_diary_task_tools, DEFAULT_DIARY_WRITING_PROMPT
        tool, read, _ = context_tool({'term':'name','definition':'text'}, 'term')
        self.assertEqual(tool['input_schema']['properties']['fields']['items']['enum'],['definition','term'])
        with self.assertRaises(ToolRejection):
            await read(['bio'])
        tools, handlers = [{'name':'save_diary'},{'name':'edit_content'}], {'save_diary':Mock(),'edit_content':Mock()}
        with patch('telegram.diary_mode.scheduled_diary_prompts', return_value={DEFAULT_DIARY_WRITING_PROMPT}):
            selected,bound = filter_diary_task_tools({'agent_type':'diary','content':'correct typo'},tools,handlers)
            self.assertEqual(set(bound),{'edit_content'})
            self.assertEqual([t['name'] for t in selected],['edit_content'])
            self.assertEqual(filter_diary_task_tools({'agent_type':'diary','content':DEFAULT_DIARY_WRITING_PROMPT},tools,handlers), (tools,handlers))


if __name__ == '__main__':
    unittest.main()
