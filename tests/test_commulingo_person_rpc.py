"""Opt-in cross-runtime test; points only at the isolated RPC container."""
import asyncio
import os
import sys
import unittest
from pathlib import Path
import uuid
from unittest.mock import patch
if os.environ.get('COMMULINGO_FRONTEND_CONTAINER') != 'commulingo-python-rpc':
    raise unittest.SkipTest('opt-in isolated RPC container required')
sys.path.insert(0, '/home/grass/leninbot')
source_root = Path(os.environ.get('COMMULINGO_TEST_SOURCE', Path(__file__).resolve().parents[1]))
import db
db.query = lambda *a, **k: []
db.query_one = lambda *a, **k: None
import runtime_tools
runtime_tools.__path__.insert(0, str(source_root / 'runtime_tools'))
from runtime_tools import commulingo_people as people
from runtime_tools.commulingo_person_service import call_person_service

class SharedPersonRPC(unittest.TestCase):
    def test_actual_python_writes_reads_and_review(self):
        source = 'Isolated archive, volume 1'
        suffix = uuid.uuid4().hex[:8]
        person_id = 'python-rpc-' + suffix
        fields = {'groupId': 'rpc-test-group', 'name': {'ko': '검증 작가 ' + suffix, 'en': 'RPC Author ' + suffix},
                  'role': {'icon': 'book-open'}, 'bio': {'ko': '원래 소개', 'en': 'Original biography'},
                  'evidence': [{'field': 'bio', 'claim': 'Original biography', 'source': source, 'locator': 'p. 1'}]}
        with patch.object(people, 'direct_apply_enabled', return_value=True):
            result = asyncio.run(people._exec_commulingo_write('person', 'create', person_id, [source], fields))
            self.assertTrue(str(result).startswith('OK — approved:'), str(result))
            person = people._get_person(person_id)
            self.assertTrue(person['revision'].startswith('v1-'))
            result = asyncio.run(people._exec_commulingo_write('person', 'update', person_id, [source],
                {'expectedRevision': person['revision'], 'bio': {'ko': '수정 소개'},
                 'evidence': [{'field': 'bio', 'claim': 'Corrected biography', 'source': source, 'locator': 'p. 2'}]}))
            self.assertTrue(str(result).startswith('OK — approved:'), str(result))
            self.assertEqual(people._get_person(person_id)['bio']['en'], 'Original biography')
            stale = asyncio.run(people._exec_commulingo_write('person', 'update', person_id, [source],
                {'expectedRevision': person['revision'], 'epithet': {'ko': '오래된 편집'}}))
            self.assertIn('revision_conflict', str(stale))
            current = people._get_person(person_id)
            result = asyncio.run(people._exec_commulingo_write('person', 'update', person_id, [source],
                {'expectedRevision': current['revision'], 'reviewFlags': ['source_conflict'],
                 'bio': {'ko': '검토할 주장'}, 'evidence': [{'field': 'bio', 'claim': 'Disputed claim', 'source': source, 'locator': 'p. 3', 'stance': 'disputes'}]}))
            self.assertTrue(str(result).startswith('OK — pending:'), str(result))
            self.assertEqual(people._get_person(person_id)['bio']['ko'], '수정 소개')
            import re
            sid = re.search(r'edit #(\d+)', str(result)).group(1)
            approved = call_person_service({'command': 'review', 'suggestionId': sid, 'approve': True, 'note': 'Compared original source', 'changedBy': 'rpc-test-review'})
            self.assertEqual(approved['status'], 'approved')
            self.assertEqual(people._get_person(person_id)['bio']['ko'], '검토할 주장')
        # Legacy backfills now pass an explicit spec through the same atomic CLI.
        from runtime_tools.commulingo_person_service import apply_person_spec
        import tempfile, json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as spec:
            json.dump({'people': [{'id': person_id, 'expectedRevision': people._get_person(person_id)['revision'],
                'sources': [source], 'epithet': {'ko': '공통 CLI 수정'}}]}, spec)
            spec.flush()
            self.assertIn('committed 1 person', apply_person_spec(spec.name))
        self.assertEqual(people._get_person(person_id)['epithet']['ko'], '공통 CLI 수정')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as spec:
            version = people._get_person(person_id)['revision']
            entries = [{'id': person_id, 'expectedRevision': version, 'sources': [source],
                        'epithet': {'ko': text}} for text in ['롤백할 첫 수정', '충돌하는 둘째 수정']]
            json.dump({'people': entries}, spec); spec.flush()
            with self.assertRaises(RuntimeError): apply_person_spec(spec.name)
        self.assertEqual(people._get_person(person_id)['epithet']['ko'], '공통 CLI 수정', 'batch failure rolls back prior writes')

        with self.assertRaisesRegex(ValueError, 'shared editorial'):
            people.apply_edit(None, 'person', 'update', person_id, {}, 'forbidden-legacy')

if __name__ == '__main__': unittest.main()
