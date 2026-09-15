"""Mail identity, cache, read coverage and delivery against an isolated test schema.

Run with MAIL_TEST_DATABASE=1 DB_NAME=leninbot_test and DB credentials.
No real IMAP connections or Telegram messages are made.
"""
import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from psycopg2.extras import RealDictCursor

from mail_runtime import store
from mail_runtime.delivery import deliver
from mail_runtime.inbox import collect


def test_read_ranges_do_not_confuse_overlap_with_full_coverage():
    assert store.merge_ranges([[0, 100], [200, 300]], 250, 300) == [[0, 100], [200, 300]]
    assert store.merge_ranges([[0, 100], [200, 300]], 100, 200) == [[0, 300]]


@pytest.fixture
def ledger(monkeypatch):
    if os.environ.get('MAIL_TEST_DATABASE') != '1':
        pytest.skip('requires explicitly selected test database')
    assert os.environ.get('DB_NAME') == 'leninbot_test'
    from dotenv import load_dotenv
    load_dotenv()
    import db
    schema = 'mail_test_' + uuid4().hex
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA {schema}')

    @contextmanager
    def connection():
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f'SET LOCAL search_path TO {schema}')
            yield conn

    def query(sql, params=None):
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()] if cur.description else []

    monkeypatch.setattr(store, 'get_conn', connection)
    monkeypatch.setattr(store, 'query', query)
    monkeypatch.setattr(store, 'query_one', lambda sql, params=None: next(iter(query(sql, params)), None))
    monkeypatch.setattr(store, 'execute', query)
    with connection() as conn, conn.cursor() as cur:
        cur.execute(Path('mail_runtime/schema.sql').read_text())
        cur.execute('CREATE TABLE telegram_tasks(id BIGINT PRIMARY KEY, user_id BIGINT)')
        cur.execute('INSERT INTO telegram_tasks VALUES (1,42),(2,42),(3,43)')
    try:
        yield query
    finally:
        with db.get_conn() as conn, conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA {schema} CASCADE')


class Imap:
    def __init__(self):
        self.box = None
        self.validity = b'100'
        self.fetches = []
        self.seen = True

    def select(self, folder, readonly=True):
        assert readonly
        self.box = folder
        return 'OK', []

    def response(self, name):
        assert name == 'UIDVALIDITY'
        return name, [self.validity]

    def uid(self, command, *args):
        if command == 'search':
            return 'OK', [b'' if args[-1] == 'UNSEEN' and self.seen else b'1 2']
        uid, spec = args
        self.fetches.append((self.box, uid, spec))
        flags = f'{uid} (UID {uid} FLAGS (' + ('\\Seen' if self.seen else '') + '))'
        if spec == '(FLAGS)':
            return 'OK', [flags.encode()]
        assert spec == '(FLAGS BODY.PEEK[])'
        raw = (f'From: sender@example.test\nSubject: Mail {uid}\n'
               'Date: Mon, 01 Jan 2024 00:00:00 +0000\n\n' + '본문' * 100).encode()
        return 'OK', [(flags.encode(), raw)]

    def logout(self):
        pass


def read(imap, task=1, **kwargs):
    from runtime_tools.registry import _parse_email_message
    return collect(connect=lambda: imap, parse=_parse_email_message, scope=(task, '42'),
                   audience='42', **kwargs)


def test_cached_pagination_and_delivery_are_independent_of_seen(ledger):
    imap = Imap()
    first = read(imap, folder='INBOX', limit=1, body_max_chars=100)['messages'][0]
    assert first['imap_read'] is True
    assert first['briefing_delivered'] is False
    assert first['body_returned_completely_in_this_task'] is False
    mail_id = first['mail_id']
    with pytest.raises(ValueError, match='read all'):
        store.prepare(1, '42', store.account_key(), [{'mail_id': mail_id, 'summary': '내용 요약'}])
    before = len(imap.fetches)
    second = read(imap, **first['next'])['messages'][0]
    assert second['body_returned_completely_in_this_task'] is True
    assert len(imap.fetches) == before  # Offline page: no IMAP fetch at all.
    items = [{'mail_id': mail_id, 'summary': '메일은 본문을 반복한다고 설명합니다.'}]
    store.prepare(1, '42', store.account_key(), items)
    assert not store.is_delivered(mail_id, '42')
    store.mark_sent(1, mail_id, '42', 123)
    assert store.is_delivered(mail_id, '42')
    assert not store.is_delivered(mail_id, '43')
    again = read(imap, folder='INBOX')['messages']
    assert mail_id not in [r['mail_id'] for r in again]
    history = read(imap, folder='INBOX', unbriefed_only=False)['messages']
    assert any(r['mail_id'] == mail_id and r['briefing_delivered'] for r in history)
    assert sum(spec == '(FLAGS BODY.PEEK[])' for _, _, spec in imap.fetches) == 2
    assert read(imap, folder='INBOX', unread_only=True)['messages'] == []


def test_folder_and_uidvalidity_prevent_identity_collisions(ledger):
    imap = Imap()
    rows = read(imap)['messages']
    assert len({r['mail_id'] for r in rows}) == 4
    old = {r['mail_id'] for r in rows}
    imap.validity = b'200'
    assert old.isdisjoint({r['mail_id'] for r in read(imap)['messages']})
    imap.validity = None
    with pytest.raises(RuntimeError, match='UIDVALIDITY'):
        read(imap)


def test_prepare_batch_is_atomic_and_requires_this_tasks_reads(ledger):
    rows = read(Imap(), folder='INBOX')['messages']
    items = [{'mail_id': r['mail_id'], 'summary': '발신자는 내용을 설명합니다.'} for r in rows]
    with pytest.raises(ValueError):
        store.prepare(2, '42', store.account_key(), items)
    with pytest.raises(ValueError):
        store.prepare(1, '42', store.account_key(), [items[0], {'mail_id': 9999, 'summary': '없는 메일'}])
    assert store.briefing_items(1, '42') == []


def test_partial_send_failure_only_marks_acknowledged_mail(ledger):
    rows = read(Imap(), folder='INBOX')['messages']
    store.prepare(1, '42', store.account_key(), [
        {'mail_id': r['mail_id'], 'summary': '원문에 따르면 중요한 내용입니다.'} for r in rows])
    items = store.briefing_items(1, '42')
    bot = SimpleNamespace(send_message=AsyncMock(side_effect=[SimpleNamespace(message_id=10), RuntimeError('offline')]))
    persist = AsyncMock()
    with pytest.raises(RuntimeError, match='offline'):
        asyncio.run(deliver(bot, 1, 42, items, persist))
    assert store.is_delivered(items[0]['mail_id'], '42')
    assert not store.is_delivered(items[1]['mail_id'], '42')
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=11))
    asyncio.run(deliver(bot, 1, 42, store.briefing_items(1, '42'), persist))
    assert bot.send_message.await_count == 1
    assert items[1]['summary'] in bot.send_message.call_args.kwargs['text']
    assert len(read(Imap(), folder='INBOX')['messages']) == 0


def test_output_budget_records_only_returned_ranges_and_preserves_prior_reads(ledger):
    from mail_runtime.inbox import render_messages
    rows = []
    for n in range(8):
        body = '\x01' * 12000
        parsed = {'body': body, 'body_chars': len(body), 'subject': '긴 메일',
                  'from': 'sender', 'date': '', 'links': [], 'date_sort_timestamp': 0}
        rows.append(store.capture(store.account_key(), 'INBOX', '1', str(n), b'raw', parsed))
    rendered = render_messages(rows, (1, '42'), True, 0, 12000)
    assert len(json.dumps(rendered, ensure_ascii=False)) < 41000
    for row, item in zip(rows, rendered):
        assert not store.read_state(row['id'], 12000, 1)['body_returned_completely_in_this_task']
        if item['returned_chars']:
            saved = ledger('SELECT ranges FROM mail_briefing_reads WHERE mail_id=%s', (row['id'],))
            assert saved[0]['ranges'] == [item['returned_chars']]
    store.record_read(1, rows[0]['id'], 0, 12000)
    later = render_messages([rows[0]], (2, '42'), False, 0, 12000)[0]
    assert later['body_fully_returned_task_id'] == 1
    assert not later['body_returned_completely_in_this_task']
