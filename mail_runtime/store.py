"""PostgreSQL ledger. IMAP read flags never stand in for delivery receipts."""
import hashlib
import json
import os

from psycopg2.extras import RealDictCursor
from db import get_conn, query, query_one, execute


def account_key():
    identity = [os.environ.get(k, '') for k in
                ('EMAIL_IMAP_HOST', 'EMAIL_IMAP_PORT', 'EMAIL_IMAP_USERNAME')]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def task_scope(task_id):
    if not task_id:
        return None
    row = query_one('SELECT id, user_id FROM telegram_tasks WHERE id=%s', (int(task_id),))
    return (int(row['id']), str(row['user_id'])) if row else None


def namespace(account, folder, validity, audience):
    return query('''SELECT m.id, m.uid, EXISTS (
        SELECT 1 FROM mail_briefing_items i WHERE i.mail_id=m.id
        AND i.audience=%s AND i.sent_at IS NOT NULL) AS briefed
        FROM mail_briefing_messages m
        WHERE m.account=%s AND m.folder=%s AND m.uidvalidity=%s''',
        (audience, account, folder, validity))


def get_message(mail_id, account):
    return query_one('SELECT * FROM mail_briefing_messages WHERE id=%s AND account=%s',
                     (mail_id, account))


def observe_flags(mail_id, seen):
    return query_one('''UPDATE mail_briefing_messages SET imap_seen=%s,
        flags_observed_at=now() WHERE id=%s RETURNING *''', (seen, mail_id))


def capture(account, folder, validity, uid, raw, parsed):
    return query_one('''INSERT INTO mail_briefing_messages
        (account, folder, uidvalidity, uid, raw, parsed) VALUES (%s,%s,%s,%s,%s,%s::jsonb)
        ON CONFLICT(account, folder, uidvalidity, uid)
        DO UPDATE SET uid=EXCLUDED.uid RETURNING *''',
        (account, folder, validity, uid, raw, json.dumps(parsed)))


def merge_ranges(ranges, start, end):
    merged = []
    for left, right in sorted([*ranges, [start, end]]):
        if right <= left:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(right, merged[-1][1])
        else:
            merged.append([left, right])
    return merged


def record_read(task_id, mail_id, start, end):
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute('''INSERT INTO mail_briefing_reads(task_id,mail_id) VALUES (%s,%s)
            ON CONFLICT DO NOTHING''', (task_id, mail_id))
        cur.execute('SELECT ranges FROM mail_briefing_reads WHERE task_id=%s AND mail_id=%s FOR UPDATE',
                    (task_id, mail_id))
        ranges = merge_ranges(cur.fetchone()['ranges'], start, end)
        cur.execute('''UPDATE mail_briefing_reads SET ranges=%s::jsonb, updated_at=now()
            WHERE task_id=%s AND mail_id=%s''', (json.dumps(ranges), task_id, mail_id))
        return ranges


def read_state(mail_id, body_chars, task_id=None):
    rows = query('''SELECT task_id,ranges,updated_at FROM mail_briefing_reads
        WHERE mail_id=%s ORDER BY updated_at DESC''', (mail_id,))
    full = [r for r in rows if not body_chars or r['ranges'] == [[0, body_chars]]]
    return {
        'body_fully_returned_at': str(full[0]['updated_at']) if full else None,
        'body_fully_returned_task_id': full[0]['task_id'] if full else None,
        'body_returned_completely_in_this_task': any(r['task_id'] == task_id for r in full),
    }


def prepare(task_id, audience, account, items):
    """Validate the entire batch before storing; delivered summaries are immutable."""
    if not 1 <= len(items) <= 20 or len({i['mail_id'] for i in items}) != len(items):
        raise ValueError('Provide 1–20 distinct mail IDs.')
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        for item in items:
            summary = item['summary'].strip()
            if not 1 <= len(summary) <= 1800:
                raise ValueError('Each source-attributed summary must be 1–1800 characters.')
            cur.execute('''SELECT m.parsed, r.ranges FROM mail_briefing_messages m
                JOIN mail_briefing_reads r ON r.mail_id=m.id AND r.task_id=%s
                WHERE m.id=%s AND m.account=%s''', (task_id, item['mail_id'], account))
            row = cur.fetchone()
            size = row['parsed']['body_chars'] if row else 0
            if not row or (size and row['ranges'] != [[0, size]]):
                raise ValueError(f"Mail {item['mail_id']}: read all cached body pages in this task before preparing.")
            cur.execute('''INSERT INTO mail_briefing_items(task_id,mail_id,audience,summary)
                VALUES (%s,%s,%s,%s) ON CONFLICT(task_id,mail_id) DO UPDATE
                SET summary=EXCLUDED.summary, prepared_at=now()
                WHERE mail_briefing_items.sent_at IS NULL''',
                (task_id, item['mail_id'], audience, summary))


def briefing_items(task_id, audience):
    return query('''SELECT i.*, m.parsed, m.folder, m.uid FROM mail_briefing_items i
        JOIN mail_briefing_messages m ON m.id=i.mail_id
        WHERE i.task_id=%s AND i.audience=%s ORDER BY i.prepared_at, i.mail_id''',
        (task_id, str(audience)))


def mark_sent(task_id, mail_id, audience, message_id):
    execute('''UPDATE mail_briefing_items SET sent_at=now(),telegram_message_id=%s
        WHERE task_id=%s AND mail_id=%s AND audience=%s AND sent_at IS NULL''',
        (message_id, task_id, mail_id, str(audience)))


def is_delivered(mail_id, audience):
    return bool(query_one('''SELECT 1 FROM mail_briefing_items
        WHERE mail_id=%s AND audience=%s AND sent_at IS NOT NULL LIMIT 1''',
        (mail_id, str(audience))))
