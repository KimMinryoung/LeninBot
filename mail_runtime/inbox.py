"""Read IMAP identity/flags live, cache immutable message contents once."""
import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone

from mail_runtime import store
from security_gateway.context import get_caller
from tool_gateway.results import ToolFailure


PAGE_MAX = 12000
_IMAP_MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')


def new_mail_window_days():
    """Unbriefed listings only look this many days back; 0 disables the bound."""
    try:
        return max(0, int(os.environ.get('MAIL_BRIEFING_WINDOW_DAYS', '7')))
    except ValueError:
        return 7


def _imap_since(days):
    day = datetime.now(timezone.utc) - timedelta(days=days)
    return f'{day.day:02d}-{_IMAP_MONTHS[day.month - 1]}-{day.year}'


def _validity(conn, folder):
    status, _ = conn.select(folder, readonly=True)
    if status != 'OK':
        raise RuntimeError(f'{folder}: SELECT failed')
    _, values = conn.response('UIDVALIDITY')
    value = values[0].decode() if values and isinstance(values[0], bytes) else ''
    if not value.isdigit():
        raise RuntimeError(f'{folder}: UIDVALIDITY missing; cannot safely reuse UID history')
    return value


def _fetch(conn, uid, cached, account, folder, validity, parse):
    status, data = conn.uid('fetch', uid, '(FLAGS)' if cached else '(FLAGS BODY.PEEK[])')
    if status != 'OK' or not data:
        raise RuntimeError(f'{folder} UID {uid}: fetch failed')
    headers = [part[0] if isinstance(part, tuple) else part for part in data]
    flags = b' '.join(p for p in headers if isinstance(p, bytes))
    match = re.search(rb'FLAGS\s*\(([^)]*)\)', flags)
    if not match:
        raise RuntimeError(f'{folder} UID {uid}: FLAGS missing')
    if not cached:
        raw = next((p[1] for p in data if isinstance(p, tuple)), None)
        if not raw:
            raise RuntimeError(f'{folder} UID {uid}: body missing')
        cached = store.capture(account, folder, validity, uid, raw,
                               parse(raw, body_max_chars=0))
    return store.observe_flags(cached['id'], b'\\Seen' in match.group(1).split())


def _render(row, scope, include_body, offset, size):
    parsed = dict(row['parsed'])
    body = parsed.pop('body')
    for key in ('body_start', 'body_end', 'body_truncated'):
        parsed.pop(key, None)
    parsed['subject'] = parsed['subject'][:1000]
    parsed['from'] = parsed['from'][:500]
    parsed['date'] = parsed['date'][:200]
    links = parsed['links']
    parsed['links'] = [link for link in links if len(link) <= 1000][:5]
    parsed['links_omitted'] = len(links) - len(parsed['links'])
    start = min(max(0, offset), len(body))
    end = min(len(body), start + size)
    return {
        'mail_id': row['id'], 'folder': row['folder'], 'uid': row['uid'],
        'uidvalidity': row['uidvalidity'], **parsed,
        'imap_read': row.get('imap_seen'), 'flags_observed_at': str(row.get('flags_observed_at')),
        'collected_at': str(row['captured_at']),
        'briefing_delivered': store.is_delivered(row['id'], scope[1]) if scope else None,
        **store.read_state(row['id'], len(body), scope[0] if scope else None),
        'body': body[start:end] if include_body else '',
        'returned_chars': [start, end] if include_body else None,
        'next': ({'mail_id': row['id'], 'body_offset': end, 'body_max_chars': PAGE_MAX,
                  'include_body': True} if include_body and end < len(body) else None),
    }


def render_messages(rows, scope, include_body, offset, size):
    """Fit below the gateway's 50k cap BEFORE recording body exposure."""
    messages = []
    for row in rows:
        page_size = min(size, max(1, 24000 // len(rows)))
        item = _render(row, scope, include_body, offset, page_size)
        while not messages and len(json.dumps([item], ensure_ascii=False)) > 40000 and page_size > 1:
            page_size = max(1, page_size // 2)
            item = _render(row, scope, include_body, offset, page_size)
        if len(json.dumps([*messages, item], ensure_ascii=False)) > 40000:
            item = _render(row, scope, False, offset, page_size)
            item['next'] = {'mail_id': row['id'], 'body_offset': offset,
                            'body_max_chars': PAGE_MAX, 'include_body': True}
            if len(json.dumps([*messages, item], ensure_ascii=False)) > 40000:
                break
        if item['returned_chars'] is not None and scope:
            store.record_read(scope[0], row['id'], *item['returned_chars'])
            item.update(store.read_state(row['id'], row['parsed']['body_chars'], scope[0]))
        messages.append(item)
    return messages


def collect(*, connect, parse, scope, audience, sender_filter='', subject_filter='',
            unread_only=False, unbriefed_only=None, limit=5, include_body=True,
            body_max_chars=12000, body_offset=0, folder='', uid='', mail_id=None):
    account = store.account_key()
    size = max(1, min(PAGE_MAX, int(body_max_chars or PAGE_MAX)))
    offset = max(0, int(body_offset or 0))
    limit = max(1, min(20, int(limit)))
    if mail_id:
        row = store.get_message(int(mail_id), account)
        if not row:
            raise ValueError('Cached mail not found for this account.')
        return {'mode': 'cached_body', 'messages': render_messages([row], scope, include_body, offset, size)}
    requested = folder.lower().strip()
    if requested not in ('', 'inbox', 'junk'):
        raise ValueError("Use folder INBOX, Junk, or omit for both.")
    folders = (['Junk'] if requested == 'junk' else ['INBOX']) if uid else (
        {'inbox': ['INBOX'], 'junk': ['Junk']}.get(requested, ['INBOX', 'Junk']))
    only_new = bool(scope) and not unread_only if unbriefed_only is None else unbriefed_only
    if only_new and not audience:
        raise ValueError('An identified audience is required for unbriefed_only.')
    window = new_mail_window_days() if only_new and not uid else 0
    since = _imap_since(window) if window else None
    conn = connect()
    if conn is None:
        raise RuntimeError('IMAP credentials not configured')
    results, coverage = [], []
    try:
        for box in folders:
            try:
                validity = _validity(conn, box)
                known = {r['uid']: r for r in store.namespace(account, box, validity, audience)}
                if uid:
                    uids = [str(uid)]
                else:
                    criteria = ['UNSEEN' if unread_only else 'ALL'] + (['SINCE', since] if since else [])
                    status, data = conn.uid('search', None, *criteria)
                    if status != 'OK':
                        raise RuntimeError('UID SEARCH failed')
                    uids = [u.decode() for u in (data[0].split() if data and data[0] else [])]
                    if only_new:
                        uids = [u for u in uids if not known.get(u, {}).get('briefed')]
                    uids.reverse()
                candidates = uids[:limit * 5]
                fetched = 0
                matched = 0
                errors = []
                for candidate in candidates:
                    if matched >= limit:
                        break
                    fetched += 1
                    cached = store.get_message(known[candidate]['id'], account) if candidate in known else None
                    try:
                        row = _fetch(conn, candidate, cached, account, box, validity, parse)
                    except RuntimeError as exc:
                        errors.append(str(exc))
                        continue
                    p = row['parsed']
                    if sender_filter.lower() not in p['from'].lower() or subject_filter.lower() not in p['subject'].lower():
                        continue
                    results.append((row, known.get(candidate, {}).get('briefed', False)))
                    matched += 1
                coverage.append({'folder': box, 'uidvalidity': validity, 'eligible_count': len(uids),
                                 'examined_count': fetched, 'unexamined_count': len(uids) - fetched,
                                 'since': since, 'errors': errors})
            except RuntimeError as exc:
                coverage.append({'folder': box, 'error': str(exc)})
    finally:
        conn.logout()
    if all('error' in c for c in coverage):
        raise RuntimeError(json.dumps(coverage, ensure_ascii=False))
    results.sort(key=lambda pair: pair[0]['parsed']['date_sort_timestamp'], reverse=True)
    messages = render_messages([row for row, _ in results[:limit]], scope,
                               include_body, offset if uid else 0, size)
    return {'mode': 'unbriefed' if only_new and not uid else 'mailbox',
            'new_mail_window_days': window or None,
            'history_note': ('Unbriefed means received within the window and without a delivery receipt; '
                             'older mail is history, not new. Set unbriefed_only=false to browse it.'
                             if window else
                             'No delivery receipt means unrecorded, not necessarily never briefed historically.'),
            'coverage': coverage, 'matched_but_not_returned': len(results) - len(messages),
            'messages': messages}


async def check_inbox(**kwargs):
    from mail_runtime import imap
    from provenance.runtime import _wrap_external
    try:
        caller = get_caller()
        scope = await asyncio.to_thread(store.task_scope, caller.task_id)
        result = await asyncio.to_thread(collect, connect=imap.connect, parse=imap.parse_message,
                                        scope=scope, audience=scope[1] if scope else caller.user_id,
                                        **kwargs)
        return _wrap_external(json.dumps(result, ensure_ascii=False), 'imap_inbox')
    except Exception as exc:
        return ToolFailure(f'Mail check failed: {exc}')


async def prepare_mail_briefing(items):
    try:
        scope = await asyncio.to_thread(store.task_scope, get_caller().task_id)
        if not scope:
            raise ValueError('A delegated Telegram task is required.')
        await asyncio.to_thread(store.prepare, *scope, store.account_key(), items)
        return ('Mail summaries prepared for this task. No mail is marked delivered yet. '
                'After successful task completion, the callback sends these exact summaries '
                'and records each successful Telegram delivery. Do not duplicate raw archives.')
    except Exception as exc:
        return ToolFailure(f'Cannot prepare mail briefing: {exc}')


PREPARE_MAIL_BRIEFING_TOOL = {
    'name': 'prepare_mail_briefing',
    'description': 'Select a direct mail-only callback: prepare source-attributed summaries for Telegram delivery on successful task completion. Use when the requested task output is a mail briefing. Read all cached body pages first. Does not send or mark delivered.',
    'input_schema': {'type': 'object', 'properties': {'items': {'type': 'array', 'minItems': 1, 'maxItems': 20,
        'items': {'type': 'object', 'properties': {'mail_id': {'type': 'integer'},
                  'summary': {'type': 'string', 'minLength': 1, 'maxLength': 1800}},
                  'required': ['mail_id', 'summary'], 'additionalProperties': False}}}, 'required': ['items']},
}
