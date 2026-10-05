"""Owner-only command for pending dictionary edits; no LLM in approvals.

Suggestions live in CommuLingo, read and decided through its admin MCP
(suggestions_list, editorial_store review). The pipeline's own independent
review runs in the frontend; this command is the owner's manual queue.
"""
import asyncio
import json
from aiogram.types import BufferedInputFile
from commulingo.mcp_client import call_tool
from commulingo.person_service import call_person_service

HELP = ('/commulingo_review list\n/commulingo_review show 번호\n'
        '/commulingo_review approve 번호 승인 사유\n/commulingo_review reject 번호 반려 사유')


def _suggestion(sid: int) -> dict | None:
    items = call_tool('suggestions_list', {'id': str(sid), 'includePatch': True, 'limit': 1})['items']
    return items[0] if items else None


def _pending() -> list[dict]:
    return call_tool('suggestions_list', {'status': 'pending', 'limit': 30})['items']


def _read_target(row: dict):
    target = 'term' if row['target_type'] == 'term' else 'person'
    if row['target_type'] not in {'person', 'person_section', 'term'}:
        return None
    return call_person_service({'command': 'read', 'target': target, 'id': row['target_id']})


async def cmd_commulingo_review(message, ctx):
    if not message.from_user or not ctx['is_allowed'](message.from_user.id) or message.chat.type != 'private':
        return
    parts = (message.text or '').split(maxsplit=3)
    action = parts[1] if len(parts) > 1 else 'list'
    try:
        if action == 'list':
            rows = await asyncio.to_thread(_pending)
            text = '\n'.join(f"#{r['id']} {r['target_type']} {r['target_id']} · {r['action']} · {r['suggested_by']}" for r in rows)
            await message.answer((text or '대기 중인 제안이 없습니다.') + '\n\n' + HELP, parse_mode=None)
            return
        if len(parts) < 3 or not parts[2].isdigit() or action not in {'show', 'approve', 'reject'}:
            await message.answer(HELP, parse_mode=None)
            return
        sid = int(parts[2])
        row = await asyncio.to_thread(_suggestion, sid)
        if not row:
            await message.answer('제안을 찾지 못했습니다.', parse_mode=None)
            return
        if action == 'show':
            current = await asyncio.to_thread(_read_target, row)
            reason = row.get('review_note') or '검토 전'
            fields = row.get('patch_json') or {}
            before = current or {}
            if row['target_type'] == 'person_section':
                before = next((section for section in before.get('sections', []) if section['slug'] == fields.get('slug')), {})
            changes = []
            for key, value in fields.items():
                if key in {'expectedRevision', 'evidence', 'sources', 'reviewFlags', 'id'}:
                    continue
                previous = json.dumps(before.get(key), ensure_ascii=False, default=str)[:140]
                proposed = json.dumps(value, ensure_ascii=False, default=str)[:140]
                changes.append(f"{key}: {previous} → {proposed}")
                if len(changes) >= 5:
                    break
            refs = '\n'.join(str(ref)[:180] for ref in (row.get('source_refs') or [])[:3])
            caption = (f"#{sid} {row['target_id']} · {row['action']} · {row['status']}\n{reason[:800]}\n\n"
                       + '\n'.join(changes) + '\n출처:\n' + refs + '\n\n전체 원문·변경안은 첨부 JSON에 있습니다.\n' + HELP)[:3900]
            data = json.dumps({'suggestion': row, 'current': current}, ensure_ascii=False, indent=2, default=str).encode()
            await message.answer(caption, parse_mode=None)
            await message.answer_document(BufferedInputFile(data, filename=f'commulingo-review-{sid}.json'))
            return
        if row['status'] != 'pending':
            await message.answer(f"이미 {row['status']} 처리된 제안입니다.", parse_mode=None)
            return
        if len(parts) < 4 or len(parts[3].strip()) < 5:
            await message.answer('근거를 확인한 승인/반려 사유를 5자 이상 적어 주세요.\n' + HELP, parse_mode=None)
            return
        result = await asyncio.to_thread(call_person_service, {
            'command': 'review', 'target': row['target_type'], 'suggestionId': str(sid), 'approve': action == 'approve',
            'note': parts[3].strip(), 'changedBy': f'telegram-owner:{message.from_user.id}'})
        await message.answer(f"#{sid} {'승인하여 반영했습니다.' if result['status'] == 'approved' else '반려했습니다.'}", parse_mode=None)
    except (ValueError, RuntimeError) as exc:
        text = str(exc)
        if 'revision_conflict' in text:
            text = '제안 이후 원문이 바뀌어 승인하지 않았습니다. 최신 내용을 확인하고 반려 후 새 제안을 작성해야 합니다.'
        await message.answer(f'처리하지 못했습니다: {text[:2500]}', parse_mode=None)
