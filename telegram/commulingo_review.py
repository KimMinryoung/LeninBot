"""Owner-only command for unresolved dictionary reviews; no LLM in approvals."""
import asyncio
import json
from aiogram.types import BufferedInputFile
from runtime_tools import commulingo_review_queue as queue
from runtime_tools.commulingo_person_service import call_person_service

HELP = ('/commulingo_review list\n/commulingo_review show 번호\n'
        '/commulingo_review approve 번호 승인 사유\n/commulingo_review reject 번호 반려 사유\n'
        '/commulingo_review retry 번호')


async def cmd_commulingo_review(message, ctx):
    if not message.from_user or not ctx['is_allowed'](message.from_user.id) or message.chat.type!='private':
        return
    parts = (message.text or '').split(maxsplit=3)
    action = parts[1] if len(parts)>1 else 'list'
    try:
        if action=='list':
            rows = await asyncio.to_thread(queue.pending)
            text = '\n'.join(f"#{r['id']} {r['target_id']} · {r['action']} · {r['review_status']}" for r in rows)
            await message.answer((text or '대기 중인 인물 제안이 없습니다.')+'\n\n'+HELP,parse_mode=None)
            return
        if len(parts)<3 or not parts[2].isdigit() or action not in {'show','approve','reject','retry'}:
            await message.answer(HELP,parse_mode=None);return
        sid=int(parts[2])
        row=await asyncio.to_thread(queue.detail,sid)
        if not row:
            await message.answer('인물/상세 절 제안을 찾지 못했습니다.',parse_mode=None);return
        if action=='show':
            current=await asyncio.to_thread(call_person_service,{'command':'read','id':row['target_id'],**({'target':'term'} if row['target_type']=='term' else {})})
            job=row.get('review_job') or {}
            reason=job.get('last_error') or (job.get('decision') or {}).get('reason') or row.get('review_note') or '검토 전'
            fields=row.get('patch_json') or {}
            before=current or {}
            if row['target_type']=='person_section':
                before=next((section for section in before.get('sections',[]) if section['slug']==fields.get('slug')), {})
            changes=[]
            for key,value in fields.items():
                if key in {'expectedRevision','evidence','sources','reviewFlags','id'}:continue
                previous=json.dumps(before.get(key),ensure_ascii=False,default=str)[:140]
                proposed=json.dumps(value,ensure_ascii=False,default=str)[:140]
                changes.append(f"{key}: {previous} → {proposed}")
                if len(changes)>=5:break
            refs='\n'.join(str(ref)[:180] for ref in (row.get('source_refs') or [])[:3])
            caption=(f"#{sid} {row['target_id']} · {row['action']} · {row['status']}\n{reason[:800]}\n\n"
                     +'\n'.join(changes)+'\n출처:\n'+refs+'\n\n전체 원문·변경안·검토 근거는 첨부 JSON에 있습니다.\n'+HELP)[:3900]
            data=json.dumps({'suggestion':row,'current_person':current},ensure_ascii=False,indent=2,default=str).encode()
            await message.answer(caption,parse_mode=None)
            await message.answer_document(BufferedInputFile(data,filename=f'commulingo-review-{sid}.json'))
            return
        if row['status']!='pending':
            await message.answer(f"이미 {row['status']} 처리된 제안입니다.",parse_mode=None);return
        if action=='retry':
            reset=await asyncio.to_thread(queue.retry,sid)
            await message.answer('다음 검토 실행에서 다시 조사합니다.' if reset else '재검토 대상이 아니거나 이미 검토 중입니다.',parse_mode=None)
            return
        if len(parts)<4 or len(parts[3].strip())<5:
            await message.answer('근거를 확인한 승인/반려 사유를 5자 이상 적어 주세요.\n'+HELP,parse_mode=None);return
        result=await asyncio.to_thread(call_person_service,{'command':'review','suggestionId':sid,'approve':action=='approve',**({'target':'term'} if row['target_type']=='term' else {}),
            'note':parts[3].strip(),'changedBy':f'telegram-owner:{message.from_user.id}'})
        await asyncio.to_thread(queue.synchronize)
        await message.answer(f"#{sid} {'승인하여 반영했습니다.' if result['status']=='approved' else '반려했습니다.'}",parse_mode=None)
    except (ValueError,RuntimeError) as exc:
        text=str(exc)
        if 'revision_conflict' in text:
            text='제안 이후 원문이 바뀌어 승인하지 않았습니다. 최신 내용을 확인하고 반려 후 새 제안을 작성해야 합니다.'
        await message.answer(f'처리하지 못했습니다: {text[:2500]}',parse_mode=None)
