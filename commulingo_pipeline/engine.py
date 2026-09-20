"""Stage orchestration; workers never carry an entire job in a conversation."""
import asyncio
from dataclasses import dataclass, field

from .store import BudgetUnavailable, LostLease


@dataclass
class Result:
    value: dict
    next_stage: str
    status: str = 'ready'
    delay_seconds: int = 0


@dataclass
class Usage:
    tracker: dict = field(default_factory=dict)
    # Unknown usage after a process/provider failure retains the reservation.
    complete: bool = False
    started: bool = False


class Engine:
    def __init__(self, store, stages, *, cap='3.39', stage_budget='0.20',
                 timeout=480, heartbeat_seconds=30, review_fraction='0.30'):
        self.store, self.stages = store, stages
        self.cap, self.stage_budget = cap, stage_budget
        self.timeout, self.heartbeat_seconds = timeout, heartbeat_seconds
        self.review_fraction = review_fraction

    async def _heartbeat(self, job):
        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            await asyncio.to_thread(self.store.heartbeat, job)

    async def run_batch(self, *, limit=12, max_seconds=1800, job_id=None,
                        draft_only=True, allow_review=False):
        """Resume the same job between committed stages; bound each invocation."""
        deadline = asyncio.get_running_loop().time() + max_seconds
        results = []
        next_job = job_id
        allowed_stages = None
        for _ in range(limit):
            # Give every admitted stage its full timeout, without killing useful
            # research just because the batch is nearing its wall-clock limit.
            if asyncio.get_running_loop().time() + self.timeout > deadline:
                break
            options = {'claim_stages':allowed_stages} if allowed_stages is not None else {}
            result = await self.run_one(job_id=next_job, draft_only=draft_only,
                                        allow_review=allow_review, **options)
            results.append(result)
            if result['status']=='budget_deferred' and job_id is None:
                allowed_stages = ['validate','judge'] + ([] if draft_only else ['submit'])
                if result.get('blocked_stage')!='review' and (not draft_only or allow_review):
                    allowed_stages.append('review')
                next_job = None
                continue
            if result['status'] in {'idle','budget_deferred','draft_ready','lease_lost'}:
                break
            next_job = (result['job_id'] if result['status']=='ready'
                        and (allowed_stages is None or result.get('stage') in allowed_stages) else job_id)
            if job_id is not None and result['status']!='ready':
                break
        # Finish the approved write at the stage-count boundary, never begin
        # another investigation. Keep the normal wall-clock and lease checks.
        if (results and len(results)==limit and not draft_only
                and results[-1].get('status')=='ready'
                and results[-1].get('stage')=='submit'
                and asyncio.get_running_loop().time()+self.timeout<=deadline):
            results.append(await self.run_one(job_id=results[-1]['job_id'],
                draft_only=False, expected_stage='submit'))
        return results

    async def run_one(self, *, group=None, job_id=None, draft_only=True, allow_review=False, expected_stage=None, claim_stages=None):
        claim_options = {'stages':claim_stages} if claim_stages is not None else {}
        job = await asyncio.to_thread(self.store.claim, group=group, job_id=job_id, **claim_options)
        if not job:
            return {'status': 'idle'}
        if expected_stage is not None and job['stage']!=expected_stage:
            await asyncio.to_thread(self.store.defer,job,'stage changed before final write',seconds=0,failed=False)
            return {'status':'stage_changed','job_id':job['id']}
        if draft_only and (job['stage']=='submit' or
                           (job['stage']=='review' and not allow_review)):
            await asyncio.to_thread(self.store.defer, job, 'draft-only execution', seconds=3600, failed=False)
            return {'status': 'draft_ready', 'job_id': job['id']}
        reservation = None
        usage = Usage()
        work = heartbeat = None
        attempt = None
        outcome, next_stage, error = 'error', None, ''
        started_at = asyncio.get_running_loop().time()
        try:
            attempt = await asyncio.to_thread(self.store.start_attempt,job)
            stage = self.stages[job['stage']]
            if getattr(stage, 'uses_llm', False):
                reservation = await asyncio.to_thread(self.store.reserve,
                    self.stage_budget, lane='review' if job['stage']=='review' else job['kind'],
                    job_id=job['id'], cap=self.cap, review_fraction=self.review_fraction)
                await asyncio.to_thread(self.store.link_attempt_budget,attempt,reservation)
            detail = await asyncio.to_thread(self.store.detail, job['id'])
            heartbeat = asyncio.create_task(self._heartbeat(job))
            work = asyncio.create_task(stage(job, detail['artifacts'], usage,
                                            float(self.stage_budget)))
            done, _ = await asyncio.wait({work, heartbeat}, timeout=self.timeout,
                                        return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                heartbeat.result()  # A lost lease stops the LLM stage immediately.
                raise LostLease(str(job['id']))
            if work not in done:
                raise TimeoutError('stage time limit')
            result = work.result()
            await asyncio.to_thread(self.store.finish_stage, job, result.value,
                next_stage=result.next_stage, status=result.status, usage=usage.tracker,
                delay_seconds=result.delay_seconds)
            outcome, next_stage = result.status, result.next_stage
            error = result.value.get('error') or result.value.get('preflight_error') or ''
            return {'status': result.status, 'stage': result.next_stage, 'job_id': job['id'],
                    'completed_stage': job['stage'],
                    'disposition': disposition(job['stage'], result),
                    'cost_usd':usage.tracker.get('total_cost',0)}
        except BudgetUnavailable as exc:
            outcome, error = 'budget_deferred', str(exc)
            reason = str(exc) or 'daily budget unavailable'
            await asyncio.to_thread(self.store.defer, job, reason, seconds=3600, failed=False)
            return {'status': 'budget_deferred', 'disposition':'budget_wait', 'job_id': job['id'], 'reason': reason, 'blocked_stage':job['stage']}
        except asyncio.CancelledError:
            outcome, error = 'cancelled', 'stage cancelled; unsettled usage remains reserved'
            raise
        except LostLease:
            outcome = 'lease_lost'
            return {'status': 'lease_lost', 'job_id': job['id']}
        except Exception as exc:
            error = str(exc)
            try:
                await asyncio.to_thread(self.store.defer, job, exc,
                                        escalate=job['attempts'] >= 3)
            except LostLease:
                pass
            return {'status': 'error', 'disposition':'failed', 'job_id': job['id'], 'error': str(exc)}
        finally:
            for task in (work, heartbeat):
                if task:
                    task.cancel()
            await asyncio.gather(*(t for t in (work, heartbeat) if t), return_exceptions=True)
            try:
                if reservation and (usage.complete or not usage.started):
                    await asyncio.to_thread(self.store.settle, reservation,
                                            usage.tracker.get('total_cost', 0) if usage.started else 0)
            finally:
                if attempt:
                    await asyncio.to_thread(self.store.finish_attempt,attempt,outcome,next_stage,error,
                        asyncio.get_running_loop().time()-started_at,usage.tracker)


def disposition(stage, result):
    if stage == 'submit' and result.value.get('status') == 'approved':
        return 'published'
    if stage == 'judge' and result.status == 'complete':
        return 'no_edit'
    if result.status == 'escalated':
        return 'held'
    if result.status == 'deferred':
        return 'deferred'
    return 'progress'
