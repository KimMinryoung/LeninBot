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

    async def run_one(self, *, group=None, job_id=None, draft_only=True, allow_review=False):
        job = await asyncio.to_thread(self.store.claim, group=group, job_id=job_id)
        if not job:
            return {'status': 'idle'}
        if draft_only and (job['stage']=='submit' or
                           (job['stage']=='review' and not allow_review)):
            await asyncio.to_thread(self.store.defer, job, 'draft-only execution', seconds=3600, failed=False)
            return {'status': 'draft_ready', 'job_id': job['id']}
        reservation = None
        usage = Usage()
        work = heartbeat = None
        try:
            stage = self.stages[job['stage']]
            if getattr(stage, 'uses_llm', False):
                reservation = await asyncio.to_thread(self.store.reserve,
                    self.stage_budget, lane='review' if job['stage']=='review' else job['kind'],
                    job_id=job['id'], cap=self.cap, review_fraction=self.review_fraction)
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
            return {'status': result.status, 'stage': result.next_stage, 'job_id': job['id'],
                    'cost_usd':usage.tracker.get('total_cost',0)}
        except BudgetUnavailable:
            await asyncio.to_thread(self.store.defer, job, 'daily budget unavailable', seconds=3600, failed=False)
            return {'status': 'budget_deferred', 'job_id': job['id']}
        except LostLease:
            return {'status': 'lease_lost', 'job_id': job['id']}
        except Exception as exc:
            try:
                await asyncio.to_thread(self.store.defer, job, exc,
                                        escalate=job['attempts'] >= 3)
            except LostLease:
                pass
            return {'status': 'error', 'job_id': job['id'], 'error': str(exc)}
        finally:
            for task in (work, heartbeat):
                if task:
                    task.cancel()
            await asyncio.gather(*(t for t in (work, heartbeat) if t), return_exceptions=True)
            if reservation and (usage.complete or not usage.started):
                await asyncio.to_thread(self.store.settle, reservation,
                                        usage.tracker.get('total_cost', 0) if usage.started else 0)
