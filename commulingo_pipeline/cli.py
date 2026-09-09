"""Operator entrypoint. Read commands never initialize schema or invoke an LLM."""
import argparse
import asyncio
import json

from .store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    commands.add_parser('costs')
    commands.add_parser('metrics')
    plan = commands.add_parser('plan')
    plan.add_argument('--apply',action='store_true')
    for name in ('show', 'retry'):
        commands.add_parser(name).add_argument('id', type=int)
    add = commands.add_parser('enqueue')
    add.add_argument('kind', choices=['person','term'])
    add.add_argument('action', choices=['create','update'])
    add.add_argument('target')
    add.add_argument('--topic', required=True)
    add.add_argument('--reason', required=True)
    add.add_argument('--baseline', default='')
    run = commands.add_parser('run')
    run.add_argument('--limit', type=int, default=1)
    run.add_argument('--publish', action='store_true')
    run.add_argument('--job-id',type=int,help='resume only this job, with normal lease and safety checks')
    run.add_argument('--review', action='store_true',
                     help='include independent review while keeping publication disabled')
    tick = commands.add_parser('tick')
    tick.add_argument('--limit',type=int,default=1)
    args = parser.parse_args()
    store = Store()
    if args.command == 'list':
        result = store.list_jobs()
    elif args.command == 'plan':
        from .planner import Planner
        result = Planner(store).plan(apply=args.apply)
    elif args.command == 'costs':
        result = store.costs()
    elif args.command == 'metrics':
        result = store.metrics()
    elif args.command == 'show':
        result = store.detail(args.id)
    elif args.command == 'retry':
        result = {'retried': store.retry(args.id)}
    elif args.command == 'enqueue':
        result = {'id': store.enqueue(**{k:v for k,v in vars(args).items() if k!='command'})}
    else:
        from .engine import Engine
        from .stages import stages
        from .config import load
        config = load()
        if not config['legacy_shared_budget']:
            parser.error('enable legacy_shared_budget after schema migration before running paid stages')
        publish = config['phase']!='draft' if args.command=='tick' else args.publish
        if publish and config['phase']=='draft':
            parser.error('publication requires phase=canary or live after evaluation')
        if publish and not config['term_editorial_service']:
            parser.error('enable the deployed term editorial service before publication')
        if not 1 <= args.limit <= 100:
            parser.error('--limit must be 1..100')
        async def run_batch():
            if args.command=='tick':
                from .planner import Planner
                await asyncio.to_thread(store.reconcile_reviews)
                await asyncio.to_thread(Planner(store).plan,apply=True)
                await asyncio.to_thread(store.expire_sources)
            engine = Engine(store, stages(store),cap=config['daily_cap_usd'],
                            stage_budget=config['stage_budget_usd'],review_fraction=config['review_fraction'])
            return [await engine.run_one(draft_only=not publish,
                        job_id=getattr(args,'job_id',None),
                        allow_review=getattr(args,'review',False)) for _ in range(args.limit)]
        result = asyncio.run(run_batch())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0
