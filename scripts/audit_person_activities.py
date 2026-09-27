#!/usr/bin/env python3
"""Read-only Jev review of public person functions; never applies classifications.

Uses the current function criteria, then checks existing source excerpts.
Biography-only candidates remain explicitly ungrounded. JSONL checkpoints can
be resumed; a Markdown report shows changes and uncertain/unsupported cases.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from commulingo.activities import activity_questions, activity_evidence, activity_basis_question, load_catalog
from llm.call_registry import decide_detailed

FEATURE = 'commulingo_classification_audit'
AUDIT_VERSION = 2
PUBLIC_URL = 'https://cyber-lenin.com/en/commulingo/api/people'


def prepare(person, catalog, threshold):
    # Only public biography fields reach Jev. Existing assignments are excluded.
    state = {k: person.get(k) for k in ('name', 'years', 'epithet', 'bio')}
    state['career'] = [{'y': c.get('y'), 'r': c.get('r')} for c in person.get('career', [])]
    evidence = activity_evidence({'evidence': [
        {**e, 'field': 'activities'} for a in person.get('activities', [])
        for e in a.get('evidence', [])]})
    if evidence:
        state['cited_activity_evidence'] = evidence
    stored = (person.get('primaryActivity') or next(
        (a for a in person.get('activities', []) if a.get('primary')), {})).get('functionId')
    fingerprint = hashlib.sha256(json.dumps({
        'version': AUDIT_VERSION, 'script': Path(__file__).read_text(),
        'questions': activity_questions(catalog, evidence), 'catalog': catalog,
        'basis_questions': [activity_basis_question(catalog, evidence, f['id'], person.get('groupId'))
                            for f in catalog['functions']],
        'state': state, 'stored': stored, 'threshold': threshold,
        'profile': profile_fingerprint(),
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return state, evidence, {'id': person['id'], 'name': person.get('name'),
        'stored': stored, 'fingerprint': fingerprint}


def profile_fingerprint():
    from llm.call_registry import resolve
    profile = resolve(FEATURE)
    return {k: str(getattr(profile, k, None)) for k in ('provider', 'model', 'enabled', 'thresholds')}


def judge(person, catalog, threshold, decide=decide_detailed):
    state, evidence, row = prepare(person, catalog, threshold)
    row.update(cost=0.0, calls=0, trace=[])
    def ask(questions, label):
        result = decide(FEATURE, state, questions, label=label)
        row['calls'] += 1
        if result.decision is None:
            raise RuntimeError(result.error or 'Jev unavailable')
        decision = result.decision
        row['cost'] += decision.cost_usd or 0.0
        row['trace'].append({'stage': label, 'model': decision.model, 'answers': decision.answers})
        return decision
    try:
        q = activity_questions(catalog, evidence)['activity_function']
        if not evidence:
            q = {**q, 'instructions': q['instructions'] +
                 ' No external source excerpts are available: choose a provisional candidate from the public biography and career only. This is not source verification.'}
        d = ask({'activity_function': q}, 'function-audit')
        function = d.choice('activity_function')
        if function not in q['criteria']:
            raise ValueError('out-of-catalog function')
        row.update(candidate=function, changed=function != row['stored'],
                   confidence=d.confidence('activity_function') or 0.0)
        if not evidence:
            row['status'] = 'needs_source'
        else:
            state['selected_activity_function'] = function
            q = activity_basis_question(catalog, evidence, function, person.get('groupId'))
            d = ask({'activity_basis': q}, 'function-audit-basis')
            basis = d.choice('activity_basis')
            row['basis_confidence'] = d.confidence('activity_basis') or 0.0
            if basis == 'unsupported':
                row['status'] = 'unsupported'
            elif isinstance(basis, str) and basis.isdigit() and int(basis) < len(evidence):
                row['evidence'] = evidence[int(basis)]
                row['status'] = ('low_confidence' if min(row['confidence'], row['basis_confidence']) < threshold
                                 else 'source_supported_candidate')
            else:
                raise ValueError('invalid evidence selection')
    except Exception as exc:
        row.update(status='error', error=str(exc))
    return row


def checkpoints(path):
    rows = {}
    if path.exists():
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
                continue  # Interrupted final write; preserve all complete records.
            if row.get('status') != 'error':
                rows[(row['id'], row['fingerprint'])] = row
    return rows


def markdown(rows, summary):
    out = ['# 인물 기능 분류 검토', '',
           'Jev 후보이며 승인된 수정안이 아닙니다. 소속은 재분류하지 않습니다.', '',
           f"대상 {len(rows)}명 · 변경 후보 {summary['changed']}명 · 상태 {summary['statuses']}",
           f"이번 실행 {summary['calls_this_run']}회 · 보고된 비용 ${summary['cost_this_run']:.6f} · 재사용 {summary['reused']}명", '']
    for row in sorted(rows, key=lambda r: (not r.get('changed', False), r['status'], r['id'])):
        if not row.get('changed') and row['status'] == 'source_supported_candidate':
            continue
        out += [f"## {row['name']} ({row['id']})", '',
                f"{row['stored']} → {row.get('candidate', '판정 실패')} · {row['status']} · 신뢰도 {row.get('confidence', 0):.3f}",
                f"https://cyber-lenin.com/commulingo/people/{row['id']}", '']
        if row.get('evidence'):
            e = row['evidence']
            out += [f"근거 신뢰도 {row['basis_confidence']:.3f} · {e['source']} · {e['locator']}", '', e['excerpt'], '']
        if row.get('error'):
            out += [row['error'], '']
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path, help='JSONL checkpoint; .md and .summary.json are also written')
    ap.add_argument('--input', type=Path, help='Saved public /api/people JSON; otherwise fetch public API')
    ap.add_argument('--function', default='government', help='Current function ID, or all')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--threshold', type=float, default=.85)
    args = ap.parse_args(argv)
    if not 1 <= args.concurrency <= 16 or not 0 <= args.threshold <= 1 or (args.limit is not None and args.limit < 1):
        ap.error('concurrency must be 1..16, threshold 0..1, limit positive')
    catalog = load_catalog()
    if args.function not in {'all'} | {f['id'] for f in catalog['functions']}:
        ap.error('unknown function')
    if args.input:
        payload = json.loads(args.input.read_text())
    else:
        with urlopen(PUBLIC_URL, timeout=60) as response:
            payload = json.load(response)
    people = sorted(payload['people'], key=lambda p: p['id'])
    people = [p for p in people if args.function == 'all' or prepare(p, catalog, args.threshold)[2]['stored'] == args.function]
    if args.limit:
        people = people[:args.limit]
    previous = checkpoints(args.out)
    if args.out.exists():
        lines = args.out.read_text().splitlines()
        if lines and lines[-1].strip():
            try:
                json.loads(lines[-1])
            except json.JSONDecodeError:
                args.out.write_text('\n'.join(lines[:-1]) + '\n')
    rows, pending = [], []
    for person in people:
        _, _, key = prepare(person, catalog, args.threshold)
        old = previous.get((key['id'], key['fingerprint']))
        if old:
            rows.append(old)
        else:
            pending.append(person)
    summary = {'reused': len(rows), 'calls_this_run': 0, 'cost_this_run': 0.0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Append checkpoints immediately in completion order; do not truncate on resume.
    with args.out.open('a') as file, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        if args.out.stat().st_size:
            file.write('\n')
        futures = [pool.submit(judge, p, catalog, args.threshold) for p in pending]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
            file.flush()
            summary['calls_this_run'] += row['calls']
            summary['cost_this_run'] += row['cost']
            print(f"{len(rows)}/{len(people)} {row['id']}: {row['status']} {row.get('candidate', '')}", flush=True)
    summary.update(people=len(rows), changed=sum(bool(r.get('changed')) for r in rows),
                   statuses=dict(Counter(r['status'] for r in rows)))
    args.out.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    args.out.with_suffix('.md').write_text(markdown(rows, summary))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return int(any(r['status'] == 'error' for r in rows))


if __name__ == '__main__':
    sys.exit(main())
