"""Evaluate citation decisions and policy, or replay saved answers without API calls."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commulingo_pipeline.citation_gate import FEATURE, QUESTIONS, verdict, settings
from llm.call_registry import Decision, decide_detailed


def summarize(records, thresholds):
    available, confident, wrong, false_rejects, missed_rejects = 0, 0, 0, 0, 0
    latencies, costs, models = [], [], set()
    for row in records:
        if row.get('answers') is None:
            continue
        available += 1
        d = Decision(answers=row['answers'], model=row['model'])
        check = verdict(d, thresholds, row.get('stance', 'supports'))
        reject = bool(check.get('reject'))
        false_rejects += reject and not row['expected_reject']
        missed_rejects += not reject and row['expected_reject']
        if (d.confidence('support') or 0) >= thresholds['reject']:
            confident += 1
            wrong += d.choice('support') != row['expected_support']
        latencies.append(row['latency_ms'])
        if row.get('cost_usd') is not None:
            costs.append(row['cost_usd'])
        models.add(row['model'])
    latencies.sort()
    return {
        'samples': len(records), 'available': available, 'unavailable': len(records) - available,
        'models': sorted(models), 'thresholds': thresholds,
        'confident_coverage': confident / len(records) if records else 0,
        'confident_support_accuracy': (confident - wrong) / confident if confident else None,
        'confident_support_errors': wrong, 'false_rejections': false_rejects,
        'missed_rejections_available_only': missed_rejects,
        'latency_p50_ms': latencies[(len(latencies) - 1) // 2] if latencies else None,
        'latency_p95_ms': latencies[math.ceil(len(latencies) * .95) - 1] if latencies else None,
        'reported_or_estimated_cost_usd': sum(costs) if costs else None,
        'cost_known_calls': len(costs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live', action='store_true', help='Call the configured Jev endpoint (billed/audited)')
    mode.add_argument('--replay', type=Path, help='Replay a previous JSON output; no API calls')
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[1] /
                        'tests/fixtures/jev_citations.json')
    args = parser.parse_args()
    if args.live:
        fixtures = json.loads(args.fixtures.read_text())
        records = []
        for case in fixtures:
            result = decide_detailed(FEATURE, case['state'], QUESTIONS, label='citation-eval')
            d = result.decision
            records.append({**case, 'answers': d.answers if d else None,
                            'model': d.model if d else None, 'latency_ms': d.latency_ms if d else None,
                            'cost_usd': d.cost_usd if d else None, 'error_kind': result.error_kind})
    else:
        records = json.loads(args.replay.read_text())['records']
    thresholds = settings()['thresholds']
    report = {'mode': 'live' if args.live else 'replay', 'records': records,
              'summary': summarize(records, thresholds),
              'threshold_sweep': [summarize(records, {**thresholds, 'reject': t})
                                  for t in (.7, .8, .85, .9, .95)]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(any(r.get('answers') is None for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
