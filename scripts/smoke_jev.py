"""Smoke-test TypeSafe Jev (System One) through the call registry.

Runs the registry feature ``system_one_smoke`` (config/llm_call_sites.json)
with a few Korean/English decision questions and prints answers, latency and
cost. Every call is audited like any other one-shot call (llm_audit_log).

    venv/bin/python scripts/smoke_jev.py

The registry entry points at the direct TypeSafe API since 2026-09-19
(provider ``typesafe``, model ``jev-1.13.0``); OpenRouter's Decisions route
remains a standby. An approved run can bypass the proxy with the key in the
environment (never on the command line history of a shared shell):

    TYPESAFE_BASE_URL=https://api.typesafe.ai TYPESAFE_API_KEY=... \\
        venv/bin/python scripts/smoke_jev.py

The samples exercise the shapes the adoption plan needs first: a Korean
spelling-correction verdict (noul), a routing class (choice) and a
citation-support check over a Russian source with a Korean claim (choice).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.call_registry import decide_detailed  # noqa: E402

FEATURE = "system_one_smoke"

SAMPLES = [
    {
        "name": "korean spelling verdict",
        "state": {
            "sentence": "1937년 소브나르콤 의장 몰로토프는 스탈린과 함께 정치국 회의에 참석했다.",
            "auto_correction": {"from": "소브나르콤", "to": "인민위원회의", "rule": "사전 표준 표기"},
        },
        "questions": {
            "changes_referent": {
                "type": "noul",
                "instructions": "The auto-correction replaces the original spelling with a term that "
                                "refers to a DIFFERENT person or institution than the original text meant.",
            },
            "same_institution_standard_form": {
                "type": "noul",
                "instructions": "The corrected term is the standard Korean name for the same "
                                "institution the original term denotes.",
            },
        },
    },
    {
        "name": "task routing",
        "state": {"task": "commulingo 인물 카드 몰로토프의 국적 필드를 그루지야로 고쳐줘"},
        "questions": {
            "routing_class": {
                "type": "choice",
                "instructions": "Which class of work does this task belong to?",
                "criteria": {
                    "public_content_edit": "Editing published dictionary/people/report content in the DB",
                    "code_config_work": "Changing source code, config, scheduler, tests",
                    "research": "Investigating a topic and writing findings",
                    "browser_automation": "Operating a web browser to complete a task",
                },
            },
        },
    },
    {
        "name": "citation support (ru source, ko claim)",
        "state": {
            "claim": {"field": "birth_year", "value": "1890"},
            "quote": "Вячеслав Михайлович Молотов родился 25 февраля (9 марта) 1890 года в слободе "
                     "Кукарка Вятской губернии.",
        },
        "questions": {
            "support": {
                "type": "choice",
                "instructions": "Does the quoted source passage support the claim value?",
                "criteria": {
                    "supports": "The passage states the claimed value for that field",
                    "contradicts": "The passage states a different value for that field",
                    "unrelated": "The passage does not speak to that field",
                },
            },
            "verbatim_value": {
                "type": "noul",
                "instructions": "The quoted passage contains the claimed value itself, not only an inference.",
            },
        },
    },
]


def main() -> int:
    total_cost = 0.0
    failures = 0
    for sample in SAMPLES:
        result = decide_detailed(FEATURE, sample["state"], sample["questions"], label="smoke")
        print(f"\n== {sample['name']}")
        if result.decision is None:
            failures += 1
            print(f"   FAILED [{result.error_kind}] {result.error}")
            continue
        d = result.decision
        total_cost += d.cost_usd or 0.0
        print(f"   model={d.model} latency={d.latency_ms}ms tokens_in={d.usage.get('input_tokens')} "
              f"cost=${(d.cost_usd or 0):.6f}")
        for key, answer in d.answers.items():
            print(f"   {key}: {json.dumps(answer, ensure_ascii=False)}")
    print(f"\n{len(SAMPLES) - failures}/{len(SAMPLES)} ok, total cost ${total_cost:.6f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
