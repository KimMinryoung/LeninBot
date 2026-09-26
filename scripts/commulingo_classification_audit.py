#!/usr/bin/env python3
"""Audit CommuLingo person classification (groupId, role) with a System One model.

For every person (or a sample) Jev is given name, years, citizenship, epithet,
career lines and the bilingual bio, and asked to pick the dictionary group and
the role (a Soviet office or a category). Disagreements with the stored value
at or above the confidence threshold are written as a Markdown report grouped
by stored→judged pair, for a curator to read. Nothing is written to the
dictionary; corrections go through the editorial service by hand.

The criteria carry the editorial rules the operator confirmed on 2026-09-19
(dev_docs/jev_system_one_adoption.md 4.11.1): republic first secretaries are
the nationalities-federal line, regional secretaries the Secretariat line,
ideology-propaganda is for pro-Soviet ideologues only, non-Soviet people never
sit in a Soviet era group, "scholar" means historians of this history; since
2026-09-21 Chinese citizens take a china-* group and the Chinese party-state
categories (CATEGORY_RULES), which no one else is offered. Pairs
the operator accepted as boundary judgements are left out of the report
(ACCEPTED_PAIRS).

    venv/bin/python scripts/commulingo_classification_audit.py [--limit N] [--threshold 0.85]
        [--out logs/commulingo/person_classification_audit_<date>.md]

Needs DB_PASSWORD (service credential env) and the LLM proxy. Cost is about
$0.00012 per person (~2.8k input tokens on the 2026-09-19 run: $0.31 for 2,341).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURE = "commulingo_classification_audit"
from commulingo.classify import build_questions, groups_for_years, offices_allowed, role_scope  # noqa: E402  shared editorial rules

# stored→judged pairs the operator has accepted as boundary judgements; a
# disagreement on these lines is not reported (2026-09-19 decisions).
ACCEPTED_PAIRS = {
    ("nationalities-federal", "party-leadership"), ("nationalities-federal", "head-of-government"),
    ("party-secretariat-cadres", "party-leadership"), ("old-regime", "international-counterrevolutionary"),
    ("comintern", "party-leadership"), ("economic-management", "party-leadership"), ("agriculture", "party-leadership"),
    ("party-leadership", "head-of-government"),
}


def load_people(limit: int | None) -> list[dict]:
    from db import query
    sql = """SELECT p.id, p.name_ko, p.years_label, p.epithet_ko, p.bio_ko, p.bio_en, p.group_id, p.citizenship_code,
                    r.category_id, r.office_id,
                    (SELECT json_agg(json_build_object('t', e.role_ko, 'y', e.period_label) ORDER BY e.sort_order)
                       FROM (SELECT role_ko, period_label, sort_order FROM commulingo_person_career_entries
                             WHERE person_id = p.id ORDER BY sort_order LIMIT 8) e) AS career
             FROM commulingo_people p LEFT JOIN commulingo_person_roles r ON r.person_id = p.id
             WHERE p.bio_ko IS NOT NULL ORDER BY p.id""" + (f" LIMIT {int(limit)}" if limit else "")
    return query(sql)


def state_of(p: dict) -> dict:
    return {"name": p["name_ko"], "years": p["years_label"], "citizenship": p["citizenship_code"], "epithet": p["epithet_ko"],
            "career": [f"{c.get('t')} ({c.get('y')})" for c in (p.get("career") or [])],
            "bio_ko": p["bio_ko"], "bio_en": p["bio_en"]}


def judge(p: dict, catalogs: tuple, decide) -> dict:
    groups, offices, categories = catalogs
    scope = role_scope(p["citizenship_code"])
    groups = groups_for_years(groups, p["years_label"])
    result = decide(FEATURE, state_of(p), build_questions(groups, offices, categories, scope == "soviet", scope=scope),
                    label="classification-audit")
    row = {"id": p["id"], "name": p["name_ko"], "years": p["years_label"],
           "stored": {"group": p["group_id"], "role": p["office_id"] or p["category_id"]}}
    if result.decision is None:
        return {**row, "error": result.error}
    d = result.decision
    return {**row, "cost": d.cost_usd or 0.0, "tokens": d.usage.get("input_tokens"),
            "jev": {k: {"choice": d.choice(k), "conf": round(d.confidence(k) or 0.0, 3),
                        "top": sorted(d.probabilities(k).items(), key=lambda kv: -kv[1])[:3]} for k in ("group", "role")}}


def report(rows: list[dict], threshold: float) -> str:
    ok = [r for r in rows if "error" not in r]
    out = [f"# CommuLingo 인물 분류 감사 — {date.today().isoformat()}", "",
           f"{len(ok)}명 판정, 오류 {len(rows) - len(ok)}건, 비용 ${sum(r['cost'] for r in ok):.3f}. "
           f"conf ≥ {threshold} 불일치만 표시하며, 운영자가 경계 판단으로 인정한 쌍(ACCEPTED_PAIRS)은 제외.", ""]
    for k in ("group", "role"):
        agree = sum(1 for r in ok if r["jev"][k]["choice"] == r["stored"][k])
        dis = [r for r in ok if r["jev"][k]["conf"] >= threshold and r["jev"][k]["choice"] != r["stored"][k]
               and (r["stored"][k], r["jev"][k]["choice"]) not in ACCEPTED_PAIRS]
        out.append(f"## {k}: 일치 {agree}/{len(ok)}, 보고 대상 불일치 {len(dis)}")
        pairs = collections.Counter((r["stored"][k], r["jev"][k]["choice"]) for r in dis)
        for (s, j), n in pairs.most_common():
            out += ["", f"### {n} × `{s}` → `{j}`", "", "| 인물 | conf | 상위 확률 |", "|---|---|---|"]
            for r in sorted((r for r in dis if (r["stored"][k], r["jev"][k]["choice"]) == (s, j)),
                            key=lambda r: -r["jev"][k]["conf"]):
                alt = ", ".join(f"{a} {p:.2f}" for a, p in r["jev"][k]["top"][:2])
                out.append(f"| [{r['id']}](https://cyber-lenin.com/commulingo/people/{r['id']}) {r['name']} ({r['years']}) "
                           f"| {r['jev'][k]['conf']:.2f} | {alt} |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=f"logs/commulingo/person_classification_audit_{date.today().isoformat()}.md")
    args = ap.parse_args()
    from llm.call_registry import decide_detailed
    from commulingo.people import _list_categories, _list_groups, _list_offices
    catalogs = (_list_groups(), _list_offices(), _list_categories())
    people = load_people(args.limit)
    with ThreadPoolExecutor(args.concurrency) as ex:
        rows = list(ex.map(lambda p: judge(p, catalogs, decide_detailed), people))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(report(rows, args.threshold))
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    ok = [r for r in rows if "error" not in r]
    print(f"{len(ok)}/{len(rows)} judged, ${sum(r['cost'] for r in ok):.3f}, report {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
