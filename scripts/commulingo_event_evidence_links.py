#!/usr/bin/env python3
"""Link people to a mass event on documented evidence, not on the roster line.

commulingo_backfill_event_links.py is a single toolless call: it sees an
epithet plus a clipped career and nothing else, so anyone whose card describes a
different chapter of their life drops out silently. 골로쇼킨 organized the
Romanov execution and his line says Kazakhstan collectivization; 예이젠시테인
joined the Red Army in 1918 and his line says 전함 포툠킨. Flipping the default
(ask who did NOT take part) does not fix that, it only turns missed links into
wrong exclusions: the exclusion pass on 2026-08-02 threw out both of the above
with the reason "documented activity is later", which is a fact about the line,
not about the person.

This pass asks Wikipedia instead. For each candidate it pulls the article and
keeps sentences that name the event's markers inside its years. Those sentences,
and only those, become the evidence a link is written from, and they are quoted
in the revision row. No evidence means the person is left alone: silence stays
"unknown", never "did not take part".

  scripts/commulingo_event_evidence_links.py --event civil-war --dry-run
  scripts/commulingo_event_evidence_links.py --event civil-war
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import query as db_query, execute as db_execute
from secrets_loader import get_secret
from runtime_tools.wiki import _exec_wiki_get
from scripts.commulingo_backfill_event_links import (
    VALID_KINDS, FALLBACK_KIND, normalize_label, resolve_person_id,
)

MODEL = "deepseek-v4-pro"
CHANGED_BY = "operator:claude-code"
REPORT_DIR = PROJECT_ROOT / "logs" / "commulingo"
# Wikimedia asks bots to go serial rather than parallel. Concurrency 8 took
# five 429s over 205 people; concurrency 3 took 251 over 451, so more than half
# the Great Patriotic War cohort was never actually read. One at a time with a
# second between calls is the rate the API is documented to accept, and this
# pass has no deadline.
FETCH_CONCURRENCY = 1
FETCH_DELAY_S = 1.0
FETCH_RETRIES = 3
FETCH_BACKOFF_S = 10.0
LABEL_BATCH = 40

# Per-event: who is even a candidate, and what counts as evidence in the article.
# `groups`/`born_by`/`alive_from` are the deterministic cohort; `markers` and
# `years` are what a sentence must contain to be evidence. Kept narrow on
# purpose: a sentence naming the Red Army in 1935 is not evidence about 1919.
EVENTS = {
    "civil-war": {
        "groups": ("bolshevik", "stalin-era"),
        "born_by": 1900,
        "alive_from": 1918,
        "markers": r"\b(Red Army|Civil War|White Army|Whites|Cheka|Red Guards)\b",
        "years": r"\b(191[7-9]|192[0-2])\b",
        "label_hint": "the Russian Civil War of 1918-1922",
    },
    "great-patriotic-war": {
        # The thaw generation is in: many of them were at the front as young men
        # and the war is the first thing their card can be linked to.
        "groups": ("bolshevik", "stalin-era", "thaw"),
        "born_by": 1925,
        "alive_from": 1941,
        "markers": r"\b(Red Army|Great Patriotic War|World War II|Eastern Front|Soviet Army|"
                   r"Wehrmacht|partisan|Stalingrad|Leningrad blockade|siege of Leningrad)\b",
        "years": r"\b(194[1-5])\b",
        "label_hint": "the Great Patriotic War of 1941-1945",
    },
    "great-terror": {
        # Not a mobilization: here "took part" splits into being purged, running
        # the purge, or presiding over it, and the evidence decides which. The
        # prompt's warning against defaulting to `target` matters most here,
        # because for this event `target` is frequently the true answer and the
        # temptation is to apply it to everyone the years touch.
        "groups": ("bolshevik", "stalin-era"),
        "born_by": 1915,
        "alive_from": 1936,
        "markers": r"\b(Great Purge|Great Terror|NKVD|purge[sd]?|arrested|executed|"
                   r"shot|show trial|rehabilitated|Gulag)\b",
        "years": r"\b(193[6-9])\b",
        "label_hint": "the Great Purge of 1937-1938",
    },
}

PROMPT = """For each person below you are given sentences from their Wikipedia article that
place them in {label_hint}. Write the label the event page will show next to their name.

The WEIGHT of an involvement is carried by the length of the label, not by whether the link
exists:
  - Decisive involvement (commanded a named formation, led a body, made a documented turning
    act): a short role phrase, as specific as the evidence supports. e.g. "제7군 사령관".
  - Ordinary involvement (enlisted, served, held a routine post while it happened): the BARE
    role noun and nothing else, a few characters. e.g. "정찰병", "적군 복무". No verbs, no
    explanation, no clause.
Never use an em dash. Join two parts with a middle dot ( · ) if you must, and a label needing
more than one join is too long.

Also classify HOW they were involved, as one "kind":
  leader | participant | executor | target | opponent | witness
Do not default to "target": that means a victim of the event.

Write only from the evidence given. If the evidence does not actually place the person in the
event, omit them entirely rather than guessing.

Being affected by an event is not taking part in it. Promotion into posts the event emptied,
holding office while it happened, or a career that merely overlaps its years are all reasons to
OMIT the person. On 2026-08-02 this produced two wrong links on the Great Purge, 보즈네센스키
and 즈베레프, both on evidence that read "was promoted rapidly during the Great Purge": they
rose into the vacancies, they did not make them. The tell is that there is nothing to write in
the label except the person's job title. If the label you want to write is a post rather than a
relation to the event, that is the signal to leave them out.

PEOPLE
{people}

Answer with ONLY a JSON object, no other text:
{{"links": [{{"person_id": "<id copied verbatim>", "name_ko": "<Korean name>",
"relation_ko": "<label>", "relation_en": "<same in English>",
"kind": "<one of the six>"}}]}}"""


def cohort(cfg: dict, event_id: str) -> list[dict]:
    return db_query(
        """SELECT p.id, p.name_ko, p.name_en, p.epithet_ko, p.years_label
             FROM commulingo_people p
            WHERE p.group_id = ANY(%(groups)s)
              AND p.years_label ~ '^[0-9]{4}'
              AND substring(p.years_label from '^([0-9]{4})')::int <= %(born_by)s
              AND (substring(p.years_label from '–([0-9]{4})') IS NULL
                   OR substring(p.years_label from '–([0-9]{4})')::int >= %(alive_from)s)
              AND NOT EXISTS (SELECT 1 FROM commulingo_history_event_people e
                               WHERE e.person_id = p.id AND e.event_id = %(event)s)
            ORDER BY p.id""",
        {"groups": list(cfg["groups"]), "born_by": cfg["born_by"],
         "alive_from": cfg["alive_from"], "event": event_id},
    )


async def evidence_for(person: dict, cfg: dict, sem: asyncio.Semaphore) -> dict:
    """Pull the article and keep the sentences that are evidence.

    _exec_wiki_get does not raise: it logs and returns "Wikipedia fetch failed:
    ...". Checking only for an exception counted every throttled fetch as "no
    evidence", which is the one answer this pass must never invent. The first
    run at concurrency 8 took five 429s and recorded them as absences. Detect
    the string, retry with backoff, and report what still failed.
    """
    marker = re.compile(rf"[^.]*{cfg['markers']}[^.]*\.", re.I)
    years = re.compile(cfg["years"])
    text, error = "", None
    for attempt in range(FETCH_RETRIES):
        async with sem:
            try:
                text = await _exec_wiki_get(title=person["name_en"], language="en", max_chars=6000)
            except Exception as exc:  # defensive: the handler is not supposed to raise
                text, error = "", str(exc)[:120]
        if text and not text.startswith("Wikipedia fetch failed"):
            error = None
            break
        error = (text or error or "empty response")[:120]
        text = ""
        await asyncio.sleep(FETCH_BACKOFF_S * (attempt + 1))
    await asyncio.sleep(FETCH_DELAY_S)
    hits = [m.group(0).strip() for m in marker.finditer(text) if years.search(m.group(0))]
    return {**person, "hits": hits[:3], "error": error}


def label_batch(client, cfg: dict, people: list[dict]) -> list[dict]:
    block = "\n\n".join(
        f"{p['id']} | {p['name_ko']} | {p['years_label']} | {p['epithet_ko']}\n"
        + "\n".join(f"  근거: {h}" for h in p["hits"])
        for p in people
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(
            label_hint=cfg["label_hint"], people=block)}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    got = json.loads(resp.choices[0].message.content or "{}").get("links")
    return got if isinstance(got, list) else []


def apply_link(event_id: str, person_id: str, relation_ko: str, relation_en: str,
               kind: str, evidence: list[str]) -> None:
    kind = kind if kind in VALID_KINDS else FALLBACK_KIND
    nxt = db_query(
        "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next FROM commulingo_history_event_people WHERE event_id = %s",
        (event_id,),
    )[0]["next"]
    db_execute(
        """INSERT INTO commulingo_history_event_people
             (event_id, person_id, sort_order, relation_ko, relation_en, relation_kind)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON CONFLICT (event_id, person_id) DO NOTHING""",
        (event_id, person_id, nxt, relation_ko, relation_en, kind),
    )
    db_execute(
        """INSERT INTO commulingo_people_revisions (entity_type, entity_id, revision_note, snapshot, changed_by)
           VALUES ('history_event_person', %s, 'event link from Wikipedia evidence', %s::jsonb, %s)""",
        (f"{event_id}/{person_id}",
         json.dumps({"after": {"event_id": event_id, "person_id": person_id,
                               "relation_ko": relation_ko, "relation_en": relation_en,
                               "relation_kind": kind},
                     "evidence": evidence}, ensure_ascii=False),
         CHANGED_BY),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="civil-war", choices=sorted(EVENTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = EVENTS[args.event]

    people = cohort(cfg, args.event)
    print(f"[evidence] cohort {len(people)}", file=sys.stderr)

    sem = asyncio.Semaphore(FETCH_CONCURRENCY)
    probed = asyncio.run(_probe_all(people, cfg, sem))
    with_ev = [p for p in probed if p["hits"]]
    failed = [p for p in probed if p["error"]]
    print(f"[evidence] wikipedia evidence for {len(with_ev)}, "
          f"none for {len(probed) - len(with_ev)}, fetch errors {len(failed)}", file=sys.stderr)

    from openai import OpenAI
    client = OpenAI(api_key=get_secret("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    by_id = {p["id"]: p for p in with_ev}
    report = {"event": args.event, "dry_run": args.dry_run, "cohort": len(people),
              "with_evidence": len(with_ev), "applied": [], "rejected": []}

    for i in range(0, len(with_ev), LABEL_BATCH):
        batch = with_ev[i:i + LABEL_BATCH]
        for prop in label_batch(client, cfg, batch):
            pid = str(prop.get("person_id") or "").strip()
            resolved = resolve_person_id(pid, str(prop.get("name_ko") or ""), with_ev)
            entry = {"person": pid,
                     "relation_ko": normalize_label(prop.get("relation_ko")),
                     "relation_en": normalize_label(prop.get("relation_en")),
                     "kind": str(prop.get("kind") or "").strip().lower()}
            if resolved and resolved[0] != pid:
                entry["person_id_as_proposed"], entry["id_repair"] = pid, resolved[1]
                pid = entry["person"] = resolved[0]
            if resolved is None:
                entry["verdict"] = "rejected: not in the evidenced set"
            elif not entry["relation_ko"] or not entry["relation_en"]:
                entry["verdict"] = "rejected: missing label"
            else:
                entry["evidence"] = by_id[pid]["hits"]
                entry["verdict"] = "proposed" if args.dry_run else "applied"
                if not args.dry_run:
                    apply_link(args.event, pid, entry["relation_ko"], entry["relation_en"],
                               entry["kind"], entry["evidence"])
                report["applied"].append(entry)
                continue
            report["rejected"].append(entry)
        print(f"[evidence] labelled {min(i + LABEL_BATCH, len(with_ev))}/{len(with_ev)}", file=sys.stderr)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"event_evidence_{args.event}{'_dry' if args.dry_run else ''}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cohort": report["cohort"], "with_evidence": report["with_evidence"],
                      "applied": len(report["applied"]), "rejected": len(report["rejected"]),
                      "report": str(path)}, ensure_ascii=False))
    return 0


async def _probe_all(people: list[dict], cfg: dict, sem: asyncio.Semaphore) -> list[dict]:
    return list(await asyncio.gather(*(evidence_for(p, cfg, sem) for p in people)))


if __name__ == "__main__":
    raise SystemExit(main())
