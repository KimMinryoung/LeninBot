#!/usr/bin/env python3
"""Batch-backfill commulingo_history_event_people links, event by event.

For each history event, hands the event card plus the full people roster
(names, epithets, condensed careers) to DeepSeek and asks for the people who
were clearly involved. Proposals are validated against the roster, filtered by
confidence, deduplicated against existing links, and applied with a revision
row per link (changed_by 'operator:claude-code') for post-hoc review.

Runs under a systemd oneshot unit (leninbot-event-backfill.service) because
the DeepSeek and DB credentials only exist in the systemd credstore.

Usage:
  ... commulingo_backfill_event_links.py --dry-run           # propose only
  ... commulingo_backfill_event_links.py                     # apply
  ... commulingo_backfill_event_links.py --events chernobyl  # subset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import query as db_query, execute as db_execute
from secrets_loader import get_secret

MODEL = "deepseek-v4-pro"
CHANGED_BY = "operator:claude-code"
REPORT_DIR = PROJECT_ROOT / "logs" / "commulingo"

# The event page groups people by manner of involvement. The model must classify
# each link into one of these; anything else (or a missing value) becomes the
# neutral "unclassified" bucket for human review — never a silent "target",
# which would libel a mere participant as a victim of the event.
VALID_KINDS = {"leader", "participant", "executor", "target", "opponent", "witness"}
FALLBACK_KIND = "unclassified"

PROMPT = """You are auditing the people-links of one historical event card on a Soviet/revolutionary
history site. Below are the event and the site's full roster of people. List every person from the
roster whose card shows they took part in this event. Judge from the roster line itself; do not
speculate beyond it. Never invent a connection the line does not support: missing a link is fine,
inventing one is not. But do not omit a real one for being ordinary. Ordinary service in a mass
event IS a link, because the reader learns as much from who was there as from who was not, and a
missing link would otherwise be indistinguishable from an absence.

The WEIGHT of an involvement is carried by the length of relation_ko/relation_en, not by whether
the link exists:
  - Decisive involvement (led it, commanded a named formation, made a documented turning act, or
    the event is the centre of this person's card): a short role phrase, as specific as the roster
    line supports. e.g. "백군 총사령관", "제7군 사령관 · 만네르헤임 선 돌파".
  - Ordinary involvement (enlisted, served, was mobilized, held a routine post while it happened):
    the BARE role noun and nothing else, at most a few characters. e.g. "정찰병", "기병학교 생도",
    "군수보급". No verbs, no explanation, no clause. If the roster line says only that they served,
    "붉은 군대 복무" is the whole label.
A generation of Soviet officers served in the Civil War; the page must show that plainly without
dressing each one up as a protagonist.

Never put an em dash (—) in a label: this site does not use them. Join two parts with a middle
dot ( · ) as the rest of the dictionary does, e.g. "제7군 사령관 · 만네르헤임 선 돌파".
A label needing more than one such join is too long for a label; cut it back to the role.

For each person also classify HOW they were involved in THIS event, as one "kind":
  leader      — directed or led the event from the top
  participant — took active part (commanders, officials, organizers, designers, soldiers)
  executor    — security/repression apparatus that carried out the event's coercion
  target      — a victim or target of the event (purged, deported, executed, deposed)
  opponent    — opposed or resisted the event
  witness     — a chronicler or observer (writer, journalist, artist), not a direct actor
The kind is event-specific: the same person may be a target in one event and an
executor in another. Do NOT default to "target" — pick the role they actually played.

EVENT
{event_block}

PEOPLE ALREADY LINKED (do not repeat): {already}

ROSTER (id | names | epithet | group | career)
{roster}

person_id MUST be COPIED VERBATIM from the id column of the ROSTER above. Never build one
from the person's name: the roster's ids do not follow a rule you can guess (some carry a given
name, some do not; transliteration varies). Copy the string, and also echo the roster's Korean
name in name_ko so a copy error can be repaired.

Answer with ONLY a JSON object, no other text:
{{"links": [{{"person_id": "<roster id, copied verbatim>", "name_ko": "<roster Korean name>",
"relation_ko": "<간결한 역할, 예: 진압 지휘>",
"relation_en": "<same in English>", "kind": "<leader|participant|executor|target|opponent|witness>",
"confidence": <0.0-1.0>, "reason": "<one short sentence>"}}]}}
Use an empty list when nobody qualifies."""


def normalize_label(value) -> str:
    """Strip an em dash out of a relation label.

    The site does not use em dashes in written content, and the model puts them
    in anyway: 56 stored labels carried one by 2026-08-02, 7 of them from that
    day's civil-war pass. The prompt now forbids it, but a prompt rule is a
    request and this is the boundary, so normalize here too. ' — ' becomes the
    middle dot the rest of the dictionary joins with; a bare '—' becomes a
    space, since it was standing in for a break rather than a join.
    """
    text = str(value or "").strip()
    text = text.replace(" — ", " · ").replace("—", " ")
    return " ".join(text.split())


def resolve_person_id(pid: str, name_ko: str, roster: list[dict]) -> tuple[str, str] | None:
    """Repair an id the model constructed instead of copying, or give up.

    Two real misses on 2026-08-02, both id-shaped rather than wrong people:
    'sergei-kruglov' for roster 'sergey-kruglov' (transliteration), and
    'boris-ponomarev' for roster 'ponomarev' (a given name the roster id does
    not carry). Both were correct identifications thrown away over spelling.

    Only unambiguous repairs are accepted. A name matching two roster people,
    or a suffix matching two ids, stays rejected: a wrong link is far worse
    than a missing one, which is the rule the prompt itself states.
    """
    ids = {p["id"] for p in roster}
    if pid in ids:
        return pid, "exact"

    name = (name_ko or "").strip()
    if name:
        by_name = [p["id"] for p in roster if (p.get("name_ko") or "").strip() == name]
        if len(by_name) == 1:
            return by_name[0], "name_ko"

    if pid:
        # 'boris-ponomarev' -> 'ponomarev': the model prepended a given name.
        tail = pid.split("-", 1)[-1]
        if tail != pid and tail in ids:
            return tail, "dropped-given-name"
        # 'ponomarev' -> 'boris-ponomarev': or omitted one the roster carries.
        by_suffix = [i for i in ids if i.endswith("-" + pid)]
        if len(by_suffix) == 1:
            return by_suffix[0], "added-given-name"
        # 'sergei-kruglov' -> 'sergey-kruglov': i/y and similar transliteration.
        squashed = pid.replace("-", "").replace("i", "y").replace("j", "y")
        by_translit = [
            i for i in ids
            if i.replace("-", "").replace("i", "y").replace("j", "y") == squashed
        ]
        if len(by_translit) == 1:
            return by_translit[0], "transliteration"
    return None


def fetch_events(only: list[str]) -> list[dict]:
    rows = db_query(
        """SELECT id, period_label, title_ko, title_en, summary_ko, outcome_ko
             FROM commulingo_history_events ORDER BY sort_order"""
    )
    return [r for r in rows if not only or r["id"] in only]


def fetch_roster() -> list[dict]:
    return db_query(
        """SELECT p.id, p.name_ko, p.name_en, p.epithet_ko, p.group_id,
                  COALESCE(LEFT(STRING_AGG(c.period_label || ' ' || c.role_ko, '; '
                    ORDER BY c.sort_order), 200), '') AS career
             FROM commulingo_people p
             LEFT JOIN commulingo_person_career_entries c ON c.person_id = p.id
            GROUP BY p.id, p.name_ko, p.name_en, p.epithet_ko, p.group_id
            ORDER BY p.id"""
    )


def existing_links() -> set[tuple[str, str]]:
    rows = db_query("SELECT event_id, person_id FROM commulingo_history_event_people")
    return {(r["event_id"], r["person_id"]) for r in rows}


def propose(client, event: dict, roster: list[dict], linked_ids: list[str]) -> list[dict]:
    event_block = (
        f"id: {event['id']}\nperiod: {event['period_label']}\n"
        f"title: {event['title_ko']} / {event['title_en']}\n"
        f"summary: {event['summary_ko']}\noutcome: {event['outcome_ko']}"
    )
    roster_lines = "\n".join(
        f"{p['id']} | {p['name_ko']} / {p['name_en']} | {p['epithet_ko']} | {p['group_id']} | {p['career']}"
        for p in roster
    )
    prompt = PROMPT.format(
        event_block=event_block,
        already=", ".join(linked_ids) or "(none)",
        roster=roster_lines,
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    payload = json.loads(resp.choices[0].message.content or "{}")
    links = payload.get("links")
    if not isinstance(links, list):
        raise ValueError(f"model returned no links array for {event['id']}")
    return links


def apply_link(event_id: str, person_id: str, relation_ko: str, relation_en: str, kind: str) -> None:
    kind = kind if kind in VALID_KINDS else FALLBACK_KIND
    row = db_query(
        "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next FROM commulingo_history_event_people WHERE event_id = %s",
        (event_id,),
    )[0]
    db_execute(
        """INSERT INTO commulingo_history_event_people
             (event_id, person_id, sort_order, relation_ko, relation_en, relation_kind)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (event_id, person_id, row["next"], relation_ko, relation_en, kind),
    )
    db_execute(
        """INSERT INTO commulingo_people_revisions (entity_type, entity_id, revision_note, snapshot, changed_by)
           VALUES ('history_event_person', %s, 'event-link backfill (batch, operator-approved)', %s::jsonb, %s)""",
        (
            f"{event_id}/{person_id}",
            json.dumps({"after": {
                "event_id": event_id, "person_id": person_id,
                "relation_ko": relation_ko, "relation_en": relation_en,
                "relation_kind": kind,
            }}, ensure_ascii=False),
            CHANGED_BY,
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="propose and report only, write nothing")
    ap.add_argument("--events", default="", help="comma-separated event ids (default: all)")
    ap.add_argument("--min-confidence", type=float, default=0.8)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between event calls")
    args = ap.parse_args()

    from openai import OpenAI

    api_key = get_secret("DEEPSEEK_API_KEY", "") or ""
    if not api_key:
        print("DEEPSEEK_API_KEY unavailable (run under the systemd unit)", file=sys.stderr)
        return 1
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    only = [e.strip() for e in args.events.split(",") if e.strip()]
    events = fetch_events(only)
    roster = fetch_roster()
    linked = existing_links()
    print(f"[event-links] {len(events)} events, roster {len(roster)}, existing links {len(linked)}", file=sys.stderr)

    report = {"dry_run": args.dry_run, "min_confidence": args.min_confidence,
              "applied": [], "rejected": [], "errors": []}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report_path = REPORT_DIR / f"event_link_backfill_{stamp}{'_dry' if args.dry_run else ''}.json"
    for i, event in enumerate(events):
        linked_ids = sorted(p for (e, p) in linked if e == event["id"])
        try:
            proposals = propose(client, event, roster, linked_ids)
        except Exception as exc:
            report["errors"].append({"event": event["id"], "error": str(exc)})
            print(f"[event-links] {event['id']}: ERROR {exc}", file=sys.stderr)
            continue
        for prop in proposals:
            pid = str(prop.get("person_id") or "").strip()
            kind = str(prop.get("kind") or "").strip().lower()
            entry = {"event": event["id"], "person": pid,
                     "relation_ko": normalize_label(prop.get("relation_ko")),
                     "relation_en": normalize_label(prop.get("relation_en")),
                     "kind": kind if kind in VALID_KINDS else FALLBACK_KIND,
                     "confidence": prop.get("confidence"), "reason": prop.get("reason")}
            try:
                conf = float(prop.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            resolved = resolve_person_id(pid, str(prop.get("name_ko") or ""), roster)
            if resolved and resolved[0] != pid:
                entry["person_id_as_proposed"] = pid
                entry["id_repair"] = resolved[1]
                pid = entry["person"] = resolved[0]
            if resolved is None:
                entry["verdict"] = "rejected: unknown person_id"
            elif (event["id"], pid) in linked:
                entry["verdict"] = "rejected: already linked"
            elif conf < args.min_confidence:
                entry["verdict"] = f"rejected: confidence {conf} < {args.min_confidence}"
            elif not entry["relation_ko"] or not entry["relation_en"]:
                entry["verdict"] = "rejected: missing relation labels"
            else:
                entry["verdict"] = "proposed" if args.dry_run else "applied"
                if not args.dry_run:
                    apply_link(event["id"], pid, entry["relation_ko"], entry["relation_en"], entry["kind"])
                linked.add((event["id"], pid))
                report["applied"].append(entry)
                continue
            report["rejected"].append(entry)
        print(f"[event-links] {event['id']}: +{sum(1 for a in report['applied'] if a['event'] == event['id'])} "
              f"(rejected {sum(1 for r in report['rejected'] if r['event'] == event['id'])})", file=sys.stderr)
        # Checkpoint after every event so a kill/timeout never loses the audit trail.
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if i + 1 < len(events):
            time.sleep(args.sleep)
    print(json.dumps({"applied": len(report["applied"]), "rejected": len(report["rejected"]),
                      "errors": len(report["errors"]), "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
