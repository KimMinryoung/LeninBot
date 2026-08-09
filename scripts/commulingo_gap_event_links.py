#!/usr/bin/env python3
"""Link the person cards an event commissioned back to that event.

`commulingo_person_create` is a finalization tool: a successful call ends the
run. The gap worker was built around that, so a curator that made a card had no
round left to call `commulingo_event_link`, and 206 of the 253 cards ordered by
an event page never linked back to the page that ordered them. Valery
Khodemchuk, the first man killed at Chernobyl, sat outside the Chernobyl event's
people list while eighteen others were in it.

The research is already done and stored: the gap row names the person, the
event, and — in the curator's own words — why that section needed them. What is
missing is only the two short bilingual strings the link carries, so this asks
the model for those and nothing else. It never invents a link: every row comes
from a gap whose event the curator chose.

The worker itself is fixed separately (it links in code after the card lands),
so this is a one-off for the backlog, not a recurring job.

Not to be confused with commulingo_backfill_event_links.py, which audits an
event's roster against the wiki. This one only ever links what the gap queue
already decided.

Usage:
  scripts/commulingo_gap_event_links.py --dry-run --limit 8
  scripts/commulingo_gap_event_links.py --limit 50
  scripts/commulingo_gap_event_links.py            # the whole backlog
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("COMMULINGO_SUGGESTED_BY", "commulingo-gap-event-links")

from bot_config import _deepseek_anthropic_client, _resolve_deepseek_model  # noqa: E402
from db import get_conn  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402
from runtime_tools.commulingo_people import (  # noqa: E402
    _HISTORY_RELATION_KINDS,
    normalize_commulingo_write,
    _run_edit,
)

logger = logging.getLogger("commulingo_backfill_event_links")
# A batch that dies takes every item in it with it, so this stays modest even
# now that thinking is off and the whole budget goes to the reply.
BATCH = 6

PROMPT = """You are filling in the connective tissue of a Korean-language site on Soviet
history. Each item below is a person the site already has a card for, an event the site
already has a page for, and the note the event's own curator wrote when it asked for that
card. The link between them is decided; you are writing only what the link says.

For each item return four short strings and one kind:

  relation_ko / relation_en — the person's position in THIS event, two to six words.
      Korean examples: 첫 희생자, 4호기 야간 근무조장, 진압을 지휘한 내무장관,
      망명 정부의 외무장관. Not a job title in the abstract: what they were to this event.
  note_ko / note_en — ONE sentence on what they did or what happened to them in this
      event. Draw it from the curator's note; do not invent facts that are not there.
      Hard limit: Korean at most 70 characters, English at most 160. This is a caption
      under a name on the event page, not a paragraph: everything that does not fit is
      already on the person's own card. Count before you answer; a note over the limit
      is thrown away and the person stays unlinked.
  kind — exactly one of: {kinds}
      leader (directed it), participant (took part), executor (carried out orders),
      target (it was done to them), opponent (worked against it), witness (recorded or
      observed it, including scholars who later wrote about it).

Writing rules, both languages:
  - No em dash (—) anywhere. Use a comma, a colon, or a new sentence.
  - Korean is 한다체 written prose: '~했다', '~이다'. No 존댓말, no bullet fragments.
  - Write 그루지야, never 조지아. Write 조선민주주의인민공화국 or 조선, never 북한.
  - Use the person's name exactly as the item gives it.
  - The two languages must say the same thing. English is not a gloss of the Korean word
    order; write it as English prose.

Reply with a JSON array and nothing else:
[{{"i": 0, "relation_ko": "...", "relation_en": "...", "note_ko": "...", "note_en": "...", "kind": "participant"}}]

Items:
"""


_PENDING_SQL = """SELECT g.resolved_id AS person_id, g.event_id, g.reason,
                          p.name_ko, p.name_en, p.epithet_ko,
                          e.title_ko AS event_ko, e.title_en AS event_en
                     FROM commulingo_curation_gaps g
                     JOIN commulingo_people p ON p.id = g.resolved_id
                     JOIN commulingo_history_events e ON e.id = g.event_id
                    WHERE g.kind = 'person'
                      AND g.resolved_id <> '' AND g.event_id <> ''
                      AND NOT EXISTS (
                            SELECT 1 FROM commulingo_history_event_people l
                             WHERE l.person_id = g.resolved_id
                               AND l.event_id = g.event_id)
"""


def _select(extra: str, params: tuple) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(_PENDING_SQL + extra, params)
            return [dict(row) for row in cur.fetchall()]


def pending_rows(limit: int) -> list[dict]:
    """Gap-commissioned cards that never linked back to their event."""
    return _select(" ORDER BY g.priority DESC, g.id LIMIT %s", (limit,))


def pending_rows_for(person_id: str, event_id: str) -> list[dict]:
    """The same row for one card, for the worker that just created it."""
    return _select(" AND g.resolved_id = %s AND g.event_id = %s LIMIT 1", (person_id, event_id))


def next_sort_order(event_id: str) -> int:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT COALESCE(MAX(sort_order), 0) AS m FROM commulingo_history_event_people WHERE event_id = %s",
                (event_id,),
            )
            return int(cur.fetchone()["m"]) + 1


async def describe(rows: list[dict], model: str) -> list[dict]:
    items = "\n".join(
        f"--- item {i} ---\n"
        f"person: {row['name_ko']} / {row['name_en']}\n"
        f"card says: {row.get('epithet_ko') or ''}\n"
        f"event: {row['event_ko']} / {row['event_en']} (id {row['event_id']})\n"
        f"why the event asked for this card: {row['reason']}"
        for i, row in enumerate(rows)
    )
    response = await _deepseek_anthropic_client.messages.create(
        model=model,
        max_tokens=12000,
        # DeepSeek V4 thinks by default and the reasoning shares max_tokens with
        # the reply, so a call that only needs text can spend the whole budget
        # deliberating and return an empty text block. llm/call_registry.py
        # disables it for exactly this reason on every one-shot generation; a
        # script that reaches for the client directly has to say so itself.
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": PROMPT.format(kinds=", ".join(_HISTORY_RELATION_KINDS)) + items}],
    )
    raw = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON array in reply (stop_reason={response.stop_reason}): {raw[:200]}")
    out: list[dict] = [{} for _ in rows]
    for entry in json.loads(raw[start:end + 1]):
        index = int(entry.get("i", -1))
        if 0 <= index < len(out):
            out[index] = entry
    return out


def acceptable(entry: dict) -> str:
    """'' if the entry can be written, else why not."""
    for key in ("relation_ko", "relation_en", "note_ko", "note_en"):
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"missing {key}"
        if "—" in value:
            return f"em dash in {key}"
        if "북한" in value or "조지아" in value:
            return f"banned spelling in {key}"
    # The event page caps the caption at 110/250 characters. Thirty-nine of the
    # first backfill's forty failures were the model writing a paragraph, and
    # each one cost a write attempt to discover. Ask here instead.
    if len(entry["note_ko"]) > 110 or len(entry["note_en"]) > 250:
        return f"note too long ({len(entry['note_ko'])}/110 ko, {len(entry['note_en'])}/250 en)"
    if entry.get("kind") not in _HISTORY_RELATION_KINDS:
        return f"kind {entry.get('kind')!r} not in {_HISTORY_RELATION_KINDS}"
    # Calibrated on what is already stored: the longest relation in the table is
    # 64 Korean characters and 164 English, the 95th percentile 30 and 82. A cap
    # tighter than the corpus refuses rows the site itself would accept.
    if len(entry["relation_ko"]) > 64 or len(entry["relation_en"]) > 164:
        return "relation longer than anything else in the table"
    return ""


def write_link(row: dict, entry: dict) -> None:
    patch = {
        "personId": row["person_id"],
        "sortOrder": next_sort_order(row["event_id"]),
        "relationKind": entry["kind"],
        "relation": {"ko": entry["relation_ko"].strip(), "en": entry["relation_en"].strip()},
        "note": {"ko": entry["note_ko"].strip(), "en": entry["note_en"].strip()},
    }
    citations = [f"결손 큐 {row['event_id']} → {row['person_id']}: {row['reason'][:200]}"]
    patch, citations, _confidence, _repairs = normalize_commulingo_write(
        "history_event_person", row["event_id"], patch, citations, None
    )
    result = _run_edit("history_event_person", "create", row["event_id"], patch, citations, None)
    if result.startswith("Error:"):
        raise RuntimeError(result)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = pending_rows(args.limit)
    logger.info("unlinked gap-commissioned cards: %d", len(rows))
    # Flash: the research is already in the gap row and this writes two short
    # captions from it. Nothing here decides what is true.
    model = _resolve_deepseek_model("deepseek_flash")
    written = skipped = 0

    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        try:
            entries = await describe(batch, model)
        except Exception as exc:
            logger.warning("batch %d failed, left alone: %s", start // BATCH, exc)
            skipped += len(batch)
            continue
        for row, entry in zip(batch, entries):
            problem = acceptable(entry)
            if problem:
                logger.warning("skip %s -> %s: %s", row["person_id"], row["event_id"], problem)
                skipped += 1
                continue
            if args.dry_run:
                print(f"{row['event_id']:<28} {row['name_ko']:<18} [{entry['kind']}] "
                      f"{entry['relation_ko']}\n    {entry['note_ko']}")
                written += 1
                continue
            try:
                write_link(row, entry)
                written += 1
            except Exception as exc:
                logger.warning("write failed %s -> %s: %s", row["person_id"], row["event_id"], exc)
                skipped += 1
        logger.info("progress: %d written, %d skipped, %d remaining",
                    written, skipped, len(rows) - start - len(batch))

    logger.info("done: %d written, %d skipped", written, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
