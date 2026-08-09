#!/usr/bin/env python3
"""Rewrite CommuLingo copy that uses the em dash, in batches.

The em dash is not this site's punctuation. On 2026-08-09 the only ones in
hand-written text were inside quoted titles (「스페인의 교훈 — 마지막 경고」);
every other one, 3,113 rows of them, came from the curator lanes, which reach
for it in both languages. `_em_dash_problem` in runtime_tools/commulingo_people.py
now stops new ones at the save. This clears the ones already stored.

It is a rewrite, not a substitution. Swapping the character for a comma
mechanically produces Korean nobody would write, so each affected text goes back
through the model with one instruction: say the same thing without the dash.
Batched, because the unit of work is a sentence or two and paying a round trip
per row would cost more than the writing.

Safety:
  - Every write records a revision snapshot, so the whole sweep is reversible.
  - A rewrite is rejected and the row left alone if it still has an em dash
    outside quotes, drifts more than 25% in length, loses a digit that was in
    the original, or introduces 북한/조지아.
  - --dry-run prints before/after and writes nothing. Start there.

Usage:
  scripts/commulingo_strip_em_dashes.py --dry-run --limit 10
  scripts/commulingo_strip_em_dashes.py --target person_sections --limit 200
  scripts/commulingo_strip_em_dashes.py            # everything
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("COMMULINGO_SUGGESTED_BY", "commulingo-em-dash-sweep")

from bot_config import _deepseek_anthropic_client, _resolve_deepseek_model  # noqa: E402
from db import query as db_query, get_conn  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402
from runtime_tools.commulingo_people import _QUOTED_SPAN_RE, _write_revision  # noqa: E402

logger = logging.getLogger("commulingo_strip_em_dashes")
LOCK_PATH = Path("/tmp/leninbot-commulingo-em-dash-sweep.lock")
CHANGED_BY = "commulingo-em-dash-sweep"
# The unit of work is a PARAGRAPH, not a field. An event body runs past 5,000
# characters, so a dozen of them in one call is tens of thousands of characters
# to reproduce and the reply is truncated long before the last one. Rewriting
# paragraph by paragraph bounds every unit to a few hundred characters, and the
# paragraphs with no em dash in them are never sent anywhere: they come out of
# the sweep byte-identical, which is the strongest guarantee available that this
# did not quietly reword the site.
BATCH_CHARS = 5000
BATCH_ITEMS = 12
DIGITS = re.compile(r"\d+")
# The one way a rewrite goes wrong that rereads as an error rather than a style
# choice: a comma left standing in front of a clause-opening conjunction. Checked
# as "introduced by the rewrite", never as an absolute rule, because prose that
# already read that way is not this sweep's business.
COMMA_CONJUNCTION = re.compile(r",\s*(하지만|그러나|그리고|그런데|but|however|and yet)\b")

# (name, table, primary key expression, key columns for the UPDATE, revision entity)
TARGETS = {
    "person_sections": ("commulingo_person_sections", ["person_id", "slug"],
                        ["body_ko", "body_en"], "person_section"),
    "person_moment": ("commulingo_people", ["id"], ["moment_ko", "moment_en"], "person"),
    "person_bio": ("commulingo_people", ["id"], ["bio_ko", "bio_en"], "person"),
    "person_epithet": ("commulingo_people", ["id"], ["epithet_ko", "epithet_en"], "person"),
    "term_definition": ("commulingo_terms", ["id"], ["definition_ko", "definition_en"], "term"),
    "term_body": ("commulingo_terms", ["id"], ["body_ko", "body_en"], "term"),
    "event_people_note": ("commulingo_history_event_people", ["event_id", "person_id"],
                          ["note_ko", "note_en"], "history_event_person"),
    "event_body": ("commulingo_history_events", ["id"], ["body_ko", "body_en"], "history_event"),
}

PROMPT = """Each item below is published encyclopedia copy from a Korean/English site about
Soviet and communist history. Every one uses the em dash (—). Rewrite each so it uses none.

Rules:
- Say exactly the same thing. Do not add, drop, soften or sharpen any claim, date, number,
  name or quotation.
- Replace the dash with what the sentence actually needs: a comma, a colon, parentheses, or
  a full stop and a new sentence. Korean must read as Korean, not as a patched string.
- Never leave a comma in front of a conjunction that starts a new clause (하지만, 그러나,
  그리고, 그런데, but, however, and yet). Those take a full stop and a new sentence.
- Prefer not to nest parentheses. If the material inside the dashes already contains a
  bracket, an apposition (…인 …) or a new sentence usually reads better than wrapping it
  again. Nest them when that genuinely is the clearest form.
- Keep the register, the length and any markdown exactly as they are.
- An em dash INSIDE a quoted title (「…」, 『…』, "…") is part of that work's name. Leave those
  alone and rewrite only the dashes outside quotation.
- When the dash separates a quotation from its ATTRIBUTION (who said it, where, when), a
  comma is wrong. Close the quotation, then give the attribution as its own sentence, or
  put it in parentheses. Never leave a bare comma sitting after a closing quotation mark.
  「…이해하지 못했다고 말했습니다.」 — 미코얀에게 답하며 (1956년 면담)
  becomes 「…이해하지 못했다고 말했습니다.」 1956년 면담에서 미코얀에게 답한 말이다.
- Change nothing else. If a text has no em dash outside quotation, return it unchanged.

Return ONLY a JSON array of objects: [{"i": <the item's number>, "text": "<the rewrite>"}]
with one entry per item, no commentary.

ITEMS:
"""


def violating(text: str | None) -> bool:
    return bool(text) and "—" in _QUOTED_SPAN_RE.sub("", text)


def fetch_rows(target: str, limit: int) -> list[dict]:
    table, keys, cols, _entity = TARGETS[target]
    where = " OR ".join(f"{c} LIKE '%—%'" for c in cols)
    rows = db_query(
        f"SELECT {', '.join(keys)}, {', '.join(cols)} FROM {table} WHERE {where} "
        f"ORDER BY {', '.join(keys)}"
    )
    out = []
    for row in rows:
        if any(violating(row[c]) for c in cols):
            out.append(dict(row))
        if limit and len(out) >= limit:
            break
    return out


def acceptable(before: str, after: str) -> str | None:
    """Why the rewrite must be refused, or None."""
    if not after or not after.strip():
        return "empty"
    if violating(after):
        return "still has an em dash outside quotation"
    ratio = len(after) / max(1, len(before))
    if not 0.75 <= ratio <= 1.25:
        return f"length drifted {ratio:.2f}x"
    lost = set(DIGITS.findall(before)) - set(DIGITS.findall(after))
    if lost:
        return f"dropped number(s) {sorted(lost)}"
    if "북한" in after and "북한" not in before:
        return "introduced 북한"
    if "조지아" in after and "조지아" not in before:
        return "introduced 조지아"
    if len(COMMA_CONJUNCTION.findall(after)) > len(COMMA_CONJUNCTION.findall(before)):
        return "left a comma in front of a clause-opening conjunction"
    # Nested parentheses are NOT checked here, though the prompt asks the model to
    # avoid them. Refusing a paragraph leaves its em dash in place, so a validator
    # is only worth spending on an actual error: a comma in front of 하지만 is one,
    # a bracket inside a bracket is a preference, and it is sometimes the right
    # call. The good rewrites in the 2026-08-09 samples came from the prompt rule,
    # which is where a preference belongs; the validator only ever subtracted.
    return None


async def rewrite_batch(texts: list[str], model: str) -> list[str]:
    items = "\n".join(
        f"--- item {i} ---\n{text}" for i, text in enumerate(texts)
    )
    response = await _deepseek_anthropic_client.messages.create(
        model=model,
        max_tokens=8000,
        # DeepSeek V4 thinks by default and the reasoning shares max_tokens with
        # the reply, so a call that only needs text can spend the whole budget
        # deliberating and return an empty text block. llm/call_registry.py
        # disables it for exactly this reason on every one-shot generation; a
        # script that reaches for the client directly has to say so itself.
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": PROMPT + items}],
    )
    raw = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON array in reply: {raw[:200]}")
    parsed = json.loads(raw[start:end + 1])
    out = list(texts)
    for entry in parsed:
        index = int(entry.get("i", -1))
        if 0 <= index < len(out) and isinstance(entry.get("text"), str):
            out[index] = entry["text"]
    return out


def apply_row(target: str, row: dict, updates: dict) -> None:
    table, keys, _cols, entity = TARGETS[target]
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            where = " AND ".join(f"{k} = %({k})s" for k in keys)
            cur.execute(
                f"SELECT * FROM {table} WHERE {where}", {k: row[k] for k in keys}
            )
            before = dict(cur.fetchone() or {})
            sets = ", ".join(f"{c} = %({c})s" for c in updates)
            cur.execute(
                f"UPDATE {table} SET {sets} WHERE {where}",
                {**{k: row[k] for k in keys}, **updates},
            )
            _write_revision(
                cur, entity, "/".join(str(row[k]) for k in keys),
                f"em dash sweep ({', '.join(sorted(updates))})",
                {"before": before, "after": updates}, CHANGED_BY,
            )


def batches(units: list) -> list[list]:
    """Group work units so no call carries more text than it can reproduce."""
    out, current, size = [], [], 0
    for unit in units:
        text_len = len(unit[-1])
        if current and (size + text_len > BATCH_CHARS or len(current) >= BATCH_ITEMS):
            out.append(current)
            current, size = [], 0
        current.append(unit)
        size += text_len
    if current:
        out.append(current)
    return out


async def sweep(target: str, limit: int, dry_run: bool) -> dict:
    _table, keys, cols, _entity = TARGETS[target]
    rows = fetch_rows(target, limit)
    # Flash, not Pro. The unit of work is "say this paragraph again without the
    # dash", which is a rewrite with one correct shape rather than a judgement,
    # and acceptable() rejects anything that drifts. Pro's price is for prose
    # nobody has written yet.
    model = _resolve_deepseek_model("deepseek_flash")
    stats = {"target": target, "rows": len(rows), "updated": 0,
             "paragraphs": 0, "refused": 0}

    # (row, col, paragraph index, paragraph text) for every paragraph that
    # actually offends, across both languages.
    units = []
    split: dict[tuple, list[str]] = {}
    # How many paragraphs each row is still waiting on. A row is written back as
    # soon as it hits zero rather than at the end of the target: person_sections
    # is 1,222 rows and would otherwise spend two hours holding every rewrite in
    # memory, with a crash costing all of it.
    outstanding: dict[int, int] = {}
    for row in rows:
        for col in cols:
            text = row.get(col) or ""
            if not violating(text):
                continue
            parts = text.split("\n\n")
            split[(id(row), col)] = parts
            for index, part in enumerate(parts):
                if violating(part):
                    units.append((row, col, index, part))
                    outstanding[id(row)] = outstanding.get(id(row), 0) + 1

    def flush(row) -> None:
        updates = {}
        for col in cols:
            parts = split.get((id(row), col))
            if not parts:
                continue
            rebuilt = "\n\n".join(parts)
            if rebuilt != (row.get(col) or ""):
                updates[col] = rebuilt
        if updates:
            apply_row(target, row, updates)
            stats["updated"] += 1

    for chunk in batches(units):
        try:
            rewrites = await rewrite_batch([unit[3] for unit in chunk], model)
        except Exception as exc:
            # A failed batch leaves its paragraphs exactly as they were; the rows
            # still flush below, writing nothing, and the em dashes survive for a
            # later pass rather than the run dying here.
            rewrites = [unit[3] for unit in chunk]
            logger.warning("batch failed, leaving it alone: %s", exc)
            stats["refused"] += len(chunk)
        for (row, col, index, before), after in zip(chunk, rewrites):
            key = "/".join(str(row[k]) for k in keys)
            problem = None if after == before else acceptable(before, after)
            if problem:
                stats["refused"] += 1
                logger.info("refused %s.%s[%d]: %s", key, col, index, problem)
            elif after != before:
                stats["paragraphs"] += 1
                if dry_run:
                    print(f"\n--- {key} .{col} ¶{index}")
                    print(f"BEFORE: {before[:400]}")
                    print(f"AFTER : {after[:400]}")
                else:
                    split[(id(row), col)][index] = after
            outstanding[id(row)] -= 1
            if outstanding[id(row)] == 0 and not dry_run:
                flush(row)
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="", choices=[""] + list(TARGETS))
    parser.add_argument("--limit", type=int, default=0, help="Rows per target; 0 for all.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lock_file = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.info("another sweep is active; exiting")
        return 0

    targets = [args.target] if args.target else list(TARGETS)
    results = [asyncio.run(sweep(t, args.limit, args.dry_run)) for t in targets]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
