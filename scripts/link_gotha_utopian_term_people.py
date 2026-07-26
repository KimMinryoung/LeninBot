#!/usr/bin/env python3
"""Add the four newly registered Germans to the term entries that already
argue about them.

commulingo_term_people is written as a whole list, so each patch below repeats
the existing person ids in their current order and puts the new ones where they
belong chronologically or by weight of the argument. Existing order was read
from commulingo_term_people before writing.

- state-socialism        wrote about Marx, Engels, Lassalle and Wilhelm
                         Liebknecht in prose and could link none of them; that
                         entry's whole argument is their quarrel.
- state-capitalism       gains Engels (the 'total national capital' passage)
                         and Wilhelm Liebknecht (the 1896 line that this is the
                         name the thing should carry).
- dictatorship-of-the-proletariat  gains Marx: the Gotha Critique holds the
                         most compressed statement of the concept in his work.
- free-association-of-producers    gains Marx and Engels for the same reason
                         the entry contrasts itself with state ownership.

Usage:
  bash scripts/run_commulingo_register.sh scripts/link_gotha_utopian_term_people.py
  bash scripts/run_commulingo_register.sh scripts/link_gotha_utopian_term_people.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIA = "https://www.marxists.org"

UPDATES = [
    {
        "id": "state-socialism",
        "people": [
            "karl-marx", "friedrich-engels", "ferdinand-lassalle",
            "wilhelm-liebknecht", "lenin", "karl-kautsky", "eduard-bernstein",
        ],
        "sources": [
            f"{MIA}/archive/marx/works/1875/gotha/ch03.htm — Marx against state-aided "
            "co-operatives, the Lassallean demand this entry describes",
            f"{MIA}/archive/marx/works/1880/soc-utop/ch03.htm — Engels's footnote calling "
            "Bismarckian state ownership 'a kind of spurious Socialism'",
            f"{MIA}/archive/liebknecht-w/1896/08/our-congress.htm — Wilhelm Liebknecht: "
            "'State Socialism is really State capitalism!'",
            "https://en.wikipedia.org/wiki/Ferdinand_Lassalle — the ADAV and the "
            "state-financed co-operatives from which the word descends",
        ],
    },
    {
        "id": "state-capitalism",
        "people": [
            "friedrich-engels", "wilhelm-liebknecht", "lenin", "trotsky",
            "amadeo-bordiga", "anton-pannekoek", "mikhail-bakunin", "karl-kautsky",
            "mao-zedong", "bukharin", "osinsky", "raya-dunayevskaya",
        ],
        "sources": [
            f"{MIA}/archive/marx/works/1880/soc-utop/ch03.htm — 'The modern state, no matter "
            "what its form, is essentially a capitalist machine… the ideal personification "
            "of the total national capital'",
            f"{MIA}/archive/liebknecht-w/1896/08/our-congress.htm — Wilhelm Liebknecht, 1896: "
            "state socialism 'is really State capitalism'",
        ],
    },
    {
        "id": "dictatorship-of-the-proletariat",
        "people": ["karl-marx", "lenin"],
        "sources": [
            f"{MIA}/archive/marx/works/1875/gotha/ch04.htm — 'Corresponding to this is also a "
            "political transition period in which the state can be nothing but the "
            "revolutionary dictatorship of the proletariat.'",
        ],
    },
    {
        "id": "free-association-of-producers",
        "people": ["karl-marx", "friedrich-engels"],
        "sources": [
            f"{MIA}/archive/marx/works/1875/gotha/ch03.htm — co-operatives 'are of value only "
            "insofar as they are the independent creations of the workers and not protégés "
            "either of the governments or of the bourgeois'",
            f"{MIA}/archive/marx/works/1880/soc-utop/ch03.htm — Engels on state ownership as "
            "no solution of the conflict, the counterpoint this entry rests on",
        ],
    },
]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    for update in UPDATES:
        print(f"{update['id']:<34} people -> {', '.join(update['people'])}")

    if not args.apply:
        print("\ndry run; pass --apply to write")
        return 0

    from runtime_tools.commulingo_people import _exec_commulingo_write

    failed = 0
    for update in UPDATES:
        result = await _exec_commulingo_write(
            "term", "update", update["id"], update["sources"],
            {"people": update["people"]}, 0.95,
        )
        print(f"\n{update['id']}: {result}")
        if result.startswith("Error:") or '"error"' in result:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
