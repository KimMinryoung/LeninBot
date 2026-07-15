#!/usr/bin/env python3
"""Find candidate person-name spelling variants in CommuLingo card text.

Compares every Korean surname in the dictionary against hangul tokens in card
prose (edit distance 1, or 2 for long surnames) and prints candidates that are
NOT the canonical spelling, NOT a registered alias, and NOT another person's
name. Candidates are for HUMAN review — approve by adding them to
config/commulingo_name_normalization.json; nothing is written automatically.

Usage:
  scripts/commulingo_find_name_variants.py            # report to stdout
  scripts/commulingo_find_name_variants.py --min-count 2
  scripts/commulingo_find_name_variants.py --min-count 2 --notify
      # weekly timer mode: remembers reported candidates in --state and
      # sends a Telegram message only when NEW ones appear
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PSQL = ROOT / "scripts" / "psql-supabase"
CONFIG = ROOT / "config" / "commulingo_name_normalization.json"

HANGUL_RUN = re.compile(r"[가-힣]{3,}")


def run_psql(stdin: str) -> str:
    result = subprocess.run([str(PSQL), "-t", "-A"], input=stdin, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"psql-supabase failed: {result.stderr.strip()}")
    return result.stdout


def query_json(sql: str) -> list:
    out = run_psql(f"SELECT COALESCE(json_agg(t), '[]'::json) FROM ({sql}) t;")
    return json.loads(out.strip() or "[]")


def edit_distance(a: str, b: str, cap: int = 3) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def notify_telegram(message: str) -> bool:
    """Send `message` to the configured Telegram chat (stale-secrets pattern)."""
    sys.path.insert(0, str(ROOT))
    try:
        from secrets_loader import get_secret
    except Exception as e:
        print(f"WARNING: cannot import secrets_loader ({e}); skipping notify", file=sys.stderr)
        return False
    import os
    import urllib.parse
    import urllib.request
    token = get_secret("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping notify",
              file=sys.stderr)
        return False
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as e:
        print(f"WARNING: telegram notify failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-count", type=int, default=1, help="hide candidates seen fewer times")
    ap.add_argument("--state", type=Path, default=ROOT / "data" / "commulingo_variant_scan_state.json",
                    help="JSON file remembering already-reported candidates")
    ap.add_argument("--notify", action="store_true",
                    help="Telegram-notify NEW candidates (vs --state) and update the state")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    known_variants = set(cfg.get("ko") or {})
    blocked = list((cfg.get("blocked") or {}).get("ko") or [])

    people = query_json("SELECT id, name_ko FROM commulingo_people")
    aliases = query_json("SELECT person_id, alias FROM commulingo_person_aliases WHERE lang='ko'")
    texts = query_json(
        """SELECT 'section:' || s.id || ':' || s.person_id AS src, s.body_ko AS txt
             FROM commulingo_person_sections s
            WHERE COALESCE(s.body_ko, '') <> ''
        UNION ALL
          SELECT 'bio:' || p.id, p.bio_ko FROM commulingo_people p
           WHERE COALESCE(p.bio_ko, '') <> ''
        UNION ALL
          SELECT 'career:' || c.id::text, c.role_ko FROM commulingo_person_career_entries c
           WHERE COALESCE(c.role_ko, '') <> ''"""
    )

    # Every exact string that must never be reported as a "variant":
    # canonical name words, registered aliases, known variants, blocked containers.
    canonical_words: set[str] = set()
    surnames: dict[str, set[str]] = defaultdict(set)  # surname -> person ids
    for p in people:
        words = [w for w in (p["name_ko"] or "").split() if len(w) >= 3]
        canonical_words.update(words)
        if words:
            surnames[words[-1]].add(p["id"])
    alias_words = {a["alias"] for a in aliases if len(a["alias"] or "") >= 3}
    skip_exact = canonical_words | alias_words | known_variants

    candidates: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in texts:
        text = row["txt"] or ""
        for b in blocked:
            text = text.replace(b, " ")
        for run in HANGUL_RUN.findall(text):
            for surname in surnames:
                if run.startswith(surname):
                    continue  # canonical + attached particle (예조프로, 샤우먄의)
                max_d = 2 if len(surname) >= 6 else 1
                for plen in (len(surname) - 1, len(surname), len(surname) + 1):
                    if plen < 3 or plen > len(run):
                        continue
                    prefix = run[:plen]
                    if prefix in skip_exact or prefix in surnames:
                        continue
                    d = edit_distance(prefix, surname, cap=max_d)
                    if 1 <= d <= max_d:
                        # Longer canonical words that merely start like the
                        # surname (스탈린그라드 vs 스탈린) are container noise.
                        if any(prefix in w for w in canonical_words if w != surname):
                            continue
                        candidates[(prefix, surname)].append(row["src"])
                        break

    rows = sorted(candidates.items(), key=lambda kv: -len(kv[1]))
    shown: list[tuple[str, str, int, str]] = []
    for (prefix, surname), sources in rows:
        if len(sources) < args.min_count:
            continue
        sample = ", ".join(sources[:4])
        shown.append((prefix, surname, len(sources), sample))
        print(f"{prefix:<12} ≈ {surname:<12} ×{len(sources):<4} {sample}")
    print(f"\n[find-variants] {len(shown)} candidates (of {len(rows)} raw) — review by hand; "
          f"add real misspellings to {CONFIG.name} (and container words to 'blocked')")

    if args.notify:
        seen: set[str] = set()
        if args.state.exists():
            try:
                seen = set(json.loads(args.state.read_text(encoding="utf-8")).get("seen") or [])
            except Exception as e:
                print(f"WARNING: state file unreadable ({e}); treating all as new", file=sys.stderr)
        new = [(p, s, n, src) for p, s, n, src in shown if f"{p}≈{s}" not in seen]
        if new:
            lines = [f"• {p} ≈ {s} (×{n}) — {src}" for p, s, n, src in new[:20]]
            more = f"\n…외 {len(new) - 20}건" if len(new) > 20 else ""
            message = (
                f"📖 CommuLingo 인물 표기 변형 후보 {len(new)}건 (신규)\n"
                + "\n".join(lines) + more
                + "\n\n진짜 오기만 config/commulingo_name_normalization.json에 추가한 뒤 "
                "scripts/commulingo_normalize_names.py 를 실행하세요. "
                "지명·동명이인은 무시(자동으로 다시 알리지 않음)."
            )
            if notify_telegram(message):
                print(f"[find-variants] notified {len(new)} new candidates")
            else:
                # Leave the state untouched so the failed batch re-notifies
                # on the next run instead of being silently lost.
                print("[find-variants] notify failed — state NOT updated", file=sys.stderr)
                return 1
        else:
            print("[find-variants] no new candidates — no notification")
        seen.update(f"{p}≈{s}" for p, s, _, _ in shown)
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(
            json.dumps({"seen": sorted(seen)}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
