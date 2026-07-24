#!/usr/bin/env python3
"""Find candidate spelling variants of CommuLingo names in site text.

Compares every Korean canonical word from the dictionaries — person surnames,
history-event title words, and glossary-term names/aliases — against WHOLE
words in card prose and public research reports (an 어절 with at most one trailing particle stripped) and prints
candidates that are NOT the canonical spelling, NOT a registered alias, and NOT
another person's name. Candidates are for HUMAN review — approve by adding them
to config/commulingo_name_normalization.json; nothing is written automatically.

Two filters keep precision usable; without them ~95% of output was noise:
  - word boundary: a candidate must be a whole word, never an arbitrary cut of
    a longer one (페테르부르크 must not report 페테르 ≈ 페테르스).
  - jamo distance: one differing syllable is a weak signal in hangul, since
    wholly unrelated syllables are one edit apart (게릴라 ≈ 게바라). Real
    transliteration variants differ by a jamo or two (게르첸/헤르첸), so the
    difference is re-measured on decomposed 초/중/종성.

Usage:
  scripts/commulingo_find_name_variants.py            # report to stdout
  scripts/commulingo_find_name_variants.py --min-count 2
  scripts/commulingo_find_name_variants.py --max-jamo 2   # looser, noisier
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

CHOSUNG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSUNG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

# Trailing particles stripped (at most one) to recover the bare word. Longest
# first so 에게서 wins over 에. Deliberately conservative: a wrong strip
# re-creates the mid-word cutting this filter exists to prevent.
PARTICLES = (
    "에게서", "으로서", "으로써", "에서는", "에게는", "이라고", "라고는",
    "에서", "에게", "께서", "으로", "라고", "이라", "이고", "부터", "까지",
    "조차", "마저", "처럼", "보다", "만이", "만을", "이나", "이란", "이는",
    "은", "는", "이", "가", "을", "를", "의", "에", "도", "와", "과", "로",
    "만", "랑", "나", "야", "여", "께",
)


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


def to_jamo(word: str) -> str:
    """Decompose hangul syllables into 초/중/종성 so distance measures sound."""
    out: list[str] = []
    for ch in word:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(CHOSUNG[code // 588])
            out.append(JUNGSUNG[(code % 588) // 28])
            jong = JONGSUNG[code % 28]
            if jong != " ":
                out.append(jong)
        else:
            out.append(ch)
    return "".join(out)


def word_forms(run: str) -> list[str]:
    """The hangul run itself, plus the form with one trailing particle removed."""
    forms = [run]
    for p in PARTICLES:
        if run.endswith(p) and len(run) - len(p) >= 3:
            forms.append(run[: -len(p)])
            break
    return forms


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
    ap.add_argument("--max-jamo", type=int, default=1,
                    help="max 초/중/종성-level difference between candidate and name")
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
    events = query_json("SELECT id, title_ko FROM commulingo_history_events")
    terms = query_json(
        """SELECT t.id, t.term_ko, COALESCE(a.aliases, '') AS aliases
             FROM commulingo_terms t
             LEFT JOIN (SELECT term_id, string_agg(alias, ' ') AS aliases
                          FROM commulingo_term_aliases WHERE lang='ko' GROUP BY term_id) a
               ON a.term_id = t.id"""
    )
    texts = query_json(
        """SELECT 'section:' || s.id || ':' || s.person_id AS src, s.body_ko AS txt
             FROM commulingo_person_sections s
            WHERE COALESCE(s.body_ko, '') <> ''
        UNION ALL
          SELECT 'bio:' || p.id, p.bio_ko FROM commulingo_people p
           WHERE COALESCE(p.bio_ko, '') <> ''
        UNION ALL
          SELECT 'career:' || c.id::text, c.role_ko FROM commulingo_person_career_entries c
           WHERE COALESCE(c.role_ko, '') <> ''
        UNION ALL
          SELECT 'event:' || e.id, COALESCE(e.summary_ko, '') || ' ' || COALESCE(e.outcome_ko, '')
             FROM commulingo_history_events e
        UNION ALL
          SELECT 'termdef:' || t.id, t.definition_ko FROM commulingo_terms t
           WHERE COALESCE(t.definition_ko, '') <> ''
        UNION ALL
          SELECT 'report:' || r.slug, r.markdown FROM research_documents r
           WHERE r.status = 'public' AND COALESCE(r.markdown, '') <> ''"""
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
    # Event title words and glossary names/aliases join the comparison targets:
    # a near-miss of 헝가리/네프맨 in prose is as much a variant as a surname miss.
    for e in events:
        cleaned = (e["title_ko"] or "").replace("(", " ").replace(")", " ")
        for word in [w for w in cleaned.split() if len(w) >= 3]:
            canonical_words.add(word)
            surnames[word].add("event:" + e["id"])
    for t in terms:
        for word in [w for w in f"{t['term_ko'] or ''} {t['aliases'] or ''}".split() if len(w) >= 3]:
            canonical_words.add(word)
            surnames[word].add("term:" + t["id"])
    alias_words = {a["alias"] for a in aliases if len(a["alias"] or "") >= 3}
    skip_exact = canonical_words | alias_words | known_variants

    candidates: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in texts:
        text = row["txt"] or ""
        for b in blocked:
            text = text.replace(b, " ")
        for run in HANGUL_RUN.findall(text):
            for word in word_forms(run):
                # A whole word only — never an arbitrary cut of a longer one.
                if word in skip_exact or word in surnames:
                    continue  # canonical, alias, or already-known variant
                word_jamo = to_jamo(word)
                for surname in surnames:
                    max_d = 2 if len(surname) >= 6 else 1
                    d = edit_distance(word, surname, cap=max_d)
                    if not 1 <= d <= max_d:
                        continue
                    # One differing syllable means little on its own; require the
                    # sounds to be close too (게르첸/헤르첸 yes, 게바라/게릴라 no).
                    if edit_distance(word_jamo, to_jamo(surname),
                                     cap=args.max_jamo) > args.max_jamo:
                        continue
                    # Longer canonical words that merely contain the surname
                    # (스탈린그라드 vs 스탈린) are container noise.
                    if any(word in w for w in canonical_words if w != surname):
                        continue
                    candidates[(word, surname)].append(row["src"])
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
                f"📖 CommuLingo 인물·사건·용어 표기 변형 후보 {len(new)}건 (신규)\n"
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
