#!/usr/bin/env python3
"""Fill every missing CommuLingo citizenship and national-origin field.

The job is deterministic, restartable, and conservative:
- existing non-empty values are never overwritten;
- missing citizenships require an explicit reviewed entry below;
- national origin defaults to the already-known citizenship, except that Soviet
  citizenship defaults to Russian background and documented exceptions are
  listed explicitly;
- applying uses optimistic WHERE clauses and then requires zero remaining gaps.

Birthplace is deliberately not an input.  ``origin_*`` describes documented
national/ethnic background, not the modern state containing a birthplace.

Usage:
  python scripts/commulingo_backfill_person_nationality.py          # dry run
  python scripts/commulingo_backfill_person_nationality.py --apply # update DB
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PSQL = Path(__file__).resolve().parent / "psql-supabase"

LABELS = {
    "soviet": ("소련", "Soviet Union"), "russia": ("러시아", "Russia"),
    "ukraine": ("우크라이나", "Ukraine"), "georgia": ("그루지야", "Georgia"),
    "armenia": ("아르메니아", "Armenia"), "azerbaijan": ("아제르바이잔", "Azerbaijan"),
    "belarus": ("벨라루스", "Belarus"), "kazakhstan": ("카자흐스탄", "Kazakhstan"),
    "latvia": ("라트비아", "Latvia"), "lithuania": ("리투아니아", "Lithuania"),
    "estonia": ("에스토니아", "Estonia"), "uzbekistan": ("우즈베키스탄", "Uzbekistan"),
    "moldova": ("몰도바", "Moldova"), "turkmenistan": ("투르크메니스탄", "Turkmenistan"),
    "tajikistan": ("타지키스탄", "Tajikistan"), "kyrgyzstan": ("키르기스스탄", "Kyrgyzstan"),
    "poland": ("폴란드", "Poland"), "finland": ("핀란드", "Finland"),
    "germany": ("독일", "Germany"), "east-germany": ("동독", "East Germany"),
    "austria": ("오스트리아", "Austria"), "hungary": ("헝가리", "Hungary"),
    "czechia": ("체코", "Czechia"), "slovakia": ("슬로바키아", "Slovakia"),
    "czechoslovakia": ("체코슬로바키아", "Czechoslovakia"), "korea": ("조선", "Korea"),
    "romania": ("루마니아", "Romania"), "bulgaria": ("불가리아", "Bulgaria"),
    "yugoslavia": ("유고슬라비아", "Yugoslavia"), "france": ("프랑스", "France"),
    "italy": ("이탈리아", "Italy"), "spain": ("스페인", "Spain"),
    "uk": ("영국", "United Kingdom"), "ireland": ("아일랜드", "Ireland"),
    "netherlands": ("네덜란드", "Netherlands"), "usa": ("미국", "United States"),
    "cuba": ("쿠바", "Cuba"), "argentina": ("아르헨티나", "Argentina"),
    "chile": ("칠레", "Chile"), "brazil": ("브라질", "Brazil"),
    "china": ("중국", "China"), "japan": ("일본", "Japan"), "india": ("인도", "India"),
    "turkey": ("튀르키예", "Turkey"), "vietnam": ("베트남", "Vietnam"),
    "north-korea": ("조선민주주의인민공화국", "North Korea"),
    "south-korea": ("대한민국", "South Korea"), "albania": ("알바니아", "Albania"),
    "angola": ("앙골라", "Angola"), "burkina-faso": ("부르키나파소", "Burkina Faso"),
    "congo": ("콩고민주공화국", "DR Congo"), "ghana": ("가나", "Ghana"),
    "guinea-bissau": ("기니비사우", "Guinea-Bissau"), "indonesia": ("인도네시아", "Indonesia"),
    "mozambique": ("모잠비크", "Mozambique"), "peru": ("페루", "Peru"),
    "trinidad": ("트리니다드 토바고", "Trinidad and Tobago"), "portugal": ("포르투갈", "Portugal"),
    "el-salvador": ("엘살바도르", "El Salvador"), "grenada": ("그레나다", "Grenada"),
    "guyana": ("가이아나", "Guyana"), "nicaragua": ("니카라과", "Nicaragua"),
    "south-africa": ("남아프리카 공화국", "South Africa"), "tanzania": ("탄자니아", "Tanzania"),
}

# Every person whose citizenship was blank at the time of the full audit.
# This is intentionally explicit: adding a new incomplete person makes the job
# fail rather than guessing citizenship from birthplace.
CITIZENSHIP_OVERRIDES = {
    "pyotr-krasikov": "soviet",
    "aime-cesaire": "france", "alfred-rosmer": "france", "andre-marty": "france",
    "angela-davis": "usa", "anna-louise-strong": "usa", "arkadi-maslow": "russia",
    "august-thalheimer": "germany", "augusto-cesar-sandino": "nicaragua",
    "carlos-marighella": "brazil", "charles-bettelheim": "france",
    "chris-hani": "south-africa", "clr-james": "trinidad", "dn-aidit": "indonesia",
    "eduard-bernstein": "germany", "ernst-bloch": "germany", "etienne-cabet": "france",
    "farabundo-marti": "el-salvador", "george-padmore": "trinidad",
    "gustav-husak": "czechoslovakia", "harry-pollitt": "uk", "heinrich-brandler": "germany",
    "isaac-deutscher": "poland", "julio-antonio-mella": "cuba",
    "julius-nyerere": "tanzania", "karl-korsch": "germany", "kim-san": "korea",
    "louis-althusser": "france", "luis-carlos-prestes": "brazil",
    "maurice-bishop": "grenada", "nazim-hikmet": "turkey", "paul-levi": "germany",
    "paul-robeson": "usa", "qu-qiubai": "china", "rajani-palme-dutt": "uk",
    "ruth-first": "south-africa", "ruth-fischer": "germany", "santiago-carrillo": "spain",
    "sylvia-pankhurst": "uk", "vo-nguyen-giap": "vietnam", "walter-rodney": "guyana",
    "yeo-un-hyeong": "korea",
    "anatoly-zheleznyakov": "russia", "bogdan-knunyants": "russia",
    "marina-raskova": "soviet", "matvei-shkiryatov": "soviet", "mikhail-vladimirsky": "soviet",
    "anna-pankratova": "soviet", "averky-aristov": "soviet",
    "nikolai-shchelokov": "soviet", "valery-sablin": "soviet",
}

# Only exceptions to origin == citizenship (or Soviet -> Russian) belong here.
# These encode documented family/national identity, never birthplace alone.
ORIGIN_OVERRIDES = {
    "bogdan-knunyants": "armenia",
    "connolly": "ireland",
    "dubcek": "slovakia",
    "gustav-husak": "slovakia",
    "rajani-palme-dutt": "india",
    "ruth-fischer": "austria",
    "victor-serge": "russia",
}


def run_psql(args: list[str], stdin: str | None = None) -> str:
    result = subprocess.run([str(PSQL), *args], input=stdin, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "psql-supabase failed")
    return result.stdout


def fetch_missing() -> list[dict[str, str]]:
    sql = """
SELECT id, name_en, COALESCE(citizenship_code,''), COALESCE(origin_code,'')
FROM commulingo_people
WHERE COALESCE(citizenship_code,'')='' OR COALESCE(origin_code,'')=''
ORDER BY sort_order,id
"""
    rows = []
    for line in run_psql(["-t", "-A", "-F", "\t", "-c", sql]).splitlines():
        if line.strip():
            pid, name, citizenship, origin = line.split("\t")
            rows.append({"id": pid, "name": name, "citizenship": citizenship, "origin": origin})
    return rows


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def plan(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    changes = []
    errors = []
    for row in rows:
        citizenship = row["citizenship"] or CITIZENSHIP_OVERRIDES.get(row["id"], "")
        if not citizenship:
            errors.append(f"{row['id']}: missing citizenship has no reviewed override")
            continue
        if citizenship not in LABELS:
            errors.append(f"{row['id']}: unsupported citizenship code {citizenship!r}")
            continue
        origin = row["origin"] or ORIGIN_OVERRIDES.get(row["id"]) or (
            "russia" if citizenship == "soviet" else citizenship
        )
        if origin not in LABELS:
            errors.append(f"{row['id']}: unsupported origin code {origin!r}")
            continue
        changes.append({
            **row,
            "new_citizenship": citizenship,
            "new_origin": origin,
            "citizenship_reason": "existing" if row["citizenship"] else "reviewed override",
            "origin_reason": (
                "existing" if row["origin"] else
                "documented national/family identity" if row["id"] in ORIGIN_OVERRIDES else
                "Soviet citizen; Russian background default" if citizenship == "soviet" else
                "national background matches citizenship"
            ),
        })
    if errors:
        raise RuntimeError("Audit coverage failed:\n  " + "\n  ".join(errors))
    return changes


def build_sql(changes: list[dict[str, str]]) -> str:
    statements = ["BEGIN;"]
    for item in changes:
        sets = []
        if not item["citizenship"]:
            ko, en = LABELS[item["new_citizenship"]]
            sets += [f"citizenship_code={sql_quote(item['new_citizenship'])}",
                     f"citizenship_label_ko={sql_quote(ko)}", f"citizenship_label_en={sql_quote(en)}"]
        if not item["origin"]:
            ko, en = LABELS[item["new_origin"]]
            sets += [f"origin_code={sql_quote(item['new_origin'])}",
                     f"origin_label_ko={sql_quote(ko)}", f"origin_label_en={sql_quote(en)}"]
        conditions = [f"id={sql_quote(item['id'])}"]
        if not item["citizenship"]:
            conditions.append("COALESCE(citizenship_code,'')=''")
        if not item["origin"]:
            conditions.append("COALESCE(origin_code,'')=''")
        statements.append(
            "UPDATE commulingo_people SET " + ", ".join(sets + ["updated_at=NOW()"]) +
            " WHERE " + " AND ".join(conditions) + ";"
        )
    statements += ["COMMIT;"]
    return "\n".join(statements) + "\n"


def missing_counts() -> tuple[int, int]:
    out = run_psql(["-t", "-A", "-F", "\t", "-c", """
SELECT count(*) FILTER (WHERE COALESCE(citizenship_code,'')=''),
       count(*) FILTER (WHERE COALESCE(origin_code,'')='')
FROM commulingo_people
"""]).strip()
    a, b = out.split("\t")
    return int(a), int(b)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="apply updates; default is dry-run")
    parser.add_argument("--report", type=Path, help="write the full audit plan as JSON")
    args = parser.parse_args()

    rows = fetch_missing()
    changes = plan(rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "people": changes,
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    citizenship_updates = sum(not c["citizenship"] for c in changes)
    origin_updates = sum(not c["origin"] for c in changes)
    print(f"audited={len(changes)} citizenship_updates={citizenship_updates} origin_updates={origin_updates}")
    for c in changes:
        print(f"{c['id']}\t{c['citizenship'] or '-'}->{c['new_citizenship']}\t{c['origin'] or '-'}->{c['new_origin']}\t{c['origin_reason']}")

    if not args.apply:
        print("dry-run only; pass --apply to update the database")
        return 0
    run_psql(["-v", "ON_ERROR_STOP=1"], build_sql(changes))
    citizenship_missing, origin_missing = missing_counts()
    print(f"post_apply citizenship_missing={citizenship_missing} origin_missing={origin_missing}")
    if citizenship_missing or origin_missing:
        raise RuntimeError("post-apply invariant failed: nationality gaps remain")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
