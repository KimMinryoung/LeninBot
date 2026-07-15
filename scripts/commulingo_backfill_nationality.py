#!/usr/bin/env python3
"""Bulk-backfill CommuLingo person nationality (citizenship + origin) from Wikidata.

Deterministic and credential-free: reads/writes commulingo_people through the
sanctioned scripts/psql-supabase helper, and resolves each person on Wikidata's
public SPARQL endpoint (no API key). For every person it reads:
  - P27 country of citizenship   -> the primary "citizenship" flag
  - P19 place of birth -> P17 country -> the secondary "origin" flag
Entities are matched by exact English (then Russian/Cyrillic) label + instance
of human (P31=Q5) and disambiguated by birth year against years_label.

Usage:
  python scripts/commulingo_backfill_nationality.py --dry-run        # report only
  python scripts/commulingo_backfill_nationality.py                  # apply
  python scripts/commulingo_backfill_nationality.py --limit 50       # first 50 missing
  python scripts/commulingo_backfill_nationality.py --all            # revisit every person
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PSQL = Path(__file__).resolve().parent / "psql-supabase"
WDQS = "https://query.wikidata.org/sparql"
UA = "BichonWebsite-commulingo/1.0 (https://cyber-lenin.com; minryoung93@gmail.com)"

# Flag codes with a vendored SVG in the frontend (data/commulingo/flag-icons.js),
# and their canonical bilingual labels. citizenship_code / origin_code must be one
# of these keys; anything else is left unset (and reported as an unmapped gap).
FLAG_NAMES = {
    "soviet": ("소련", "Soviet Union"),
    "russia": ("러시아", "Russia"),
    "ukraine": ("우크라이나", "Ukraine"),
    "georgia": ("조지아", "Georgia"),
    "armenia": ("아르메니아", "Armenia"),
    "azerbaijan": ("아제르바이잔", "Azerbaijan"),
    "belarus": ("벨라루스", "Belarus"),
    "kazakhstan": ("카자흐스탄", "Kazakhstan"),
    "latvia": ("라트비아", "Latvia"),
    "lithuania": ("리투아니아", "Lithuania"),
    "estonia": ("에스토니아", "Estonia"),
    "poland": ("폴란드", "Poland"),
    "finland": ("핀란드", "Finland"),
    "germany": ("독일", "Germany"),
    "austria": ("오스트리아", "Austria"),
    "hungary": ("헝가리", "Hungary"),
    "france": ("프랑스", "France"),
    "italy": ("이탈리아", "Italy"),
    "uk": ("영국", "United Kingdom"),
    "usa": ("미국", "United States"),
    "china": ("중국", "China"),
    "netherlands": ("네덜란드", "Netherlands"),
    "bulgaria": ("불가리아", "Bulgaria"),
    "cuba": ("쿠바", "Cuba"),
    "spain": ("스페인", "Spain"),
    "romania": ("루마니아", "Romania"),
    "czechia": ("체코", "Czechia"),
    "east-germany": ("동독", "East Germany"),
    "uzbekistan": ("우즈베키스탄", "Uzbekistan"),
    "moldova": ("몰도바", "Moldova"),
    "turkmenistan": ("투르크메니스탄", "Turkmenistan"),
    "tajikistan": ("타지키스탄", "Tajikistan"),
    "kyrgyzstan": ("키르기스스탄", "Kyrgyzstan"),
    "japan": ("일본", "Japan"),
    "india": ("인도", "India"),
    "turkey": ("튀르키예", "Turkey"),
    "argentina": ("아르헨티나", "Argentina"),
    "chile": ("칠레", "Chile"),
    "north-korea": ("조선", "North Korea"),
    "south-korea": ("대한민국", "South Korea"),
    "vietnam": ("베트남", "Vietnam"),
    "albania": ("알바니아", "Albania"),
    "angola": ("앙골라", "Angola"),
    "burkina-faso": ("부르키나파소", "Burkina Faso"),
    "congo": ("콩고민주공화국", "DR Congo"),
    "ghana": ("가나", "Ghana"),
    "guinea-bissau": ("기니비사우", "Guinea-Bissau"),
    "indonesia": ("인도네시아", "Indonesia"),
    "mozambique": ("모잠비크", "Mozambique"),
    "peru": ("페루", "Peru"),
    "trinidad": ("트리니다드 토바고", "Trinidad and Tobago"),
}

# Wikidata QIDs whose citizenship means "Soviet" for our purposes: the USSR, the
# RSFSR, and the union republics.
SOVIET_QIDS = {
    "Q15180",   # Soviet Union
    "Q2184",    # Russian SFSR
    "Q130229",  # Ukrainian SSR
    "Q206855",  # Byelorussian SSR
    "Q207521",  # Georgian SSR
    "Q207318",  # Armenian SSR
    "Q207522",  # Azerbaijan SSR
    "Q133428",  # Kazakh SSR
    "Q724003",  # Latvian SSR
    "Q725456",  # Lithuanian SSR
    "Q83483",   # Estonian SSR
    "Q214006",  # Moldavian SSR
    "Q131142",  # Uzbek SSR
    "Q170895",  # Turkmen SSR
    "Q173950",  # Tajik SSR
    "Q184759",  # Kirghiz SSR
}

# Country/state QID -> flag code (historical predecessors fold into the modern flag).
COUNTRY_QID = {
    "Q159": "russia", "Q34266": "russia", "Q139319": "russia",  # Russia, Russian Empire, Russian Republic
    "Q2305208": "russia", "Q1187142": "russia",
    "Q230": "georgia", "Q207521": "georgia",
    "Q212": "ukraine", "Q130229": "ukraine", "Q1223840": "ukraine",
    "Q211": "latvia", "Q724003": "latvia",
    "Q37": "lithuania", "Q725456": "lithuania", "Q1120520": "lithuania",
    "Q191": "estonia", "Q83483": "estonia",
    "Q399": "armenia", "Q207318": "armenia",
    "Q227": "azerbaijan", "Q207522": "azerbaijan",
    "Q184": "belarus", "Q206855": "belarus",
    "Q232": "kazakhstan", "Q133428": "kazakhstan",
    "Q36": "poland", "Q207272": "poland", "Q154741": "poland",  # Poland, Congress Poland, Duchy of Warsaw
    "Q33": "finland", "Q34266_fi": "finland",
    "Q183": "germany", "Q43287": "germany", "Q2415901": "germany",  # Germany, German Empire, German Reich
    "Q1206012": "germany", "Q7318": "germany", "Q41304": "germany",  # Weimar, Nazi Germany, Weimar Rep.
    "Q40": "austria", "Q28513": "austria", "Q131964": "austria", "Q533534": "austria",
    "Q28": "hungary", "Q171150": "hungary",
    "Q142": "france", "Q17054": "france", "Q71084": "france",  # France, Martinique, French Third Republic
    "Q38": "italy", "Q172579": "italy",
    "Q145": "uk", "Q174193": "uk", "Q179876": "uk",  # UK, UK of GB & Ireland, Great Britain
    "Q30": "usa",
    "Q148": "china", "Q8733": "china", "Q13426199": "china", "Q29520": "china",  # PRC, Qing, ROC
    "Q55": "netherlands", "Q29999": "netherlands",
    "Q219": "bulgaria",
    "Q241": "cuba",
    "Q29": "spain", "Q29999_es": "spain",
    "Q218": "romania",
    "Q213": "czechia", "Q33946": "czechia", "Q11205": "czechia",  # Czechia, Czechoslovakia
    "Q16957": "east-germany",
    "Q265": "uzbekistan", "Q131142": "uzbekistan",
    "Q217": "moldova", "Q214006": "moldova",
    "Q874": "turkmenistan", "Q170895": "turkmenistan",
    "Q863": "tajikistan", "Q173950": "tajikistan",
    "Q813": "kyrgyzstan", "Q184759": "kyrgyzstan",
    "Q17": "japan",
    "Q668": "india", "Q129286": "india",  # India, British Raj
    "Q43": "turkey", "Q12560": "turkey",  # Turkey, Ottoman Empire
    "Q414": "argentina",
    "Q298": "chile",
    "Q423": "north-korea", "Q423_kp": "north-korea",
    "Q884": "south-korea",
    "Q881": "vietnam", "Q83891": "vietnam",  # Vietnam, North Vietnam (DRV)
    # People's republics / historical citizenship states seen in P27.
    "Q211274": "poland",  # People's Republic of Poland
    "Q846739": "bulgaria", "Q841628": "bulgaria", "Q7842": "bulgaria",  # PR/Kingdom/Principality Bulgaria
    "Q28513_hu": "hungary",
    "Q222": "albania",
    "Q916": "angola", "Q2208280": "angola",  # Angola, People's Republic of Angola
    "Q965": "burkina-faso",  # Burkina Faso (incl. renamed Upper Volta)
    "Q974": "congo", "Q618399": "congo",  # DR Congo, Congo-Léopoldville
    "Q117": "ghana",
    "Q1007": "guinea-bissau",
    "Q252": "indonesia", "Q188161": "indonesia",  # Indonesia, Dutch East Indies
    "Q1029": "mozambique", "Q617078": "mozambique",  # Mozambique, People's Republic of Mozambique
    "Q419": "peru",
    "Q754": "trinidad", "Q116282722": "trinidad",  # Trinidad and Tobago, Crown colony
}


def run_psql(args: list[str], stdin: str | None = None) -> str:
    result = subprocess.run(
        [str(PSQL), *args],
        input=stdin,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"psql-supabase failed: {result.stderr.strip()}")
    return result.stdout


def fetch_people(limit: int, include_all: bool) -> list[dict]:
    where = "" if include_all else "WHERE COALESCE(citizenship_code,'') = ''"
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    sql = (
        "SELECT id, name_en, cyrillic, COALESCE(birth_year::text,'') "
        f"FROM commulingo_people {where} ORDER BY sort_order, id {limit_clause}"
    )
    out = run_psql(["-t", "-A", "-F", "\t", "-c", sql])
    people = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        pid, name_en, cyrillic, birth = parts[0], parts[1], parts[2], parts[3]
        people.append({
            "id": pid.strip(),
            "name_en": name_en.strip(),
            "cyrillic": cyrillic.strip(),
            "birth_year": int(birth) if birth.strip().isdigit() else None,
        })
    return people


def sparql(query: str, retries: int = 4) -> list[dict]:
    data = urllib.parse.urlencode({"query": query}).encode()
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            WDQS,
            data=data,
            headers={"User-Agent": UA, "Accept": "application/sparql-results+json",
                     "Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.load(resp)
            return payload["results"]["bindings"]
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"SPARQL failed after {retries} attempts: {last_err}")


def _qid(binding: dict, key: str) -> str:
    val = binding.get(key, {}).get("value", "")
    return val.rsplit("/", 1)[-1] if val else ""


def wbsearch(name: str, lang: str) -> list[str]:
    """Wikidata entity search -> candidate QIDs, in relevance order. This is
    robust to transliteration variants and label-normalization quirks that exact
    SPARQL rdfs:label matching silently misses (e.g. 'Rosa Luxemburg')."""
    if not name:
        return []
    params = urllib.parse.urlencode({
        "action": "wbsearchentities", "search": name, "language": lang,
        "uselang": lang, "format": "json", "type": "item", "limit": "5",
    })
    req = urllib.request.Request(
        f"https://www.wikidata.org/w/api.php?{params}", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except Exception:
        return []
    return [it["id"] for it in payload.get("search", []) if str(it.get("id", "")).startswith("Q")]


def fetch_claims(qids: list[str]) -> dict[str, dict]:
    """Batched claim fetch for candidate QIDs: human flag, birth year, and the
    citizenship / birthplace-country QID sets."""
    if not qids:
        return {}
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""SELECT ?p ?type ?dob ?cit ?bpCountry WHERE {{
  VALUES ?p {{ {values} }}
  OPTIONAL {{ ?p wdt:P31 ?type . }}
  OPTIONAL {{ ?p wdt:P569 ?dob . }}
  OPTIONAL {{ ?p wdt:P27 ?cit . }}
  OPTIONAL {{ ?p wdt:P19 ?bp . ?bp wdt:P17 ?bpCountry . }}
}}"""
    out: dict[str, dict] = {}
    for b in sparql(query):
        qid = _qid(b, "p")
        if not qid:
            continue
        dob = b.get("dob", {}).get("value", "")
        year = int(dob[:4]) if dob[:4].isdigit() else None
        ent = out.setdefault(qid, {"human": False, "dob": None, "cit": set(), "bp": set()})
        if _qid(b, "type") == "Q5":
            ent["human"] = True
        if ent["dob"] is None and year is not None:
            ent["dob"] = year
        if _qid(b, "cit"):
            ent["cit"].add(_qid(b, "cit"))
        if _qid(b, "bpCountry"):
            ent["bp"].add(_qid(b, "bpCountry"))
    return out


def pick_candidate(cands: list[dict], birth_year: int | None) -> dict | None:
    """Choose the entity for a person: prefer a human whose birth year matches
    (exact, then +-1), else the top human search hit, else the top hit."""
    if not cands:
        return None
    humans = [c for c in cands if c["human"]]
    pool = humans or cands
    if birth_year is not None:
        for c in pool:
            if c["dob"] == birth_year:
                return c
        for c in pool:
            if c["dob"] and abs(c["dob"] - birth_year) <= 1:
                return c
    return pool[0]


def decide_nationality(ent: dict) -> tuple[str, str, list[str]]:
    """Return (citizenship_code, origin_code, unmapped_qids)."""
    unmapped: list[str] = []
    cit_qids = ent["cit"]
    bp_qids = ent["bp"]

    bp_codes = {COUNTRY_QID[q] for q in bp_qids if q in COUNTRY_QID}
    cit_codes = sorted({COUNTRY_QID[q] for q in cit_qids if q in COUNTRY_QID})
    # A competing foreign citizenship is any mapped country other than Russia
    # (Russia/Russian Empire folds into the Soviet story, so it never competes).
    foreign = [c for c in cit_codes if c != "russia"]

    # Citizenship priority for a Soviet-history dictionary:
    #   1. Soviet, if held AND there is no competing foreign citizenship — so a
    #      mere Soviet-army stint can't override a foreign founder's own state
    #      (Kim Il-sung reads 조선), while genuine Soviets born across the old
    #      empire (Dzerzhinsky, a Pole) still read Soviet.
    #   2. Russia / Russian Empire, likewise only when nothing foreign competes,
    #      so pre-1922 figures and émigrés keep their Russian-sphere citizenship.
    #   3. else a citizenship matching the birthplace, then the first mappable one.
    if (cit_qids & SOVIET_QIDS) and not foreign:
        citizenship = "soviet"
    elif (cit_qids & {"Q34266", "Q159", "Q139319"}) and not foreign:
        citizenship = "russia"
    else:
        match = next((c for c in cit_codes if c in bp_codes), None)
        citizenship = match or (foreign[0] if foreign else (cit_codes[0] if cit_codes else ""))
        if not citizenship and (cit_qids & SOVIET_QIDS):
            citizenship = "soviet"
        if not citizenship:
            unmapped += [q for q in cit_qids if q not in COUNTRY_QID and q not in SOVIET_QIDS]

    # Origin: birthplace country -> code.
    origin = ""
    for q in bp_qids:
        if q in COUNTRY_QID:
            origin = COUNTRY_QID[q]
            break
    if not origin and bp_qids:
        unmapped += [q for q in bp_qids if q not in COUNTRY_QID]

    # Drop a redundant origin identical to citizenship (e.g. a plain French citizen
    # born in France); keep it when citizenship is the Soviet umbrella.
    if origin and origin == citizenship:
        origin = ""
    return citizenship, origin, unmapped


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_update(pid: str, citizenship: str, origin: str) -> str:
    cit_ko, cit_en = FLAG_NAMES.get(citizenship, ("", ""))
    org_ko, org_en = FLAG_NAMES.get(origin, ("", ""))
    return (
        "UPDATE commulingo_people SET "
        f"citizenship_code={sql_quote(citizenship)}, "
        f"citizenship_label_ko={sql_quote(cit_ko)}, citizenship_label_en={sql_quote(cit_en)}, "
        f"origin_code={sql_quote(origin)}, "
        f"origin_label_ko={sql_quote(org_ko)}, origin_label_en={sql_quote(org_en)} "
        f"WHERE id={sql_quote(pid)};"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="max people to process (0 = all)")
    ap.add_argument("--batch-size", type=int, default=25, help="candidate QIDs per claim query")
    ap.add_argument("--all", action="store_true", help="revisit every person, not just those missing citizenship")
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between claim-fetch batches")
    ap.add_argument("--search-sleep", type=float, default=0.08, help="seconds between entity searches")
    args = ap.parse_args()

    people = fetch_people(args.limit, args.all)
    print(f"[backfill] {len(people)} people to process", file=sys.stderr)

    updates: list[str] = []
    resolved = citizenship_set = origin_set = 0
    unresolved: list[str] = []
    unmapped_tally: dict[str, int] = {}

    # Phase 1: resolve candidate QIDs per person via entity search (English name,
    # then Cyrillic). Search tolerates transliteration and label quirks.
    cand_qids: dict[str, list[str]] = {}
    for i, p in enumerate(people):
        qids = wbsearch(p["name_en"], "en")
        if not qids and p["cyrillic"]:
            qids = wbsearch(p["cyrillic"], "ru")
        cand_qids[p["id"]] = qids
        time.sleep(args.search_sleep)
        if (i + 1) % 100 == 0:
            print(f"[backfill]   searched {i + 1}/{len(people)}", file=sys.stderr)

    # Phase 2: fetch claims for every distinct candidate entity, in batches.
    uniq = list(dict.fromkeys(q for qs in cand_qids.values() for q in qs))
    print(f"[backfill] fetching claims for {len(uniq)} candidate entities", file=sys.stderr)
    claims: dict[str, dict] = {}
    for start in range(0, len(uniq), args.batch_size):
        claims.update(fetch_claims(uniq[start:start + args.batch_size]))
        time.sleep(args.sleep)

    # Phase 3: pick the matching entity per person and decide nationality.
    for p in people:
        cands = [claims[q] for q in cand_qids[p["id"]] if q in claims]
        ent = pick_candidate(cands, p["birth_year"])
        if not ent:
            unresolved.append(p["id"])
            continue
        resolved += 1
        citizenship, origin, unmapped = decide_nationality(ent)
        for q in unmapped:
            unmapped_tally[q] = unmapped_tally.get(q, 0) + 1
        if not citizenship and not origin:
            continue
        if citizenship:
            citizenship_set += 1
        if origin:
            origin_set += 1
        updates.append(build_update(p["id"], citizenship, origin))
        if args.dry_run:
            print(f"  {p['id']:<26} cit={citizenship or '-':<9} origin={origin or '-'}")

    print(
        f"[backfill] resolved {resolved}/{len(people)} | citizenship set {citizenship_set} | "
        f"origin set {origin_set} | unresolved {len(unresolved)}",
        file=sys.stderr,
    )
    if unmapped_tally:
        top = sorted(unmapped_tally.items(), key=lambda kv: -kv[1])[:25]
        print("[backfill] unmapped country QIDs (add flag+code): " +
              ", ".join(f"{q}×{n}" for q, n in top), file=sys.stderr)
    if unresolved:
        print(f"[backfill] unresolved ids ({len(unresolved)}): " + ", ".join(unresolved[:40]) +
              (" ..." if len(unresolved) > 40 else ""), file=sys.stderr)

    if args.dry_run:
        print(f"[backfill] DRY RUN — {len(updates)} updates NOT applied", file=sys.stderr)
        return 0

    if updates:
        script = "BEGIN;\n" + "\n".join(updates) + "\nCOMMIT;\n"
        run_psql(["-q"], stdin=script)
        print(f"[backfill] applied {len(updates)} updates", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
