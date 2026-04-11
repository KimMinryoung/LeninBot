#!/usr/bin/env python3
"""
Migrate KG entities from v2.1 → v2.2 schema additions:

  Person → Role        (when name is purely a title with no individual)
  Asset  → Industry    (when name is a sector/value-chain abstraction)
  delete internal noise (task IDs, env vars, file names — should never have been ingested)

Heuristic shortlist + Gemini batch confirmation. Internal noise is deleted
without LLM (pattern is unambiguous). Persons whose name ends in an actual
individual's name (e.g. "CEO Kim Mi-kyung") stay as Person — only pure
titles become Role.

Usage:
    python migrate_to_v22_schema.py            # dry-run (default)
    python migrate_to_v22_schema.py --execute  # actually apply
"""
import argparse
import json
import os
import re
import time

from dotenv import load_dotenv
load_dotenv("/home/grass/leninbot/.env")

from neo4j import GraphDatabase

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASSWORD"]
NEO4J_DB = os.environ.get("NEO4J_DATABASE", "neo4j")

# ── Heuristics ────────────────────────────────────────────────────────────────

ROLE_TITLE_PATTERNS = [
    r"\b(Secretary|Minister|Director|Chair|Chairman|Chairwoman|Chief|President|"
    r"Vice President|VP|CEO|COO|CFO|CTO|Commissioner|Governor|Mayor|"
    r"Vice Mayor|Senator|Representative|Ambassador|Speaker|Justice|"
    r"Premier|Prime Minister|Chancellor|Head of|Leader of|"
    r"Minority Leader|Majority Leader|Whip|Supreme Leader)\b"
]

INDUSTRY_PATTERNS = [
    r"\b(industry|sector|mining|shipping|fisheries|drilling|"
    r"manufacturing|retail|hospitality|tourism|logistics|"
    r"farming|fishing|telecommunications|broadcasting)\b"
]

NOISE_PATTERNS = [
    r"^[Tt]ask\s*#\d+",                # Task #210
    r"^[A-Z][A-Z0-9_]+_[A-Z0-9_]+$",   # ENV_VAR_STYLE
    r"^[a-z_]+\.(py|js|ts|md|json|yml|yaml|toml)$",  # filenames
    r"^[A-Z]\w*Error$",                # NameError, ValueError
    r"^commit [a-f0-9]{6,}",           # commit 60547e1
    r"^pr #\d+", r"^PR #\d+",          # PR #123
    r"^issue #\d+", r"^Issue #\d+",
]


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


# ── Gemini batch classification ───────────────────────────────────────────────

ROLE_PROMPT = """\
You are validating entity classifications in a knowledge graph.
For each name+summary below, decide:
  "ROLE" if the name refers to a job title or position (e.g. "Secretary of Energy",
  "Senate Minority Leader", "President of Venezuela") with NO specific person mentioned.
  "PERSON" if the name refers to a specific individual, even if it includes a title prefix
  (e.g. "CEO Kim Mi-kyung", "Prime Minister Trudeau" — these are still people).
  "OTHER" if neither applies.

Output JSON only — no markdown fences. Schema:
[{"id": "uuid_short", "verdict": "ROLE"|"PERSON"|"OTHER"}]

Inputs:
"""

INDUSTRY_PROMPT = """\
You are validating entity classifications in a knowledge graph.
For each name+summary below, decide:
  "INDUSTRY" if the name refers to an economic sector or industry abstraction
  (e.g. "semiconductor industry", "Bitcoin mining", "oil sector", "shipping industry").
  "ASSET" if it refers to a specific tangible/intangible product, technology, or thing
  (e.g. "Claude Opus 4.6", "Patriot missile system", "MV Ever Given").
  "OTHER" if neither applies.

Output JSON only — no markdown fences. Schema:
[{"id": "uuid_short", "verdict": "INDUSTRY"|"ASSET"|"OTHER"}]

Inputs:
"""


def gemini_classify(prompt_template: str, candidates: list[dict]) -> list[dict]:
    """Send a batch to Gemini and return classification list."""
    from google import genai
    from google.genai.types import GenerateContentConfig

    if not candidates:
        return []

    inputs = [
        {"id": c["uuid"][:8], "name": c["name"], "summary": (c.get("summary") or "")[:200]}
        for c in candidates
    ]
    prompt = prompt_template + json.dumps(inputs, ensure_ascii=False)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(temperature=0.0, max_output_tokens=4096),
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    return json.loads(text)


# ── Phase 1: candidate discovery ──────────────────────────────────────────────

def fetch_persons(driver) -> list[dict]:
    with driver.session(database=NEO4J_DB) as s:
        rows = s.run("""
            MATCH (n:Entity) WHERE 'Person' IN labels(n) AND n.name IS NOT NULL
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary
        """).data()
    return rows


def fetch_assets(driver) -> list[dict]:
    with driver.session(database=NEO4J_DB) as s:
        rows = s.run("""
            MATCH (n:Entity) WHERE 'Asset' IN labels(n) AND n.name IS NOT NULL
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary
        """).data()
    return rows


def fetch_all(driver) -> list[dict]:
    with driver.session(database=NEO4J_DB) as s:
        rows = s.run("""
            MATCH (n:Entity) WHERE n.name IS NOT NULL
            RETURN n.uuid AS uuid, n.name AS name,
                   [l IN labels(n) WHERE l <> 'Entity'] AS types
        """).data()
    return rows


# ── Phase 2: apply changes ────────────────────────────────────────────────────

def relabel_to(driver, uuid: str, old_label: str, new_label: str) -> bool:
    with driver.session(database=NEO4J_DB) as s:
        try:
            s.run(
                f"MATCH (n:Entity {{uuid: $uuid}}) "
                f"REMOVE n:{old_label} SET n:{new_label}",
                uuid=uuid,
            )
            return True
        except Exception as exc:
            print(f"  [ERROR] relabel {uuid[:8]}: {exc}")
            return False


def delete_entity(driver, uuid: str) -> bool:
    with driver.session(database=NEO4J_DB) as s:
        try:
            s.run("MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n", uuid=uuid)
            return True
        except Exception as exc:
            print(f"  [ERROR] delete {uuid[:8]}: {exc}")
            return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--execute", action="store_true", help="Apply changes")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip Gemini confirmation (use heuristics only — less safe)")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    print("=" * 70)
    print(f"  KG schema migration v2.1 → v2.2  [{'EXECUTE' if args.execute else 'DRY RUN'}]")
    print("=" * 70)

    # ── Phase 1: candidates by heuristic ──
    persons = fetch_persons(driver)
    assets = fetch_assets(driver)
    everything = fetch_all(driver)

    role_candidates = [r for r in persons if matches_any(r["name"], ROLE_TITLE_PATTERNS)]
    industry_candidates = [r for r in assets if matches_any(r["name"].lower(), INDUSTRY_PATTERNS)]
    noise_candidates = [r for r in everything if matches_any(r["name"], NOISE_PATTERNS)]

    print(f"\n  Heuristic shortlist:")
    print(f"    Person→Role candidates:  {len(role_candidates)}")
    print(f"    Asset→Industry candidates: {len(industry_candidates)}")
    print(f"    Internal noise to delete: {len(noise_candidates)}")

    # ── Phase 2: LLM confirmation for role/industry ──
    confirmed_roles: list[dict] = []
    confirmed_industries: list[dict] = []

    if not args.skip_llm:
        if role_candidates:
            print(f"\n  Asking Gemini about {len(role_candidates)} role candidates...")
            verdicts = gemini_classify(ROLE_PROMPT, role_candidates)
            verdict_map = {v["id"]: v["verdict"] for v in verdicts}
            for c in role_candidates:
                v = verdict_map.get(c["uuid"][:8])
                marker = "→ ROLE" if v == "ROLE" else f"→ KEEP ({v or 'no verdict'})"
                print(f"    {c['name'][:40]:40s}  {marker}")
                if v == "ROLE":
                    confirmed_roles.append(c)

        if industry_candidates:
            print(f"\n  Asking Gemini about {len(industry_candidates)} industry candidates...")
            verdicts = gemini_classify(INDUSTRY_PROMPT, industry_candidates)
            verdict_map = {v["id"]: v["verdict"] for v in verdicts}
            for c in industry_candidates:
                v = verdict_map.get(c["uuid"][:8])
                marker = "→ INDUSTRY" if v == "INDUSTRY" else f"→ KEEP ({v or 'no verdict'})"
                print(f"    {c['name'][:40]:40s}  {marker}")
                if v == "INDUSTRY":
                    confirmed_industries.append(c)
    else:
        confirmed_roles = role_candidates
        confirmed_industries = industry_candidates

    # ── Phase 3: noise list (no LLM, heuristic patterns are unambiguous) ──
    print(f"\n  Internal noise to delete ({len(noise_candidates)}):")
    for c in noise_candidates:
        print(f"    {c['name'][:40]:40s}  types={c['types']}")

    # ── Phase 4: summary + execute ──
    print("\n" + "=" * 70)
    print(f"  PLAN:")
    print(f"    relabel Person→Role: {len(confirmed_roles)}")
    print(f"    relabel Asset→Industry: {len(confirmed_industries)}")
    print(f"    delete noise entities: {len(noise_candidates)}")
    print("=" * 70)

    if not args.execute:
        print("\n  [DRY RUN] No changes made. Re-run with --execute.")
        driver.close()
        return

    print("\n  Applying changes...")
    role_done = sum(1 for c in confirmed_roles if relabel_to(driver, c["uuid"], "Person", "Role"))
    industry_done = sum(1 for c in confirmed_industries if relabel_to(driver, c["uuid"], "Asset", "Industry"))
    noise_done = sum(1 for c in noise_candidates if delete_entity(driver, c["uuid"]))

    print(f"\n  Done:")
    print(f"    Person→Role relabeled: {role_done}")
    print(f"    Asset→Industry relabeled: {industry_done}")
    print(f"    noise entities deleted: {noise_done}")

    driver.close()


if __name__ == "__main__":
    main()
