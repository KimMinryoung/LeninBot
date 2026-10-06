"""Daily news → knowledge graph.

    python -m jobs.news_kg [--dry-run] [--max-stories N] [--notify-on-error] [--json]

1. Collect: one Korean and one English news search per domain in
   ``config/news_kg.json`` (web gateway, topic=news, time_range=day).
2. Dedupe: URLs already processed (Postgres ``kg_news_items``) are dropped;
   titles covered in the last few days go to the selector as context.
3. Select: registry site ``news_kg_select`` groups same-event articles and
   picks at most ``max_stories`` major stories, each with a KG group and up
   to two articles from different domains.
4. Fetch full text (free paths first, paid extract last). A story without
   any fetched article is skipped; snippets alone are never extracted.
5. Extract: registry site ``news_kg_extraction`` turns a story's articles
   into ≤8 agent-schema facts whose ``valid_at`` is the event date.
6. Write with ``add_kg_structured`` (agent ``news``): tier ``corroborated``
   when two domains were fetched, else ``single``. Facts carry
   ``source_url``/``news_*`` attributes but no ``sync_key``, so agents can
   still retract a wrong news fact; the URL table prevents rewrites.

Runs under ``systemd/leninbot-news-kg.timer`` at 07:30 KST. Ad-hoc runs, even
``--dry-run``, need ``NEO4J_PASSWORD`` (the extraction step resolves entities
read-only to drop self-loops); writes also need ``LENINBOT_ALLOW_WRITE=1`` for the URL table (the graph itself is not covered
by the Postgres write guard — back up first and prefer ``--dry-run``).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from kg_runtime import doc_extract as dx

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "news_kg.json"
AGENT = "news"
SELECT_FEATURE = "news_kg_select"
EXTRACT_FEATURE = "news_kg_extraction"
GROUPS = ("geopolitics_conflict", "diplomacy", "economy", "korea_domestic", "agent_knowledge")
MAX_SEARCHES = 8
RESULTS_PER_SEARCH = 8
MAX_FETCHES = 12
MAX_ARTICLES_PER_STORY = 2
MAX_ARTICLE_CHARS = 8000
MIN_ARTICLE_CHARS = 300
MAX_STORY_FACTS = 8
RECENT_DAYS = 3
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
KST = ZoneInfo("Asia/Seoul")

NEWS_EXTRACTION_SYSTEM = (
    "You extract knowledge-graph facts about one current news event for a Korean-language "
    "political-economy knowledge base. The articles below report the same event.\n"
    + dx.FACT_SCHEMA_RULES.replace(
        '"object_aliases": [str]}',
        '"object_aliases": [str], "article": int}',
    )
    + "Guidelines:\n"
    + f"- At most {MAX_STORY_FACTS} facts: who did what, to whom, where, with what result. Prefer\n"
    "  concrete actions, decisions, numbers and named participants over background and commentary.\n"
    "- valid_at is the date the event or action happened, NOT the article's publication date. Use the\n"
    "  date the article states. Resolve relative or year-less dates (지난달 30일, 지난 2일, last Tuesday)\n"
    "  from THAT article's publication date, including the year: its Published line, or when that is\n"
    "  unknown the registration date printed in the article text (입력/등록/게재 date), and only when\n"
    "  neither exists, Today. Use null when unsure.\n"
    "- \"article\" is the number of the article that states the fact.\n"
    "- Model the event as ONE specifically named Incident or Campaign (e.g. 2026년 프랑스 학생 시위) and\n"
    "  write that exact same name in every fact that refers to the event; never a second variant of it.\n"
    "  Connect named actors to it (Involvement, Participation, Presence, Statement). Counts, casualties,\n"
    "  arrests, prices and amounts go in the fact sentence, never in an entity name: not 체포자 5060명,\n"
    "  학교 24곳, 학생 190명, 브렌트유 가격. Unnamed groups (당국, 시위대, 투자자) are not entities; name the\n"
    "  concrete organization (프랑스 교육부) or attach the claim to the Incident.\n"
    "- Countries are Organization under their English name, never Location and never a Korean variant:\n"
    "  South Korea (not 한국, 대한민국), North Korea, United States, Ukraine, Russia, France.\n"
    "- Direction: the actor is always the subject. Involvement/Participation go actor → Incident/Campaign,\n"
    "  never Incident → actor. Presence goes thing/event → Location; a Location is never its subject.\n"
    "- Use only facts about this event. Ignore other topics the page happens to contain (sidebars,\n"
    "  related headlines, unrelated sections of a report).\n"
    + dx.ENTITY_GUIDELINES
    + "- Never use the article, the news outlet or the reporter as an entity unless the outlet itself acts\n"
    "  in the event. Skip quotes of anonymous sources and speculation about what may happen.\n"
    + dx.DISTINCT_ENDPOINTS_GUIDELINE
)

SELECT_SYSTEM = """You pick the day's major news events for a Korean-language political-economy knowledge base.
Domains of interest: international politics and war, diplomacy, economy and markets, South Korean
domestic politics and society, labour and social movements (strikes, unions, protests).
Return ONLY a JSON object: {"stories": [{"title": str, "group_id": G, "items": [int, ...]}]}
G ∈ {geopolitics_conflict, diplomacy, economy, korea_domestic, agent_knowledge}
- Group candidates that report the same event; "items" lists their candidate numbers, best first.
- Pick at most %d stories, most significant first, spread across the domains when possible.
- Skip celebrity, sports, lifestyle, weather trivia, opinion columns, and market tickers without an event.
- Skip events listed under "Already covered" unless a candidate reports a clearly new development.
- group_id follows where the event happens, not the outlet's country: korea_domestic ONLY for events
  inside South Korea. A Korean outlet reporting a French protest is geopolitics_conflict (or economy).
- Labour and social-movement stories: korea_domestic when in Korea, otherwise economy or geopolitics_conflict.
- "items" must be news articles about the event itself: never a journalist/author profile, section,
  tag or index page, a research report or a market-data page. A story needs at least one such article.
- "title" is a short Korean description of the event."""


# ── Collection ────────────────────────────────────────────────────────────────

def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def parse_search_results(text: str) -> list[dict]:
    """Parse the web gateway's search text (``web_gateway/search.py:_format_results``)."""
    items: list[dict] = []
    cur: dict | None = None
    for line in str(text or "").splitlines():
        if line.startswith("### "):
            cur = {"title": line[4:].strip(), "url": "", "published": "", "snippet": []}
            items.append(cur)
        elif cur is None or line.strip() == "</external>":
            continue
        elif line.startswith("[source_kind=") and not cur["url"]:
            m = re.search(r"publication=([^;\]]*)", line)
            pub = (m.group(1).strip() if m else "")
            cur["published"] = "" if pub == "unknown" else pub
        elif not cur["url"] and re.match(r"^https?://\S+$", line.strip()):
            cur["url"] = line.strip()
        elif cur["url"]:
            cur["snippet"].append(line)
    out = []
    for it in items:
        if not it["url"]:
            continue
        title = it["title"]
        if it["published"] and title.endswith(f" ({it['published']})"):
            title = title[: -len(it["published"]) - 3].rstrip()
        out.append({"title": title, "url": it["url"], "published": it["published"],
                    "snippet": "\n".join(it["snippet"]).strip()})
    return out


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not k.lower().startswith("utm_")])
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()[:32]


def domain_of(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


async def collect_candidates(config: dict) -> tuple[list[dict], dict]:
    from runtime_tools.web_search import execute_web_search
    from tool_gateway.results import is_failure

    queries = [(d["name"], q) for d in config["domains"] for q in d["queries"]][:MAX_SEARCHES]
    results = await asyncio.gather(*(
        execute_web_search(q, max_results=RESULTS_PER_SEARCH, search_depth="basic",
                           topic="news", time_range="day",
                           exclude_domains=config.get("exclude_domains") or None)
        for _, q in queries))
    seen, out, failed = set(), [], 0
    for (domain, query), text in zip(queries, results):
        if is_failure(text):
            failed += 1
            logger.warning("[news-kg] search failed (%s): %s", query, str(text)[:200])
            continue
        for item in parse_search_results(text):
            h = url_hash(item["url"])
            if h in seen:
                continue
            seen.add(h)
            out.append({**item, "url_hash": h, "domain": domain})
    return out, {"searches": len(queries), "searches_failed": failed}


# ── State (Postgres) ──────────────────────────────────────────────────────────

_table_ensured = False


def _ensure_table() -> None:
    global _table_ensured
    if _table_ensured:
        return
    from db import execute as db_execute
    db_execute("""
        CREATE TABLE IF NOT EXISTS kg_news_items (
            url_hash      TEXT PRIMARY KEY,
            url           TEXT NOT NULL,
            title         TEXT,
            story_key     TEXT,
            story_title   TEXT,
            group_id      TEXT,
            published     TEXT,
            status        TEXT NOT NULL,
            facts_written INTEGER NOT NULL DEFAULT 0,
            processed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    _table_ensured = True


def _table_exists() -> bool:
    from db import query_one
    row = query_one("SELECT to_regclass('kg_news_items') AS name")
    return bool(row and row["name"])


def processed_hashes(hashes: list[str]) -> set[str]:
    if not hashes or not _table_exists():
        return set()
    from db import query
    rows = query("SELECT url_hash FROM kg_news_items WHERE url_hash = ANY(%s)", (hashes,))
    return {r["url_hash"] for r in rows}


def recent_story_titles(days: int = RECENT_DAYS) -> list[str]:
    if not _table_exists():
        return []
    from db import query
    rows = query("SELECT DISTINCT story_title FROM kg_news_items "
                 "WHERE processed_at > NOW() - make_interval(days => %s) AND status = 'written' "
                 "AND story_title IS NOT NULL", (days,))
    return [r["story_title"] for r in rows]


def record_items(rows: list[dict]) -> None:
    if not rows:
        return
    _ensure_table()
    from db import execute as db_execute
    for r in rows:
        db_execute(
            """
            INSERT INTO kg_news_items (url_hash, url, title, story_key, story_title, group_id,
                                       published, status, facts_written)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url_hash) DO UPDATE SET
                status = EXCLUDED.status, facts_written = EXCLUDED.facts_written,
                story_key = EXCLUDED.story_key, story_title = EXCLUDED.story_title,
                group_id = EXCLUDED.group_id, processed_at = NOW()
            """,
            (r["url_hash"], r["url"], r.get("title"), r.get("story_key"), r.get("story_title"),
             r.get("group_id"), r.get("published"), r["status"], int(r.get("facts_written") or 0)),
        )


def today_label(run_date: str) -> str:
    return f"Today: {run_date} ({date.fromisoformat(run_date).strftime('%A')}, Asia/Seoul)"


# ── Selection ─────────────────────────────────────────────────────────────────

def build_select_prompt(candidates: list[dict], recent: list[str], run_date: str) -> str:
    lines = [today_label(run_date), "", "Already covered:"]
    lines += [f"- {t}" for t in recent] or ["- (none)"]
    lines += ["", "Candidates:"]
    for i, c in enumerate(candidates):
        snippet = re.sub(r"\s+", " ", c["snippet"])[:300]
        lines.append(f"[{i}] {c['title']} | {domain_of(c['url'])} | {c['published'] or '?'} | {snippet}")
    return "\n".join(lines)


def parse_selection(raw: str, candidates: list[dict], max_stories: int) -> list[dict]:
    """Validate the selector's JSON: known groups, real candidate numbers, distinct domains."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    data = json.loads(text)
    stories, used = [], set()
    for s in (data.get("stories") or []) if isinstance(data, dict) else []:
        if not isinstance(s, dict) or s.get("group_id") not in GROUPS:
            continue
        picked, domains = [], set()
        for idx in s.get("items") or []:
            if not isinstance(idx, int) or not 0 <= idx < len(candidates) or idx in used:
                continue
            c = candidates[idx]
            dom = domain_of(c["url"])
            if dom in domains:
                continue
            domains.add(dom)
            used.add(idx)
            picked.append(c)
        if not picked:
            continue
        title = re.sub(r"\s+", " ", str(s.get("title") or picked[0]["title"])).strip()[:200]
        key = hashlib.sha256("|".join(sorted(c["url_hash"] for c in picked)).encode()).hexdigest()[:12]
        stories.append({"title": title, "group_id": s["group_id"], "candidates": picked, "story_key": key})
        if len(stories) >= max_stories:
            break
    return stories


def select_stories(candidates: list[dict], recent: list[str], run_date: str, max_stories: int) -> list[dict]:
    from llm.call_registry import generate_sync
    raw = generate_sync(SELECT_FEATURE, build_select_prompt(candidates, recent, run_date),
                        system=SELECT_SYSTEM % max_stories)
    if not raw:
        raise RuntimeError("news selection returned no response")
    return parse_selection(raw, candidates, max_stories)


# ── Fetch ─────────────────────────────────────────────────────────────────────

async def fetch_articles(stories: list[dict]) -> int:
    """Attach ``articles`` (fetched text, ≤2 per story) to each story; returns fetch count."""
    from content_fetch.urls import fetch_url_content_async

    fetches = 0
    for story in stories:
        story["articles"] = []
        for c in story["candidates"]:
            if len(story["articles"]) >= MAX_ARTICLES_PER_STORY or fetches >= MAX_FETCHES:
                break
            fetches += 1
            try:
                text = await fetch_url_content_async(c["url"], max_chars=MAX_ARTICLE_CHARS)
            except Exception as exc:
                logger.info("[news-kg] fetch failed %s: %s", c["url"][:80], type(exc).__name__)
                text = None
            if text and len(str(text).strip()) >= MIN_ARTICLE_CHARS:
                story["articles"].append({**c, "text": str(text).strip()})
            else:
                c["fetch_failed"] = True
    return fetches


# ── Extraction ────────────────────────────────────────────────────────────────

def build_extract_prompt(story: dict, run_date: str) -> str:
    parts = [f"Event: {story['title']}", today_label(run_date)]
    for n, a in enumerate(story["articles"], 1):
        parts.append(f"\n=== Article {n}: {a['title']}\nOutlet: {domain_of(a['url'])}\n"
                     f"Published: {_published_label(a['published'], run_date)}\nURL: {a['url']}\n---\n{a['text']}")
    return "\n".join(parts)


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def published_date(value) -> str | None:
    """ISO date from a search result's publication field (RFC 2822 or ISO), else None."""
    from email.utils import parsedate_to_datetime
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError, IndexError):
        return None


def _published_label(value, run_date: str) -> str:
    d = published_date(value)
    if d:
        return d
    # time_range=day: the article surfaced in a past-day news search.
    return f"unknown (found by a past-day news search on {run_date})"


def normalize_facts(story: dict, raw_facts: list[dict], run_date: str) -> tuple[list[dict], list[dict]]:
    """Return (valid facts, [{"fact", "reason"}] schema rejections for the repair round)."""
    from graph_memory.structured_writer import validate_fact
    from kg_runtime.identity import normalize_alias_key

    articles = story["articles"]
    out, rejected = [], []
    for i, f in enumerate(raw_facts[:MAX_STORY_FACTS]):
        fact = {k: f.get(k) for k in ("subject_name", "subject_type", "predicate",
                                      "object_name", "object_type", "fact")}
        for side in ("subject", "object"):
            al = f.get(f"{side}_aliases")
            if isinstance(al, list):
                fact[f"{side}_aliases"] = [_clean(a) for a in al
                                           if a and _clean(a) != fact.get(f"{side}_name")]
        n = f.get("article")
        art = articles[n - 1] if isinstance(n, int) and 1 <= n <= len(articles) else articles[0]
        valid_at = str(f.get("valid_at") or "")[:10]
        valid_at = valid_at if _DATE_RE.match(valid_at) else None
        invalid_at = str(f.get("invalid_at") or "")[:10]
        if valid_at:
            fact["valid_at"] = valid_at
        if _DATE_RE.match(invalid_at) and (not valid_at or invalid_at > valid_at):
            fact["invalid_at"] = invalid_at
        if f.get("confirm_new_entities") is True:
            fact["confirm_new_entities"] = True
        fact["attributes"] = {
            "source_url": art["url"], "news_published": published_date(art["published"]),
            "news_outlet": domain_of(art["url"]), "story_key": story["story_key"],
            "extraction": "llm", "verification_status": "unverified",
        }
        fact["attributes"] = {k: v for k, v in fact["attributes"].items() if v not in (None, "")}
        err = validate_fact(fact, i)
        if err:
            logger.info("[news-kg] %s: rejected fact %d: %s", story["story_key"], i, err)
            rejected.append({"fact": fact, "reason": err})
            continue
        if (fact["subject_type"] == fact["object_type"]
                and normalize_alias_key(fact["subject_name"]) == normalize_alias_key(fact["object_name"])):
            continue
        out.append(fact)
    return out, rejected


REPAIR_INSTRUCTIONS = """These facts were rejected by the knowledge-graph schema or the duplicate-entity guard.
Fix each one according to its reason and return ONLY {"facts": [ ... ]} in the same fact format
(including "article"). When the reason says the reverse direction is allowed, swap subject and object
(with their types and aliases) and rewrite nothing else. When it says the KG already has an entity,
use that existing name exactly if it is the same entity; set "confirm_new_entities": true only when it
is clearly a different entity. Choose a predicate the reason lists as allowed when one fits the
meaning. Omit a fact that cannot be fixed without changing what the article says."""


def repair_facts(story: dict, rejected: list[dict], run_date: str) -> list[dict]:
    """One LLM repair round over rejected facts; returns valid repaired facts (rest dropped)."""
    from llm.call_registry import generate_sync

    urls = [a["url"] for a in story["articles"]]
    items = []
    for r in rejected:
        f = {k: v for k, v in r["fact"].items() if k not in ("attributes", "confirm_new_entities")}
        src = (r["fact"].get("attributes") or {}).get("source_url")
        f["article"] = urls.index(src) + 1 if src in urls else 1
        items.append({"fact": f, "reason": r["reason"]})
    prompt = (f"Event: {story['title']}\n{today_label(run_date)}\n\n{REPAIR_INSTRUCTIONS}\n\n"
              + json.dumps(items, ensure_ascii=False, indent=1))
    raw = generate_sync(EXTRACT_FEATURE, prompt, system=NEWS_EXTRACTION_SYSTEM)
    if not raw:
        logger.warning("[news-kg] %s: repair returned no response", story["story_key"])
        return []
    try:
        fixed, still = normalize_facts(story, dx.parse_llm_facts(raw, strict=True), run_date)
    except ValueError as exc:
        logger.warning("[news-kg] %s: repair output unusable: %s", story["story_key"], exc)
        return []
    for r in still:
        logger.info("[news-kg] %s: dropped after repair: %s", story["story_key"], r["reason"])
    return fixed


def extract_story_facts(story: dict, run_date: str) -> list[dict]:
    from llm.call_registry import generate_sync
    raw = generate_sync(EXTRACT_FEATURE, build_extract_prompt(story, run_date), system=NEWS_EXTRACTION_SYSTEM)
    if not raw:
        raise RuntimeError(f"news extraction returned no response: {story['story_key']}")
    facts, rejected = normalize_facts(story, dx.parse_llm_facts(raw, strict=True), run_date)
    if rejected:
        facts += repair_facts(story, rejected, run_date)
    facts, _skipped = dx.filter_resolved_self_loops(facts)
    return facts


def trust_tier(story: dict) -> str:
    return "corroborated" if len({domain_of(a["url"]) for a in story["articles"]}) >= 2 else "single"


def _write(story: dict, facts: list[dict]) -> dict:
    from kg_runtime.writes import add_kg_structured
    footer = f"news: {story['title']}\n" + "\n".join(f"url: {a['url']}" for a in story["articles"])
    return add_kg_structured(facts, group_id=story["group_id"], agent=AGENT, trust_tier=trust_tier(story),
                             provenance_footer=footer, cross_script_guard=True)


def write_story(story: dict, facts: list[dict], run_date: str) -> dict:
    """Write, then send the writer's rejections (e.g. a cross-script duplicate name) through one
    repair round and write the repaired facts. Returns combined counts."""
    res = _write(story, facts)
    rejected = [{"fact": r["fact"], "reason": r["reason"]} for r in res.get("rejected_facts") or []]
    if not rejected:
        return res
    written = int(res.get("facts_written") or 0)
    repaired = repair_facts(story, rejected, run_date)
    second = _write(story, repaired) if repaired else {"facts_written": 0, "facts_rejected": 0}
    written += int(second.get("facts_written") or 0)
    final_rejected = len(rejected) - len(repaired) + int(second.get("facts_rejected") or 0)
    infra_error = next((r.get("message") for r in (res, second)
                        if r.get("status") == "error" and not r.get("facts_rejected")), None)
    return {"status": "error" if infra_error else ("ok" if not final_rejected else
                                                   "partial_success" if written else "rejected"),
            "message": infra_error, "facts_written": written, "facts_rejected": final_rejected,
            "facts_repaired": len(repaired)}


# ── Run ───────────────────────────────────────────────────────────────────────

def _item_rows(story: dict, status: str, facts_written: int) -> list[dict]:
    fetched = {a["url_hash"] for a in story.get("articles", [])}
    rows = []
    for c in story["candidates"]:
        if c["url_hash"] not in fetched and not c.get("fetch_failed"):
            continue  # never fetched (cap reached): leave it for a later run
        rows.append({**c, "story_key": story["story_key"], "story_title": story["title"],
                     "group_id": story["group_id"],
                     "status": status if c["url_hash"] in fetched else "fetch_failed",
                     "facts_written": facts_written if c["url_hash"] in fetched else 0})
    return rows


def rejection_alert(stats: dict) -> str | None:
    """A run where most facts stay rejected after repair points at a systemic cause (schema or prompt
    drift), not at single bad facts, so it fails the run; isolated rejections stay silent."""
    rejected, written = stats.get("facts_rejected") or 0, stats.get("facts_written") or 0
    if rejected and rejected > written:
        return f"{rejected} facts rejected after repair vs {written} written"
    return None


def run(*, max_stories: int | None = None, dry_run: bool = False) -> dict:
    config = load_config()
    max_stories = max_stories or int(config.get("max_stories") or 6)
    run_date = datetime.now(KST).date().isoformat()

    candidates, stats = asyncio.run(collect_candidates(config))
    stats["candidates"] = len(candidates)
    if stats["searches_failed"] == stats["searches"]:
        return {**stats, "error": "all news searches failed"}
    done = processed_hashes([c["url_hash"] for c in candidates])
    candidates = [c for c in candidates if c["url_hash"] not in done]
    stats["new_candidates"] = len(candidates)
    if not candidates:
        return {**stats, "stories": []}

    stories = select_stories(candidates, recent_story_titles(), run_date, max_stories)
    stats["fetches"] = asyncio.run(fetch_articles(stories))

    stats.update({"facts_written": 0, "facts_rejected": 0, "stories": []})
    errors = []
    for story in stories:
        entry = {"title": story["title"], "group_id": story["group_id"],
                 "urls": [a["url"] for a in story["articles"]]}
        stats["stories"].append(entry)
        if not story["articles"]:
            entry["status"] = "fetch_failed"
            if not dry_run:
                record_items(_item_rows(story, "fetch_failed", 0))
            continue
        entry["tier"] = trust_tier(story)
        try:
            facts = extract_story_facts(story, run_date)
        except Exception as exc:
            logger.exception("[news-kg] extraction failed: %s", story["title"])
            entry.update(status="extract_error", error=str(exc)[:300])
            errors.append(f"{story['title']}: {exc}")
            continue
        entry["facts"] = len(facts)
        if dry_run:
            entry.update(status="dry_run", sample=[
                {k: f.get(k) for k in ("subject_name", "predicate", "object_name", "fact", "valid_at")}
                for f in facts])
            continue
        if not facts:
            entry["status"] = "no_facts"
            record_items(_item_rows(story, "no_facts", 0))
            continue
        res = write_story(story, facts, run_date)
        written = int(res.get("facts_written") or 0)
        entry.update(status=res.get("status"), facts_written=written,
                     facts_rejected=int(res.get("facts_rejected") or 0),
                     facts_repaired=int(res.get("facts_repaired") or 0))
        stats["facts_written"] += written
        stats["facts_rejected"] += entry["facts_rejected"]
        if res.get("status") == "error" and not res.get("facts_rejected"):
            entry["error"] = str(res.get("message") or "")[:300]
            errors.append(f"{story['title']}: {entry['error']}")
            continue  # infrastructure failure, not recorded: a later run retries these URLs
        # Schema rejections that survived the repair round are a quality outcome, not a run failure.
        record_items(_item_rows(story, "written" if written else "rejected", written))
    if not dry_run and (msg := rejection_alert(stats)):
        errors.append(msg)
    if errors:
        stats["error"] = "; ".join(errors)[:1000]
    stats["dry_run"] = dry_run
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-stories", type=int, default=None, help="cap selected stories (default: config)")
    parser.add_argument("--dry-run", action="store_true", help="search, select and extract; write nothing")
    parser.add_argument("--notify-on-error", action="store_true", help="Telegram notify on failure")
    parser.add_argument("--json", action="store_true", help="print stats as JSON only")
    args = parser.parse_args(argv)
    if args.max_stories is not None and args.max_stories < 1:
        parser.error("--max-stories must be positive")

    logging.basicConfig(level=logging.WARNING if args.json else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.dry_run:
            stats = run(max_stories=args.max_stories, dry_run=True)
        else:
            from kg_runtime.locks import kg_write_lock
            with kg_write_lock("news"):
                stats = run(max_stories=args.max_stories)
    except Exception as exc:
        logger.exception("[news-kg] failed")
        stats = {"error": str(exc)}

    print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    failed = bool(stats.get("error"))
    if failed and args.notify_on_error:
        try:
            sys.path.insert(0, "scripts")
            from _notify import notify_telegram
            notify_telegram(f"⚠️ 뉴스 KG 보강 실패 ({datetime.now().strftime('%m-%d %H:%M')})\n"
                            f"{str(stats['error'])[:500]}")
        except Exception as exc:  # notification is best-effort
            logger.warning("[news-kg] notify failed: %s", exc)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
