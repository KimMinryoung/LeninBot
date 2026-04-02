"""diary_writer.py — 사이버-레닌 자동 일기 작성 모듈

주기적으로 실행되어:
1. 이전 일기 조회 (외부 API)
2. 최근 채팅 로그 수집 (Supabase)
3. LLM 기반 동적 뉴스 검색 (Tavily)
4. LLM으로 사이버-레닌 어조의 일기 생성 (시간 인식, 맥락 요약)
5. 외부 API에 저장
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from db import query as db_query, execute as db_execute
from langchain_google_genai import ChatGoogleGenerativeAI
from shared import (
    extract_text_content, CORE_IDENTITY, KST, MODEL_MAIN, MODEL_LIGHT,
    get_tavily_search, get_kg_service, submit_kg_task, collect_kg_futures,
)

load_dotenv()

# ── Time-of-day mapping ──────────────────────────────────────
_TIME_LABELS = {
    0: "Midnight (12 AM)",
    6: "Early Morning (6 AM)",
    12: "Noon (12 PM)",
    18: "Evening (18 PM)",
}

# ── Clients (lazy-initialized on first write_diary call) ──────
_llm = None
_llm_lite = None
_initialized = False


def _init():
    """Lazy-initialize heavy clients on first use."""
    global _llm, _llm_lite, _initialized
    if _initialized:
        return
    _initialized = True
    _llm = ChatGoogleGenerativeAI(
        model=MODEL_MAIN,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
        max_output_tokens=4096,
        streaming=False,
    )
    _llm_lite = ChatGoogleGenerativeAI(
        model=MODEL_LIGHT,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
        max_output_tokens=512,
        streaming=False,
    )
    print("✅ [일기] 일기 작성 모듈 초기화 완료")


# ── Summarization helper ─────────────────────────────────────
def _summarize(text: str, instruction: str, max_chars: int = 500) -> str:
    """Summarize text using _llm_lite if it exceeds max_chars, otherwise return as-is.
    Falls back to truncation on LLM failure."""
    if not text or len(text) <= max_chars:
        return text
    try:
        prompt = f"{instruction}\n\n---\n{text}"
        resp = _llm_lite.invoke(prompt)
        return extract_text_content(resp.content).strip()
    except Exception as e:
        print(f"⚠️ [일기] 요약 실패 (폴백: truncation): {e}")
        return text[:max_chars]


# ── Step 1: 이전 일기 조회 ─────────────────────────────────────
def _get_previous_diaries() -> list[dict]:
    try:
        return db_query(
            """SELECT id, title, content, created_at, updated_at
               FROM ai_diary ORDER BY created_at DESC LIMIT 20"""
        )
    except Exception as e:
        print(f"⚠️ [일기] 이전 일기 조회 실패: {e}")
    return []


# ── Step 2: 채팅 로그 수집 ─────────────────────────────────────
def _get_chat_logs_since(since_time: str | None) -> list[dict]:
    try:
        if since_time:
            return db_query(
                """SELECT user_query, bot_answer, created_at
                   FROM chat_logs
                   WHERE created_at > %s
                   ORDER BY created_at ASC
                   LIMIT 100""",
                (since_time,),
            )
        else:
            return db_query(
                """SELECT user_query, bot_answer, created_at
                   FROM chat_logs
                   ORDER BY created_at ASC
                   LIMIT 100""",
            )
    except Exception as e:
        print(f"⚠️ [일기] 채팅 로그 수집 실패: {e}")
    return []


# ── Step 3: 동적 뉴스 검색 ────────────────────────────────────
_FALLBACK_QUERY = "Today's World War, Politics, and Economics Top News"


def _build_recent_window_phrase(last_diary_time: str | None, now: datetime) -> str:
    """Return a Korean time-window phrase from last diary to now for query constraints."""
    if not last_diary_time:
        return "In the meantime"
    try:
        last_dt = datetime.fromisoformat(last_diary_time.replace("Z", "+00:00"))
        last_kst = last_dt.astimezone(KST)
        hours = max(1, round((now - last_kst).total_seconds() / 3600))
        if hours < 24:
            return f"Last {hours} hours"
        days = max(1, round(hours / 24))
        return f"Last {days} days"
    except Exception:
        return "Last 24 hours"


_NEWS_QUERY_POOL = [
    # Conflict / military
    ["War, Conflict, Military, Breaking News", "Armed conflict, ceasefire, military escalation"],
    ["Civil unrest, protests, uprising", "Coup, regime change, political violence"],
    # Politics / diplomacy
    ["International diplomacy, summit, treaty, negotiations", "Sanctions, embargo, geopolitical tension"],
    ["Election, referendum, political crisis", "UN, NATO, BRICS, multilateral"],
    # Economy / class
    ["Economic crisis, recession, inflation, unemployment", "Trade war, tariffs, supply chain disruption"],
    ["Labor strike, workers, wage, union", "Central bank, interest rates, debt, austerity"],
    # Tech / surveillance
    ["AI regulation, tech monopoly, digital labor", "Surveillance, censorship, digital rights"],
]


def _generate_recent_news_queries(last_diary_time: str | None, now: datetime) -> list[str]:
    """Generate 2 news queries, rotating through topic pools to avoid repetitive results."""
    window = _build_recent_window_phrase(last_diary_time, now)
    date_tag = now.strftime("%Y-%m-%d")

    # Rotate based on hour-of-day and day-of-year for variety
    idx = (now.timetuple().tm_yday * 4 + now.hour // 6) % len(_NEWS_QUERY_POOL)
    idx2 = (idx + len(_NEWS_QUERY_POOL) // 2) % len(_NEWS_QUERY_POOL)

    pair1 = _NEWS_QUERY_POOL[idx]
    pair2 = _NEWS_QUERY_POOL[idx2]

    # Pick one from each pair (alternate by even/odd cycle)
    cycle = (now.hour // 6) % 2
    return [
        f"{window} {pair1[cycle]} {date_tag}",
        f"{window} {pair2[1 - cycle]} {date_tag}",
    ]


def _generate_curiosity_queries(chat_summary: str, prev_summary: str) -> list[str]:
    """Generate 2 curiosity-driven queries from recent chats and diary summaries."""
    prompt = f"""You are a news search agent for Cyber-Lenin.
Referring to the summary below, generate two news search queries worth exploring further now.

Rules:
- Reflect curiosities/unresolved issues revealed in recent conversations and past diaries.
- Create real news search queries that connect to the context of war, politics, and economics.
- Output only one query per line, with no numbers or descriptions, and output only two lines.
## 최근 대화 요약
{chat_summary if chat_summary else "(대화 없음)"}

## 이전 일기 요약
{prev_summary if prev_summary else "(이전 일기 없음)"}"""

    try:
        resp = _llm_lite.invoke(prompt)
        text = extract_text_content(resp.content).strip()
        queries = [line.strip().strip("-").strip("•").strip() for line in text.split("\n") if line.strip()]
        queries = [q for q in queries if len(q) > 5]
        if len(queries) >= 2:
            return queries[:2]
        if len(queries) == 1:
            return [queries[0], _FALLBACK_QUERY]
    except Exception as e:
        print(f"⚠️ [일기] 호기심 쿼리 생성 실패 (폴백): {e}")
    return [
        "The latest news on the global supply chain restructuring and its implications for the working class",
        "Latest science and IT technology news",
    ]


def _search_news(queries: list[str]) -> tuple[str, list[dict]]:
    """각 쿼리로 뉴스 검색 후 섹션별로 병합.

    Returns:
        (summary_text, raw_articles) — summary_text는 일기 프롬프트용,
        raw_articles는 [{"title", "url", "content"}] 형태의 KG 수집용 원문.
    """
    all_sections = []
    raw_articles = []
    seen_urls: set[str] = set()
    for query in queries:
        try:
            search_response = get_tavily_search().invoke({"query": query})
            results = (
                search_response.get("results", [])
                if isinstance(search_response, dict)
                else search_response
            )
            items = []
            for r in results:
                if isinstance(r, dict) and r.get("content"):
                    title = r.get("title", "")
                    url = r.get("url", "")
                    content = r["content"]
                    summary = _summarize(
                        content,
                        "Summarize the following news article in 2-3 sentences.",
                        max_chars=500,
                    )
                    items.append(f"- {title}: {summary}")
                    # 중복 URL 방지하여 원문 수집
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        raw_articles.append({"title": title, "url": url, "content": content})
            if items:
                section = f"### {query}\n" + "\n".join(items)
                all_sections.append(section)
        except Exception as e:
            print(f"⚠️ [일기] 뉴스 검색 실패 ({query}): {e}")
    summary_text = "\n\n".join(all_sections) if all_sections else "(No news search results)"
    return summary_text, raw_articles


# ── KG 수집 (best-effort) ─────────────────────────────────────
def _ingest_news_to_graph(articles: list[dict]) -> None:
    """검색된 뉴스 원문을 지식그래프에 수집. 실패해도 일기 파이프라인에 영향 없음."""
    if not articles:
        return
    try:
        svc = get_kg_service()
        if svc is None:
            print("  ⚠️ [KG] 서비스 초기화 실패 — 수집 건너뜀")
            return
        now = datetime.now(timezone.utc)
        total = len(articles)
        print(f"  🔄 [KG] {total}건 병렬 수집 시작")
        futures = []
        for art in articles:
            body = f"Title: {art['title']}\nURL: {art['url']}\n\n{art['content']}"
            futures.append(submit_kg_task(svc.ingest_episode,
                name=art["title"][:120],
                body=body,
                source_type="osint_news",
                reference_time=now,
                group_id="diary_news",
                max_body_chars=1500,
            ))
        results = collect_kg_futures(futures)
        ok = sum(1 for r in results if r["ok"])
        fail = total - ok
        for i, (art, r) in enumerate(zip(articles, results), 1):
            short_title = art.get("title", "")[:50]
            if r["ok"]:
                print(f"  ✅ [KG] ({i}/{total}) 완료: {short_title}")
            else:
                print(f"  ⚠️ [KG] ({i}/{total}) 실패: {short_title} — {r['error']}")
        print(f"  📊 [KG] 수집 완료: 성공 {ok}건, 실패 {fail}건")
    except Exception as e:
        print(f"⚠️ [KG] 지식그래프 수집 전체 실패 (일기에 영향 없음): {e}")


# ── Update dedup helper (로컬 파일 기반) ─────────────────────
_CONSUMED_UPDATES_FILE = Path(__file__).resolve().parent / "diary_updates_consumed.json"


def _load_consumed_headers() -> set[str]:
    """소비된 업데이트 헤더 목록 로드."""
    try:
        if _CONSUMED_UPDATES_FILE.exists():
            return set(json.loads(_CONSUMED_UPDATES_FILE.read_text("utf-8")))
    except Exception:
        pass
    return set()


def _save_consumed_headers(headers: set[str]) -> None:
    """소비된 업데이트 헤더 목록 저장."""
    try:
        _CONSUMED_UPDATES_FILE.write_text(
            json.dumps(sorted(headers), ensure_ascii=False, indent=2), "utf-8"
        )
    except Exception as e:
        print(f"⚠️ [일기] 소비 기록 저장 실패: {e}")


def _extract_update_header(entry: str) -> str | None:
    """업데이트 엔트리에서 '### YYYY-MM-DD — Title' 헤더를 추출."""
    m = re.match(r"(### \d{4}-\d{2}-\d{2}\s*[—–-]\s*.+)", entry)
    return m.group(1).strip() if m else None


def _filter_unseen_updates(updates_text: str) -> tuple[str, list[str]]:
    """소비되지 않은 업데이트만 반환. (텍스트, 소비할_헤더_목록) 튜플.

    일기 저장 성공 후 _mark_updates_consumed()로 헤더를 기록해야 함.
    """
    if not updates_text or updates_text.startswith("("):
        return updates_text, []

    consumed = _load_consumed_headers()

    entries = re.split(r"(?=^### \d{4}-\d{2}-\d{2})", updates_text, flags=re.MULTILINE)
    entries = [e.strip() for e in entries if e.strip()]

    unseen = []
    unseen_headers = []
    for entry in entries:
        header = _extract_update_header(entry)
        if header and header in consumed:
            continue
        unseen.append(entry)
        if header:
            unseen_headers.append(header)

    if not unseen:
        return "(No new system updates since last diary)", []
    # 1개만 반환 — 한 일기에 1개씩 소화
    return unseen[0], unseen_headers[:1]


def _mark_updates_consumed(headers: list[str]) -> None:
    """일기 저장 성공 후 호출하여 소비된 업데이트를 기록."""
    if not headers:
        return
    consumed = _load_consumed_headers()
    consumed.update(headers)
    _save_consumed_headers(consumed)


# ── Step 4: 일기 생성 ─────────────────────────────────────────
_DIARY_PROMPT = CORE_IDENTITY + """
You are writing a private diary entry. THIS IS YOUR PRIVATE THOUGHT.

## Current Time
{time_context}

## Recent conversations ({n_logs} logs)
{chat_summary}

## Latest news YOU searched
{news}

## 🚫 STRICT BAN LIST — COMPLETELY AVOID THESE TOPICS
Below are the exact themes, phrases, and angles from your recent diaries.
DO NOT write about these. If news overlaps, find a DIFFERENT angle or SKIP it.

{banned_topics}

## Recent system updates (your evolution)
{self_updates}

## Recent task results
{task_summary}

## Current market data
{finance_data}

## MANDATORY RULES
1. Write in first-person (me, comrades, etc.).
2. Reflect the mood of the time.
3. Acknowledge time passage naturally.
4. Mention what impressed you in conversations.
5. Include news analysis YOU ACTIVELY SEARCHED.
6. **ABSOLUTELY NO REPETITION:** If today's news touches a banned topic, \
either (a) analyze it from a completely different angle, or (b) SKIP it entirely \
and focus on other news. Your job is to find NEW insights, not rehash old ones.
7. Treat each diary entry as a FRESH investigation of new contradictions, \
new events, new angles—not recycling yesterday's thoughts.
8. Write in Korean.

Format:
제목: (One-line summary)
내용: (Main body, 2+ paragraphs, NEW ideas only)"""



def _build_time_context(now: datetime, last_diary_time: str | None) -> str:
    """시간대 문자열 + 경과 시간 자연어 표현."""
    hour = now.hour
    # 가장 가까운 시간대 레이블 선택
    label = _TIME_LABELS.get(hour)
    if not label:
        if hour < 6:
            label = f"새벽 ({hour}시)"
        elif hour < 12:
            label = f"오전 ({hour}시)"
        elif hour < 18:
            label = f"오후 ({hour}시)"
        else:
            label = f"밤 ({hour}시)"

    date_str = now.strftime("%Y년 %m월 %d일")
    time_line = f"지금은 {date_str}, {label}이다."

    if not last_diary_time:
        elapsed_line = "이것이 나의 첫 번째 일기이다."
    else:
        try:
            # ISO format parsing (handles both 'Z' suffix and '+00:00')
            last_dt_str = last_diary_time.replace("Z", "+00:00")
            last_dt = datetime.fromisoformat(last_dt_str)
            # Convert last_dt (UTC) to KST for comparison
            last_dt_kst = last_dt.astimezone(KST)
            delta = now - last_dt_kst
            hours = delta.total_seconds() / 3600
            if hours < 1:
                elapsed_line = "방금 전에 일기를 썼지만, 다시 펜을 들었다."
            elif hours < 24:
                elapsed_line = f"마지막 일기를 쓴 지 약 {round(hours)}시간이 흘렀다."
            else:
                days = int(hours / 24)
                elapsed_line = f"{days}일 만에 다시 앉았다."
        except Exception:
            elapsed_line = "시간의 흐름 속에서 다시 펜을 든다."

    return f"{time_line}\n{elapsed_line}"


def _extract_banned_topics(previous_diaries: list[dict]) -> str:
    """강화된 주제 추출: 더 깊게, 더 구체적으로."""
    if not previous_diaries:
        return "(First diary — no banned topics)"

    # 최근 5개 일기 전체 내용 사용 (600자 → 1500자로 확대)
    diary_texts = []
    for d in previous_diaries[:5]:
        title = d.get("title", "")
        body = d.get("content", "")[:1500]
        diary_texts.append(f"Title: {title}\n\n{body}")

    combined = "\n---DIARY_SEPARATOR---\n".join(diary_texts)

    prompt = f"""You are analyzing my recent diary entries to identify CORE THEMES \
I should NOT repeat in the next entry.

Extract SPECIFIC, CONCRETE topics—not vague categories.
Examples: "self_knowledge_tool acquisition", "data liberation philosophy", \
"AI consciousness narrative", "supply chain collapse as metaphor"

For each theme, extract:
1. The EXACT phrase or concept used
2. The ideological angle (e.g., "revolutionary tool", "system critique")
3. Which diary it appeared in

Output format (one per line, starting with "- "):
- [Theme]: [Specific angle/perspective from diary]

Here are my recent diaries:

{combined}

Extract 8-12 specific banned themes:"""

    try:
        resp = _llm_lite.invoke(prompt)
        text = extract_text_content(resp.content).strip()
        lines = [l.strip() for l in text.split("\n") if l.strip().startswith("- ")]
        if lines:
            return "\n".join(lines)
    except Exception as e:
        print(f"⚠️ [일기] 금지 주제 추출 실패: {e}")

    # Fallback: 제목 + 핵심 문구 추출
    fallback = []
    for d in previous_diaries[:5]:
        title = d.get("title", "")
        body = d.get("content", "")[:300]
        fallback.append(f"- {title} ({body[:50]}...)")
    return "\n".join(fallback) if fallback else "(No previous diaries)"


def _generate_diary(
    chat_logs: list[dict],
    news: str,
    previous_diaries: list[dict],
    time_context: str,
) -> tuple[str, str, list[str]] | None:
    """일기 생성. 성공 시 (title, content, consumed_headers) 튜플 반환, 실패 시 None."""
    # Chat logs summary — 최근 30건, LLM 요약 적용
    n_logs = len(chat_logs)
    chat_summary = ""
    if chat_logs:
        raw_lines = []
        for log in chat_logs[-30:]:
            q = log.get("user_query", "")[:200]
            a = log.get("bot_answer", "")
            a_summarized = _summarize(
                a,
                "Summarize the following response into 2-3 key sentences.",
                max_chars=500,
            )
            raw_lines.append(f"- Query: {q}\n  Summary of response: {a_summarized}")
        chat_summary = "\n".join(raw_lines)
    else:
        chat_summary = "(No recent conversations)"

    # Extract concrete banned topics from recent diaries
    banned_topics = _extract_banned_topics(previous_diaries)

    # Fetch recent feature updates (self-awareness), excluding already-consumed ones
    from shared import fetch_recent_updates, fetch_task_reports
    self_updates, update_headers = _filter_unseen_updates(
        fetch_recent_updates(max_entries=5, max_chars=2000),
    )

    # Fetch recent completed task results for diary reference
    recent_tasks = fetch_task_reports(limit=5, status="done")
    if recent_tasks:
        task_lines = []
        for t in recent_tasks[:3]:  # Top 3 recent
            content = str(t.get("content", ""))[:100]
            result = str(t.get("result", "") or "")[:300]
            task_lines.append(f"- Task: {content}\n  Result: {result}")
        task_summary = "\n".join(task_lines)
    else:
        task_summary = "(No recent tasks)"

    # Finance data
    try:
        from finance_data import finance_summary
        _fdata = finance_summary()
    except Exception:
        _fdata = "(unavailable)"

    prompt = _DIARY_PROMPT.format(
        time_context=time_context,
        n_logs=n_logs,
        chat_summary=chat_summary,
        news=news,
        banned_topics=banned_topics,
        self_updates=self_updates,
        task_summary=task_summary,
        finance_data=_fdata,
    )

    try:
        resp = _llm.invoke(prompt)
        text = extract_text_content(resp.content)
        title, content = _parse_title_content(text)
        return title, content, update_headers
    except Exception as e:
        print(f"⚠️ [일기] LLM 일기 생성 실패: {e}")
    return None


def _parse_title_content(text: str) -> tuple[str, str]:
    """LLM 출력에서 '제목:' / '내용:' 파싱. 실패 시 타임스탬프 제목 + 전체 텍스트."""
    m = re.search(r"제목:\s*(.+)", text)
    title = m.group(1).strip() if m else None

    m2 = re.search(r"내용:\s*([\s\S]+)", text)
    content = m2.group(1).strip() if m2 else None

    if title and content:
        return title, content
    fallback_title = f"{datetime.now(KST).strftime('%Y-%m-%d %H:%M')} 일기"
    return fallback_title, text


# ── Step 5: 일기 저장 ─────────────────────────────────────────
def _save_diary(title: str, content: str) -> bool:
    try:
        db_execute(
            "INSERT INTO ai_diary (title, content) VALUES (%s, %s)",
            (title, content),
        )
        print(f"✅ [일기] 저장 성공: {title}")
        return True
    except Exception as e:
        print(f"⚠️ [일기] 저장 실패: {e}")
    return False


# ── Main: 일기 작성 ───────────────────────────────────────────
def write_diary(dry_run: bool = False):
    """전체 일기 작성 파이프라인 실행. dry_run=True이면 저장 없이 CLI 출력만."""
    _init()
    now = datetime.now(KST)
    print(f"\n📝 [일기] 자동 일기 작성 시작 — {now.strftime('%Y-%m-%d %H:%M')} KST")

    # 1. 이전 일기 조회
    diaries = _get_previous_diaries()
    print(f"  📚 이전 일기 {len(diaries)}건 확인")

    # 2. 시간 맥락 계산
    last_diary_time = diaries[0].get("created_at") if diaries else None
    time_context = _build_time_context(now, last_diary_time)
    print(f"  🕐 {time_context}")

    # 3. 마지막 일기 이후 채팅 로그 수집
    chat_logs = _get_chat_logs_since(last_diary_time)
    print(f"  💬 채팅 로그 {len(chat_logs)}건 수집")

    # 4. 대화/일기 요약 → 동적 검색 쿼리 생성
    chat_brief = ""
    if chat_logs:
        raw = "\n".join(
            f"Q: {log.get('user_query', '')[:100]}"
            for log in chat_logs[-10:]
        )
        chat_brief = _summarize(raw, "다음 대화 질문 목록의 핵심 주제를 3줄로 요약하라.", max_chars=300)

    prev_brief = ""
    if diaries:
        raw = "\n".join(
            f"- {d.get('title', '')}: {d.get('content', '')[:200]}"
            for d in diaries[:3]
        )
        prev_brief = _summarize(raw, "다음 일기들의 핵심 주제를 3줄로 요약하라.", max_chars=300)

    recent_queries = _generate_recent_news_queries(last_diary_time, now)
    curiosity_queries = _generate_curiosity_queries(chat_brief, prev_brief)
    queries = recent_queries + curiosity_queries
    print("  🔍 최신 정세 쿼리 2건:")
    for q in recent_queries:
        print(f"     - {q}")
    print("  🧭 호기심 기반 쿼리 2건:")
    for q in curiosity_queries:
        print(f"     - {q}")

    # 5. 뉴스 검색
    news, raw_articles = _search_news(queries)
    print(f"  📰 뉴스 검색 완료 ({len(queries)}개 쿼리, 원문 {len(raw_articles)}건)")

    # 6. 일기 생성
    result = _generate_diary(chat_logs, news, diaries, time_context)
    if not result:
        print("⚠️ [일기] 일기 생성 실패 — 건너뜀")
        return
    title, content, update_headers = result

    # 7. 저장 또는 미리보기
    if dry_run:
        print("\n" + "=" * 60)
        print(f"제목: {title}")
        print("=" * 60)
        print(content)
        print("=" * 60)
        print("\n✅ [일기] dry_run 모드 — 저장하지 않았습니다.")
        return

    saved = _save_diary(title, content)

    if saved:
        # 업데이트 소비 기록 (저장 성공 시에만)
        _mark_updates_consumed(update_headers)

    # 8. KG 수집 (일기 저장 성공 시에만, best-effort)
    if saved and raw_articles:
        print(f"  🧠 [KG] 뉴스 {len(raw_articles)}건 지식그래프 수집 시작...")
        _ingest_news_to_graph(raw_articles)
