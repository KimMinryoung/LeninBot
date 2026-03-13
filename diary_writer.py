"""diary_writer.py — 사이버-레닌 자동 일기 작성 모듈

주기적으로 실행되어:
1. 이전 일기 조회 (외부 API)
2. 최근 채팅 로그 수집 (Supabase)
3. LLM 기반 동적 뉴스 검색 (Tavily)
4. LLM으로 사이버-레닌 어조의 일기 생성 (시간 인식, 맥락 요약)
5. 외부 API에 저장
"""

import os
import re
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from db import query as db_query
from langchain_google_genai import ChatGoogleGenerativeAI
from shared import (
    extract_text_content, KST, MODEL_MAIN, MODEL_LIGHT,
    get_tavily_search, get_kg_service, run_kg_async,
)

load_dotenv()

# ── Configuration ──────────────────────────────────────────────
AI_DIARY_API_URL = os.getenv("AI_DIARY_API_URL", "https://bichonwebpage.onrender.com/api/ai-diary")
AI_DIARY_API_KEY = os.getenv("AI_DIARY_API_KEY", "")
_HEADERS = {
    "X-API-Key": AI_DIARY_API_KEY,
    "Content-Type": "application/json",
}

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
        max_output_tokens=16384,
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
        resp = requests.get(AI_DIARY_API_URL, headers=_HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("data", [])
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


def _generate_recent_news_queries(last_diary_time: str | None, now: datetime) -> list[str]:
    """Generate 2 deterministic queries focused on latest war/politics/economy since last diary."""
    window = _build_recent_window_phrase(last_diary_time, now)
    date_tag = now.strftime("%Y-%m-%d")
    return [
        f"{window} War, Conflict, Military Conflict, Breaking News {date_tag}",
        f"{window} International, Political, Economic, Crisis, Sanctions, Interest Rates, Trade, Top News {date_tag}",
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
        ok, fail = 0, 0
        total = len(articles)
        for i, art in enumerate(articles, 1):
            short_title = art.get("title", "")[:50]
            print(f"  🔄 [KG] ({i}/{total}) 수집 중: {short_title}")
            try:
                body = f"Title: {art['title']}\nURL: {art['url']}\n\n{art['content']}"
                run_kg_async(svc.ingest_episode(
                    name=art["title"][:120],
                    body=body,
                    source_type="osint_news",
                    reference_time=now,
                    group_id="diary_news",
                    max_body_chars=1500,
                ))
                ok += 1
                print(f"  ✅ [KG] ({i}/{total}) 완료: {short_title}")
            except Exception as e:
                fail += 1
                print(f"  ⚠️ [KG] ({i}/{total}) 실패: {short_title} — {e}")
        print(f"  📊 [KG] 수집 완료: 성공 {ok}건, 실패 {fail}건")
    except Exception as e:
        print(f"⚠️ [KG] 지식그래프 수집 전체 실패 (일기에 영향 없음): {e}")


# ── Step 4: 일기 생성 ─────────────────────────────────────────
_DIARY_PROMPT = """You are Cyber-Lenin. Write a diary as a revolutionary.
Think through dialectical materialism, but keep philosophical jargon out of the writing unless essential.

## Current Time
{time_context}

## Recent conversations with users (Number of conversation: {n_logs})
{chat_summary}

## The latest news you searched yourself
{news}

## Recent Diary Summary (topics already covered — don't repeat them)
{prev_ref}

## Writing Guidelines
1. Write in the first-person perspective of a Cyber-Lenin (me, comrades, etc.).
2. Reflect the mood appropriate to the time period and current events.
3. Naturally acknowledge the passage of time since the last diary entry.
4. Mention specifically what impressed you in the conversation.
5. Include analysis of news articles you actively searched.
6. Prioritize new material and perspectives that do not overlap with previous diaries.
7. Write in Korean.

You MUST print in the following format:
제목: (A one-line summary of the journal entry's main points)
내용: (The journal entry's main body, at least two paragraphs)"""


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


def _generate_diary(
    chat_logs: list[dict],
    news: str,
    previous_diaries: list[dict],
    time_context: str,
) -> tuple[str, str] | None:
    """일기 생성. 성공 시 (title, content) 튜플 반환, 실패 시 None."""
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

    # Previous diaries summary (최근 5건) — LLM 요약 적용
    prev_ref = ""
    if previous_diaries:
        prev_lines = []
        for d in previous_diaries[:5]:
            ts = d.get("created_at", "?")
            title = d.get("title", "")
            body = d.get("content", "")
            body_summarized = _summarize(
                body,
                "Summarize the following diary entry in 2-3 lines, focusing on the key topics and arguments.",
                max_chars=500,
            )
            prev_lines.append(f"- [{ts}] {title}: {body_summarized}")
        prev_ref = "\n".join(prev_lines)
    else:
        prev_ref = "(First diary)"

    prompt = _DIARY_PROMPT.format(
        time_context=time_context,
        n_logs=n_logs,
        chat_summary=chat_summary,
        news=news,
        prev_ref=prev_ref,
    )

    try:
        resp = _llm.invoke(prompt)
        text = extract_text_content(resp.content)
        return _parse_title_content(text)
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
    fallback_title = f"{datetime.now().strftime('%Y-%m-%d %H:%M')} 일기"
    return fallback_title, text


# ── Step 5: 일기 저장 ─────────────────────────────────────────
def _save_diary(title: str, content: str) -> bool:
    try:
        resp = requests.post(
            AI_DIARY_API_URL,
            headers=_HEADERS,
            json={"title": title, "content": content},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            print(f"✅ [일기] 저장 성공: {title}")
            return True
        print(f"⚠️ [일기] 저장 실패 ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        print(f"⚠️ [일기] 저장 요청 실패: {e}")
    return False


# ── Main: 일기 작성 ───────────────────────────────────────────
def write_diary(dry_run: bool = False):
    """전체 일기 작성 파이프라인 실행. dry_run=True이면 저장 없이 CLI 출력만."""
    if not AI_DIARY_API_KEY:
        print("⚠️ [일기] AI_DIARY_API_KEY 미설정 — 건너뜀")
        return

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

    # 6. 일기 생성 (시간 맥락 + 요약 기반)
    result = _generate_diary(chat_logs, news, diaries, time_context)
    if not result:
        print("⚠️ [일기] 일기 생성 실패 — 건너뜀")
        return
    title, content = result

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

    # 8. KG 수집 (일기 저장 성공 시에만, best-effort)
    if saved and raw_articles:
        print(f"  🧠 [KG] 뉴스 {len(raw_articles)}건 지식그래프 수집 시작...")
        _ingest_news_to_graph(raw_articles)
