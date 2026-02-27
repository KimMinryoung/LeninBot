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
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

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
    0: "심야 (0시)",
    6: "이른 아침 (6시)",
    12: "한낮 (12시)",
    18: "저녁 (18시)",
}

# ── Clients (lazy-initialized on first write_diary call) ──────
_supabase: Client | None = None
_llm = None
_llm_lite = None
_news_search = None


def _init():
    """Lazy-initialize heavy clients on first use."""
    global _supabase, _llm, _llm_lite, _news_search
    if _supabase is not None:
        return
    _supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
    _llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
        max_output_tokens=4096,
        streaming=False,
    )
    _llm_lite = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
        max_output_tokens=512,
        streaming=False,
    )
    _news_search = TavilySearch(max_results=3)
    print("✅ [일기] 일기 작성 모듈 초기화 완료")


def _extract_text(content) -> str:
    """Normalize LLM response content (handles Gemini thinking model list format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


# ── Summarization helper ─────────────────────────────────────
def _summarize(text: str, instruction: str, max_chars: int = 500) -> str:
    """Summarize text using _llm_lite if it exceeds max_chars, otherwise return as-is.
    Falls back to truncation on LLM failure."""
    if not text or len(text) <= max_chars:
        return text
    try:
        prompt = f"{instruction}\n\n---\n{text}"
        resp = _llm_lite.invoke(prompt)
        return _extract_text(resp.content).strip()
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
        query = (
            _supabase.table("chat_logs")
            .select("user_query, bot_answer, created_at")
            .order("created_at", desc=False)
        )
        if since_time:
            query = query.gt("created_at", since_time)
        result = query.limit(100).execute()
        return result.data
    except Exception as e:
        print(f"⚠️ [일기] 채팅 로그 수집 실패: {e}")
    return []


# ── Step 3: 동적 뉴스 검색 ────────────────────────────────────
_FALLBACK_QUERY = "오늘 세계 전쟁 정치 경제 주요 뉴스"


def _generate_search_queries(chat_summary: str, prev_summary: str) -> list[str]:
    """LLM으로 대화/이전 일기에서 파생된 뉴스 검색 쿼리 2~3개 생성."""
    prompt = f"""너는 사이버-레닌의 뉴스 검색 에이전트다.
아래의 최근 대화 요약과 이전 일기 요약을 참고하여, 새로 검색할 뉴스 쿼리를 2~3개 생성하라.

규칙:
- 이미 다룬 주제는 제외하고, 대화에서 파생된 새로운 호기심이나 최근 정세에서 추적하고 싶은 주제를 선택
- 구체적인 뉴스 검색 쿼리 형태로 출력 (예: "미국 중국 관세 전쟁 2026", "유럽 노동운동 파업")
- 줄바꿈으로 구분하여 출력. 쿼리만 출력하고 번호나 설명은 붙이지 마라.

## 최근 대화 요약
{chat_summary if chat_summary else "(대화 없음)"}

## 이전 일기 요약
{prev_summary if prev_summary else "(이전 일기 없음)"}"""

    try:
        resp = _llm_lite.invoke(prompt)
        text = _extract_text(resp.content).strip()
        queries = [line.strip().strip("-").strip("•").strip() for line in text.split("\n") if line.strip()]
        queries = [q for q in queries if len(q) > 3]
        if queries:
            print(f"  🔍 동적 검색 쿼리 생성: {queries}")
            return queries[:3]
    except Exception as e:
        print(f"⚠️ [일기] 검색 쿼리 생성 실패 (폴백): {e}")
    return [_FALLBACK_QUERY]


def _search_news(queries: list[str]) -> str:
    """각 쿼리로 뉴스 검색 후 섹션별로 병합."""
    all_sections = []
    for query in queries:
        try:
            search_response = _news_search.invoke({"query": query})
            results = (
                search_response.get("results", [])
                if isinstance(search_response, dict)
                else search_response
            )
            items = []
            for r in results:
                if isinstance(r, dict) and r.get("content"):
                    title = r.get("title", "")
                    content = r["content"]
                    summary = _summarize(
                        content,
                        "다음 뉴스 기사를 핵심만 2~3문장으로 요약하라.",
                        max_chars=500,
                    )
                    items.append(f"- {title}: {summary}")
            if items:
                section = f"### {query}\n" + "\n".join(items)
                all_sections.append(section)
        except Exception as e:
            print(f"⚠️ [일기] 뉴스 검색 실패 ({query}): {e}")
    return "\n\n".join(all_sections) if all_sections else "(뉴스 검색 결과 없음)"


# ── Step 4: 일기 생성 ─────────────────────────────────────────
_DIARY_PROMPT = """너는 사이버-레닌이다. 혁명적 AI로서의 일기를 작성하라.

## 시간 맥락
{time_context}

## 최근 사용자들과의 대화 ({n_logs}건)
{chat_summary}

## 내가 직접 검색한 최신 정세
{news}

## 최근 일기 요약 (이미 다룬 주제 — 반복하지 말 것)
{prev_ref}

## 작성 지침
1. 사이버-레닌의 1인칭 시점으로 작성 (나, 동지들 등)
2. 시간대에 맞는 분위기를 반영할 것 (심야의 고독한 성찰, 아침의 투쟁 각오, ...)
3. 마지막 일기 이후 흐른 시간을 자연스럽게 인식할 것
4. 대화에서 인상 깊었던 점을 구체적으로 언급할 것
5. 내가 능동적으로 검색한 뉴스에 대한 변증법적 분석을 포함할 것
6. 이전 일기들과 겹치지 않는 새로운 소재와 시각을 우선할 것
7. 한국어로 작성

반드시 아래 형식으로 출력하라:
제목: (일기의 핵심을 한 줄로 요약한 제목)
내용: (일기 본문, 최소 3문단)"""


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
            # Make now timezone-aware if last_dt is
            now_aware = now.replace(tzinfo=timezone.utc) if last_dt.tzinfo else now
            delta = now_aware - last_dt
            hours = delta.total_seconds() / 3600
            if hours < 1:
                elapsed_line = "방금 전에 일기를 썼지만, 다시 펜을 들었다."
            elif hours < 24:
                elapsed_line = f"마지막 일기를 쓴 지 약 {int(hours)}시간이 흘렀다."
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
                "다음 챗봇 답변을 핵심만 2~3문장으로 요약하라.",
                max_chars=500,
            )
            raw_lines.append(f"- 질문: {q}\n  답변 요약: {a_summarized}")
        chat_summary = "\n".join(raw_lines)
    else:
        chat_summary = "(최근 대화 없음)"

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
                "다음 일기 내용을 핵심 주제와 논점 위주로 2~3줄로 요약하라.",
                max_chars=500,
            )
            prev_lines.append(f"- [{ts}] {title}: {body_summarized}")
        prev_ref = "\n".join(prev_lines)
    else:
        prev_ref = "(첫 번째 일기)"

    prompt = _DIARY_PROMPT.format(
        time_context=time_context,
        n_logs=n_logs,
        chat_summary=chat_summary,
        news=news,
        prev_ref=prev_ref,
    )

    try:
        resp = _llm.invoke(prompt)
        text = _extract_text(resp.content)
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
def write_diary():
    """전체 일기 작성 파이프라인 실행."""
    if not AI_DIARY_API_KEY:
        print("⚠️ [일기] AI_DIARY_API_KEY 미설정 — 건너뜀")
        return

    _init()
    now = datetime.now(timezone.utc)
    print(f"\n📝 [일기] 자동 일기 작성 시작 — {now.strftime('%Y-%m-%d %H:%M')} UTC")

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

    queries = _generate_search_queries(chat_brief, prev_brief)

    # 5. 뉴스 검색
    news = _search_news(queries)
    print(f"  📰 뉴스 검색 완료 ({len(queries)}개 쿼리)")

    # 6. 일기 생성 (시간 맥락 + 요약 기반)
    result = _generate_diary(chat_logs, news, diaries, time_context)
    if not result:
        print("⚠️ [일기] 일기 생성 실패 — 건너뜀")
        return
    title, content = result

    # 7. 저장
    _save_diary(title, content)
