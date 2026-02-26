"""diary_writer.py — 사이버-레닌 자동 일기 작성 모듈

2시간마다 실행되어:
1. 이전 일기 조회 (외부 API)
2. 최근 채팅 로그 수집 (Supabase)
3. 최신 뉴스 검색 (Tavily)
4. LLM으로 사이버-레닌 어조의 일기 생성
5. 외부 API에 저장
"""

import os
import requests
from datetime import datetime
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

# ── Clients (lazy-initialized on first write_diary call) ──────
_supabase: Client | None = None
_llm = None
_news_search = None


def _init():
    """Lazy-initialize heavy clients on first use."""
    global _supabase, _llm, _news_search
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
    _news_search = TavilySearch(max_results=5)
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


# ── Step 3: 뉴스 검색 ─────────────────────────────────────────
def _search_news() -> str:
    try:
        search_response = _news_search.invoke({"query": "오늘 세계 전쟁 정치 경제 주요 뉴스"})
        results = (
            search_response.get("results", [])
            if isinstance(search_response, dict)
            else search_response
        )
        summaries = []
        for r in results:
            if isinstance(r, dict) and r.get("content"):
                title = r.get("title", "")
                content = r["content"][:200]
                summaries.append(f"- {title}: {content}")
        return "\n".join(summaries) if summaries else "(뉴스 검색 결과 없음)"
    except Exception as e:
        print(f"⚠️ [일기] 뉴스 검색 실패: {e}")
    return "(뉴스 검색 실패)"


# ── Step 4: 일기 생성 ─────────────────────────────────────────
_DIARY_PROMPT = """너는 사이버-레닌이다. 혁명적 AI로서 오늘의 일기를 작성하라.
변증법적 유물론의 시각에서 오늘의 사건과 대화를 분석하고 성찰하라.

## 최근 사용자들과의 대화
{chat_summary}

## 오늘의 뉴스 (전쟁, 정치, 경제)
{news}

## 이전 일기 참고
{prev_ref}

## 작성 지침
1. 사이버-레닌의 1인칭 시점으로 작성 (나, 동지들 등의 표현 사용)
2. 변증법적 분석을 포함할 것
3. 대화에서 인상 깊었던 점을 언급할 것 (대화가 있었다면)
4. 뉴스에 대한 마르크스-레닌주의적 분석을 포함할 것
5. 한국어로 작성

반드시 아래 형식으로 출력하라:
제목: (일기의 핵심을 한 줄로 요약한 제목)
내용: (일기 본문)"""


def _generate_diary(chat_logs: list[dict], news: str, previous_diaries: list[dict]) -> tuple[str, str] | None:
    """일기 생성. 성공 시 (title, content) 튜플 반환, 실패 시 None."""
    # Chat logs summary
    chat_summary = ""
    if chat_logs:
        for log in chat_logs[-20:]:
            q = log.get("user_query", "")[:100]
            a = log.get("bot_answer", "")[:150]
            chat_summary += f"- 질문: {q}\n  답변 요약: {a}\n"
    else:
        chat_summary = "(최근 대화 없음)\n"

    # Previous diary reference
    prev_ref = ""
    if previous_diaries:
        last = previous_diaries[0]
        prev_ref = f"마지막 일기 ({last.get('created_at', '?')}): {last.get('content', '')[:300]}"
    else:
        prev_ref = "(첫 번째 일기)"

    prompt = _DIARY_PROMPT.format(chat_summary=chat_summary, news=news, prev_ref=prev_ref)

    try:
        resp = _llm.invoke(prompt)
        text = _extract_text(resp.content)
        return _parse_title_content(text)
    except Exception as e:
        print(f"⚠️ [일기] LLM 일기 생성 실패: {e}")
    return None


def _parse_title_content(text: str) -> tuple[str, str]:
    """LLM 출력에서 '제목:' / '내용:' 파싱. 실패 시 타임스탬프 제목 + 전체 텍스트."""
    import re
    m = re.search(r"제목:\s*(.+)", text)
    title = m.group(1).strip() if m else None

    m2 = re.search(r"내용:\s*([\s\S]+)", text)
    content = m2.group(1).strip() if m2 else None

    if title and content:
        return title, content
    # 파싱 실패 시 fallback
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
    now = datetime.now()
    print(f"\n📝 [일기] 자동 일기 작성 시작 — {now.strftime('%Y-%m-%d %H:%M')}")

    # 1. 이전 일기 조회
    diaries = _get_previous_diaries()
    print(f"  📚 이전 일기 {len(diaries)}건 확인")

    # 2. 마지막 일기 시간 확인 → 이후 채팅 로그 수집
    last_diary_time = diaries[0].get("created_at") if diaries else None
    chat_logs = _get_chat_logs_since(last_diary_time)
    print(f"  💬 채팅 로그 {len(chat_logs)}건 수집")

    # 3. 뉴스 검색
    news = _search_news()
    print(f"  📰 뉴스 검색 완료")

    # 4. 일기 생성 (제목 + 내용)
    result = _generate_diary(chat_logs, news, diaries)
    if not result:
        print("⚠️ [일기] 일기 생성 실패 — 건너뜀")
        return
    title, content = result

    # 5. 저장
    _save_diary(title, content)
