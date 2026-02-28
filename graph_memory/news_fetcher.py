"""
news_fetcher — Tavily 뉴스 검색 → 지식그래프 수집 유틸리티
==========================================================

GraphMemoryService와 함께 사용하여 최신 뉴스를 자동으로
에피소드로 변환·수집하는 재사용 가능한 함수를 제공한다.
"""

import asyncio
from datetime import datetime, timezone

from langchain_tavily import TavilySearch

from .service import GraphMemoryService

# Graphiti add_episode 시 Gemini가 올바른 JSON을 생성할 수 있는 안전 본문 길이.
# 길수록 엔티티/엣지가 많아져 JSON 구문 오류 발생 확률이 급증한다.
# (gemini-2.5-flash max_output_tokens=65536이므로 토큰 한도는 아니지만,
#  긴 JSON 생성 시 모델이 구문 실수를 저지름)
MAX_INGEST_BODY_CHARS = 1500

# Graphiti 에피소드 1건당 ~15-20 LLM 호출 발생.
# Tier 1 RPM 한도를 고려해 에피소드 간 충분한 대기 필요.
DEFAULT_DELAY_BETWEEN = 30

# 재시도 가능한 에러 키워드 (Gemini API 에러 메시지 기준)
_RETRYABLE_KEYWORDS = [
    "rate limit", "429", "resource_exhausted",  # rate limit
    "503", "unavailable",                        # server overload
    "500", "internal",                           # transient server error
]


def _is_retryable_error(err_msg: str) -> bool:
    """에러 메시지가 재시도 가능한 서버 에러인지 판별."""
    lower = err_msg.lower()
    return any(kw in lower for kw in _RETRYABLE_KEYWORDS)


async def fetch_and_ingest_news(
    service: GraphMemoryService,
    query: str,
    group_id: str = "geopolitics_conflict",
    max_results: int = 5,
    time_range: str = "day",
    delay_between: float = DEFAULT_DELAY_BETWEEN,
    preprocess_news: bool = True,
    max_body_chars: int = MAX_INGEST_BODY_CHARS,
) -> dict:
    """Tavily 뉴스 검색 → 에피소드 변환 → 그래프 수집.

    Args:
        service: 초기화된 GraphMemoryService 인스턴스.
        query: Tavily 검색 쿼리.
        group_id: 에피소드 그룹 ID (e.g., 'geopolitics_conflict').
        max_results: 검색할 최대 기사 수.
        time_range: 검색 기간 ('day' | 'week' | 'month').
        delay_between: 에피소드 간 대기 시간(초). Gemini rate limit 대응.
        preprocess_news: 뉴스 본문을 LLM으로 팩트 목록으로 정제할지 여부.
        max_body_chars: 수집 본문 최대 길이. 초과 시 truncate.

    Returns:
        {"succeeded": int, "failed": int,
         "articles": [{"title": str, "url": str, "status": str}]}
    """
    # 1. Tavily 뉴스 검색
    tavily = TavilySearch(
        max_results=max_results,
        topic="news",
        search_depth="advanced",
        include_raw_content=True,
        time_range=time_range,
    )

    print(f"🔍 Tavily 검색: '{query}' (max={max_results}, range={time_range})", flush=True)
    try:
        raw_results = await tavily.ainvoke(query)
    except Exception as e:
        print(f"⚠️ Tavily 검색 실패: {e}", flush=True)
        return {"succeeded": 0, "failed": 0, "articles": []}

    # ainvoke는 dict{"results": [...]} 반환, 에러 시 str
    if isinstance(raw_results, str):
        print(f"⚠️ Tavily가 문자열을 반환함: {raw_results[:200]}", flush=True)
        return {"succeeded": 0, "failed": 0, "articles": []}

    articles = raw_results.get("results", []) if isinstance(raw_results, dict) else []
    print(f"📰 {len(articles)}건 기사 수신", flush=True)

    # 2. 각 기사 → 에피소드 수집
    MAX_RETRIES = 3
    succeeded = 0
    failed = 0
    article_log = []
    ref_time = datetime.now(timezone.utc)

    for i, article in enumerate(articles):
        title = article.get("title", article.get("url", f"article-{i}"))
        url = article.get("url", "")
        raw_content = article.get("raw_content") or article.get("content", "")

        body = f"{title}\n\n{raw_content}" if raw_content else title
        name = f"news-{query[:30].replace(' ', '-')}-{i}"

        print(f"\n[{i+1}/{len(articles)}] {title[:80]}", flush=True)
        print(f"    [raw] 원본 {len(body)}자", flush=True)

        # 전처리 전 원본이 너무 길면 truncate
        if len(body) > max_body_chars * 10:
            body = body[: max_body_chars * 10]
            print(f"    [raw] → {len(body)}자로 잘라냄 (전처리 입력 한도)", flush=True)

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await service.ingest_episode(
                    name=name,
                    body=body,
                    source_type="osint_news",
                    reference_time=ref_time,
                    group_id=group_id,
                    source_description=f"Tavily news: {url}",
                    preprocess_news=preprocess_news,
                    max_body_chars=max_body_chars,
                )
                print(f"  ✅ 수집 완료 (시도 {attempt})", flush=True)
                success = True
                succeeded += 1
                break
            except Exception as e:
                err_msg = str(e)
                if _is_retryable_error(err_msg):
                    wait = 30 * attempt  # 30, 60, 90초
                    print(f"  ⚠️ 서버 에러 (시도 {attempt}/{MAX_RETRIES}). {wait}초 대기...", flush=True)
                    print(f"      {err_msg[:120]}", flush=True)
                    await asyncio.sleep(wait)
                else:
                    print(f"  ❌ 실패: {err_msg[:200]}", flush=True)
                    break

        if not success:
            failed += 1

        article_log.append({
            "title": title[:120],
            "url": url,
            "status": "ok" if success else "failed",
        })

        # 다음 에피소드 전 대기
        if i < len(articles) - 1 and success:
            print(f"  ⏳ {delay_between}초 대기...", flush=True)
            await asyncio.sleep(delay_between)

    print(f"\n📊 결과: {succeeded} 성공, {failed} 실패 / 총 {len(articles)}건", flush=True)
    return {"succeeded": succeeded, "failed": failed, "articles": article_log}
