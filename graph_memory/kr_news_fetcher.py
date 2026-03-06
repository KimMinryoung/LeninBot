"""
한국 국내 뉴스 수집 + 인물 프로파일 보강 → 지식그래프 수집
==========================================================

Functions:
  fetch_kr_news()                  — Tavily로 한국 국내 뉴스 검색
  extract_persons_from_articles()  — LLM으로 기사에서 인물명 추출
  fetch_person_profile()           — Tavily로 인물 프로파일 검색
  ingest_news_to_kg()              — 뉴스 기사 KG 수집
  ingest_profile_to_kg()           — 프로파일 KG 수집
  run_full_pipeline()              — 전체 파이프라인 오케스트레이션
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message

from .service import GraphMemoryService


# ============================================================
# LLM 클라이언트 (인물 추출용)
# ============================================================

def _get_llm_client() -> GeminiClient:
    """인물 추출 전용 경량 LLM 클라이언트."""
    return GeminiClient(
        config=LLMConfig(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model="gemini-3.1-flash-lite-preview",
            small_model="gemini-2.5-flash-lite",
        )
    )


def _extract_text_from_response(response) -> str:
    """LLM 응답에서 텍스트 추출 (service.py 패턴 재사용)."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    for attr in ("text", "content", "output_text", "response"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(response, dict):
        for key in ("text", "content", "output_text", "response"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(response).strip()


def _parse_json_from_text(text: str):
    """텍스트에서 JSON 파싱 (코드펜스 처리)."""
    raw = text.strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # {} 또는 [] 찾기
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = raw.find(open_ch)
        end = raw.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


# ============================================================
# 1. 한국 국내 뉴스 수집
# ============================================================

_KOREA_KEYWORDS = {
    "korea", "korean", "seoul", "busan", "lee jae", "이재명",
    "yoon", "윤석열", "rok", "south korea", "pyongyang", "dprk",
}


def _is_korea_relevant(title: str, content: str) -> bool:
    """title + content 앞 500자에서 한국 관련 키워드를 확인한다."""
    text = (title + " " + content[:500]).lower()
    return any(kw in text for kw in _KOREA_KEYWORDS)


async def _extract_article_body(raw_content: str) -> str:
    """Tavily raw_content에서 기사 본문만 LLM으로 추출한다.

    사이드바, 광고, 관련 기사, 푸터 등 노이즈를 제거하고
    기사 제목 + 본문만 반환한다.
    """
    if not raw_content or len(raw_content) < 100:
        return raw_content or ""

    llm = _get_llm_client()
    prompt = (
        "Extract ONLY the main article headline and body text from the web page content below.\n"
        "Remove ALL of the following: navigation menus, sidebars, advertisements, "
        "related articles, footer, author bios, social media links, comments, "
        "cookie notices, subscription prompts.\n\n"
        "Return ONLY the cleaned article text (headline + body). "
        "Do NOT add any commentary or formatting.\n\n"
        f"[WEB PAGE CONTENT]\n{raw_content[:6000]}"
    )

    try:
        response = await llm.generate_response(
            [Message(role="user", content=prompt)],
            model_size=ModelSize.small,
        )
        extracted = _extract_text_from_response(response)
        if extracted and len(extracted) > 50:
            return extracted
    except Exception as e:
        print(f"⚠️ 본문 정제 실패, 원본 사용: {e}", flush=True)

    return raw_content


def _sanitize_filename(text: str, max_len: int = 60) -> str:
    """파일명에 사용할 수 없는 문자를 제거한다."""
    sanitized = re.sub(r'[<>:"/\\|?*\n\r\t]', '', text)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    return sanitized[:max_len]


async def fetch_kr_news(
    query: str = "South Korea president politics economy 2026",
    max_results: int = 5,
    time_range: str = "week",
    save_dir: str | None = "docs/news",
) -> list[dict]:
    """Tavily로 한국 국내 뉴스를 검색하고 본문을 정제한다.

    Args:
        save_dir: 정제된 기사를 저장할 디렉터리. None이면 저장하지 않음.

    Returns:
        [{"title": str, "url": str, "content": str}, ...]
    """
    tavily = TavilySearch(
        max_results=max_results,
        topic="news",
        search_depth="advanced",
        include_raw_content=True,
        time_range=time_range,
    )

    print(f"🔍 한국 뉴스 검색: '{query}' (max={max_results})", flush=True)
    try:
        raw = await tavily.ainvoke(query)
    except Exception as e:
        print(f"⚠️ Tavily 검색 실패: {e}", flush=True)
        return []

    if isinstance(raw, str):
        print(f"⚠️ Tavily 문자열 반환: {raw[:200]}", flush=True)
        return []

    results = raw.get("results", []) if isinstance(raw, dict) else []
    articles = []
    for r in results:
        content = r.get("raw_content") or r.get("content", "")
        if not content or len(content) < 50:
            continue
        articles.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": content,
        })

    # 관련성 필터: 한국 관련 키워드가 없는 기사 제거
    before = len(articles)
    articles = [a for a in articles if _is_korea_relevant(a["title"], a["content"])]
    if before != len(articles):
        print(f"🔎 관련성 필터: {before}건 → {len(articles)}건", flush=True)

    # 본문 정제: LLM으로 사이드바/광고/관련기사 노이즈 제거
    print(f"🧹 {len(articles)}건 본문 정제 중...", flush=True)
    for art in articles:
        art["content"] = await _extract_article_body(art["content"])

    # docs/news/에 정제된 기사 저장
    if save_dir and articles:
        save_path = Path(save_dir)
        if not save_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            save_path = project_root / save_dir
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, art in enumerate(articles):
            fname = f"{timestamp}_{i}_{_sanitize_filename(art['title'])}.txt"
            fpath = save_path / fname
            file_content = f"{art['url']}\n{art['title']}\n---\n{art['content']}"
            fpath.write_text(file_content, encoding="utf-8")
        print(f"💾 {len(articles)}건 저장 → {save_path}", flush=True)

    print(f"📰 {len(articles)}건 유효 기사 수신", flush=True)
    return articles


# ============================================================
# 2. 기사에서 인물명 추출
# ============================================================

PERSON_EXTRACTION_PROMPT = """\
Extract **key South Korean public figures** mentioned in the news articles below.

[RULES]
1. Only South Korean nationals or people active in South Korea
2. Prioritize: politicians, business leaders, senior officials, party leaders
3. Deduplicate (each person appears once)
4. Maximum 5 persons
5. If no South Korean person is explicitly named, look for implied references to South Korean leaders, presidents, party heads, etc.
6. You MUST return at least 1 person if the article mentions South Korea at all — infer the most relevant leader (e.g., the current South Korean president or opposition leader)

[OUTPUT FORMAT — JSON array ONLY, no other text]
[
  {{"name_ko": "이재명", "name_en": "Lee Jae-myung", "role": "Leader of the Democratic Party of Korea"}},
  ...
]

[ARTICLES]
{articles_text}
"""


async def extract_persons_from_articles(articles: list[dict]) -> list[dict]:
    """뉴스 기사 본문에서 한국 인물명을 LLM으로 추출한다.

    Returns:
        [{"name_ko": str, "name_en": str, "role": str}, ...]
    """
    # 기사 본문 합치기 (각 최대 800자)
    parts = []
    for i, art in enumerate(articles):
        snippet = art["content"][:800]
        parts.append(f"--- 기사 {i+1}: {art['title']} ---\n{snippet}")
    articles_text = "\n\n".join(parts)

    llm = _get_llm_client()
    prompt = PERSON_EXTRACTION_PROMPT.format(articles_text=articles_text)

    print("🧠 LLM 인물 추출 중...", flush=True)
    response = await llm.generate_response([Message(role="user", content=prompt)])
    text = _extract_text_from_response(response)
    parsed = _parse_json_from_text(text)

    if not isinstance(parsed, list):
        print(f"⚠️ JSON 파싱 실패, raw: {text[:300]}", flush=True)
        return []

    persons = []
    for item in parsed:
        if isinstance(item, dict) and "name_ko" in item:
            persons.append({
                "name_ko": item.get("name_ko", ""),
                "name_en": item.get("name_en", ""),
                "role": item.get("role", "unknown"),
            })

    print(f"👤 {len(persons)}명 추출 완료", flush=True)
    return persons


# ============================================================
# 3. 인물 프로파일 Tavily 검색
# ============================================================

PROFILE_SYNTHESIS_PROMPT = """\
Synthesize the search results below into a person profile.

CRITICAL: The search results below contain the MOST RECENT information.
If the search results state a person holds a specific position (e.g., president),
ALWAYS use that information, even if it differs from your training data.
Your training data may be outdated — the search results are authoritative.

[RULES]
1. Write in English (for knowledge graph ingestion)
2. ONLY use facts from the search results below (no speculation, no training data)
3. Structure: name, CURRENT role/position, party/organization, career highlights, recent activities
4. For the "role" field, use the person's CURRENT position as stated in the search results
5. For the "summary" field, START with the person's CURRENT position as stated in search results

[OUTPUT FORMAT — JSON only, no other text]
{{
  "name": "Lee Jae-myung",
  "name_ko": "이재명",
  "role": "President of South Korea",
  "party_or_org": "Democratic Party of Korea",
  "nationality": "KR",
  "summary": "Lee Jae-myung is the President of South Korea ... (200-400 words, fact-based, English)"
}}

[SEARCH RESULTS]
{search_results}
"""


async def fetch_person_profile(
    name_ko: str,
    name_en: str,
) -> dict:
    """Tavily 검색으로 인물 프로파일을 수집 → LLM으로 구조화한다.

    Returns:
        {"name": str, "name_ko": str, "role": str,
         "party_or_org": str, "nationality": str, "summary": str}
    """
    # Tavily 일반 검색 (뉴스가 아닌 general)
    tavily = TavilySearch(
        max_results=5,
        topic="general",
        search_depth="advanced",
        include_raw_content=True,
    )

    query = f"{name_en} {name_ko} current position role president 2026"
    print(f"🔍 프로파일 검색: '{query}'", flush=True)

    try:
        raw = await tavily.ainvoke(query)
    except Exception as e:
        print(f"⚠️ Tavily 프로파일 검색 실패: {e}", flush=True)
        return {"name": name_en, "name_ko": name_ko, "summary": f"Profile search failed: {e}"}

    if isinstance(raw, str):
        results_text = raw[:3000]
    else:
        items = raw.get("results", []) if isinstance(raw, dict) else []
        parts = []
        for r in items:
            content = r.get("raw_content") or r.get("content", "")
            parts.append(f"[{r.get('title', '')}]\n{content[:600]}")
        results_text = "\n\n".join(parts)

    if not results_text or len(results_text) < 30:
        return {
            "name": name_en, "name_ko": name_ko,
            "role": "unknown", "party_or_org": "unknown",
            "nationality": "KR",
            "summary": f"Insufficient search results for {name_en}.",
        }

    # LLM으로 프로파일 종합
    llm = _get_llm_client()
    prompt = PROFILE_SYNTHESIS_PROMPT.format(search_results=results_text[:4000])

    print("🧠 LLM 프로파일 종합 중...", flush=True)
    response = await llm.generate_response([Message(role="user", content=prompt)])
    text = _extract_text_from_response(response)
    parsed = _parse_json_from_text(text)

    if isinstance(parsed, dict) and "summary" in parsed:
        parsed.setdefault("name", name_en)
        parsed.setdefault("name_ko", name_ko)
        parsed.setdefault("nationality", "KR")
        print(f"✅ 프로파일 종합 완료: {parsed['name']}", flush=True)
        return parsed

    # 파싱 실패 시 raw text를 summary로
    return {
        "name": name_en, "name_ko": name_ko,
        "role": "unknown", "party_or_org": "unknown",
        "nationality": "KR",
        "summary": text[:500] if text else "Profile synthesis failed.",
    }


# ============================================================
# 4. 뉴스 기사 KG 수집
# ============================================================

async def ingest_news_to_kg(
    service: GraphMemoryService,
    articles: list[dict],
    group_id: str = "korea_domestic",
    delay_between: float = 5.0,
) -> dict:
    """뉴스 기사 목록을 KG에 에피소드로 수집한다.

    Returns:
        {"succeeded": int, "failed": int, "articles": [{"title", "status"}]}
    """
    ref_time = datetime.now(timezone.utc)
    succeeded = 0
    failed = 0
    article_log = []

    for i, art in enumerate(articles):
        title = art["title"]
        body = f"{title}\n\n{art['content']}"
        name = f"kr-news-{i}"

        print(f"\n[{i+1}/{len(articles)}] 뉴스 수집: {title[:60]}", flush=True)

        try:
            await service.ingest_episode(
                name=name,
                body=body,
                source_type="osint_news",
                reference_time=ref_time,
                group_id=group_id,
                source_description=f"Korean news: {art.get('url', '')}",
                preprocess_news=True,
                max_body_chars=1500,
            )
            succeeded += 1
            article_log.append({"title": title[:120], "status": "ok"})
            print(f"  ✅ 수집 완료", flush=True)
        except Exception as e:
            import traceback
            failed += 1
            err_msg = f"{type(e).__name__}: {e}"
            article_log.append({"title": title[:120], "status": f"failed: {err_msg}"})
            print(f"  ❌ 실패: {err_msg[:300]}", flush=True)
            traceback.print_exc()

        # 다음 에피소드 전 대기 (rate limit)
        if i < len(articles) - 1:
            print(f"  ⏳ {delay_between}초 대기...", flush=True)
            await asyncio.sleep(delay_between)

    print(f"\n📊 뉴스 수집 결과: {succeeded} 성공, {failed} 실패", flush=True)
    return {"succeeded": succeeded, "failed": failed, "articles": article_log}


# ============================================================
# 5. 프로파일 KG 수집
# ============================================================

async def ingest_profile_to_kg(
    service: GraphMemoryService,
    profile: dict,
    group_id: str = "korea_domestic",
) -> dict:
    """인물 프로파일을 KG에 에피소드로 수집한다.

    Returns:
        {"status": "ok"|"failed", "name": str}
    """
    name = profile.get("name", "unknown")
    summary = profile.get("summary", "")
    role = profile.get("role", "")
    party = profile.get("party_or_org", "")

    body = (
        f"Person Profile: {name}\n"
        f"Korean name: {profile.get('name_ko', '')}\n"
        f"Role: {role}\n"
        f"Organization: {party}\n"
        f"Nationality: {profile.get('nationality', 'KR')}\n\n"
        f"{summary}"
    )

    print(f"📝 프로파일 KG 수집: {name}", flush=True)
    ref_time = datetime.now(timezone.utc)

    try:
        await service.ingest_episode(
            name=f"profile-{name.lower().replace(' ', '-')}",
            body=body,
            source_type="osint_news",
            reference_time=ref_time,
            group_id=group_id,
            source_description=f"Person profile: {name}",
            preprocess_news=False,   # 프로파일은 이미 정제됨
            max_body_chars=2000,
        )
        print(f"  ✅ 프로파일 수집 완료: {name}", flush=True)
        return {"status": "ok", "name": name}
    except Exception as e:
        print(f"  ❌ 프로파일 수집 실패: {str(e)[:200]}", flush=True)
        return {"status": "failed", "name": name, "error": str(e)}


# ============================================================
# 전체 파이프라인 (standalone 실행용)
# ============================================================

async def run_full_pipeline():
    """전체 파이프라인: 뉴스 수집 → 인물 추출 → 프로파일 보강 → KG 수집."""
    service = GraphMemoryService()
    await service.initialize()

    try:
        # 1. 한국 뉴스 3건 수집
        print("\n" + "=" * 60)
        print("Phase 1: 한국 국내 뉴스 수집")
        print("=" * 60)
        articles = await fetch_kr_news(
            max_results=5,
            time_range="week",
        )
        if not articles:
            print("❌ 뉴스 수집 실패. 중단.")
            return

        # 2. 인물 추출
        print("\n" + "=" * 60)
        print("Phase 2: 인물명 추출")
        print("=" * 60)
        persons = await extract_persons_from_articles(articles)
        print(f"추출된 인물: {[p['name_ko'] for p in persons]}")

        # 3. 뉴스 KG 수집 (최대 3건)
        print("\n" + "=" * 60)
        print("Phase 3: 뉴스 KG 수집")
        print("=" * 60)
        news_result = await ingest_news_to_kg(
            service=service,
            articles=articles[:3],
            group_id="korea_domestic",
        )

        # 4. 인물 프로파일 보강 + KG 수집
        print("\n" + "=" * 60)
        print("Phase 4: 인물 프로파일 보강 → KG 수집")
        print("=" * 60)
        for p in persons[:2]:  # 상위 2명
            print(f"\n--- 프로파일 수집: {p['name_ko']} ---")
            await asyncio.sleep(5)  # rate limit 대기
            profile = await fetch_person_profile(p["name_ko"], p["name_en"])
            await asyncio.sleep(5)
            await ingest_profile_to_kg(service, profile, group_id="korea_domestic")

        # 5. 검증
        print("\n" + "=" * 60)
        print("Phase 5: KG 검증")
        print("=" * 60)
        search_result = await service.search(
            query="South Korea politics economy leader president party",
            group_ids=["korea_domestic"],
            num_results=15,
        )
        print(f"노드 {len(search_result['nodes'])}개, 엣지 {len(search_result['edges'])}개")
        for n in search_result["nodes"][:10]:
            print(f"  [node] {n['name']} ({', '.join(n['labels'])})")
        for e in search_result["edges"][:10]:
            print(f"  [edge] {e['fact'][:80]}")

        print("\n✅ 전체 파이프라인 완료!")

    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(run_full_pipeline())
