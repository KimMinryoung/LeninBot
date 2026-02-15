import requests
from bs4 import BeautifulSoup
import os
import time
import re
import argparse
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, quote

# 설정
OUTPUT_DIR = "./docs/modern_analysis"
DELAY = 0.3
MAX_DEPTH = 3

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

visited_urls = set()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def get_soup(url):
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        res.encoding = res.apparent_encoding
        return BeautifulSoup(res.text, 'html.parser')
    except Exception as e:
        print(f"  접속 실패: {url} ({e})")
        return None


def save_document(url, title, content, prefix):
    """문서를 txt 파일로 저장합니다."""
    if len(content) < 300:
        return False

    # 영문 slug 파일명 생성
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', title[:80]).strip('_').lower()
    if not slug:
        slug = re.sub(r'[^a-zA-Z0-9]+', '_', url.split('/')[-1])[:60].strip('_').lower()
    file_name = f"{prefix}_{slug}.txt"
    save_path = os.path.join(OUTPUT_DIR, file_name)

    # 중복 파일명 방지
    counter = 1
    while os.path.exists(save_path):
        file_name = f"{prefix}_{slug}_{counter}.txt"
        save_path = os.path.join(OUTPUT_DIR, file_name)
        counter += 1

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"Source: {url}\nTitle: {title}\n\n{content}")
        print(f"  저장: {file_name} ({title[:50]})")
        return True
    except Exception as e:
        print(f"  저장 실패: {e}")
        return False


# ─────────────────────────────────────────────
# 1. marxists.org 현대 분석 섹션
# ─────────────────────────────────────────────

MARXISTS_SOURCES = [
    ("https://www.marxists.org/subject/economy/index.htm", "marxists.org/subject/economy/", "mxo_economy"),
    ("https://www.marxists.org/subject/imperialism/index.htm", "marxists.org/subject/imperialism/", "mxo_imperialism"),
    ("https://www.marxists.org/subject/war/index.htm", "marxists.org/subject/war/", "mxo_war"),
    ("https://www.marxists.org/subject/science/index.htm", "marxists.org/subject/science/", "mxo_science"),
    ("https://www.marxists.org/archive/mandel/index.htm", "marxists.org/archive/mandel/", "mxo_mandel"),
    ("https://www.marxists.org/reference/archive/marcuse/index.htm", "marxists.org/reference/archive/marcuse/", "mxo_marcuse"),
]


def is_content_page(url, soup):
    text_content = soup.get_text()
    return len(text_content) > 1500 and not url.endswith('index.htm')


def crawl_marxists_recursive(url, url_filter, prefix, depth):
    if depth > MAX_DEPTH or url in visited_urls:
        return
    visited_urls.add(url)
    soup = get_soup(url)
    if not soup: return

    if is_content_page(url, soup):
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        for nav in soup.find_all(['nav', 'header', 'footer', 'table']):
            nav.decompose()
        paragraphs = soup.find_all(['p', 'h3', 'h4', 'blockquote'])
        content = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text()) > 30])
        save_document(url, title, content, prefix)
        time.sleep(DELAY)
        return

    print(f"  탐색 중 (Depth {depth}): {url}")
    links = soup.find_all('a', href=True)
    for a in links:
        href = a['href']
        full_url = urljoin(url, href).split('#')[0]
        if url_filter in full_url:
            if full_url.endswith('.htm') or full_url.endswith('.html') or full_url.endswith('/'):
                crawl_marxists_recursive(full_url, url_filter, prefix, depth + 1)


def crawl_marxists():
    print("\n[marxists.org] 현대 분석 섹션 크롤링 시작")
    for start_url, url_filter, prefix in MARXISTS_SOURCES:
        print(f"\n  소스: {start_url}")
        crawl_marxists_recursive(start_url, url_filter, prefix, 0)
    print("[marxists.org] 완료")


# ─────────────────────────────────────────────
# 2. 나무위키 한국어 문서
# ─────────────────────────────────────────────

NAMUWIKI_ARTICLES = {
    "자본주의": "capitalism",
    "신자유주의": "neoliberalism",
    "제국주의": "imperialism",
    "헤지펀드": "hedge_fund",
    "세계화": "globalization",
    "2008년 세계금융위기": "2008_financial_crisis",
    "금융위기": "financial_crisis",
    "양적완화": "quantitative_easing",
    "군산복합체": "military_industrial_complex",
    "전쟁경제": "war_economy",
    "인공지능": "artificial_intelligence",
    "자동화": "automation",
    "기술적 실업": "technological_unemployment",
    "4차 산업혁명": "fourth_industrial_revolution",
    "플랫폼 경제": "platform_economy",
    "긱 경제": "gig_economy",
    "프라이버시": "privacy",
}


def crawl_namuwiki():
    print("\n[나무위키] 한국어 문서 크롤링 시작")
    count = 0
    for article_name, slug in NAMUWIKI_ARTICLES.items():
        url = f"https://namu.wiki/w/{quote(article_name)}"
        print(f"  크롤링: {article_name}")

        soup = get_soup(url)
        if not soup:
            continue

        # 나무위키 본문 추출
        body = soup.find('article') or soup.find('div', class_='wiki-heading-content') or soup
        # 불필요한 요소 제거
        for tag in body.find_all(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        paragraphs = body.find_all(['p', 'div', 'h2', 'h3', 'h4', 'li'])
        content_parts = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 20:
                content_parts.append(text)

        content = "\n\n".join(content_parts)
        if save_document(url, article_name, content, f"namu_{slug}"):
            count += 1
        time.sleep(DELAY)

    print(f"[나무위키] 완료 ({count}개 문서)")


# ─────────────────────────────────────────────
# 3. arXiv 논문 초록
# ─────────────────────────────────────────────

ARXIV_QUERIES = [
    "artificial intelligence labor",
    "AI inequality",
    "automation unemployment",
    "technology social impact",
]
ARXIV_MAX_RESULTS = 50


def crawl_arxiv():
    print("\n[arXiv] 논문 초록 크롤링 시작")
    count = 0
    for query in ARXIV_QUERIES:
        print(f"  검색: {query}")
        encoded_query = quote(query)
        api_url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{encoded_query}&start=0&max_results={ARXIV_MAX_RESULTS}"
        )

        try:
            res = requests.get(api_url, headers=headers, timeout=30)
            res.raise_for_status()
        except Exception as e:
            print(f"  arXiv API 실패: {e}")
            continue

        root = ET.fromstring(res.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entries = root.findall('atom:entry', ns)
        print(f"  결과: {len(entries)}건")

        for entry in entries:
            title_el = entry.find('atom:title', ns)
            summary_el = entry.find('atom:summary', ns)
            link_el = entry.find('atom:id', ns)

            if title_el is None or summary_el is None:
                continue

            title = ' '.join(title_el.text.strip().split())
            abstract = summary_el.text.strip()
            paper_url = link_el.text.strip() if link_el is not None else ""

            # 저자 추출
            authors = []
            for author in entry.findall('atom:author', ns):
                name_el = author.find('atom:name', ns)
                if name_el is not None:
                    authors.append(name_el.text.strip())
            author_str = ", ".join(authors[:5])
            if len(authors) > 5:
                author_str += " et al."

            content = f"Authors: {author_str}\n\nAbstract:\n{abstract}"
            slug = query.replace(' ', '_')
            if save_document(paper_url, title, content, f"arxiv_{slug}"):
                count += 1

        time.sleep(3)  # arXiv API 예의

    print(f"[arXiv] 완료 ({count}편)")


# ─────────────────────────────────────────────
# 4. BIS 워킹페이퍼
# ─────────────────────────────────────────────

BIS_START = 1100
BIS_END = 1230  # 최근 ~130편


def crawl_bis():
    print("\n[BIS] 워킹페이퍼 크롤링 시작")
    print(f"  범위: work{BIS_START} ~ work{BIS_END}")
    count = 0

    for num in range(BIS_START, BIS_END + 1):
        paper_url = f"https://www.bis.org/publ/work{num}.htm"
        paper_soup = get_soup(paper_url)
        if not paper_soup:
            continue

        # 제목 추출
        title_tag = paper_soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else f"BIS Working Paper {num}"

        # #body div에서 Abstract 섹션 추출
        body_div = paper_soup.find('div', id='body')
        if not body_div:
            continue

        body_text = body_div.get_text(separator='\n', strip=True)
        # Abstract ~ JEL/Keywords 사이 텍스트 추출
        abstract_match = re.search(
            r'Abstract\s*\n(.*?)(?:\nJEL|\nKeywords|\nThe views expressed)',
            body_text, re.DOTALL
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()
        else:
            # fallback: Summary + Focus + Contribution + Findings
            sections = []
            for section in ['Summary', 'Focus', 'Contribution', 'Findings']:
                match = re.search(
                    rf'{section}\s*\n(.*?)(?:\n(?:Focus|Contribution|Findings|Abstract|JEL|Keywords|The views)|\Z)',
                    body_text, re.DOTALL
                )
                if match:
                    sections.append(f"{section}:\n{match.group(1).strip()}")
            abstract = "\n\n".join(sections) if sections else ""

        if not abstract:
            continue

        content = f"Abstract:\n{abstract}"
        if save_document(paper_url, title, content, f"bis_wp{num}"):
            count += 1
        time.sleep(DELAY)

    print(f"[BIS] 완료 ({count}편)")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="modern_analysis 레이어 문헌 크롤러")
    parser.add_argument(
        "--source",
        choices=["marxists", "namuwiki", "arxiv", "bis", "all"],
        default="all",
        help="크롤링 소스 선택 (기본: all)"
    )
    args = parser.parse_args()

    print(f"modern_analysis 크롤러 시작 (소스: {args.source})")
    print(f"출력 디렉토리: {OUTPUT_DIR}\n")

    if args.source in ("marxists", "all"):
        crawl_marxists()
    if args.source in ("namuwiki", "all"):
        crawl_namuwiki()
    if args.source in ("arxiv", "all"):
        crawl_arxiv()
    if args.source in ("bis", "all"):
        crawl_bis()

    # 결과 요약
    txt_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
    print(f"\n크롤링 완료! {OUTPUT_DIR}에 총 {len(txt_files)}개 문서 저장됨")
    print("다음 명령으로 벡터 DB에 적재하세요:")
    print(f"  python update_knowledge.py --layer modern_analysis --source-dir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
