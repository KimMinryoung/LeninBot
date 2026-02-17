import requests
from bs4 import BeautifulSoup
import os
import time
import re
import argparse
from urllib.parse import urljoin, quote

# 설정
OUTPUT_DIR = "./docs/modern_analysis"
DELAY = 0.5

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

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
    if len(content) < 300:
        return False

    slug = re.sub(r'[^a-zA-Z0-9가-힣]+', '_', title[:80]).strip('_')
    if not slug:
        slug = re.sub(r'[^a-zA-Z0-9]+', '_', url.split('/')[-1])[:60].strip('_')
    file_name = f"{prefix}_{slug}.txt"
    save_path = os.path.join(OUTPUT_DIR, file_name)

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
# 1. uprising.kr
# ─────────────────────────────────────────────

UPRISING_TAGS = [
    "domestic",
    "international",
    "gugjejeongse",
    "culture",
    "translation",
    "us",
    "theory",
]


def crawl_uprising_tag(tag):
    """태그 페이지에서 글 URL 수집 후 각 글 크롤링"""
    page = 1
    count = 0
    while True:
        list_url = f"https://uprising.kr/tag/{tag}/page/{page}/" if page > 1 else f"https://uprising.kr/tag/{tag}/"
        print(f"  목록: {list_url}")
        soup = get_soup(list_url)
        if not soup:
            break

        # 글 링크 수집 - h2 > a 태그에서 상대 경로 추출
        article_links = set()
        skip_prefixes = ('/tag/', '/page/', '/author/', '/categories', '/about')
        for h2 in soup.find_all('h2'):
            a = h2.find('a', href=True)
            if not a:
                continue
            href = a['href']
            # 상대 경로 슬러그 (e.g., /some-article-slug/)
            if href.startswith('/') and not any(href.startswith(p) for p in skip_prefixes):
                full_url = f"https://uprising.kr{href}"
                article_links.add(full_url)

        if not article_links:
            break

        for article_url in article_links:
            if crawl_uprising_article(article_url):
                count += 1
            time.sleep(DELAY)

        # 다음 페이지 존재 여부 확인
        next_link = soup.find('a', class_='next') or soup.find('a', string=re.compile(r'Next|다음|›|»'))
        if not next_link:
            break
        page += 1

    return count


def crawl_uprising_article(url):
    """개별 글 본문 크롤링"""
    soup = get_soup(url)
    if not soup:
        return False

    title_tag = soup.find('h1', class_='entry-title') or soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else ""

    # 본문 추출
    content_div = soup.find('div', class_='entry-content') or soup.find('article')
    if not content_div:
        return False

    for tag in content_div.find_all(['script', 'style', 'nav', 'footer']):
        tag.decompose()

    paragraphs = content_div.find_all(['p', 'h2', 'h3', 'h4', 'blockquote', 'li'])
    content = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text()) > 20])

    return save_document(url, title, content, "uprising")


def crawl_uprising():
    print("\n[uprising.kr] 크롤링 시작")
    total = 0
    for tag in UPRISING_TAGS:
        print(f"\n  태그: {tag}")
        total += crawl_uprising_tag(tag)
    print(f"[uprising.kr] 완료 ({total}개 문서)")


# ─────────────────────────────────────────────
# 2. bolky.jinbo.net (XpressEngine 게시판)
# ─────────────────────────────────────────────

BOLKY_BOARDS = {
    "board_OpxD90": "볼셰비키 강령",
    "board_FKwQ53": "주제별 분류",
    "board_kbol88": "국제 소식",
    "board_qWpn39": "트로츠키 저작",
    "board_oynw78": "자료실",
}

BOLKY_BASE = "https://bolky.jinbo.net/index.php"


def crawl_bolky_board(board_id, board_name):
    """게시판의 글 목록 → 개별 글 크롤링"""
    page = 1
    count = 0
    seen_srls = set()

    while True:
        list_url = f"{BOLKY_BASE}?mid={board_id}&page={page}"
        print(f"  목록: {list_url}")
        soup = get_soup(list_url)
        if not soup:
            break

        # document_srl 링크 수집
        article_srls = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            match = re.search(r'document_srl=(\d+)', href)
            if match and board_id in href:
                srl = match.group(1)
                if srl not in seen_srls:
                    seen_srls.add(srl)
                    article_srls.append(srl)

        if not article_srls:
            break

        for srl in article_srls:
            article_url = f"{BOLKY_BASE}?mid={board_id}&document_srl={srl}"
            if crawl_bolky_article(article_url, board_id):
                count += 1
            time.sleep(DELAY)

        # 다음 페이지
        next_link = soup.find('a', href=re.compile(rf'mid={board_id}&page={page + 1}'))
        if not next_link:
            break
        page += 1

    return count


def crawl_bolky_article(url, board_id):
    """개별 글 본문 크롤링"""
    soup = get_soup(url)
    if not soup:
        return False

    # 제목 추출 - og:title이 가장 안정적 (XE CMS)
    og_title = soup.find('meta', property='og:title')
    if og_title and og_title.get('content'):
        title = og_title['content'].strip()
        # "게시판명 - 글제목" 패턴에서 글제목만 추출
        if ' - ' in title:
            title = title.split(' - ', 1)[1]
    else:
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ""

    # 본문 추출 - XE CMS의 본문 영역
    content_div = (
        soup.find('div', class_='read_body')
        or soup.find('div', class_='document_content')
        or soup.find('div', class_='xe_content')
    )
    if not content_div:
        return False

    for tag in content_div.find_all(['script', 'style']):
        tag.decompose()

    paragraphs = content_div.find_all(['p', 'div', 'h2', 'h3', 'h4', 'blockquote', 'li', 'span'])
    content_parts = []
    for p in paragraphs:
        text = p.get_text(strip=True)
        if len(text) > 20:
            content_parts.append(text)

    # fallback: 전체 텍스트
    if not content_parts:
        text = content_div.get_text(separator='\n', strip=True)
        content_parts = [line for line in text.split('\n') if len(line.strip()) > 20]

    content = "\n\n".join(content_parts)
    return save_document(url, title, content, "bolky")


def crawl_bolky():
    print("\n[bolky.jinbo.net] 크롤링 시작")
    total = 0
    for board_id, board_name in BOLKY_BOARDS.items():
        print(f"\n  게시판: {board_name} ({board_id})")
        total += crawl_bolky_board(board_id, board_name)
    print(f"[bolky.jinbo.net] 완료 ({total}개 문서)")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="한국 ML 조직 웹사이트 크롤러")
    parser.add_argument(
        "--source",
        choices=["uprising", "bolky", "all"],
        default="all",
        help="크롤링 소스 선택 (기본: all)"
    )
    args = parser.parse_args()

    print(f"한국 ML 조직 크롤러 시작 (소스: {args.source})")
    print(f"출력 디렉토리: {OUTPUT_DIR}\n")

    if args.source in ("uprising", "all"):
        crawl_uprising()
    if args.source in ("bolky", "all"):
        crawl_bolky()

    txt_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
    print(f"\n크롤링 완료! {OUTPUT_DIR}에 총 {len(txt_files)}개 문서 저장됨")
    print("다음 명령으로 벡터 DB에 적재하세요:")
    print(f"  python update_knowledge.py --layer modern_analysis --source-dir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
