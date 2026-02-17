import requests
from bs4 import BeautifulSoup
import os
import time
import re
import argparse
from urllib.parse import urljoin

# 설정
OUTPUT_DIR = "./docs/theorists"
DELAY = 0.3
MAX_DEPTH = 3

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

visited_urls = set()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

THEORIST_SOURCES = {
    "trotsky": {
        "start_url": "https://www.marxists.org/archive/trotsky/works/index.htm",
        "url_filter": "marxists.org/archive/trotsky/works/",
        "prefix": "trotsky",
        "label": "Trotsky",
    },
    "luxemburg": {
        "start_url": "https://www.marxists.org/archive/luxemburg/index.htm",
        "url_filter": "marxists.org/archive/luxemburg/",
        "prefix": "luxemburg",
        "label": "Rosa Luxemburg",
    },
    "gramsci": {
        "start_url": "https://www.marxists.org/archive/gramsci/index.htm",
        "url_filter": "marxists.org/archive/gramsci/",
        "prefix": "gramsci",
        "label": "Gramsci",
    },
    "bukharin": {
        "start_url": "https://www.marxists.org/archive/bukharin/index.htm",
        "url_filter": "marxists.org/archive/bukharin/",
        "prefix": "bukharin",
        "label": "Bukharin",
    },
    "mao": {
        "start_url": "https://www.marxists.org/reference/archive/mao/selected-works/index.htm",
        "url_filter": "marxists.org/reference/archive/mao/",
        "prefix": "mao",
        "label": "Mao",
    },
}


def get_soup(url):
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        res.encoding = res.apparent_encoding
        return BeautifulSoup(res.text, 'html.parser')
    except Exception as e:
        print(f"  접속 실패: {url} ({e})")
        return None


def is_content_page(url, soup):
    text_content = soup.get_text()
    return len(text_content) > 1500 and not url.endswith('index.htm')


def save_document(url, title, content, prefix):
    if len(content) < 500:
        return False

    slug = re.sub(r'[^a-zA-Z0-9]+', '_', title[:80]).strip('_').lower()
    if not slug:
        slug = re.sub(r'[^a-zA-Z0-9]+', '_', url.split('/')[-1])[:60].strip('_').lower()
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


def crawl_recursive(url, url_filter, prefix, depth):
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
                crawl_recursive(full_url, url_filter, prefix, depth + 1)


def crawl_theorist(key):
    src = THEORIST_SOURCES[key]
    print(f"\n[{src['label']}] 크롤링 시작: {src['start_url']}")
    crawl_recursive(src['start_url'], src['url_filter'], src['prefix'], 0)
    print(f"[{src['label']}] 완료")


def main():
    parser = argparse.ArgumentParser(description="marxists.org 이론가 저작 크롤러")
    parser.add_argument(
        "--source",
        choices=list(THEORIST_SOURCES.keys()) + ["all"],
        default="all",
        help="크롤링 대상 선택 (기본: all)"
    )
    args = parser.parse_args()

    print(f"이론가 크롤러 시작 (대상: {args.source})")
    print(f"출력 디렉토리: {OUTPUT_DIR}\n")

    if args.source == "all":
        for key in THEORIST_SOURCES:
            crawl_theorist(key)
    else:
        crawl_theorist(args.source)

    txt_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
    print(f"\n크롤링 완료! {OUTPUT_DIR}에 총 {len(txt_files)}개 문서 저장됨")
    print("다음 명령으로 벡터 DB에 적재하세요:")
    print(f"  python update_knowledge.py --layer core_theory --source-dir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
