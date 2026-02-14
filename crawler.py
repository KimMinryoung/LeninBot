import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin

# 1. 설정
# 가장 방대한 목록이 있는 연도별 색인 페이지를 시작점으로 잡습니다.
START_URL = "https://www.marxists.org/archive/lenin/works/index.htm"
OUTPUT_DIR = "./docs/lenin"
DELAY = 0.3 
MAX_DEPTH = 3 # 목차 -> 하위 목차 -> 챕터까지 들어갈 수 있도록 설정

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

visited_urls = set()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

def get_soup(url):
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        res.encoding = res.apparent_encoding
        return BeautifulSoup(res.text, 'html.parser')
    except Exception as e:
        print(f"❌ 접속 실패: {url} ({e})")
        return None

def is_content_page(url, soup):
    """실제 본문이 들어있는 페이지인지 판단합니다."""
    # 190x/xx/xx.htm 식의 패턴이거나 텍스트 양이 많으면 본문으로 간주
    text_content = soup.get_text()
    return len(text_content) > 1500 and not url.endswith('index.htm')

def crawl(url, depth):
    if depth > MAX_DEPTH or url in visited_urls:
        return
    
    visited_urls.add(url)
    soup = get_soup(url)
    if not soup: return

    # 1. 만약 현재 페이지가 본문이라면 저장
    if is_content_page(url, soup):
        save_document(url, soup)
        return # 본문 페이지 안의 링크는 더 이상 타지 않음

    # 2. 목차 페이지라면 하위 링크 수집 및 재귀 탐색
    print(f"📂 탐색 중 (Depth {depth}): {url}")
    links = soup.find_all('a', href=True)
    
    for a in links:
        href = a['href']
        full_url = urljoin(url, href).split('#')[0] # 앵커 제거

        # 레닌의 저작 경로(works) 내부에 있고, htm 확장자이며, 이미 방문하지 않은 경우만
        if "marxists.org/archive/lenin/works/" in full_url:
            if full_url.endswith('.htm') or full_url.endswith('.html') or full_url.endswith('/'):
                crawl(full_url, depth + 1)

def save_document(url, soup):
    # 문헌 제목 추출 (nav 제거 전에 수행)
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # 내비게이션 요소 제거
    for nav in soup.find_all(['nav', 'header', 'footer', 'table']):
        nav.decompose()

    # 본문 추출
    paragraphs = soup.find_all(['p', 'h3', 'h4', 'blockquote'])
    content = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text()) > 30])

    if len(content) < 500: return # 너무 짧은 데이터는 버림

    # 파일명 생성 (URL 구조 반영)
    path_part = url.split('works/')[-1]
    file_name = re.sub(r'[/\\?%*:|"<>]', '_', path_part).replace('.htm', '.txt').replace('.html', '.txt')

    save_path = os.path.join(OUTPUT_DIR, file_name)

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"Source: {url}\nTitle: {title}\n\n{content}")
        print(f"  └─ ✅ 저장 완료: {file_name} ({title})")
    except Exception as e:
        print(f"  └─ ❌ 저장 실패: {e}")
    
    time.sleep(DELAY)

# 실행
print("🚀 레닌 전집 전체 크롤링을 시작합니다. (상당한 시간이 소요될 수 있습니다.)")
crawl(START_URL, 0)
print(f"\n🎉 작업 완료! 총 {len(visited_urls)}개의 페이지를 검사했습니다.")