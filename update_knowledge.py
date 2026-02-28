import os
import glob
import re
from dotenv import load_dotenv
from tqdm import tqdm
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch

load_dotenv()

# 1. 초기화
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# BGE-M3 임베딩 모델 (1024차원, 다국어 지원)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[System] Using device: {device}")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="lenin_corpus",
    query_name="match_documents",
)

source_directory = "./docs/lenin"
log_file = "processed_files.txt"

# 2. 파일 확장자별 로더 매핑 (클래스 자체를 매핑)
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader
}

# 저자-디렉토리 매핑 (source_directory 경로로 판별)
AUTHOR_BY_DIR = {
    "lenin": "Lenin",
    "marx_engels": "Marx & Engels",
}

# 이론가 prefix → 저자 이름 매핑
THEORIST_AUTHORS = {
    "trotsky": "Trotsky",
    "luxemburg": "Rosa Luxemburg",
    "gramsci": "Gramsci",
    "bukharin": "Bukharin",
    "mao": "Mao",
}

# modern_analysis prefix → 저자/출처 매핑
MODERN_AUTHORS = {
    "arxiv": None,       # 파일 내 Authors: 헤더에서 추출
    "bis": "BIS",
    "mxo_mandel": "Ernest Mandel",   # 더 구체적인 prefix를 먼저 배치
    "mxo_marcuse": "Herbert Marcuse",
    "mxo": "Marxists.org",
    "uprising": "Uprising(반란) - Korean Marx-Lenin-Maoism group",
    "bolky": "Bolky Group (볼셰비키그룹) - Korean Trotskyist organization",
}


def _extract_author_year(file_path, source_dir):
    """파일 경로와 헤더에서 author, year 메타데이터를 추출한다."""
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    author = None
    year = None

    # --- 디렉토리 기반 저자 판별 ---
    dir_name = os.path.basename(os.path.normpath(source_dir))
    if dir_name in AUTHOR_BY_DIR:
        author = AUTHOR_BY_DIR[dir_name]

    # --- 이론가 prefix 기반 저자 판별 ---
    if dir_name == "theorists":
        for prefix, name in THEORIST_AUTHORS.items():
            if base_name.startswith(prefix + "_"):
                author = name
                break

    # --- modern_analysis prefix 기반 저자 판별 ---
    if dir_name == "modern_analysis":
        for prefix, name in MODERN_AUTHORS.items():
            if base_name.startswith(prefix + "_"):
                author = name  # arxiv은 None → 아래 헤더 파싱에서 덮어씀
                break

    # --- 파일 헤더 파싱 (Source/Title/Author(s)/Year 줄) ---
    source_url = ""
    title_line = ""
    year_from_header = None
    try:
        with open(file_path, 'r', encoding='utf-8') as tf:
            for line in tf:
                line_s = line.strip()
                if line_s.startswith("Source:"):
                    source_url = line_s[7:].strip()
                elif line_s.startswith("Title:"):
                    title_line = line_s[6:].strip()
                elif line_s.startswith("Authors:") and not author:
                    author = line_s[8:].strip()
                elif line_s.startswith("Author:") and not author:
                    author = line_s[7:].strip()
                elif line_s.startswith("Year:") and not year_from_header:
                    year_from_header = line_s[5:].strip()
                elif not line_s.startswith(("Source:", "Title:", "Authors:", "Author:", "Year:", "")):
                    break  # 헤더 영역을 벗어나면 중단
    except Exception:
        pass

    # --- 연도 추출: 헤더의 Year: 필드 (가장 우선) ---
    if year_from_header:
        year = year_from_header

    # --- 연도 추출: 파일명 앞 4자리 ---
    if not year:
        year_match = re.match(r'^(\d{4})_', base_name)
        if year_match:
            year = year_match.group(1)

    # --- 연도 추출 fallback: Source URL에서 /YYYY/ 패턴 ---
    if not year and source_url:
        url_year = re.search(r'/(\d{4})/', source_url)
        if url_year:
            year = url_year.group(1)

    # --- 연도 추출 fallback: arXiv URL에서 YYMM 패턴 (e.g. arxiv.org/abs/2304.07859) ---
    if not year and source_url and 'arxiv.org/abs/' in source_url:
        arxiv_match = re.search(r'arxiv\.org/abs/(\d{2})\d{2}\.', source_url)
        if arxiv_match:
            year = "20" + arxiv_match.group(1)

    # --- 연도 추출 fallback: Title에서 연도 패턴 ---
    if not year and title_line:
        title_year = re.search(r'\b(1[7-9]\d{2}|20[0-2]\d)\b', title_line)
        if title_year:
            year = title_year.group(1)

    return author, year


def update_knowledge(layer="core_theory"):
    print(f"📂 {source_directory} 폴더에서 새 문서를 탐색 중... (layer: {layer})")
    
    # 로그 파일에서 처리된 파일 목록 읽기
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()

    # 하위 폴더를 포함하여 모든 .txt, .pdf 파일 찾기
    all_files = []
    for ext in LOADER_MAPPING.keys():
        all_files.extend(glob.glob(os.path.join(source_directory, f"**/*{ext}"), recursive=True))

    # 경로를 통일하여 비교 (역슬래시 문제를 방지하기 위해 os.path.normpath 사용)
    new_files = [f for f in all_files if os.path.normpath(f) not in processed_files]

    if not new_files:
        print("✅ 추가할 새 문서가 없습니다.")
        return

    total_chunks = 0
    file_bar = tqdm(new_files, desc="파일 처리", unit="file", position=0)
    for file_path in file_bar:
        file_name = os.path.basename(file_path)
        file_bar.set_postfix_str(file_name[:40])
        ext = os.path.splitext(file_name)[1].lower()

        try:
            # 3. 단일 파일 로더 실행
            if ext == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                loader = LOADER_MAPPING[ext](file_path)

            docs = loader.load()

            # 4. 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # 4.5. 메타데이터에 layer, source, author, year 주입
            # 파일 내 Title: 헤더가 있으면 문헌 제목을 source로 사용
            source_name = os.path.splitext(file_name)[0]
            try:
                with open(file_path, 'r', encoding='utf-8') as tf:
                    for line in tf:
                        line = line.strip()
                        if line.startswith("Title:") and line[6:].strip():
                            source_name = line[6:].strip()
                            break
                        if not line.startswith("Source:") and line:
                            break  # 헤더 영역을 벗어나면 중단
            except Exception:
                pass

            author, year = _extract_author_year(file_path, source_directory)

            for doc in splits:
                doc.metadata["layer"] = layer
                doc.metadata["source"] = source_name
                if author:
                    doc.metadata["author"] = author
                if year:
                    doc.metadata["year"] = year

            # 5. Supabase 전송 (배치 처리)
            batch_size = 5
            for i in range(0, len(splits), batch_size):
                vectorstore.add_documents(documents=splits[i:i+batch_size])
            total_chunks += len(splits)

            # 6. 로그에 추가 (성공 시에만)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(os.path.normpath(file_path) + "\n")

        except Exception as e:
            tqdm.write(f"❌ 에러 발생 ({file_name}): {e}")
    file_bar.close()

    print(f"\n✨ 지식 업데이트 완료! 총 {len(new_files)}개 문서, {total_chunks}개 청크를 추가했습니다.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="core_theory",
                        help="Metadata layer tag (e.g. core_theory, modern_analysis)")
    parser.add_argument("--source-dir",
                        help="Override source directory (default: ./docs/lenin)")
    args = parser.parse_args()
    if args.source_dir:
        source_directory = args.source_dir  # noqa: F841 - used by update_knowledge
    update_knowledge(layer=args.layer)