import os
import glob
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

            # 4.5. 메타데이터에 layer 및 source 주입
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
            for doc in splits:
                doc.metadata["layer"] = layer
                doc.metadata["source"] = source_name

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