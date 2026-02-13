import os
import glob
from dotenv import load_dotenv
from tqdm import tqdm
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. 초기화
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
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

    for file_path in tqdm(new_files, desc="전체 진행률"):
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        
        try:
            # 3. 단일 파일 로더 실행
            if ext == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                loader = LOADER_MAPPING[ext](file_path)
            
            docs = loader.load()
        
            # 4. 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)

            # 4.5. 메타데이터에 layer 및 source 주입
            source_name = os.path.splitext(file_name)[0]
            for doc in splits:
                doc.metadata["layer"] = layer
                doc.metadata["source"] = source_name

            # 5. Supabase 전송 (배치 처리)
            # tqdm을 중첩해서 쓰지 않고 파일 단위로만 표시하거나, 내부 전송도 표시할 수 있습니다.
            for i in range(0, len(splits), 100):
                vectorstore.add_documents(documents=splits[i:i+100])
            
            # 6. 로그에 추가 (성공 시에만)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(os.path.normpath(file_path) + "\n")
                
        except Exception as e:
            print(f"❌ 에러 발생 ({file_name}): {e}")

    print(f"\n✨ 지식 업데이트 완료! 총 {len(new_files)}개의 문서를 추가했습니다.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="core_theory",
                        help="Metadata layer tag (e.g. core_theory, modern_analysis)")
    parser.add_argument("--source-dir",
                        help="Override source directory (default: ./docs/lenin)")
    args = parser.parse_args()
    if args.source_dir:
        source_directory = args.source_dir
    update_knowledge(layer=args.layer)