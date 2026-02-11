import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from tqdm import tqdm

# 1. 설정 및 로더 매핑
persist_directory = "./db_storage"
source_directory = "./docs"
log_file = "processed_files.txt"
model_name = "jhgan/ko-sroberta-multitask"

# 확장자별 문서 로더 매핑
loaders = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
}

# 2. 임베딩 모델 준비
embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# 3. 신규 파일 체크 및 인덱싱
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8") as f:
        processed_files = set(f.read().splitlines())
else:
    processed_files = set()

all_files = {f for f in os.listdir(source_directory) if os.path.splitext(f)[1].lower() in loaders}
new_files = all_files - processed_files

if new_files:
    for file_name in tqdm(new_files, desc="문서 분석 중"):
        file_path = os.path.join(source_directory, file_name)
        ext = os.path.splitext(file_name)[1].lower()
        
        # 해당 확장자에 맞는 로더로 실행
        loader = loaders[ext](file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        # 100개씩 끊어서 DB에 저장
        for i in range(0, len(splits), 100):
            vectorstore.add_documents(documents=splits[i:i+100])
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(file_name + "\n")

# 4. RAG 체인 구성 (출처 표시를 위해 return_source_documents=True 설정)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 관련 조각 3개 추출

# 5. 질의응답 루프
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("시스템 환경변수에 'OPENAI_API_KEY'가 설정되어 있지 않다!")

# [중요] 1. 대화 기록을 기반으로 질문을 재구성하는 프롬프트
# 예: "그거 요약해줘" -> "레닌의 '어떻게 할 것인가' 책 내용을 요약해줘"로 변환
contextualize_q_system_prompt = """
이전 대화 내용과 최신 사용자 질문이 주어졌을 때, 
이 질문이 이전 대화 맥락과 관련이 있다면, 
대화 내용을 참고하여 독립적인 질문으로 다시 작성하세요. 
질문에 답변하지 말고, 검색에 최적화된 질문으로 바꾸기만 하세요.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 2. 질문 재구성용 LLM 체인 생성
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# [중요] 3. 실제 답변을 생성하는 프롬프트 (페르소나 주입)
qa_system_prompt = """
당신은 레닌의 사상과 저작을 깊이 연구한 전문가입니다.
아래의 [Context]를 기반으로 질문에 답하세요.

만약 [Context]에 명확한 답이 없다면, "제공된 문서에는 해당 내용이 없습니다"라고 말하지 말고,
문서의 핵심 사상(전위당, 직업 혁명가, 규율 등)을 바탕으로 논리적으로 추론하여 답변하세요.
단, 추론한 내용은 "문서 내용을 바탕으로 추론하자면"이라고 밝히세요.

[Context]:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 4. 최종 체인 조립 (문서 검색 + 답변 생성)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 5. 대화 루프 (메모리 기능 추가)
chat_history = []  # 대화 기록 저장소

print("\n" + "="*50)
print("🤖 지능형 레닌 봇 준비 완료! (종료: 'exit')")
print("="*50)

while True:
    query = input("\n질문: ")
    if query.lower() in ['exit', 'quit', 'q']: break
    if not query.strip(): continue
    
    print("🤔 생각 중...")
    
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    # 답변 출력
    answer = result['answer']
    print(f"\n[AI]: {answer}")
    
    # 출처 명시
    print("\n[참고한 문서 조각]")
    for i, doc in enumerate(result['context']):
        print(f"- {doc.page_content[:50]}...")

    # 대화 기록 업데이트
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))