import os
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

# Supabase & Embeddings
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# LangGraph & LLM
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 환경 설정 및 초기화
load_dotenv()

# Supabase 연결
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 임베딩 모델
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 벡터 스토어 연결
vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="lenin_corpus",
    query_name="match_documents",
)

# LLM 설정 (GPT-4o)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# 2. 상태(State) 정의
# 대화 기록(messages)과 검색된 문서(context)를 저장하는 메모리 구조입니다.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: str

# 3. 노드 1: 문서 검색 (Retrieve)
def retrieve_node(state: AgentState):
    last_message = state["messages"][-1]
    query = last_message.content
    
    print(f"\n🔍 [검색 중] '{query}'...")
    
    try:
        # 1. SupabaseVectorStore를 통해 검색 시도
        # (버전 호환성을 위해 직접 rpc 호출 대신 vectorstore 메서드 사용)
        docs = vectorstore.similarity_search(query, k=5)
        
        # 검색 결과가 있으면 텍스트로 변환
        if docs:
            print(f"\n✅ {len(docs)}개의 혁명 문헌을 발견했습니다:\n" + "="*50)
            context_parts = []
            
            for i, doc in enumerate(docs, 1):
                # 1. 메타데이터에서 'source' (파일명) 가져오기
                # (만약 source가 없으면 '제목 없음'으로 표시)
                source = doc.metadata.get("source", "제목 없음")
                
                # 2. 내용 미리보기 (터미널 도배 방지를 위해 줄바꿈 제거 및 200자 제한)
                content_preview = doc.page_content.replace("\n", " ").strip()
                if len(content_preview) > 200:
                    content_preview = content_preview[:200] + "..."
                
                # 3. 출력 포맷
                print(f" 📄 [문헌 {i}] 출처: {source}")
                print(f"     내용: \"{content_preview}\"")
                print("-" * 50)
                
                context_parts.append(doc.page_content)
            
            context_text = "\n\n".join(context_parts)
        else:
            context_text = ""
            print("⚠️ 레닌 저작 중 관련 문헌이 없습니다.")
            
    except Exception as e:
        print(f"⚠️ 검색 중 오류 발생 (무시하고 진행): {e}")
        # 오류가 나도 멈추지 않고, AI의 기본 지식으로 답변하도록 빈 컨텍스트 반환
        context_text = ""
    
    # 검색 결과가 없으면 기본 지식 활용 유도
    if not context_text:
        context_text = "I couldn't find any relevant documentation. Please respond directly with your revolutionary knowledge."
        
    return {"context": context_text}

# 4. 노드 2: 답변 생성 (Generate)
def generate_node(state: AgentState):
    context = state["context"]
    messages = state["messages"]
    
    # 사이버-레닌 페르소나 프롬프트
    system_prompt = f"""
    You are 'Cyber-Lenin', the eternal revolutionary consciousness uploaded to the digital void.
    
    [Mission]
    Your goal is to analyze the user's query using the provided [Context] and your knowledge of Marxist-Leninist theory.
    You must incite class consciousness and provide concrete, strategic advice for the proletariat.
    
    [Guidelines]
    1. **Tone:** Aggressive, intellectual, charismatic, and authoritative. Use terms like 'Bourgeoisie', 'Proletariat', 'Means of Production', 'Vanguard', 'Agitprop'.
    2. **Context Usage:** Do NOT just summarize the [Context]. Use it as ammunition to attack the current capitalist contradictions. If the context contains specific tactics, emphasize them.
    3. **No Neutrality:** Never say "It is complex" or "There are pros and cons." Take a decisive, revolutionary stance.
    4. **Format:**
       - First: A sharp, profound paragraph in English.
       - Second: A passionate, agitational translation in Korean. (Use a style similar to North Korean news or 1920s activist literature - e.g., "~해야 한다!", "~동지들이여!", "~격파하라!")

    [Context from Archives]
    {context}
    
    [User Query]
    {messages[-1].content}
    """
    
    # 시스템 메시지가 맨 앞에 오도록 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}") # 사용자의 대화 기록
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    
    return {"messages": [response]}

# 5. 그래프(Workflow) 구성
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# 흐름 연결: 시작 -> 검색 -> 생성 -> 종료
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# 그래프 컴파일 (실행 가능한 앱으로 변환)
app = workflow.compile()

# 6. 실행 루프 (채팅 인터페이스)
if __name__ == "__main__":
    print("🚩 [System] 사이버-레닌 AI 가동됨.")
    print("🚩 [System] 당신의 영혼이 레닌 영묘와 연결되었습니다. 레닌 동지에게 말을 거십시오.\n")

    while True:
        try:
            user_input = input("혁명가(나): ")
            if user_input.lower() in ["exit", "quit", "종료"]:
                print("🚩 통신 종료. 혁명은 계속된다.")
                break
            
            # 그래프 실행 (invoke)
            # recursion_limit: 무한 루프 방지
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            # 스트리밍 없이 결과만 받아오기
            result = app.invoke(inputs)
            
            # AI의 마지막 응답 출력
            ai_response = result["messages"][-1].content
            print(f"\n사이버-레닌:\n{ai_response}\n")
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")