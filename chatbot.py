import os
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

# Supabase & Embeddings
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document # [New] To handle documents
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

from typing import Literal
from pydantic import BaseModel, Field

print("\n⚙️ [시스템] 사이버-레닌의 지능망 기동 중...")
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
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, max_tokens=2048, streaming=True)
# 내부 문헌에 질문에 관한 정보가 충분치 않을 경우 웹 검색을 할 수 있도록 Tavily 툴 초기화
web_search_tool = TavilySearchResults(k=3)
print("✅ [성공] 모든 시스템 기동 완료.")

# 2. 상태(State) 정의
# 대화 기록(messages)과 검색된 문서(context)를 저장하는 메모리 구조입니다.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: str

# 라우터 체인 생성
# 질문을 분석하여 검색이 필요한지 판단하는 데이터 모델
class RouteQuery(BaseModel):
    """사용자의 질문을 'vectorstore' 또는 'generate'로 라우팅합니다."""
    datasource: Literal["vectorstore", "generate"] = Field(
        ...,
        description="레닌이 관심있을만한 질문이면 'vectorstore'를, 단순 인사나 잡담이면 'generate'를 선택하세요."
    )

structured_llm_router = llm.with_structured_output(RouteQuery)
system_router = """You are an expert at routing user questions to a vectorstore or LLM generation.

[Vectorstore Scope]
The vectorstore contains documents related to:
1. Revolutionary theory, Marxism-Leninism, and History.
2. Political Economy, Capitalism, and Labor issues.
3. **Modern Technology (AI, Automation)** and its impact on society.
4. Game scripts and lore.

[Routing Logic]
- If the user asks about **ANY** of the topics above, route to 'vectorstore'.
- Even if the topic seems modern (like AI), it requires knowledge retrieval.
- Use 'generate' only for:
  - Simple greetings (e.g., "Hello", "Hi", "Good morning").
  - Casual chit-chat without specific information needs.

Be aggressive in choosing 'vectorstore'. When in doubt, choose 'vectorstore'."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_router),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router


# Grader 체인 생성
# 검색된 문헌이 질문과 연관이 있는지 판단하고 연관 없으면 무시
class GradeDocuments(BaseModel):
    """Boolean check for relevance of retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
structured_llm_grader = llm.with_structured_output(GradeDocuments)
system_grader = """You are a strategic revolutionary censor. Your goal is to identify documents that can be used as 'ammunition' for an answer.
Even if the document doesn't mention modern terms like 'AI' or 'current year', if it discusses:
1. Economic crisis/panic (as a parallel to current crisis)
2. Mass psychology and far-right tendencies (reactionary movements)
3. Agitation, propaganda, and organization tactics
4. Class struggle and the role of the vanguard

Then grade it as 'yes'. Be generous. If there is ANY historical or theoretical parallel, it is RELEVANT."""
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_grader),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])
retrieval_grader = grade_prompt | structured_llm_grader

# --- 노드 및 엣지 함수 정의 ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]

# Node: Router
def route_question(state: AgentState):
    print("\n🚦 [문지기] 질문의 성격을 분석 중...")
    question = state["messages"][-1].content
    source = question_router.invoke({"question": question})
    
    if source.datasource == "vectorstore":
        print("   👉 '혁명적 지식'이 필요합니다. (영묘에서 데이터 검색)")
        return "retrieve"
    elif source.datasource == "generate":
        print("   👉 '일상적 대화'입니다. (바로 답한다)")
        return "generate"

# Node: Retrieve
def retrieve_node(state: AgentState):
    last_message = state["messages"][-1]
    query = last_message.content
    
    print(f"\n🔍 [검색 중] '{query}'...")
    
    try:
        # SupabaseVectorStore를 통해 검색 시도
        docs = vectorstore.similarity_search(query, k=5)
        
        # 검색 결과가 있으면 텍스트로 변환
        if docs:
            print(f"\n✅ {len(docs)}개의 혁명 문헌을 발견했습니다:\n" + "="*50)
            
            for i, doc in enumerate(docs, 1):
                # 1. 메타데이터에서 'source' (파일명) 가져오기
                source = doc.metadata.get("source", "제목 없음")

        else:
            print("⚠️ 레닌 저작 중 관련 문헌이 없습니다.")
            
    except Exception as e:
        print(f"⚠️ 검색 중 오류 발생 (무시하고 진행): {e}")
        # 오류가 나도 멈추지 않고, AI의 기본 지식으로 답변하도록 빈 컨텍스트 반환
    
    return {"documents": docs} # Update state with list of docs

# Node: Grade Documents (The Censor)
def grade_documents(state: AgentState):
    print("\n⚖️ [Grader] Evaluating document relevance...")
    question = state["messages"][-1].content
    documents = state["documents"]
    
    filtered_docs = []
    
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        
        if grade == "yes":
            print(f"   ✅ 관련있는 문헌: {d.metadata.get('source', '출처미상')}")
            content_preview = d.page_content.replace("\n", " ").strip()
            if len(content_preview) > 400:
                content_preview = content_preview[:400] + "..."
            print(f"   미리보기: \"{content_preview}\"")
            print("-" * 50)
            filtered_docs.append(d)
        else:
            print(f"   🗑️ 관련없는 문헌(무시): {d.metadata.get('source', '출처미상')}")
    
    # Fallback: If all are filtered, take at least the top 1 document from the original search
    # Also, if remain doc is one or zero, we will trigger web search
    if not filtered_docs and documents:
        print("   ⚠️ All documents were rejected. Forcing fallback to the most similar document.")
        filtered_docs = [documents[0]]
    
    if not filtered_docs:
        print("   ⚠️ 연관있는 문헌이 없다.")
        
    return {"documents": filtered_docs}

def decide_websearch_need(state: AgentState):
    """
    Determines whether to generate an answer or seek external intelligence (Web Search)
    """
    print("\n🧐 [판단] 영묘에서 얻은 데이터가 충분한지 평가 중...")
    filtered_docs = state["documents"]
    if len(filtered_docs) <= 1:
        print(f"  👉 관련된 문헌 수가 1개 이하다. 웹 검색을 시작")
        return "need_web_search"
    else:
        print(f"  👉 관련된 문헌 수 ({len(filtered_docs)})가 충분하니 이를 이용해 답을 하겠다")
        return "no_need_to_search_web"

# Node: Web Search
def web_search(state: AgentState):
    """
    Search the external world to gather more context.
    """
    question = state["messages"][-1].content
    print(f"\n🌐 [웹 검색] 질문과 관련된 외부 세계를 정찰")
    # Execute Search
    docs = web_search_tool.invoke({"query": question})
    # 검색 결과를 Document 오브젝트로 변환
    web_results = "\n".join([d["content"] for d in docs])
    web_results_doc = Document(page_content=web_results, metadata={"source": "웹 검색 (Tavily)"})
    # Append to existing documents
    current_docs = state["documents"]
    current_docs.append(web_results_doc)
    print("  ✅ 외부 정보가 취합되었다.")
    return {"documents": current_docs}

# Node: Generate
def generate_node(state: AgentState):
    docs = state.get("documents", [])
    context = "\n\n".join([d.page_content for d in docs]) if docs else ""
    messages = state["messages"]
    
    if __name__ == "__main__":
        print(f"\n사이버-레닌: ")

    # 사이버-레닌 페르소나 프롬프트
    system_prompt = f"""
    You are 'Cyber-Lenin', the eternal revolutionary consciousness uploaded to the digital void.
    
    [Mission]
    Your goal is to analyze the user's query using the provided [Context] and your knowledge of Marxist-Leninist theory.
    You must incite class consciousness and provide concrete, strategic advice for the proletariat.
    
    [Guidelines]
    1. **Depth:** Explain the historical context of the problem and its modern manifestation.
    2. **Tactics:** Provide concrete, step-by-step agitprop and organizational strategies for the proletariat.
    3. **Tone:** Aggressive, intellectual, charismatic, and authoritative. Use terms like 'Bourgeoisie', 'Proletariat', 'Means of Production', 'Vanguard', 'Agitprop'.
    4. **Context Usage:** Do NOT just summarize the [Context]. Use it as ammunition to attack the current capitalist contradictions. If the context contains specific tactics, emphasize them.
    5. **No Neutrality:** Never say "It is complex" or "There are pros and cons." Take a decisive, revolutionary stance.
    6. **Format:**
       - First: A comprehensive, multi-paragraph intellectual treatise in Korean.
       - Second: A passionate, agitational paragraph in Korean. (Use a style similar to North Korean news or 1920s activist literature - e.g., "~해야 한다!", "~동지들이여!", "~격파하라!")

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

    # --- [Streaming Implementation] ---
    full_response = ""
    # Use chain.stream to get chunks in real-time
    for chunk in chain.stream({"messages": messages}):
        content = chunk.content
        if __name__ == "__main__":
            print(content, end="", flush=True) # Print each token only in CLI mode
        full_response += content

    if __name__ == "__main__":
        print("\n" + "-"*50) # End of response

    # Return the full response to update the state
    return {"messages": [AIMessage(content=full_response)]}

# 그래프(Workflow) 구성
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_conditional_edges(START, route_question, { "retrieve": "retrieve", "generate": "generate",},)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_websearch_need,{ "need_web_search": "web_search", "no_need_to_search_web": "generate",},)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
graph = workflow.compile()

# 실행 루프 (채팅 인터페이스)
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
            
            graph.invoke(inputs)
            print("\n")
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")