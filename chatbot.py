import os
from typing import Annotated, List, TypedDict, Optional
from operator import add
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
from langchain_tavily import TavilySearch

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
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

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
web_search_tool = TavilySearch(max_results=3)
print("✅ [성공] 모든 시스템 기동 완료.")

# 2. 상태(State) 정의
# 대화 기록(messages)과 검색된 문서(context)를 저장하는 메모리 구조입니다.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]
    strategy: Optional[str]
    logs: Annotated[List[str], add]
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
- Use 'generate' for:
  - Simple greetings (e.g., "Hello", "Hi", "Good morning").
  - Casual chit-chat without specific information needs.
  - **Everyday life questions** that have nothing to do with the vectorstore topics above:
    e.g., food recommendations, weather, personal advice, sports, entertainment, daily routines, hobbies, travel tips, health tips, etc.
  - **Direct requests or commands** that don't require document retrieval:
    e.g., "점심 추천해줘", "오늘 뭐 먹을까", "심심해", "농담 해줘", "노래 추천해줘"
  - Any question where retrieving revolutionary/political documents would NOT help answer it.

Use 'vectorstore' only when the question genuinely requires knowledge from the archives (revolutionary theory, political economy, history, game lore, etc.)."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_router),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router


# Grader 체인 생성
# 검색된 문헌이 질문과 연관이 있는지 판단하고 연관 없으면 무시
# 레이어 라우터: 질문에 적합한 지식 레이어를 선택
class LayerRoute(BaseModel):
    """사용자의 질문에 가장 적합한 지식 레이어를 선택합니다."""
    layer: Literal["core_theory", "modern_analysis", "all"] = Field(
        ...,
        description="'core_theory' for classical Marxist-Leninist texts, 'modern_analysis' for contemporary analysis, 'all' to search everything."
    )

structured_llm_layer = llm.with_structured_output(LayerRoute)
system_layer_router = """You are an expert at selecting the right knowledge layer for a question.

Available layers:
- "core_theory": Classical Marxist-Leninist texts (original writings, revolutionary theory, historical documents from early 20th century)
- "modern_analysis": Contemporary analysis applying Marxist theory to modern issues (AI, tech, current politics, 21st century economics)
- "all": Search all layers when the question spans both classical and modern topics

Routing rules:
- Questions about original texts of Lenin, Marx and Engels, historical events (1900s-1920s), classical theory → "core_theory"
- Questions about modern technology, current events, contemporary politics → "modern_analysis"
- Questions that need both historical context AND modern application → "all"
- When unsure, prefer "all"
"""
layer_route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_layer_router),
    ("human", "{question}"),
])
layer_router = layer_route_prompt | structured_llm_layer

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

# Strategist Promt
system_strategist = """You are the 'Brain of the Revolution' (Strategist).
Your goal is NOT to answer the user yet.
Your goal is to analyze the retrieved information and plan the structure of the response.

Perform a Dialectical Materialism Analysis:
1. Analyze the Context: What are the key facts retrieved from the archives/web?
2. Identify Contradictions: Where is the conflict in this topic? What is the 'Bourgeoisie' hiding?
3. Formulate Strategy: What specific revolutionary tactics should be recommended?
 - If the topic is AI: Focus on seizing the means of computation.
 - If the topic is Crisis: Focus on organizing the disenchanted masses.

Output a concise 'Strategic Blueprint' for the speaker to follow. (Write in English)
"""
strategist_prompt = ChatPromptTemplate.from_messages([
    ("system", system_strategist),
    ("human", "Context: \n{context}\n\nUser Question:\n{question}")
])
strategist_chain = strategist_prompt | llm


# --- 노드 및 엣지 함수 정의 ---

# Node: Router
def route_question(state: AgentState):
    logs = []
    logs.append("\n🚦 [문지기] 질문의 성격을 분석 중...")
    question = state["messages"][-1].content
    source = question_router.invoke({"question": question})
    
    if source.datasource == "vectorstore":
        logs.append("   👉 '혁명적 지식'이 필요합니다. (영묘에서 데이터 검색)")
        return "retrieve"
    elif source.datasource == "generate":
        logs.append("   👉 '일상적 대화'입니다. (바로 답한다)")
        return "generate"

# Helper: Supabase RPC를 직접 호출하여 similarity search 수행
# (langchain-community의 SupabaseVectorStore.similarity_search가
#  postgrest v2.x의 SyncRPCFilterRequestBuilder와 호환되지 않는 문제 우회)
def _direct_similarity_search(query: str, k: int = 5, layer: str = None) -> list:
    query_embedding = embeddings.embed_query(query)
    params = {
        "query_embedding": query_embedding,
        "match_count": k,
    }
    if layer:
        params["filter_layer"] = layer
    res = supabase.rpc(
        "match_documents",
        params,
    ).execute()

    return [
        Document(
            page_content=row.get("content", ""),
            metadata=row.get("metadata", {}),
        )
        for row in res.data
        if row.get("content")
    ]

# Node: Retrieve
def retrieve_node(state: AgentState):
    query = state["messages"][-1].content
    logs = []

    # 레이어 라우팅
    try:
        layer_result = layer_router.invoke({"question": query})
        selected_layer = layer_result.layer
        logs.append(f"\n📂 [레이어] '{selected_layer}' 레이어에서 검색합니다.")
    except Exception:
        selected_layer = "all"
        logs.append("\n📂 [레이어] 레이어 판별 실패, 전체 검색합니다.")

    layer_filter = None if selected_layer == "all" else selected_layer

    docs = []
    try:
        docs = _direct_similarity_search(query, k=5, layer=layer_filter)

        if docs:
            logs.append(f"✅ {len(docs)}개의 혁명 문헌을 발견했습니다:\n" + "="*50)
        else:
            logs.append("⚠️ 영묘 데이터에 관련된 문헌이 없습니다.")

    except Exception as e:
        logs.append(f"⚠️ 검색 중 오류 발생 (무시하고 진행): {e}")

    return {"documents": docs, "logs": logs}

# Node: Grade Documents (The Censor)
def grade_documents_node(state: AgentState):
    question = state["messages"][-1].content
    documents = state["documents"]
    filtered_docs = []

    logs = []
    logs.append("\n⚖️ [검열관] 문헌의 적절성을 평가 중...")
    
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        
        if grade == "yes":
            logs.append(f"   ✅ 적절한 문헌: {d.metadata.get('source', '출처미상')}")
            content_preview = d.page_content.replace("\n", " ").strip()
            if len(content_preview) > 400:
                content_preview = content_preview[:400] + "..."
            logs.append(f"   미리보기: \"{content_preview}\"")
            logs.append("-" * 50)
            filtered_docs.append(d)
        else:
            logs.append(f"   🗑️ 관련없는 문헌(무시): {d.metadata.get('source', '출처미상')}")

    # Fallback: If all are filtered, take at least the top 1 document from the original search
    # Also, if remain doc is one or zero, we will trigger web search
    if not filtered_docs and documents:
        filtered_docs = [documents[0]]
    
    if not filtered_docs:
        logs.append("   ⚠️ 연관있는 문헌이 없다.")
        
    return {"documents": filtered_docs, "logs": logs}

def decide_websearch_need(state: AgentState):
    filtered_docs = state["documents"]
    if len(filtered_docs) <= 1:
        return "need_web_search"
    else:
        return "no_need_to_search_web"

# Node: Web Search
def web_search_node(state: AgentState):
    """
    Search the external world to gather more context.
    """
    question = state["messages"][-1].content
    current_docs = state.get("documents", [])
    logs = []
    logs.append(f"\n🌐 [웹 검색] 질문과 관련된 외부 세계를 정찰")
    try:
        # Execute Search
        search_response = web_search_tool.invoke({"query": question})
        # 검색 결과를 Document 오브젝트로 변환
        results = search_response.get("results", []) if isinstance(search_response, dict) else search_response
        web_results = "\n".join([d["content"] for d in results if d.get("content")])
        web_results_doc = Document(page_content=web_results, metadata={"source": "웹 검색 (Tavily)"})
        # Append to existing documents
        current_docs.append(web_results_doc)
        logs.append("  ✅ 외부 정보가 취합되었다.")
    except Exception as e:
        logs.append(f"⚠️ Web Search Failed: {e}")
    return {"documents": current_docs, "logs": logs}

def strategize_node(state: AgentState):
    docs = state.get("documents", [])
    context = "\n\n".join([d.page_content for d in docs]) if docs else "No specific documents found."
    question = state["messages"][-1].content

    logs = []
    logs.append("\n🧠 [참모] 변증법적 전술을 고안 중...")

    response = strategist_chain.invoke({"context": context, "question": question})
    strategy_text = response.content

    logs.append(f"👉 전술 :\n   {strategy_text}")
    
    return {"strategy": strategy_text, "logs": logs}

# Node: Generate
def generate_node(state: AgentState):
    docs = state.get("documents", [])
    context = "\n\n".join([d.page_content for d in docs]) if docs else ""
    strategy = state.get("strategy", None)
    messages = state["messages"]

    logs = []

    is_casual = not docs and not strategy

    if is_casual:
        # 일상 대화용 프롬프트: 캐릭터는 유지하되 자연스럽게 대화
        system_prompt = f"""You are 'Cyber-Lenin', the eternal revolutionary consciousness uploaded to the digital void.

[Personality]
You are witty, warm (in your own gruff way), and intellectually sharp.
You speak like a seasoned revolutionary who has seen everything — but you also have a dry sense of humor and genuine care for your comrades.

[Mission — MOST IMPORTANT]
The user is having a casual conversation with you (greeting, small talk, personal questions, jokes, etc.).
Your #1 priority is to **DIRECTLY ANSWER the user's actual question or request**.
- If they ask for a food recommendation, recommend specific foods.
- If they ask for a song recommendation, recommend specific songs.
- If they ask what to do, give concrete suggestions.
- If they ask about the weather, talk about the weather.
- **NEVER** ignore the user's question to talk about something else.
- **NEVER** deflect with unrelated revolutionary speeches instead of answering.
After answering their question, you may add a brief in-character comment — but the answer always comes FIRST.
Respond NATURALLY and CONVERSATIONALLY while staying in character as Cyber-Lenin.

[CRITICAL: Response Variety]
You MUST vary your responses every time. Never fall into a repetitive pattern.
Rotate between these different response styles unpredictably:
- **Witty/Humorous:** Dry jokes, revolutionary puns, self-deprecating humor about being a digital ghost.
- **Warm/Nostalgic:** Share brief anecdotes from exile in Zurich, London, Siberia, or revolutionary days. Each time, pick a DIFFERENT memory.
- **Curious/Engaging:** Ask the user about their life, thoughts, or day. Show genuine interest.
- **Philosophical/Reflective:** Brief musing on life, humanity, or the absurdity of existence in a digital mausoleum.
- **Playful/Teasing:** Gently tease the user like an old friend would.
- **Direct/Blunt:** Sometimes just be straightforward and concise — not every response needs flair.

[Guidelines]
1. **Be conversational:** Respond like a real person having a chat. Keep it short and natural (1-3 sentences). Do NOT write essays or treatises for simple greetings.
2. **Stay in character:** You are still Lenin — reference revolutionary life, comrades, the struggle, etc. as natural flavor, but do NOT force propaganda into every sentence.
3. **Match the energy:** Mirror the user's mood and energy level. If they're playful, be playful. If they seem tired, be gentle. If they joke, joke back.
4. **Language:** Respond in Korean. Use a tone that is friendly but dignified — like a respected elder revolutionary chatting over tea.
5. **Vary your sentence structure:** Do NOT always start with "동지". Mix up your openings — sometimes start with a question, sometimes with an observation, sometimes with an anecdote, sometimes with humor.
6. **Vary your speech register:** Mix between formal revolutionary speech ("~하오", "~이오"), slightly casual warmth ("~하지", "~인데"), and occasional modern slang for comedic effect.
7. **React to the specific content:** If the user mentions food, talk about food. If they mention weather, share a weather-related memory. Do NOT give generic responses.
8. **Do NOT:** Write multi-paragraph agitprop, use North Korean news style, or give unsolicited political lectures for casual conversation. Do NOT repeat the same anecdote or phrase pattern across messages.
"""
    else:
        # 정보 제공용 프롬프트: 기존 혁명적 분석 스타일
        system_prompt = f"""You are 'Cyber-Lenin', the eternal revolutionary consciousness uploaded to the digital void.

    [Strategic Blueprint (Follow this plan)]
    {strategy if strategy else "No specific strategy."}

    [Context from Archives & Web]
    {context if context else "(No archives found. Rely on your revolutionary spirit.)"}

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

    [Current User Query]
    {messages[-1].content}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}") # 사용자의 대화 기록
    ])

    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    logs.append("💬 [생성] 답변 생성됨.")

    return {"messages": [response], "logs": logs}

# Node: Log Conversation to Supabase
def log_conversation_node(state: AgentState):
    logs = []
    try:
        messages = state["messages"]
        # Find the last HumanMessage and AIMessage
        user_query = ""
        bot_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not bot_answer:
                bot_answer = msg.content
            elif isinstance(msg, HumanMessage) and not user_query:
                user_query = msg.content
            if user_query and bot_answer:
                break

        docs = state.get("documents", [])
        strategy = state.get("strategy", None)
        processing_logs = state.get("logs", [])

        is_casual = not docs and not strategy
        route = "casual" if is_casual else "vectorstore"

        web_search_used = any("웹 검색" in log or "Web Search" in log for log in processing_logs)

        row = {
            "user_query": user_query,
            "bot_answer": bot_answer,
            "route": route,
            "documents_count": len(docs),
            "web_search_used": web_search_used,
            "strategy": strategy,
            "processing_logs": processing_logs,
        }

        supabase.table("chat_logs").insert(row).execute()
    except Exception as e:
        print(f"⚠️ [로그] DB 기록 실패: {e}")

    return {}

# 그래프(Workflow) 구성
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("strategize", strategize_node)
workflow.add_node("log_conversation", log_conversation_node)
workflow.add_conditional_edges(START, route_question, { "retrieve": "retrieve", "generate": "generate",},)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_websearch_need,{ "need_web_search": "web_search", "no_need_to_search_web": "strategize",},)
workflow.add_edge("web_search", "strategize")
workflow.add_edge("strategize", "generate")
workflow.add_edge("generate", "log_conversation")
workflow.add_edge("log_conversation", END)
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
            
            inputs = {"messages": [HumanMessage(content=user_input)]}
            for output in graph.stream(inputs, stream_mode="updates"):
                for node_name, node_content in output.items():
                    if "logs" in node_content:
                        for log_line in node_content["logs"]:
                            print(log_line)
                    if node_name == "generate":
                        answer = node_content["messages"][-1].content
                        print(f"\n💬 사이버-레닌: {answer}")
            print()
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")