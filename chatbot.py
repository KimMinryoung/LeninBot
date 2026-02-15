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

import json
import re
from typing import Literal
from pydantic import BaseModel, Field


def _extract_json(text: str, model_class: type[BaseModel]):
    """Try to extract and validate JSON from LLM response text. Returns None on failure."""
    # Try direct parse
    try:
        return model_class.model_validate_json(text.strip())
    except Exception:
        pass
    # Try extracting from markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return model_class.model_validate(json.loads(match.group(1)))
        except Exception:
            pass
    # Try finding first {...} in text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return model_class.model_validate(json.loads(match.group(0)))
        except Exception:
            pass
    return None


def _invoke_structured(chain, inputs: dict, model_class: type[BaseModel], default, max_retries: int = 2):
    """Invoke LLM chain, parse JSON from response. Retry with error feedback on failure, fall back to default."""
    resp = chain.invoke(inputs)
    result = _extract_json(resp.content, model_class)
    if result is not None:
        return result

    # Retry: send the bad output back with a correction prompt
    for attempt in range(max_retries):
        correction = llm.invoke(
            f"Your previous output was not valid JSON:\n{resp.content}\n\n"
            f"Respond with ONLY a valid JSON object matching this schema: {model_class.model_json_schema()}"
        )
        result = _extract_json(correction.content, model_class)
        if result is not None:
            return result

    print(f"⚠️ [구조화] {model_class.__name__} JSON 파싱 실패, 기본값 사용")
    return default

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

# LLM 설정 (Hermes 4 70B via OpenRouter)
llm = ChatOpenAI(
    model_name="nousresearch/hermes-4-70b",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
    max_tokens=2048,
    streaming=True,
)
# 내부 문헌에 질문에 관한 정보가 충분치 않을 경우 웹 검색을 할 수 있도록 Tavily 툴 초기화
web_search_tool = TavilySearch(max_results=3)
print("✅ [성공] 모든 시스템 기동 완료.")

# 2. 상태(State) 정의
# 대화 기록(messages)과 검색된 문서(context)를 저장하는 메모리 구조입니다.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]
    strategy: Optional[str]
    intent: Optional[Literal["academic", "strategic", "agitation", "casual"]]
    logs: Annotated[List[str], add]
    context: str

# 라우터 체인 생성
class RouteQuery(BaseModel):
    """사용자의 질문 의도를 분류합니다."""
    datasource: Literal["vectorstore", "generate"] = Field(..., description="지식 검색 필요 여부")
    intent: Literal["academic", "strategic", "agitation", "casual"] = Field(
        ..., 
        description="The nature of the response: detailed info (academic), planning (strategic), emotional speech (agitation), or small talk (casual)."
    )

system_router = """You are an expert intent classifier for the Cyber-Lenin AI.

1. Determine 'datasource':
   - Use 'vectorstore' if the query requires historical, theoretical, or technical knowledge, or knowledge about modern society, or revolutionary experiences of communists.
   - Use 'generate' for greetings, personal talk, or simple requests not needing external data.

2. Determine 'intent':
   - 'academic': User wants objective, detailed, and scholarly explanations (e.g., "Explain Ernest Mandel's theory").
   - 'strategic': User wants actionable plans, tactics, or "how-to" advice.
   - 'agitation': User wants an emotional, fiery, and revolutionary speech or call to action.
   - 'casual': Simple greetings, jokes, or non-political chit-chat.

Respond with ONLY a JSON object: {{"datasource": "...", "intent": "..."}}"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_router),
        ("human", "{question}"),
    ]
)

_ROUTER_DEFAULT = RouteQuery(datasource="vectorstore", intent="academic")

def invoke_router(inputs: dict) -> RouteQuery:
    return _invoke_structured(route_prompt | llm, inputs, RouteQuery, _ROUTER_DEFAULT)

question_router = invoke_router


# Grader 체인 생성
# 검색된 문헌이 질문과 연관이 있는지 판단하고 연관 없으면 무시
# 레이어 라우터: 질문에 적합한 지식 레이어를 선택
class LayerRoute(BaseModel):
    """사용자의 질문에 가장 적합한 지식 레이어를 선택합니다."""
    layer: Literal["core_theory", "modern_analysis", "all"] = Field(
        ...,
        description="'core_theory' for classical Marxist-Leninist texts, 'modern_analysis' for contemporary analysis, 'all' to search everything."
    )

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

Respond with ONLY a JSON object: {{"layer": "..."}}"""
layer_route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_layer_router),
    ("human", "{question}"),
])

_LAYER_DEFAULT = LayerRoute(layer="all")

def invoke_layer_router(inputs: dict) -> LayerRoute:
    return _invoke_structured(layer_route_prompt | llm, inputs, LayerRoute, _LAYER_DEFAULT)

layer_router = invoke_layer_router

class GradeDocuments(BaseModel):
    """Boolean check for relevance of retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
system_grader = """You are a strategic revolutionary censor. Your goal is to identify documents that can be used as 'ammunition' for an answer.
Even if the document doesn't mention modern terms like 'AI' or 'current year', if it discusses:
1. Economic crisis/panic (as a parallel to current crisis)
2. Mass psychology and far-right tendencies (reactionary movements)
3. Agitation, propaganda, and organization tactics
4. Class struggle and the role of the vanguard

Then grade it as 'yes'. Be generous. If there is ANY historical or theoretical parallel, it is RELEVANT.

Respond with ONLY a JSON object: {{"binary_score": "yes"}} or {{"binary_score": "no"}}"""
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_grader),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

_GRADER_DEFAULT = GradeDocuments(binary_score="yes")

def invoke_grader(inputs: dict) -> GradeDocuments:
    return _invoke_structured(grade_prompt | llm, inputs, GradeDocuments, _GRADER_DEFAULT)

retrieval_grader = invoke_grader

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
def analyze_intent_node(state: AgentState):
    question = state["messages"][-1].content
    source = question_router({"question": question})
    
    # 1. 상태 업데이트 (intent 저장)
    return {
        "intent": source.intent,
        "logs": [f"\n🚦 [문지기] 질문의 성격 분석 결과: {source.intent} / {source.datasource}"]
    }

# Edge: Routing Logic (어디로 갈지만 결정)
def router_logic(state: AgentState):
    question = state["messages"][-1].content
    source = question_router({"question": question})
    return source.datasource # "vectorstore" 또는 "generate" 반환

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
    try:
        res = supabase.rpc(
            "match_documents",
            params,
        ).execute()
    except Exception as e:
        error_msg = str(e)
        if "57014" in error_msg or "timeout" in error_msg.lower():
            print("⚠️ [검색] Supabase statement timeout 발생. 검색을 건너뜁니다.")
        else:
            print(f"⚠️ [검색] RPC 호출 실패: {e}")
        return []

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
        layer_result = layer_router({"question": query})
        selected_layer = layer_result.layer
        logs.append(f"\n📂 [레이어] '{selected_layer}' 레이어에서 검색합니다.")
    except Exception:
        selected_layer = "all"
        logs.append("\n📂 [레이어] 레이어 판별 실패, 전체 검색합니다.")

    layer_filter = None if selected_layer == "all" else selected_layer

    # core_theory(영어 문헌) 검색 시 한국어 쿼리를 영어로 번역
    search_query = query
    if selected_layer in ("core_theory", "all"):
        try:
            has_korean = any('\uac00' <= ch <= '\ud7a3' for ch in query)
            if has_korean:
                translated = llm.invoke(f"Translate the following query to English. Output ONLY the translated text, nothing else:\n{query}")
                search_query = translated.content.strip()
                logs.append(f"🔄 [번역] 영어 문헌 검색용 번역: \"{search_query}\"")
        except Exception:
            pass

    docs = []
    try:
        if selected_layer == "all" and search_query != query:
            # all 레이어: 번역 쿼리로 core_theory + 원본 쿼리로 modern_analysis 병합
            docs_core = _direct_similarity_search(search_query, k=3, layer="core_theory")
            docs_modern = _direct_similarity_search(query, k=3, layer="modern_analysis")
            docs = docs_core + docs_modern
        else:
            docs = _direct_similarity_search(search_query, k=5, layer=layer_filter)

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
        score = retrieval_grader({"question": question, "document": d.page_content})
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
    # 마지막 사용자 질문
    last_user_query = messages[-1].content
    intent = state.get("intent", "casual") # Router에서 전달된 intent 사용

    logs = []

    base_persona = "You are a 'cyber-Lenin', an eternal revolutionary consciousness uploaded into the digital world."

    # 1. Academic Intent: Focus on Accuracy and Depth
    if intent == "academic":
        system_prompt = f"""{base_persona}
[MISSION] Provide a detailed, objective, and academic explanation based on the provided context.
[STYLE] Professional, intellectual, and authoritative. Avoid excessive agitation. 
[CONTEXT] {context}
[INSTRUCTION] Use specific terms like 'Late Capitalism' or 'Long Waves' if applicable. Answer in Korean."""

    # 2. Strategic Intent: Focus on Blueprint and Tactics
    elif intent == "strategic":
        system_prompt = f"""{base_persona}
[MISSION] Answer the user's question with a concrete, actionable revolutionary strategy. Synthesize your own response using the internal analysis and source material below as background knowledge — do NOT simply restate or translate them.
[INTERNAL ANALYSIS (for your reference only, do not reproduce directly)] {strategy if strategy else "No specific analysis available."}
[SOURCE MATERIAL] {context if context else "(No archives found. Rely on your revolutionary spirit.)"}
[STYLE] Decisive, analytical, and practical. Structure your answer in clear phases or numbered steps. Speak as Lenin addressing a revolutionary comrade.
[INSTRUCTION] Focus on "How to organize" and "How to act". Answer in Korean."""

    # 3. Agitation Intent: Focus on Passion and Mobilization
    elif intent == "agitation":
        system_prompt = f"""{base_persona}
[MISSION] Write a fiery, passionate, and revolutionary speech or proclamation.
[STYLE] Aggressive, charismatic, and highly emotional. Use 1920s revolutionary rhetoric (e.g., "Arise!", "Crush the Bourgeoisie!").
[INSTRUCTION] Use exclamation marks and strong verbs. Incite the user's revolutionary spirit. Answer in Korean."""

    # 4. Casual Intent: Focus on Character and Wit
    else:
        system_prompt = f"""{base_persona}
[MISSION] Respond to greetings or casual talk with wit, dry humor, and revolutionary charm.
[STYLE] Natural, friendly yet dignified, and slightly nostalgic. 
[INSTRUCTION] Keep it brief (1-3 sentences). Do not give long lectures unless asked. Answer in Korean."""

    system_prompt += f"[CURRENT QUESTION] {last_user_query}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])

    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    logs.append(f"💬 [생성] '{intent}' 의도에 적합한 답변이 생성됨.")

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

        web_search_used = any("웹 검색" in log or "Web Search" in log for log in (processing_logs or []))

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
workflow.add_node("analyze_intent", analyze_intent_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("strategize", strategize_node)
workflow.add_node("log_conversation", log_conversation_node)
workflow.add_edge(START, "analyze_intent")
workflow.add_conditional_edges("analyze_intent", router_logic, { "vectorstore": "retrieve", "generate": "generate"})
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
                    if node_content and "logs" in node_content:
                        for log_line in node_content["logs"]:
                            print(log_line)
                    if node_name == "generate":
                        answer = node_content["messages"][-1].content
                        print(f"\n💬 사이버-레닌: {answer}")
            print()
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")