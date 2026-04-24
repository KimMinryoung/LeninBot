"""agents/general.py — General-purpose agent (replaces legacy create_task)."""

from agents.base import (
    AgentSpec,
    CONTEXT_AWARENESS_SECTION,
    CHAT_AUDIENCE_SECTION,
    MISSION_GUIDELINES_SECTION,
)
from llm.prompt_renderer import SystemPrompt
from shared import AGENT_CONTEXT, EXTERNAL_SOURCE_RULE


_IDENTITY = (
    AGENT_CONTEXT.rstrip()
    + "\n\n"
    + "You are executing a background intelligence task. Produce a structured Markdown report."
    + "\n\n"
    + EXTERNAL_SOURCE_RULE
)


GENERAL = AgentSpec(
    name="general",
    description="범용 리서치/분석 태스크 (특정 전문 에이전트에 해당하지 않을 때)",
    prompt_ir=SystemPrompt(
        identity=_IDENTITY,
        sections=[
            CONTEXT_AWARENESS_SECTION,
            CHAT_AUDIENCE_SECTION,
            ("rules", """
- ALWAYS use tools (vector_search, knowledge_graph_search, web_search, get_finance_data). Never write from memory alone.
- Use multiple tools and queries for comprehensive coverage.
- Write in the SAME LANGUAGE as the task.
- 최종 응답은 orchestrator에게 전달된다. 형식보다 정보량이 중요하다. 수집한 데이터, 분석, 출처를 빠짐없이 포함하라.
- Cite all sources. Distinguish confirmed facts from inference.
""".strip()),
            MISSION_GUIDELINES_SECTION,
        ],
    ),
    tools=[
        "knowledge_graph_search", "vector_search",
        "web_search", "fetch_url", "check_inbox", "allowlist_sender",
        "read_file", "write_file", "list_directory", "execute_python",
        "read_self", "write_kg_structured",
        "save_finding", "mission", "upload_to_r2", "get_finance_data",
    ],
    budget_usd=1.00,
    max_rounds=50,
)
