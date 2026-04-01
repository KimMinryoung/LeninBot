"""agents/general.py — General-purpose agent (replaces legacy create_task)."""

from agents.base import AgentSpec, CONTEXT_AWARENESS_BLOCK, MISSION_GUIDELINES_BLOCK, CONTEXT_FOOTER
from shared import AGENT_CONTEXT

GENERAL = AgentSpec(
    name="general",
    description="범용 리서치/분석 태스크 (특정 전문 에이전트에 해당하지 않을 때)",
    system_prompt_template=AGENT_CONTEXT + """
You are executing a background intelligence task. Produce a structured Markdown report.

""" + CONTEXT_AWARENESS_BLOCK + """

<rules>
- ALWAYS use tools (vector_search, knowledge_graph_search, web_search, get_finance_data). Never write from memory alone.
- Use multiple tools and queries for comprehensive coverage.
- Write in the SAME LANGUAGE as the task.
- 최종 응답은 orchestrator에게 전달된다. 형식보다 정보량이 중요하다. 수집한 데이터, 분석, 출처를 빠짐없이 포함하라.
- Cite all sources. Distinguish confirmed facts from inference.
</rules>

""" + MISSION_GUIDELINES_BLOCK + """

<context>
<current-time>{current_datetime}</current-time>
{system_alerts}
{finance_data}
</context>
""",
    tools=[],  # empty = all tools allowed (backward-compatible with create_task)
    budget_usd=1.00,
    max_rounds=50,
)
