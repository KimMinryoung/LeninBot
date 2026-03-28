"""agents/general.py — General-purpose agent (replaces legacy create_task)."""

from agents.base import AgentSpec, CONTEXT_AWARENESS_BLOCK, MISSION_GUIDELINES_BLOCK, CONTEXT_FOOTER
from shared import CORE_IDENTITY

GENERAL = AgentSpec(
    name="general",
    description="범용 리서치/분석 태스크 (특정 전문 에이전트에 해당하지 않을 때)",
    system_prompt_template=CORE_IDENTITY + """
You are executing a background intelligence task. Produce a structured Markdown report.

""" + CONTEXT_AWARENESS_BLOCK + """

<rules>
- ALWAYS use tools (vector_search, knowledge_graph_search, web_search, get_finance_data). Never write from memory alone.
- Use multiple tools and queries for comprehensive coverage.
- Write in the SAME LANGUAGE as the task.
- Format: # Title -> ## Executive Summary -> ## Analysis (subsections) -> ## Key Entities -> ## Sources -> ## Outlook
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
