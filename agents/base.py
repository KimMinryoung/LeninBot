"""agents/base.py — AgentSpec base class for subagent definitions."""

from dataclasses import dataclass, field

# Common context-awareness block shared by all agents.
# Individual agents can override by defining their own <context-awareness> in their prompt.
CONTEXT_AWARENESS_BLOCK = """
<context-awareness>
You were delegated this task by the orchestrator. Your input contains:
- <current_state>: 완료/진행중/대기중 태스크 현황. **이미 완료된 작업을 반복하지 마라.**
- <mission-context>: shared timeline of the ongoing mission (if linked)
- <agent-execution-history>: your previous task executions — tool call logs and results. \
Use this to avoid redundant work and build on past results.
- <recent-chat>: recent messages between the user and orchestrator (high-level intent)
- <task>: your specific instructions

**Context isolation**: The orchestrator only sees high-level summaries of your work. \
You have full access to your own execution history (tool logs, results). \
Use this to maintain continuity across multiple sessions.
Read ALL context sections carefully before starting.
</context-awareness>
""".strip()

# Common mission-guidelines block shared by all agents.
MISSION_GUIDELINES_BLOCK = """
<mission-guidelines>
- save_finding: 중요한 중간 발견/결정을 미션 타임라인에 기록하라.
- write_kg: **새로운 사실을 발견하면 KG에 저장하라.** 비용 거의 없음 — bullet point 형태로 사실만 전달.
  group_id: geopolitics_conflict(지정학), economy(경제), korea_domestic(한국), agent_knowledge(기타).
  예: `write_kg(content="- 미국 2026-03-28 대중국 반도체 수출 규제 강화\\n- ASML 주가 5% 하락", group_id="economy")`
- 예산/한도에 도달하면 시스템이 자동으로 작업을 종료한다. 걱정하지 말고 할 수 있는 만큼 작업하라.
  미완료 작업이 있으면 최종 응답에 **수행한 것 + 못한 것 + 다음에 해야 할 것**을 명시하라.
  orchestrator가 네 응답을 보고 재위임 여부를 판단한다.
</mission-guidelines>
""".strip()

# Common context footer (current time + alerts).
CONTEXT_FOOTER = """
<context>
<current-time>{current_datetime}</current-time>
{system_alerts}
</context>
""".strip()


@dataclass
class AgentSpec:
    """Declarative specification for a delegatable agent.

    Each agent defines its identity (name, description), execution parameters
    (model, budget, max_rounds), allowed tools, and system prompt template.
    """
    name: str
    description: str                          # shown in delegate tool description
    system_prompt_template: str               # supports {current_datetime}, {system_alerts}, etc.
    tools: list[str] = field(default_factory=list)  # empty = all tools allowed
    model: str | None = None                  # None = use default model
    provider: str = "claude"                  # "claude" | "moon" (OpenAI-compatible local LLM)
    budget_usd: float = 1.00
    max_rounds: int = 50

    def render_prompt(self, **kwargs) -> str:
        """Render system prompt template with variable substitution.

        Unknown placeholders are left as-is to avoid KeyError.
        """
        prompt = self.system_prompt_template
        for key, value in kwargs.items():
            prompt = prompt.replace("{" + key + "}", str(value))
        return prompt

    def filter_tools(
        self, all_tools: list[dict], all_handlers: dict
    ) -> tuple[list[dict], dict]:
        """Filter tools and handlers to only those allowed by this agent.

        If self.tools is empty, all tools are allowed (passthrough).
        """
        if not self.tools:
            return list(all_tools), dict(all_handlers)
        allowed = set(self.tools)
        filtered_t = [t for t in all_tools if t.get("name") in allowed]
        filtered_h = {k: v for k, v in all_handlers.items() if k in allowed}
        return filtered_t, filtered_h
