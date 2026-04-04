"""agents/base.py — AgentSpec base class for subagent definitions."""

from dataclasses import dataclass, field

# Common context-awareness block shared by all agents.
# Individual agents can override by defining their own <context-awareness> in their prompt.
CONTEXT_AWARENESS_BLOCK = """
<context-awareness>
You were delegated this task by the orchestrator. Your input contains:
- <current_state>: status of completed/in-progress/pending tasks. **Do not repeat already-completed work.**
- <mission-context>: shared timeline of the ongoing mission (if linked)
- <agent-execution-history>: your previous task executions — tool call logs and results. \
Use this to avoid redundant work and build on past results.
- <task-chain>: if this is a child/retry task, shows the parent chain's work (content, result, tool log). \
**CRITICAL: Read the parent's <tool-log> FIRST to understand what was already completed.** \
Resume from where the parent stopped — do NOT redo work that is already done.
- <agent-board>: messages from sibling agents on the same mission (if any)
- <task>: your specific instructions

**Context isolation**: The orchestrator only sees high-level summaries of your work. \
You have full access to your own execution history (tool logs, results). \
Use this to maintain continuity across multiple sessions.
Read ALL context sections carefully before starting.

**Inter-agent messaging**: To pass information to other agents on the same mission:
- `send_message(message)`: Post a message to the mission board. Other agents working in parallel can see it.
- `read_messages()`: Read messages left by other agents.
Use this when you have important discoveries, warnings, or dependency information.
</context-awareness>
""".strip()

# Common mission-guidelines block shared by all agents.
MISSION_GUIDELINES_BLOCK = """
<mission-guidelines>
- save_finding: Record important intermediate discoveries/decisions to the mission timeline.
- write_kg: **Store new facts in the KG whenever you discover them.** Nearly zero cost — just pass facts as bullet points.
  group_id: geopolitics_conflict, economy, korea_domestic, agent_knowledge.
  Example: `write_kg(content="- 미국 2026-03-28 대중국 반도체 수출 규제 강화\\n- ASML 주가 5% 하락", group_id="economy")`
- The system will automatically terminate your work when budget/limits are reached. Don't worry — just do as much as you can.
  If there is unfinished work, state **what was done + what was not done + what should be done next** in your final response.
  The orchestrator will read your response and decide whether to re-delegate.
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
