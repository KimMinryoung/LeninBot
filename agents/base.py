"""agents/base.py — AgentSpec base class for subagent definitions."""

from dataclasses import dataclass, field


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
