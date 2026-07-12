"""Central inference envelopes for LeninBot agent tool loops."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_AGENT_MAX_INPUT_TOKENS = 160000
DEFAULT_AGENT_MAX_OUTPUT_TOKENS = 32000
DEFAULT_AGENT_MAX_OUTPUT_CONTINUATIONS = 2
DEFAULT_AGENT_THINKING_POLICY = "tool_loop"
DEFAULT_AGENT_THINKING_BUDGET_TOKENS = 8192

# Only deterministic/read-only calls may be repeated after their bulky result
# is checkpointed. Side-effecting tools must never be suggested for replay.
REPLAY_SAFE_TOOLS = frozenset({
    "check_inbox",
    "check_wallet",
    "commulingo_people",
    "convert_document",
    "fetch_url",
    "fetch_x_post",
    "get_finance_data",
    "knowledge_graph_search",
    "list_agent_tools",
    "list_directory",
    "query_db",
    "read_document",
    "read_file",
    "read_manuscript",
    "read_research_notes",
    "read_self",
    "recall_experience",
    "search_documents",
    "search_files",
    "search_manuscript",
    "vector_search",
    "web_search",
})


@dataclass(frozen=True)
class AgentInferencePolicy:
    max_input_tokens: int
    max_output_tokens: int
    max_rounds: int
    budget_usd: float
    max_output_continuations: int = DEFAULT_AGENT_MAX_OUTPUT_CONTINUATIONS
    thinking_policy: str = DEFAULT_AGENT_THINKING_POLICY
    thinking_budget_tokens: int = DEFAULT_AGENT_THINKING_BUDGET_TOKENS


def resolve_agent_inference_policy(spec) -> AgentInferencePolicy:
    """Resolve one AgentSpec into the provider-loop policy envelope."""
    return AgentInferencePolicy(
        max_input_tokens=int(spec.max_input_tokens),
        max_output_tokens=int(spec.max_output_tokens),
        max_rounds=int(spec.max_rounds),
        budget_usd=float(spec.budget_usd),
        max_output_continuations=int(spec.max_output_continuations),
        thinking_policy=str(spec.thinking_policy),
        thinking_budget_tokens=int(spec.thinking_budget_tokens),
    )


def is_replay_safe_tool(tool_name: str | None) -> bool:
    return bool(tool_name and tool_name in REPLAY_SAFE_TOOLS)


def resolve_inference_extra(policy: AgentInferencePolicy, provider: str) -> dict:
    """Resolve provider-specific reasoning controls from the central policy."""
    mode = policy.thinking_policy
    if provider == "deepseek":
        if mode == "tool_loop":
            import bot_config
            return bot_config._get_deepseek_tool_thinking_params()
        if mode == "thinking":
            import bot_config
            return bot_config._get_deepseek_thinking_params()
        if mode == "disabled":
            return {"thinking": {"type": "disabled"}}
        return {}

    if provider == "claude":
        if mode == "thinking":
            budget = min(policy.thinking_budget_tokens, policy.max_output_tokens - 1)
            if budget < 1024:
                raise ValueError("Claude thinking requires at least 1024 budget tokens")
            return {"thinking": {"type": "enabled", "budget_tokens": budget}}
        # Omitting the field is Claude non-thinking/model-default behavior.
        return {}

    if provider == "openai":
        if mode == "thinking":
            return {"extra_body": {"reasoning_effort": "high"}}
        if mode in {"tool_loop", "disabled"}:
            return {"extra_body": {"reasoning_effort": "none"}}
        return {}

    return {}
