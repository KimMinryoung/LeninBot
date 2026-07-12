"""Central inference envelopes for LeninBot agent tool loops."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_AGENT_MAX_INPUT_TOKENS = 96000
DEFAULT_AGENT_MAX_OUTPUT_TOKENS = 16384
DEFAULT_AGENT_MAX_OUTPUT_CONTINUATIONS = 2
DEFAULT_AGENT_THINKING_POLICY = "tool_loop"


@dataclass(frozen=True)
class AgentInferencePolicy:
    max_input_tokens: int
    max_output_tokens: int
    max_rounds: int
    budget_usd: float
    max_output_continuations: int = 2
    thinking_policy: str = "tool_loop"


def resolve_agent_inference_policy(spec) -> AgentInferencePolicy:
    """Resolve one AgentSpec into the provider-loop policy envelope."""
    return AgentInferencePolicy(
        max_input_tokens=int(spec.max_input_tokens),
        max_output_tokens=int(spec.max_output_tokens),
        max_rounds=int(spec.max_rounds),
        budget_usd=float(spec.budget_usd),
        max_output_continuations=int(spec.max_output_continuations),
        thinking_policy=str(spec.thinking_policy),
    )


def resolve_inference_extra(policy: AgentInferencePolicy, provider: str) -> dict:
    """Resolve provider-specific thinking controls from the central policy."""
    if provider == "deepseek" and policy.thinking_policy == "tool_loop":
        import bot_config
        return bot_config._get_deepseek_tool_thinking_params()
    if provider == "deepseek" and policy.thinking_policy == "thinking":
        import bot_config
        return bot_config._get_deepseek_thinking_params()
    if provider == "deepseek" and policy.thinking_policy == "disabled":
        return {"thinking": {"type": "disabled"}}
    return {}
