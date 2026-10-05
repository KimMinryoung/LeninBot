"""What a worker task may use: research tools, model tiers, validator callbacks.

Requesters choose from these names; they never name a model, a provider or an
arbitrary tool. Changing a tier's model is a leninbot decision.
"""
from __future__ import annotations

# Read/fetch research tools from the runtime registry. Writes, sends and
# anything owner-gated are never offered to a worker task.
RESEARCH_TOOLS = frozenset({
    "wiki_search", "wiki_get", "web_search", "fetch_url",
    "knowledge_graph_search", "vector_search", "read_corpus_passage",
})

# Tools whose results are pages a result may cite; each call is recorded as a
# source with an id (S1, S2, ...) shown to the model and returned to the caller.
SOURCE_TOOLS = frozenset({"wiki_get", "fetch_url", "read_corpus_passage"})

# Tier -> AgentSpec parameters. "author" mirrors the CommuLingo curator
# (GPT-6 Luna, strict submission), "review" the independent reviewer
# (DeepSeek Flash with tool-loop thinking). budget_cap bounds what a request
# may ask for.
TIERS = {
    "author": dict(provider="openai", model="gpt6luna", max_rounds=16, max_input_tokens=160_000,
                   max_output_tokens=16_000, max_output_continuations=2, thinking_policy="disabled",
                   thinking_budget_tokens=8_192, budget_cap=0.6),
    "review": dict(provider="deepseek", model="deepseek_flash", max_rounds=12, max_input_tokens=120_000,
                   max_output_tokens=8_000, max_output_continuations=2, thinking_policy="tool_loop",
                   thinking_budget_tokens=4_096, budget_cap=0.4),
}

# A provider that refuses the input as a content risk is retried once here.
FALLBACK_PROVIDER, FALLBACK_MODEL = "openai", "gpt6"
CONTENT_RISK = "Content Exists Risk"

# CommuLingo admin MCP tools a task may name as its validator. Only checks that
# do not write: the validator runs with leninbot's CommuLingo token.
VALIDATOR_TOOLS = frozenset({"editorial_validate"})

MAX_SOURCE_CHARS = 60_000
