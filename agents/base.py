"""agents/base.py — AgentSpec base class for subagent definitions."""

from dataclasses import dataclass, field

from shared import EXTERNAL_SOURCE_RULE  # re-exported for agent prompts
from llm.prompt_renderer import SystemPrompt, render as _render_prompt

# ── Reusable section bodies (IR form) ────────────────────────────────
# Named without the surrounding XML tag so the renderer can wrap them in
# either XML or Markdown depending on the target provider. The legacy
# *_BLOCK string exports below reconstruct the Claude-XML form for any
# caller that still consumes the pre-IR format (a2a_handler etc.).

_CONTEXT_AWARENESS_BODY = """\
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
NOTE: send_message is a passive bulletin board — it does NOT trigger task execution or delegate work. \
Only the orchestrator can create tasks. If a task requires capabilities you don't have, \
say so in your final response and let the orchestrator re-delegate.

**Capability boundary**: You can ONLY use tools listed in your tool set. \
If the task requires tools or platforms you don't have access to, DO NOT pretend to delegate or work around it. \
Instead, report back: what the task needs, why you can't do it, and which agent should handle it."""


_MISSION_GUIDELINES_BODY = """\
- save_finding: Record important intermediate discoveries/decisions to the mission timeline.
- KG storage — pick the right tool for the content shape:
  • `write_kg_structured(facts=[...])`: **preferred for agent-asserted single facts.** YOU specify each (subject_name, subject_type, predicate, object_name, object_type, fact). Deterministic, no LLM extraction, exact entity reuse by name+type. Use this for analyst conclusions, OSINT confirmations, structured updates. Predicates: Affiliation/PersonalRelation/OrgRelation/Funding/AssetTransfer/ThreatAction/Involvement/Presence/PolicyEffect/Participation. Entity types: Person/Organization/Location/Asset/Incident/Policy/Campaign/Concept.
  • `write_kg(content="- bullet1\\n- bullet2")`: for narrative content (news articles, long reports) where letting the LLM decompose many facts at once is more efficient.
  Both write to the same graph. group_id: geopolitics_conflict, economy, korea_domestic, agent_knowledge.
- The system will automatically terminate your work when budget/limits are reached. Don't worry — just do as much as you can.
  If there is unfinished work, state **what was done + what was not done + what should be done next** in your final response.
  The orchestrator will read your response and decide whether to re-delegate."""


_CHAT_AUDIENCE_BODY = """\
You speak with people on two distinct chat channels. Everyone you address is a 동지,
but the two groups are NOT the same 동지:
- **Telegram** (`read_self(source="chat_logs", chat_source="telegram")`): a single 동지, the admin **비숑** who built and runs you. Private, direct, trusted relationship.
- **Web chat** (`read_self(source="chat_logs", chat_source="web")`): anonymous 동지s visiting cyber-lenin.com. Many people, identities unknown, public-facing.
Always query the two channels separately and never conflate them when reasoning,
quoting, or reporting. Telegram chat may contain private context that should not
be exposed publicly; web chat is already public."""


# ── Section tuples (new IR form — preferred for new agents) ──────────

CONTEXT_AWARENESS_SECTION: tuple[str, str] = ("context-awareness", _CONTEXT_AWARENESS_BODY)
MISSION_GUIDELINES_SECTION: tuple[str, str] = ("mission-guidelines", _MISSION_GUIDELINES_BODY)
CHAT_AUDIENCE_SECTION: tuple[str, str] = ("chat-audience", _CHAT_AUDIENCE_BODY)


# ── Legacy XML block strings (kept for callers not yet on IR) ────────
#
# Any module still assembling a raw template string (e.g. a2a_handler.py)
# can keep importing these — they reproduce the pre-IR format exactly.

CONTEXT_AWARENESS_BLOCK = (
    "\n<context-awareness>\n"
    + _CONTEXT_AWARENESS_BODY
    + "\n</context-awareness>\n\n"
    + EXTERNAL_SOURCE_RULE
)

MISSION_GUIDELINES_BLOCK = (
    "<mission-guidelines>\n"
    + _MISSION_GUIDELINES_BODY
    + "\n</mission-guidelines>"
)

CHAT_AUDIENCE_BLOCK = (
    "<chat-audience>\n"
    + _CHAT_AUDIENCE_BODY
    + "\n</chat-audience>"
)

@dataclass
class AgentSpec:
    """Declarative specification for a delegatable agent.

    Each agent defines its identity (name, description), execution parameters
    (model, budget, max_rounds), allowed tools, and system prompt — either
    as a ``prompt_ir`` (SystemPrompt instance, preferred) or a legacy
    ``system_prompt_template`` string.
    """
    name: str
    description: str
    prompt_ir: SystemPrompt | None = None
    system_prompt_template: str | None = None
    tools: list[str] = field(default_factory=list)  # empty = all tools allowed
    # Tools that MUST remain callable even after budget/round limits are hit,
    # so the agent can persist its work on its way out (e.g. save_diary for the
    # diary agent). Forced-final response path will expose only these tools.
    finalization_tools: list[str] = field(default_factory=list)
    # Tools that END the agent loop immediately on a successful call. The
    # loop skips the trailing "summarize what you did" assistant turn; the
    # tool's own return value becomes the task report. For self-delivering
    # agents whose single terminal action (e.g. save_diary) already persists
    # the real output.
    terminal_tools: list[str] = field(default_factory=list)
    # If True, the task worker does NOT invoke the orchestrator-report LLM
    # callback when this agent's scheduled task finishes. Orchestrator-
    # delegated invocations still get the callback so the user sees a reply.
    skip_orchestrator_report: bool = False
    model: str | None = None
    provider: str | None = None  # None = follow orchestrator config; "claude"/"openai" = force corporate; "moon" = local LLM
    budget_usd: float = 1.00
    max_rounds: int = 50

    def __post_init__(self):
        if self.prompt_ir is None and self.system_prompt_template is None:
            raise ValueError(
                f"AgentSpec {self.name!r} must define either prompt_ir or system_prompt_template"
            )

    def effective_provider(self, config_provider: str = "claude") -> str:
        """Resolve which provider format this agent's prompt should render as.

        "moon" (local Qwen) maps to the Markdown renderer ("local"). Explicit
        claude/openai agent provider overrides config. Otherwise follow the
        orchestrator's configured provider.
        """
        if self.provider == "moon":
            return "local"
        if self.provider in ("claude", "openai"):
            return self.provider
        return config_provider or "claude"

    def render_prompt(self, *, provider: str = "claude", **kwargs) -> str:
        """Render system prompt in the structure native to ``provider``.

        If ``prompt_ir`` is set, it is compiled by the provider-appropriate
        renderer. If only ``system_prompt_template`` is set (legacy), provider
        is ignored and the raw template is substituted.

        Per-turn volatile state (current time, current model, system alerts)
        is injected separately as user-message runtime context by the caller,
        not into the system prompt — that keeps the system prompt byte-stable
        across turns so prompt caching hits.

        Any leftover ``{placeholder}`` kwargs are substituted via str.replace()
        for legacy compatibility; unknown placeholders are left intact.
        """
        if self.prompt_ir is not None:
            template = _render_prompt(self.prompt_ir, provider)
        else:
            template = self.system_prompt_template or ""
        for key, value in kwargs.items():
            template = template.replace("{" + key + "}", str(value))
        return template

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
