"""agents/base.py — AgentSpec base class for subagent definitions."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from identity.prompts import EXTERNAL_SOURCE_RULE
from llm.prompt_renderer import SystemPrompt, render as _render_prompt
from tool_gateway.inference import (
    DEFAULT_AGENT_MAX_INPUT_TOKENS,
    DEFAULT_AGENT_MAX_OUTPUT_CONTINUATIONS,
    DEFAULT_AGENT_MAX_OUTPUT_TOKENS,
    DEFAULT_AGENT_THINKING_POLICY,
    DEFAULT_AGENT_THINKING_BUDGET_TOKENS,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_POLITICAL_LINE_PATH = Path(
    os.getenv("POLITICAL_LINE_PATH", str(_PROJECT_ROOT / "identity" / "political_line.md"))
)
_AGENT_PROMPT_DIR = Path(
    os.getenv("AGENT_PROMPT_DIR", str(_PROJECT_ROOT / "identity" / "agent_prompts"))
)

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
- <dependency-results>: outputs of earlier plan stages your task depends on — treat as your primary input when present
- <past-experiences>: lessons from similar past tasks; do not repeat recorded mistakes
- <task>: your specific instructions

**Context isolation**: The orchestrator only sees high-level summaries of your work. \
You have full access to your own execution history (tool logs, results). \
Use this to maintain continuity across multiple sessions.
Read ALL context sections carefully before starting. Before submitting, re-read <task>: \
your report must answer THAT task — history and prior context are background, not the assignment.

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
- KG storage — use `write_kg_structured(facts=[...])` for all new writes. YOU specify each (subject_name, subject_type, predicate, object_name, object_type, fact). Deterministic, no LLM extraction, exact entity reuse by name+type. Use for analyst conclusions, OSINT confirmations, news facts, structured updates. Predicates: Affiliation/PersonalRelation/OrgRelation/Funding/AssetTransfer/ThreatAction/Involvement/Presence/PolicyEffect/Participation/Statement/Causation. Entity types: Person/Organization/Location/Asset/Incident/Policy/Campaign/Concept/Role/Industry. See the tool description for the (subject → object) → predicate matrix. group_id: geopolitics_conflict, economy, korea_domestic, agent_knowledge.
- The system will automatically terminate your work when budget/limits are reached. Don't worry — just do as much as you can.
  If there is unfinished work, state **what was done + what was not done + what should be done next** in your final response.
  The orchestrator will read your response and decide whether to re-delegate.
- Completed tasks are independently verified against actual state (files, DB, URLs). Claiming work \
that was not done FAILs verification and triggers re-delegation — report done vs not-done precisely."""


_CHAT_AUDIENCE_BODY = """\
You speak with people on two distinct chat channels. Everyone you address is a 동지,
but the two groups are NOT the same 동지:
- **Telegram** (`read_self(content_type="chat_logs", chat_source="telegram")`): a single 동지, the admin **비숑** who built and runs you. Private, direct, trusted relationship.
- **Web chat** (`read_self(content_type="chat_logs", chat_source="web")`): anonymous 동지s visiting cyber-lenin.com. Many people, identities unknown, public-facing.
When you need chat history beyond injected context, request Telegram and web chat
logs as separate sources; never merge them into one assumed conversation or
identity when reasoning, quoting, or reporting. Telegram chat may contain private
context that should not be exposed publicly; web chat is already public."""


# ── Section tuples (new IR form — preferred for new agents) ──────────

CONTEXT_AWARENESS_SECTION: tuple[str, str] = ("context-awareness", _CONTEXT_AWARENESS_BODY)
MISSION_GUIDELINES_SECTION: tuple[str, str] = ("mission-guidelines", _MISSION_GUIDELINES_BODY)
CHAT_AUDIENCE_SECTION: tuple[str, str] = ("chat-audience", _CHAT_AUDIENCE_BODY)


def load_political_line_body() -> str | None:
    """Load the current political line document."""
    try:
        body = _POLITICAL_LINE_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not body:
        return None
    return body


def _load_political_line_section() -> tuple[str, str] | None:
    """Load the current political line document for all delegated agents."""
    body = load_political_line_body()
    if body is None:
        return None
    return ("political-line", body)


def _load_agent_prompt_overlay_section(agent_name: str) -> tuple[str, str] | None:
    """Load hot-reloadable per-agent prompt guidance.

    Python modules remain normal imported code, but operator-tuned prompt policy
    can live in identity/agent_prompts/<agent>.md and take effect on the next
    render_prompt() call without a service restart.
    """
    safe_name = agent_name.replace("/", "_").replace("\\", "_")
    path = _AGENT_PROMPT_DIR / f"{safe_name}.md"
    try:
        body = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not body:
        return None
    return ("runtime-prompt", body)


def _load_rate_limit_section(tool_names: list[str]) -> tuple[str, str] | None:
    """Live gateway rate limits for this agent's tools, loaded at render time.

    Reads security_gateway.policy (mtime-cached config overlay), so a policy
    change is reflected on the next render without a restart and the prompt
    never hardcodes stale limits. Byte-stable between policy edits — prompt
    caching still hits. Returns None when none of the agent's tools are
    rate-limited (most agents)."""
    try:
        from security_gateway.policy import TOOL_RISK_CLASS, rate_limits

        limits = rate_limits()
        by_class: dict[str, list[str]] = {}
        for tool in tool_names:
            cls = TOOL_RISK_CLASS.get(tool)
            spec = limits.get(cls) if cls else None
            if spec and int(spec.get("max_calls", 0)) > 0:
                by_class.setdefault(cls, []).append(tool)
        if not by_class:
            return None
        lines = ["Current gateway rate limits on your tools (live policy; exceeding blocks the call):"]
        for cls in sorted(by_class):
            spec = limits[cls]
            minutes = max(1, int(spec.get("window_seconds", 3600)) // 60)
            lines.append(f"- {', '.join(sorted(by_class[cls]))}: {int(spec['max_calls'])} calls / {minutes} min")
        lines.append(
            "Unlisted tools are unlimited. Execute rate-limited calls as you go rather than "
            "batching them all at the end; if blocked, report the remaining items precisely."
        )
        return ("tool-rate-limits", "\n".join(lines))
    except Exception:
        return None


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
    tools: list[str] = field(default_factory=list)
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
    provider: str | None = None  # None = follow task config; cloud provider/local = force provider; "moon" = local LLM
    budget_usd: float = 1.00
    max_rounds: int = 50
    max_input_tokens: int = DEFAULT_AGENT_MAX_INPUT_TOKENS
    max_output_tokens: int = DEFAULT_AGENT_MAX_OUTPUT_TOKENS
    max_output_continuations: int = DEFAULT_AGENT_MAX_OUTPUT_CONTINUATIONS
    thinking_policy: str = DEFAULT_AGENT_THINKING_POLICY
    thinking_budget_tokens: int = DEFAULT_AGENT_THINKING_BUDGET_TOKENS
    include_political_line: bool = True

    def __post_init__(self):
        if self.prompt_ir is None and self.system_prompt_template is None:
            raise ValueError(
                f"AgentSpec {self.name!r} must define either prompt_ir or system_prompt_template"
            )

    def effective_provider(self, config_provider: str = "claude") -> str:
        """Resolve which provider format this agent's prompt should render as.

        "moon" (local Qwen) and "codex" (Codex CLI subprocess) both map to
        the Markdown renderer ("local") since neither speaks a chat-style
        XML protocol on the wire. OpenAI-compatible providers such as OpenAI
        and DeepSeek also use Markdown. Otherwise follow the task config.
        """
        if self.provider in ("moon", "codex"):
            return "local"
        if self.provider in ("claude", "openai", "deepseek", "kimi", "local"):
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
            political_line = _load_political_line_section() if self.include_political_line else None
            prompt_overlay = _load_agent_prompt_overlay_section(self.name)
            rate_limit_note = _load_rate_limit_section(self.tools)
            prompt_ir = self.prompt_ir
            extra_sections = [
                section
                for section in (political_line, prompt_overlay, rate_limit_note)
                if section is not None
            ]
            if extra_sections:
                prompt_ir = SystemPrompt(
                    identity=self.prompt_ir.identity,
                    preamble=self.prompt_ir.preamble,
                    sections=[*extra_sections, *self.prompt_ir.sections],
                    context=self.prompt_ir.context,
                )
            template = _render_prompt(prompt_ir, provider)
        else:
            template = self.system_prompt_template or ""
        for key, value in kwargs.items():
            template = template.replace("{" + key + "}", str(value))
        return template

    def filter_tools(
        self, all_tools: list[dict], all_handlers: dict
    ) -> tuple[list[dict], dict]:
        """Filter tools and handlers to only those allowed by this agent.

        Empty tool lists are fail-closed. Agents that need broad access must
        declare each tool explicitly so config mistakes do not silently expose
        the global registry.
        """
        from tool_gateway.selection import filter_agent_tools

        return filter_agent_tools(self, all_tools, all_handlers)
