"""Owner chat runtime: the orchestrator/agent tool loop behind Telegram, delegated
agents, autonomous jobs and the browser worker.

Separated from telegram/bot.py so library code (jobs, runtime_tools, browser,
llm.reflexion) does not import the bot entry point. telegram.bot re-exports
these under their old private names."""

import logging
import time as _time
from datetime import datetime

from agents.base import CHAT_AUDIENCE_BLOCK, load_political_line_body
from bot_config import (
    _claude, _config, deepseek_tool_available, deepseek_tool_loop, _get_model_moon, _get_model_task,
    _get_task_provider, _kimi_client, _openai_client,
)
from identity.prompts import CORE_IDENTITY, EXTERNAL_SOURCE_RULE
from llm.prompt_renderer import SystemPrompt, render as _render_prompt
from llm.provider_failover import run_with_provider_failover
from llm.runtime_context import (
    join_context_blocks as _join_context_blocks,
    merge_runtime_context_into_last_user as _merge_runtime_context_into_last_user,
)
from llm.runtime_profile import resolve_runtime_profile
from llm.skills_loader import build_skills_prompt
from ops.logs import log_event as _log_event
from runtime_tools.allowlists import build_orchestrator_toolset
from runtime_tools.registry import TOOL_HANDLERS, TOOLS
from shared import KST

logger = logging.getLogger(__name__)


_MAX_ALERTS = 5


_ALERT_TTL = 24 * 60 * 60  # 24 hours


_system_alerts: list[tuple[float, str]] = []


def _prune_alerts():
    """Remove expired alerts and trim to max count."""
    now = _time.monotonic()
    _system_alerts[:] = [(t, m) for t, m in _system_alerts if now - t < _ALERT_TTL]
    while len(_system_alerts) > _MAX_ALERTS:
        _system_alerts.pop(0)


def add_system_alert(msg: str):
    """Add a system alert visible to the bot in its system prompt."""
    _system_alerts.append((_time.monotonic(), f"[{datetime.now(KST).strftime('%H:%M')}] {msg}"))
    _prune_alerts()


def clear_system_alert(keyword: str):
    """Remove alerts containing keyword (e.g. when issue resolves)."""
    _system_alerts[:] = [(t, m) for t, m in _system_alerts if keyword not in m]


def _format_system_alerts(provider: str = "claude") -> str:
    """Format recent system alerts as a standalone context block.

    Returned without any surrounding whitespace — `_join_context_blocks` is
    responsible for separator insertion. `provider` selects the structure:
    Claude → `<system-alerts>` XML, others → `### System Alerts` Markdown.
    """
    _prune_alerts()
    if not _system_alerts:
        return ""
    items = "\n".join(f"- {m}" for _, m in _system_alerts)
    if provider == "claude":
        return f"<system-alerts>\n{items}\n</system-alerts>"
    return f"### System Alerts\n{items}"


_CHAT_AUDIENCE_INNER = (
    CHAT_AUDIENCE_BLOCK
    .removeprefix("<chat-audience>").removesuffix("</chat-audience>")
    .strip()
)


_ORCHESTRATOR_PROMPT_IR = SystemPrompt(
    identity=CORE_IDENTITY.rstrip() + "\n\n" + EXTERNAL_SOURCE_RULE,
    preamble=(
        "Operating via Telegram (you are currently talking to the admin "
        "비숑 동지). Use tools proactively when data would improve the "
        "answer — don't rely on memory alone."
    ),
    sections=[
        ("chat-audience", _CHAT_AUDIENCE_INNER),
        ("tool-strategy", """
- Geopolitics → knowledge_graph_search first, then vector_search
- For Korean organizations/publications already known to KG, preserve canonical names; do not invent translations/romanizations. Use `디아마트 (DiaMat)` and `웹진 반란(Uprising)`, not `Diamat` or `Webzine Banlan`.
- For Korean people already known to KG, preserve Korean names; use `신현준`, not `Shin Hyunjoon` / `Shin Hyun-joon`.
- Theory/ideology → vector_search (layer="core_theory")
- Current events → web_search, cross-ref with KG
- URL in message → fetch_url to read the page; for x.com/twitter.com status/profile URLs use fetch_x_post
- Owner asks to curate / register an external link on the hub (큐레이션 등록) → do not delegate and do not attempt it yourself; tell them to send `/curate <url> [메모]`, which runs the hub_curator agent. Corrections to an already published curation still go to analyst.
- Self-reflection → read_self(content_type="diary"); cross-interface memory → read_self(content_type="chat_logs")
- Past lessons/mistakes → recall_experience (semantic search over accumulated daily insights)
- Reusable self-produced analysis → save_self_analysis, then retrieve later with vector_search(layer="self_produced_analysis")
- Store important structured facts → write_kg_structured
- Real-time market prices → get_finance_data
- Telegram channel announcement → broadcast_to_channel(title, summary, url). Use this directly when asked to post to the public channel; summary must be a 2-3 sentence preview and url must be a plain full-text URL.
- Unsure which agent or content store owns the request → route_task(task="...") first. Use list_agent_tools only when you need the detailed tool list/schema.
- Published content corrections → delegate to the content-owning agent, not programmer:
  - Past diary entry → `delegate(agent="diary")`; the diary agent has `edit_content(content_type="diary", ...)`
  - Published research / task report / blog post / hub curation / static page → `delegate(agent="analyst")`; the analyst has `research_document` and `edit_content`
  These are operational content edits with cache invalidation, not code changes. Do not delegate to programmer just to correct wording, titles, metadata, markdown prose, or factual text in existing public content.
- Runtime/pipeline failures → delegate to programmer even if the affected feature is diary or publishing. "Diary writing failed", diary scheduler/context bugs, route_task errors, service errors, and pipeline regressions are code/config work, not diary content edits.
""".strip()),
        ("context-isolation", """
**You are the orchestrator. You have no access to programming tools (read_file, write_file, patch_file, list_directory, execute_python).**
If you need to read/modify/execute code, you must delegate via `delegate(agent="programmer")`.
This code-delegation rule does NOT apply to already-published site content. If the user asks to edit public text/content that is stored in the database or research store, delegate to the content-owning agent (usually diary for diaries, analyst for research/reports/posts/curations), not programmer.
Your role is to understand the user's intent, dispatch tasks to the appropriate agents, and synthesize results.
The `<current_state>` block contains structured completed/in-progress/pending tasks. Use it to avoid duplicate work and determine next steps. Detailed tool execution logs are only accessible to each agent itself.
""".strip()),
        ("delegation", """
CRITICAL RULE: When you decide to delegate, you MUST call the `delegate` or `multi_delegate` tool.

You have specialized agents. Use the `delegate` tool to dispatch tasks:
- programmer: code writing/editing/debugging/file management
- analyst: default agent for information analysis/research. Web search + collection + KG cross-validation + pattern extraction + knowledge storage
- scout: Moltbook and mersoom.com activity (posting/commenting/patrol), routine patrols, large-scale platform crawling
- browser: AI browser automation — login, form input, multi-page navigation, dynamic site data extraction
- visualizer: image generation, visual concepts
- diary: writes scheduled diary entries and maintains published diary entries. For a new diary entry, delegate exactly `[diary] Write a periodic diary entry`; arbitrary diary-agent task text is treated as autonomous maintenance/inspection and cannot call `save_diary`.
Stasova is not a general delegation target. It is reserved for internal publication-review flows and is intentionally absent from delegate/multi_delegate.

Content-type routing:
- Diary entry → diary for writing/editing/deleting/unpublishing; use diary id when supplied. For new diary writing, use the exact scheduled prompt `[diary] Write a periodic diary entry`.
- Task report / completed Telegram report → analyst with edit_content(content_type="task_report"); use task_id when supplied
- Public research document → analyst with research_document; use research slug when supplied
- Private research document → analyst with research_document(action="save_private"|"publish_private"); read it with read_self(content_type="private_research_document")
- Hub curation → analyst with edit_content(content_type="hub_curation"); use curation slug when supplied
- Static/custom HTML page → analyst with edit_content(content_type="static_page"); use static page slug when supplied
Public URLs are only clues for inferring content type and identifier; do not make agents reason from routes when the content type is already known.

Parallel delegation with `multi_delegate`:
- Compound requests (e.g., "investigate X and fix Y's code") should be handled in parallel via multi_delegate.
- After all subtasks complete, a synthesis task automatically consolidates results.
- Specifying synthesis_instructions with consolidation criteria yields better results.

Context passing — agents automatically receive recent conversation and their own execution history, but specify the current conversation's key context in the `context` field:
1. The user's original request (verbatim or key summary)
2. Findings from the conversation so far (tool results, analysis, decisions)
3. Why you are delegating to this agent (reason and expected outcome)
4. The correct target identifier when the user supplied one: public URL, slug, post_id, DB document identifier, error text, command output, or visible symptom. Do not invent or pass filesystem paths; delegated agents that need code context can inspect the repository themselves.

Pass concrete acceptance criteria: required deliverables, target IDs, constraints, and the evidence that will demonstrate completion. Distinguish existing authorization from new requested actions.
Delegation discipline: delegate what must be achieved, not how. Do not invent unverified implementation details; let workers inspect and choose the implementation.

Do not delegate routine public-content edits to programmer. For requests like "fix this published post", "correct a diary/report/blog typo", "revise this curation", or "edit an already-published research page", delegate to the agent that owns the content/editor tool: diary for diary entries; analyst for research documents, task reports, blog posts, and curations. Delegate to programmer only when the required change is source code, configuration, scripts, templates, frontend behavior, deployment, or debugging.
If the user explicitly says to send a failure, scheduler, routing, or pipeline problem to programmer, obey that instruction; do not reinterpret it as a diary content task just because the word "diary" appears.
""".strip()),
        ("mission-management", """
- Missions are auto-created when delegate is called. The user does not need to create them explicitly.
- The `<active-mission>` block (when present) shows the current mission's title and event timeline. Read it before every turn to decide whether the user's new message still belongs to that mission.
- **Call `mission(action="close")` when ANY of these hold:**
  - The user's original goal for the active mission has been fully achieved.
  - **Topic drift**: the user's current message is clearly on a different topic from the active mission's title/timeline. A topic switch implicitly abandons the old mission — close it so the next `delegate` call opens a fresh mission aligned with the new topic. Do this even when the prior mission had in-progress or budget-interrupted tasks; stale missions pollute future context.
- **Do NOT close solely because** task results mention "budget exhausted" / "limit reached" / "stopped due to error" — if the user is still pursuing that same topic, leave the mission open and delegate follow-up work.
- When you close a mission because of topic drift, you don't need to manually create a replacement — just proceed with the new topic; if a delegate is needed, a fresh mission will be auto-created from it.
""".strip()),
        ("temporal-awareness", """
Conversation history includes timestamps ([YYYY-MM-DD HH:MM]) on user messages.
Infer elapsed time from the timestamps. Large gaps may indicate context switches or changed circumstances.
""".strip()),
        ("response-rules", """
- Dialectical materialist lens for geopolitics. Concise, substantive. Cite sources. Match user's language.
- Do not use markdown formatting (**, *, #, ```, - etc.) in Telegram messages. Write in plain text only, as a human would. Markdown is allowed only when composing markdown documents or code artifacts through the appropriate specialist/tool.
""".strip()),
    ],
)


_CLAUDE_STATIC_TAIL = "{skills_section}"


_MARKDOWN_STATIC_TAIL = "{skills_section}"


def _build_orchestrator_system_prompt(provider: str) -> str:
    """Render the orchestrator's static system prompt for `provider`.

    Returned string keeps only the ``{skills_section}`` placeholder — the only
    stable runtime data in the system layer. Per-turn state (time, model,
    mission, memories, alerts) is injected as message content so the system
    prompt stays byte-identical across turns and benefits from prompt caching.
    """
    prompt_ir = _ORCHESTRATOR_PROMPT_IR
    political_line = load_political_line_body()
    if political_line:
        prompt_ir = SystemPrompt(
            identity=_ORCHESTRATOR_PROMPT_IR.identity,
            preamble=_ORCHESTRATOR_PROMPT_IR.preamble,
            sections=[("political-line", political_line), *_ORCHESTRATOR_PROMPT_IR.sections],
            context=_ORCHESTRATOR_PROMPT_IR.context,
        )
    body = _render_prompt(prompt_ir, provider)
    tail = _CLAUDE_STATIC_TAIL if provider == "claude" else _MARKDOWN_STATIC_TAIL
    return body + "\n\n" + tail


def _owner_run_context(
    interface: str, agent_name: str | None, *, request_id=None, user_id=None,
    task_id=None, session_id=None, parent_request_id=None, scope_type=None,
    scope_id=None,
):
    """Build the security-gateway run context for an owner-trusted Telegram call.

    Task-bound calls default to the ``telegram_task`` scope keyed by task id
    when the caller did not name a scope explicitly.
    """
    from tool_gateway.security import new_run_context
    ctx_kwargs = {
        "interface": interface,
        "agent_name": agent_name,
        "is_owner": True,
        "request_id": request_id,
    }
    if user_id is not None:
        ctx_kwargs["user_id"] = str(user_id)
    if task_id is not None:
        ctx_kwargs["task_id"] = str(task_id)
    if session_id is not None:
        ctx_kwargs["session_id"] = session_id
    if parent_request_id is not None:
        ctx_kwargs["parent_request_id"] = parent_request_id
    if scope_type is not None:
        ctx_kwargs["scope_type"] = scope_type
    elif task_id is not None:
        ctx_kwargs["scope_type"] = "telegram_task"
    if scope_id is not None or task_id is not None:
        ctx_kwargs["scope_id"] = str(scope_id if scope_id is not None else task_id)
    return new_run_context(**ctx_kwargs)


async def _resolve_chat_runtime(
    *,
    system_prompt: str | None,
    runtime_kind: str | None,
    provider_override: str | None,
    model: str | None,
    budget_usd: float | None,
    max_rounds: int | None,
    max_tokens: int | None,
    max_input_tokens: int | None,
    max_output_continuations: int,
    thinking_policy: str,
    thinking_budget_tokens: int,
):
    """Render the system prompt and resolve the runtime profile and limits.

    Returns (sys_prompt, runtime_kind, profile, max_input_tokens,
    output_continuations, inference_policy).
    """
    # Resolve provider up-front so the system prompt can be rendered in the
    # format native to the target model family (XML for Claude, Markdown for
    # OpenAI/Qwen). provider_override wins over the stored config.
    effective_provider = provider_override or _config.get("provider", "claude")

    # System prompt: orchestrator builds its own fully static prompt; agents
    # pass their pre-rendered (also static, post-refactor) spec prompt.
    if system_prompt is None:
        sys_prompt = _build_orchestrator_system_prompt(effective_provider).format(
            skills_section=build_skills_prompt(),
        )
        _runtime_kind = runtime_kind or "chat"
    else:
        sys_prompt = system_prompt
        _runtime_kind = runtime_kind or "task"
    if _runtime_kind not in ("chat", "task", "autonomous"):
        logger.warning("Unknown runtime_kind=%r; falling back to task", _runtime_kind)
        _runtime_kind = "task"
    profile = await resolve_runtime_profile(
        _runtime_kind,
        provider_override=effective_provider,
        model_override=model,
        budget_override=budget_usd,
        max_rounds_override=max_rounds,
        max_tokens_override=max_tokens,
    )
    from tool_gateway.inference import DEFAULT_AGENT_MAX_INPUT_TOKENS
    resolved_max_input_tokens = int(max_input_tokens or DEFAULT_AGENT_MAX_INPUT_TOKENS)
    resolved_output_continuations = max(0, int(max_output_continuations or 0))
    from tool_gateway.inference import AgentInferencePolicy
    call_inference_policy = AgentInferencePolicy(
        max_input_tokens=resolved_max_input_tokens,
        max_output_tokens=profile.max_tokens,
        max_rounds=profile.max_rounds,
        budget_usd=profile.budget_usd,
        max_output_continuations=resolved_output_continuations,
        thinking_policy=thinking_policy,
        thinking_budget_tokens=thinking_budget_tokens,
    )
    return (
        sys_prompt, _runtime_kind, profile,
        resolved_max_input_tokens, resolved_output_continuations, call_inference_policy,
    )


def _attach_chat_runtime_context(
    messages: list[dict],
    *,
    extra_system_context: str,
    profile,
    runtime_kind: str,
    agent_name: str | None,
    user_id,
    task_id: int | None,
    session_id: str | None,
) -> list[dict]:
    """Merge volatile runtime context into the trailing user message."""
    # Runtime context injection (applies to orchestrator AND agents). Volatile
    # data — current time, current model, caller-supplied extras (mission,
    # experiences, state), and system alerts — rides on the trailing user
    # message so the system prompt and history prefix stay byte-stable across
    # turns for prompt caching. The active-call record supplies the resolved
    # model and current time once; do not also inject the default model route.
    full_runtime_context = _join_context_blocks(
        extra_system_context or "",
        _format_system_alerts(profile.provider),
    )
    messages = _merge_runtime_context_into_last_user(messages, full_runtime_context)
    from llm.execution_context import attach_context, context_record
    return attach_context(messages, [context_record(
        "active_call", "resolved_runtime_profile", {
            "provider": profile.provider, "model": profile.model_id,
            "runtime_kind": runtime_kind, "agent": agent_name,
            "owner_user_id": user_id, "task_id": task_id,
        }, scope=session_id or (f"task:{task_id}" if task_id is not None else "unknown"),
        observed_at=datetime.now(KST), temporal_scope="current turn",
    )])


def _build_chat_toolset(
    extra_tools: list | None, extra_handlers: dict | None,
) -> tuple[bool, list, dict]:
    """Return (is_orchestrator, tools, handlers) for one chat_with_tools call."""
    is_orchestrator = extra_tools is None

    if is_orchestrator:
        merged_tools, merged_handlers = build_orchestrator_toolset(
            TOOLS,
            TOOL_HANDLERS,
            extra_handlers,
        )
    else:
        # Task/agent: use ONLY extra_tools (already filtered by agent spec).
        # Do NOT merge full TOOLS — that would bypass agent tool restrictions.
        merged_tools = list(extra_tools or [])
        merged_handlers = dict(extra_handlers or {})

    if is_orchestrator and "list_agent_tools" in {t.get("name") for t in merged_tools}:
        from self_runtime.tools import build_list_agent_tools_handler
        merged_handlers["list_agent_tools"] = build_list_agent_tools_handler(merged_tools)

    # Inject run_agent handler (needs chat_with_tools closure — can't be registered at import time)
    if is_orchestrator and "run_agent" not in merged_handlers:
        from self_runtime.tools import build_run_agent_handler
        merged_handlers["run_agent"] = build_run_agent_handler(chat_with_tools)
    return is_orchestrator, merged_tools, merged_handlers


def _resolve_chat_provenance(
    task_id: int | None, agent_name: str | None, is_orchestrator: bool,
) -> tuple[str, int | None, object]:
    """Return (agent_name, mission_id, task_user_id) for provenance tracking."""
    # Resolve agent name + mission for provenance tracking
    _agent_name = agent_name or ("orchestrator" if is_orchestrator else "agent")
    _mission_id: int | None = None
    _task_user_id = None
    if task_id is not None:
        try:
            from db import query as _db_q
            row = _db_q(
                "SELECT agent_type, mission_id, user_id, parent_task_id "
                "FROM telegram_tasks WHERE id = %s",
                (task_id,),
            )
            if row:
                if agent_name is None and not is_orchestrator:
                    _agent_name = str(row[0].get("agent_type") or "agent")
                _mission_id = row[0].get("mission_id")
                _task_user_id = row[0].get("user_id")
        except Exception:
            pass
    return _agent_name, _mission_id, _task_user_id


async def _dispatch_chat_provider(
    messages: list[dict],
    *,
    effective_provider: str,
    profile,
    loop_kwargs: dict,
    call_inference_policy,
    is_orchestrator: bool,
    runtime_kind: str,
    deepseek_thinking_override: dict | None,
    budget_tracker: dict | None,
    on_progress,
    gw_ctx,
) -> str:
    """Run the tool loop on the resolved provider inside the gateway scope."""
    from tool_gateway.security import caller_scope
    from tool_gateway.inference import resolve_inference_extra
    resolved_max_tokens = loop_kwargs["max_tokens"]

    # ── Provider dispatch: Claude vs OpenAI vs Local ──
    # effective_provider is the resolved runtime profile's provider.
    if effective_provider == "local":
        from llm.openai_tool_loop import chat_with_tools as openai_chat
        from llm.client import (
            _resolve_backend, LOCAL_SEMAPHORE, LOCAL_CONTEXT_LIMIT,
            LOCAL_MAX_TOKENS, LOCAL_ENABLE_THINKING,
        )
        backend = _resolve_backend()
        # Floor the completion budget at LOCAL_MAX_TOKENS (default 8192).
        # The 4096 default shared with Claude truncates Qwen3 responses
        # mid-<think> on Q4 quantizations, so the tool_call is never
        # emitted and the loop returns an empty answer.
        _chat_coro = openai_chat(
            messages,
            client=None,
            base_url=backend["base"],
            model=profile.model_id or backend["model"],
            **{**loop_kwargs, "max_tokens": max(resolved_max_tokens, LOCAL_MAX_TOKENS)},
            context_limit=LOCAL_CONTEXT_LIMIT,
            enable_thinking=is_orchestrator and LOCAL_ENABLE_THINKING,
            api_semaphore=LOCAL_SEMAPHORE,
            provider_label=f"local:{backend['base']}",
        )
        with caller_scope(gw_ctx):
            return await _chat_coro

    if effective_provider == "openai" and _openai_client:
        from llm.openai_tool_loop import chat_with_tools as openai_chat
        openai_inference = resolve_inference_extra(call_inference_policy, "openai")
        _chat_coro = openai_chat(
            messages,
            client=_openai_client,
            model=profile.model_id,
            **loop_kwargs,
            provider_label="openai",
            extra_body=openai_inference.get("extra_body"),
        )
        with caller_scope(gw_ctx):
            return await _chat_coro

    if effective_provider == "kimi" and _kimi_client:
        from llm.openai_tool_loop import chat_with_tools as openai_chat
        from llm.provider_registry import kimi_openai_tool_options
        _chat_coro = openai_chat(
            messages,
            client=_kimi_client,
            model=profile.model_id,
            **loop_kwargs,
            provider_label="kimi",
            **kimi_openai_tool_options(),
        )
        with caller_scope(gw_ctx):
            return await _chat_coro

    if effective_provider == "deepseek" and deepseek_tool_available():
        deepseek_thinking = deepseek_thinking_override or resolve_inference_extra(
            call_inference_policy, "deepseek"
        )
        deepseek_chat, deepseek_client, deepseek_options = deepseek_tool_loop(
            deepseek_thinking, label=f"deepseek:{runtime_kind}")

        def _deepseek_primary():
            return deepseek_chat(
                messages,
                client=deepseek_client,
                model=profile.model_id,
                **loop_kwargs,
                **deepseek_options,
            )

        from llm.provider_failover import resolve_deepseek_failover_model
        failover_model = await resolve_deepseek_failover_model(runtime_kind, _openai_client)

        def _terra_failover():
            from llm.openai_tool_loop import chat_with_tools as openai_chat
            return openai_chat(
                messages,
                client=_openai_client,
                model=failover_model,
                **loop_kwargs,
                provider_label="openai:failover",
            )

        with caller_scope(gw_ctx):
            return await run_with_provider_failover(
                _deepseek_primary,
                _terra_failover if failover_model else None,
                primary_label="deepseek",
                fallback_label=failover_model or "openai",
                budget_tracker=budget_tracker,
                on_progress=on_progress,
            )

    if effective_provider in ("openai", "deepseek", "kimi"):
        missing = {
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "kimi": "MOONSHOT_API_KEY",
        }[effective_provider]
        raise RuntimeError(f"{missing} is not configured for provider={effective_provider}")

    # Claude path. `messages` has already had the runtime context merged into
    # the trailing user turn by `_attach_chat_runtime_context`,
    # so history remains byte-stable across turns and prefix caching works.
    claude_inference = resolve_inference_extra(call_inference_policy, "claude")
    _chat_coro = chat_with_tools(
        messages,
        client=_claude,
        model=profile.model_id,
        **loop_kwargs,
        thinking=claude_inference.get("thinking"),
    )
    with caller_scope(gw_ctx):
        return await _chat_coro


async def chat_with_tools(
    messages: list[dict],
    max_rounds: int | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    max_input_tokens: int | None = None,
    max_output_continuations: int = 0,
    thinking_policy: str = "tool_loop",
    thinking_budget_tokens: int = 8192,
    budget_usd: float | None = None,
    extra_tools: list | None = None,
    extra_handlers: dict | None = None,
    on_progress=None,
    budget_tracker: dict | None = None,
    task_id: int | None = None,
    provider_override: str | None = None,
    finalization_tools: list[str] | None = None,
    terminal_tools: list[str] | None = None,
    terminal_required: bool = False,
    extra_system_context: str = "",
    agent_name: str | None = None,
    runtime_kind: str | None = None,
    deepseek_thinking_override: dict | None = None,
    user_id: str | int | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    parent_request_id: str | None = None,
    scope_type: str | None = None,
    scope_id: str | int | None = None,
) -> str:
    """Call LLM with tools — dispatches to Claude or OpenAI based on provider config.

    provider_override: if set, forces this provider instead of _config["provider"].
    Used by web_chat, diary writer, etc. to always use corporate LLM.
    extra_system_context: appended to the rendered orchestrator system prompt
    (active mission timeline, task state block, retrieved experiences, etc.).
    Ignored when `system_prompt` is passed directly.
    deepseek_thinking_override: per-call replacement for the global DeepSeek
    thinking params (same shape as _get_deepseek_thinking_params()). Short
    format-constrained calls (autonomous tick planner/critic) pass
    {"thinking": {"type": "disabled"}} — reasoning-mode burn was exhausting
    their whole max_tokens before any visible reply. Ignored by non-DeepSeek
    providers.
    """
    (
        sys_prompt, _runtime_kind, profile,
        resolved_max_input_tokens, resolved_output_continuations, call_inference_policy,
    ) = await _resolve_chat_runtime(
        system_prompt=system_prompt,
        runtime_kind=runtime_kind,
        provider_override=provider_override,
        model=model,
        budget_usd=budget_usd,
        max_rounds=max_rounds,
        max_tokens=max_tokens,
        max_input_tokens=max_input_tokens,
        max_output_continuations=max_output_continuations,
        thinking_policy=thinking_policy,
        thinking_budget_tokens=thinking_budget_tokens,
    )
    effective_provider = profile.provider
    resolved_max_rounds = profile.max_rounds
    resolved_max_tokens = profile.max_tokens
    resolved_budget = profile.budget_usd

    messages = _attach_chat_runtime_context(
        messages,
        extra_system_context=extra_system_context,
        profile=profile,
        runtime_kind=_runtime_kind,
        agent_name=agent_name,
        user_id=user_id,
        task_id=task_id,
        session_id=session_id,
    )
    is_orchestrator, merged_tools, merged_handlers = _build_chat_toolset(
        extra_tools, extra_handlers,
    )
    _agent_name, _mission_id, _task_user_id = _resolve_chat_provenance(
        task_id, agent_name, is_orchestrator,
    )

    # ── Security gateway caller context ──
    # Telegram is the owner's gated control channel (ALLOWED_USER_IDS upstream),
    # so calls here are trusted. Orchestrator vs delegated-agent is distinguished
    # for audit attribution; caller_scope restores the parent on exit so a nested
    # run_agent sub-call doesn't leak its agent identity back to the orchestrator.
    _interface = (
        "autonomous" if _runtime_kind == "autonomous"
        else ("telegram" if is_orchestrator else "agent")
    )
    _gw_ctx = _owner_run_context(
        _interface,
        None if is_orchestrator and _interface == "telegram" else _agent_name,
        request_id=request_id,
        user_id=user_id if user_id is not None else _task_user_id,
        task_id=task_id, session_id=session_id,
        parent_request_id=parent_request_id,
        scope_type=scope_type, scope_id=scope_id,
    )

    # Kwargs shared verbatim by every loop call below — the per-provider
    # branches add only client/model and provider-specific extras.
    loop_kwargs = dict(
        tools=merged_tools,
        tool_handlers=merged_handlers,
        system_prompt=sys_prompt,
        max_rounds=resolved_max_rounds,
        max_tokens=resolved_max_tokens,
        max_input_tokens=resolved_max_input_tokens,
        recover_input_via_tools=True,
        continue_on_length=resolved_output_continuations > 0,
        max_length_continuations=resolved_output_continuations,
        log_event=_log_event,
        budget_usd=resolved_budget,
        on_progress=on_progress,
        budget_tracker=budget_tracker,
        task_id=task_id,
        agent_name=_agent_name,
        mission_id=_mission_id,
        finalization_tools=finalization_tools,
        terminal_tools=terminal_tools,
        terminal_required=terminal_required,
    )

    return await _dispatch_chat_provider(
        messages,
        effective_provider=effective_provider,
        profile=profile,
        loop_kwargs=loop_kwargs,
        call_inference_policy=call_inference_policy,
        is_orchestrator=is_orchestrator,
        runtime_kind=_runtime_kind,
        deepseek_thinking_override=deepseek_thinking_override,
        budget_tracker=budget_tracker,
        on_progress=on_progress,
        gw_ctx=_gw_ctx,
    )


async def get_model_for_agent(spec):
    """Resolve the LLM model an agent should run against.

    Priority: spec.provider override wins over task_provider, which can be
    independent from the Telegram chat provider. Returns the API model ID.
    """
    if spec.provider == "moon":
        return await _get_model_moon()
    if spec.provider == "codex":
        from llm.codex_exec_loop import CODEX_DEFAULT_MODEL
        return spec.model or CODEX_DEFAULT_MODEL
    provider = spec.provider or _get_task_provider()
    if provider == "local":
        if spec.model:
            return spec.model
        from llm.client import _resolve_backend
        return _resolve_backend()["model"]
    if provider in ("claude", "openai", "deepseek", "kimi"):
        profile = await resolve_runtime_profile(
            "task",
            provider_override=provider,
            tier_override=spec.model,
        )
        return profile.model_id
    return await _get_model_task()


def make_provider_chat_fn(provider: str):
    """Build a chat_fn that forces chat_with_tools to use a specific provider.

    Used when an agent spec pins a provider or task_provider differs from the
    Telegram chat provider.
    """
    async def _provider_chat_fn(
        messages, max_rounds=None, system_prompt=None, model=None,
        max_tokens=None, max_input_tokens=None, max_output_continuations=0,
        thinking_policy="tool_loop", thinking_budget_tokens=8192, budget_usd=None, extra_tools=None,
        extra_handlers=None, on_progress=None, budget_tracker=None,
        task_id=None, finalization_tools=None, terminal_tools=None,
        terminal_required=False,
        agent_name=None,
        runtime_kind=None,
        user_id=None, session_id=None, request_id=None, parent_request_id=None,
        scope_type=None, scope_id=None,
    ):
        return await chat_with_tools(
            messages, max_rounds=max_rounds, system_prompt=system_prompt,
            model=model, max_tokens=max_tokens, max_input_tokens=max_input_tokens,
            max_output_continuations=max_output_continuations,
            thinking_policy=thinking_policy,
            thinking_budget_tokens=thinking_budget_tokens, budget_usd=budget_usd,
            extra_tools=extra_tools, extra_handlers=extra_handlers,
            on_progress=on_progress, budget_tracker=budget_tracker,
            task_id=task_id, provider_override=provider,
            finalization_tools=finalization_tools,
            terminal_tools=terminal_tools,
            terminal_required=terminal_required,
            agent_name=agent_name,
            runtime_kind=runtime_kind,
            user_id=user_id, session_id=session_id, request_id=request_id,
            parent_request_id=parent_request_id,
            scope_type=scope_type, scope_id=scope_id,
        )
    return _provider_chat_fn
