"""agent_loop.py — Unified agent tool-use loop engine.

The round-loop and forced-final control flow shared by claude_loop.py
(Anthropic protocol: Claude, DeepSeek) and openai_tool_loop.py (OpenAI
protocol: OpenAI, Kimi, local llama-server) lives here, once. Provider
mechanics — message shapes, API calls/streaming/retry, cost math, response
parsing, protocol recovery — stay in the two protocol modules and are
exposed to the engine through a protocol-adapter object.

History: the two loops evolved in parallel from telegram_bot.py, and every
control-flow fix (budget warnings, terminal tools, finalization tools,
followup-skip, length continuation, stream idle guards …) had to be mirrored
by hand across both files. tests/test_claude_loop_rounds.py and
tests/test_openai_loop_rounds.py pin the public contracts this engine
preserves.

Adapter contract (duck-typed; see _ClaudeProtocolAdapter and
_OpenAIProtocolAdapter for the two implementations):

  bind(state)                      — receive the shared LoopState
  normalize(messages)              — inbound history → provider wire format
  call_model(msgs, round_num)      — one API call incl. retry/recovery; may
                                     mutate msgs in place (recovery); returns
                                     the provider response, or None to abort
                                     straight to the forced-final phase
  track_cost(response, label)      — add cost to state, log usage; True if
                                     usage was present (engine then emits
                                     budget progress + Redis state)
  parse_turn(response, round_num)  — response → Turn; may raise LoopEarlyReturn
  append_assistant(msgs, turn)
  run_batch(batch, round_num)      — execute tool calls via the module's
                                     execute_tools_batch (patchable in tests)
  note_exec_results(round_num, exec_results)
  append_tool_results(msgs, turn, exec_results, missing, warning_texts)
                                   — returns True when warning_texts were
                                     actually delivered to the model
  append_length_continuation(msgs, turn, partial, next_index, max_count)
                                   — async; True to continue the loop
  on_truncated_final()             — async; hook for a truncated final reply
  was_still_working(response)      — bool for the forced-final report
  build_finalization_tools(names)  — (allowed_names, tools_payload_or_None)
  append_user_text(msgs, text)
  call_final(msgs, tools, limit_reason) / parse_final(response, names)
  append_final_assistant(msgs, final_turn)
  append_final_results(msgs, final_turn, exec_results)
  call_followup(msgs) / extract_text_parts(response)
  recover_final_failure(msgs, err, limit_reason, was_still_working)
                                   — async; return replacement text parts,
                                     raise LoopEarlyReturn, or re-raise
  update_tracker(tracker, *, rounds_used, was_interrupted, tool_work_details,
                 response)
  make_result(parts, **meta)       — final text (or metadata dict)
"""

import json
import logging

from llm.gateway import check_llm_call, record_llm_call
from tool_loop_common import (
    build_budget_warning,
    build_limit_message,
    build_round_warning,
    check_cancelled,
    emit_progress,
    save_redis_progress,
    update_redis_state,
    validate_budget,
)

logger = logging.getLogger(__name__)

# Skip the followup roundtrip when the forced-final call already produced
# this much closing text alongside its finalization tool calls (see the
# followup-skip rationale in the engine body).
FOLLOWUP_SKIP_MIN_CHARS = 200


class LoopEarlyReturn(Exception):
    """Raised by a protocol adapter to end the loop with a ready answer.

    Covers provider outcomes that terminate the turn outside the normal
    text-response path: refusals, API failures after side-effect tools have
    already run, fully-malformed tool-call turns, and forced-final responses
    that failed even after protocol stripping.
    """

    def __init__(self, text, *, finish_reason="stop", truncated=False,
                 limit_reason=None, was_still_working=False, interrupted=False):
        super().__init__(text)
        self.text = text
        self.finish_reason = finish_reason
        self.truncated = truncated
        self.limit_reason = limit_reason
        self.was_still_working = was_still_working
        self.interrupted = interrupted


class Turn:
    """Normalized view of one model response inside the round loop."""

    def __init__(self, *, is_tool_round, text_parts=None, tool_calls=None,
                 malformed=None, truncated_by_length=False, finish_reason="stop",
                 raw=None, extra=None):
        self.is_tool_round = is_tool_round
        self.text_parts = list(text_parts or [])
        self.tool_calls = list(tool_calls or [])   # [(id, name, input_dict)]
        self.malformed = list(malformed or [])     # [(id, name, error_text)]
        self.truncated_by_length = truncated_by_length
        self.finish_reason = finish_reason
        self.raw = raw
        self.extra = extra or {}


class FinalTurn:
    """Normalized view of the forced-final response."""

    def __init__(self, *, text_parts=None, batch=None, has_protocol=False,
                 raw=None, extra=None):
        self.text_parts = list(text_parts or [])
        self.batch = list(batch or [])             # [(id, name, args)] whitelisted
        self.has_protocol = has_protocol           # tool protocol must be appended
        self.raw = raw
        self.extra = extra or {}


class LoopState:
    """Cost state shared between the engine and its adapter.

    add_cost doubles as the LLM-gateway seam for loop calls: both protocol
    adapters already report every model round's cost here, so recording the
    audit event in this one place covers every provider without mirroring.
    """

    def __init__(self, budget_usd: float, *, agent_name: str = "agent"):
        self.budget_usd = budget_usd
        self.total_cost = 0.0
        self.agent_name = agent_name

    def add_cost(self, cost: float, *, model: str | None = None, label: str | None = None,
                 tokens_in: int = 0, tokens_out: int = 0,
                 cache_read: int = 0, cache_create: int = 0):
        self.total_cost += cost
        record_llm_call(
            surface="loop", caller=self.agent_name, model=model, label=label,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cache_read=cache_read, cache_create=cache_create,
            cost_usd=cost,
        )


async def run_tool_loop(
    adapter,
    messages: list[dict],
    *,
    max_rounds: int = 50,
    max_tokens: int = 4096,
    budget_usd: float = 0.30,
    budget_tracker: dict | None = None,
    on_progress=None,
    task_id: int | None = None,
    log_event=None,
    agent_name: str = "agent",
    mission_id: int | None = None,
    finalization_tools: list[str] | None = None,
    terminal_tools: list[str] | None = None,
    continue_on_length: bool = False,
    max_length_continuations: int = 1,
):
    """Drive the tool-use loop for one agent turn through a protocol adapter."""
    budget_usd = validate_budget(budget_usd)

    # LLM-gateway policy gate: one check per agent turn (kill switch,
    # blocked provider/model, daily budget). Raises only in enforce mode.
    check_llm_call(
        surface="loop", caller=agent_name,
        model=getattr(adapter, "model", None),
    )

    # Per-agent-run provenance buffer for KG write/read trust tracking.
    from provenance.runtime import init_provenance_buffer
    init_provenance_buffer(agent=agent_name, mission_id=mission_id)

    state = LoopState(budget_usd, agent_name=agent_name)
    adapter.bind(state)

    working_msgs = adapter.normalize(messages)
    tool_call_log: list[str] = []
    tool_work_details: list[str] = []
    accumulated_text_parts: list[str] = []  # text collected from tool rounds
    budget_warning_sent = False
    length_continuations = 0
    round_num = 0
    response = None

    # A max-token continuation needs a fresh model round even when the
    # truncation happened on the final normal tool round. Reserve those rounds
    # up front; they are reachable only after an actual max_tokens stop.
    continuation_limit = max(0, int(max_length_continuations or 0))
    continuation_rounds = continuation_limit if continue_on_length else 0
    max_total_round_limit = max_rounds + continuation_rounds
    total_round_limit = max_rounds

    def _tracked(parts, *, finish_reason="stop", truncated=False,
                 limit_reason=None, was_still_working=False, interrupted=False,
                 response_obj=None):
        adapter.update_tracker(
            budget_tracker,
            rounds_used=round_num,
            was_interrupted=interrupted,
            tool_work_details=tool_work_details,
            response=response_obj,
        )
        return adapter.make_result(
            parts,
            finish_reason=finish_reason,
            truncated=truncated,
            limit_reason=limit_reason,
            was_still_working=was_still_working,
            continuations_used=length_continuations,
            rounds_used=round_num,
        )

    try:
        while round_num < total_round_limit:
            round_num += 1
            check_cancelled(task_id)

            response = await adapter.call_model(working_msgs, round_num)
            if response is None:
                # Unrecoverable in-round failure; the adapter has prepared
                # working_msgs (e.g. stripped protocol) for the forced final.
                break

            if adapter.track_cost(response, f"Round {round_num}"):
                await emit_progress(
                    on_progress, "budget",
                    f"[{round_num}] ${state.total_cost:.3f}/${budget_usd:.2f}",
                )
                update_redis_state(task_id, round_num, state.total_cost)

            turn = adapter.parse_turn(response, round_num)

            # ── Final (non-tool) response ──
            if not turn.is_tool_round:
                if turn.truncated_by_length:
                    logger.warning(
                        "Response truncated by max_tokens (%d) at round %d/%d",
                        max_tokens, round_num, max_rounds,
                    )
                    if log_event:
                        log_event(
                            "warning", "chat",
                            f"Response truncated by max_tokens ({max_tokens}) "
                            f"at round {round_num}/{max_rounds}",
                        )
                    if continue_on_length and length_continuations < continuation_limit:
                        partial_text = "\n".join(
                            t.strip() for t in turn.text_parts if t.strip()
                        ).strip()
                        if await adapter.append_length_continuation(
                            working_msgs, turn, partial_text,
                            length_continuations + 1, continuation_limit,
                        ):
                            length_continuations += 1
                            total_round_limit = min(
                                max_total_round_limit, total_round_limit + 1
                            )
                            if partial_text:
                                accumulated_text_parts.append(partial_text)
                            continue
                if turn.truncated_by_length:
                    await adapter.on_truncated_final()
                return _tracked(
                    accumulated_text_parts + turn.text_parts,
                    finish_reason=turn.finish_reason,
                    truncated=turn.truncated_by_length,
                    response_obj=response,
                )

            # ── Tool round ──
            # Budget exceeded → process this response's tool calls, then break.
            budget_exceeded = state.total_cost >= budget_usd
            if budget_exceeded:
                logger.warning(
                    "Budget exhausted: $%.4f >= $%.2f at round %d — "
                    "processing final tool calls before exit",
                    state.total_cost, budget_usd, round_num,
                )

            for text in turn.text_parts:
                stripped = text.strip()
                # Accumulate substantial round text for the final result.
                if stripped and len(stripped) > 20:
                    accumulated_text_parts.append(stripped)
                if stripped:
                    await emit_progress(on_progress, "thinking", f"[{round_num}] {stripped}")

            adapter.append_assistant(working_msgs, turn)

            for _tid, tname, error_text in turn.malformed:
                tool_work_details.append(f"  [{round_num}] {tname}(malformed) → {error_text}")

            exec_results = []
            if turn.tool_calls:
                exec_results = await adapter.run_batch(turn.tool_calls, round_num)
                for tid, tname, tinput, result, is_error in exec_results:
                    input_summary = json.dumps(tinput, ensure_ascii=False)
                    tool_call_log.append(f"  [{round_num}/{max_rounds}] {tname}({input_summary})")
                    tool_work_details.append(f"  [{round_num}] {tname}({input_summary}) → {result}")
                    save_redis_progress(task_id, round_num, tname, input_summary, result, is_error)
                adapter.note_exec_results(round_num, exec_results)

            # Safety net: every tool call must get a result (adapter appends a
            # synthetic error in its own protocol shape for the missing ones).
            resolved_ids = {r[0] for r in exec_results}
            resolved_ids.update(tid for tid, _name, _err in turn.malformed)
            missing = [
                (tid, tname) for tid, tname, _tinput in turn.tool_calls
                if tid not in resolved_ids
            ]

            warning_texts = []
            budget_warning_due = (
                not budget_warning_sent and state.total_cost > budget_usd * 0.8
            )
            if budget_warning_due:
                warning_texts.append(build_budget_warning(state.total_cost, budget_usd))
            if round_num == total_round_limit - 2:
                warning_texts.append(build_round_warning(round_num, max_rounds))

            warnings_delivered = adapter.append_tool_results(
                working_msgs, turn, exec_results, missing, warning_texts,
            )
            if budget_warning_due and warnings_delivered:
                budget_warning_sent = True

            # Terminal-tool short-circuit: if any spec-declared terminal tool
            # succeeded in this round, end the loop without a trailing
            # assistant text turn. The tool's return value becomes the report.
            if terminal_tools and exec_results:
                terminal_hit = next(
                    (
                        (tname, result)
                        for _tid, tname, _tinput, result, is_error in exec_results
                        if tname in terminal_tools and not is_error
                    ),
                    None,
                )
                if terminal_hit:
                    _tname, _tresult = terminal_hit
                    terminal_report = str(_tresult).strip() or f"{_tname} completed"
                    return _tracked(
                        [p for p in accumulated_text_parts + [terminal_report] if p],
                        finish_reason="terminal_tool",
                        response_obj=response,
                    )

            # Budget break AFTER tool results are properly appended.
            if budget_exceeded:
                break

        # ══════════════════════════════════════════════════════════════
        # Limit reached (rounds or budget) — force final response
        # ══════════════════════════════════════════════════════════════
        budget_exhausted = state.total_cost >= budget_usd
        was_still_working = adapter.was_still_working(response) if response else False
        log_detail = "\n".join(tool_call_log) if tool_call_log else ""
        logger.warning(
            "Limit reached (rounds=%d/%d, normal_rounds=%d, budget=$%.4f/$%.2f, "
            "still_working=%s). Forcing final response. Calls:\n%s",
            round_num, total_round_limit, max_rounds, state.total_cost, budget_usd,
            was_still_working, log_detail,
        )
        limit_reason = "예산 소진" if budget_exhausted else "도구 호출 한도 도달"

        # Finalization tool whitelist (subset of the agent's allowed tools):
        # if provided, only these remain available on the forced-final call so
        # the agent can persist its work (e.g. save_diary) on the way out.
        final_tool_names, final_tools = adapter.build_finalization_tools(finalization_tools)
        adapter.append_user_text(
            working_msgs,
            build_limit_message(
                limit_reason, state.total_cost, budget_usd,
                round_num, total_round_limit, was_still_working,
                finalization_tools=final_tool_names or None,
            ),
        )

        tracker_response = None
        try:
            final_response = await adapter.call_final(working_msgs, final_tools, limit_reason)
            adapter.track_cost(final_response, "Forced-final")
            tracker_response = final_response
            final_turn = adapter.parse_final(final_response, final_tool_names)
            text_parts = list(final_turn.text_parts)

            if final_turn.has_protocol:
                adapter.append_final_assistant(working_msgs, final_turn)
                final_exec = []
                if final_turn.batch:
                    logger.info(
                        "Forced-final: executing %d finalization tool call(s)",
                        len(final_turn.batch),
                    )
                    final_exec = await adapter.run_batch(final_turn.batch, round_num + 1)
                    for tid, tname, tinput, result, is_error in final_exec:
                        input_summary = json.dumps(tinput, ensure_ascii=False)
                        tool_call_log.append(f"  [final] {tname}({input_summary})")
                        tool_work_details.append(f"  [final] {tname}({input_summary}) → {result}")
                        save_redis_progress(
                            task_id, round_num + 1, tname, input_summary, result, is_error,
                        )
                adapter.append_final_results(working_msgs, final_turn, final_exec)

                # Skip the followup roundtrip when the forced-final already
                # produced substantive closing text. The finalization tool ran
                # (durable persistence) and the pre-tool text typically already
                # contains the self-critique the agent was prompted for —
                # re-sending the whole conversation just to collect a "done"
                # line at full input pricing is the dominant cost in
                # autonomous_project ticks.
                pre_tool_text = "\n".join(p for p in final_turn.text_parts if p).strip()
                if final_turn.batch and len(pre_tool_text) >= FOLLOWUP_SKIP_MIN_CHARS:
                    tool_summary = ", ".join(
                        f"{tname}{'(error)' if is_error else ''}"
                        for _tid, tname, _ti, _r, is_error in final_exec
                    )
                    logger.info(
                        "Forced-final: skipping followup (pre-tool text %d chars; tools=%s)",
                        len(pre_tool_text), tool_summary,
                    )
                    text_parts = [pre_tool_text]
                else:
                    followup = await adapter.call_followup(working_msgs)
                    adapter.track_cost(followup, "Forced-final followup")
                    tracker_response = followup
                    text_parts = text_parts + adapter.extract_text_parts(followup)
        except LoopEarlyReturn:
            raise
        except Exception as final_err:
            text_parts = await adapter.recover_final_failure(
                working_msgs, final_err, limit_reason, was_still_working,
            )
            tracker_response = None

        return _tracked(
            accumulated_text_parts + text_parts,
            finish_reason=(
                "forced_final" if (was_still_working or budget_exhausted) else "stop"
            ),
            truncated=was_still_working,
            limit_reason=limit_reason,
            was_still_working=was_still_working,
            interrupted=was_still_working,
            response_obj=tracker_response,
        )
    except LoopEarlyReturn as early:
        return _tracked(
            [early.text],
            finish_reason=early.finish_reason,
            truncated=early.truncated,
            limit_reason=early.limit_reason,
            was_still_working=early.was_still_working,
            interrupted=early.interrupted,
        )
