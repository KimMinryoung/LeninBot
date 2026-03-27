"""openai_tool_loop.py — OpenAI-compatible tool-use loop for local LLM (MOON PC).

claude_loop.py와 동일한 인터페이스를 제공하되, OpenAI 호환 API
(/v1/chat/completions with function calling)를 사용한다.
llama-server, vLLM 등 OpenAI 호환 백엔드에서 동작.

사용 예:
    from openai_tool_loop import chat_with_tools

    result = await chat_with_tools(
        messages=history,
        base_url="http://127.0.0.1:8080",
        model="qwen3.5-9b",
        tools=tool_defs,          # Anthropic 포맷 → 내부에서 자동 변환
        tool_handlers=handlers,
        system_prompt="...",
    )
"""

import json
import logging
import httpx

logger = logging.getLogger(__name__)


# ── Anthropic → OpenAI tool format conversion ────────────────────────

def _convert_tool_anthropic_to_openai(tool: dict) -> dict:
    """Anthropic tool definition → OpenAI function tool definition.

    Anthropic: {"name", "description", "input_schema": {type, properties, required}}
    OpenAI:    {"type": "function", "function": {"name", "description", "parameters": {...}}}
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert a list of Anthropic-format tools to OpenAI format.

    Auto-detects: if already in OpenAI format (has "type": "function"), pass through.
    """
    converted = []
    for t in tools:
        if t.get("type") == "function":
            converted.append(t)
        elif "input_schema" in t:
            converted.append(_convert_tool_anthropic_to_openai(t))
        else:
            converted.append(t)
    return converted


# ── Message normalization ─────────────────────────────────────────────

def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize message history to plain text OpenAI chat format.

    Strips any Anthropic-specific block structures (tool_use, tool_result)
    and converts them to readable text, keeping only role + content strings.
    """
    clean: list[dict] = []
    for raw in messages:
        if not isinstance(raw, dict):
            clean.append({"role": "user", "content": str(raw)})
            continue

        role = raw.get("role", "user")
        if role not in ("user", "assistant", "system", "tool"):
            role = "assistant" if role in ("model", "bot") else "user"

        content = raw.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        name = block.get("name", "?")
                        inp = json.dumps(block.get("input", {}), ensure_ascii=False)[:500]
                        parts.append(f"[tool call: {name}({inp})]")
                    elif btype == "tool_result":
                        parts.append(block.get("content", ""))
                    else:
                        parts.append(str(block))
            text = "\n".join(p for p in parts if p)
        else:
            text = str(content)

        if clean and clean[-1]["role"] == role:
            clean[-1]["content"] += "\n" + text
        else:
            clean.append({"role": role, "content": text})

    return clean


# ── Core API call ─────────────────────────────────────────────────────

async def _call_api(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout: int = 300,
) -> dict:
    """Single call to OpenAI-compatible /v1/chat/completions."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


# ── Tool-use loop ────────────────────────────────────────────────────

async def chat_with_tools(
    messages: list[dict],
    *,
    base_url: str = "http://127.0.0.1:8080",
    model: str = "qwen3.5-9b",
    tools: list[dict],
    tool_handlers: dict,
    system_prompt: str,
    max_rounds: int = 30,
    max_tokens: int = 4096,
    log_event=None,
    budget_usd: float = 0.0,
    budget_tracker: dict | None = None,
    on_progress=None,
) -> str:
    """Call OpenAI-compatible LLM with tools, execute tool calls, loop until text response.

    Interface mirrors claude_loop.chat_with_tools() for drop-in use.
    budget_usd is accepted for compatibility but not enforced (local LLM = free).
    """
    openai_tools = _convert_tools(tools)
    working_msgs = _normalize_messages(messages)

    # Prepend system prompt
    if system_prompt:
        working_msgs.insert(0, {"role": "system", "content": system_prompt})

    tool_call_log = []
    tool_work_details = []

    for round_num in range(1, max_rounds + 1):
        try:
            data = await _call_api(
                base_url=base_url,
                model=model,
                messages=working_msgs,
                tools=openai_tools if round_num <= max_rounds else None,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("OpenAI API call failed at round %d: %s", round_num, e)
            if log_event:
                log_event("error", "openai_loop", f"API call failed: {e}")
            raise

        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        tool_calls = message.get("tool_calls")

        # No tool calls → return text response
        if not tool_calls or finish_reason == "stop":
            text = message.get("content", "") or ""
            if budget_tracker is not None:
                budget_tracker.update({
                    "total_cost": 0.0,
                    "rounds_used": round_num,
                    "was_interrupted": False,
                    "tool_work_details": list(tool_work_details),
                })
            return text.strip() if text.strip() else "응답을 생성하지 못했습니다."

        # Has tool calls → execute them
        # Append the assistant message (with tool_calls) to history
        assistant_msg = {"role": "assistant", "content": message.get("content") or ""}
        assistant_msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            }
            for tc in tool_calls
        ]
        working_msgs.append(assistant_msg)

        for tc in tool_calls:
            tc_id = tc["id"]
            func_name = tc["function"]["name"]
            try:
                func_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                func_args = {}

            input_summary = json.dumps(func_args, ensure_ascii=False)

            if on_progress:
                try:
                    await on_progress("tool_call", f"[{round_num}] {func_name}({input_summary})")
                except Exception:
                    pass

            handler = tool_handlers.get(func_name)
            if handler:
                logger.info("Tool call: %s(%s)", func_name, input_summary[:200])
                try:
                    result = await handler(**func_args)
                    is_error = False
                except Exception as e:
                    logger.error("Tool %s execution error: %s", func_name, e)
                    if log_event:
                        log_event("warning", "tool", f"Tool {func_name} failed: {e}")
                    result = f"Tool execution failed: {e}"
                    is_error = True
            else:
                result = f"Unknown tool: {func_name}"
                is_error = True

            if not isinstance(result, str) or result is None:
                result = str(result) if result is not None else "(no result)"

            # Append tool result as "tool" role message (OpenAI format)
            working_msgs.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })

            if on_progress:
                status = "error" if is_error else "ok"
                try:
                    await on_progress("tool_result", f"  [{status}] {result[:200]}")
                except Exception:
                    pass

            tool_call_log.append(f"  [{round_num}/{max_rounds}] {func_name}({input_summary})")
            tool_work_details.append(f"  [{round_num}] {func_name}({input_summary}) -> {result}")

    # Max rounds exhausted → force final text response (no tools)
    logger.warning(
        "Max rounds reached (%d). Forcing final response. Calls:\n%s",
        max_rounds, "\n".join(tool_call_log),
    )
    working_msgs.append({
        "role": "user",
        "content": (
            "[SYSTEM] 도구 호출 한도 도달. 추가 도구를 사용하지 말고, "
            "지금까지 수집한 정보만으로 최선의 답변을 완성하세요."
        ),
    })

    try:
        data = await _call_api(
            base_url=base_url,
            model=model,
            messages=working_msgs,
            tools=None,  # no tools → force text response
            max_tokens=max_tokens,
        )
        text = data["choices"][0]["message"].get("content", "")
    except Exception as e:
        logger.error("Final forced response failed: %s", e)
        text = f"도구 호출 한도 도달 후 응답 생성 실패: {e}"

    if budget_tracker is not None:
        budget_tracker.update({
            "total_cost": 0.0,
            "rounds_used": max_rounds,
            "was_interrupted": True,
            "tool_work_details": list(tool_work_details),
        })
    return text.strip() if text.strip() else "응답을 생성하지 못했습니다."
