"""codex_exec_loop.py — Delegate the entire task to OpenAI Codex CLI.

Wraps `codex exec --full-auto` as a chat_with_tools()-compatible function.
Used by agents whose AgentSpec has provider="codex". The agent's tool list
and chat history are flattened into a single Codex prompt; Codex runs
autonomously inside its workspace-write sandbox (its own bash/edit tools)
and returns one final message via -o file.

Auth: Codex CLI's existing OAuth login (~/.codex/auth.json), backed by the
ChatGPT Plus/Pro subscription. No OpenAI API key billing.

Interface mirrors claude_loop.chat_with_tools / openai_tool_loop.chat_with_tools
so the telegram bot dispatch can swap it in via spec.provider == "codex".
The `tools`/`tool_handlers` arguments are accepted for compatibility but
ignored — Codex uses its own built-in toolset.
"""

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from typing import Optional

from tool_loop_common import emit_progress, build_budget_tracker

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
CODEX_DEFAULT_MODEL = os.environ.get("CODEX_MODEL", "gpt-5.5")


def _extract_text(content) -> str:
    """Pull plain text out of an Anthropic/OpenAI message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits: list[str] = []
        for b in content:
            if isinstance(b, dict):
                btype = b.get("type")
                if btype == "text":
                    bits.append(b.get("text", ""))
                elif btype == "tool_use":
                    name = b.get("name", "?")
                    inp = json.dumps(b.get("input", {}), ensure_ascii=False)[:300]
                    bits.append(f"[tool_call: {name}({inp})]")
                elif btype == "tool_result":
                    rc = b.get("content", "")
                    if isinstance(rc, list):
                        rc = "\n".join(
                            x.get("text", str(x)) if isinstance(x, dict) else str(x)
                            for x in rc
                        )
                    bits.append(f"[tool_result: {str(rc)[:1500]}]")
            elif isinstance(b, str):
                bits.append(b)
        return "\n".join(p for p in bits if p)
    return str(content) if content is not None else ""


_EXECUTION_POLICY = """\
<execution-policy>
- You have full autonomy: read/edit any file under the workspace, run shell commands, install deps as needed.
- DO NOT run `systemctl restart leninbot-*` or any service restart — that requires explicit user approval.
- Git commits MUST use author Cyber-Lenin <lenin@cyber-lenin.com>:
    git -c user.name=Cyber-Lenin -c user.email=lenin@cyber-lenin.com commit -m "..."
- The project's Python venv is at ./venv (NOT .venv). Activate it for any Python invocation.
- After finishing, output a concise final report: changed files, what was modified, verification done, anything left undone.
- If a service restart is required for the change to take effect, say so explicitly in the final report — do not restart.
</execution-policy>"""


def _flatten_messages_to_prompt(messages: list[dict], system_prompt: str) -> str:
    """Render the agent's full context into a single Codex prompt string.

    Codex exec takes one prompt and runs autonomously — there is no chat
    protocol on the wire. We embed system instructions and prior turns
    as readable XML blocks so Codex can use them as background context
    while focusing on the latest task.
    """
    parts: list[str] = []

    if system_prompt:
        parts.append(f"<orchestrator-instructions>\n{system_prompt.strip()}\n</orchestrator-instructions>")

    user_msgs = [m for m in messages if m.get("role") == "user"]
    latest_task = _extract_text(user_msgs[-1].get("content")) if user_msgs else ""

    history = messages[:-1] if user_msgs else messages
    history_lines: list[str] = []
    for m in history:
        role = m.get("role", "user")
        if role not in ("user", "assistant", "system"):
            continue
        text = _extract_text(m.get("content"))
        if text.strip():
            history_lines.append(f"<{role}>\n{text.strip()}\n</{role}>")
    if history_lines:
        parts.append("<prior-conversation>\n" + "\n".join(history_lines) + "\n</prior-conversation>")

    parts.append(f"<task>\n{latest_task.strip()}\n</task>")
    parts.append(_EXECUTION_POLICY)

    return "\n\n".join(parts)


async def chat_with_tools(
    messages: list[dict],
    *,
    base_url: str = "",
    model: Optional[str] = None,
    tools: Optional[list[dict]] = None,
    tool_handlers: Optional[dict] = None,
    system_prompt: str = "",
    max_rounds: int = 50,
    max_tokens: int = 8192,
    log_event=None,
    budget_usd: float = 0.0,
    budget_tracker: Optional[dict] = None,
    on_progress=None,
    task_id: Optional[int] = None,
    context_limit: int = 0,
    enable_thinking: bool = False,
    agent_name: str = "codex",
    mission_id: Optional[int] = None,
    finalization_tools: Optional[list[str]] = None,
    terminal_tools: Optional[list[str]] = None,
    api_semaphore: Optional[asyncio.Semaphore] = None,
) -> str:
    """Run the entire task in one Codex CLI invocation.

    The `tools`/`tool_handlers` args are accepted for interface compatibility
    but ignored — Codex uses its own built-in toolset (read/write/exec)
    inside its workspace-write sandbox.
    """
    prompt = _flatten_messages_to_prompt(messages, system_prompt)
    chosen_model = model or CODEX_DEFAULT_MODEL

    # Final-message file ensures clean extraction without parsing JSONL events.
    fd, last_msg_path = tempfile.mkstemp(
        suffix=".txt", prefix=f"codex_{agent_name}_", dir="/tmp",
    )
    os.close(fd)

    cmd = [
        CODEX_BIN, "exec",
        "--full-auto",
        "-C", PROJECT_ROOT,
        "-m", chosen_model,
        "--ephemeral",
        "--json",
        "-o", last_msg_path,
        "-",
    ]

    logger.info(
        "[codex] agent=%s model=%s prompt_len=%d task=%s",
        agent_name, chosen_model, len(prompt), task_id,
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PROJECT_ROOT,
    )

    assert proc.stdin and proc.stdout and proc.stderr
    proc.stdin.write(prompt.encode())
    await proc.stdin.drain()
    proc.stdin.close()

    rounds = 0

    async def _pump_events():
        nonlocal rounds
        async for raw in proc.stdout:
            try:
                event = json.loads(raw.decode().strip())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            etype = str(event.get("type") or event.get("event") or "")
            # Codex's JSONL event taxonomy varies across versions; match by
            # suffix on the event-type string so we stay robust.
            if etype.endswith("agent_message_delta"):
                delta = event.get("delta") or event.get("text") or ""
                if delta:
                    await emit_progress(on_progress, "text_delta", delta)
            elif etype.endswith("agent_message"):
                msg = event.get("message") or event.get("text") or ""
                if msg:
                    await emit_progress(on_progress, "thinking", f"[codex] {str(msg)[:200]}")
            elif etype.endswith("exec_command_begin"):
                rounds += 1
                cmd_field = event.get("command")
                if isinstance(cmd_field, list):
                    cmd_str = " ".join(str(x) for x in cmd_field)
                else:
                    cmd_str = str(cmd_field or "")
                await emit_progress(
                    on_progress, "thinking",
                    f"[codex round {rounds}] $ {cmd_str[:160]}",
                )
            elif etype.endswith("patch_apply_begin"):
                rounds += 1
                await emit_progress(
                    on_progress, "thinking",
                    f"[codex round {rounds}] patch_apply",
                )

    async def _drain_stderr() -> bytes:
        chunks: list[bytes] = []
        async for raw in proc.stderr:
            chunks.append(raw)
        return b"".join(chunks)

    pump_task = asyncio.create_task(_pump_events())
    stderr_task = asyncio.create_task(_drain_stderr())
    try:
        rc = await proc.wait()
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise
    finally:
        # Drain any remaining events before we read the final-message file.
        try:
            await asyncio.wait_for(pump_task, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump_task

    try:
        stderr_bytes = await asyncio.wait_for(stderr_task, timeout=5.0)
    except (asyncio.TimeoutError, Exception):
        stderr_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await stderr_task
        stderr_bytes = b""

    final_text = ""
    try:
        with open(last_msg_path, "r", encoding="utf-8") as f:
            final_text = f.read().strip()
    except OSError as e:
        logger.warning("[codex] could not read last-message file %s: %s", last_msg_path, e)
    finally:
        try:
            os.unlink(last_msg_path)
        except OSError:
            pass

    if rc != 0 and not final_text:
        err_tail = stderr_bytes.decode(errors="replace")[-1500:]
        logger.error("[codex] exit=%d stderr=%s", rc, err_tail)
        if log_event:
            log_event("error", "codex", f"codex exec failed (rc={rc}): {err_tail}")
        final_text = f"⚠️ Codex 실행 실패 (exit {rc}):\n{err_tail}"

    # ChatGPT Plus/Pro subscription is flat-rate — actual USD cost is 0
    # from our perspective. Round count reflects Codex's own tool actions.
    if budget_tracker is not None:
        budget_tracker.update(build_budget_tracker(0.0, max(rounds, 1), False, []))

    logger.info(
        "[codex] agent=%s exit=%d rounds=%d output_len=%d",
        agent_name, rc, rounds, len(final_text),
    )

    return final_text or "⚠️ Codex returned no output."
