"""Interactive CLI for the local agent."""

import asyncio
import logging
import os
import sys

logger = logging.getLogger(__name__)

# ── History Compression Constants ─────────────────────────────────────
_HISTORY_TOKEN_LIMIT = 20_000  # compress if history exceeds this
_RECENT_TURNS_KEEP = 5         # keep last N turns uncompressed (5 turns = 10 msgs)
_SUMMARY_CHUNK_SIZE = 10       # messages per summary chunk

# ANSI color codes
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _print_tool_call(event: str, detail: str):
    """Print tool call notification to stderr (so it doesn't mix with piped output)."""
    print(f"  {_DIM}{detail}{_RESET}", file=sys.stderr)


def _print_banner(tool_count: int):
    print(f"""
{_BOLD}{'=' * 50}
  Cyber-Lenin Local Agent
  Model: Claude Sonnet 4.6
  Tools: {tool_count}
{'=' * 50}{_RESET}

Commands: {_DIM}/quit  /clear  /tasks  /history{_RESET}
""")


async def _handle_command(cmd: str, messages: list[dict]) -> bool:
    """Handle special /commands. Returns True if handled."""
    cmd = cmd.strip().lower()

    if cmd in ("/quit", "/exit", "/q"):
        print(f"\n{_DIM}Shutting down...{_RESET}")
        # Clean up Playwright
        try:
            from local_agent.crawler import close
            await close()
        except Exception:
            pass
        return True  # Signal to exit

    if cmd == "/clear":
        messages.clear()
        from local_agent.local_db import execute
        execute("DELETE FROM conversations")
        print(f"{_DIM}Conversation cleared (memory + DB).{_RESET}")
        return True

    if cmd == "/tasks":
        from local_agent.local_db import query
        rows = query("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 20")
        if not rows:
            print(f"{_DIM}No local tasks.{_RESET}")
        else:
            for r in rows:
                status_color = _GREEN if r["status"] == "done" else _YELLOW
                print(f"  #{r['id']} {status_color}[{r['status']}]{_RESET} {r['content'][:80]}")
        return True

    if cmd == "/history":
        if not messages:
            print(f"{_DIM}No conversation history.{_RESET}")
        else:
            for m in messages[-20:]:
                role = m["role"]
                content = m["content"] if isinstance(m["content"], str) else "(complex)"
                color = _CYAN if role == "user" else _GREEN
                print(f"  {color}{role}{_RESET}: {content[:120]}")
        return True

    return False


# ── History Compression ───────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~3 chars/token for Korean+English mix)."""
    if not isinstance(text, str):
        text = str(text)
    return len(text) // 3


async def _maybe_summarize_chunk():
    """Create a summary chunk if enough unsummarized messages have accumulated."""
    from local_agent.local_db import query, execute

    # Find where the last summary ends
    last = query(
        "SELECT chunk_end_id FROM chat_summaries ORDER BY chunk_end_id DESC LIMIT 1"
    )
    raw_after = last[0]["chunk_end_id"] if last else 0

    # Bootstrap: if no summaries yet, only consider recent messages
    if not last:
        latest = query("SELECT MAX(id) AS max_id FROM conversations")
        if latest and latest[0]["max_id"]:
            raw_after = max(0, latest[0]["max_id"] - _SUMMARY_CHUNK_SIZE * 4)

    rows = query(
        "SELECT id, role, content FROM conversations "
        "WHERE id > ? ORDER BY id ASC LIMIT ?",
        (raw_after, _SUMMARY_CHUNK_SIZE + 5),
    )

    if len(rows) < _SUMMARY_CHUNK_SIZE:
        return

    chunk = rows[:_SUMMARY_CHUNK_SIZE]
    chunk_start_id = chunk[0]["id"]
    chunk_end_id = chunk[-1]["id"]

    conversation_text = "\n".join(
        f"[{r['role']}] {r['content'][:500]}" for r in chunk
    )
    summary_prompt = (
        "아래 대화를 핵심 정보만 남기고 간결하게 요약해라. "
        "사용자가 어떤 주제를 물었고, 어떤 결론/답변이 나왔는지 위주로. "
        "고유명사, 수치, 날짜는 보존. 300자 이내.\n\n"
        + conversation_text
    )

    try:
        import anthropic
        client = anthropic.AsyncAnthropic()
        resp = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        summary = "\n".join(b.text for b in resp.content if b.type == "text")
    except Exception as e:
        logger.warning("Chunk summarization failed: %s", e)
        return

    execute(
        "INSERT INTO chat_summaries "
        "(chunk_start_id, chunk_end_id, summary, msg_count) "
        "VALUES (?, ?, ?, ?)",
        (chunk_start_id, chunk_end_id, summary, len(chunk)),
    )
    logger.info(
        "Chunk summary created: msgs #%d~#%d (%d msgs)",
        chunk_start_id, chunk_end_id, len(chunk),
    )


async def _compress_history(messages: list[dict]) -> list[dict]:
    """Compress history if it exceeds the token limit."""
    total_tokens = sum(_estimate_tokens(m.get("content", "")) for m in messages)
    if total_tokens <= _HISTORY_TOKEN_LIMIT:
        return messages

    # Split into old (to summarize) and recent (to keep)
    keep_count = _RECENT_TURNS_KEEP * 2  # user+assistant pairs
    if len(messages) <= keep_count:
        return messages

    old_msgs = messages[:-keep_count]
    recent_msgs = messages[-keep_count:]

    logger.info(
        "Compressing history: %d msgs (%d tokens) → summarize %d old, keep %d recent",
        len(messages), total_tokens, len(old_msgs), len(recent_msgs),
    )

    conversation_text = "\n".join(
        f"[{m['role']}] {m.get('content', '')[:1000]}" for m in old_msgs
    )
    summary_prompt = (
        "아래 대화를 핵심 정보만 남기고 간결하게 요약해라. "
        "사용자가 어떤 주제를 물었고, 어떤 결론/답변이 나왔는지 위주로. "
        "고유명사, 수치, 날짜는 보존. 300자 이내.\n\n"
        + conversation_text
    )

    try:
        import anthropic
        client = anthropic.AsyncAnthropic()
        resp = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        summary = "\n".join(b.text for b in resp.content if b.type == "text")
    except Exception as e:
        logger.warning("History compression failed: %s — truncating", e)
        return recent_msgs

    compressed = [
        {"role": "user", "content": f"[이전 대화 요약]\n{summary}"},
        {"role": "assistant", "content": "네, 이전 대화 내용을 파악했습니다. 이어서 진행하겠습니다."},
    ] + recent_msgs

    new_tokens = sum(_estimate_tokens(m.get("content", "")) for m in compressed)
    logger.info("History compressed: %d → %d tokens", total_tokens, new_tokens)
    return compressed


async def main():
    """Main REPL loop."""
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Show info for our modules only
    logging.getLogger("local_agent").setLevel(logging.INFO)

    # Initialize SQLite
    from local_agent.local_db import init_db
    init_db()

    # Load tools to show count
    from local_agent.agent import _get_tools_and_handlers
    tools, _ = _get_tools_and_handlers()
    _print_banner(len(tools))

    # Load conversation history from SQLite (last 40 messages = 20 turns)
    from local_agent.local_db import query as db_query
    rows = db_query(
        "SELECT role, content FROM conversations ORDER BY id DESC LIMIT 40"
    )
    rows.reverse()
    messages: list[dict] = [{"role": r["role"], "content": r["content"]} for r in rows]
    if messages:
        print(f"{_DIM}Restored {len(messages)} messages from previous session.{_RESET}")

    session_cost = 0.0

    while True:
        try:
            user_input = await asyncio.to_thread(input, f"{_CYAN}You>{_RESET} ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_DIM}Goodbye.{_RESET}")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle /commands
        if user_input.startswith("/"):
            should_exit = await _handle_command(user_input, messages)
            if user_input.strip().lower() in ("/quit", "/exit", "/q"):
                break
            if should_exit:
                continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Save to local DB
        from local_agent.local_db import execute
        execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("user", user_input))

        # Compress history if needed, then call the agent
        compressed = await _compress_history(messages)

        print(f"{_DIM}Thinking...{_RESET}", end="", flush=True)
        try:
            from local_agent.agent import chat
            reply, budget_info = await chat(compressed, on_tool_call=_print_tool_call)

            # Clear the "Thinking..." line
            print(f"\r{' ' * 40}\r", end="")

            # Print response
            print(f"{_GREEN}{reply}{_RESET}")

            # Print cost info
            turn_cost = budget_info.get("total_cost", 0)
            rounds_used = budget_info.get("rounds_used", 0)
            session_cost += turn_cost
            was_interrupted = budget_info.get("was_interrupted", False)
            cost_line = f"${turn_cost:.4f} ({rounds_used}r)"
            if was_interrupted:
                cost_line += f" {_YELLOW}[interrupted]{_RESET}"
            print(f"{_DIM}  cost: {cost_line} | session: ${session_cost:.4f}{_RESET}\n")

            # Add to history
            messages.append({"role": "assistant", "content": reply})

            # Save to local DB
            execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", reply))

            # Background: summarize old chunks if accumulated
            asyncio.create_task(_maybe_summarize_chunk())

        except KeyboardInterrupt:
            print(f"\n{_YELLOW}Interrupted.{_RESET}")
            # Remove the unanswered user message
            if messages and messages[-1]["role"] == "user":
                messages.pop()
        except Exception as e:
            print(f"\r{' ' * 40}\r", end="")
            print(f"{_YELLOW}Error: {e}{_RESET}\n")
            logger.error("Agent error", exc_info=True)
            # Remove the unanswered user message
            if messages and messages[-1]["role"] == "user":
                messages.pop()
