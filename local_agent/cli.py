"""Interactive CLI for the local agent."""

import asyncio
import json
import logging
import sys

logger = logging.getLogger(__name__)

# ANSI color codes
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _print_tool_call(name: str, input_summary: str):
    """Print tool call notification to stderr (so it doesn't mix with piped output)."""
    print(f"  {_DIM}[tool] {name}({input_summary}){_RESET}", file=sys.stderr)


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

        # Call the agent
        print(f"{_DIM}Thinking...{_RESET}", end="", flush=True)
        try:
            from local_agent.agent import chat
            reply = await chat(messages, on_tool_call=_print_tool_call)

            # Clear the "Thinking..." line
            print(f"\r{' ' * 40}\r", end="")

            # Print response
            print(f"{_GREEN}{reply}{_RESET}\n")

            # Add to history
            messages.append({"role": "assistant", "content": reply})

            # Save to local DB
            execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", reply))

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
