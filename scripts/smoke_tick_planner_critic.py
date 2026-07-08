"""Smoke test for the Phase-4 autonomous tick Planner + Critic.

Hermetic: stub chat fns, monkeypatched event logging and model resolution —
no LLM, no Postgres. Prints every intermediate result in full.

Run:  venv/bin/python scripts/smoke_tick_planner_critic.py
"""

import asyncio
import inspect
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASSED = []
FAILED = []


def check(name: str, cond: bool, detail: str = ""):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


PLANNER_REPLY = (
    "OBJECTIVE: Fact-check the staged draft 'hbm-rents' against its cited sources and publish it.\n"
    "ARTIFACT: publication\n"
    "WHY: The draft has been staged for two ticks and editorial notes are addressed."
)

PROJECT = {"id": 42, "title": "T", "topic": "semis", "goal": "Track HBM political economy", "state": "researching"}


async def main():
    import bot_config
    import autonomous_project as ap

    logged_events = []
    ap._log_event = lambda pid, et, content="", meta=None, **kw: logged_events.append((et, content, meta))

    async def fake_profile(provider):
        return "deepseek", SimpleNamespace(model_id="deepseek-v4-flash")

    ap._cheap_reviewer_profile = fake_profile

    def make_chat(response, raise_exc=False):
        calls = []

        async def chat(messages, **kwargs):
            calls.append((messages, kwargs))
            if raise_exc:
                raise RuntimeError("provider down")
            return response

        chat.calls = calls
        return chat

    print("=" * 72)
    print("1. Pre-tick planner (_plan_tick_objective)")
    print("=" * 72)

    big_prompt = "<project>...</project>\n" + ("context " * 5000)  # > 20k chars
    chat = make_chat(PLANNER_REPLY)
    logged_events.clear()
    obj = await ap._plan_tick_objective(PROJECT, big_prompt, "deepseek", chat)
    sent = chat.calls[0][0][0]["content"]
    kwargs = chat.calls[0][1]
    print(f"  objective:\n{obj}")
    print(f"  kwargs: model={kwargs.get('model')}, budget={kwargs.get('budget_usd')}, agent={kwargs.get('agent_name')}")
    print(f"  context sent: {len(sent)} chars (prompt was {len(big_prompt)})")
    check("planner returns objective text", obj == PLANNER_REPLY)
    check("tick_objective event logged", any(e[0] == "tick_objective" for e in logged_events))
    check("planner context capped at ~20k", len(sent) < ap._TICK_PLANNER_CONTEXT_CAP + 200)
    check("planner uses cheap model + tiny budget", kwargs.get("model") == "deepseek-v4-flash" and kwargs.get("budget_usd") == 0.05)

    logged_events.clear()
    obj = await ap._plan_tick_objective(PROJECT, "ctx", "deepseek", make_chat("I think you should do research."))
    check("malformed reply -> None, no event", obj is None and not logged_events)

    obj = await ap._plan_tick_objective(PROJECT, "ctx", "deepseek", make_chat("", raise_exc=True))
    check("planner exception -> None", obj is None)

    bot_config._config["autonomous_tick_planner"] = False
    chat = make_chat(PLANNER_REPLY)
    obj = await ap._plan_tick_objective(PROJECT, "ctx", "deepseek", chat)
    check("flag off -> no call", obj is None and len(chat.calls) == 0)
    bot_config._config["autonomous_tick_planner"] = True

    print()
    print("=" * 72)
    print("2. Objective injection block")
    print("=" * 72)

    xml_block = ap._format_tick_objective_block(PLANNER_REPLY, "claude")
    md_block = ap._format_tick_objective_block(PLANNER_REPLY, "openai")
    print(f"  xml head: {xml_block[:80]!r}")
    print(f"  md head: {md_block[:80]!r}")
    check("xml providers get <tick-objective>", xml_block.startswith("<tick-objective>") and PLANNER_REPLY in xml_block)
    check("markdown providers get ### section", md_block.startswith("### Tick Objective") and PLANNER_REPLY in md_block)

    print()
    print("=" * 72)
    print("3. Actions summary (_summarize_tick_actions_for_review)")
    print("=" * 72)

    summary = ap._summarize_tick_actions_for_review({
        "notes": ["Found that SK hynix HBM capacity is sold out through 2027." * 20],
        "staged_drafts": ["[공개 초안] HBM rents draft"],
        "publications": [],
        "plan_rationale": "Reordered plan to prioritize publication",
        "state_change": None,
    })
    print(f"  summary:\n{summary}")
    check(
        "summary lists notes/drafts/plan with caps",
        "research note:" in summary and "staged draft:" in summary and "plan revision:" in summary
        and len(summary.splitlines()[0]) < 340,
    )
    check("empty actions -> explicit no-op text", "(no durable actions" in ap._summarize_tick_actions_for_review({}))

    print()
    print("=" * 72)
    print("4. Post-tick critic (_review_tick_outcome)")
    print("=" * 72)

    actions = {"notes": ["New finding: X"], "staged_drafts": [], "publications": [], "plan_rationale": None, "state_change": None}

    for reply, expected in [
        ("VERDICT: advanced\nREASON: staged draft was published as planned.", "advanced"),
        ("VERDICT: Partial\nREASON: note saved but the objective named publication.", "partial"),
        ("VERDICT: no-op\nREASON: the note restates the synthesis.", "no-op"),
        ("The tick went okay I suppose.", "inconclusive"),
    ]:
        logged_events.clear()
        out = await ap._review_tick_outcome(PROJECT, PLANNER_REPLY, actions, "deepseek", make_chat(reply))
        ev = next((e for e in logged_events if e[0] == "tick_review"), None)
        print(f"  reply={reply.splitlines()[0][:60]!r} -> verdict={out['verdict']}, event_meta={ev[2] if ev else None}")
        check(f"verdict parsed: {expected}", out["verdict"] == expected and ev is not None and ev[2]["verdict"] == expected)

    chat = make_chat("VERDICT: advanced\nREASON: ok")
    await ap._review_tick_outcome(PROJECT, None, actions, "deepseek", chat)
    user_msg = chat.calls[0][0][0]["content"]
    check("no objective -> judged against one-concrete-step standard", "one-concrete-step" in user_msg)
    check("critic sees goal + actions", "Track HBM political economy" in user_msg and "New finding: X" in user_msg)

    out = await ap._review_tick_outcome(PROJECT, PLANNER_REPLY, actions, "deepseek", make_chat("", raise_exc=True))
    check("critic exception -> None", out is None)

    bot_config._config["autonomous_tick_critic"] = False
    chat = make_chat("VERDICT: advanced\nREASON: ok")
    out = await ap._review_tick_outcome(PROJECT, PLANNER_REPLY, actions, "deepseek", chat)
    check("flag off -> no call", out is None and len(chat.calls) == 0)
    bot_config._config["autonomous_tick_critic"] = True

    print()
    print("=" * 72)
    print("5. Next-tick warning feed")
    print("=" * 72)

    check("tick_review is an attention event type", "tick_review" in ap._TICK_ATTENTION_EVENT_TYPES)
    src = inspect.getsource(ap._recent_tick_attention_events)
    check(
        "warnings query filters to partial/no-op verdicts",
        "meta->>'verdict' IN ('partial', 'no-op')" in src,
    )

    print()
    print("=" * 72)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED CHECKS:")
        for name in FAILED:
            print(f"  - {name}")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
