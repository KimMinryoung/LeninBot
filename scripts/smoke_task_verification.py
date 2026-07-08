"""Smoke test for the Phase-1 task verification wiring (Critic activation).

Hermetic: monkeypatches telegram.tasks._execute so no DB is touched, and uses
stub chat/model fns so no LLM is called. Prints every intermediate result in
full so the operator can inspect the actual behavior.

Run:  venv/bin/python scripts/smoke_task_verification.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import telegram.tasks as tasks

PASSED = []
FAILED = []


def check(name: str, cond: bool, detail: str = ""):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


# ── Capture DB writes instead of executing them ─────────────────────
CAPTURED_SQL = []


def _fake_execute(sql, params=None):
    CAPTURED_SQL.append((" ".join(sql.split()), params))


tasks._execute = _fake_execute


def make_task(task_id, agent_type, metadata=None, content="Analyze X and report."):
    return {
        "id": task_id,
        "user_id": 0,
        "agent_type": agent_type,
        "content": content,
        "metadata": json.dumps(metadata) if metadata is not None else None,
    }


GOOD_REPORT = (
    "## Executive Summary\n"
    "The analysis is complete: X grew 12% QoQ driven by policy changes.\n\n"
    "## Details\nFull reasoning here."
)


async def main():
    print("=" * 72)
    print("1. Default verification policy synthesis (_normalize_verification_policy)")
    print("=" * 72)

    for agent, expect_checks in [
        ("programmer", ["task_report", "server_logs"]),
        ("analyst", ["task_report"]),
        ("scout", ["task_report"]),
        ("diplomat", ["task_report"]),
    ]:
        policy = tasks._normalize_verification_policy(make_task(1, agent))
        print(f"  [{agent}] -> {policy}")
        check(
            f"default policy for {agent}",
            policy is not None and policy["checks"] == expect_checks and policy["required"],
        )

    for agent in ("visualizer", "diary", "browser", "stasova"):
        policy = tasks._normalize_verification_policy(make_task(1, agent))
        print(f"  [{agent}] -> {policy}")
        check(f"no default policy for {agent}", policy is None)

    explicit = tasks._normalize_verification_policy(
        make_task(1, "visualizer", metadata={"verification": {"checks": ["task_report"], "urls": ["https://cyber-lenin.com"]}})
    )
    print(f"  [visualizer + explicit policy] -> {explicit}")
    check(
        "explicit policy overrides missing default",
        explicit is not None and "url_access" in explicit["checks"],
    )

    optout = tasks._normalize_verification_policy(
        make_task(1, "programmer", metadata={"verification": {"required": False}})
    )
    print(f"  [programmer + required=false] -> {optout}")
    check("opt-out yields required=False", optout is not None and optout["required"] is False)

    print()
    print("=" * 72)
    print("2. Verifier tool surface (_build_verifier_toolset)")
    print("=" * 72)

    shadow_tools, shadow_handlers = tasks._build_verifier_toolset("shadow")
    shadow_names = sorted(t["name"] for t in shadow_tools)
    print(f"  shadow tools: {shadow_names}")
    check(
        "shadow toolset is read-only",
        "restart_service" not in shadow_names and set(shadow_names) == set(shadow_handlers),
    )
    check(
        "shadow toolset has the observation tools",
        {"read_self", "read_file", "search_files", "list_directory", "fetch_url"} <= set(shadow_names),
    )

    enforce_tools, _ = tasks._build_verifier_toolset("enforce")
    enforce_names = sorted(t["name"] for t in enforce_tools)
    print(f"  enforce tools: {enforce_names}")
    check("enforce toolset adds restart_service", "restart_service" in enforce_names)

    print()
    print("=" * 72)
    print("3. _run_verification end-to-end with stub LLM")
    print("=" * 72)

    async def stub_model():
        return "stub-low-model"

    def make_stub_chat(response):
        async def stub_chat(messages, **kwargs):
            print(f"  [stub llm] called with model={kwargs.get('model')}, budget={kwargs.get('budget_usd')}")
            print(f"  [stub llm] prompt head: {messages[0]['content'][:160]!r}")
            return response
        return stub_chat

    # 3a. PASS verdict
    CAPTURED_SQL.clear()
    v = await tasks._run_verification(
        None, make_task(999901, "analyst"), GOOD_REPORT,
        chat_with_tools_fn=make_stub_chat("VERDICT: PASS\nReason: report is substantiated."),
        get_model_fn=stub_model,
    )
    print(f"  result: {v}")
    print(f"  SQL: {CAPTURED_SQL}")
    check("analyst PASS verdict -> passed", v["status"] == "passed")
    check(
        "passed status persisted",
        any("verification_status" in sql and params[0] == "passed" for sql, params in CAPTURED_SQL),
    )

    # 3b. FAIL verdict
    CAPTURED_SQL.clear()
    v = await tasks._run_verification(
        None, make_task(999902, "analyst"), GOOD_REPORT,
        chat_with_tools_fn=make_stub_chat("VERDICT: FAIL\nReason: claimed file was never modified."),
        get_model_fn=stub_model,
    )
    print(f"  result: {v}")
    check("analyst FAIL verdict -> failed", v["status"] == "failed")
    check("failure reason recorded", "claimed file was never modified" in v["details"])

    # 3c. empty report fails the automated check without calling the LLM
    CAPTURED_SQL.clear()
    v = await tasks._run_verification(
        None, make_task(999903, "analyst"), "",
        chat_with_tools_fn=make_stub_chat("VERDICT: PASS"),
        get_model_fn=stub_model,
    )
    print(f"  result: {v}")
    check("empty report -> failed without LLM", v["status"] == "failed" and "llm_verification" not in v["details"])

    # 3d. opt-out short-circuits to passed-by-default
    CAPTURED_SQL.clear()
    v = await tasks._run_verification(
        None, make_task(999904, "programmer", metadata={"verification": {"required": False}}), GOOD_REPORT,
        chat_with_tools_fn=make_stub_chat("VERDICT: FAIL"),
        get_model_fn=stub_model,
    )
    print(f"  result: {v}")
    check("opt-out -> passed by default", v["status"] == "passed" and "No verification policy" in v["details"])

    # 3e. agent without default policy short-circuits too
    CAPTURED_SQL.clear()
    v = await tasks._run_verification(
        None, make_task(999905, "visualizer"), GOOD_REPORT,
        chat_with_tools_fn=make_stub_chat("VERDICT: FAIL"),
        get_model_fn=stub_model,
    )
    print(f"  result: {v}")
    check("visualizer (no policy) -> passed by default", v["status"] == "passed")

    # 3f. verifier LLM exception degrades to auto-check result, not a crash
    async def broken_chat(messages, **kwargs):
        raise RuntimeError("provider down")

    CAPTURED_SQL.clear()
    v = await tasks._run_verification(
        None, make_task(999906, "analyst"), GOOD_REPORT,
        chat_with_tools_fn=broken_chat,
        get_model_fn=stub_model,
    )
    print(f"  result: {v}")
    check(
        "LLM error degrades gracefully",
        v["status"] == "passed" and "llm_verification: error" in v["details"],
    )

    print()
    print("=" * 72)
    print("4. Config plumbing")
    print("=" * 72)
    from bot_config import get_task_verification_mode, _CONFIG_DEFAULTS, _CONFIG_META
    mode = get_task_verification_mode()
    print(f"  task_verification_mode = {mode!r}")
    print(f"  default = {_CONFIG_DEFAULTS['task_verification_mode']!r}")
    print(f"  meta = {_CONFIG_META['task_verification_mode']}")
    check("mode is valid", mode in ("off", "shadow", "enforce"))
    check("shipping default is shadow", _CONFIG_DEFAULTS["task_verification_mode"] == "shadow")

    print()
    print("=" * 72)
    print("5. Delegation schema carries verification policy")
    print("=" * 72)
    from self_runtime.tools import SELF_TOOLS
    delegate = next(t for t in SELF_TOOLS if t["name"] == "delegate")
    multi = next(t for t in SELF_TOOLS if t["name"] == "multi_delegate")
    dv = delegate["input_schema"]["properties"].get("verification")
    mv = multi["input_schema"]["properties"]["tasks"]["items"]["properties"].get("verification")
    print(f"  delegate.verification keys: {sorted((dv or {}).get('properties', {}))}")
    print(f"  multi_delegate item verification present: {mv is not None}")
    check("delegate schema has verification object", dv is not None and dv["type"] == "object")
    check("multi_delegate items have verification object", mv is not None)

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
