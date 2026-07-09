"""Smoke test for the Phase-2 generalized Reflexion pass (diagnose → revise).

Hermetic: stub chat/model fns, monkeypatched DB access — no LLM, no Postgres.
Prints every intermediate result in full so the operator can inspect behavior.

Run:  venv/bin/python scripts/smoke_reflexion.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASSED = []
FAILED = []


def check(name: str, cond: bool, detail: str = ""):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


LONG_REPORT = (
    "## Executive Summary\n"
    "Semiconductor exports rose 31% YoY in June, driven by HBM demand.\n\n"
    "## Analysis\n" + ("The structural shift toward AI-memory demand concentrates rents. " * 40)
)

NOTES = (
    '1. [사실성] "rose 31% YoY" — no source or period basis given in the text; direction: state the data source and comparison window.\n'
    "남길 것: the structural rent-concentration argument."
)


async def main():
    from llm.reflexion import (
        DIAGNOSIS_PASS_MARKER,
        build_report_revision_prompt,
        diagnose,
        diagnosis_is_pass,
        _diagnosis_system_prompt,
    )

    print("=" * 72)
    print("1. llm/reflexion.py primitives")
    print("=" * 72)

    check("PASS marker detection", diagnosis_is_pass("PASS") and diagnosis_is_pass("pass — clean"))
    check("notes are not PASS", not diagnosis_is_pass(NOTES) and not diagnosis_is_pass(None))

    sys_report = _diagnosis_system_prompt("task_report")
    sys_draft = _diagnosis_system_prompt("research_draft")
    print(f"  task_report system prompt: {len(sys_report)} chars")
    print(f"  research_draft system prompt: {len(sys_draft)} chars")
    check(
        "draft prompt adds publication check, report prompt does not",
        "공개 적합성" in sys_draft and "공개 적합성" not in sys_report,
    )
    check(
        "both demand PASS-or-numbered-notes format",
        DIAGNOSIS_PASS_MARKER in sys_report and "numbered diagnosis" in sys_report,
    )

    captured = {}

    async def stub_chat(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return NOTES

    notes = await diagnose(
        LONG_REPORT,
        chat_fn=stub_chat,
        model="stub-flash",
        content_kind="task_report",
        context="Analyze June semiconductor exports.",
        agent_name="reflexion_diagnosis",
    )
    user_msg = captured["messages"][0]["content"]
    print(f"  diagnose() user message head: {user_msg[:140]!r}")
    print(f"  diagnose() chat kwargs: model={captured['kwargs'].get('model')}, "
          f"budget={captured['kwargs'].get('budget_usd')}, agent_name={captured['kwargs'].get('agent_name')}")
    check("diagnose returns the editor notes", notes == NOTES)
    check("context and content reach the editor", "June semiconductor" in user_msg and "rose 31%" in user_msg)
    check("passthrough chat kwargs preserved", captured["kwargs"].get("agent_name") == "reflexion_diagnosis")

    empty = await diagnose("", chat_fn=stub_chat, model="stub-flash")
    check("empty content skips the call", empty is None)

    rev_prompt = build_report_revision_prompt(LONG_REPORT, NOTES, context="Analyze June semiconductor exports.")
    print(f"  revision prompt head: {rev_prompt[:120]!r}")
    check(
        "revision prompt carries notes + task + report + no-tools rule",
        all(s in rev_prompt for s in ("Editor's notes", "Original task", "Your report", "no tools")),
    )

    print()
    print("=" * 72)
    print("2. Task-report hook (_maybe_reflexion_revise_report)")
    print("=" * 72)

    import bot_config
    import telegram.tasks as tasks

    async def model_fn():
        return "stub-model"

    def make_chat(response):
        calls = []

        async def chat(messages, **kwargs):
            calls.append((messages, kwargs))
            return response

        chat.calls = calls
        return chat

    async def run_hook(*, report=LONG_REPORT, agent="analyst", diag_response=NOTES,
                       rev_response=None, diag_fn_present=True):
        diag = make_chat(diag_response)
        rev = make_chat(rev_response if rev_response is not None else LONG_REPORT + "\n(31% per KITA June customs data)")
        out = await tasks._maybe_reflexion_revise_report(
            task={"id": 1, "agent_type": agent},
            content="Analyze June semiconductor exports.",
            report=report,
            task_system_prompt="You are the analyst.",
            diagnose_chat_fn=diag if diag_fn_present else None,
            diagnose_model_fn=model_fn if diag_fn_present else None,
            revise_chat_fn=rev,
            revise_model_fn=model_fn,
        )
        return out, diag, rev

    out, diag, rev = await run_hook()
    print(f"  full flow: diagnosis calls={len(diag.calls)}, revision calls={len(rev.calls)}, revised={out is not None}")
    check("notes trigger author revision", out is not None and "KITA June customs" in out)
    # Text-only must be an EXPLICIT empty toolset: with extra_tools absent
    # (None), telegram _chat_with_tools treats the call as the orchestrator
    # and grants the full toolset — the opposite of text-only.
    check(
        "revision turn is text-only (explicit empty toolset, single round)",
        rev.calls[0][1].get("extra_tools") == []
        and rev.calls[0][1].get("extra_handlers") == {}
        and rev.calls[0][1].get("max_rounds") == 1,
    )
    check("revision reuses task system prompt", rev.calls[0][1].get("system_prompt") == "You are the analyst.")

    out, diag, rev = await run_hook(diag_response="PASS")
    check("PASS diagnosis keeps original (no revision call)", out is None and len(rev.calls) == 0)

    out, diag, rev = await run_hook(agent="programmer")
    check("non-report agent skipped", out is None and len(diag.calls) == 0)

    out, diag, rev = await run_hook(report="short report")
    check("short report skipped", out is None and len(diag.calls) == 0)

    out, diag, rev = await run_hook(diag_fn_present=False)
    check("missing diagnoser fns skipped", out is None)

    out, diag, rev = await run_hook(rev_response="I fixed it.")
    check("too-short revision rejected, original kept", out is None)

    bot_config._config["reflexion_task_reports"] = False
    out, diag, rev = await run_hook()
    check("config flag off disables the pass", out is None and len(diag.calls) == 0)
    bot_config._config["reflexion_task_reports"] = True

    print()
    print("=" * 72)
    print("3. Autonomous editorial diagnosis")
    print("=" * 72)

    import autonomous_project as ap
    import research_store

    # Monkeypatch DB surfaces
    logged_events = []
    ap._log_event = lambda pid, et, content="", meta=None, **kw: logged_events.append((et, content, meta))
    ap.db_query = lambda *a, **kw: []  # cache miss
    ap._recent_staged_research_drafts = lambda pid, limit=8: [
        {"slug": "hbm-rents", "project_match": True, "updated_at": None},
        {"slug": "other-project-draft", "project_match": False, "updated_at": None},
    ]
    research_store.get_document = lambda slug, include_private=False: {
        "slug": slug, "status": "staged", "title": "HBM and rent concentration",
        "markdown": "# Draft\n" + ("AI memory rents concentrate. " * 50), "updated_at": None,
    }

    diag_calls = []

    async def stub_tick_chat(messages, **kwargs):
        diag_calls.append(kwargs)
        return NOTES

    project = {"id": 42, "title": "T", "topic": "semis", "goal": "Track HBM political economy", "state": "researching"}
    block = await ap._diagnose_staged_drafts_for_tick(project, "deepseek", stub_tick_chat)
    print(f"  injection block:\n{block}")
    print(f"  logged events: {[(e[0], (e[2] or {}).get('slug'), (e[2] or {}).get('verdict')) for e in logged_events]}")
    check("only project-matched staged draft diagnosed", len(diag_calls) == 1)
    check("notes produce injection block with slug anchor", block is not None and "[draft: hbm-rents]" in block)
    check(
        "diagnosis event logged with slug + verdict",
        any(e[0] == "editorial_diagnosis" and (e[2] or {}).get("slug") == "hbm-rents"
            and (e[2] or {}).get("verdict") == "notes" for e in logged_events),
    )

    # Cache hit path: db_query returns a newer diagnosis than the draft
    from datetime import datetime, timezone
    ap.db_query = lambda *a, **kw: [{"content": NOTES, "created_at": datetime.now(timezone.utc)}]
    research_store.get_document = lambda slug, include_private=False: {
        "slug": slug, "status": "staged", "title": "t",
        "markdown": "body", "updated_at": datetime(2020, 1, 1, tzinfo=timezone.utc),
    }
    diag_calls.clear()
    block2 = await ap._diagnose_staged_drafts_for_tick(project, "deepseek", stub_tick_chat)
    check("unchanged draft reuses cached diagnosis (no LLM call)", block2 is not None and len(diag_calls) == 0)

    # PASS diagnosis → no injection
    ap.db_query = lambda *a, **kw: []
    research_store.get_document = lambda slug, include_private=False: {
        "slug": slug, "status": "staged", "title": "t", "markdown": "body", "updated_at": None,
    }

    async def pass_chat(messages, **kwargs):
        return "PASS"

    logged_events.clear()
    block3 = await ap._diagnose_staged_drafts_for_tick(project, "deepseek", pass_chat)
    check(
        "PASS diagnosis logs event but injects nothing",
        block3 is None and any(e[0] == "editorial_diagnosis" and (e[2] or {}).get("verdict") == "pass" for e in logged_events),
    )

    bot_config._config["reflexion_autonomous_publish"] = False
    block4 = await ap._diagnose_staged_drafts_for_tick(project, "deepseek", stub_tick_chat)
    check("config flag off disables autonomous diagnosis", block4 is None)
    bot_config._config["reflexion_autonomous_publish"] = True

    print()
    print("=" * 72)
    print("4. Prompt injection in _build_task_prompt")
    print("=" * 72)

    ap._recent_notes = lambda p: []
    ap._synthesis_context = lambda pid: (None, None)
    ap._recent_tick_attention_events = lambda pid: []
    ap._fetch_last_tick_tool_log = lambda pid: None
    ap._recent_staged_research_drafts = lambda pid, limit=8: []

    proj = {"id": 42, "title": "T", "topic": "semis", "goal": "G", "state": "researching", "plan": None, "turn_count": 3}
    xml_prompt = ap._build_task_prompt(proj, 20, provider="claude", editorial_diagnosis="[draft: hbm-rents]\n" + NOTES)
    md_prompt = ap._build_task_prompt(proj, 20, provider="openai", editorial_diagnosis="[draft: hbm-rents]\n" + NOTES)
    none_prompt = ap._build_task_prompt(proj, 20, provider="claude", editorial_diagnosis=None)
    idx_staged = xml_prompt.find("<staged-research-drafts>")
    idx_diag = xml_prompt.find("<editorial-diagnosis>")
    print(f"  xml block present at {idx_diag} (staged at {idx_staged})")
    check("xml prompt injects <editorial-diagnosis> after staged drafts", 0 < idx_staged < idx_diag)
    check("xml guidance names stage_public workflow", "stage_public" in xml_prompt)
    check("markdown prompt injects section", "### Editorial Diagnosis (staged drafts)" in md_prompt)
    check("no diagnosis -> no block", "<editorial-diagnosis>" not in none_prompt)

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
