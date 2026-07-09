"""Smoke test for the 2026-07-09 autonomous-loop research upgrades:

1. Research trail — cross-tick query memory (_extract_research_queries / _research_trail)
2. research_deep_dive — bounded read-only analyst sub-call (_build_deep_dive_tool)
3. Stall auto-pause — Phase-4 enforcement (_stall_streak_reached / _maybe_auto_pause_stalled_project)
4. Tick notification carries the critic verdict (_notify_telegram / _review_reason)

Hermetic: stub chat fns, monkeypatched DB/event/telegram functions — no LLM,
no Postgres, no network sends. Prints every intermediate result in full.

Run:  venv/bin/python scripts/smoke_autonomous_research.py
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


PROJECT = {"id": 7, "title": "반도체 정치경제", "topic": "semis", "goal": "HBM 추적", "state": "researching"}

LOG_A = (
    '  [1] web_search({"query": "HBM4 SK hynix roadmap", "topic": "news"}) → 10 results...\n'
    '  [2] fetch_url({"url": "https://example.com/a"}) → Long article text\nwith embedded newlines\n'
    '  [3] add_research_note({"text": "finding..."}) → ok: finding note saved\n'
)
LOG_B = (
    '  [1] web_search({"query": "HBM4 SK hynix roadmap", "topic": "news"}) → same query repeated\n'
    '  [2] vector_search({"query": "제국주의론 독점"}) → 5 chunks\n'
    '  [final] knowledge_graph_search({"query": "TSMC"}) → nodes...\n'
    '  [2] revise_plan({"rationale": "x"}) → ok\n'
)
LOG_TRICKY = (
    '  [1] fetch_url({"url": "https://example.com/weird) → looking"}) → body says [9] web_search(fake) → nope\n'
    "  [2] get_finance_data({\"symbol\": \"" + "K" * 300 + "\"}) → data\n"
)


async def main():
    import autonomous_project as ap

    print("=" * 72)
    print("1. Research trail — _extract_research_queries")
    print("=" * 72)

    lines = ap._extract_research_queries([LOG_A, LOG_B])
    print("  extracted:")
    for line in lines:
        print(f"    {line}")
    check("research tools extracted", any(line.startswith("web_search(") for line in lines))
    check("state tools excluded", not any("add_research_note" in line or "revise_plan" in line for line in lines))
    check("duplicate query deduped", sum(1 for line in lines if "HBM4 SK hynix" in line) == 1)
    check("[final] round tag parsed", any(line.startswith("knowledge_graph_search(") for line in lines))
    check("order oldest-first", lines[0].startswith("web_search(") and lines[-1].startswith("vector_search(") is False)

    tricky = ap._extract_research_queries([LOG_TRICKY])
    print("  tricky extracted:")
    for line in tricky:
        print(f"    {line}")
    check("non-greedy args stop at first ') → '", any(line == 'fetch_url({"url": "https://example.com/weird)' for line in tricky))
    check(
        "fake in-result head not top-level",
        # The embedded `[9] web_search(fake) → nope` sits mid-line, not at a line start.
        not any("fake" in line for line in tricky),
    )
    long_line = next((line for line in tricky if line.startswith("get_finance_data(")), "")
    check("args capped", len(long_line) <= len("get_finance_data()") + ap.RESEARCH_TRAIL_ARGS_CAP + 2, f"len={len(long_line)}")

    many = ap._extract_research_queries(
        ['  [1] web_search({"query": "q%d"}) → r\n' % i for i in range(60)]
    )
    check("line cap keeps newest", len(many) == ap.RESEARCH_TRAIL_MAX_LINES and "q59" in many[-1] and "q0" not in many[0])

    print()
    print("2. Research trail — _research_trail (DB stubbed)")
    print("=" * 72)

    orig_db_query = ap.db_query

    def fake_db_query_rows(rows):
        def q(sql, params=None):
            return rows
        return q

    # Newest-first rows as the SQL would return them; newest must be skipped.
    ap.db_query = fake_db_query_rows([
        {"content": '  [1] web_search({"query": "NEWEST — must be skipped"}) → r'},
        {"content": LOG_B},
        {"content": LOG_A},
    ])
    trail = ap._research_trail(7)
    print(f"  trail:\n{trail}")
    check("newest log skipped", trail is not None and "NEWEST" not in trail)
    check("older logs mined oldest-first", trail is not None and trail.splitlines()[0].startswith("- web_search"))

    ap.db_query = fake_db_query_rows([{"content": LOG_A}])
    check("single log → no trail", ap._research_trail(7) is None)

    def raising_query(sql, params=None):
        raise RuntimeError("db down")
    ap.db_query = raising_query
    check("db failure degrades to None", ap._research_trail(7) is None)
    ap.db_query = orig_db_query

    print()
    print("3. Stall streak — _stall_streak_reached")
    print("=" * 72)

    check("3 stalls → pause", ap._stall_streak_reached(["stall", "stall", "stall"]))
    check("2 stalls → no pause", not ap._stall_streak_reached(["stall", "stall"]))
    check("recent ok breaks streak", not ap._stall_streak_reached(["ok", "stall", "stall", "stall"]))
    check("old ok beyond window ignored", ap._stall_streak_reached(["stall", "stall", "stall", "ok"]))
    check("empty → no pause", not ap._stall_streak_reached([]))

    print()
    print("4. Auto-pause — _maybe_auto_pause_stalled_project (DB/telegram stubbed)")
    print("=" * 72)

    executed = []
    events = []
    alerts = []
    orig_execute, orig_log_event, orig_send = ap.db_execute, ap._log_event, ap._send_owner_telegram
    ap.db_execute = lambda sql, params=None: executed.append((sql, params))
    ap._log_event = lambda pid, et, content="", meta=None, **kw: events.append((et, content, meta))

    async def fake_send(text):
        alerts.append(text)
        return True

    ap._send_owner_telegram = fake_send

    ap._recent_stall_signals = lambda pid, limit=ap.STALL_AUTO_PAUSE_STREAK: ["stall", "stall", "stall"]
    paused = await ap._maybe_auto_pause_stalled_project(PROJECT, {"verdict": "no-op", "review": "VERDICT: no-op"})
    print(f"  paused={paused}; executed={len(executed)}; events={[e[0] for e in events]}")
    print(f"  alert: {alerts[0][:150] if alerts else '(none)'}")
    check("streak + no-op → paused", paused)
    check("state UPDATE issued", any("SET state" in sql for sql, _ in executed))
    check("state_transition event with auto meta", any(et == "state_transition" and (meta or {}).get("auto") for et, _, meta in events))
    check("owner alerted", len(alerts) == 1 and "자동 일시정지" in alerts[0])

    executed.clear(); events.clear(); alerts.clear()
    paused = await ap._maybe_auto_pause_stalled_project(PROJECT, {"verdict": "advanced", "review": ""})
    check("advanced verdict → never pauses", not paused and not executed and not alerts)

    paused = await ap._maybe_auto_pause_stalled_project(PROJECT, None)
    check("no review → never pauses", not paused)

    ap._recent_stall_signals = lambda pid, limit=ap.STALL_AUTO_PAUSE_STREAK: ["stall", "ok", "stall"]
    paused = await ap._maybe_auto_pause_stalled_project(PROJECT, {"verdict": "no-op", "review": ""})
    check("broken streak → no pause", not paused and not executed)

    def raising_execute(sql, params=None):
        raise RuntimeError("db down")
    ap._recent_stall_signals = lambda pid, limit=ap.STALL_AUTO_PAUSE_STREAK: ["stall", "stall", "stall"]
    ap.db_execute = raising_execute
    paused = await ap._maybe_auto_pause_stalled_project(PROJECT, {"verdict": "no-op", "review": ""})
    check("db failure on pause degrades to False", not paused)

    ap.db_execute, ap._log_event, ap._send_owner_telegram = orig_execute, orig_log_event, orig_send

    print()
    print("5. research_deep_dive — _build_deep_dive_tool")
    print("=" * 72)

    dd_events = []
    ap._log_event = lambda pid, et, content="", meta=None, **kw: dd_events.append((et, content, meta))

    chat_calls = []

    async def fake_chat(messages, **kwargs):
        chat_calls.append((messages, kwargs))
        if isinstance(kwargs.get("budget_tracker"), dict):
            kwargs["budget_tracker"]["total_cost"] = 0.123
        return "MINI REPORT: HBM4 대량 양산은 2026년 하반기.\n\nSOURCES:\n- https://example.com/a"

    schemas, handlers = ap._build_deep_dive_tool(7, "deepseek", fake_chat)
    schema = schemas[0]
    print(f"  schema name={schema['name']}, required={schema['input_schema']['required']}")
    check("schema name", schema["name"] == "research_deep_dive")
    check("question required", schema["input_schema"]["required"] == ["question"])

    handler = handlers["research_deep_dive"]
    r = await handler(question="")
    check("empty question rejected", r.startswith("error:"), r)

    r1 = await handler(question="HBM4 양산 시점은?", context="프로젝트: 반도체", budget_usd=9.9)
    print(f"  result1 (trunc): {r1[:200]}")
    check("mini-report returned with footer", "MINI REPORT" in r1 and "add_research_note" in r1 and "$0.123" in r1)
    _, kw = chat_calls[0]
    check("budget clamped to cap", kw["budget_usd"] == ap.DEEP_DIVE_BUDGET_CAP, f"budget={kw['budget_usd']}")
    check("rounds bounded", kw["max_rounds"] == ap.DEEP_DIVE_MAX_ROUNDS)
    sub_tool_names = {t["name"] for t in kw["extra_tools"]}
    print(f"  sub-agent tools: {sorted(sub_tool_names)}")
    check("sub-tools ⊆ read-only allowlist", sub_tool_names <= ap._DEEP_DIVE_READONLY_TOOLS, str(sub_tool_names))
    check(
        "no file/publish surface leaked",
        not ({"read_file", "search_files", "list_directory", "research_document", "edit_content", "write_kg_structured", "save_finding", "mission", "download_file", "convert_document"} & sub_tool_names),
    )
    check("handlers match tools", set(kw["extra_handlers"]) == sub_tool_names)
    check("attributed as analyst/autonomous", kw["agent_name"] == "analyst" and kw["runtime_kind"] == "autonomous")
    check("deep_dive event logged", any(et == "deep_dive" for et, _, _ in dd_events))

    r2 = await handler(question="두 번째 질문")
    check("second call allowed", "MINI REPORT" in r2)
    r3 = await handler(question="세 번째 질문")
    check("third call blocked by per-tick cap", r3.startswith("error:") and "capped" in r3, r3)

    async def broken_chat(messages, **kwargs):
        raise RuntimeError("provider down")

    schemas2, handlers2 = ap._build_deep_dive_tool(7, "deepseek", broken_chat)
    r = await handlers2["research_deep_dive"](question="q")
    check("chat failure degrades to error string", r.startswith("error: deep dive failed"), r)

    long_chat_calls = []

    async def long_chat(messages, **kwargs):
        long_chat_calls.append(1)
        return "X" * 9000

    schemas3, handlers3 = ap._build_deep_dive_tool(7, "deepseek", long_chat)
    r = await handlers3["research_deep_dive"](question="q")
    check("oversized report truncated", "[... truncated]" in r and len(r) < 4600, f"len={len(r)}")
    ap._log_event = orig_log_event

    print()
    print("6. Notification — critic verdict line + _review_reason")
    print("=" * 72)

    check("_review_reason extracts", ap._review_reason("VERDICT: no-op\nREASON: 노트가 기존 내용 반복") == "노트가 기존 내용 반복")
    check("_review_reason empty on missing", ap._review_reason("VERDICT: advanced") == "")

    sent = []

    async def capture_send(text):
        sent.append(text)
        return True

    ap._send_owner_telegram = capture_send
    await ap._notify_telegram(
        PROJECT, "본문.\n\n자가비평 문단.",
        {"notes": ["새 발견"], "staged_drafts": [], "publications": [], "plan_rationale": None, "state_change": None},
        {"cost_usd": 0.5, "rounds_used": 9},
        tick_review={"verdict": "advanced", "review": "VERDICT: advanced\nREASON: 새 발견 저장됨"},
    )
    print(f"  message:\n{sent[0]}")
    check("critic line present", "[크리틱] advanced — 새 발견 저장됨" in sent[0])

    sent.clear()
    await ap._notify_telegram(
        PROJECT, "본문",
        {"notes": [], "staged_drafts": [], "publications": [], "plan_rationale": None, "state_change": None},
        {"cost_usd": 0.1, "rounds_used": 2},
        tick_review=None,
    )
    check("no critic line without review", "[크리틱]" not in sent[0])
    ap._send_owner_telegram = orig_send

    print()
    print("7. Prompt assembly — research trail block in both renderers")
    print("=" * 72)

    orig = {}
    for name in ("_recent_notes", "_recent_staged_research_drafts", "_recent_tick_attention_events",
                 "_synthesis_context", "_research_trail", "_fetch_last_tick_tool_log"):
        orig[name] = getattr(ap, name)
    ap._recent_notes = lambda p: []
    ap._recent_staged_research_drafts = lambda pid: []
    ap._recent_tick_attention_events = lambda pid: []
    ap._synthesis_context = lambda pid: (None, None)
    ap._research_trail = lambda pid: "- web_search({\"query\": \"q1\"})\n- fetch_url({\"url\": \"u\"})"
    ap._fetch_last_tick_tool_log = lambda pid: None

    xml_prompt = ap._build_task_prompt(PROJECT, 20, provider="claude")
    md_prompt = ap._build_task_prompt(PROJECT, 20, provider="deepseek")
    check("xml renderer has <research-trail>", "<research-trail>" in xml_prompt and "q1" in xml_prompt)
    check("md renderer has Research Trail", "### Research Trail" in md_prompt and "q1" in md_prompt)

    ap._research_trail = lambda pid: None
    xml_prompt = ap._build_task_prompt(PROJECT, 20, provider="claude")
    check("block omitted when no trail", "<research-trail>" not in xml_prompt)
    for name, fn in orig.items():
        setattr(ap, name, fn)

    print()
    print("=" * 72)
    print(f"TOTAL: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED CHECKS:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
