"""Smoke test for Phase-3 Plan-and-Execute dependency DAGs in multi_delegate.

Hermetic: task creation, DB queries, and chat history are stubbed — no
Postgres, no LLM. Prints full intermediate output.

Run:  venv/bin/python scripts/smoke_plan_dag.py
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


def staged_tasks():
    return [
        {"agent": "scout", "task": "Collect June HBM export coverage from Korean press."},
        {"agent": "analyst", "task": "Analyze the collected coverage for rent-concentration signals.", "depends_on": [0]},
        {"agent": "analyst", "task": "Draft a research document from the analysis.", "depends_on": [1]},
    ]


async def main():
    import db
    import task_store
    import self_runtime.tools as st
    import telegram.tasks as tasks
    import memory_store.experiential as exp
    from prompt_context import format_dependency_results

    # ── Stub the world ───────────────────────────────────────────────
    created = []
    next_id = {"v": 100}

    def fake_create(content, user_id=0, priority="normal", **kw):
        if kw.get("agent_type") == "failagent":
            return {"status": "error", "error": "boom"}
        tid = next_id["v"]
        next_id["v"] += 1
        created.append({"task_id": tid, "content": content, **kw})
        return {"status": "ok", "task_id": tid}

    executed = []
    task_store.create_task_in_db = fake_create
    db.execute = lambda sql, params=None: executed.append((" ".join(sql.split()), params))
    db.query = lambda sql, params=None: []
    st._resolve_recent_operator_user_id = lambda: None

    print("=" * 72)
    print("1. Plan validation guards")
    print("=" * 72)

    out = await st._exec_multi_delegate([
        {"agent": "scout", "task": "a", "depends_on": [1]},
        {"agent": "analyst", "task": "b"},
    ])
    print(f"  forward ref -> {out[:100]}")
    check("forward reference rejected", "invalid depends_on" in out)

    out = await st._exec_multi_delegate([
        {"agent": "scout", "task": "a"},
        {"agent": "analyst", "task": "b", "depends_on": [-1]},
    ])
    check("negative index rejected", "invalid depends_on" in out)

    out = await st._exec_multi_delegate([{"agent": "analyst", "task": f"t{i}"} for i in range(9)])
    print(f"  9 tasks -> {out[:80]}")
    check("plan size capped at 8", "at most 8 tasks" in out)

    print()
    print("=" * 72)
    print("2. Staged plan creation")
    print("=" * 72)

    created.clear()
    executed.clear()
    out = await st._exec_multi_delegate(staged_tasks(), synthesis_instructions="Combine into one verdict.")
    print(f"  result message:\n{out}")
    subtasks = [c for c in created if c.get("plan_role") == "subtask"]
    synthesis = [c for c in created if c.get("plan_role") == "synthesis"]
    print(f"  created: {[(c['task_id'], c.get('status'), (c.get('metadata') or {}).get('depends_on_task_ids')) for c in created]}")

    check("three subtasks + one synthesis created", len(subtasks) == 3 and len(synthesis) == 1)
    check("stage 1 starts pending", subtasks[0].get("status") == "pending" and "metadata" in subtasks[0] and not (subtasks[0].get("metadata") or {}).get("depends_on_task_ids"))
    check(
        "stage 2 blocked on stage 1's real task id",
        subtasks[1].get("status") == "blocked"
        and (subtasks[1]["metadata"] or {}).get("depends_on_task_ids") == [subtasks[0]["task_id"]],
    )
    check(
        "stage 3 blocked on stage 2's real task id",
        subtasks[2].get("status") == "blocked"
        and (subtasks[2]["metadata"] or {}).get("depends_on_task_ids") == [subtasks[1]["task_id"]],
    )
    check("synthesis created blocked", synthesis[0].get("status") == "blocked")
    check("result message marks plan as staged", "staged subtasks" in out and "blocked until #" in out)

    created.clear()
    parallel_out = await st._exec_multi_delegate([
        {"agent": "scout", "task": "a"},
        {"agent": "analyst", "task": "b"},
    ])
    check(
        "pure parallel plan unchanged (all pending, message says parallel)",
        all(c.get("status") == "pending" for c in created if c.get("plan_role") == "subtask")
        and "parallel subtasks" in parallel_out,
    )

    print()
    print("=" * 72)
    print("3. Failed dependency creation skips dependents")
    print("=" * 72)

    created.clear()
    out = await st._exec_multi_delegate([
        {"agent": "failagent", "task": "will fail"},
        {"agent": "analyst", "task": "depends on the failure", "depends_on": [0]},
        {"agent": "scout", "task": "independent"},
    ])
    # failagent isn't in _DELEGATABLE_AGENTS -> whole call rejected earlier; use monkeypatched failure instead
    print(f"  (agent-validation path) -> {out[:90]}")
    check("unknown agent still rejected upfront", "Cannot delegate" in out)

    orig_create = task_store.create_task_in_db

    def failing_first(content, user_id=0, priority="normal", **kw):
        if "will fail" in content:
            return {"status": "error", "error": "db down"}
        return fake_create(content, user_id, priority, **kw)

    task_store.create_task_in_db = failing_first
    created.clear()
    out = await st._exec_multi_delegate([
        {"agent": "scout", "task": "will fail"},
        {"agent": "analyst", "task": "depends on the failure", "depends_on": [0]},
        {"agent": "analyst", "task": "independent"},
    ])
    print(f"  result message:\n{out}")
    check("dependent of failed creation skipped", "SKIPPED [analyst]" in out)
    check("independent task still created", any("independent" in c["content"] for c in created))
    task_store.create_task_in_db = orig_create

    print()
    print("=" * 72)
    print("4. Worker unblock pass (_unblock_dependency_tasks_sync)")
    print("=" * 72)

    def run_unblock(blocked_rows, blocking_count):
        calls = {"selects": [], "updates": []}

        def q(sql, params=None):
            calls["selects"].append((sql, params))
            if "COUNT(*)" in sql:
                return [{"c": blocking_count}]
            return blocked_rows

        def e(sql, params=None):
            calls["updates"].append((" ".join(sql.split()), params))

        tasks._query = q
        tasks._execute = e
        tasks._unblock_dependency_tasks_sync()
        return calls

    row = {"id": 200, "metadata": '{"depends_on_task_ids": [101, 102]}'}
    calls = run_unblock([row], blocking_count=0)
    unblocks = [u for u in calls["updates"] if "SET status = 'pending'" in u[0]]
    print(f"  deps terminal -> updates: {[u[0][:60] for u in calls['updates']]}")
    check("all deps terminal -> task unblocked", len(unblocks) == 1 and unblocks[0][1] == (200,))

    calls = run_unblock([row], blocking_count=1)
    unblocks = [u for u in calls["updates"] if "SET status = 'pending'" in u[0]]
    check("pending dep -> stays blocked", len(unblocks) == 0)

    calls = run_unblock([], blocking_count=0)
    watchdogs = [u for u in calls["updates"] if "WATCHDOG" in str(u[1])]
    check("watchdog failsafe always issued", len(watchdogs) == 1 and "48" in str(watchdogs[0][1][0]))

    print()
    print("=" * 72)
    print("5. Dependency-results injection")
    print("=" * 72)

    dep_rows = [
        {"id": 101, "agent_type": "scout", "content": "Collect coverage", "result": "Found 12 articles on HBM.", "status": "done"},
        {"id": 102, "agent_type": "analyst", "content": "Analyze", "result": "provider crashed", "status": "failed"},
    ]
    xml = format_dependency_results(dep_rows, "claude")
    md = format_dependency_results(dep_rows, "openai")
    print(f"  xml block:\n{xml[:500]}")
    check("xml block carries ids + statuses", '<dependency id="101" agent="scout" status="done">' in xml and 'status="failed"' in xml)
    check("failure-handling guidance present", "report the blockage" in xml)
    check("markdown variant renders", "#### Dependency #101 [scout] — done" in md)

    exp.recall_experiences_block = lambda q, p="claude", k=3: ""
    tasks._query = lambda sql, params=None: dep_rows if "id = ANY" in sql else []
    task = {"id": 300, "user_id": 0, "content": "", "parent_task_id": None, "mission_id": None,
            "agent_type": "analyst", "plan_id": 100, "plan_role": "subtask",
            "metadata": '{"depends_on_task_ids": [101, 102]}'}
    built = tasks._build_task_context_content(task, "Analyze the collected coverage.", context_provider="claude")
    print(f"  built head:\n{built[:300]}")
    check("subtask prompt contains <dependency-results>", "<dependency-results>" in built)
    check("task text preserved", "Analyze the collected coverage." in built)

    tasks._query = lambda sql, params=None: (_ for _ in ()).throw(RuntimeError("db down")) if "id = ANY" in sql else []
    built = tasks._build_task_context_content(task, "Analyze X.", context_provider="claude")
    check("injection failure degrades gracefully", "<dependency-results>" not in built and "Analyze X." in built)

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
