"""Smoke test for Phase-5 experiential memory extension.

Covers: shared recall helper, task-worker + autonomous-tick injection,
failure write-back hooks, dedupe plumbing. Hermetic — embeddings, DB, and
search are stubbed. Prints full intermediate output.

Run:  venv/bin/python scripts/smoke_experience_recall.py
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


FIXTURE_ROWS = [
    {"category": "mistake", "content": "Verifier failed an analyst task that cited numbers without sources.", "similarity": 0.8},
    {"category": "lesson", "content": "KITA customs data is the reliable source for Korean export figures.", "similarity": 0.7},
]


async def main():
    import memory_store.experiential as exp

    print("=" * 72)
    print("1. Shared recall helper (recall_experiences_block)")
    print("=" * 72)

    orig_search = exp.search_experiential_memory
    exp.search_experiential_memory = lambda q, k=5: FIXTURE_ROWS

    xml_block = exp.recall_experiences_block("analyze exports", "claude", 3)
    md_block = exp.recall_experiences_block("analyze exports", "deepseek", 3)
    print(f"  xml block:\n{xml_block}")
    print(f"  md head: {md_block.splitlines()[0]!r}")
    check(
        "claude gets <past-experiences> with categories",
        xml_block.startswith("<past-experiences>") and "[mistake]" in xml_block and "[lesson]" in xml_block,
    )
    check("non-claude gets markdown section", md_block.startswith("### Past Experiences"))

    exp.search_experiential_memory = lambda q, k=5: []
    check("no relevant memories -> empty string", exp.recall_experiences_block("x", "claude") == "")

    def boom(q, k=5):
        raise RuntimeError("embedding server down")
    exp.search_experiential_memory = boom
    check("search failure -> empty string, no raise", exp.recall_experiences_block("x", "claude") == "")
    exp.search_experiential_memory = orig_search

    print()
    print("=" * 72)
    print("2. Chat loop delegates to the shared helper")
    print("=" * 72)

    import telegram.commands as commands
    exp.search_experiential_memory = lambda q, k=5: FIXTURE_ROWS
    chat_block = await commands._fetch_relevant_experiences("analyze exports", "claude")
    check("chat block identical to shared helper output", chat_block == xml_block)
    exp.search_experiential_memory = orig_search

    print()
    print("=" * 72)
    print("3. Dedupe plumbing (save_experiential_memory)")
    print("=" * 72)

    import db
    # Import BEFORE patching is_duplicate_experience: experience_writer binds
    # the function at import time (from-import alias).
    import experience_writer
    check("experience_writer shares the canonical dedupe", experience_writer._is_duplicate is exp.is_duplicate_experience)

    class FakeEmb:
        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    executed = []
    orig_get_emb = exp._get_exp_embeddings
    orig_execute = db.execute
    orig_dup = exp.is_duplicate_experience
    exp._get_exp_embeddings = lambda: FakeEmb()
    db.execute = lambda sql, params=None: executed.append((" ".join(sql.split()), params))

    exp.is_duplicate_experience = lambda e, threshold=0.85: True
    ok = exp.save_experiential_memory("repeat lesson", "mistake", "task_verification", dedupe=True)
    check("duplicate + dedupe=True -> skipped, no insert", ok is False and len(executed) == 0)

    ok = exp.save_experiential_memory("repeat lesson", "mistake", "task_verification")
    check("dedupe=False keeps legacy behavior (insert)", ok is True and len(executed) == 1)

    exp.is_duplicate_experience = lambda e, threshold=0.85: False
    executed.clear()
    ok = exp.save_experiential_memory("fresh lesson", "mistake", "autonomous_tick", dedupe=True)
    print(f"  insert params: {executed[0][1][:3] if executed else None}")
    check(
        "fresh + dedupe=True -> inserted with category/source",
        ok is True and executed and executed[0][1][1] == "mistake" and executed[0][1][2] == "autonomous_tick",
    )

    exp._get_exp_embeddings = orig_get_emb
    db.execute = orig_execute
    exp.is_duplicate_experience = orig_dup

    print()
    print("=" * 72)
    print("4. Task worker injection (_build_task_context_content)")
    print("=" * 72)

    import telegram.tasks as tasks

    exp.recall_experiences_block = lambda q, p="claude", k=3: (
        f"<past-experiences>\n- [mistake] stub for query {q[:40]!r}\n</past-experiences>"
    )
    task = {"id": 7, "user_id": 0, "content": "", "parent_task_id": None, "mission_id": None, "agent_type": "analyst"}
    built = tasks._build_task_context_content(task, "Analyze June semiconductor exports.", context_provider="claude")
    print(f"  built content:\n{built[:400]}")
    check("task prompt contains past-experiences block", "<past-experiences>" in built)
    check("task content preserved inside <task>", "Analyze June semiconductor exports." in built)

    def recall_boom(q, p="claude", k=3):
        raise RuntimeError("down")
    exp.recall_experiences_block = recall_boom
    built = tasks._build_task_context_content(task, "Analyze X.", context_provider="claude")
    check("recall failure degrades to no block", "<past-experiences>" not in built and "Analyze X." in built)

    print()
    print("=" * 72)
    print("5. Autonomous tick injection (_build_task_prompt)")
    print("=" * 72)

    import autonomous_project as ap

    ap._recent_notes = lambda p: []
    ap._synthesis_context = lambda pid: (None, None)
    ap._recent_tick_attention_events = lambda pid: []
    ap._fetch_last_tick_tool_log = lambda pid: None
    ap._recent_staged_research_drafts = lambda pid, limit=8: []

    recall_queries = []

    def stub_recall(q, p="claude", k=3):
        recall_queries.append(q)
        return "<past-experiences>\n- [mistake] stub tick lesson\n</past-experiences>"

    exp.recall_experiences_block = stub_recall
    proj = {"id": 42, "title": "HBM watch", "topic": "semis", "goal": "Track HBM political economy",
            "state": "researching", "plan": None, "turn_count": 3}
    prompt = ap._build_task_prompt(proj, 20, provider="claude")
    idx_exp = prompt.find("<past-experiences>")
    idx_warn = prompt.find("<recent-tick-warnings>")
    print(f"  recall query: {recall_queries[0]!r}")
    print(f"  block at {idx_exp}, warnings at {idx_warn}")
    check("tick prompt contains past-experiences before warnings", 0 < idx_exp < idx_warn)
    check("recall keyed on title+topic+goal", "HBM watch" in recall_queries[0] and "Track HBM political economy" in recall_queries[0])

    exp.recall_experiences_block = recall_boom
    prompt = ap._build_task_prompt(proj, 20, provider="claude")
    check("tick recall failure degrades to no block", "<past-experiences>" not in prompt)

    print()
    print("=" * 72)
    print("6. Failure write-back hooks")
    print("=" * 72)

    saved = []
    exp.save_experiential_memory = lambda content, category, source_type, dedupe=False: saved.append(
        {"content": content, "category": category, "source_type": source_type, "dedupe": dedupe}
    ) or True

    tasks._record_failure_experience("[analyst] Task failed independent verification. Task: X | Verifier: bad numbers", "task_verification")
    print(f"  task hook saved: {saved[-1]}")
    check(
        "verification hook writes deduped mistake row",
        saved[-1]["category"] == "mistake" and saved[-1]["source_type"] == "task_verification" and saved[-1]["dedupe"] is True,
    )

    reason = tasks._verifier_reason_head(
        "task_report: ok — summary extracted\nllm_verification: failed — claimed file was never modified"
    )
    print(f"  extracted reason: {reason!r}")
    check("verifier reason extraction prefers llm line", reason.startswith("llm_verification: failed"))
    check("reason extraction handles empty details", tasks._verifier_reason_head("") == "(no details)")

    ap._record_tick_experience("[autonomous] Project 'HBM watch': tick aimed at 'X' but produced no real progress")
    print(f"  tick hook saved: {saved[-1]}")
    check(
        "tick hook writes deduped autonomous_tick row",
        saved[-1]["source_type"] == "autonomous_tick" and saved[-1]["dedupe"] is True,
    )

    def save_boom(content, category, source_type, dedupe=False):
        raise RuntimeError("db down")
    exp.save_experiential_memory = save_boom
    try:
        tasks._record_failure_experience("x", "task_verification")
        ap._record_tick_experience("y")
        check("hooks never raise on save failure", True)
    except Exception as e:
        check("hooks never raise on save failure", False, str(e))

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
