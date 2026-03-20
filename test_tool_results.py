"""Unit tests for sanitize_messages (consolidated tool_result handler).

Tests claude_loop.sanitize_messages directly (no DB/API dependencies needed).
"""
import logging

logger = logging.getLogger("test_tool_results")
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

from claude_loop import sanitize_messages as _sanitize_messages, _prepare_messages_for_api


# ── Helpers ──

def _make_assistant_tool_use(tool_id: str, name: str = "test_tool") -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Calling tool..."},
            {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
        ],
    }

def _make_user_tool_result(tool_id: str, result: str = "ok") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": result},
        ],
    }

def _make_assistant_server_tool(server_id: str, name: str = "web_search") -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Searching..."},
            {"type": "server_tool_use", "id": server_id, "name": name, "input": {}},
        ],
    }

def _make_assistant_server_resolved(server_id: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Searching..."},
            {"type": "server_tool_use", "id": server_id, "name": "web_search", "input": {}},
            {"type": "web_search_tool_result", "tool_use_id": server_id, "content": []},
        ],
    }

def _make_assistant_mixed(tool_id: str, server_id: str, server_resolved: bool = True) -> dict:
    content = [
        {"type": "text", "text": "Working..."},
        {"type": "server_tool_use", "id": server_id, "name": "web_search", "input": {}},
        {"type": "tool_use", "id": tool_id, "name": "test_tool", "input": {}},
    ]
    if server_resolved:
        content.insert(2, {"type": "web_search_tool_result", "tool_use_id": server_id, "content": []})
    return {"role": "assistant", "content": content}


def count_tool_results(msgs_list, tid):
    count = 0
    for m in msgs_list:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == tid:
                    count += 1
    return count


def _find_unresolved_tool_uses(msgs_list):
    """Return unresolved tool_use ids as [(assistant_index, [ids...]), ...]."""
    unresolved = []
    for i, m in enumerate(msgs_list):
        if m.get("role") != "assistant":
            continue
        c = m.get("content", [])
        if not isinstance(c, list):
            continue
        tool_ids = {
            b.get("id")
            for b in c
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
        }
        if not tool_ids:
            continue
        resolved = set()
        if i + 1 < len(msgs_list) and msgs_list[i + 1].get("role") == "user":
            nc = msgs_list[i + 1].get("content", [])
            if isinstance(nc, list):
                resolved = {
                    b.get("tool_use_id")
                    for b in nc
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                }
        missing = sorted(tid for tid in tool_ids if tid not in resolved)
        if missing:
            unresolved.append((i, missing))
    return unresolved


# ── Tests ──

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            print(f"  ✗ {name}: {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for e in self.errors:
                print(f"  - {e}")
        print(f"{'='*60}")
        return self.failed == 0


T = TestResults()


# ──────────────────────────────────────────────────────────
print("\n=== Test 1: Normal case (no fix needed) ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    _make_user_tool_result("t1"),
    {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
]
result = _sanitize_messages(msgs)
T.check("message count unchanged", len(result) == 4, f"got {len(result)}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 2: Missing tool_result ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    {"role": "assistant", "content": [{"type": "text", "text": "next"}]},
]
result = _sanitize_messages(msgs)
T.check("user message injected", len(result) == 4, f"got {len(result)}")
T.check("injected msg is user", result[2].get("role") == "user")
injected_content = result[2].get("content", [])
T.check("has tool_result", any(
    isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == "t1"
    for b in injected_content
))


# ──────────────────────────────────────────────────────────
print("\n=== Test 3: server_tool_use dummy in assistant ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_server_tool("s1"),
    {"role": "user", "content": [{"type": "text", "text": "continue"}]},
]
result = _sanitize_messages(msgs)
assistant_content = result[1].get("content", [])
has_server_result = any(
    isinstance(b, dict) and b.get("type") == "web_search_tool_result" and b.get("tool_use_id") == "s1"
    for b in assistant_content
)
T.check("server dummy in assistant", has_server_result)
user_content = result[2].get("content", [])
has_server_in_user = any(
    isinstance(b, dict) and b.get("type") == "web_search_tool_result"
    for b in (user_content if isinstance(user_content, list) else [])
)
T.check("no server result in user", not has_server_in_user)


# ──────────────────────────────────────────────────────────
print("\n=== Test 4: Already resolved server ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_server_resolved("s1"),
]
result = _sanitize_messages(msgs)
assistant_content = result[1].get("content", [])
server_results = [b for b in assistant_content if isinstance(b, dict) and b.get("type") == "web_search_tool_result"]
T.check("no duplicate server result", len(server_results) == 1, f"got {len(server_results)}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 5: Mixed tool_use + server_tool_use ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_mixed("t1", "s1", server_resolved=False),
]
result = _sanitize_messages(msgs)
assistant_content = result[1].get("content", [])
has_server = any(isinstance(b, dict) and b.get("type") == "web_search_tool_result" for b in assistant_content)
T.check("server dummy in assistant (mixed)", has_server)
T.check("user msg injected (mixed)", len(result) == 3, f"got {len(result)}")
user_content = result[2].get("content", [])
has_custom = any(isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == "t1" for b in user_content)
T.check("custom dummy in user (mixed)", has_custom, f"user content={user_content}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 6: Mixed with existing user msg ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_mixed("t1", "s1", server_resolved=True),
    _make_user_tool_result("t1"),
]
result = _sanitize_messages(msgs)
T.check("no changes needed (mixed resolved)", len(result) == 3, f"got {len(result)}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 7: Consecutive assistants ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    _make_assistant_tool_use("t2"),
    _make_user_tool_result("t2"),
]
result = _sanitize_messages(msgs)
T.check("user injected between assistants", len(result) == 5, f"got {len(result)}")
T.check("injected at index 2", result[2].get("role") == "user")
injected = result[2].get("content", [])
has_t1_result = any(isinstance(b, dict) and b.get("tool_use_id") == "t1" for b in injected)
T.check("has t1 tool_result", has_t1_result)
T.check("t2 still resolved", result[4].get("role") == "user")


# ──────────────────────────────────────────────────────────
print("\n=== Test 8: Multiple tool_use in one assistant ===")
msgs = [
    {"role": "user", "content": "hello"},
    {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
            {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
        ],
    },
]
result = _sanitize_messages(msgs)
user_content = result[2].get("content", [])
has_t2 = any(isinstance(b, dict) and b.get("tool_use_id") == "t2" for b in user_content)
T.check("missing t2 injected", has_t2)


# ──────────────────────────────────────────────────────────
print("\n=== Test 9: Idempotency ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
]
first_pass = _sanitize_messages(msgs)
second_pass = _sanitize_messages(first_pass)
T.check("idempotent length", len(first_pass) == len(second_pass), f"{len(first_pass)} vs {len(second_pass)}")
T.check("idempotent: no double dummy", count_tool_results(second_pass, "t1") == 1)


# ──────────────────────────────────────────────────────────
print("\n=== Test 10: String content in assistant (from DB) ===")
msgs = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "just text"},
    {"role": "user", "content": "ok"},
]
result = _sanitize_messages(msgs)
T.check("string content skipped", len(result) == 3)


# ──────────────────────────────────────────────────────────
print("\n=== Test 11: Assistant at end (last msg) ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
]
result = _sanitize_messages(msgs)
T.check("user injected at end", len(result) == 3, f"got {len(result)}")
T.check("last msg is user", result[-1].get("role") == "user")


# ──────────────────────────────────────────────────────────
print("\n=== Test 12: User message with no tool_result after tool_use ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    {"role": "user", "content": [{"type": "text", "text": "no tool result here"}]},
]
result = _sanitize_messages(msgs)
user_content = result[2].get("content", [])
has_t1 = any(isinstance(b, dict) and b.get("tool_use_id") == "t1" for b in user_content)
T.check("missing tool_result injected into existing user msg", has_t1, f"content={user_content}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 13: No false positives ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    _make_user_tool_result("t1"),
]
result = _sanitize_messages(msgs)
user_content = result[2].get("content", [])
tool_results = [b for b in user_content if isinstance(b, dict) and b.get("type") == "tool_result"]
T.check("no extra tool_result", len(tool_results) == 1, f"got {len(tool_results)}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 14: Complex multi-round scenario ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_mixed("t1", "s1", server_resolved=False),
    _make_assistant_tool_use("t2"),
    _make_user_tool_result("t2"),
    _make_assistant_server_tool("s2"),
    {"role": "user", "content": [{"type": "text", "text": "continue"}]},
]
result = _sanitize_messages(msgs)

all_tool_use_ids = set()
all_tool_result_ids = set()
all_server_use_ids = set()
all_server_result_ids = set()
for m in result:
    c = m.get("content", [])
    if not isinstance(c, list):
        continue
    for b in c:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "tool_use":
            all_tool_use_ids.add(b["id"])
        elif b.get("type") == "tool_result":
            all_tool_result_ids.add(b.get("tool_use_id"))
        elif b.get("type") == "server_tool_use":
            all_server_use_ids.add(b["id"])
        elif b.get("type") == "web_search_tool_result":
            all_server_result_ids.add(b.get("tool_use_id"))

T.check("all custom resolved", all_tool_use_ids.issubset(all_tool_result_ids),
        f"use={all_tool_use_ids} result={all_tool_result_ids}")
T.check("all server resolved", all_server_use_ids.issubset(all_server_result_ids),
        f"use={all_server_use_ids} result={all_server_result_ids}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 15: tool_use at message boundary (last msg) ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    _make_user_tool_result("t1"),
    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    {"role": "user", "content": "thanks"},
    _make_assistant_tool_use("t2"),
]
result = _sanitize_messages(msgs)
T.check("t2 resolved at boundary", count_tool_results(result, "t2") >= 1)
T.check("last msg is user", result[-1].get("role") == "user")


# ──────────────────────────────────────────────────────────
print("\n=== Test 16: User message with string content after tool_use ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    {"role": "user", "content": "just a string, no tool_result"},
]
result = _sanitize_messages(msgs)
user_content = result[2].get("content", [])
has_t1 = any(isinstance(b, dict) and b.get("tool_use_id") == "t1" for b in user_content)
T.check("string user content wrapped + dummy added", has_t1, f"content={user_content}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 17: Broken history input (non-alternating, malformed mix) ===")
msgs = [
    {"role": "assistant", "content": [{"type": "tool_use", "id": "tb1", "name": "x", "input": {}}]},
    {"role": "assistant", "content": [{"type": "text", "text": "second assistant without user"}]},
    {"role": "user", "content": "late user"},
]
result = _sanitize_messages(msgs)
unresolved = _find_unresolved_tool_uses(result)
T.check("broken history repaired (no unresolved tool_use)", len(unresolved) == 0, f"{unresolved}")
T.check("alternating around first tool_use", result[1].get("role") == "user", f"roles={[m.get('role') for m in result]}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 18: JSON-string content input ===")
msgs = [
    {"role": "user", "content": "hello"},
    {
        "role": "assistant",
        "content": "[{\"type\":\"tool_use\",\"id\":\"tj1\",\"name\":\"json_tool\",\"input\":{}}]",
    },
    {
        "role": "user",
        "content": "[{\"type\":\"text\",\"text\":\"continue from json string\"}]",
    },
]
result = _sanitize_messages(msgs)
json_unresolved = _find_unresolved_tool_uses(result)
T.check("json string parsed and resolved", len(json_unresolved) == 0, f"{json_unresolved}")
json_user_content = result[2].get("content", [])
T.check(
    "tool_result injected into json-string user msg",
    any(isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == "tj1" for b in json_user_content),
    f"content={json_user_content}",
)


# ──────────────────────────────────────────────────────────
print("\n=== Test 19: Orphan tool_result input ===")
msgs = [
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "orphan1", "content": "x"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
]
result = _sanitize_messages(msgs)
first_content = result[0].get("content", [])
if isinstance(first_content, list):
    orphan_still_structured = any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in first_content
    )
else:
    orphan_still_structured = False
T.check("orphan tool_result does not remain as protocol block", not orphan_still_structured, f"content={first_content}")
T.check("no unresolved tool_use introduced by orphan input", len(_find_unresolved_tool_uses(result)) == 0)


# ──────────────────────────────────────────────────────────
print("\n=== Test 20: API payload strips server tool blocks ===")
msgs = [
    {"role": "user", "content": "hello"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "searching + local tool"},
            {"type": "server_tool_use", "id": "srv1", "name": "web_search", "input": {}},
            {"type": "web_search_tool_result", "tool_use_id": "srv1", "content": []},
            {"type": "tool_use", "id": "tc1", "name": "knowledge_graph_search", "input": {"query": "x"}},
        ],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tc1", "content": "ok"}]},
]
prepared = _prepare_messages_for_api(msgs)
assistant_content = prepared[1].get("content", [])
T.check(
    "server blocks removed for API payload",
    not any(isinstance(b, dict) and b.get("type") in ("server_tool_use", "web_search_tool_result") for b in assistant_content),
    f"content={assistant_content}",
)
T.check("custom tool_use preserved", any(isinstance(b, dict) and b.get("type") == "tool_use" for b in assistant_content))
T.check("custom tool_result still paired", len(_find_unresolved_tool_uses(prepared)) == 0, f"{_find_unresolved_tool_uses(prepared)}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 21: API payload preserves server result as text ===")
msgs = [
    {"role": "user", "content": "hello"},
    {
        "role": "assistant",
        "content": [
            {"type": "server_tool_use", "id": "srvx", "name": "web_search", "input": {"query": "gold price drop"}},
            {"type": "web_search_tool_result", "tool_use_id": "srvx", "content": [{"type": "text", "text": "Result A"}, {"type": "text", "text": "Result B"}]},
            {"type": "tool_use", "id": "tx1", "name": "knowledge_graph_search", "input": {"query": "금값 하락 원인"}},
        ],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tx1", "content": "ok"}]},
]
prepared = _prepare_messages_for_api(msgs)
assistant_content = prepared[1].get("content", [])
text_blob = "\n".join(
    b.get("text", "")
    for b in assistant_content
    if isinstance(b, dict) and b.get("type") == "text"
)
T.check("server result text preserved", "Result A" in text_blob and "Result B" in text_blob, f"text={text_blob}")
T.check("server protocol blocks removed", not any(
    isinstance(b, dict) and b.get("type") in ("server_tool_use", "web_search_tool_result")
    for b in assistant_content
))


# ── Summary ──
success = T.summary()
exit(0 if success else 1)
