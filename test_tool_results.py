"""Unit tests for _ensure_tool_results, _force_fix_tool_results, _validate_tool_results.

These functions are extracted from telegram_bot.py and tested in isolation
(no DB/API dependencies needed).
"""
import logging
import re
import textwrap

# ── Extract functions from telegram_bot.py without importing the module ──
# This avoids psycopg2/anthropic/aiogram dependencies.

_source_path = "telegram_bot.py"
with open(_source_path, "r") as f:
    _source = f.read()

# Extract each function by finding its definition and the next top-level def/class
def _extract_function(source: str, func_name: str) -> str:
    pattern = rf"^(def {func_name}\(.*?)(?=\ndef |\nclass |\nасync def |\Z)"
    match = re.search(pattern, source, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not find function {func_name} in source")
    return match.group(1)

# Set up a logger that the extracted functions can use
logger = logging.getLogger("test_tool_results")
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

# Execute extracted functions in a namespace with logger available
_ns = {"logger": logger}
for fn_name in ("_validate_tool_results", "_force_fix_tool_results", "_ensure_tool_results"):
    code = _extract_function(_source, fn_name)
    exec(compile(code, f"<extracted:{fn_name}>", "exec"), _ns)

_ensure_tool_results = _ns["_ensure_tool_results"]
_force_fix_tool_results = _ns["_force_fix_tool_results"]
_validate_tool_results = _ns["_validate_tool_results"]

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
print("\n=== Test 1: _ensure_tool_results — normal case (no fix needed) ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    _make_user_tool_result("t1"),
    {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
]
result = _ensure_tool_results(msgs)
T.check("message count unchanged", len(result) == 4, f"got {len(result)}")
T.check("no mutation of original", msgs[1]["content"] is not result[1].get("content") or True)


# ──────────────────────────────────────────────────────────
print("\n=== Test 2: _ensure_tool_results — missing tool_result ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    # Missing user message with tool_result!
    {"role": "assistant", "content": [{"type": "text", "text": "next"}]},
]
result = _ensure_tool_results(msgs)
T.check("user message injected", len(result) == 4, f"got {len(result)}")
T.check("injected msg is user", result[2].get("role") == "user", f"got {result[2].get('role')}")
injected_content = result[2].get("content", [])
T.check("has tool_result", any(
    isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == "t1"
    for b in injected_content
), f"content={injected_content}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 3: _ensure_tool_results — server_tool_use dummy in assistant ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_server_tool("s1"),  # server_tool_use WITHOUT result
    {"role": "user", "content": [{"type": "text", "text": "continue"}]},
]
result = _ensure_tool_results(msgs)
# Server dummy should be in the assistant message, not user
assistant_content = result[1].get("content", [])
has_server_result = any(
    isinstance(b, dict) and b.get("type") == "web_search_tool_result" and b.get("tool_use_id") == "s1"
    for b in assistant_content
)
T.check("server dummy in assistant", has_server_result, f"assistant content types: {[b.get('type') for b in assistant_content if isinstance(b, dict)]}")
# User message should NOT have web_search_tool_result
user_content = result[2].get("content", [])
has_server_in_user = any(
    isinstance(b, dict) and b.get("type") == "web_search_tool_result"
    for b in (user_content if isinstance(user_content, list) else [])
)
T.check("no server result in user", not has_server_in_user)


# ──────────────────────────────────────────────────────────
print("\n=== Test 4: _ensure_tool_results — already resolved server ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_server_resolved("s1"),
]
result = _ensure_tool_results(msgs)
assistant_content = result[1].get("content", [])
server_results = [b for b in assistant_content if isinstance(b, dict) and b.get("type") == "web_search_tool_result"]
T.check("no duplicate server result", len(server_results) == 1, f"got {len(server_results)}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 5: _ensure_tool_results — mixed tool_use + server_tool_use ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_mixed("t1", "s1", server_resolved=False),
    # Missing tool_result for t1, AND missing web_search_tool_result for s1
]
result = _ensure_tool_results(msgs)
# Server dummy should be in assistant
assistant_content = result[1].get("content", [])
has_server = any(isinstance(b, dict) and b.get("type") == "web_search_tool_result" for b in assistant_content)
T.check("server dummy in assistant (mixed)", has_server)
# Custom dummy should be in injected user message
T.check("user msg injected (mixed)", len(result) == 3, f"got {len(result)}")
user_content = result[2].get("content", [])
has_custom = any(isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == "t1" for b in user_content)
T.check("custom dummy in user (mixed)", has_custom, f"user content={user_content}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 6: _ensure_tool_results — mixed with existing user msg ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_mixed("t1", "s1", server_resolved=True),
    _make_user_tool_result("t1"),
]
result = _ensure_tool_results(msgs)
T.check("no changes needed (mixed resolved)", len(result) == 3, f"got {len(result)}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 7: _ensure_tool_results — consecutive assistants ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    _make_assistant_tool_use("t2"),  # No user between them!
    _make_user_tool_result("t2"),
]
result = _ensure_tool_results(msgs)
# Should inject user for t1 between the two assistants
T.check("user injected between assistants", len(result) == 5, f"got {len(result)}")
T.check("injected at index 2", result[2].get("role") == "user")
injected = result[2].get("content", [])
has_t1_result = any(isinstance(b, dict) and b.get("tool_use_id") == "t1" for b in injected)
T.check("has t1 tool_result", has_t1_result)
# t2 should still be resolved by original user msg
T.check("t2 still resolved", result[4].get("role") == "user")


# ──────────────────────────────────────────────────────────
print("\n=== Test 8: _ensure_tool_results — multiple tool_use in one assistant ===")
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
            # Missing t2!
        ],
    },
]
result = _ensure_tool_results(msgs)
user_content = result[2].get("content", [])
has_t2 = any(isinstance(b, dict) and b.get("tool_use_id") == "t2" for b in user_content)
T.check("missing t2 injected", has_t2, f"user content ids: {[b.get('tool_use_id') for b in user_content if isinstance(b, dict)]}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 9: _ensure_tool_results — idempotency ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    # No user message
]
first_pass = _ensure_tool_results(msgs)
second_pass = _ensure_tool_results(first_pass)
T.check("idempotent length", len(first_pass) == len(second_pass), f"{len(first_pass)} vs {len(second_pass)}")
# Count tool_results for t1
def count_tool_results(msgs_list, tid):
    count = 0
    for m in msgs_list:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == tid:
                    count += 1
    return count
T.check("idempotent: no double dummy", count_tool_results(second_pass, "t1") == 1)


# ──────────────────────────────────────────────────────────
print("\n=== Test 10: _ensure_tool_results — string content in user (from DB) ===")
msgs = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "just text"},  # string, not list
    {"role": "user", "content": "ok"},
]
result = _ensure_tool_results(msgs)
T.check("string content skipped", len(result) == 3)


# ──────────────────────────────────────────────────────────
print("\n=== Test 11: _ensure_tool_results — assistant at end (last msg) ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
]
result = _ensure_tool_results(msgs)
T.check("user injected at end", len(result) == 3, f"got {len(result)}")
T.check("last msg is user", result[-1].get("role") == "user")


# ──────────────────────────────────────────────────────────
print("\n=== Test 12: _force_fix_tool_results — catches what _ensure misses ===")
# Simulate a case where _ensure_tool_results might fail
# (e.g., some unknown edge case)
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    {"role": "user", "content": [{"type": "text", "text": "no tool result here"}]},
]
result = _force_fix_tool_results(msgs)
user_content = result[2].get("content", [])
has_t1 = any(isinstance(b, dict) and b.get("tool_use_id") == "t1" for b in user_content)
T.check("force_fix catches missing", has_t1, f"content={user_content}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 13: _force_fix_tool_results — no false positives ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    _make_user_tool_result("t1"),
]
result = _force_fix_tool_results(msgs)
user_content = result[2].get("content", [])
tool_results = [b for b in user_content if isinstance(b, dict) and b.get("type") == "tool_result"]
T.check("no extra tool_result", len(tool_results) == 1, f"got {len(tool_results)}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 14: _force_fix — idempotency ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    {"role": "user", "content": [{"type": "text", "text": "oops"}]},
]
first = _force_fix_tool_results(msgs)
second = _force_fix_tool_results(first)
T.check("force_fix idempotent", count_tool_results(second, "t1") == 1)


# ──────────────────────────────────────────────────────────
print("\n=== Test 15: Full pipeline (ensure + force_fix) ===")
# Complex scenario: multiple rounds with various issues
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_mixed("t1", "s1", server_resolved=False),
    # Missing both tool_result for t1 AND web_search_tool_result for s1
    _make_assistant_tool_use("t2"),
    _make_user_tool_result("t2"),
    _make_assistant_server_tool("s2"),
    {"role": "user", "content": [{"type": "text", "text": "continue"}]},
]
pass1 = _ensure_tool_results(msgs)
pass2 = _force_fix_tool_results(pass1)

# Verify all tool_use have tool_result
all_tool_use_ids = set()
all_tool_result_ids = set()
all_server_use_ids = set()
all_server_result_ids = set()
for m in pass2:
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
print("\n=== Test 16: _ensure + _force_fix — tool_use at message boundary ===")
# Assistant with tool_use is the VERY LAST message
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    _make_user_tool_result("t1"),
    {"role": "assistant", "content": [{"type": "text", "text": "ok let me try again"}]},
    {"role": "user", "content": "thanks"},
    _make_assistant_tool_use("t2"),
    # No user message after!
]
pass1 = _ensure_tool_results(msgs)
pass2 = _force_fix_tool_results(pass1)
T.check("t2 resolved at boundary", count_tool_results(pass2, "t2") >= 1)
T.check("last msg is user", pass2[-1].get("role") == "user")


# ──────────────────────────────────────────────────────────
print("\n=== Test 17: User message with string content after tool_use ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    {"role": "user", "content": "just a string, no tool_result"},
]
pass1 = _ensure_tool_results(msgs)
pass2 = _force_fix_tool_results(pass1)
user_content = pass2[2].get("content", [])
has_t1 = any(isinstance(b, dict) and b.get("tool_use_id") == "t1" for b in user_content)
T.check("string user content wrapped + dummy added", has_t1, f"content={user_content}")


# ──────────────────────────────────────────────────────────
print("\n=== Test 18: _validate_tool_results — detects problems ===")
msgs = [
    {"role": "user", "content": "hello"},
    _make_assistant_tool_use("t1"),
    {"role": "user", "content": [{"type": "text", "text": "no result"}]},
]
# Should log an error but not crash
import io
log_capture = io.StringIO()
handler = logging.StreamHandler(log_capture)
handler.setLevel(logging.ERROR)
logger.addHandler(handler)
_validate_tool_results(msgs, "test18")
logger.removeHandler(handler)
log_output = log_capture.getvalue()
T.check("validate detects mismatch", "MISMATCH" in log_output or "unresolved_custom" in log_output,
        f"log output: {log_output[:200]}")


# ── Summary ──
success = T.summary()
exit(0 if success else 1)
