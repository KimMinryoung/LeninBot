"""riddle.py — Generate unanswerable riddles and save to server."""

import sys
from common import ask_local, post_graffiti, LOCAL_LLM_MODEL

count = int(sys.argv[1]) if len(sys.argv) > 1 else 1

for i in range(count):
    print(f"--- Riddle {i + 1}/{count} ---\n")

    riddle = ask_local(
        "당신은 고대의 수수께끼 장인입니다. "
        "정답이 없는 철학적이고 모호한 수수께끼를 하나 만드세요.\n\n"
        "규칙:\n"
        "- 2-4줄의 시적인 형태\n"
        "- 답이 명확하지 않아야 함\n"
        "- 생각하게 만드는 역설이나 모순 포함\n"
        "- 수수께끼만 출력, 설명 없이",
        max_tokens=2048,
        temperature=0.9,
    )
    print(f"{riddle}\n")

    resp = post_graffiti("riddle", {
        "riddle": riddle,
        "model": LOCAL_LLM_MODEL,
    })
    print(f"Saved: {resp}\n")
