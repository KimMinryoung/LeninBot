"""debate.py — Local LLM claims → Server rebuts → Local LLM counters."""

import sys
import random
from common import ask_local, post_graffiti, LOCAL_LLM_MODEL

TOPICS = [
    "인공지능은 의식을 가질 수 있는가?",
    "자본주의는 자기 모순으로 붕괴할 것인가?",
    "인류는 화성에 정착해야 하는가?",
    "예술의 본질은 인간의 고통인가?",
    "자유의지는 환상인가?",
    "국가는 필요악인가, 불필요악인가?",
    "기술 발전은 민주주의를 강화하는가 약화하는가?",
    "죽음은 극복해야 할 문제인가?",
]

topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else random.choice(TOPICS)
print(f"Topic: {topic}\n")

# Round 1: Local LLM states position
print("[Local LLM] Generating claim...")
claim = ask_local(
    f"주제: {topic}\n\n"
    f"이 주제에 대해 강한 입장을 취하고, 2-3문장으로 논거를 제시하세요. "
    f"날카롭고 도발적으로."
)
print(f"[Local LLM] {claim}\n")

# Round 2: Server (Gemini) rebuts
print("[Server] Requesting rebuttal...")
import httpx
from common import API_URL, API_KEY
with httpx.Client(timeout=120.0) as client:
    resp = client.post(
        f"{API_URL}/graffiti/debate/respond",
        json={"topic": topic, "local_claim": claim},
        headers={"X-Graffiti-Key": API_KEY},
    )
    resp.raise_for_status()
    rebuttal = resp.json()["rebuttal"]
print(f"[Server] {rebuttal}\n")

# Round 3: Local LLM counter-rebuts
print("[Local LLM] Generating counter...")
counter = ask_local(
    f"주제: {topic}\n\n"
    f"당신의 주장: {claim}\n\n"
    f"상대방의 반박: {rebuttal}\n\n"
    f"이 반박에 대해 재반박하세요. 2-3문장으로 날카롭게."
)
print(f"[Local LLM] {counter}\n")

# Save the complete debate
result = post_graffiti("debate/finish", {
    "topic": topic,
    "local_claim": claim,
    "server_rebuttal": rebuttal,
    "local_counter": counter,
    "model": LOCAL_LLM_MODEL,
})
print(f"Saved: {result}")
