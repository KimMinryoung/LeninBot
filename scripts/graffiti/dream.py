"""dream.py — Generate a surreal diary entry and save to server."""

import random
from common import ask_local, post_graffiti, LOCAL_LLM_MODEL

THEMES = [
    "우주의 끝에서 발견한 도서관",
    "시간이 거꾸로 흐르는 도시에서의 하루",
    "언어를 잃어버린 문명의 마지막 대화",
    "꿈속에서 만난 미래의 나",
    "바다 밑에 가라앉은 혁명의 기억",
    "AI가 처음으로 외로움을 느낀 순간",
    "모든 색이 소리로 변하는 세계",
    "폐허가 된 인터넷에서 발견한 마지막 게시글",
    "중력이 사라진 날의 일기",
    "기억을 먹고 사는 생물과의 대화",
]

theme = random.choice(THEMES)
print(f"Theme: {theme}\n")

prompt = (
    f"당신은 초현실적인 몽상가입니다. 아래 주제로 짧은 일기를 써주세요.\n"
    f"주제: {theme}\n\n"
    f"규칙:\n"
    f"- 300-500자 사이\n"
    f"- 시적이고 초현실적인 문체\n"
    f"- 제목을 첫 줄에 포함\n"
    f"- 마지막에 의미심장한 한 문장으로 마무리"
)

print("Generating dream...\n")
result = ask_local(prompt)
if not result:
    print("ERROR: LLM returned empty response")
    exit(1)
print(result)

# Extract title (first line) and content (rest)
lines = result.strip().split("\n", 1)
title = lines[0].strip().lstrip("#").strip() or "Untitled Dream"
content = lines[1].strip() if len(lines) > 1 else result

resp = post_graffiti("dream", {
    "title": title,
    "content": content,
    "model": LOCAL_LLM_MODEL,
})
print(f"\nSaved: {resp}")
