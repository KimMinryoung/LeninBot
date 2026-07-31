"""GPT-5.6 Luna 경량 호출부 스모크.

레지스트리(config/llm_call_sites.json)에 등록된 경량 호출부를 실제 API로 한 번씩
호출해 openai 계약(max_completion_tokens / temperature 생략 / reasoning_effort)이
맞는지 확인한다. generate_sync는 실패를 삼키고 None을 돌려주므로, 여기서는
executor를 직접 불러 예외를 그대로 드러낸다.

OPENAI_API_KEY가 필요하다. credstore에서 받으려면:

  sudo systemd-run --pipe --wait --property=User=grass \
    --property=WorkingDirectory=/home/grass/leninbot \
    --property=LoadCredentialEncrypted=openai_api_key:/etc/credstore.encrypted/openai_api_key.cred \
    /home/grass/leninbot/venv/bin/python scripts/smoke_luna_call_sites.py
"""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from llm.call_registry import _EXECUTORS, resolve  # noqa: E402

# 호출부별 max_completion_tokens 목표치 — 추론 토큰까지 포함하는 상한이라
# 기존 gemini 상한(32~4096)보다 여유를 둔다.
TARGET_MAX_TOKENS = {
    "chunk_summary": 2048,
    "conversation_reflection": 1536,
    "scout_kg_classify": 512,
    "experience_extraction": 8192,
    "research_spelling_proofread": 1024,
}


def luna_profile(feature: str):
    """레지스트리 전환 전에도 검증할 수 있도록 luna 대상 프로파일을 강제한다."""
    return dataclasses.replace(
        resolve(feature),
        provider="openai",
        model="gpt-5.6-luna",
        max_tokens=TARGET_MAX_TOKENS[feature],
        extra={"reasoning_effort": "minimal"},
    )

CASES = {
    "chunk_summary": (
        "다음 대화를 3문장 이내로 요약하라.\n"
        "사용자: 오늘 배포한 가격표 수정 확인했어?\n"
        "봇: 네, Terra와 Luna 단가를 인하분에 맞춰 갱신했습니다.\n"
        "사용자: 예산 집행에 바로 반영되나?\n"
        "봇: 다음 호출부터 반영됩니다.",
        None,
    ),
    "conversation_reflection": (
        "다음 대화에서 기억할 만한 사실 한 가지를 한 문장으로 적어라.\n"
        "사용자: 나는 커밋을 항상 main에 직접 푸시하는 편이야.",
        None,
    ),
    "scout_kg_classify": (
        "다음 리포트의 group_id를 하나만 골라 단어로만 답하라 "
        "(agent_knowledge / world_events / user_context).\n"
        "리포트: OpenAI가 GPT-5.6 Terra와 Luna의 API 단가를 인하했다.",
        None,
    ),
    "experience_extraction": (
        "다음 활동 로그에서 경험 항목을 JSON 배열로 뽑아라.\n"
        "- 가격표의 낡은 단가를 발견해 수정하고 main에 푸시했다.\n"
        "- 경량 호출부를 저가 모델로 교체하기 전에 계약 불일치를 먼저 찾아냈다.",
        None,
    ),
    "research_spelling_proofread": (
        "다음은 보고서 저장 시 자동 적용된 표기 교정 목록이다. 문맥상 잘못된 교정의 "
        '번호만 반환하라.\n1. "레닌" → "블라디미르 레닌"\n2. "쏘련" → "소련"\n\n'
        'JSON 한 줄로만 답하라: {"revert": [번호, ...]}',
        "당신은 교정 검수 담당이다. 확신이 없으면 교정을 유지한다.",
    ),
}


def main() -> int:
    failures = []
    for feature, (prompt, system) in CASES.items():
        profile = luna_profile(feature)
        print("=" * 72)
        print(f"[{feature}] {profile.provider}/{profile.model} "
              f"max={profile.max_tokens} extra={profile.extra}")
        try:
            out = _EXECUTORS[profile.provider](profile, prompt, system)
        except Exception as e:
            print(f"FAIL  {type(e).__name__}: {e}")
            failures.append((feature, f"{type(e).__name__}: {e}"))
            continue
        print(f"--- output ({len(out)} chars) ---")
        print(out if out else "(EMPTY — 추론 토큰이 상한을 다 먹었을 수 있다)")
        if not out:
            failures.append((feature, "empty output"))

    print("=" * 72)
    if failures:
        print(f"FAILED {len(failures)}/{len(CASES)}")
        for feature, why in failures:
            print(f"  - {feature}: {why}")
        return 1
    print(f"OK {len(CASES)}/{len(CASES)} — luna 경량 호출부 계약 확인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
