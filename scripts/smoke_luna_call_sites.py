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

# 호출부별 (max_completion_tokens, reasoning_effort) 목표치.
#
# GPT-5.6의 기본 effort는 medium이다. 아래 호출부는 대부분 요약·분류·추출이라
# 추론이 필요 없어 "none"으로 끈다 (OpenAI reasoning 가이드가 분류·빠른 검색을
# none 권장 용도로 명시). 추론을 끄면 max_completion_tokens를 본문이 다 쓰므로
# 상한도 기존 gemini 값 수준으로 되돌린다.
#
# 예외는 research_spelling_proofread — 기계 교정이 문맥상 맞는지 판단하는
# 유일한 판단형 작업이라 low를 준다.
TARGETS = {
    "chunk_summary": (1024, "none"),
    "conversation_reflection": (768, "none"),
    "scout_kg_classify": (64, "none"),
    "experience_extraction": (4096, "none"),
    "research_spelling_proofread": (1024, "low"),
}


def luna_profile(feature: str):
    """레지스트리 전환 전에도 검증할 수 있도록 luna 대상 프로파일을 강제한다."""
    max_tokens, effort = TARGETS[feature]
    return dataclasses.replace(
        resolve(feature),
        provider="openai",
        model="gpt-5.6-luna",
        max_tokens=max_tokens,
        extra={"reasoning_effort": effort},
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


def reasoning_token_probe() -> str | None:
    """effort=none이 실제로 추론 토큰을 0으로 만드는지 raw 호출로 확인한다."""
    from openai import OpenAI

    from llm.call_registry import resolve_provider_connection

    connection = resolve_provider_connection("openai")
    client = OpenAI(
        api_key=connection.api_key,
        base_url=connection.base_url,
        timeout=60,
    )
    msgs = [{"role": "user", "content": "다음 리포트의 group_id를 한 단어로만 답하라 "
                                        "(agent_knowledge / world_events / user_context): "
                                        "OpenAI가 API 단가를 인하했다."}]
    for effort in ("none", "medium"):
        resp = client.chat.completions.create(
            model="gpt-5.6-luna", messages=msgs,
            max_completion_tokens=512, reasoning_effort=effort,
        )
        u = resp.usage
        details = getattr(u, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None)
        print(f"  effort={effort:6s} completion={u.completion_tokens:4d} "
              f"reasoning={reasoning} → {(resp.choices[0].message.content or '').strip()[:40]!r}")
        if effort == "none" and reasoning:
            return f"effort=none인데 추론 토큰 {reasoning}개가 계상됐다"
    return None


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
    print("[reasoning token probe] effort=none이 추론을 실제로 끄는지 확인")
    try:
        problem = reasoning_token_probe()
    except Exception as e:
        problem = f"{type(e).__name__}: {e}"
    if problem:
        print(f"FAIL  {problem}")
        failures.append(("reasoning_token_probe", problem))

    print("=" * 72)
    checks = len(CASES) + 1  # 호출부 + 추론 토큰 프로브
    if failures:
        print(f"FAILED {len(failures)}/{checks}")
        for feature, why in failures:
            print(f"  - {feature}: {why}")
        return 1
    print(f"OK {checks}/{checks} — luna 경량 호출부 계약 + 추론 해제 확인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
