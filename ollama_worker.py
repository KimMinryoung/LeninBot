"""
ollama_worker.py — LLM 문학 분석 위임 스크립트 (MOON PC 우선, 로컬 폴백)

사용법:
    python ollama_worker.py              # 전체 태스크 실행
    python ollama_worker.py --task <id>  # 특정 태스크만 (결과: temp_dev/)
    python ollama_worker.py --list       # 태스크 목록 출력

결과:
    전체 실행: literature/analysis/<id>_analysis.json
    단일 테스트: temp_dev/<id>_analysis.json
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path
from ollama_client import ask, check_ollama

# ── 설정 ─────────────────────────────────────────────
OUTPUT_DIR   = Path("literature/analysis")
TEST_DIR     = Path("temp_dev")

# ── 태스크 정의 ──────────────────────────────────────
TASKS = [
    {
        "id": "yun_dongju_seosi",
        "type": "style_analysis",
        "title": "서시",
        "author": "윤동주",
        "file": "literature/poetry/yun_dongju/서시.txt",
        "prompt_template": "style_analysis",
    },
    {
        "id": "yun_dongju_jawhsang",
        "type": "style_analysis",
        "title": "자화상",
        "author": "윤동주",
        "file": "literature/poetry/yun_dongju/자화상.txt",
        "prompt_template": "style_analysis",
    },
    {
        "id": "yun_dongju_byulhenunn",
        "type": "style_analysis",
        "title": "별헤는밤",
        "author": "윤동주",
        "file": "literature/poetry/yun_dongju/별헤는밤.txt",
        "prompt_template": "style_analysis",
    },
    {
        "id": "yun_dongju_sipjaga",
        "type": "style_analysis",
        "title": "십자가",
        "author": "윤동주",
        "file": "literature/poetry/yun_dongju/십자가.txt",
        "prompt_template": "style_analysis",
    },
    {
        "id": "yun_dongju_chamhoirok",
        "type": "style_analysis",
        "title": "참회록",
        "author": "윤동주",
        "file": "literature/poetry/yun_dongju/참회록.txt",
        "prompt_template": "style_analysis",
    },
    {
        "id": "hyunjeongeon_unsujoeunnal",
        "type": "narrative_analysis",
        "title": "운수좋은날",
        "author": "현진건",
        "file": "literature/fiction/운수좋은날_현진건.txt",
        "prompt_template": "narrative_analysis",
    },
    {
        "id": "isang_nalgae",
        "type": "narrative_analysis",
        "title": "날개",
        "author": "이상",
        "file": "literature/fiction/날개_이상.txt",
        "prompt_template": "narrative_analysis",
    },
]

# ── 프롬프트 템플릿 ───────────────────────────────────
def build_prompt(task: dict, text: str) -> str:
    author = task["author"]
    title  = task["title"]

    if task["prompt_template"] == "style_analysis":
        return f"""다음은 {author}의 시 「{title}」이다.

---
{text}
---

아래 항목을 분석하라. 각 항목은 3~5문장으로 구체적으로 서술할 것.

1. **문체적 특징** — 문장 길이, 어조, 반복 구조, 리듬감
2. **핵심 이미지** — 가장 강렬한 시각적·감각적 이미지와 그 기능
3. **주제 의식** — 시가 드러내는 중심 사상 또는 정서
4. **시대적 맥락** — 이 시가 쓰인 역사적 배경과 시의 관계
5. **문학사적 위치** — 한국 근대시에서 이 작품의 의미

분석은 학술적이되 명료하게 작성하라."""

    elif task["prompt_template"] == "narrative_analysis":
        return f"""다음은 {author}의 소설 「{title}」이다.

---
{text}
---

아래 항목을 분석하라. 각 항목은 4~6문장으로 구체적으로 서술할 것.

1. **서술 기법** — 시점, 서술자의 태도, 문체적 특징
2. **구성과 플롯** — 사건 전개 방식, 극적 아이러니, 반전
3. **인물 분석** — 주요 인물의 심리와 사회적 의미
4. **주제 의식** — 소설이 드러내는 중심 메시지
5. **시대적 맥락** — 일제강점기와 이 작품의 관계
6. **문학사적 위치** — 한국 근대소설에서 이 작품의 의미

분석은 학술적이되 명료하게 작성하라."""

    else:
        return f"{author}의 「{title}」을 분석하라:\n\n{text}"


# ── LLM 호출 ─────────────────────────────────────────
def query_ollama(prompt: str) -> str:
    return ask(prompt, temperature=0.7)


# ── 단일 태스크 실행 ──────────────────────────────────
def run_task(task: dict, output_dir: Path = OUTPUT_DIR) -> dict:
    task_id = task["id"]
    file_path = Path(task["file"])

    # 텍스트 로드
    if not file_path.exists():
        raise FileNotFoundError(f"파일 없음: {file_path}")
    text = file_path.read_text(encoding="utf-8")

    # 프롬프트 빌드
    prompt = build_prompt(task, text)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ {task['author']} 「{task['title']}」 분석 중...")

    # LLM 호출
    result_text = query_ollama(prompt)

    # 결과 구성
    output = {
        "id":        task_id,
        "type":      task["type"],
        "author":    task["author"],
        "title":     task["title"],
        "source":    task["file"],
        "model":     check_ollama().get("model", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "analysis":  result_text,
    }

    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task_id}_analysis.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  ✅ 저장: {out_path}")
    return output


# ── 전체 실행 ─────────────────────────────────────────
def run_all():
    status = check_ollama()
    print(f"=== ollama_worker 시작 ({status.get('backend', '?')}: {status.get('model', '?')}) ===")
    print(f"총 {len(TASKS)}개 태스크\n")

    results = []
    failed  = []

    for task in TASKS:
        try:
            out = run_task(task, OUTPUT_DIR)
            results.append(out)
        except Exception as e:
            print(f"  ❌ 실패: {task['id']} — {e}")
            failed.append({"id": task["id"], "error": str(e)})

    # 요약 리포트
    print(f"\n=== 완료 ===")
    print(f"성공: {len(results)}개 / 실패: {len(failed)}개")

    summary = {
        "run_at":   datetime.now().isoformat(),
        "model":    MODEL,
        "success":  len(results),
        "failed":   len(failed),
        "failures": failed,
    }
    summary_path = OUTPUT_DIR / "_run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"요약 저장: {summary_path}")


# ── 진입점 ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="로컬 LLM 문학 분석 워커")
    parser.add_argument("--task", help="특정 태스크 ID만 실행 — 결과는 temp_dev/ 에 저장")
    parser.add_argument("--list", action="store_true", help="태스크 목록 출력")
    args = parser.parse_args()

    if args.list:
        print("등록된 태스크:")
        for t in TASKS:
            print(f"  {t['id']:40s} {t['author']} 「{t['title']}」")
    elif args.task:
        target = next((t for t in TASKS if t["id"] == args.task), None)
        if not target:
            print(f"태스크 없음: {args.task}")
        else:
            run_task(target, TEST_DIR)   # ← temp_dev 고정
    else:
        run_all()
