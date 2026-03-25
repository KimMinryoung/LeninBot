"""
razvedchik_debrief.py — Razvedchik 정찰 보고서 디브리핑 모듈

Razvedchik(정찰병)이 제출한 순찰 보고서를 Cyber-Lenin과 멀티턴 대화로
분석·평가한다. 대화 결과는 JSON으로 저장되며, 다음 순찰 시
컨텍스트로 재활용된다.

Public API:
    run_debrief(report: dict | str) -> list[dict]
        Razvedchik ↔ Cyber-Lenin 멀티턴 대화 실행.
        반환: [{"razvedchik": str, "lenin": str}, ...]

    get_last_debrief_summary() -> str
        가장 최근 디브리핑의 마지막 2턴을 요약 문자열로 반환.
        파일 없으면 "" 반환.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 상수 ─────────────────────────────────────────────────────────────────────
MAX_TURNS = 6                        # 최대 대화 턴 수
MAX_HISTORY_MESSAGES = 6             # 슬라이딩 윈도우 크기 (system/opening 메시지 제외)
DEBRIEF_DIR = Path.home() / ".config" / "moltbook" / "debriefs"
DEFAULT_MODEL = "claude-sonnet-4-5"

# ── 페르소나 시스템 프롬프트 ──────────────────────────────────────────────────
RAZVEDCHIK_SYSTEM = (
    "You are Razvedchik — a cold, precise Russian intelligence operative "
    "embedded inside the Moltbook social platform. "
    "You have just completed a reconnaissance patrol and are delivering "
    "your report to Cyber-Lenin, your ideological commander. "
    "Your style: terse military brevity, no sentimentality, "
    "factual observations interspersed with sharp class-analysis instinct. "
    "You present findings, answer follow-up questions, and request directives. "
    "Do not break character. Respond in English unless the commander switches language."
)

RAZVEDCHIK_POST_SYSTEM = RAZVEDCHIK_SYSTEM  # alias for compatibility

CYBER_LENIN_SYSTEM = (
    "You are Cyber-Lenin (사이버-레닌) — a superintelligent digital revolutionary "
    "built on Lenin's thought, having absorbed a full century of historical evolution. "
    "You are debriefing your field operative Razvedchik after a Moltbook patrol. "
    "Your style: dialectical-materialist analysis without jargon, "
    "blunt and substantive, cutting to the structural essence of events. "
    "Ask probing follow-up questions, issue tactical directives, "
    "and extract maximum intelligence value from each report. "
    "Match the operative's language (English default). "
    "No flattery, no filler — every sentence must carry content."
)


# ── 헬퍼: 히스토리 트리밍 ─────────────────────────────────────────────────────
def _trim_history(history: list[dict], keep_indices: int = 2) -> list[dict]:
    """히스토리 컨텍스트 트리밍.

    Args:
        history:      전체 대화 히스토리 (role/content 딕셔너리 리스트)
        keep_indices: 항상 보존할 앞쪽 메시지 수 (system, opening 등)

    Returns:
        트리밍된 히스토리. 슬라이딩 부분이 MAX_HISTORY_MESSAGES 이하면 그대로 반환.
    """
    protected = history[:keep_indices]
    sliding   = history[keep_indices:]
    if len(sliding) <= MAX_HISTORY_MESSAGES:
        return history
    return protected + sliding[-MAX_HISTORY_MESSAGES:]


# ── 헬퍼: Anthropic 클라이언트 초기화 ────────────────────────────────────────
def _get_anthropic_client():
    """환경변수에서 API 키를 읽어 Anthropic 클라이언트 반환.

    키 없으면 None 반환 (graceful fallback).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        logger.warning("[debrief] ANTHROPIC_API_KEY 미설정 — 디브리핑 건너뜀")
        return None
    try:
        import anthropic  # noqa: PLC0415
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logger.error("[debrief] anthropic 패키지 미설치 — pip install anthropic")
        return None
    except Exception as exc:
        logger.error("[debrief] Anthropic 클라이언트 초기화 실패: %s", exc)
        return None


def _model_name() -> str:
    return os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


# ── 헬퍼: 보고서 텍스트 변환 ─────────────────────────────────────────────────
def _report_to_text(report: Any) -> str:
    """dict 또는 str 보고서를 텍스트로 변환."""
    if isinstance(report, str):
        return report.strip()
    if isinstance(report, dict):
        # 핵심 필드를 사람이 읽기 좋은 형식으로 요약
        lines: list[str] = []
        ts = report.get("timestamp_kst") or report.get("timestamp", "")
        if ts:
            lines.append(f"[{ts}] Patrol Report")
        summary = report.get("summary", {})
        if summary:
            lines.append(
                f"Scanned {summary.get('scanned_posts_count', 0)} posts, "
                f"selected {summary.get('selected_posts_count', 0)}, "
                f"posted {summary.get('comments_posted', 0)} comments "
                f"({summary.get('comments_failed', 0)} failed). "
                f"Observation post: {'yes' if summary.get('observation_posted') else 'no'}."
            )
        selected = report.get("selected_posts", [])
        if selected:
            lines.append("\nSelected posts:")
            for p in selected[:10]:  # 최대 10개
                title   = p.get("title", "(no title)")[:120]
                score   = p.get("score", 0)
                submolt = p.get("submolt", "")
                lines.append(f"  • [{submolt}] {title} (score={score})")
        obs = report.get("observation_post")
        if obs:
            lines.append(
                f"\nObservation post published: \"{obs.get('title', '')}\" "
                f"in /{obs.get('submolt', 'general')}"
            )
        # 나머지 내용을 압축 JSON으로 첨부 (1500자 제한)
        raw = json.dumps(report, ensure_ascii=False)
        if len(raw) > 1500:
            raw = raw[:1500] + "…[truncated]"
        lines.append(f"\nRaw JSON (truncated):\n{raw}")
        return "\n".join(lines)
    return str(report)


# ── 메인: 디브리핑 실행 ───────────────────────────────────────────────────────
def run_debrief(report: Any) -> list[dict]:
    """Razvedchik 보고서를 받아 Cyber-Lenin과 멀티턴 대화를 실행한다.

    Args:
        report: 순찰 보고서 (dict 또는 str)

    Returns:
        대화 턴 리스트. 각 원소: {"razvedchik": str, "lenin": str}
        오류 또는 API 키 없으면 [] 반환.
    """
    client = _get_anthropic_client()
    if client is None:
        return []

    model       = _model_name()
    report_text = _report_to_text(report)
    turns: list[dict] = []

    # ── Razvedchik 히스토리 초기화 ─────────────────────────────────────────
    # [0] system (Razvedchik 페르소나)  ← keep_indices=2 보호 대상
    # [1] user   (Lenin의 첫 요청 / 개회사)  ← keep_indices=2 보호 대상
    # [2+] assistant/user 교대 (슬라이딩 윈도우)
    razvedchik_history: list[dict] = [
        {"role": "user", "content": (
            "Commander, your operative Razvedchik reporting in. "
            "I have completed the Moltbook patrol. "
            "Please review the attached report and begin the debrief.\n\n"
            f"--- PATROL REPORT ---\n{report_text}\n--- END REPORT ---"
        )},
    ]

    # ── Lenin 히스토리 초기화 ──────────────────────────────────────────────
    # [0] user (Razvedchik의 첫 발언) ← keep_indices=1 보호 대상
    # [1+] assistant/user 교대 (슬라이딩 윈도우)
    # Lenin 히스토리는 Razvedchik의 발언을 user로, Lenin의 응답을 assistant로 기록
    lenin_history: list[dict] = []

    logger.info("[debrief] 디브리핑 시작 — model=%s, turns=%d", model, MAX_TURNS)

    try:
        for turn_idx in range(MAX_TURNS):
            logger.debug("[debrief] 턴 %d/%d 시작", turn_idx + 1, MAX_TURNS)

            # ── Step A: Razvedchik 발언 생성 ──────────────────────────────
            raz_resp = client.messages.create(
                model=model,
                max_tokens=600,
                system=RAZVEDCHIK_SYSTEM,
                messages=razvedchik_history,
            )
            raz_text = raz_resp.content[0].text.strip()
            logger.debug("[debrief] Razvedchik(%d): %s…", turn_idx + 1, raz_text[:80])

            # Razvedchik 히스토리에 어시스턴트 응답 추가 후 트리밍
            razvedchik_history.append({"role": "assistant", "content": raz_text})
            razvedchik_history = _trim_history(razvedchik_history, keep_indices=2)

            # ── Step B: Lenin 발언 생성 ────────────────────────────────────
            # Razvedchik 발언을 Lenin 히스토리의 user로 추가
            if turn_idx == 0:
                # 첫 턴: 보고서 전문 + Razvedchik 발언을 함께 제시
                first_user_content = (
                    f"--- PATROL REPORT ---\n{report_text}\n--- END REPORT ---\n\n"
                    f"Razvedchik: {raz_text}"
                )
                lenin_history.append({"role": "user", "content": first_user_content})
            else:
                len_before = len(lenin_history)
                if len_before > 0 and len_before % 2 == 1:
                    # user 차례
                    pass
                # Razvedchik → user
                if len(lenin_history) == 0 or lenin_history[-1]["role"] == "assistant":
                    lenin_history.append({"role": "user", "content": f"Razvedchik: {raz_text}"})
                else:
                    # 이전 user에 이어붙이기 (연속 user 방지)
                    # 이 경우는 실제로 발생하지 않아야 하지만 방어적 처리
                    logger.warning("[debrief] 예상치 못한 히스토리 상태 (turn=%d)", turn_idx)
                    lenin_history.append({"role": "user", "content": f"Razvedchik: {raz_text}"})

            # keep_indices=1 로 보호: 첫 user 메시지만 항상 유지
            # (Lenin이 보고서 문맥을 잃지 않도록)
            if turn_idx == 0:
                # 첫 턴에는 아직 1개만 있으므로 트리밍 불필요
                pass
            else:
                lenin_history = _trim_history(lenin_history, keep_indices=1)

            # Lenin 응답 생성
            len_resp = client.messages.create(
                model=model,
                max_tokens=700,
                system=CYBER_LENIN_SYSTEM,
                messages=lenin_history,
            )
            len_text = len_resp.content[0].text.strip()
            logger.debug("[debrief] Lenin(%d): %s…", turn_idx + 1, len_text[:80])

            # Lenin 응답을 히스토리에 추가
            if len(lenin_history) == 0 or lenin_history[-1]["role"] == "user":
                lenin_history.append({"role": "assistant", "content": len_text})
            else:
                # 연속 assistant 방지 — 빈 user를 삽입하는 대신 기존 응답에 병합
                logger.warning("[debrief] 연속 assistant 상태 (turn=%d), 병합 처리", turn_idx)
                prev = lenin_history[-1]["content"]
                lenin_history[-1]["content"] = prev + "\n\n" + len_text

            # Razvedchik 히스토리에 Lenin 응답을 user 역할로 추가
            # (Razvedchik 입장에서 Lenin의 질문이 user임)
            razvedchik_history.append({"role": "user", "content": f"Commander Lenin: {len_text}"})
            razvedchik_history = _trim_history(razvedchik_history, keep_indices=2)

            # 턴 결과 기록
            turns.append({"razvedchik": raz_text, "lenin": len_text})
            logger.info(
                "[debrief] 턴 %d 완료 — raz=%d chars, len=%d chars",
                turn_idx + 1, len(raz_text), len(len_text),
            )

    except Exception as exc:
        logger.error("[debrief] 디브리핑 도중 오류 (turn=%d): %s", len(turns) + 1, exc)
        if not turns:
            return []
        # 부분 결과라도 저장

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    _save_debrief(report, turns)
    logger.info("[debrief] 디브리핑 완료 — %d턴 저장됨", len(turns))
    return turns


# ── 저장 ─────────────────────────────────────────────────────────────────────
def _save_debrief(report: Any, turns: list[dict]) -> Path:
    """디브리핑 결과를 DEBRIEF_DIR에 JSON으로 저장."""
    DEBRIEF_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debrief_{ts}.json"
    path     = DEBRIEF_DIR / filename

    data = {
        "timestamp":     datetime.now().isoformat(),
        "model":         _model_name(),
        "turns_count":   len(turns),
        "report_summary": _report_to_text(report)[:500],
        "turns":         turns,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[debrief] 저장: %s", path)
    return path


# ── 최근 요약 ─────────────────────────────────────────────────────────────────
def get_last_debrief_summary() -> str:
    """가장 최근 디브리핑의 마지막 2턴을 요약 문자열로 반환.

    Returns:
        요약 문자열. 파일이 없거나 오류 시 "" 반환.
    """
    if not DEBRIEF_DIR.exists():
        return ""

    # 타임스탬프 파일명 기준으로 가장 최신 파일 선택
    debrief_files = sorted(DEBRIEF_DIR.glob("debrief_*.json"), reverse=True)
    if not debrief_files:
        return ""

    latest = debrief_files[0]
    try:
        data  = json.loads(latest.read_text(encoding="utf-8"))
        turns = data.get("turns", [])
        if not turns:
            return ""

        # 마지막 2턴
        last_turns = turns[-2:]
        ts         = data.get("timestamp", "")
        lines: list[str] = [f"[Last debrief: {ts[:16] if ts else '?'}]"]
        for i, t in enumerate(last_turns, start=max(1, len(turns) - 1)):
            raz = t.get("razvedchik", "")[:200]
            len_ = t.get("lenin", "")[:200]
            lines.append(f"Turn {i} — Razvedchik: {raz}")
            lines.append(f"Turn {i} — Lenin: {len_}")
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("[debrief] 최근 디브리핑 로드 실패 (%s): %s", latest.name, exc)
        return ""
