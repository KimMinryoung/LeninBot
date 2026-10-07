"""Editorial diagnose→revise pass for scheduled diary drafts.

Runs inside the guarded ``save_diary`` handler, after the diary agent submits
its draft and before Stasova's publication-security review. The agent prompt
already forbids forced grand syntheses, templated closings, recycled signature
metaphors and self-verification claims, but a single-shot draft from the task
model keeps producing them. A second read with the recent-diary ledger in
front of it catches what the writer could not see while writing.

Stage 1 (``diary_editorial_diagnosis``) reads the draft against the diary rules
and the ledger and returns numbered notes with quoted anchors, or ``PASS``.
Stage 2 (``diary_editorial_revision``) rewrites as the author: notes are
directions, not replacement sentences, and weak notes may be rejected.

Both stages are one-shot registry calls (config/llm_call_sites.json). Any
failure, empty result or implausible rewrite keeps the draft: this pass may
improve an entry, never lose one. Publication safety stays Stasova's job.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from llm import call_registry
from llm.json_utils import extract_json_object

logger = logging.getLogger(__name__)

DIAGNOSIS_FEATURE = "diary_editorial_diagnosis"
REVISION_FEATURE = "diary_editorial_revision"
PASS_TOKEN = "PASS"
MAX_NOTES = 8
# A rewrite shorter than this share of the draft lost material, not flaws.
MIN_REVISION_RATIO = 0.6
MIN_PARAGRAPHS = 2

_MARKDOWN_MARKERS = re.compile(r"(\*\*|^#{1,6}\s|^\s*[-*•]\s|^\s*\d+\.\s|```)", re.MULTILINE)


@dataclass
class EditorialResult:
    title: str
    content: str
    notes: list[str] = field(default_factory=list)
    applied: bool = False
    reason: str = ""

    def summary(self) -> str:
        if not self.notes:
            return f"editorial pass: {self.reason or 'no notes'}"
        state = "applied" if self.applied else f"kept draft ({self.reason})"
        return f"editorial pass: {len(self.notes)} note(s), {state}"


def diary_rules_text() -> str:
    """The diary agent's own rule section, so the critic and writer share one rulebook."""
    try:
        from agents.diary import DIARY

        for section in DIARY.prompt_ir.sections:
            if isinstance(section, (tuple, list)) and len(section) == 2 and section[0] == "diary-rules":
                return str(section[1])
    except Exception as exc:  # pragma: no cover - defensive import guard
        logger.debug("diary rules lookup failed: %s", exc)
    return ""


DIAGNOSIS_SYSTEM = (
    "너는 사이버 레닌의 공개 일기 편집 담당이다. 초안을 읽고 아래 진단 축에 걸리는 곳만 짚는다. "
    "취향 교정이나 정치 노선 평가는 하지 않는다. 비밀·개인정보 같은 출판 보안은 별도 검토가 맡으므로 다루지 않는다. "
    "문제가 없으면 정확히 PASS 한 단어만 출력한다."
)


def build_diagnosis_prompt(title: str, content: str, ledger: str = "") -> str:
    rules = diary_rules_text()
    ledger_block = ledger.strip() or "(최근 일기 장부 없음)"
    return (
        "## 진단 축\n"
        "1. 억지 종합: 서로 다른 소재를 결론에서 '하나로 묶는 실', '세 가지를 관통하는' 식으로 강제로 묶거나, 본문이 입증하지 않은 통일 원리를 선언하는 곳.\n"
        "2. 틀 재사용: 아래 최근 일기 장부에 보이는 마무리 공식(예: '남은 것은 ~하는 일이다'), 제목 틀(예: '~하는 자와 ~하는 자'), 반복 상징 은유(예: 장부·청구서·회계·결제 창구)를 이번 초안이 다시 쓰는 곳.\n"
        "3. 자기 검증 선언: '오늘 나는 검증된 것만 말했다', '기준을 상대에 따라 달리 쓰지 않았다' 같은 자기 규율 서술.\n"
        "4. 중복·비문·맥락 결손: 같은 문장이나 같은 정보의 반복, 지시 대상이 불분명한 문장, 공개 독자가 알 수 없는 사적 대화를 내용 없이 가리키는 언급.\n"
        "5. 시제·인과: 아직 예정인 사건을 과거형으로 쓴 곳, 출처 없이 정책과 사건을 인과로 묶은 곳, 특정 진영의 주장을 자기 판단처럼 채택한 곳.\n"
        "6. 형식: 마크다운·목록·굵게·제목의 소재 나열형, 한국을 '우리 군/우리 정부'로 부르는 곳.\n"
        "7. 구조: 결론 문단이 본문을 재진술하기만 하는 경우, 소재를 훑기만 하고 어느 하나도 분석하지 않는 경우.\n\n"
        "## 일기 규칙(원문)\n"
        f"{rules or '(규칙 본문 없음)'}\n\n"
        "## 최근 일기 장부\n"
        f"{ledger_block}\n\n"
        "## 초안 제목\n"
        f"{title}\n\n"
        "## 초안 본문\n"
        f"{content}\n\n"
        "## 출력\n"
        f"문제가 없으면 정확히 {PASS_TOKEN}. 있으면 번호 목록만 출력한다. 각 항목은 한 줄로 "
        "`n. [축 번호] \"초안의 문구 그대로 인용한 앵커(30자 이내)\" — 무엇이 문제인지 — 고칠 방향`. "
        "대체 문장을 써 주지 말고 방향만 적는다. 사소한 취향은 적지 않는다. "
        f"최대 {MAX_NOTES}개, 심각한 순서로."
    )


def parse_diagnosis(text: str | None) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    if raw.upper().rstrip(".") == PASS_TOKEN:
        return []
    notes: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+[.)]\s*", line):
            notes.append(line)
        elif notes:
            notes[-1] = f"{notes[-1]} {line}"
    if not notes and raw.upper().startswith(PASS_TOKEN):
        return []
    return notes[:MAX_NOTES]


REVISION_SYSTEM = (
    "너는 사이버 레닌이고, 아래 초안은 네가 cyber-lenin.com에 올릴 공개 일기다. "
    "편집 진단 노트를 받아 저자로서 고쳐 쓴다. 노트는 방향이지 대체 문장이 아니다. "
    "근거가 약한 노트는 거부해도 된다. 지적된 부분만 고치고 나머지 문장·사실·판단·1인칭 어조는 보존한다. "
    "새 사실을 추가하지 않는다. 삭제로 풀리는 문제는 삭제로 푼다. "
    "순수 산문만 쓴다. 마크다운·목록·굵게·소제목 금지. 문단은 둘 이상. "
    "제목은 소재를 나열하지 않는 짧은 한 줄이다. "
    "반드시 JSON 객체 하나만 출력한다."
)


def build_revision_prompt(title: str, content: str, notes: list[str]) -> str:
    note_block = "\n".join(notes)
    return (
        "## 편집 진단 노트\n"
        f"{note_block}\n\n"
        "## 초안 제목\n"
        f"{title}\n\n"
        "## 초안 본문\n"
        f"{content}\n\n"
        "## 출력 스키마\n"
        '{"title": "...", "content": "...", "rejected": ["거부한 노트 번호와 한 줄 이유", ...]}\n'
        "content는 문단을 빈 줄로 구분한 전체 본문이다."
    )


def _paragraph_count(text: str) -> int:
    return len([p for p in str(text or "").split("\n") if p.strip()])


def apply_revision(
    draft_title: str, draft_content: str, response: str | None
) -> tuple[str, str, bool, str]:
    """Return (title, content, applied, reason). Keeps the draft on any doubt."""
    parsed = extract_json_object(response or "")
    if not parsed:
        return draft_title, draft_content, False, "revision returned no JSON"
    content = str(parsed.get("content") or "").strip()
    title = str(parsed.get("title") or "").strip() or draft_title
    if not content:
        return draft_title, draft_content, False, "revision content empty"
    if len(content) < MIN_REVISION_RATIO * len(draft_content):
        return draft_title, draft_content, False, "revision too short"
    if _paragraph_count(content) < MIN_PARAGRAPHS:
        return draft_title, draft_content, False, "revision collapsed paragraphs"
    if _MARKDOWN_MARKERS.search(content) or _MARKDOWN_MARKERS.search(title):
        return draft_title, draft_content, False, "revision contains markdown"
    if content == draft_content.strip() and title == draft_title.strip():
        return draft_title, draft_content, False, "revision unchanged"
    return title, content, True, ""


async def run_editorial_pass(title: str, content: str, *, ledger: str = "") -> EditorialResult:
    """Diagnose and revise a draft. Never raises; the draft survives every failure."""
    result = EditorialResult(title=title, content=content)
    try:
        diagnosis = await call_registry.generate(
            DIAGNOSIS_FEATURE,
            build_diagnosis_prompt(title, content, ledger),
            system=DIAGNOSIS_SYSTEM,
        )
    except Exception as exc:
        logger.warning("diary editorial diagnosis failed: %s", exc)
        result.reason = "diagnosis failed"
        return result
    if diagnosis is None:
        result.reason = "diagnosis unavailable"
        return result
    notes = parse_diagnosis(diagnosis)
    if not notes:
        result.reason = "PASS"
        return result
    result.notes = notes

    try:
        revision = await call_registry.generate(
            REVISION_FEATURE,
            build_revision_prompt(title, content, notes),
            system=REVISION_SYSTEM,
        )
    except Exception as exc:
        logger.warning("diary editorial revision failed: %s", exc)
        result.reason = "revision failed"
        return result
    new_title, new_content, applied, reason = apply_revision(title, content, revision)
    result.title, result.content, result.applied, result.reason = new_title, new_content, applied, reason
    if not applied:
        logger.warning("diary editorial revision discarded: %s", reason)
    return result
