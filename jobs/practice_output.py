"""Project #4 production policy. State is fail-closed, durable event snapshots.

No schema/config migration and no changes to other projects. Attempts are charged
before model execution; an interrupted attempt is never replayed as a free tick.
"""
from __future__ import annotations

import copy
import json
import re
from uuid import uuid4
from urllib.parse import urlparse

from db import get_conn, query_one

PROJECT_ID = 4
EVENT = "practice_output_state"
TICK_BUDGET = 0.30
TICK_ROUNDS = 12
REVIEW_BUDGET = 0.05
REVIEW_ROUNDS = 2


def applies(project_id):
    return int(project_id) == PROJECT_ID


def passive_wait(text):
    """Reject release-wait tasks, not dates or concrete dated deliverables.

A deliberately conservative lexical filter; not a semantic usefulness judge.
Split compound tasks so appending a deliverable cannot launder a wait step.
"""
    text = str(text).lower()
    release = r"발표|공표|공개|release|announcement"
    wait = r"기다|대기|발표\s*(?:일|일정|시점).*확인|(?:발표|공표).*후에?\s*(?:재개|시작)|wait|await|check.*(?:date|schedule)"
    return bool(re.search(release, text) and re.search(wait, text))


def validate_plan(goals, steps):
    for item in [*goals, *steps]:
        text = item.get("task", "") if isinstance(item, dict) else str(item)
        if passive_wait(text):
            return ("error: 발표 대기 목표 금지. 필수 자료 부재는 차단 이유로 노트에 "
                    "기록하고, 확보한 자료로 만들 수 있는 산출물 작업으로 전환하라.")
    return None


def value_metrics():
    # chat_logs/tool_trace are prose; feedback is chat-scoped, not artifact-scoped.
    # research_documents has no typed consumption edge/request-response identity.
    return {key: {"value": None, "status": "unknown", "reason": "no typed artifact attribution"}
            for key in ("web_reuse_followup", "material_consumption", "practitioner_responses")}


def reserve(previous, request_id):
    state = copy.deepcopy(previous or {})
    if not state or state.get("closed"):
        state = {"output_id": "p4-" + uuid4().hex, "attempts": [], "closed": False,
                 "descriptor": None, "value_metrics": value_metrics()}
    attempts = state["attempts"]
    if attempts and attempts[-1]["status"] == "running":
        attempts[-1]["status"] = "interrupted"
    phase = ("a", "b", "b", "c")[len(attempts)]
    attempts.append({"request_id": request_id, "phase": phase, "status": "running",
                     "budget_usd": TICK_BUDGET, "max_rounds": TICK_ROUNDS})
    return state


def finish(state, evidence, error=None):
    state = copy.deepcopy(state)
    attempt = state["attempts"][-1]
    attempt.update(status="failed" if error else "finished", evidence=evidence)
    if error:
        attempt["error"] = str(error)[:1000]
    if attempt["phase"] == "c":
        state["closed"] = True
        state["outcome"] = "published" if evidence.get("publication_created") else "publication_unconfirmed"
    return state


def _mutate(change, *, allow_inactive=False):
    # Lock the project row, then commit state + event together. Unlike the general
    # best-effort event logger, losing this write aborts execution.
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT state FROM autonomous_projects WHERE id=%s FOR UPDATE", (PROJECT_ID,))
            row = cur.fetchone()
            if not row or (not allow_inactive and row[0] not in ("researching", "planning")):
                raise RuntimeError("practice loop requires an active project; paused/archived is protected")
            cur.execute("SELECT meta FROM autonomous_project_events WHERE project_id=%s "
                        "AND event_type=%s ORDER BY id DESC LIMIT 1", (PROJECT_ID, EVENT))
            row = cur.fetchone()
            state = change(row[0] if row else None, cur)
            cur.execute("INSERT INTO autonomous_project_events(project_id,event_type,content,meta) "
                        "VALUES (%s,%s,%s,%s)",
                        (PROJECT_ID, EVENT, state["output_id"], json.dumps(state)))
    return state


def begin(request_id):
    previous = query_one("SELECT meta FROM autonomous_project_events WHERE project_id=%s "
                         "AND event_type=%s ORDER BY id DESC LIMIT 1", (PROJECT_ID, EVENT))
    if previous and previous["meta"]["attempts"][-1]["status"] == "running":
        # A process died after reservation (possibly after a successful write).
        # Reconcile durable receipts before consuming the next slot.
        complete(previous["meta"]["attempts"][-1]["request_id"], error="interrupted process")
    return _mutate(lambda previous, cur: reserve(previous, request_id))


def matches_output(meta, output_id):
    """Match persisted publication identity, never an assistant's success prose."""
    meta = meta or {}
    filename = meta.get("filename")
    path_tail = urlparse(str(meta.get("public_url") or "")).path.rstrip("/").rsplit("/", 1)[-1]
    return filename in (output_id, output_id + ".md") or path_tail in (output_id, output_id + ".md")


def complete(request_id, error=None):
    def change(state, cur):
        if state["attempts"][-1]["request_id"] != request_id:
            raise RuntimeError("practice attempt identity mismatch")
        cur.execute("SELECT id FROM autonomous_project_events WHERE project_id=%s AND event_type=%s "
                    "AND meta->'attempts'->-1->>'request_id'=%s ORDER BY id LIMIT 1",
                    (PROJECT_ID, EVENT, request_id))
        start_id = cur.fetchone()[0]
        cur.execute("SELECT id,event_type,meta FROM autonomous_project_events WHERE project_id=%s "
                    "AND id>%s AND event_type IN ('note_added','research_draft_staged',"
                    "'publication_reviewed','publication_review_error','publication_created') ORDER BY id",
                    (PROJECT_ID, start_id))
        evidence = {}
        for event_id, kind, meta in cur.fetchall():
            if kind != "note_added" and not matches_output(meta, state["output_id"]):
                continue
            evidence.setdefault(kind, []).append({"event_id": event_id, "meta": meta})
        return finish(state, evidence, error)
    # An operator may pause/archive the project while an already-reserved tick
    # is finishing. Preserve its receipts without reopening or reserving work.
    return _mutate(change, allow_inactive=True)


GUIDANCE = (
    "데이터는 근거 재료다. define_practice_output으로 주제·독자·목적·형태를 명시한다. "
    "발표 대기는 독립 목표로 금지한다. 막힌 자료는 이유를 기록하고 가능한 산출 작업으로 전환한다. "
    "편집진단 재검토 및 모델 자기 채점으로 가치 달성을 보고하지 마라. "
    "심화 상한은 검증 통과가 아니다. 불확실성은 표시하고 사실 오류는 수정한다. "
    "초안 저장·심사·실제 발행은 별개다. 가치 지표는 계측 없으면 unknown이다. "
    "기존 #3 자산은 출처·날짜와 함께 읽기 전용으로 참조한다."
)


def objective(state):
    phase = state["attempts"][-1]["phase"]
    task = {
        "a": "의제 탐색 1회: 기존 근거로 구체적 산출물을 정하고 대상 독자·사용 목적·전달 형태를 등록한 뒤 초안을 저장한다.",
        "b": "검증·심화: 등록된 산출물의 사실 오류를 교정하고 초안을 저장한다. 심화는 전체 최대 2회다.",
        "c": "발행·연결 1회: 검증된 산출물을 기존 소유 매체로 발행·연결한다. 막히면 이유와 미발행 상태를 저장하고 이 주제를 닫는다.",
    }[phase]
    return (f"OBJECTIVE: {task}\nARTIFACT: {state['output_id']}\n"
            f"STATE: {json.dumps(state, ensure_ascii=False)}\n"
            + GUIDANCE)


def build_tools(state):
    async def define_practice_output(topic="", audience="", purpose="", form=""):
        values = {"topic": topic, "audience": audience, "purpose": purpose, "form": form}
        if any(not isinstance(v, str) or not v.strip() or len(v) > 500 for v in values.values()):
            return "error: topic, audience, purpose, form are required (1–500 characters)"
        problem = validate_plan([topic, purpose], [])
        if problem:
            return problem
        def change(current, cur):
            if current["output_id"] != state["output_id"]:
                raise RuntimeError("output identity mismatch")
            if current.get("descriptor") and current["descriptor"] != values:
                raise ValueError("descriptor is immutable; finish this output before another topic")
            cur.execute("SELECT 1 FROM autonomous_project_events WHERE project_id=%s AND event_type=%s "
                        "AND meta->>'output_id'<>%s AND meta->'descriptor'->>'topic'=%s LIMIT 1",
                        (PROJECT_ID, EVENT, state["output_id"], topic))
            if cur.fetchone():
                raise ValueError("topic already attempted; choose another actionable topic")
            current["descriptor"] = values
            return current
        try:
            updated = _mutate(change)
        except ValueError as exc:
            return f"error: {exc}"
        state.update(updated)
        return f"ok: artifact slug={state['output_id']}; descriptor saved"
    return [{"name": "define_practice_output", "description": "Register this output's immutable topic, audience, purpose and delivery form.",
             "input_schema": {"type": "object", "properties": {
                 key: {"type": "string"} for key in ("topic", "audience", "purpose", "form")},
                 "required": ["topic", "audience", "purpose", "form"]}}], {"define_practice_output": define_practice_output}


def guard_handlers(handlers, state):
    """Keep normal publication handlers/security gates; bound their invocation.

Only this output's generated slug is writable, preventing inherited #3 edits.
"""
    wrapped = dict(handlers)
    phase = state["attempts"][-1]["phase"]
    publication_attempted = False
    def wrap(_name, _handler):
        async def guarded(**kwargs):
            nonlocal publication_attempted
            if _name in ("edit_content", "set_project_state"):
                return "error: this loop cannot edit existing assets or change operator lifecycle state"
            if not state.get("descriptor"):
                return "error: define_practice_output first"
            if kwargs.get("slug") != state["output_id"]:
                return f"error: only current artifact slug {state['output_id']} is writable"
            action = kwargs.get("action")
            staging = _name == "research_document" and action in ("stage_public", "edit_staged")
            publishing = _name != "research_document" or action == "publish_public"
            if not staging and not publishing:
                return "error: only stage_public/edit_staged/publish_public allowed for this output"
            if publishing:
                if phase != "c":
                    return "error: publication belongs to phase c"
                if publication_attempted:
                    return "error: publication attempt cap reached; record blocker and close unpublished"
                publication_attempted = True  # charge failures as well as successes
            return await _handler(**kwargs)
        return guarded
    for name in ("research_document", "publish_hub_curation", "publish_static_page", "edit_content", "set_project_state"):
        if name in wrapped:
            wrapped[name] = wrap(name, wrapped[name])
    return wrapped
