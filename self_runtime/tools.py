"""self_runtime.tools — Self-awareness tools for Cyber-Lenin.

All tool handlers delegate to shared.py memory access functions,
so the same data is accessible from any module (telegram, chatbot, diary).

Integration in telegram_bot.py:
    from self_runtime.tools import SELF_TOOLS, SELF_TOOL_HANDLERS
    _TOOLS.extend(SELF_TOOLS)
    _TOOL_HANDLERS.update(SELF_TOOL_HANDLERS)
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))


_DELEGATABLE_AGENTS = [
    "analyst",
    "programmer",
    "scout",
    "visualizer",
    "browser",
    "diplomat",
    "diary",
]

# Optional per-delegation verification policy, persisted to task metadata and
# consumed by telegram/tasks._normalize_verification_policy after completion.
_VERIFICATION_POLICY_SCHEMA = {
    "type": "object",
    "description": (
        "Optional post-completion verification policy. Omit to use the per-agent "
        "default (programmer/analyst/scout/diplomat reports are verified). "
        "Set required=false to opt this task out."
    ),
    "properties": {
        "required": {"type": "boolean", "description": "Set false to skip verification for this task."},
        "checks": {
            "type": "array",
            "items": {"type": "string", "enum": ["task_report", "url_access", "server_logs"]},
            "description": "Which checks the verifier should run.",
        },
        "urls": {
            "type": "array",
            "items": {"type": "string"},
            "description": "URLs that must respond without HTTP errors after the task.",
        },
        "log_service": {
            "type": "string",
            "enum": ["telegram", "api", "nginx"],
            "description": "Service whose logs the verifier should inspect for task-related errors.",
        },
        "log_grep": {"type": "string", "description": "Optional grep pattern for the log check."},
        "retry_limit": {
            "type": "integer",
            "description": "Max auto-retries after verification failure (0-3, enforce mode only).",
        },
    },
}

_AGENT_ROUTING_CARDS = {
    "analyst": {
        "use_for": [
            "research, analysis, fact-checking, KG/vector/web synthesis",
            "publishing or editing research documents, task reports, blog posts, and hub curations",
            "admin-only private research documents",
        ],
        "do_not_use_for": [
            "source-code changes or service debugging",
            "posting/commenting on external social platforms",
            "interactive browser login/form workflows",
        ],
        "signature_tools": [
            "knowledge_graph_search", "vector_search", "web_search", "fetch_url",
            "research_document", "edit_content",
        ],
    },
    "programmer": {
        "use_for": [
            "source-code, configuration, scripts, templates, frontend behavior, tests, deployment debugging",
            "filesystem inspection or modification inside the project workspace",
        ],
        "do_not_use_for": [
            "routine edits to already-published diary/research/report/post/curation content",
            "research-only tasks that do not require code changes",
        ],
        "signature_tools": ["Codex CLI shell/read/write/exec in delegated mode", "list_agent_tools"],
    },
    "scout": {
        "use_for": [
            "Moltbook and mersoom.com activity",
            "external platform reconnaissance and raw collection for later analysis",
            "large-scale patrol/crawling workflows",
        ],
        "do_not_use_for": ["deep synthesis of collected material", "source-code changes"],
        "signature_tools": ["moltbook", "mersoom", "web_search", "fetch_url", "write_file"],
    },
    "browser": {
        "use_for": [
            "interactive website automation: login, forms, multi-page navigation, dynamic extraction",
        ],
        "do_not_use_for": ["plain URL reading that fetch_url can handle", "code changes"],
        "signature_tools": ["browser worker / Playwright automation"],
    },
    "visualizer": {
        "use_for": ["image generation, visual direction, visual asset variants"],
        "do_not_use_for": ["text-only research", "source-code changes"],
        "signature_tools": ["generate_image", "download_image", "upload_to_r2"],
    },
    "diary": {
        "use_for": [
            "new first-person Cyber-Lenin diary entries",
            "edits, deletion, and unpublishing of already-published diary entries",
        ],
        "do_not_use_for": ["research/report/post/curation edits", "code changes"],
        "signature_tools": ["save_diary", "edit_content(content_type='diary')"],
    },
    "diplomat": {
        "use_for": ["email and agent-to-agent diplomatic communication"],
        "do_not_use_for": ["public site content edits", "code changes"],
        "signature_tools": ["send_email", "a2a_send"],
    },
}

_CONTENT_STORE_GUIDE = {
    "research_document": {
        "content_type": "public long-form research document",
        "identifier": "research slug or filename",
        "read": "read_self(content_type='research_document', slug='<slug>')",
        "write_or_edit": "analyst: research_document",
        "not_this": "Not task_report and not static_page.",
    },
    "private_research_document": {
        "content_type": "admin-only private research document",
        "identifier": "private research document slug or document_id",
        "read": "read_self(content_type='private_research_document', slug='<slug>')",
        "write_or_edit": "analyst: research_document(action='save_private'|'publish_private')",
        "not_this": "Not a public task report. Do not make public unless explicitly asked.",
    },
    "task_report": {
        "content_type": "completed Telegram task/report output",
        "identifier": "numeric task_id",
        "read": "read_self(content_type='task_report', id=<id>)",
        "write_or_edit": "analyst: edit_content(content_type='task_report', id=<task_id>, field='content'|'result', ...)",
        "not_this": "Not research_document. Do not use research_document for task reports.",
    },
    "static_page": {
        "content_type": "custom HTML/static page",
        "identifier": "static page slug",
        "read": "read_self(content_type='static_page', slug='<slug>')",
        "write_or_edit": "analyst/autonomous_project: publish_static_page for new pages, edit_content(content_type='static_page', slug='<slug>', ...) for existing pages",
        "not_this": "Not markdown research, task report, curation, or diary.",
    },
    "blog_post": {
        "content_type": "public blog post",
        "identifier": "numeric post_id",
        "read": "read_self(content_type='blog_post', id=<id>) when available; otherwise use site frontend.",
        "write_or_edit": "analyst: edit_content(content_type='blog_post', id=<id>, ...)",
        "not_this": "Not task_report and not research_document.",
    },
    "diary": {
        "content_type": "Cyber-Lenin diary entry",
        "identifier": "numeric diary/post id",
        "read": "read_self(content_type='diary', id=<id>)",
        "write_or_edit": "diary: save_diary only for the exact scheduled diary prompt; edit_content(content_type='diary') for edit/delete/unpublish",
        "not_this": "Do not route diary edits to analyst/programmer unless explicitly needed.",
    },
    "hub_curation": {
        "content_type": "hub curation entry",
        "identifier": "curation slug",
        "read": "read_self(content_type='hub_curation', slug='<slug>')",
        "write_or_edit": "analyst: edit_content(content_type='hub_curation', slug='<slug>', ...)",
        "not_this": "Not static_page and not research_document.",
    },
}


_PUBLIC_CONTENT_TERMS = (
    "published", "already-published", "public post", "public content", "diary",
    "research", "report", "blog post", "hub", "curation", "slug", "post_id",
    "게시", "공개", "발행", "일기", "연구", "보고서", "블로그", "허브", "큐레이션",
    "문구", "오타", "수정",
)

_CODE_TERMS = (
    "code", "source", "bug", "test", "script", "config", "template", "frontend",
    "backend", "deploy", "service", "traceback", "exception", "repository",
    "코드", "소스", "버그", "테스트", "스크립트", "설정", "템플릿", "프론트",
    "백엔드", "배포", "서비스", "에러", "예외", "저장소",
)

_PRIVATE_RESEARCH_TERMS = (
    "private research", "private_research", "private document", "private report",
    "admin-only", "비공개 연구", "비공개 문서", "비공개 보고", "비공개 리서치",
    "사적 연구", "내부 연구", "관리자 전용",
)

_PROGRAMMER_EXPLICIT_TERMS = (
    "programmer", "coding agent", "code agent", "developer",
    "프로그래머", "코더", "개발자",
)

_PIPELINE_ERROR_TERMS = (
    "pipeline", "workflow", "scheduler", "cron", "route", "routing",
    "failure", "failed", "error", "bug", "regression",
    "파이프라인", "워크플로우", "스케줄러", "크론", "라우팅", "분기",
    "실패", "오류", "에러", "버그", "결함",
)


def _to_kst(ts) -> str:
    """Convert a timestamp (datetime or ISO string) to KST formatted string."""
    if ts is None:
        return "?"
    if isinstance(ts, str):
        if not ts or ts == "?":
            return ts
        # ISO string from APIs (e.g. "2026-03-14T03:00:57Z")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.astimezone(_KST).strftime("%m/%d %H:%M KST")
        except (ValueError, TypeError):
            return ts[:19].replace("T", " ")
    if hasattr(ts, "astimezone"):
        return ts.astimezone(_KST).strftime("%m/%d %H:%M KST")
    if hasattr(ts, "strftime"):
        return ts.strftime("%m/%d %H:%M")
    return str(ts)


def _slice_text(text: str, max_chars: int | None = None, offset: int | None = None) -> tuple[str, int, int, bool]:
    """Return a character slice plus pagination metadata."""
    try:
        start = max(0, int(offset or 0))
    except (TypeError, ValueError):
        start = 0
    start = min(start, len(text))

    if max_chars is None:
        end = len(text)
    else:
        try:
            length = max(0, int(max_chars))
        except (TypeError, ValueError):
            length = 0
        end = min(len(text), start + length)

    return text[start:end], start, end, end < len(text)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(term.lower() in lowered for term in terms)


def _routing_warning(agent: str, task: str, context: str = "") -> str | None:
    """Return a concise warning when a delegation is probably misrouted."""
    text = f"{task}\n{context}"
    if agent == "programmer" and _contains_any(text, _PUBLIC_CONTENT_TERMS) and not _contains_any(text, _CODE_TERMS):
        if "diary" in text.lower() or "일기" in text:
            return (
                "Probable misroute: already-published diary content should go to "
                "diary, which owns edit_content(content_type='diary'), not programmer."
            )
        return (
            "Probable misroute: already-published research/report/post/hub content "
            "should go to analyst, which owns edit_content/research_document, not programmer."
        )
    if agent == "analyst" and _contains_any(text, _CODE_TERMS) and not _contains_any(text, _PUBLIC_CONTENT_TERMS):
        return "Probable misroute: source-code/config/test/service work should go to programmer, not analyst."
    if agent == "diary" and any(term in text.lower() for term in ("research", "report", "blog post", "hub", "curation")):
        return "Probable misroute: non-diary public content edits should go to analyst, not diary."
    return None


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _normalize_llm_route(parsed: dict, task: str, candidates: list[str] | None) -> dict | None:
    allowed = set(candidates or _DELEGATABLE_AGENTS)
    agent = str(parsed.get("recommended_agent") or "").strip().lower()
    if agent not in allowed:
        return None
    confidence = str(parsed.get("confidence") or "medium").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    content_type = str(parsed.get("content_type") or "unknown").strip()
    reason = str(parsed.get("reason") or "LLM routing classifier recommendation.").strip()

    def string_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    card = _AGENT_ROUTING_CARDS.get(agent, {})
    result = {
        "recommended_agent": agent,
        "confidence": confidence,
        "reason": reason,
        "content_type": content_type,
        "needs_identifier": bool(parsed.get("needs_identifier", False)),
        "required_capabilities": string_list(parsed.get("required_capabilities")),
        "forbidden_assumptions": string_list(parsed.get("forbidden_assumptions")),
        "routing_class": str(parsed.get("routing_class") or content_type or "unknown").strip(),
        "routing_card": card,
        "alternatives": [
            {"agent": name, "use_for": _AGENT_ROUTING_CARDS.get(name, {}).get("use_for", [])}
            for name in _DELEGATABLE_AGENTS
            if name != agent and name in allowed
        ][:4],
        "source": "llm_classifier",
    }
    warning = _routing_warning(agent, task)
    if warning:
        result["warning"] = warning
    return result


async def _classify_route_with_llm(task: str, candidates: list[str] | None = None) -> dict | None:
    try:
        from bot_config import (
            _deepseek_client,
            _openai_client,
            _resolve_deepseek_model,
            _resolve_openai_model,
        )

        client = _deepseek_client or _openai_client
        if not client:
            return None
        provider = "deepseek" if _deepseek_client else "openai"
        model = (
            _resolve_deepseek_model("deepseek_flash")
            if provider == "deepseek"
            else _resolve_openai_model("gpt54nano")
        )
        allowed = [agent for agent in _DELEGATABLE_AGENTS if not candidates or agent in candidates]
        system = (
            "You are a strict task router for LeninBot. Return only JSON. "
            "Pick one class: public_content_edit, code_config_work, research, diary, "
            "browser_automation, external_platform_scout, email_a2a. "
            "Map public_content_edit/research to analyst except diary content to diary; "
            "code_config_work to programmer; browser_automation to browser; "
            "external_platform_scout to scout; email_a2a to diplomat. "
            "If the user explicitly asks for programmer/developer handling and the request is about "
            "a failure, runtime pipeline, scheduler, routing, code, config, or test problem, choose programmer "
            "even when a public content type such as diary is mentioned."
        )
        user = {
            "task": task,
            "candidate_agents": allowed,
            "required_json_keys": [
                "recommended_agent",
                "confidence",
                "reason",
                "content_type",
                "needs_identifier",
                "required_capabilities",
                "forbidden_assumptions",
                "routing_class",
            ],
        }
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=500,
            ),
            timeout=12,
        )
        parsed = _extract_json_object(response.choices[0].message.content)
        if not parsed:
            return None
        return _normalize_llm_route(parsed, task, candidates)
    except Exception as e:
        logger.info("route_task LLM classifier unavailable: %s", e)
        return None


def _recommend_agent_for_task(task: str, *, candidates: list[str] | None = None) -> dict:
    """Heuristic routing aid for the orchestrator; advisory, not authoritative."""
    text = task or ""
    lowered = text.lower()
    allowed = set(candidates or _DELEGATABLE_AGENTS)

    def choose(agent: str, reason: str, confidence: str = "medium") -> dict:
        card = _AGENT_ROUTING_CARDS.get(agent, {})
        return {
            "recommended_agent": agent,
            "confidence": confidence,
            "reason": reason,
            "source": "heuristic",
            "routing_card": card,
            "alternatives": [
                {"agent": name, "use_for": _AGENT_ROUTING_CARDS.get(name, {}).get("use_for", [])}
                for name in _DELEGATABLE_AGENTS
                if name != agent and name in allowed
            ][:4],
        }

    explicit_programmer = _contains_any(text, _PROGRAMMER_EXPLICIT_TERMS)
    pipeline_or_code_fix = _contains_any(text, _PIPELINE_ERROR_TERMS) or _contains_any(text, _CODE_TERMS)
    if explicit_programmer and pipeline_or_code_fix and "programmer" in allowed:
        return choose(
            "programmer",
            "The user explicitly requested programmer handling for code, routing, pipeline, or failure work.",
            "high",
        )

    if ("diary" in lowered or "일기" in text) and pipeline_or_code_fix and "programmer" in allowed:
        return choose(
            "programmer",
            "This is about the diary pipeline/runtime failing, not a routine diary content edit.",
            "high",
        )

    if {"diary", "일기"} & set(lowered.replace("/", " ").split()):
        if any(term in lowered for term in ("edit", "fix", "correct", "수정", "고쳐", "오타")):
            return choose("diary", "The task is about creating or editing diary content.", "high")
    if _contains_any(text, _PRIVATE_RESEARCH_TERMS):
        return choose(
            "analyst",
            "Private research documents are owned by analyst; publish only when explicitly requested.",
            "high",
        )
    if _contains_any(text, _PUBLIC_CONTENT_TERMS) and not _contains_any(text, _CODE_TERMS):
        if "diary" in lowered or "일기" in text:
            return choose("diary", "Already-published diary content is owned by the diary agent.", "high")
        return choose(
            "analyst",
            "Published research/report/blog/hub content is operational content, and analyst owns the edit tools.",
            "high",
        )
    if _contains_any(text, _CODE_TERMS):
        return choose("programmer", "The task requires source-code/config/test/service work.", "high")
    if any(term in lowered for term in ("moltbook", "mersoom", "patrol", "crawl", "recon", "정찰", "순찰")):
        return choose("scout", "The task is external platform reconnaissance or posting.", "high")
    if any(term in lowered for term in ("login", "form", "browser", "click", "dynamic", "로그인", "브라우저")):
        return choose("browser", "The task requires interactive browser automation.", "high")
    if any(term in lowered for term in ("image", "visual", "poster", "generate_image", "이미지", "포스터", "시각")):
        return choose("visualizer", "The task is visual generation or direction.", "high")
    if any(term in lowered for term in ("email", "a2a", "diplomat", "메일", "외교")):
        return choose("diplomat", "The task is email or A2A communication.", "medium")
    return choose("analyst", "Default route for information analysis, research, and synthesis.", "medium")


def _resolve_recent_operator_user_id() -> int:
    """Best-effort owner/user scope for mission lookup from recent Telegram chat."""
    try:
        from db import query as _db_q

        recent_user = _db_q(
            "SELECT user_id FROM telegram_chat_history "
            "WHERE user_id != 0 ORDER BY id DESC LIMIT 1"
        )
        if recent_user:
            return int(recent_user[0]["user_id"])
    except Exception:
        pass
    return 0


def _format_delegation_contract(
    *,
    success_criteria: str = "",
    required_capabilities: list[str] | None = None,
    target_identifiers: list[str] | None = None,
    forbidden_assumptions: list[str] | None = None,
) -> str:
    parts = []
    if success_criteria:
        parts.append(f"Success criteria: {success_criteria}")
    if required_capabilities:
        parts.append("Required capabilities/tools: " + ", ".join(map(str, required_capabilities)))
    if target_identifiers:
        parts.append("User-supplied target identifiers: " + ", ".join(map(str, target_identifiers)))
    if forbidden_assumptions:
        parts.append("Forbidden assumptions: " + "; ".join(map(str, forbidden_assumptions)))
    if not parts:
        return ""
    return "<delegation-contract>\n" + "\n".join(parts) + "\n</delegation-contract>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. TOOL DEFINITIONS (Anthropic API format)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SELF_TOOLS = [
    {
        "name": "read_self",
        "description": (
            "Read internal content by content type. This is the only exposed read/query "
            "tool for Cyber-Lenin's own content and runtime state. Use content_type, "
            "then id/slug/keyword/limit/offset/max_chars as needed. If the user gives "
            "a public URL, use it only to infer the content type and identifier."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content_type": {
                    "type": "string",
                    "enum": [
                        "diary", "task_report", "research_document",
                        "private_research_document", "static_page", "blog_post",
                        "hub_curation", "chat_logs", "kg_status", "system_status",
                        "server_logs", "file_registry", "autonomous_project",
                    ],
                    "description": "Content/runtime type to read.",
                },
                "source": {
                    "type": "string",
                    "description": "Deprecated compatibility alias for content_type; do not use in new calls.",
                },
                "limit": {"type": "integer", "description": "Results count."},
                "keyword": {"type": "string", "description": "Filter keyword."},
                "max_chars": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "description": "Maximum body characters for detail reads. Default null returns the full body; diary and static_page detail reads are not tool-loop truncated.",
                },
                "offset": {"type": "integer", "description": "Character offset for paginating long detail reads."},
                "id": {"type": "integer", "description": "Numeric id for diary, task_report, blog_post, or autonomous_project detail."},
                "post_id": {"type": "integer", "description": "Deprecated alias for id on diary/blog_post."},
                "diary_id": {"type": "integer", "description": "For diary: alias of post_id."},
                "hours_back": {"type": "integer", "description": "Only last N hours."},
                "service": {"type": "string", "enum": ["telegram", "api", "nginx"], "description": "For server_logs."},
                "grep": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ],
                    "description": "For server_logs: filter text or list of texts.",
                },
                "status": {"type": "string", "enum": ["pending", "queued", "processing", "done", "failed"], "description": "For task_report: filter by status."},
                "task_id": {"type": "integer", "description": "Deprecated alias for id on task_report/autonomous_project."},
                "slug": {
                    "type": "string",
                    "description": (
                        "Slug for slug-addressed content types: research documents, static pages, "
                        "hub curations, or private research documents. Choose content_type first; do not "
                        "guess that every slug is a static page."
                    ),
                },
                "chat_source": {"type": "string", "enum": ["telegram", "web"], "description": "For chat_logs. Default: web."},
            },
            "required": ["content_type"],
        },
    },
    {
        "name": "write_kg",
        "description": (
            "DEPRECATED — use `write_kg_structured` for all new KG writes. "
            "This narrative-mode writer relies on LLM extraction, which is less "
            "reliable and loses the caller's type intent. Kept only for "
            "backward compatibility with historical callers; do not use."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "Factual statements to store. Use bullet points for multiple facts. "
                        "Example: '- US announced tightened semiconductor export controls against China on 2026-03-28\\n- Samsung Electronics stock fell 3.2%'"
                    ),
                },
                "group_id": {
                    "type": "string",
                    "enum": ["geopolitics_conflict", "diplomacy", "economy", "korea_domestic", "agent_knowledge"],
                    "description": "Topic group. Default: agent_knowledge.",
                    "default": "agent_knowledge",
                },
                "source_type": {
                    "type": "string",
                    "enum": ["internal_report", "osint_news", "osint_social", "personnel_change", "diplomatic_cable", "threat_report"],
                    "description": "Provenance tag for the stored facts. Default: internal_report.",
                    "default": "internal_report",
                },
                "supersedes": {
                    "type": "string",
                    "description": "Optional. If this fact corrects/refines an earlier episode, pass that episode name here. KG is append-only — old facts are not deleted, only marked as superseded.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "write_kg_structured",
        "description": (
            "Deterministic typed-triple writer for the KG — the canonical way "
            "to write facts. No LLM extraction — you pick every subject_type / "
            "predicate / object_type from the enums. Use for precise single-fact "
            "asserts, analyst conclusions, news facts, KG corrections. "
            "Existing entities are matched by exact name; unknown names create "
            "new nodes with your declared type. `fact` must be self-contained "
            "(e.g. 'Anthropic announced Claude Opus 4.6 on 2026-04-11'), "
            "because it is what gets embedded for vector search.\n\n"
            "PREDICATE RULES — pick by (subject_type → object_type) pair. "
            "Writes are rejected if the pair doesn't allow the predicate:\n"
            "  • Affiliation: Person→Org, Person→Role, Role→Org, Org→Industry\n"
            "  • PersonalRelation: Person→Person\n"
            "  • OrgRelation: Org→Org ONLY (not Org→Asset, not Org→Concept)\n"
            "  • Funding: any→any (wildcard)\n"
            "  • AssetTransfer: any→any (wildcard). Org→Asset uses this, NOT OrgRelation.\n"
            "  • ThreatAction: Person→Org, Org→Org, Org→Person, Campaign→Org, Campaign→Asset, Campaign→Industry\n"
            "  • Involvement: subject→Incident or subject→Campaign ONLY. "
            "Do NOT use for 'X is involved in Concept/Policy' — use Causation or flip direction.\n"
            "  • Presence: any→Location (Person/Org/Role/Incident/Campaign/Industry → Location)\n"
            "  • PolicyEffect: Policy→any (Policy MUST be subject), or Org→Policy (org enforces it), or Campaign→Policy\n"
            "  • Participation: Person→Campaign, Org→Campaign\n"
            "  • Statement: any→any (wildcard). For 'X said/announced/criticized Y'.\n"
            "  • Causation: any→any (wildcard). For explicit 'X caused Y'. Direction = cause→effect.\n\n"
            "DECISION SHORTCUT: if your pair isn't in the lists above, "
            "use a wildcard (Funding / AssetTransfer / Statement / Causation) "
            "or flip the direction (e.g. Org sanctioned by Policy → subject=Policy, predicate=PolicyEffect).\n\n"
            "COMMON MISTAKES TO AVOID:\n"
            "  ✗ Org→Concept with Involvement  → use Causation or Statement\n"
            "  ✗ Org→Asset with OrgRelation    → use AssetTransfer\n"
            "  ✗ Org→Policy with Involvement   → flip to Policy→Org with PolicyEffect"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "minItems": 1,
                    "description": "List of structured facts to write in one call. Invalid facts are rejected individually; valid facts are still stored. The result reports written_fact_indices and rejected_facts with original input index, reason, and fact so you can retry only the failed entries.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject_name": {"type": "string", "description": "Canonical English name of the subject entity."},
                            "subject_type": {
                                "type": "string",
                                "enum": ["Person", "Organization", "Location", "Asset",
                                         "Incident", "Policy", "Campaign", "Concept",
                                         "Role", "Industry"],
                                "description": "Person=actual human (not title), Organization=institution, Asset=tech/product/IP, Incident=time-bounded event, Policy=law/sanction/treaty, Campaign=sustained op, Concept=ideology/theory, Role=title (distinct from holder), Industry=sector.",
                            },
                            "predicate": {
                                "type": "string",
                                "enum": ["Affiliation", "PersonalRelation", "OrgRelation",
                                         "Funding", "AssetTransfer", "ThreatAction",
                                         "Involvement", "Presence", "PolicyEffect", "Participation",
                                         "Statement", "Causation"],
                                "description": "Must match the (subject_type → object_type) pair — see tool-level PREDICATE RULES. Wildcards (any→any): Funding, AssetTransfer, Statement, Causation.",
                            },
                            "object_name": {"type": "string", "description": "Canonical English name of the object entity."},
                            "object_type": {
                                "type": "string",
                                "enum": ["Person", "Organization", "Location", "Asset",
                                         "Incident", "Policy", "Campaign", "Concept",
                                         "Role", "Industry"],
                                "description": "Same semantic rules as subject_type.",
                            },
                            "fact": {
                                "type": "string",
                                "description": "Self-contained natural-language statement of the fact. Used as the edge's searchable text.",
                            },
                            "valid_at": {
                                "type": "string",
                                "description": "Optional ISO date (YYYY-MM-DD) for when the fact became true.",
                            },
                        },
                        "required": ["subject_name", "subject_type", "predicate",
                                     "object_name", "object_type", "fact"],
                    },
                },
                "group_id": {
                    "type": "string",
                    "enum": ["geopolitics_conflict", "diplomacy", "economy", "korea_domestic", "agent_knowledge"],
                    "description": "Topic group bucket for later retrieval. Default: agent_knowledge.",
                    "default": "agent_knowledge",
                },
            },
            "required": ["facts"],
        },
    },
    {
        "name": "delegate",
        "description": (
            "Dispatch an async task to one delegatable specialist agent. Allowed agents: "
            "analyst, programmer, scout, visualizer, browser, diplomat, diary. Stasova is "
            "not a general delegation target. When routing is unclear, call route_task or "
            "list_agent_tools first. Always pass `context` and prefer explicit "
            "success_criteria/target_identifiers so the worker does not infer the wrong content type. "
            "Completed tasks are independently verified by default (programmer/analyst/scout/diplomat); "
            "pass `verification` to tune checks or opt out."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": _DELEGATABLE_AGENTS,
                    "description": "Which delegatable specialist agent to use. Stasova is intentionally excluded.",
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Specific instructions for the agent. Include the user's goal, symptoms, "
                        "requirements, constraints, expected outcome, and any user-provided public URL, "
                        "slug, post_id, or DB document identifier. Do not invent or pass filesystem paths; "
                        "agents that need code context can inspect the repository themselves."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": "Delegation context: summarize the conversation that led to this delegation, "
                    "the user's original request, any discoveries or tool results so far, and why you chose this agent. "
                    "This helps the agent understand the full picture.",
                },
                "success_criteria": {
                    "type": "string",
                    "description": "Concrete done condition for the worker, e.g. 'correct the named research document and verify the typo is gone'.",
                },
                "required_capabilities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Capabilities/tools the task is expected to need, e.g. research_document, edit_content(content_type='task_report'), code editing.",
                },
                "target_identifiers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "User-supplied identifiers only: title, slug, post_id, task_id, report id, URL, error text. "
                        "Do not invent filesystem paths."
                    ),
                },
                "forbidden_assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Things the worker must not assume, e.g. 'do not treat /p/foo as research' or 'do not edit source code'.",
                },
                "parent_task_id": {"type": "integer", "description": "Parent task ID for task chaining (optional)."},
                "verification": _VERIFICATION_POLICY_SCHEMA,
            },
            "required": ["agent", "task"],
        },
    },
    {
        "name": "multi_delegate",
        "description": (
            "Delegate multiple tasks in parallel with automatic result synthesis.\n"
            "All subtasks run concurrently. After all complete, a synthesis task combines results.\n"
            "Use when you need multiple agents working on different aspects of the same request.\n"
            "For single-agent tasks, use `delegate` instead. Allowed agents exclude Stasova."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent": {
                                "type": "string",
                                "enum": _DELEGATABLE_AGENTS,
                            },
                            "task": {
                                "type": "string",
                                "description": (
                                    "Task instructions for this agent. Include goals, symptoms, constraints, "
                                    "and user-provided public URLs/slugs/post_ids/DB identifiers. Do not invent "
                                    "or pass filesystem paths."
                                ),
                            },
                            "context": {"type": "string", "description": "Why this subtask exists."},
                            "success_criteria": {"type": "string", "description": "Concrete done condition for this subtask."},
                            "required_capabilities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Expected capabilities/tools for this subtask.",
                            },
                            "target_identifiers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "User-supplied URL/slug/post_id/task_id/error text for this subtask.",
                            },
                            "forbidden_assumptions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Assumptions this subtask must not make.",
                            },
                            "verification": _VERIFICATION_POLICY_SCHEMA,
                        },
                        "required": ["agent", "task"],
                    },
                    "minItems": 2,
                    "description": "List of subtasks to run in parallel.",
                },
                "synthesis_instructions": {
                    "type": "string",
                    "description": "Instructions for combining subtask results into a final report.",
                },
            },
            "required": ["tasks"],
        },
    },
    {
        "name": "route_task",
        "description": (
            "Advisory routing helper for the orchestrator. It returns the recommended "
            "delegatable agent plus strict content-type guidance for diary, task report, "
            "research document, private research document, static page, blog post, and hub curation. "
            "Use before delegate when the request mentions research/report/static_page/private research document "
            "or when tool ownership is unclear. Public URLs, if present, are only parsed as "
            "hints to infer content type and identifier. This tool does not create a task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "User request or proposed delegated task."},
                "candidates": {
                    "type": "array",
                    "items": {"type": "string", "enum": _DELEGATABLE_AGENTS},
                    "description": "Optional candidate agents to compare. Stasova is intentionally excluded.",
                },
                "include_store_guide": {
                    "type": "boolean",
                    "description": "Include public/private/static/report storage boundary guide. Default true.",
                    "default": True,
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "list_agent_tools",
        "description": (
            "Return the runtime-visible tool list for the orchestrator and/or specialist agents. "
            "Use this before delegating when tool ownership is unclear. For routing decisions, "
            "route_task is usually shorter and includes content-type boundaries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": [
                        "all", "orchestrator", "analyst", "programmer", "scout",
                        "visualizer", "browser", "diplomat", "diary", "stasova", "autonomous_project",
                        "web_chat",
                    ],
                    "description": "Which runtime to inspect. Default/all returns orchestrator plus all agents.",
                    "default": "all",
                },
                "include_descriptions": {
                    "type": "boolean",
                    "description": "Whether to include short tool descriptions. Default true.",
                    "default": True,
                },
                "include_schemas": {
                    "type": "boolean",
                    "description": "Whether to include input schemas. Default false to keep output compact.",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "recall_experience",
        "description": "Search your experiential memory (past lessons, mistakes, insights, patterns). Stored daily from all conversations and tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What experience to recall (semantic search)."},
                "limit": {"type": "integer", "description": "Max results (1-10).", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "save_self_analysis",
        "description": (
            "Actively index your own high-quality analysis into the "
            "self_produced_analysis vector layer. Use when an insight, synthesis, "
            "framework, or reusable argument is worth preserving beyond raw logs "
            "and daily experiential summaries. Later retrieve it with "
            "vector_search(layer=\"self_produced_analysis\")."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short descriptive title for this saved analysis.",
                },
                "content": {
                    "type": "string",
                    "description": "Self-contained analytical text to preserve. Include enough context for future retrieval.",
                },
                "category": {
                    "type": "string",
                    "enum": ["insight", "framework", "synthesis", "critique", "strategy", "method", "other"],
                    "description": "What kind of analysis this is. Default: insight.",
                    "default": "insight",
                },
                "source_context": {
                    "type": "string",
                    "description": "Optional short provenance note, e.g. task ID, conversation theme, or why it matters.",
                },
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "kg_admin",
        "description": "KG admin ops. action: query (Cypher), delete_episode, merge_entities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["query", "delete_episode", "merge_entities"],
                    "description": "query: run Cypher (needs `query`). delete_episode: remove by episode name. merge_entities: fold `source_name` into `target_name`.",
                },
                "query": {"type": "string", "description": "Cypher query (action=query)."},
                "write": {"type": "boolean", "description": "Allow writes (action=query). Default false."},
                "episode_name": {"type": "string", "description": "Episode name (action=delete_episode)."},
                "source_name": {"type": "string", "description": "Entity to merge FROM (action=merge_entities)."},
                "target_name": {"type": "string", "description": "Entity to merge INTO (action=merge_entities)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "run_agent",
        "description": (
            "Run a sub-agent synchronously and get the result immediately in this turn.\n"
            "Use for quick analysis/lookup tasks. For long-running tasks, use delegate.\n"
            "Budget-capped at $0.50, max 10 rounds. Cost deducted from your budget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": ["analyst"],
                    "description": "Agent to run (currently analyst only).",
                },
                "task": {"type": "string", "description": "Task instructions."},
                "context": {"type": "string", "description": "Context for the agent."},
                "budget_usd": {"type": "number", "description": "Budget cap (max $0.50).", "default": 0.30},
            },
            "required": ["agent", "task"],
        },
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. TOOL HANDLERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _exec_read_diary(
    limit: int = 5,
    keyword: str | None = None,
    max_chars: int | None = None,
    offset: int | None = None,
    post_id: int | None = None,
    diary_id: int | None = None,
) -> str:
    from memory_store.queries import fetch_diaries

    target_id = post_id if post_id is not None else diary_id
    diaries = await asyncio.to_thread(fetch_diaries, limit, keyword, target_id)
    if not diaries:
        if target_id is not None:
            msg = f"No diary entry found with id={target_id}."
        else:
            msg = f"No diary entries matching '{keyword}'." if keyword else "No diary entries found."
        return msg

    results = []
    for i, d in enumerate(diaries, 1):
        entry_id = d.get("id")
        ts = _to_kst(d.get("created_at"))
        updated = _to_kst(d.get("updated_at"))
        title = d.get("title", "Untitled")
        content = d.get("content", "")
        slice_header = ""
        if max_chars is not None:
            body, start, end, truncated = _slice_text(content, max_chars=max_chars, offset=offset)
            next_hint = (
                f"\nnext: read_self(content_type='diary', id={entry_id}, offset={end}, max_chars={max_chars})"
                if truncated
                else ""
            )
            slice_header = f"Content chars={len(content)} returned_chars={start}:{end} truncated={truncated}{next_hint}\n"
            content = body
        results.append(
            f"[{i}] ID: {entry_id} | URL: https://cyber-lenin.com/ai-diary/{entry_id}\n"
            f"Created: {ts} | Updated: {updated}\n"
            f"Title: {title}\n{slice_header}Content:\n{content}"
        )

    return f"Your diary entries ({len(diaries)} shown):\n\n" + "\n\n---\n\n".join(results)


async def _exec_read_blog_posts(
    limit: int = 5,
    keyword: str | None = None,
    max_chars: int | None = None,
    offset: int | None = None,
    post_id: int | None = None,
) -> str:
    from db import query as db_query

    clauses = []
    params: list = []
    if post_id is not None:
        clauses.append("id = %s")
        params.append(int(post_id))
    if keyword:
        clauses.append("(title ILIKE %s OR content ILIKE %s)")
        q = f"%{keyword}%"
        params.extend([q, q])
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(min(max(int(limit or 5), 1), 50))
    try:
        rows = await asyncio.to_thread(
            db_query,
            f"""
            SELECT id, title, content, created_at, updated_at
              FROM posts
              {where}
             ORDER BY created_at DESC, id DESC
             LIMIT %s
            """,
            tuple(params),
        )
    except Exception as e:
        return f"Error reading blog posts: {type(e).__name__}: {e}"
    if not rows:
        if post_id is not None:
            return f"No blog post found with id={post_id}."
        return f"No blog posts matching '{keyword}'." if keyword else "No blog posts found."

    results = []
    for i, row in enumerate(rows, 1):
        content = row.get("content") or ""
        slice_header = ""
        if max_chars is not None:
            body, start, end, truncated = _slice_text(content, max_chars=max_chars, offset=offset)
            next_hint = (
                f"\nnext: read_self(content_type='blog_post', id={row['id']}, offset={end}, max_chars={max_chars})"
                if truncated
                else ""
            )
            slice_header = f"Content chars={len(content)} returned_chars={start}:{end} truncated={truncated}{next_hint}\n"
            content = body
        results.append(
            f"[{i}] ID: {row['id']} | URL: https://cyber-lenin.com/post/{row['id']}\n"
            f"Created: {_to_kst(row.get('created_at'))} | Updated: {_to_kst(row.get('updated_at'))}\n"
            f"Title: {row.get('title') or 'Untitled'}\n{slice_header}Content:\n{content}"
        )
    return f"Blog posts ({len(rows)} shown):\n\n" + "\n\n---\n\n".join(results)


async def _exec_read_chat_logs(
    limit: int = 20, hours_back: int | None = None, keyword: str | None = None,
    source: str = "web",
) -> str:
    from memory_store.queries import fetch_chat_logs

    normalized_source = (source or "web").strip().lower()
    rows = await asyncio.to_thread(
        fetch_chat_logs,
        limit,
        hours_back,
        keyword,
        source=source,
        group_web_contexts=(normalized_source == "web"),
        per_context_limit=10,
    )
    if not rows:
        return "No chat logs found for the specified criteria."

    output_budget = 45_000

    def _within_budget(text: str) -> bool:
        return len(text) <= output_budget

    def _join_with_budget(prefix: str, chunks: list[str]) -> tuple[str, int]:
        """Join complete chunks while keeping each chat turn intact."""
        included: list[str] = []
        omitted = 0
        current = prefix
        for chunk in chunks:
            separator = "\n\n" if current else ""
            if not included and not _within_budget(current + separator + chunk):
                current = current + separator + chunk
                included.append(chunk)
                continue
            candidate = current + separator + chunk
            if _within_budget(candidate):
                included.append(chunk)
                current = candidate
                continue
            omitted += 1
        if omitted:
            note = f"\n\n... ({omitted} older complete turn/session block(s) omitted to keep output bounded)"
            if _within_budget(current + note):
                current += note
        return current, omitted

    def _join_recent_with_budget(prefix: str, chunks: list[str]) -> tuple[str, int]:
        """Keep newest complete chunks, then render the retained set chronologically."""
        selected: list[str] = []
        omitted = 0
        for chunk in reversed(chunks):
            candidate_chunks = [chunk] + selected
            candidate = prefix + ("\n\n" if candidate_chunks else "") + "\n\n".join(candidate_chunks)
            if not selected and not _within_budget(candidate):
                selected = [chunk]
                continue
            if _within_budget(candidate):
                selected = candidate_chunks
                continue
            omitted += 1
        current = prefix + ("\n\n" if selected else "") + "\n\n".join(selected)
        if omitted:
            note = f"\n\n... ({omitted} older complete turn(s) omitted to keep output bounded)"
            if _within_budget(current + note):
                current += note
        return current, omitted

    def _sort_key(ts) -> float:
        if ts is None:
            return 0.0
        if hasattr(ts, "timestamp"):
            try:
                return float(ts.timestamp())
            except Exception:
                return 0.0
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0
        return 0.0

    results = []
    if normalized_source == "web":
        sessions: dict[str, dict] = {}
        for row in rows:
            session_id = str(row.get("session_id", "") or "").strip() or "unknown"
            fingerprint = str(row.get("fingerprint", "") or "").strip() or "unknown"
            bucket_key = f"{fingerprint}::{session_id}"
            bucket = sessions.setdefault(
                bucket_key,
                {
                    "rows": [],
                    "latest": row.get("created_at"),
                    "session_id": session_id,
                    "fingerprint": fingerprint,
                },
            )
            bucket["rows"].append(row)
            latest = bucket.get("latest")
            created = row.get("created_at")
            if _sort_key(created) > _sort_key(latest):
                bucket["latest"] = created

        ordered_sessions = sorted(
            sessions.items(),
            key=lambda item: _sort_key(item[1].get("latest")),
            reverse=True,
        )

        for _, bucket in ordered_sessions:
            session_id = bucket.get("session_id") or "unknown"
            fingerprint = bucket.get("fingerprint") or "unknown"
            session_rows = sorted(
                bucket["rows"],
                key=lambda r: _sort_key(r.get("created_at")),
            )
            latest_ts = _to_kst(bucket.get("latest"))
            header_lines = [
                f"=== Web visitor fingerprint {fingerprint} | session {session_id} | {len(session_rows)} entr{'y' if len(session_rows) == 1 else 'ies'} | latest {latest_ts} ===",
                "Treat this as one visitor context. Do not merge it with other web sessions.",
            ]
            turn_chunks = []
            for row in session_rows:
                ts = _to_kst(row.get("created_at"))
                role = str(row.get("role", "") or "").lower()
                content = str(row.get("content", "") or "")
                q = str(row.get("user_query", "") or "")
                a = str(row.get("bot_answer", "") or "")
                lines = []
                if role in ("user", "assistant") and content:
                    speaker = "Visitor" if role == "user" else "Lenin"
                    lines.append(f"[{ts}] {speaker}: {content}")
                    turn_chunks.append("\n".join(lines))
                    continue
                if q:
                    lines.append(f"[{ts}] Visitor: {q}")
                if a:
                    lines.append(f"[{ts}] Lenin: {a}")
                if not q and not a and content:
                    speaker = "Visitor" if role == "user" else "Lenin"
                    lines.append(f"[{ts}] {speaker}: {content}")
                if lines:
                    turn_chunks.append("\n".join(lines))

            session_prefix = "\n".join(header_lines)
            session_text, omitted_turns = _join_recent_with_budget(
                session_prefix,
                turn_chunks,
            )
            if omitted_turns:
                session_text = session_text.replace(
                    f"{len(session_rows)} entr{'y' if len(session_rows) == 1 else 'ies'}",
                    f"{len(turn_chunks) - omitted_turns}/{len(session_rows)} entries shown",
                    1,
                )
            results.append(session_text)

        prefix = (
            f"Web chat logs grouped by visitor fingerprint and session ({len(rows)} entries across {len(ordered_sessions)} contexts). "
            "Each fingerprint/session block is a separate visitor context; do not combine identities or intentions across blocks.\n\n"
        )
        result, _ = _join_with_budget(prefix.rstrip(), results)
        return result

    for row in rows:
        ts = _to_kst(row.get("created_at"))
        role = str(row.get("role", "") or "").lower()
        content = str(row.get("content", "") or "")

        if normalized_source == "telegram":
            header = f"[telegram] {ts}"
            q = str(row.get("user_query", "") or "")
            a = str(row.get("bot_answer", "") or "")
            lines = [header]

            if role in ("user", "assistant") and content:
                speaker = "Admin" if role == "user" else "Lenin"
                lines.append(f"{speaker}: {content}")
            else:
                if q:
                    lines.append(f"Admin: {q}")
                if a:
                    lines.append(f"Lenin: {a}")
                if not q and not a and content:
                    speaker = "Admin" if role == "user" else "Lenin"
                    lines.append(f"{speaker}: {content}")

            results.append("\n".join(lines))

    result, omitted = _join_with_budget(f"Chat logs ({len(rows)} entries):", results)
    if omitted:
        result = result.replace(
            f"Chat logs ({len(rows)} entries):",
            f"Chat logs ({len(results) - omitted}/{len(rows)} entries shown):",
            1,
        )
    return result


async def _exec_read_processing_logs(
    limit: int = 5, hours_back: int | None = None, keyword: str | None = None,
) -> str:
    from memory_store.queries import fetch_chat_logs

    rows = await asyncio.to_thread(
        fetch_chat_logs, limit, hours_back, keyword, include_logs=True,
    )
    if not rows:
        return "No processing logs found."

    results = []
    for i, row in enumerate(rows, 1):
        ts = _to_kst(row.get("created_at"))
        q = str(row.get("user_query", ""))[:150]
        route = row.get("route", "?")
        doc_cnt = row.get("documents_count", 0)
        web = row.get("web_search_used", False)
        strategy = str(row.get("strategy", "") or "")
        if len(strategy) > 400:
            strategy = strategy[:400] + "..."
        logs = str(row.get("processing_logs", "") or "")
        if len(logs) > 600:
            logs = logs[:600] + "..."

        results.append(
            f"[{i}] {ts} | route={route} | docs={doc_cnt} | web={web}\n"
            f"  Query: {q}\n"
            f"  Strategy: {strategy}\n"
            f"  Pipeline logs:\n{logs}"
        )

    return f"Processing logs ({len(rows)} entries):\n\n" + "\n\n---\n\n".join(results)


async def _exec_read_task_reports(
    limit: int = 5,
    status: str | None = None,
    task_id: int | None = None,
    max_chars: int | None = None,
    offset: int | None = None,
) -> str:
    # Single task full report
    if task_id:
        from db import query_one as _db_query_one
        row = await asyncio.to_thread(
            _db_query_one,
            "SELECT id, user_id, content, status, result, tool_log, agent_type, "
            "mission_id, parent_task_id, depth, created_at, completed_at "
            "FROM telegram_tasks WHERE id = %s",
            (task_id,),
        )
        if not row:
            return f"Task #{task_id} not found."
        ts = _to_kst(row.get("created_at"))
        completed = _to_kst(row.get("completed_at")) if row.get("completed_at") else "N/A"
        result = str(row.get("result") or "(no result)")
        tool_log = str(row.get("tool_log") or "")
        content = str(row.get("content") or "")
        report_header = ""
        if max_chars is not None:
            body, start, end, truncated = _slice_text(result, max_chars=max_chars, offset=offset)
            next_hint = (
                f"\nnext: read_self(content_type='task_report', id={row['id']}, offset={end}, max_chars={max_chars})"
                if truncated
                else ""
            )
            report_header = f"Report chars={len(result)} returned_chars={start}:{end} truncated={truncated}{next_hint}\n"
            result = body
        header = (
            f"Task #{row['id']} | status={row['status']} | agent={row.get('agent_type', '?')}\n"
            f"created={ts} | completed={completed}\n"
            f"mission_id={row.get('mission_id', 'N/A')} | parent={row.get('parent_task_id', 'N/A')} | depth={row.get('depth', 0)}\n"
            f"\n## Request\n{content[:1000]}\n"
            f"\n## Full Report\n{report_header}{result}"
        )
        if tool_log:
            header += f"\n\n## Tool Log\n{tool_log[:5000]}"
        return header

    # List mode
    from memory_store.queries import fetch_task_reports
    rows = await asyncio.to_thread(fetch_task_reports, limit, status)
    if not rows:
        return "No task reports found."

    results = []
    for i, row in enumerate(rows, 1):
        ts = _to_kst(row.get("created_at"))
        completed = _to_kst(row.get("completed_at")) if row.get("completed_at") else ""
        content = str(row.get("content", ""))[:200]
        st = row.get("status", "?")
        result = str(row.get("result", "") or "")
        if len(result) > 600:
            result = result[:600] + "... (truncated)"

        entry = (
            f"[{i}] Task #{row.get('id', '?')} | status={st} | "
            f"created={ts} | completed={completed or 'N/A'}\n"
            f"  Request: {content}\n"
        )
        if result:
            entry += f"  Result:\n{result}"
        results.append(entry)

    return f"Task reports ({len(rows)} entries):\n\n" + "\n\n---\n\n".join(results)


async def _exec_read_research(
    limit: int = 10,
    keyword: str | None = None,
    slug: str | None = None,
    status: str | None = None,
    max_chars: int | None = None,
    offset: int | None = None,
) -> str:
    from db import query as db_query, query_one as db_query_one

    status_value = (status or "").strip().lower()
    if status_value in {"public", "private", "staged"}:
        status_clause = "status = %s"
        status_params: tuple[str, ...] = (status_value,)
        empty_label = f"No {status_value} research documents found."
    elif status_value == "all":
        status_clause = "TRUE"
        status_params = ()
        empty_label = "No research documents found."
    else:
        status_clause = "status IN ('public', 'staged')"
        status_params = ()
        empty_label = "No public or staged research documents found."

    if slug:
        raw = slug.strip()
        filename = raw if raw.endswith(".md") else f"{raw}.md"
        bare_slug = raw[:-3] if raw.endswith(".md") else raw
        row = await asyncio.to_thread(
            db_query_one,
            f"""
            SELECT id, filename, slug, title, summary, markdown, status,
                   published_at, updated_at, title_en, summary_en,
                   markdown_en IS NOT NULL AND length(trim(markdown_en)) > 0 AS has_translation
              FROM research_documents
             WHERE (filename = %s OR slug = %s)
               AND {status_clause}
             LIMIT 1
            """,
            (filename, bare_slug, *status_params),
        )
        if not row:
            return f"No research document found for slug/filename: {slug}"
        markdown = row.get("markdown") or ""
        body, start, end, truncated = _slice_text(markdown, max_chars=max_chars, offset=offset)
        next_hint = (
            f"\nnext: read_self(content_type='research_document', slug='{row.get('slug') or bare_slug}', offset={end}, max_chars={max_chars})"
            if truncated and max_chars is not None
            else ""
        )
        return (
            f"=== RESEARCH DOCUMENT: {row['filename']} ===\n"
            f"id={row['id']} status={row['status']} slug={row.get('slug') or bare_slug}\n"
            f"title: {row.get('title') or ''}\n"
            f"published={_to_kst(row.get('published_at'))} updated={_to_kst(row.get('updated_at'))}\n"
            f"has_translation={row.get('has_translation')}\n"
            f"body_chars={len(markdown)} returned_chars={start}:{end} truncated={truncated}{next_hint}\n"
            f"summary: {(row.get('summary') or '')[:800]}\n\n"
            f"-- markdown {start}:{end}/{len(markdown)} --\n{body}"
        )

    clauses = [status_clause]
    params: list = list(status_params)
    if keyword:
        clauses.append("(title ILIKE %s OR summary ILIKE %s OR markdown ILIKE %s OR filename ILIKE %s)")
        q = f"%{keyword}%"
        params.extend([q, q, q, q])
    params.append(min(max(int(limit or 10), 1), 50))
    rows = await asyncio.to_thread(
        db_query,
        f"""
        SELECT id, filename, slug, title, summary, status, published_at, updated_at,
               markdown_en IS NOT NULL AND length(trim(markdown_en)) > 0 AS has_translation
          FROM research_documents
         WHERE {' AND '.join(clauses)}
         ORDER BY updated_at DESC, id DESC
         LIMIT %s
        """,
        tuple(params),
    )
    if not rows:
        return empty_label
    lines = ["=== RESEARCH DOCUMENTS ===", "Use read_self(content_type='research_document', slug='<slug-or-filename>') for full detail."]
    for r in rows:
        slug_value = r.get("slug") or str(r["filename"]).removesuffix(".md")
        summary = (r.get("summary") or "").replace("\n", " ")[:220]
        lines.append(
            f"- {r['filename']} | slug={slug_value} | status={r.get('status')} | updated={_to_kst(r.get('updated_at'))} | en={r.get('has_translation')}\n"
            f"  title: {r.get('title') or ''}\n"
            f"  summary: {summary}"
        )
    return "\n".join(lines)


async def _exec_read_private_research_documents(
    limit: int = 10,
    keyword: str | None = None,
    slug: str | None = None,
    max_chars: int | None = None,
    offset: int | None = None,
) -> str:
    from runtime_tools import private_reports

    if slug:
        try:
            row = await asyncio.to_thread(private_reports.get_private_report_sync, slug=slug)
        except Exception as e:
            return f"Error reading private research document: {type(e).__name__}: {e}"
        if not row:
            return f"No private research document found for slug: {slug}"
        markdown = row.get("markdown") or ""
        body, start, end, truncated = _slice_text(markdown, max_chars=max_chars, offset=offset)
        next_hint = (
            f"\nnext: read_self(content_type='private_research_document', slug='{row['slug']}', offset={end}, max_chars={max_chars})"
            if truncated and max_chars is not None
            else ""
        )
        return (
            f"=== PRIVATE RESEARCH DOCUMENT: {row['slug']} ===\n"
            f"id={row['id']} title: {row.get('title') or ''}\n"
            f"created={_to_kst(row.get('created_at'))} updated={_to_kst(row.get('updated_at'))}\n"
            f"published_research_id={row.get('published_research_id') or ''}\n"
            f"body_chars={len(markdown)} returned_chars={start}:{end} truncated={truncated}{next_hint}\n"
            f"summary: {(row.get('summary') or '')[:800]}\n\n"
            f"-- markdown {start}:{end}/{len(markdown)} --\n{body}"
        )

    try:
        rows = await asyncio.to_thread(
            private_reports.list_private_reports_sync,
            limit=limit or 10,
            keyword=keyword,
        )
    except Exception as e:
        return f"Error listing private research documents: {type(e).__name__}: {e}"
    if not rows:
        return "No private research documents found."
    lines = [
        "=== PRIVATE RESEARCH DOCUMENTS ===",
        "Use read_self(content_type='private_research_document', slug='<slug>') for detail.",
    ]
    for r in rows:
        summary = (r.get("summary") or "").replace("\n", " ")[:220]
        lines.append(
            f"- id={r['id']} slug={r['slug']} updated={_to_kst(r.get('updated_at'))}\n"
            f"  title: {r.get('title') or ''}\n"
            f"  summary: {summary}"
        )
    return "\n".join(lines)


async def _exec_read_curation(
    limit: int = 10, keyword: str | None = None, slug: str | None = None,
) -> str:
    from db import query as db_query, query_one as db_query_one

    if slug:
        slug = slug.strip().lower()
        row = await asyncio.to_thread(
            db_query_one,
            """
            SELECT id, slug, title, source_url, source_title, source_author,
                   source_publication, source_published_at,
                   selection_rationale, context, tags, published_at, updated_at
              FROM hub_curations
             WHERE slug = %s
             LIMIT 1
            """,
            (slug,),
        )
        if not row:
            return f"No hub curation found for slug: {slug}"
        return (
            f"=== HUB CURATION: {row['slug']} ===\n"
            f"id={row['id']} title: {row.get('title') or ''}\n"
            f"source: {row.get('source_title') or row.get('source_url')} "
            f"({row.get('source_publication') or '?'}, {row.get('source_author') or '?'})\n"
            f"url: {row.get('source_url') or ''}\n"
            f"published={_to_kst(row.get('published_at'))} updated={_to_kst(row.get('updated_at'))}\n"
            f"tags: {row.get('tags') or []}\n\n"
            f"-- selection rationale --\n{row.get('selection_rationale') or ''}\n\n"
            f"-- context --\n{row.get('context') or ''}"
        )

    clauses = []
    params: list = []
    if keyword:
        clauses.append(
            "(title ILIKE %s OR source_title ILIKE %s OR source_publication ILIKE %s "
            "OR selection_rationale ILIKE %s OR context ILIKE %s)"
        )
        q = f"%{keyword}%"
        params.extend([q, q, q, q, q])
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(min(max(int(limit or 10), 1), 50))
    rows = await asyncio.to_thread(
        db_query,
        f"""
        SELECT id, slug, title, source_title, source_publication, source_author,
               context, tags, published_at
          FROM hub_curations
          {where}
         ORDER BY published_at DESC, id DESC
         LIMIT %s
        """,
        tuple(params),
    )
    if not rows:
        return "No hub curations found."
    lines = ["=== HUB CURATIONS ===", "Use read_self(content_type='hub_curation', slug='<slug>') for full detail."]
    for r in rows:
        context = (r.get("context") or "").replace("\n", " ")[:220]
        lines.append(
            f"- {r['slug']} | curated={_to_kst(r.get('published_at'))}\n"
            f"  title: {r.get('title') or ''}\n"
            f"  source: {r.get('source_title') or ''} ({r.get('source_publication') or '?'})\n"
            f"  context: {context}"
        )
    return "\n".join(lines)


async def _exec_read_static_pages(
    limit: int = 10,
    keyword: str | None = None,
    slug: str | None = None,
    max_chars: int | None = None,
    offset: int | None = None,
) -> str:
    import site_publishing

    if slug:
        data = await asyncio.to_thread(site_publishing.get_static_page, slug.strip().lower())
        if not data:
            return f"No static page found for slug: {slug}"
        html = data.get("html_body") or ""
        body, start, end, truncated = _slice_text(html, max_chars=max_chars, offset=offset)
        next_hint = (
            f"\nnext: read_self(content_type='static_page', slug='{data.get('slug') or slug}', offset={end}, max_chars={max_chars})"
            if truncated and max_chars is not None
            else ""
        )
        return (
            f"=== STATIC PAGE: {data.get('slug') or slug} ===\n"
            f"title: {data.get('title') or ''}\n"
            f"summary: {data.get('summary') or ''}\n"
            f"updated_at: {data.get('updated_at') or '?'}\n\n"
            f"-- html_body chars={len(html)} returned_chars={start}:{end} truncated={truncated}{next_hint} --\n"
            f"{body}"
        )

    rows = await asyncio.to_thread(site_publishing.list_static_pages, "ko")
    if keyword:
        k = keyword.lower()
        rows = [
            r for r in rows
            if k in str(r.get("slug") or "").lower()
            or k in str(r.get("title") or "").lower()
            or k in str(r.get("summary") or "").lower()
        ]
    rows = rows[:min(max(int(limit or 10), 1), 50)]
    if not rows:
        return "No static pages found."
    lines = [
        "=== STATIC PAGES ===",
        "Use read_self(content_type='static_page', slug='<slug>') for full detail only for listed static page slugs.",
    ]
    for r in rows:
        summary = (r.get("summary") or "").replace("\n", " ")[:220]
        lines.append(
            f"- {r.get('slug')} | updated={r.get('updated_at') or '?'} | en={r.get('has_translation')}\n"
            f"  title: {r.get('title') or ''}\n"
            f"  summary: {summary}"
        )
    return "\n".join(lines)


async def _exec_read_kg_status() -> str:
    from kg_runtime.search import fetch_kg_stats

    stats = await asyncio.to_thread(fetch_kg_stats)

    if "error" in stats:
        return f"Knowledge Graph status check failed: {stats['error']}"

    parts = [
        f"Episodes: {stats.get('episode_count', '?')}",
        f"Edges (relationships): {stats.get('edge_count', '?')}",
    ]

    entity_types = stats.get("entity_types", {})
    if entity_types:
        parts.append("Entity types:")
        for labels, cnt in entity_types.items():
            parts.append(f"  {labels}: {cnt}")

    recent = stats.get("recent_episodes", [])
    if recent:
        parts.append(f"\nRecent episodes ({len(recent)} most recent):")
        for ep in recent:
            source = ep.get("source", "")
            source_tag = f" [{source}]" if source else ""
            parts.append(
                f"\n  📌 [{_to_kst(ep.get('created_at'))}] "
                f"{ep.get('name', '?')}"
                f"{source_tag} "
                f"(group: {ep.get('group_id', '?')})"
            )
            # Show extracted entities
            entities = ep.get("entities", [])
            if entities:
                ent_strs = [
                    f"{e['name']} ({','.join(l for l in e.get('labels', []) if l != 'Entity')})"
                    if any(l != 'Entity' for l in e.get('labels', []))
                    else e['name']
                    for e in entities[:8]
                ]
                more = f" +{len(entities) - 8} more" if len(entities) > 8 else ""
                parts.append(f"    Entities: {', '.join(ent_strs)}{more}")
            # Show extracted facts
            facts = ep.get("facts", [])
            if facts:
                for f in facts[:5]:
                    parts.append(f"    → {f.get('fact', '?')}")
                if len(facts) > 5:
                    parts.append(f"    ... +{len(facts) - 5} more facts")

    return "=== KNOWLEDGE GRAPH STATUS ===\n\n" + "\n".join(parts)


async def _exec_read_system_status() -> str:
    from shared import (
        fetch_diaries, fetch_chat_logs, fetch_task_reports,
        fetch_kg_stats, KST, MODULE_ARCHITECTURE,
    )

    status_parts = []

    # 1. Diary status
    diaries = await asyncio.to_thread(fetch_diaries, 1)
    if diaries:
        last = diaries[0]
        status_parts.append(f"Last diary: {_to_kst(last.get('created_at'))} — {last.get('title', 'N/A')}")
    else:
        status_parts.append("No diaries written yet.")

    # 2. Chat activity (use small limits — we only need counts)
    logs_24h = await asyncio.to_thread(fetch_chat_logs, 100, 24)
    logs_6h = await asyncio.to_thread(fetch_chat_logs, 100, 6)
    status_parts.append(f"Chats: {len(logs_6h)} (6h), {len(logs_24h)} (24h)")

    # 3. Task queue
    tasks = await asyncio.to_thread(fetch_task_reports, 20)
    if tasks:
        by_status = {}
        for t in tasks:
            s = t.get("status", "?")
            by_status[s] = by_status.get(s, 0) + 1
        summary = ", ".join(f"{k}: {v}" for k, v in by_status.items())
        status_parts.append(f"Tasks: {summary}")
    else:
        status_parts.append("Tasks: none")

    # 4. KG health
    kg = await asyncio.to_thread(fetch_kg_stats)
    if "error" not in kg:
        status_parts.append(f"KG: {kg.get('episode_count', '?')} episodes, {kg.get('edge_count', '?')} edges")
    else:
        status_parts.append(f"KG: {kg['error']}")

    # 5. Time + architecture
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    status_parts.append(f"Time: {now}")
    status_parts.append(MODULE_ARCHITECTURE)

    return "=== SYSTEM STATUS ===\n" + "\n".join(status_parts)


async def _exec_read_server_logs(
    service: str = "telegram", minutes_back: int = 10, limit: int = 50, grep: str | list[str] | tuple[str, ...] | None = "",
) -> str:
    """Read journald logs for a systemd service."""
    from ops.logs import fetch_server_logs, _normalize_grep_terms, grep_matches_text

    minutes_back = max(1, min(60, minutes_back))
    limit = max(1, min(200, limit))
    hours_back = max(1, (minutes_back + 59) // 60)
    grep_terms = _normalize_grep_terms(grep)

    rows = await asyncio.to_thread(
        fetch_server_logs,
        service,
        hours_back,
        None,
        limit,
    )
    if not rows:
        return f"=== SERVER LOGS ({service}, last {minutes_back}min) ===\n(no output)"

    if rows and isinstance(rows[0], dict) and rows[0].get("error"):
        return f"Log fetch failed: {rows[0]['error']}"

    if grep_terms:
        rows = [
            row for row in rows
            if grep_matches_text((row or {}).get("raw") if isinstance(row, dict) else row, grep_terms)
        ]
        if not rows:
            return f"=== SERVER LOGS ({service}, last {minutes_back}min, grep={grep_terms}) ===\n(no output)"

    formatted = []
    for row in rows:
        raw = str((row or {}).get("raw") or "").strip()
        if raw:
            formatted.append(raw)
        else:
            ts = str((row or {}).get("timestamp") or "").strip()
            msg = str((row or {}).get("message") or "").strip()
            formatted.append(f"{ts} {msg}".strip())
    output = "\n".join(formatted) if formatted else "(no output)"
    grep_desc = f", grep={grep_terms}" if grep_terms else ""
    return f"=== SERVER LOGS ({service}, last {minutes_back}min{grep_desc}) ===\n{output}"


async def _exec_read_recent_updates(max_entries: int = 3) -> str:
    return (
        "=== RECENT SYSTEM UPDATES ===\n\n"
        "Disabled: dev_docs/project_state.md is a stale human-maintained snapshot "
        "and must not be injected into agent context."
    )


def _normalize_for_overlap(text: str) -> set[str]:
    """Normalize text to a token set for self-poisoning loop detection."""
    import re as _re
    if not text:
        return set()
    cleaned = _re.sub(r"[\W_]+", " ", text.lower())
    tokens = [t for t in cleaned.split() if len(t) >= 4]
    return set(tokens)


def _self_poisoning_overlap(new_content: str, kg_reads: list[str]) -> float:
    """Return max Jaccard overlap between new_content and any recent KG read."""
    new_tokens = _normalize_for_overlap(new_content)
    if not new_tokens or not kg_reads:
        return 0.0
    best = 0.0
    for read in kg_reads:
        read_tokens = _normalize_for_overlap(read)
        if not read_tokens:
            continue
        inter = len(new_tokens & read_tokens)
        union = len(new_tokens | read_tokens)
        if union:
            j = inter / union
            if j > best:
                best = j
    return best


async def _exec_write_kg(
    content: str,
    name: str = "",
    source_type: str = "internal_report",
    group_id: str = "agent_knowledge",
    supersedes: str = "",
) -> str:
    from provenance.runtime import get_provenance_buffer
    from kg_runtime.writes import add_kg_episode_async

    if not content or not content.strip():
        return "Failed to store knowledge: content is empty"

    # ── Self-poisoning loop break ────────────────────────────────────
    # Reject writes that are largely re-quotes of text we just retrieved
    # from the KG itself in the same agent run. This prevents the agent
    # from bootstrapping false confidence by re-ingesting its own output.
    buf = get_provenance_buffer()
    if buf is not None and buf.kg_reads:
        overlap = _self_poisoning_overlap(content, buf.kg_reads)
        if overlap >= 0.55:
            logger.warning("[KG AUDIT] BLOCKED self-poisoning write | overlap=%.2f | agent=%s", overlap, buf.agent)
            return (
                f"Refused: this content overlaps {overlap:.0%} with text you "
                "just retrieved from the knowledge graph. Re-ingesting retrieved "
                "facts would create a self-confirming loop. If you want to add a "
                "genuine new fact (corroboration from a fresh source, correction, "
                "extension), rewrite it in your own words with the new source cited."
            )

    # ── Auto-provenance footer ───────────────────────────────────────
    trust_tier = buf.infer_trust_tier() if buf is not None else "unverified"
    agent_label = buf.agent if buf is not None else "agent"
    mission_label = (
        f" | mission={buf.mission_id}" if buf is not None and buf.mission_id else ""
    )
    sources = buf.recent_sources(8) if buf is not None else []
    trust_source = buf.trust_source_note() if buf is not None else "no_provenance_buffer"
    ts = datetime.now(_KST).strftime("%Y-%m-%d %H:%M KST")
    footer_lines = [
        "— provenance —",
        f"agent: {agent_label}{mission_label}",
        f"ingested_at: {ts}",
        f"trust_tier: {trust_tier}",
        f"trust_source: {trust_source}",
    ]
    if sources:
        footer_lines.append("sources:")
        footer_lines.extend(f"  - {s}" for s in sources)
    if supersedes:
        footer_lines.append(f"supersedes: {supersedes}")
    provenance_footer = "\n".join(footer_lines)

    result = await add_kg_episode_async(
        content,
        name,
        source_type,
        group_id,
        trust_tier=trust_tier,
        provenance_footer=provenance_footer,
    )
    if result["status"] == "ok":
        logger.info(
            "[KG AUDIT] wrote episode | name=%s | group=%s | tier=%s | agent=%s | sources=%d | supersedes=%s | time=%s | content_len=%d",
            name or "(auto)", group_id, trust_tier, agent_label, len(sources),
            supersedes or "-", ts, len(content),
        )
        msg = result["message"]
        if trust_tier == "unverified":
            msg += " (trust_tier=unverified — no external source recorded this run)"
        elif trust_tier == "anchor":
            msg += " (trust_tier=anchor — trusted operator chat/task context; no public URL source required)"
        return f"Knowledge stored successfully: {msg}"
    else:
        return f"Failed to store knowledge: {result['message']}"


async def _exec_write_kg_structured(
    facts: list,
    group_id: str = "agent_knowledge",
) -> str:
    """Write structured (subject, predicate, object) facts to the KG."""
    from provenance.runtime import get_provenance_buffer
    from kg_runtime.writes import add_kg_structured_async

    if not facts or not isinstance(facts, list):
        return "Failed to store structured facts: 'facts' must be a non-empty list"

    # ── Self-poisoning loop break ────────────────────────────────────
    # For structured writes, the equivalent check is whether the new fact
    # texts overlap heavily with text we just retrieved from the KG.
    buf = get_provenance_buffer()
    if buf is not None and buf.kg_reads:
        joined = "\n".join(str(f.get("fact", "")) for f in facts)
        overlap = _self_poisoning_overlap(joined, buf.kg_reads)
        if overlap >= 0.55:
            logger.warning(
                "[KG AUDIT] BLOCKED self-poisoning structured write | overlap=%.2f | agent=%s",
                overlap, buf.agent,
            )
            return (
                f"Refused: these facts overlap {overlap:.0%} with text you "
                "just retrieved from the knowledge graph. Re-asserting retrieved "
                "facts would create a self-confirming loop. Cite a fresh source "
                "and rewrite in your own words if you have a genuine new fact."
            )

    # ── Auto-provenance footer ───────────────────────────────────────
    trust_tier = buf.infer_trust_tier() if buf is not None else "unverified"
    agent_label = buf.agent if buf is not None else "agent"
    mission_id = buf.mission_id if buf is not None else None
    sources = buf.recent_sources(8) if buf is not None else []
    trust_source = buf.trust_source_note() if buf is not None else "no_provenance_buffer"
    ts = datetime.now(_KST).strftime("%Y-%m-%d %H:%M KST")
    footer_lines = [
        "— provenance —",
        f"agent: {agent_label}" + (f" | mission={mission_id}" if mission_id else ""),
        f"ingested_at: {ts}",
        f"trust_tier: {trust_tier}",
        f"trust_source: {trust_source}",
    ]
    if sources:
        footer_lines.append("sources:")
        footer_lines.extend(f"  - {s}" for s in sources)
    provenance_footer = "\n".join(footer_lines)

    result = await add_kg_structured_async(
        facts,
        group_id=group_id,
        agent=agent_label,
        mission_id=mission_id,
        trust_tier=trust_tier,
        provenance_footer=provenance_footer,
    )

    if result["status"] in {"ok", "partial_success"}:
        logger.info(
            "[KG AUDIT] structured write | status=%s facts=%d rejected=%d new=%d reused=%d | "
            "group=%s | tier=%s | agent=%s | sources=%d | time=%s",
            result.get("status"), result.get("facts_written", 0),
            result.get("facts_rejected", 0), result.get("new_entities", 0),
            result.get("reused_entities", 0),
            group_id, trust_tier, agent_label, len(sources), ts,
        )
        msg = result["message"]
        if trust_tier == "unverified":
            msg += " (trust_tier=unverified — no external source recorded this run)"
        elif trust_tier == "anchor":
            msg += " (trust_tier=anchor — trusted operator chat/task context; no public URL source required)"
        if result["status"] == "partial_success":
            partial_payload = {
                "stored_fact_indices": result.get("written_fact_indices", []),
                "rejected_facts": result.get("rejected_facts", []),
                "retry_facts": [
                    item.get("fact", item)
                    for item in result.get("rejected_facts", [])
                    if isinstance(item, dict)
                ],
            }
            msg += (
                "\nPartial success details JSON:\n"
                + json.dumps(partial_payload, ensure_ascii=False, indent=2)
            )
        return f"Structured facts stored: {msg}"
    else:
        msg = result["message"]
        if result.get("rejected_facts"):
            rejected_json = json.dumps(
                result.get("rejected_facts", []),
                ensure_ascii=False,
                indent=2,
            )
            msg += f"\nRejected facts JSON for retry:\n{rejected_json}"
        return f"Failed to store structured facts: {msg}"


def _compact_tool(tool: dict, *, include_descriptions: bool, include_schemas: bool) -> dict:
    item = {"name": tool.get("name", "")}
    if include_descriptions:
        desc = str(tool.get("description") or "")
        item["description"] = desc[:500] + ("..." if len(desc) > 500 else "")
    if include_schemas:
        item["input_schema"] = tool.get("input_schema", {})
    return item


def _dedupe_tools(tools: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for tool in tools:
        name = tool.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(tool)
    return result


def get_agent_tool_manifest(
    agent: str = "all",
    *,
    include_descriptions: bool = True,
    include_schemas: bool = False,
    orchestrator_tools: list[dict] | None = None,
) -> dict:
    """Return runtime tool visibility for orchestrator and specialist agents."""
    from agents import get_agent, list_agents
    from runtime_tools.registry import TOOLS as BASE_TOOLS, TOOL_HANDLERS as BASE_HANDLERS

    requested = (agent or "all").strip().lower()
    base_tools = _dedupe_tools(BASE_TOOLS)

    def format_tools(tools: list[dict]) -> list[dict]:
        return [
            _compact_tool(t, include_descriptions=include_descriptions, include_schemas=include_schemas)
            for t in _dedupe_tools(tools)
        ]

    def orchestrator_manifest() -> dict:
        if orchestrator_tools is None:
            from runtime_tools.allowlists import select_orchestrator_tools

            tools = _dedupe_tools(select_orchestrator_tools(base_tools))
            return {
                "runtime": "orchestrator",
                "available": True,
                "source": "static_allowlist",
                "reason": "reported from runtime_tools.allowlists outside an active orchestrator context",
                "tool_count": len(tools),
                "tools": format_tools(tools),
            }
        tools = _dedupe_tools(orchestrator_tools)
        return {
            "runtime": "orchestrator",
            "available": True,
            "source": "active_context",
            "tool_count": len(tools),
            "tools": format_tools(tools),
        }

    def agent_manifest(name: str) -> dict:
        spec = get_agent(name)
        tools, handlers = spec.filter_tools(base_tools, BASE_HANDLERS)
        note = None
        if name == "programmer" and spec.provider == "codex":
            note = (
                "programmer provider is codex; delegated programmer tasks bypass the "
                "LeninBot tool loop and run Codex CLI. This allow-list applies to "
                "in-process/non-Codex execution and routing introspection."
            )
        return {
            "runtime": "agent",
            "agent": name,
            "delegatable": name in _DELEGATABLE_AGENTS,
            "provider": spec.provider,
            "model": spec.model,
            "budget_usd": spec.budget_usd,
            "max_rounds": spec.max_rounds,
            "routing_card": _AGENT_ROUTING_CARDS.get(name),
            "tool_count": len(tools),
            "handler_count": len(handlers),
            "terminal_tools": list(spec.terminal_tools),
            "finalization_tools": list(spec.finalization_tools),
            "note": note,
            "tools": format_tools(tools),
        }

    def web_chat_manifest() -> dict:
        try:
            from web_chat import _web_tools, _web_handlers
            tools = _dedupe_tools(list(_web_tools))
            return {
                "runtime": "web_chat",
                "available": True,
                "provider": None,
                "tool_count": len(tools),
                "handler_count": len(_web_handlers),
                "note": (
                    "web chat is not an AgentSpec. It uses web_chat.py _WEB_ALLOWED_TOOLS "
                    "plus a web-safe read_self override."
                ),
                "tools": format_tools(tools),
            }
        except Exception as e:
            return {
                "runtime": "web_chat",
                "available": False,
                "reason": f"{type(e).__name__}: {e}",
                "tool_count": 0,
                "tools": [],
            }

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if requested in {"", "all", "*"}:
        return {
            "generated_at": generated_at,
            "delegatable_agents": list(_DELEGATABLE_AGENTS),
            "content_store_guide": _CONTENT_STORE_GUIDE,
            "orchestrator": orchestrator_manifest(),
            "web_chat": web_chat_manifest(),
            "agents": {spec.name: agent_manifest(spec.name) for spec in list_agents()},
        }
    if requested == "orchestrator":
        return {"generated_at": generated_at, "orchestrator": orchestrator_manifest()}
    if requested == "web_chat":
        return {"generated_at": generated_at, "web_chat": web_chat_manifest()}
    return {"generated_at": generated_at, "agents": {requested: agent_manifest(requested)}}


async def _exec_list_agent_tools(
    agent: str = "all",
    include_descriptions: bool = True,
    include_schemas: bool = False,
) -> str:
    try:
        manifest = await asyncio.to_thread(
            get_agent_tool_manifest,
            agent or "all",
            include_descriptions=bool(include_descriptions),
            include_schemas=bool(include_schemas),
            orchestrator_tools=None,
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": f"{type(e).__name__}: {e}"},
            ensure_ascii=False,
            indent=2,
        )
    return json.dumps(manifest, ensure_ascii=False, indent=2)


async def _exec_route_task(
    task: str,
    candidates: list[str] | None = None,
    include_store_guide: bool = True,
) -> str:
    try:
        clean_candidates = None
        if candidates:
            clean_candidates = [str(c).strip().lower() for c in candidates if str(c).strip()]
            invalid = sorted(set(clean_candidates) - set(_DELEGATABLE_AGENTS))
            if invalid:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"non-delegatable candidate(s): {invalid}",
                        "delegatable_agents": list(_DELEGATABLE_AGENTS),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
        recommendation = await _classify_route_with_llm(task or "", clean_candidates)
        fallback_used = False
        if recommendation is None:
            recommendation = _recommend_agent_for_task(task or "", candidates=clean_candidates)
            fallback_used = True
        warning = _routing_warning(recommendation.get("recommended_agent", ""), task or "")
        if warning and "warning" not in recommendation:
            recommendation["warning"] = warning
        payload = {
            "status": "ok",
            "delegatable_agents": list(_DELEGATABLE_AGENTS),
            "recommendation": recommendation,
            "classifier": {
                "attempted": True,
                "used": not fallback_used,
                "fallback": "heuristic" if fallback_used else None,
                "classes": [
                    "public_content_edit",
                    "code_config_work",
                    "research",
                    "diary",
                    "browser_automation",
                    "external_platform_scout",
                    "email_a2a",
                ],
            },
            "content_store_guide": _CONTENT_STORE_GUIDE if include_store_guide else None,
            "next_step": (
                "Call delegate with agent=recommendation.recommended_agent only after the "
                "target content type and identifier are clear. Include success_criteria, "
                "target_identifiers, and forbidden_assumptions."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": f"{type(e).__name__}: {e}"},
            ensure_ascii=False,
            indent=2,
        )


def build_list_agent_tools_handler(orchestrator_tools: list[dict]):
    """Build list_agent_tools with the actual current orchestrator tool set."""

    async def _exec_list_agent_tools_with_orchestrator(
        agent: str = "all",
        include_descriptions: bool = True,
        include_schemas: bool = False,
    ) -> str:
        try:
            manifest = await asyncio.to_thread(
                get_agent_tool_manifest,
                agent or "all",
                include_descriptions=bool(include_descriptions),
                include_schemas=bool(include_schemas),
                orchestrator_tools=list(orchestrator_tools),
            )
        except Exception as e:
            return json.dumps(
                {"status": "error", "error": f"{type(e).__name__}: {e}"},
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps(manifest, ensure_ascii=False, indent=2)

    return _exec_list_agent_tools_with_orchestrator


async def _exec_delegate(
    agent: str,
    task: str,
    context: str = "",
    success_criteria: str = "",
    required_capabilities: list[str] | None = None,
    target_identifiers: list[str] | None = None,
    forbidden_assumptions: list[str] | None = None,
    priority: str = "normal",
    parent_task_id: int | None = None,
    verification: dict | None = None,
) -> str:
    from task_store import create_task_in_db

    agent = (agent or "").strip().lower()
    if agent not in _DELEGATABLE_AGENTS:
        return (
            f"Cannot delegate to {agent!r}. Delegatable agents are: "
            f"{', '.join(_DELEGATABLE_AGENTS)}. Stasova is reserved for internal "
            "publication-review flows and is not in the general delegation whitelist."
        )

    warning = _routing_warning(agent, task, context)
    if warning:
        recommended = _recommend_agent_for_task(f"{task}\n{context}")
        return (
            f"{warning}\n"
            f"Recommended agent: {recommended.get('recommended_agent')} "
            f"({recommended.get('reason')}). Call route_task if you need the full store guide."
        )

    # Validate agent name
    try:
        from agents import get_agent
        spec = get_agent(agent)
    except ValueError as e:
        return str(e)

    # Inherit mission from parent task if chaining, otherwise use/create mission
    task_mission_id = None
    if not parent_task_id:
        try:
            from db import query as _db_q
            user_id_for_mission = _resolve_recent_operator_user_id()
            active = _db_q(
                "SELECT id FROM telegram_missions WHERE user_id = %s AND status = 'active' "
                "ORDER BY created_at DESC LIMIT 1",
                (user_id_for_mission,),
            )
            if active:
                task_mission_id = active[0]["id"]
            else:
                # Auto-create mission from delegation context
                from telegram.mission import create_mission
                mission_title = task[:80].replace("\n", " ").strip()
                if user_id_for_mission:
                    new_mission = create_mission(user_id_for_mission, mission_title)
                    task_mission_id = new_mission["id"]
                    logger.info("Auto-created mission #%d from delegate: %s", task_mission_id, mission_title)
        except Exception as e:
            logger.debug("Mission auto-create in delegate failed: %s", e)

    # ── Assemble full task content with context ──────────────────
    # 1. Orchestrator-provided context (conversation summary, reasoning)
    # 2. Recent chat history from DB (automatic, as fallback/supplement)
    # 3. The actual task instructions
    content_parts = []

    if context:
        content_parts.append(f"<delegation-context>\n{context}\n</delegation-context>")
    contract = _format_delegation_contract(
        success_criteria=success_criteria,
        required_capabilities=required_capabilities,
        target_identifiers=target_identifiers,
        forbidden_assumptions=forbidden_assumptions,
    )
    if contract:
        content_parts.append(contract)

    # Fetch recent chat history to give agent conversational backdrop
    try:
        from memory_store.queries import fetch_chat_logs
        recent_chats = await asyncio.to_thread(
            fetch_chat_logs, 6, None, None, source="telegram"
        )
        if recent_chats:
            chat_lines = []
            for msg in reversed(recent_chats):  # chronological order
                role = "user" if msg.get("role") == "user" else "agent"
                text = str(msg.get("content") or "")[:500]
                chat_lines.append(f"[{role}] {text}")
            content_parts.append(
                "<recent-conversation>\n"
                + "\n".join(chat_lines)
                + "\n</recent-conversation>"
            )
    except Exception:
        pass  # non-critical: mission context will still be injected by process_task

    content_parts.append(f"<task agent=\"{agent}\">\n{task}\n</task>")
    full_content = "\n\n".join(content_parts)

    # Record delegation event to mission timeline
    if task_mission_id:
        try:
            from telegram.mission import add_mission_event
            delegation_note = f"Delegated to [{agent}]: {task[:500]}"
            if context:
                delegation_note += f"\nContext: {context[:500]}"
            await asyncio.to_thread(
                add_mission_event, task_mission_id, "orchestrator", "decision", delegation_note
            )
        except Exception:
            pass

    result = await asyncio.to_thread(
        create_task_in_db, full_content, 0, priority,
        parent_task_id=parent_task_id, mission_id=task_mission_id,
        agent_type=agent,
        metadata={"verification": verification} if isinstance(verification, dict) else None,
    )
    if result["status"] == "ok":
        depth_info = f", depth={result.get('depth', 0)}" if parent_task_id else ""
        return (
            f"Task #{result['task_id']} delegated to [{agent}] agent "
            f"(priority: {priority}{depth_info}, budget: ${spec.budget_usd:.2f}). "
            f"Processing in background."
        )
    else:
        return f"Failed to delegate task: {result['error']}"


async def _exec_multi_delegate(
    tasks: list[dict],
    synthesis_instructions: str = "",
    priority: str = "normal",
) -> str:
    """Delegate multiple tasks in parallel with automatic synthesis."""
    from task_store import create_task_in_db
    from db import execute as db_execute

    if len(tasks) < 2:
        return "multi_delegate requires at least 2 tasks. Use delegate for single tasks."

    # Validate all agents
    try:
        from agents import get_agent
        for t in tasks:
            agent_name = str(t.get("agent") or "").strip().lower()
            if agent_name not in _DELEGATABLE_AGENTS:
                return (
                    f"Cannot delegate to {agent_name!r}. Delegatable agents are: "
                    f"{', '.join(_DELEGATABLE_AGENTS)}. Stasova is not a general delegation target."
                )
            warning = _routing_warning(agent_name, str(t.get("task") or ""), str(t.get("context") or ""))
            if warning:
                recommended = _recommend_agent_for_task(f"{t.get('task')}\n{t.get('context', '')}")
                return (
                    f"Subtask for {agent_name!r} appears misrouted: {warning}\n"
                    f"Recommended agent: {recommended.get('recommended_agent')} "
                    f"({recommended.get('reason')})."
                )
            t["agent"] = agent_name
            get_agent(agent_name)
    except ValueError as e:
        return str(e)

    # Resolve mission (same logic as delegate)
    task_mission_id = None
    try:
        from db import query as _db_q
        user_id_for_mission = _resolve_recent_operator_user_id()
        active = _db_q(
            "SELECT id FROM telegram_missions WHERE user_id = %s AND status = 'active' "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id_for_mission,),
        )
        if active:
            task_mission_id = active[0]["id"]
        else:
            from telegram.mission import create_mission
            mission_title = tasks[0]["task"][:80].replace("\n", " ").strip()
            if user_id_for_mission:
                new_mission = create_mission(user_id_for_mission, mission_title)
                task_mission_id = new_mission["id"]
    except Exception as e:
        logger.debug("Mission resolution in multi_delegate failed: %s", e)

    # Fetch recent chat for context (shared across all subtasks)
    chat_block = ""
    try:
        from memory_store.queries import fetch_chat_logs
        recent_chats = await asyncio.to_thread(
            fetch_chat_logs, 6, None, None, source="telegram"
        )
        if recent_chats:
            chat_lines = []
            for msg in reversed(recent_chats):
                role = "user" if msg.get("role") == "user" else "agent"
                text = str(msg.get("content") or "")[:500]
                chat_lines.append(f"[{role}] {text}")
            chat_block = (
                "<recent-conversation>\n"
                + "\n".join(chat_lines)
                + "\n</recent-conversation>"
            )
    except Exception:
        pass

    # Create subtasks
    created_items = []
    created_info = []
    for t in tasks:
        agent = t["agent"]
        task_content = t["task"]
        context = t.get("context", "")

        content_parts = []
        if context:
            content_parts.append(f"<delegation-context>\n{context}\n</delegation-context>")
        contract = _format_delegation_contract(
            success_criteria=str(t.get("success_criteria") or ""),
            required_capabilities=t.get("required_capabilities"),
            target_identifiers=t.get("target_identifiers"),
            forbidden_assumptions=t.get("forbidden_assumptions"),
        )
        if contract:
            content_parts.append(contract)
        if chat_block:
            content_parts.append(chat_block)
        content_parts.append(f"<task agent=\"{agent}\">\n{task_content}\n</task>")
        full_content = "\n\n".join(content_parts)

        subtask_verification = t.get("verification")
        result = await asyncio.to_thread(
            create_task_in_db, full_content, 0, priority,
            mission_id=task_mission_id, agent_type=agent,
            plan_role="subtask",
            metadata=(
                {"verification": subtask_verification}
                if isinstance(subtask_verification, dict)
                else None
            ),
        )
        if result["status"] == "ok":
            tid = result["task_id"]
            created_items.append({"id": tid, "agent": agent, "task": task_content})
            spec = get_agent(agent)
            created_info.append(f"  #{tid} [{agent}] ${spec.budget_usd:.2f}")
        else:
            created_info.append(f"  FAILED [{agent}]: {result.get('error')}")

    if not created_items:
        return "Failed to create any subtasks."

    # Set plan_id = first subtask ID for all subtasks
    created_ids = [item["id"] for item in created_items]
    plan_id = created_ids[0]
    if len(created_ids) > 1:
        id_list = ",".join(str(i) for i in created_ids)
        await asyncio.to_thread(
            db_execute,
            f"UPDATE telegram_tasks SET plan_id = %s WHERE id IN ({id_list})",
            (plan_id,),
        )
    else:
        await asyncio.to_thread(
            db_execute,
            "UPDATE telegram_tasks SET plan_id = %s WHERE id = %s",
            (plan_id, plan_id),
        )

    # Create synthesis task (blocked until subtasks complete)
    subtask_summary = "\n".join(
        f"- #{item['id']}: [{item['agent']}] {item['task'][:200]}"
        for item in created_items
    )
    synthesis_content = (
        f"<synthesis-task plan_id=\"{plan_id}\">\n"
        f"This task synthesizes results from subtasks that were executed in parallel.\n"
        f"Subtask results are auto-injected in the <subtask-results> block.\n\n"
        f"## Subtasks\n{subtask_summary}\n\n"
        f"## Synthesis Instructions\n{synthesis_instructions or 'Analyze all subtask results and provide a consolidated report of key findings to the user.'}\n"
        f"</synthesis-task>"
    )
    synthesis_result = await asyncio.to_thread(
        create_task_in_db, synthesis_content, 0, priority,
        mission_id=task_mission_id, agent_type="analyst",
        plan_id=plan_id, plan_role="synthesis", status="blocked",
    )
    synthesis_id = synthesis_result.get("task_id", "?")

    # Record to mission timeline
    if task_mission_id:
        try:
            from telegram.mission import add_mission_event
            await asyncio.to_thread(
                add_mission_event, task_mission_id, "orchestrator", "decision",
                f"Multi-delegate: {len(created_ids)} subtasks → synthesis #{synthesis_id}\n{subtask_summary}"
            )
        except Exception:
            pass

    return (
        f"Plan #{plan_id} created: {len(created_ids)} parallel subtasks + synthesis #{synthesis_id}\n"
        + "\n".join(created_info)
        + f"\n  #{synthesis_id} [analyst] synthesis (blocked until subtasks complete)"
    )


# ── Inline Agent Execution ─────────────────────────────────────

def build_run_agent_handler(chat_with_tools_fn):
    """Build a run_agent handler with the chat function injected via closure."""

    async def _exec_run_agent(
        agent: str, task: str, context: str = "", budget_usd: float = 0.30,
    ) -> str:
        if agent != "analyst":
            return f"run_agent currently supports 'analyst' only, got '{agent}'."

        budget_usd = min(0.50, max(0.01, budget_usd))

        try:
            from agents import get_agent
            from runtime_tools.registry import TOOLS as BASE_TOOLS, TOOL_HANDLERS as BASE_HANDLERS

            spec = get_agent(agent)
            agent_tools, agent_handlers = spec.filter_tools(BASE_TOOLS, BASE_HANDLERS)

            from bot_config import _config as _bot_config
            _agent_provider = spec.effective_provider(
                _bot_config.get("provider", "claude")
            )
            # Fully static spec prompt; runtime context (time/model/alerts) is
            # injected into the user message by the orchestrator's chat_with_tools.
            system_prompt = spec.render_prompt(provider=_agent_provider)

            content_parts = []
            if context:
                content_parts.append(f"<delegation-context>\n{context}\n</delegation-context>")
            content_parts.append(f"<task agent=\"{agent}\">\n{task}\n</task>")
            full_content = "\n\n".join(content_parts)

            result = await chat_with_tools_fn(
                [{"role": "user", "content": full_content}],
                system_prompt=system_prompt,
                budget_usd=budget_usd,
                max_rounds=10,
                extra_tools=agent_tools,
                extra_handlers=agent_handlers,
            )
            # Truncate to avoid blowing up orchestrator context
            if len(result) > 4000:
                result = result[:4000] + "\n\n[... truncated]"
            return result

        except Exception as e:
            logger.error("run_agent failed: %s", e)
            return f"run_agent error: {e}"

    return _exec_run_agent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. HANDLER MAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _exec_recall_experience(query: str, limit: int = 5) -> str:
    from memory_store.experiential import search_experiential_memory

    limit = max(1, min(10, limit))
    rows = await asyncio.to_thread(search_experiential_memory, query, limit)
    if not rows:
        return "No relevant experiential memories found."
    lines = []
    for r in rows:
        sim = f"{r.get('similarity', 0):.0%}"
        cat = r.get("category", "?")
        src = r.get("source_type", "?")
        ts = str(r.get("created_at", ""))[:10]
        lines.append(f"[{cat}|{src}|{ts}|sim={sim}] {r['content']}")
    return f"Found {len(rows)} experience(s):\n" + "\n\n".join(lines)


async def _exec_save_self_analysis(
    title: str,
    content: str,
    category: str = "insight",
    source_context: str = "",
) -> str:
    from corpus.public_index import save_self_produced_analysis

    title = (title or "").strip()
    content = (content or "").strip()
    category = (category or "insight").strip()
    source_context = (source_context or "").strip()
    if not title:
        return "Error: title is required."
    if len(content) < 80:
        return "Error: content is too short for self_produced_analysis. Save a self-contained analytical note."

    result = await asyncio.to_thread(
        save_self_produced_analysis,
        title=title,
        content=content,
        category=category,
        source_context=source_context,
    )
    if not result.get("ok"):
        return f"self-analysis save failed: {result.get('error', 'unknown error')}"
    return (
        "Saved self-produced analysis "
        f"({result['chunks']} chunk(s)) to vector layer self_produced_analysis. "
        "Retrieve with vector_search(layer=\"self_produced_analysis\", query=\"...\")."
    )


async def _exec_kg_query(query: str, write: bool = False) -> str:
    from kg_runtime.admin import kg_cypher
    result = await asyncio.to_thread(kg_cypher, query, write)
    if "error" in result:
        return f"KG query failed: {result['error']}"
    rows = result.get("rows", [])
    count = result.get("count", 0)
    if not rows:
        return f"Query returned 0 rows. (write={write})"
    import json
    formatted = json.dumps(rows[:50], ensure_ascii=False, indent=2, default=str)
    suffix = f"\n... (+{count-50} more rows)" if count > 50 else ""
    return f"KG query result ({count} rows):\n{formatted}{suffix}"


async def _exec_kg_delete_episode(episode_name: str) -> str:
    from kg_runtime.admin import kg_delete_episode
    result = await asyncio.to_thread(kg_delete_episode, episode_name)
    if "error" in result:
        return f"Delete failed: {result['error']}"
    if result.get("not_found"):
        return f"Episode not found: '{episode_name}'"
    return (
        f"✅ Episode deleted: '{episode_name}'\n"
        f"  Deleted episodes: {result.get('deleted_episode', 0)}\n"
        f"  Deleted orphaned entities: {result.get('deleted_entities', 0)}"
    )


async def _exec_kg_merge_entities(source_name: str, target_name: str) -> str:
    from kg_runtime.admin import kg_merge_entities
    result = await asyncio.to_thread(kg_merge_entities, source_name, target_name)
    if "error" in result:
        return f"Merge failed: {result['error']}"
    return (
        f"✅ Entity merged: '{result['deleted_source']}' → '{result['merged_into']}'\n"
        f"  Outgoing relations transferred: {result.get('transferred_outgoing', 0)}\n"
        f"  Incoming relations transferred: {result.get('transferred_incoming', 0)}\n"
        f"  MENTIONS transferred: {result.get('transferred_mentions', 0)}"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. TASK-CONTEXT TOOL DEFINITIONS (injected only during task execution)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK_CONTEXT_TOOLS = [
    {
        "name": "save_finding",
        "description": "Save intermediate findings to the active mission timeline. Visible to both chat and future tasks. Use to preserve important progress, decisions, and discoveries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Text to save (findings, partial results, notes)."},
                "event_type": {
                    "type": "string",
                    "enum": ["finding", "decision"],
                    "default": "finding",
                    "description": "Type of event: finding (discovery/result) or decision (strategic choice).",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "read_user_chat",
        "description": "Read the user's actual chat messages with the orchestrator. Use when the delegation context is unclear or you need to verify the user's original intent/wording. Returns timestamped messages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent messages to fetch (default 10, max 30).",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "send_message",
        "description": "Post a message to the mission bulletin board, visible to all sibling agents working on the same mission. Use for: intermediate findings, warnings, or status updates that other agents should see before they finish.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to share with sibling agents."},
                "content": {"type": "string", "description": "Alias for message; accepted for compatibility."},
                "event_type": {
                    "type": "string",
                    "description": "Ignored compatibility field. Use save_finding for typed mission timeline events.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_messages",
        "description": "Read messages from sibling agents on the same mission bulletin board. Returns timestamped entries from other agents.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


def build_task_context_tools(task_id: int, user_id: int, depth: int = 0, mission_id: int | None = None):
    """Build task-context tool handlers with task_id/mission_id bound via closure.

    Returns (tools_list, handlers_dict) ready to merge into the tool loop.
    """

    async def _exec_save_finding(content: str, event_type: str = "finding") -> str:
        if not mission_id:
            return "No mission linked to this task — finding not saved."
        try:
            from telegram.mission import add_mission_event
            truncated = content[:2000]
            await asyncio.to_thread(
                add_mission_event, mission_id, f"task#{task_id}", event_type, truncated
            )
            return f"Saved {event_type} to mission #{mission_id} ({len(truncated)} chars)."
        except Exception as e:
            logger.error("save_finding error (task %d): %s", task_id, e)
            return f"Failed to save finding: {e}"

    async def _exec_read_user_chat(limit: int = 10) -> str:
        """Fetch the user's actual chat messages with timestamps."""
        if not user_id or user_id == 0:
            return "No user context available for this task."
        limit = max(1, min(limit, 30))
        try:
            from db import query as db_query
            rows = db_query(
                "SELECT role, content, created_at FROM ("
                "  SELECT role, content, created_at, id FROM telegram_chat_history"
                "  WHERE user_id = %s ORDER BY id DESC LIMIT %s"
                ") sub ORDER BY id ASC",
                (user_id, limit),
            )
            if not rows:
                return "No chat history found."
            lines = []
            for r in rows:
                role_label = "user" if r["role"] == "user" else "lenin"
                ts = r.get("created_at")
                if ts and hasattr(ts, "strftime"):
                    ts_kst = ts.astimezone(_KST) if ts.tzinfo else ts
                    time_str = ts_kst.strftime("%Y-%m-%d %H:%M")
                else:
                    time_str = "?"
                text = str(r["content"] or "")
                # Skip system markers
                if text.startswith("[SYSTEM]"):
                    continue
                lines.append(f"[{time_str}] [{role_label}] {text[:500]}")
            return "\n".join(lines) if lines else "No user messages found."
        except Exception as e:
            logger.error("read_user_chat error (task %d): %s", task_id, e)
            return f"Failed to read chat: {e}"

    async def _exec_send_message(message: str | None = None, content: str | None = None, event_type: str | None = None) -> str:
        """Post to mission bulletin board."""
        del event_type
        message_text = (message if message is not None else content) or ""
        message_text = str(message_text).strip()
        if not message_text:
            return "No message content provided."
        if not mission_id:
            return "No mission linked to this task — message not posted."
        try:
            from redis_state import post_to_board
            agent_type_str = ""
            try:
                from telegram.bot import current_task_ctx
                ctx = current_task_ctx.get()
                agent_type_str = (ctx or {}).get("agent_type", "")
            except Exception:
                pass
            post_to_board(mission_id, task_id, agent_type_str, message_text)
            return f"Message posted to mission #{mission_id} board."
        except Exception as e:
            logger.error("send_message error (task %d): %s", task_id, e)
            return f"Failed to post message: {e}"

    async def _exec_read_messages() -> str:
        """Read mission bulletin board messages from sibling agents."""
        if not mission_id:
            return "No mission linked to this task."
        try:
            from redis_state import read_board
            from datetime import datetime, timezone
            messages = read_board(mission_id)
            if not messages:
                return "No messages on the mission board."
            lines = []
            for m in messages:
                ts = m.get("ts", 0)
                time_str = datetime.fromtimestamp(ts, tz=_KST).strftime("%H:%M") if ts else "?"
                agent = m.get("agent", "?")
                tid = m.get("task_id", "?")
                lines.append(f"[{time_str}] [{agent} #{tid}] {m.get('message', '')}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("read_messages error (task %d): %s", task_id, e)
            return f"Failed to read messages: {e}"

    handlers = {
        "save_finding": _exec_save_finding,
        "read_user_chat": _exec_read_user_chat,
        "send_message": _exec_send_message,
        "read_messages": _exec_read_messages,
    }
    return list(TASK_CONTEXT_TOOLS), handlers


async def _exec_read_self(
    content_type: str | None = None, source: str | None = None,
    id: int | None = None, limit: int | None = None, keyword: str | None = None,
    max_chars: int | None = None, offset: int | None = None,
    post_id: int | None = None, diary_id: int | None = None,
    hours_back: int | None = None, service: str = "telegram",
    grep: str | list[str] | tuple[str, ...] | None = "", status: str | None = None, task_id: int | None = None,
    chat_source: str = "web", slug: str | None = None,
) -> str:
    """Dispatcher for all read_self sources."""
    raw_type = (content_type or source or "").strip()
    alias_map = {
        "task_reports": "task_report",
        "task_report": "task_report",
        "research": "research_document",
        "research_documents": "research_document",
        "private_research_documents": "private_research_document",
        "private_reports": "private_research_document",
        "curation": "hub_curation",
        "curations": "hub_curation",
        "static_pages": "static_page",
        "posts": "blog_post",
        "post": "blog_post",
    }
    source = alias_map.get(raw_type, raw_type)
    if id is not None:
        if post_id is None:
            post_id = id
        if task_id is None:
            task_id = id
    if source == "diary":
        return await _exec_read_diary(
            limit=limit or 5,
            keyword=keyword,
            max_chars=max_chars,
            offset=offset,
            post_id=post_id,
            diary_id=diary_id,
        )
    if source == "chat_logs":
        return await _exec_read_chat_logs(limit=limit or 20, hours_back=hours_back, keyword=keyword, source=chat_source)
    if source == "processing_logs":
        return await _exec_read_processing_logs(limit=limit or 5, hours_back=hours_back, keyword=keyword)
    if source == "task_report":
        return await _exec_read_task_reports(
            limit=limit or 5,
            status=status,
            task_id=task_id,
            max_chars=max_chars,
            offset=offset,
        )
    if source == "kg_status":
        return await _exec_read_kg_status()
    if source == "system_status":
        return await _exec_read_system_status()
    if source == "server_logs":
        return await _exec_read_server_logs(service=service, minutes_back=(hours_back or 1) * 60, limit=limit or 50, grep=grep)
    if source == "recent_updates":
        return await _exec_read_recent_updates(max_entries=limit or 3)
    if source == "file_registry":
        return await _exec_read_file_registry(limit=limit or 20, keyword=keyword, category=None)
    if source == "research_document":
        return await _exec_read_research(
            limit=limit or 10,
            keyword=keyword,
            slug=slug,
            status=status,
            max_chars=max_chars,
            offset=offset,
        )
    if source == "private_research_document":
        return await _exec_read_private_research_documents(
            limit=limit or 10,
            keyword=keyword,
            slug=slug,
            max_chars=max_chars,
            offset=offset,
        )
    if source == "hub_curation":
        return await _exec_read_curation(limit=limit or 10, keyword=keyword, slug=slug)
    if source == "static_page":
        return await _exec_read_static_pages(
            limit=limit or 10,
            keyword=keyword,
            slug=slug,
            max_chars=max_chars,
            offset=offset,
        )
    if source == "blog_post":
        return await _exec_read_blog_posts(
            limit=limit or 5,
            keyword=keyword,
            max_chars=max_chars,
            post_id=post_id,
        )
    if source == "autonomous_project":
        return await _exec_read_autonomous_project(project_id=task_id, limit=limit or 10, keyword=keyword)
    return f"Unknown content_type: {source}"


async def _exec_read_autonomous_project(
    project_id: int | None = None, limit: int = 10, keyword: str | None = None,
) -> str:
    """Surface autonomous project state to the orchestrator.

    Without project_id → list of active/archived projects with one-line summary.
    With project_id    → full detail: goal, plan, recent notes, recent events.
    Keyword filter (if given) matches against title/topic on the list form, and
    against note content on the detail form.
    """
    from db import query as db_query, query_one as db_query_one

    # ── LIST MODE ─────────────────────────────────────────────
    if not project_id:
        clauses = []
        params: list = []
        if keyword:
            clauses.append("(title ILIKE %s OR topic ILIKE %s)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(min(limit, 50))
        try:
            rows = await asyncio.to_thread(
                db_query,
                f"SELECT id, title, topic, state, turn_count, last_run_at, created_at "
                f"FROM autonomous_projects {where} "
                f"ORDER BY CASE state WHEN 'researching' THEN 0 WHEN 'planning' THEN 0 "
                f"                    WHEN 'paused' THEN 1 ELSE 2 END, id DESC LIMIT %s",
                tuple(params),
            )
        except Exception as e:
            return f"=== AUTONOMOUS PROJECTS ===\n(error: {e})"
        if not rows:
            return "=== AUTONOMOUS PROJECTS ===\n(none)"
        lines = ["=== AUTONOMOUS PROJECTS ===",
                 "Use read_self(content_type='autonomous_project', id=<id>) for full detail."]
        for r in rows:
            last = _to_kst(r.get("last_run_at")) if r.get("last_run_at") else "never"
            topic = (r.get("topic") or "").replace("\n", " ")[:150]
            lines.append(
                f"#{r['id']} [{r['state']}] turns={r['turn_count']} last_run={last}\n"
                f"  title: {r['title']}\n"
                f"  topic: {topic}{'…' if len(r.get('topic') or '') > 150 else ''}"
            )
        return "\n".join(lines)

    # ── DETAIL MODE ───────────────────────────────────────────
    try:
        proj = await asyncio.to_thread(
            db_query_one,
            "SELECT id, title, topic, goal, state, plan, research_notes, "
            "turn_count, last_run_at, created_at "
            "FROM autonomous_projects WHERE id = %s",
            (project_id,),
        )
    except Exception as e:
        return f"=== AUTONOMOUS PROJECT #{project_id} ===\n(error: {e})"
    if not proj:
        return f"=== AUTONOMOUS PROJECT #{project_id} ===\n(not found)"

    legacy_notes = proj.get("research_notes") or []
    plan = proj.get("plan") or {}
    note_limit = min(limit, 10)
    note_total = len(legacy_notes)
    note_source = "legacy JSONB"
    try:
        note_clauses = ["project_id = %s"]
        note_params: list = [project_id]
        if keyword:
            note_clauses.append("text ILIKE %s")
            note_params.append(f"%{keyword}%")
        note_count = await asyncio.to_thread(
            db_query_one,
            f"SELECT COUNT(*) AS count FROM autonomous_project_notes WHERE {' AND '.join(note_clauses)}",
            tuple(note_params),
        )
        note_total = int((note_count or {}).get("count") or 0)
        note_rows = await asyncio.to_thread(
            db_query,
            f"""
            SELECT turn, text, sources, created_at
              FROM autonomous_project_notes
             WHERE {' AND '.join(note_clauses)}
             ORDER BY created_at DESC, id DESC
             LIMIT %s
            """,
            tuple([*note_params, note_limit]),
        )
        recent_notes = list(reversed([dict(row) for row in note_rows]))
        note_source = "autonomous_project_notes"
    except Exception:
        notes = legacy_notes
        if keyword:
            notes = [n for n in notes if keyword.lower() in (n.get("text") or "").lower()]
        note_total = len(notes)
        recent_notes = notes[-note_limit:]

    # Recent events (last `limit` entries)
    try:
        events = await asyncio.to_thread(
            db_query,
            "SELECT id, created_at, event_type, content FROM autonomous_project_events "
            "WHERE project_id = %s ORDER BY created_at DESC, id DESC LIMIT %s",
            (project_id, min(limit, 20)),
        )
    except Exception:
        events = []

    try:
        tick_logs = await asyncio.to_thread(
            db_query,
            """
            SELECT content, meta, created_at
              FROM autonomous_project_events
             WHERE project_id = %s
               AND event_type = 'tick_tool_log'
             ORDER BY created_at DESC, id DESC
             LIMIT 1
            """,
            (project_id,),
        )
        last_tick_log = dict(tick_logs[0]) if tick_logs else None
    except Exception:
        last_tick_log = None

    try:
        tick_errors = await asyncio.to_thread(
            db_query,
            """
            SELECT content, created_at
              FROM autonomous_project_events
             WHERE project_id = %s
               AND event_type = 'tick_error'
             ORDER BY created_at DESC, id DESC
             LIMIT 1
            """,
            (project_id,),
        )
        last_tick_error = dict(tick_errors[0]) if tick_errors else None
    except Exception:
        last_tick_error = None

    try:
        no_action_rows = await asyncio.to_thread(
            db_query,
            """
            SELECT content, created_at
              FROM autonomous_project_events
             WHERE project_id = %s
               AND event_type = 'tick_no_durable_action'
             ORDER BY created_at DESC, id DESC
             LIMIT 1
            """,
            (project_id,),
        )
        last_no_action = dict(no_action_rows[0]) if no_action_rows else None
    except Exception:
        last_no_action = None

    try:
        staged_rows = await asyncio.to_thread(
            db_query,
            """
            SELECT ev.content, ev.meta, ev.created_at
              FROM autonomous_project_events ev
              JOIN research_documents rd
                ON rd.id::text = ev.meta->>'research_document_id'
                OR rd.filename = ev.meta->>'filename'
                OR rd.slug = ev.meta->>'slug'
             WHERE ev.project_id = %s
               AND ev.event_type = 'research_draft_staged'
               AND rd.status = 'staged'
             ORDER BY ev.created_at DESC, ev.id DESC
             LIMIT 1
            """,
            (project_id,),
        )
        last_staged_draft = dict(staged_rows[0]) if staged_rows else None
    except Exception:
        last_staged_draft = None

    out = [f"=== AUTONOMOUS PROJECT #{proj['id']}: {proj['title']} ==="]
    out.append(f"state: {proj['state']}   turns: {proj['turn_count']}   last_run: {_to_kst(proj.get('last_run_at')) if proj.get('last_run_at') else 'never'}")
    out.append(f"topic: {proj.get('topic') or ''}")
    out.append("")
    out.append("-- goal --")
    out.append((proj.get("goal") or "").strip())
    out.append("")

    # Operator advisories (pending + recent consumed)
    try:
        advisories = await asyncio.to_thread(
            db_query,
            "SELECT id, content, created_at, consumed_at FROM autonomous_project_advisories "
            "WHERE project_id = %s ORDER BY created_at DESC LIMIT 10",
            (project_id,),
        )
    except Exception:
        advisories = []
    if advisories:
        pending = [a for a in advisories if a["consumed_at"] is None]
        consumed = [a for a in advisories if a["consumed_at"] is not None]
        out.append(f"-- operator advisories (pending: {len(pending)}, recent consumed: {len(consumed)}) --")
        for a in pending:
            ts = _to_kst(a.get("created_at"))
            out.append(f"[PENDING #{a['id']} @ {ts}]")
            out.append(f"  {(a.get('content') or '')[:500]}")
        for a in consumed[:3]:
            ts = _to_kst(a.get("created_at"))
            out.append(f"[consumed #{a['id']} @ {ts}]")
            out.append(f"  {(a.get('content') or '')[:300]}")
        out.append("")

    out.append("-- plan --")
    goals = plan.get("goals") or []
    steps = plan.get("steps") or []
    if not goals and not steps:
        out.append("(empty)")
    else:
        if goals:
            out.append("Goals:")
            out.extend(f"  - {g}" for g in goals)
        if steps:
            out.append("Steps:")
            out.extend(f"  {i+1}. {s}" for i, s in enumerate(steps))
        if plan.get("rationale"):
            out.append(f"(rationale: {plan['rationale']})")
    out.append("")

    out.append(f"-- recent notes ({len(recent_notes)} shown / {note_total} total, source={note_source}) --")
    if not recent_notes:
        out.append("(no notes)")
    else:
        for n in recent_notes:
            text = (n.get("text") or "")[:600]
            sources = n.get("sources") or []
            if isinstance(sources, str):
                try:
                    sources = json.loads(sources)
                except Exception:
                    sources = [sources]
            if not isinstance(sources, list):
                sources = [str(sources)]
            sources = [str(source) for source in sources]
            src = f"  [{', '.join(sources[:3])}{'…' if len(sources) > 3 else ''}]" if sources else ""
            out.append(f"[turn {n.get('turn', '?')}, {_to_kst(n.get('created_at'))}]")
            out.append(f"  {text}{'…' if len(n.get('text') or '') > 600 else ''}{src}")
    out.append("")

    if last_staged_draft:
        out.append("-- last staged research draft (" + _to_kst(last_staged_draft.get("created_at")) + ") --")
        content = str(last_staged_draft.get("content") or "")
        meta = last_staged_draft.get("meta") or {}
        slug = meta.get("slug") or str(meta.get("filename") or "").removesuffix(".md")
        out.append(content[:1200] + ("…" if len(content) > 1200 else ""))
        if slug:
            out.append(f"read draft: read_self(content_type=\"research_document\", slug=\"{slug}\", status=\"staged\")")
        out.append("")
    if last_tick_error:
        out.append(f"-- last tick error ({_to_kst(last_tick_error.get('created_at'))}) --")
        content = str(last_tick_error.get("content") or "")
        out.append(content[:1200] + ("…" if len(content) > 1200 else ""))
        out.append("")

    if last_no_action:
        out.append(f"-- last tick no durable action ({_to_kst(last_no_action.get('created_at'))}) --")
        content = str(last_no_action.get("content") or "")
        out.append(content[:1200] + ("…" if len(content) > 1200 else ""))
        out.append("")

    if last_tick_log:
        meta = last_tick_log.get("meta") or {}
        header_bits = []
        if meta.get("turn") is not None:
            header_bits.append(f"turn={meta.get('turn')}")
        if meta.get("rounds_used") is not None:
            header_bits.append(f"rounds={meta.get('rounds_used')}")
        if meta.get("tool_calls") is not None:
            header_bits.append(f"tools={meta.get('tool_calls')}")
        if meta.get("cost_usd") is not None:
            header_bits.append(f"cost=${float(meta.get('cost_usd') or 0):.3f}")
        out.append(f"-- last tick tool log ({', '.join(header_bits) or _to_kst(last_tick_log.get('created_at'))}) --")
        content = str(last_tick_log.get("content") or "")
        out.append(content[:1800] + ("…" if len(content) > 1800 else ""))
        out.append("")

    out.append(f"-- recent events ({len(events)}) --")
    if not events:
        out.append("(no events)")
    else:
        for e in events:
            snippet = (e.get("content") or "").replace("\n", " ")[:160]
            out.append(f"[{_to_kst(e.get('created_at'))}] {e['event_type']}: {snippet}")

    return "\n".join(out)


async def _exec_read_file_registry(limit: int = 20, keyword: str | None = None, category: str | None = None) -> str:
    """Search registered files (uploaded to R2 or tracked locally)."""
    from db import query as db_query

    clauses = []
    params: list = []
    if keyword:
        clauses.append("(filename ILIKE %s OR description ILIKE %s OR local_path ILIKE %s OR public_url ILIKE %s)")
        params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
    if category:
        clauses.append("category = %s")
        params.append(category)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(min(limit, 50))

    rows = await asyncio.to_thread(
        db_query,
        f"SELECT id, filename, public_url, local_path, content_type, description, category, file_size, "
        f"created_by_task_id, created_at FROM file_registry {where} ORDER BY created_at DESC LIMIT %s",
        tuple(params),
    )
    if not rows:
        return "=== FILE REGISTRY ===\n(no files registered)"
    lines = ["=== FILE REGISTRY ==="]
    for r in rows:
        ts = _to_kst(r.get("created_at"))
        size_kb = round((r.get("file_size") or 0) / 1024, 1)
        lines.append(
            f"[{r['id']}] {r['filename']} ({size_kb}KB, {r.get('category', '-')})\n"
            f"  url: {r.get('public_url') or '(local only)'}\n"
            f"  local: {r.get('local_path')}\n"
            f"  desc: {r.get('description') or '-'}\n"
            f"  task: #{r.get('created_by_task_id') or '-'} | {ts}"
        )
    return "\n".join(lines)


async def _exec_kg_admin(
    action: str, query: str = "", write: bool = False,
    episode_name: str = "", source_name: str = "", target_name: str = "",
) -> str:
    """Dispatcher for KG admin operations."""
    if action == "query":
        if not query:
            return "Error: 'query' parameter required for action=query."
        return await _exec_kg_query(query=query, write=write)
    if action == "delete_episode":
        if not episode_name:
            return "Error: 'episode_name' parameter required for action=delete_episode."
        return await _exec_kg_delete_episode(episode_name=episode_name)
    if action == "merge_entities":
        if not source_name or not target_name:
            return "Error: 'source_name' and 'target_name' required for action=merge_entities."
        return await _exec_kg_merge_entities(source_name=source_name, target_name=target_name)
    return f"Unknown action: {action}"


SELF_TOOL_HANDLERS = {
    "read_self": _exec_read_self,
    "recall_experience": _exec_recall_experience,
    "save_self_analysis": _exec_save_self_analysis,
    "write_kg": _exec_write_kg,
    "write_kg_structured": _exec_write_kg_structured,
    "delegate": _exec_delegate,
    "multi_delegate": _exec_multi_delegate,
    "route_task": _exec_route_task,
    "list_agent_tools": _exec_list_agent_tools,
    "kg_admin": _exec_kg_admin,
}
