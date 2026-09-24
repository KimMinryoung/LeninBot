"""Scout-report-to-KG ingestion heuristic."""

import logging
from datetime import datetime

from kg_runtime.writes import add_kg_episode

logger = logging.getLogger(__name__)
from shared import KST

# KG group ids the classifier may choose from (self_runtime/tools.py enum과 동일).
KG_GROUP_IDS = (
    "geopolitics_conflict",
    "diplomacy",
    "economy",
    "korea_domestic",
    "agent_knowledge",
)

_GROUP_CLASSIFY_PROMPT = """\
You are routing an OSINT scout report into a knowledge-graph group.
Pick exactly ONE group id from this list:

- geopolitics_conflict: wars, military actions, sanctions, territorial disputes, security tensions
- diplomacy: negotiations, treaties, summits, alliances, formal inter-state relations
- economy: markets, industry, trade, investment, labor, technology business
- korea_domestic: South Korean internal politics and society
- agent_knowledge: none of the above fits clearly

Output ONLY the group id, nothing else.

[Task instructions]
{task}

[Report findings]
{findings}
"""


FACT_FILTER_FEATURE = "scout_kg_fact_filter"
FACT_FILTER_CANDIDATES = 12   # lines judged per report
FACT_FILTER_KEEP = 7          # facts stored per episode
_FACT_INSTRUCTIONS = (
    "This line reports a fact about the outside world (a news event, a figure, an organisation, a statement by a "
    "public actor, a development in politics, economy or technology) that is worth storing in a knowledge base. "
    "It is NOT a note about the agent's own procedure: mailbox state, UID numbers, what was saved or skipped, "
    "timestamps of checks, reasoning about how to proceed, or table headers."
)


def _filter_fact_lines(task_content: str, lines: list[str]) -> tuple[list[str], dict]:
    """Keep the lines a System One judgement says report a fact about the world.

    Scout reports are mostly mailbox bookkeeping ("INBOX 최고 UID 345로 변동
    없음", "Let me reconsider…"); on 2026-09-19, 92 of 100 sampled lines were
    that, and all of them were being written to the knowledge graph as facts.
    One decision per report asks a noul per line (registry
    ``scout_kg_fact_filter``, threshold ``thresholds.keep``). When the model is
    unavailable every line is kept, as before, so an outage never drops news.
    Returns (kept lines, summary for logs/metrics).
    """
    candidates = [line for line in lines if line.strip()][:FACT_FILTER_CANDIDATES]
    if not candidates:
        return [], {"judged": 0, "kept": 0, "unavailable": False}
    from llm.call_registry import decide_sync, resolve

    profile = resolve(FACT_FILTER_FEATURE)
    extra = profile.extra or {}
    if not extra.get("enabled", True):
        return candidates[:FACT_FILTER_KEEP], {"judged": 0, "kept": len(candidates[:FACT_FILTER_KEEP]), "unavailable": False}
    keep = float((extra.get("thresholds") or {}).get("keep", 0.8))
    state = {"task": (task_content or "").strip()[:300],
             "lines": {f"line_{i + 1}": line[:600] for i, line in enumerate(candidates)}}
    questions = {f"line_{i + 1}": {"type": "noul", "instructions": f"Regarding line_{i + 1}: " + _FACT_INSTRUCTIONS}
                 for i in range(len(candidates))}
    decision = decide_sync(FACT_FILTER_FEATURE, state, questions)
    if decision is None:
        logger.warning("[Scout→KG] fact filter unavailable; keeping all %d lines", len(candidates))
        return candidates[:FACT_FILTER_KEEP], {"judged": 0, "kept": len(candidates[:FACT_FILTER_KEEP]), "unavailable": True}
    scored = [(decision.noul(f"line_{i + 1}") or 0.0, line) for i, line in enumerate(candidates)]
    kept = [line for p, line in scored if p >= keep][:FACT_FILTER_KEEP]
    logger.info("[Scout→KG] fact filter kept %d/%d lines (threshold %.2f)", len(kept), len(candidates), keep)
    return kept, {"judged": len(candidates), "kept": len(kept), "unavailable": False,
                  "scores": [round(p, 2) for p, _ in scored]}


def _classify_group_id(task_content: str, findings: str) -> str:
    """Classify a scout report into a KG group via the LLM call registry.

    Falls back to 'agent_knowledge' when the call fails or the model answers
    outside the known set — misrouting into the default group is cheaper than
    blocking ingestion. Model/options: config/llm_call_sites.json
    ("scout_kg_classify").
    """
    from llm.call_registry import generate_sync

    prompt = _GROUP_CLASSIFY_PROMPT.format(
        task=(task_content or "").strip()[:500] or "(none)",
        findings=(findings or "").strip()[:1500],
    )
    answer = (generate_sync("scout_kg_classify", prompt) or "").strip().lower()
    if not answer:
        logger.warning("[Scout→KG] group classification failed; using agent_knowledge")
        return "agent_knowledge"
    for group in KG_GROUP_IDS:
        if group in answer:
            return group
    logger.warning("[Scout→KG] classifier answered %r; using agent_knowledge", answer[:80])
    return "agent_knowledge"

def process_scout_report_to_kg(
    report: str,
    task_content: str = "",
    agent_type: str = "scout",
    task_id: int | None = None,
) -> dict:
    """
    Parse scout task report and auto-save factual findings to Knowledge Graph.

    This function:
    1. Extracts key findings from the scout report
    2. Determines appropriate group_id (geopolitics, economy, korea_domestic)
    3. Calls add_kg_episode() with source_type='internal_report'

    Args:
        report: Full task report text (markdown)
        task_content: Original task instructions (for context)
        agent_type: Agent type (default 'scout')

    Returns:
        dict with status, message, and episode_name
    """
    if agent_type != "scout":
        return {"status": "skip", "message": "Not a scout task"}

    if not report or not report.strip():
        return {"status": "skip", "message": "Empty report"}

    try:
        # Extract key sections from report
        # Look for Summary, Findings, or findings sections
        findings_section = ""
        for marker in ("## Findings", "## 발견사항", "## Summary", "## 요약"):
            idx = report.find(marker)
            if idx != -1:
                after = report[idx + len(marker):].strip()
                # Find next ## heading
                next_heading = after.find("\n## ")
                if next_heading != -1:
                    findings_section = after[:next_heading].strip()
                else:
                    findings_section = after.strip()
                if findings_section:
                    break

        if not findings_section:
            # Fallback: use first 1000 chars after first heading
            lines = report.split("\n")
            findings_section = "\n".join(lines[2:10]) if len(lines) > 2 else report[:1000]

        # Build factual content: bullet points from findings
        content_lines = []
        for line in findings_section.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                content_lines.append(line.lstrip("-•* ").strip())
            elif line and not line.startswith("#"):
                # Include non-heading lines as facts
                if len(line) > 20 and ":" in line:  # likely a fact statement
                    content_lines.append(line)

        if not content_lines:
            # Fallback: split findings by sentences
            import re
            sentences = re.split(r"[.。]", findings_section)
            content_lines = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15][:5]

        if not content_lines:
            return {"status": "skip", "message": "No factual content extracted"}

        # Keep only lines that state a fact about the world (fact filter); a
        # report that is all bookkeeping — most mailbox briefings — writes no
        # episode at all instead of a process log dressed as facts.
        content_lines, filter_summary = _filter_fact_lines(task_content, content_lines)
        if not content_lines:
            return {"status": "skip", "message": "No factual content after fact filter", "fact_filter": filter_summary}

        # Determine group_id with a light LLM call over the kept facts (keyword
        # substring matching misrouted anything containing "ai"/"정책" etc.)
        group_id = _classify_group_id(task_content, "\n".join(f"- {line}" for line in content_lines))

        # Build episode content
        ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
        episode_content = "\n".join(f"- {line}" for line in content_lines)
        import re
        reported_urls = list(dict.fromkeys(re.findall(r'https?://[^\s<>\]\)]+', report)))[:10]
        episode_content = (
            f"[Internal Scout Report: {ts}; task_id={task_id or 'unknown'}]\n"
            "Derived agent claims, not a primary news article or independent corroboration.\n"
            f"Full report: telegram_tasks:{task_id or 'unknown'}\n"
            f"URLs cited by the report (not independently verified here): {reported_urls}\n\n"
            + episode_content
        )

        # Write to KG
        result = add_kg_episode(
            content=episode_content,
            name=f"scout-patrol-{datetime.now(KST).strftime('%Y%m%d-%H%M%S')}"
                 + (f"-t{task_id}" if task_id else ""),
            source_type="internal_report",
            group_id=group_id,
        )

        if result["status"] == "ok":
            logger.info(
                "[Scout→KG] Successfully saved scout report to %s group | episode=%s",
                group_id, result.get("message")
            )
            return {
                "status": "ok",
                "message": result["message"],
                "group_id": group_id,
                "facts_count": len(content_lines),
                "fact_filter": filter_summary,
            }
        else:
            logger.warning("[Scout→KG] Failed to save: %s", result.get("message"))
            return {
                "status": "error",
                "message": result.get("message", "Unknown KG error"),
            }

    except Exception as e:
        logger.error("[Scout→KG] processing error: %s", e)
        return {
            "status": "error",
            "message": str(e),
        }
