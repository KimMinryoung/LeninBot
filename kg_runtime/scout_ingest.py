"""Scout-report-to-KG ingestion heuristic."""

import logging
import os
from datetime import datetime, timedelta, timezone

from kg_runtime.writes import add_kg_episode

logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))

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


def _classify_group_id(task_content: str, findings: str) -> str:
    """Classify a scout report into a KG group with a light Gemini call.

    Falls back to 'agent_knowledge' when the key is missing, the call fails,
    or the model answers outside the known set — misrouting into the default
    group is cheaper than blocking ingestion.
    """
    try:
        from secrets_loader import get_secret

        api_key = (get_secret("GEMINI_API_KEY", "") or "").strip()
        if not api_key:
            logger.warning("[Scout→KG] no GEMINI_API_KEY; group defaults to agent_knowledge")
            return "agent_knowledge"

        from google import genai
        from google.genai.types import GenerateContentConfig

        prompt = _GROUP_CLASSIFY_PROMPT.format(
            task=(task_content or "").strip()[:500] or "(none)",
            findings=(findings or "").strip()[:1500],
        )
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("SCOUT_KG_CLASSIFY_MODEL", "gemini-3.1-flash-lite"),
            contents=prompt,
            config=GenerateContentConfig(temperature=0.0, max_output_tokens=32),
        )
        answer = (response.text or "").strip().lower()
        for group in KG_GROUP_IDS:
            if group in answer:
                return group
        logger.warning("[Scout→KG] classifier answered %r; using agent_knowledge", answer[:80])
        return "agent_knowledge"
    except Exception as e:
        logger.warning("[Scout→KG] group classification failed (%s); using agent_knowledge", e)
        return "agent_knowledge"

def process_scout_report_to_kg(
    report: str,
    task_content: str = "",
    agent_type: str = "scout",
) -> dict:
    """
    Parse scout task report and auto-save factual findings to Knowledge Graph.

    This function:
    1. Extracts key findings from the scout report
    2. Determines appropriate group_id (geopolitics, economy, korea_domestic)
    3. Calls add_kg_episode() with source_type='osint_news'

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

        # Determine group_id with a light LLM call (keyword substring matching
        # misrouted anything containing "ai"/"정책" etc.)
        group_id = _classify_group_id(task_content, findings_section)

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

        # Limit to 5-7 key facts
        content_lines = content_lines[:7]

        # Build episode content
        ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
        episode_content = "\n".join(f"- {line}" for line in content_lines)
        episode_content = f"[Scout Report: {ts}]\n\n{episode_content}"

        # Write to KG
        result = add_kg_episode(
            content=episode_content,
            name=f"scout-patrol-{datetime.now(KST).strftime('%Y%m%d-%H%M%S')}",
            source_type="osint_news",
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
