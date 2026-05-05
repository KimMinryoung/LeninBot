"""Scout-report-to-KG ingestion heuristic."""

import logging
from datetime import datetime, timedelta, timezone

from kg_runtime.writes import add_kg_episode

logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))

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

        # Determine group_id from task_content keywords
        group_id = "agent_knowledge"  # default
        content_lower = (task_content + " " + report[:500]).lower()

        if any(k in content_lower for k in ("미국", "중국", "러시아", "북한", "전쟁", "제재", "외교", "정책", "영토", "분쟁")):
            group_id = "geopolitics_conflict"
        elif any(k in content_lower for k in ("ai", "기술", "투자", "주가", "시장", "경제", "산업", "노동", "실업", "임금")):
            group_id = "economy"
        elif any(k in content_lower for k in ("한국", "대한민국", "서울", "울산", "광주", "부산", "정부", "국회", "청와대", "정당")):
            group_id = "korea_domestic"

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

        # Check for potential duplicates (simple heuristic)
        # If any of the fact lines appear in recent episodes, skip
        try:
            from db import query as _db_query
            recent = _db_query(
                "SELECT name FROM telegram_tasks WHERE agent_type = 'scout' "
                "AND status = 'done' AND completed_at > NOW() - INTERVAL 1 DAY "
                "ORDER BY completed_at DESC LIMIT 5"
            )
            # Could add more sophisticated dedup here if needed
        except Exception:
            pass  # Non-critical

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
