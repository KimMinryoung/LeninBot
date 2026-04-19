"""ingest_reports_to_kg.py — Batch ingest task reports into Knowledge Graph.

Uses local LLM (moon PC, cost=0) to extract key facts from task reports,
then stores them via add_kg_episode() → Graphiti entity/relationship extraction.

Usage:
    python scripts/ingest_reports_to_kg.py [--limit 60] [--dry-run]
"""

import os
import sys
import time
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from db import query as db_query
from shared import add_kg_episode
from llm.client import ask as llm_ask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """\
아래 태스크 리포트에서 **Knowledge Graph에 저장할 가치가 있는 사실**만 bullet point로 추출하라.

규칙:
- 고유명사, 수치, 날짜, 인과관계 위주
- 코드 변경/시스템 설정 같은 내부 작업은 제외
- 지정학, 경제, 사회, 기술 관련 사실만 포함
- 사실이 없으면 "SKIP"이라고만 써라
- 최대 10개 bullet point, 각 1~2문장

리포트:
{report}
"""

GROUP_PROMPT = """\
아래 사실 목록의 주제를 분류하라. 다음 중 하나만 답하라:
geopolitics_conflict, diplomacy, economy, korea_domestic, agent_knowledge

사실:
{facts}
"""

GROUP_KEYWORDS = {
    "geopolitics_conflict": ["전쟁", "군사", "제재", "분쟁", "NATO", "핵", "미사일", "침공"],
    "diplomacy": ["외교", "정상회담", "조약", "협상", "대사", "UN", "summit"],
    "economy": ["경제", "금리", "주가", "GDP", "인플레", "무역", "수출", "투자", "시장", "반도체"],
    "korea_domestic": ["한국", "국회", "정부", "대통령", "선거", "부동산", "고용"],
}


def classify_group_heuristic(facts: str) -> str:
    """Fast keyword-based group classification (no LLM needed)."""
    lower = facts.lower()
    scores = {}
    for group, keywords in GROUP_KEYWORDS.items():
        scores[group] = sum(1 for k in keywords if k.lower() in lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "agent_knowledge"


def process_report(task_id: int, agent_type: str, result: str, dry_run: bool = False) -> dict:
    """Extract facts from one report and store in KG."""
    # Truncate very long reports
    report_text = result[:6000]

    # Extract facts via local LLM
    prompt = EXTRACT_PROMPT.format(report=report_text)
    facts = llm_ask(prompt, temperature=0.3)

    if not facts or "SKIP" in facts.strip().upper()[:10]:
        return {"task_id": task_id, "status": "skipped", "reason": "no facts"}

    # Classify group (heuristic, no LLM call)
    group_id = classify_group_heuristic(facts)

    if dry_run:
        logger.info("[DRY-RUN] Task #%d → group=%s\n%s", task_id, group_id, facts[:300])
        return {"task_id": task_id, "status": "dry_run", "group_id": group_id, "facts": facts[:200]}

    # Store in KG
    episode_name = f"task-report-{task_id}"
    result = add_kg_episode(
        content=facts,
        name=episode_name,
        source_type="internal_report",
        group_id=group_id,
    )

    if result["status"] == "ok":
        logger.info("Task #%d → KG (%s): %s", task_id, group_id, result["message"])
        return {"task_id": task_id, "status": "ok", "group_id": group_id}
    else:
        logger.warning("Task #%d → KG failed: %s", task_id, result["message"])
        return {"task_id": task_id, "status": "error", "message": result["message"]}


def main():
    parser = argparse.ArgumentParser(description="Ingest task reports into KG")
    parser.add_argument("--limit", type=int, default=60, help="Number of reports to process")
    parser.add_argument("--dry-run", action="store_true", help="Extract facts but don't write to KG")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between reports (for LLM rate limit)")
    args = parser.parse_args()

    rows = db_query(
        "SELECT id, agent_type, result FROM telegram_tasks "
        "WHERE status = 'done' AND result IS NOT NULL AND result != '' "
        "ORDER BY completed_at DESC LIMIT %s",
        (args.limit,),
    )
    logger.info("Found %d task reports to process", len(rows))

    stats = {"ok": 0, "skipped": 0, "error": 0, "dry_run": 0}
    for i, row in enumerate(rows, 1):
        task_id = row["id"]
        agent_type = row.get("agent_type") or "?"
        result_text = row["result"]

        logger.info("[%d/%d] Processing task #%d (agent=%s, len=%d)",
                     i, len(rows), task_id, agent_type, len(result_text))

        try:
            out = process_report(task_id, agent_type, result_text, dry_run=args.dry_run)
            stats[out["status"]] = stats.get(out["status"], 0) + 1
        except Exception as e:
            logger.error("Task #%d failed: %s", task_id, e)
            stats["error"] += 1

        if i < len(rows):
            time.sleep(args.delay)

    logger.info("Done. Results: %s", stats)


if __name__ == "__main__":
    main()
