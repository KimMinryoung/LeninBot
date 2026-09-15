"""Bound task callback context while keeping saved findings accessible."""

REPORT_CONTEXT_CHARS = 24000

RESULT_RELAY_GUIDANCE = (
    "Lead with the substantive answer to the original request. For a briefing, "
    "summarize the collected contents by source; file sizes and collection status "
    "do not substitute for the contents. Attribute source claims and keep caveats "
    "brief and relevant to the answer. If needed content is cropped, use read_self "
    "to read the saved task report in this turn before replying. Reading existing results "
    "is not re-delegation. Do not defer an available read to a future turn. "
    "If reading fails or the turn budget is exhausted, relay the findings available "
    "and state the specific remaining gap without inventing contents."
)


def report_for_callback(task_id: int, report: str) -> str:
    """Include ordinary reports whole, with an exact continuation for larger ones."""
    if len(report) <= REPORT_CONTEXT_CHARS:
        return report
    return (
        report[:REPORT_CONTEXT_CHARS]
        + f"\n\n[DISPLAY CROP: showing {REPORT_CONTEXT_CHARS} of {len(report)} chars. "
        "The full stored result is available; this says nothing about task completion. "
        "Read the omitted contents in this turn before summarizing them: "
        f"read_self(content_type='task_report', id={task_id}, "
        f"offset={REPORT_CONTEXT_CHARS}, max_chars=12000). "
        "Follow its next-page hint as needed. Do not re-delegate to recover stored text.]"
    )
