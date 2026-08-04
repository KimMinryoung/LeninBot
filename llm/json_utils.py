"""Robust JSON-object extraction from LLM output."""
import json
import re


def extract_json_object(text: str) -> dict | None:
    """Parse the first JSON object out of model output.

    Tolerates ``` / ```json fences and surrounding prose (falls back to the
    outermost {...} slice). Returns None when nothing parses to a dict.
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
