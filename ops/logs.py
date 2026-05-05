"""Systemd journal log query helpers."""

import logging

logger = logging.getLogger(__name__)

def _normalize_grep_terms(grep) -> list[str]:
    """Normalize grep input into a clean list of lowercase terms.

    Accepts string, list/tuple/set, nested combinations, or arbitrary scalars.
    This exists because verification paths may pass grep as a list, and some
    older call sites may stringify collections before reaching this function.
    """
    if grep is None:
        return []

    if isinstance(grep, str):
        candidate_items = [grep]
    elif isinstance(grep, (list, tuple, set)):
        candidate_items = list(grep)
    else:
        candidate_items = [grep]

    normalized_terms: list[str] = []
    seen_terms: set[str] = set()
    for item in candidate_items:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            for nested in _normalize_grep_terms(item):
                if nested not in seen_terms:
                    normalized_terms.append(nested)
                    seen_terms.add(nested)
            continue
        text = str(item).strip().lower()
        if text and text not in seen_terms:
            normalized_terms.append(text)
            seen_terms.add(text)
    return normalized_terms


def grep_matches_text(text, grep) -> bool:
    """Return True when text matches the provided grep filter(s)."""
    terms = _normalize_grep_terms(grep)
    if not terms:
        return True
    text_lower = str(text or "").lower()
    return any(term in text_lower for term in terms)



def fetch_server_logs(service: str = "api", hours_back: int = 1, grep: str | list[str] | tuple[str, ...] | None = None, limit: int = 200) -> list[dict]:
    """Fetch local systemd/journald service logs.

    Args:
        service: Logical service name (api, telegram, nginx).
        hours_back: How many recent hours to inspect.
        grep: Optional substring filter(s) applied after fetching.
              Accepts a string or collection of strings.
        limit: Max returned log lines.

    Returns list of dicts: timestamp, message, raw.
    """
    import subprocess

    service_map = {
        "api": "leninbot-api",
        "telegram": "leninbot-telegram",
        "nginx": "nginx",
    }

    if isinstance(service, (list, tuple, set)):
        service = next((str(item).strip() for item in service if str(item).strip()), "api")
    service_name = str(service or "api").strip().lower()
    unit = service_map.get(service_name)
    if not unit:
        return [{"error": f"Unknown service: {service}"}]

    hours_back = max(1, min(int(hours_back or 1), 168))
    limit = max(1, min(int(limit or 200), 1000))
    grep_terms_lower = _normalize_grep_terms(grep)

    cmd = [
        "journalctl",
        "-u",
        unit,
        "--since",
        f"-{hours_back} hour",
        "-n",
        str(limit),
        "--no-pager",
        "-o",
        "short-iso",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except Exception as e:
        logger.error("[shared] fetch_server_logs error: %s", e)
        return [{"error": str(e)}]

    if proc.returncode not in (0, 1):
        err = (proc.stderr or proc.stdout or "journalctl failed").strip()
        logger.error("[shared] fetch_server_logs journalctl failure: %s", err)
        return [{"error": err}]

    rows = []
    for line in (proc.stdout or "").splitlines():
        text = line.strip()
        if not text:
            continue
        if grep_terms_lower and not grep_matches_text(text, grep_terms_lower):
            continue
        timestamp = ""
        message = text
        if " " in text:
            first_sep = text.find(" ")
            second_sep = text.find(" ", first_sep + 1)
            third_sep = text.find(" ", second_sep + 1) if second_sep != -1 else -1
            if third_sep != -1:
                timestamp = text[:third_sep]
                message = text[third_sep + 1 :]
        rows.append({"timestamp": timestamp, "message": message, "raw": text})
    return rows


