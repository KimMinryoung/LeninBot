"""Small timestamp helpers shared across modules."""
from datetime import datetime


def timestamp_sort_key(ts) -> float:
    """Sort key for mixed datetime / ISO-string / None timestamps (None → 0)."""
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
