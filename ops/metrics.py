"""Server metric history readers shared by the Telegram /stats command and
``scripts/metrics_snapshot.py``."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from ops.paths import PROJECT_ROOT


JSON_DATA_DIR = PROJECT_ROOT / "data" / "metrics"


# ─── JSON 데이터 파서 ────────────────────────────────────
def _load_json_snapshots(hours_back: int) -> list[dict]:
    """
    hours_back 시간에 해당하는 월별 JSON 파일을 자동으로 합쳐서 반환.
    예: hours=720 → 2개 월 파일 합산.
    """
    now = datetime.now()
    cutoff = now - timedelta(hours=hours_back)

    # 필요한 연월 목록 생성
    months = set()
    cur = cutoff.replace(day=1)
    while cur <= now:
        months.add((cur.year, cur.month))
        # 다음 달로
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    all_snaps = []
    for year, month in sorted(months):
        path = JSON_DATA_DIR / f"{year:04d}-{month:02d}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_snaps.extend(data.get("snapshots", []))
            except (json.JSONDecodeError, IOError) as e:
                print(f"[경고] {path} 읽기 실패: {e}", file=sys.stderr)

    # cutoff 이후 데이터만 필터링 + 시간순 정렬
    filtered = []
    for s in all_snaps:
        try:
            dt = datetime.fromisoformat(s["ts"])
            if dt >= cutoff:
                filtered.append(s)
        except (KeyError, ValueError):
            pass

    return sorted(filtered, key=lambda x: x["ts"])


def parse_cpu_json(hours_back: int) -> list[tuple[datetime, float]]:
    snaps = _load_json_snapshots(hours_back)
    return [(datetime.fromisoformat(s["ts"]), s["cpu_pct"]) for s in snaps if "cpu_pct" in s]


def parse_memory_json(hours_back: int) -> list[tuple[datetime, float]]:
    snaps = _load_json_snapshots(hours_back)
    return [(datetime.fromisoformat(s["ts"]), s["mem_pct"]) for s in snaps if "mem_pct" in s]


def parse_disk_io_json(hours_back: int) -> list[tuple[datetime, float]]:
    snaps = _load_json_snapshots(hours_back)
    return [(datetime.fromisoformat(s["ts"]), s["disk_tps"]) for s in snaps if "disk_tps" in s]


def parse_disk_usage_json(hours_back: int) -> list[tuple[datetime, float]]:
    snaps = _load_json_snapshots(hours_back)
    return [(datetime.fromisoformat(s["ts"]), s["disk_pct"]) for s in snaps if "disk_pct" in s]


# ─── ASCII 그래프 렌더러 ──────────────────────────────────
def _sparkline(values: list[float]) -> str:
    """미니 스파크라인 (Unicode block chars)"""
    chars = " ▁▂▃▄▅▆▇█"
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1
    return "".join(chars[int((v - lo) / span * 8)] for v in values)
