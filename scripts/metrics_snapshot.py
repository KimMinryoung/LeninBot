#!/usr/bin/env python3
"""
metrics_snapshot.py — 서버 자원 사용 추이를 ASCII 텍스트 그래프로 출력.

데이터 소스:
  --source sar   : sar 파일에서 실시간 읽기 (기본값, 최대 7일)
  --source json  : data/metrics/YYYY-MM.json 파일에서 읽기 (무한 보존)

사용법:
    python3 scripts/metrics_snapshot.py [--hours N] [--source sar|json] [--metric cpu|mem|disk|all]
    python3 scripts/metrics_snapshot.py --source json --hours 720  # 30일치
"""

import subprocess
import sys
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import re


# ─── 설정 ───────────────────────────────────────────────
GRAPH_WIDTH   = 50   # ASCII 그래프 막대 최대 너비 (문자 수)
SAR_DIR       = "/var/log/sysstat"
JSON_DATA_DIR = Path("/home/grass/leninbot/data/metrics")


# ─── sar 호출 헬퍼 ───────────────────────────────────────
def _run_sar(flag: str, hours_back: int) -> list[str]:
    """
    flag: '-u' (CPU) | '-r' (메모리) | '-d' (디스크 I/O) | '-n DEV' (네트워크)
    hours_back: 1 / 3 / 12 / 24 / ...
    반환: 파싱 대상 텍스트 라인 리스트 (헤더 제외, 시계열 순)
    """
    now = datetime.now()
    cutoff = now - timedelta(hours=hours_back)

    lines = []

    # 어제 파일이 필요한 경우
    if cutoff.date() < now.date():
        yesterday = now.date() - timedelta(days=1)
        sa_file = f"{SAR_DIR}/sa{yesterday.day:02d}"
        args = ["sar"] + flag.split() + ["-f", sa_file]
        r = subprocess.run(args, capture_output=True, text=True)
        lines += r.stdout.splitlines()

    # 오늘 파일
    today = now.date()
    sa_file = f"{SAR_DIR}/sa{today.day:02d}"
    args = ["sar"] + flag.split() + ["-f", sa_file]
    r = subprocess.run(args, capture_output=True, text=True)
    lines += r.stdout.splitlines()

    return lines


def _parse_time(t_str: str, ref_date) -> datetime | None:
    """'HH:MM:SS AM/PM' 문자열을 datetime으로 변환."""
    try:
        dt = datetime.strptime(t_str, "%I:%M:%S %p")
        return dt.replace(year=ref_date.year, month=ref_date.month, day=ref_date.day)
    except ValueError:
        return None


def _filter_by_hours(rows: list[tuple], hours_back: int) -> list[tuple]:
    """(datetime, value) 리스트에서 최근 N시간만 반환."""
    if not rows:
        return []
    cutoff = datetime.now() - timedelta(hours=hours_back)
    return [(dt, v) for dt, v in rows if dt >= cutoff]


# ─── sar 데이터 파서 ─────────────────────────────────────
def parse_cpu_sar(hours_back: int) -> list[tuple[datetime, float]]:
    """CPU %idle → %used = 100 - %idle"""
    lines = _run_sar("-u", hours_back)
    today = datetime.now().date()
    results = []
    current_date = today

    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        if "Average" in line or "CPU" in line or "Linux" in line:
            if "03/" in line or "/" in line:
                m = re.search(r"(\d{2}/\d{2}/\d{4})", line)
                if m:
                    current_date = datetime.strptime(m.group(1), "%m/%d/%Y").date()
            continue
        if len(parts) >= 9 and parts[2] == "all":
            dt = _parse_time(f"{parts[0]} {parts[1]}", current_date)
            if dt is None:
                continue
            try:
                idle = float(parts[8])
                used = round(100.0 - idle, 2)
                results.append((dt, used))
            except (ValueError, IndexError):
                pass

    return _filter_by_hours(results, hours_back)


def parse_memory_sar(hours_back: int) -> list[tuple[datetime, float]]:
    """%memused"""
    lines = _run_sar("-r", hours_back)
    today = datetime.now().date()
    results = []
    current_date = today

    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        if "Average" in line or "kbmemfree" in line or "Linux" in line:
            if "/" in line:
                m = re.search(r"(\d{2}/\d{2}/\d{4})", line)
                if m:
                    current_date = datetime.strptime(m.group(1), "%m/%d/%Y").date()
            continue
        if len(parts) >= 6:
            dt = _parse_time(f"{parts[0]} {parts[1]}", current_date)
            if dt is None:
                continue
            try:
                mem_pct = float(parts[5])
                results.append((dt, mem_pct))
            except (ValueError, IndexError):
                pass

    return _filter_by_hours(results, hours_back)


def parse_disk_io_sar(hours_back: int) -> list[tuple[datetime, float]]:
    """전체 디스크 tps (transactions/sec) 합산"""
    lines = _run_sar("-d", hours_back)
    today = datetime.now().date()
    current_date = today
    time_buckets: dict[datetime, float] = {}

    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        if "Average" in line or "DEV" in line or "Linux" in line:
            if "/" in line:
                m = re.search(r"(\d{2}/\d{2}/\d{4})", line)
                if m:
                    current_date = datetime.strptime(m.group(1), "%m/%d/%Y").date()
            continue
        if len(parts) >= 5:
            dt = _parse_time(f"{parts[0]} {parts[1]}", current_date)
            if dt is None:
                continue
            try:
                tps = float(parts[3])
                time_buckets[dt] = time_buckets.get(dt, 0.0) + tps
            except (ValueError, IndexError):
                pass

    results = sorted(time_buckets.items())
    return _filter_by_hours(results, hours_back)


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


# ─── 소스 통합 래퍼 ──────────────────────────────────────
def parse_cpu(hours_back: int, source: str) -> list[tuple[datetime, float]]:
    return parse_cpu_json(hours_back) if source == "json" else parse_cpu_sar(hours_back)

def parse_memory(hours_back: int, source: str) -> list[tuple[datetime, float]]:
    return parse_memory_json(hours_back) if source == "json" else parse_memory_sar(hours_back)

def parse_disk_io(hours_back: int, source: str) -> list[tuple[datetime, float]]:
    return parse_disk_io_json(hours_back) if source == "json" else parse_disk_io_sar(hours_back)


# ─── ASCII 그래프 렌더러 ──────────────────────────────────
def _bar(value: float, max_val: float, width: int = GRAPH_WIDTH) -> str:
    if max_val == 0:
        return ""
    filled = int(round(value / max_val * width))
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def _sparkline(values: list[float]) -> str:
    """미니 스파크라인 (Unicode block chars)"""
    chars = " ▁▂▃▄▅▆▇█"
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1
    return "".join(chars[int((v - lo) / span * 8)] for v in values)


def render_chart(
    title: str,
    unit: str,
    rows: list[tuple[datetime, float]],
    max_scale: float | None = None,
    alert_threshold: float | None = None,
) -> str:
    if not rows:
        return f"\n[{title}] 데이터 없음\n"

    vals = [v for _, v in rows]
    mx = max_scale or max(vals) or 1.0
    avg = sum(vals) / len(vals)
    peak = max(vals)
    cur = vals[-1]

    lines = []
    lines.append(f"\n{'═' * 62}")
    lines.append(f"  {title}  |  현재: {cur:.1f}{unit}  평균: {avg:.1f}{unit}  최고: {peak:.1f}{unit}")
    lines.append(f"{'─' * 62}")

    # 최대 20개 샘플만 표시 (너무 많으면 축약)
    display_rows = rows
    if len(rows) > 20:
        step = len(rows) / 20
        display_rows = [rows[int(i * step)] for i in range(20)]
        display_rows.append(rows[-1])  # 항상 최신 포함

    for dt, val in display_rows:
        time_str = dt.strftime("%m/%d %H:%M")
        bar = _bar(val, mx)
        flag = " ⚠" if alert_threshold and val >= alert_threshold else ""
        lines.append(f"  {time_str}  [{bar}] {val:5.1f}{unit}{flag}")

    lines.append(f"{'─' * 62}")
    lines.append(f"  스파크라인: {_sparkline(vals)}")
    lines.append(f"{'═' * 62}")
    return "\n".join(lines)


# ─── 메인 ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="서버 자원 사용 추이 출력")
    parser.add_argument(
        "--hours", type=int, default=1,
        help="조회 기간 (기본값: 1시간). --source sar 시: 1/3/6/12/24 권장. json 시: 제한 없음"
    )
    parser.add_argument(
        "--metric", choices=["cpu", "mem", "disk", "all"], default="all",
        help="출력할 지표 (기본값: all)"
    )
    parser.add_argument(
        "--source", choices=["sar", "json"], default="sar",
        help="데이터 소스: sar (7일 한도, 기본값) | json (무한 보존)"
    )
    args = parser.parse_args()

    now = datetime.now()
    print(f"\n{'━' * 62}")
    print(f"  📊 서버 자원 스냅샷  —  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  조회 기간: 최근 {args.hours}시간  |  소스: {args.source.upper()}")
    print(f"{'━' * 62}")

    if args.source == "json":
        # JSON 파일 존재 여부 확인
        if not JSON_DATA_DIR.exists() or not list(JSON_DATA_DIR.glob("*.json")):
            print(f"\n[오류] JSON 데이터 없음: {JSON_DATA_DIR}")
            print("먼저 metrics_collector.py를 실행하거나 cron이 등록되어 있는지 확인하세요.")
            sys.exit(1)

    if args.metric in ("cpu", "all"):
        rows = parse_cpu(args.hours, args.source)
        print(render_chart(
            title="CPU 사용률",
            unit="%",
            rows=rows,
            max_scale=100.0,
            alert_threshold=80.0,
        ))

    if args.metric in ("mem", "all"):
        rows = parse_memory(args.hours, args.source)
        print(render_chart(
            title="메모리 사용률",
            unit="%",
            rows=rows,
            max_scale=100.0,
            alert_threshold=85.0,
        ))

    if args.metric in ("disk", "all"):
        rows = parse_disk_io(args.hours, args.source)
        print(render_chart(
            title="디스크 I/O (tps 합산)",
            unit=" tps",
            rows=rows,
            alert_threshold=None,
        ))

    print()


if __name__ == "__main__":
    main()
