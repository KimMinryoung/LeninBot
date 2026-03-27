#!/usr/bin/env python3
"""
metrics_collector.py — psutil 기반 자원 스냅샷을 월별 JSON 파일에 누적 저장.

저장 경로: /home/grass/leninbot/data/metrics/YYYY-MM.json
사용법:
    python3 scripts/metrics_collector.py          # 1회 수집 후 저장
    python3 scripts/metrics_collector.py --dry-run # 수집만 하고 저장 안 함 (테스트용)
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

import psutil

# ─── 설정 ─────────────────────────────────────────────────────
DATA_DIR = Path("/home/grass/leninbot/data/metrics")


# ─── 수집 함수들 ───────────────────────────────────────────────

def collect_cpu() -> float:
    """CPU 사용률(%). interval=1초 블로킹 측정."""
    return round(psutil.cpu_percent(interval=1), 2)


def collect_memory() -> dict:
    """메모리 정보: pct, used_gib, total_gib."""
    vm = psutil.virtual_memory()
    return {
        "mem_pct": round(vm.percent, 2),
        "mem_used_gib": round(vm.used / 1024 ** 3, 2),
        "mem_total_gib": round(vm.total / 1024 ** 3, 2),
    }


def collect_disk() -> dict:
    """루트 파티션 디스크 사용 정보: used_gb, total_gb, pct."""
    usage = psutil.disk_usage("/")
    return {
        "disk_used_gb": round(usage.used / 1024 ** 3, 1),
        "disk_total_gb": round(usage.total / 1024 ** 3, 1),
        "disk_pct": round(usage.percent, 2),
    }


def _get_net_interface() -> str:
    """eth0 우선, 없으면 lo·docker·br- 제외 첫 번째 인터페이스."""
    stats = psutil.net_if_stats()
    candidates = [
        name for name, s in stats.items()
        if s.isup and not name.startswith(("lo", "docker", "br-", "veth"))
    ]
    if "eth0" in candidates:
        return "eth0"
    return candidates[0] if candidates else "eth0"


def collect_network(iface: str) -> dict:
    """지정 인터페이스의 rx/tx kB/s (1초 측정)."""
    counters_before = psutil.net_io_counters(pernic=True).get(iface)
    if counters_before is None:
        return {"net_rx_kbs": 0.0, "net_tx_kbs": 0.0}

    import time
    time.sleep(1)
    counters_after = psutil.net_io_counters(pernic=True).get(iface)
    if counters_after is None:
        return {"net_rx_kbs": 0.0, "net_tx_kbs": 0.0}

    rx_kbs = round((counters_after.bytes_recv - counters_before.bytes_recv) / 1024, 2)
    tx_kbs = round((counters_after.bytes_sent - counters_before.bytes_sent) / 1024, 2)
    return {"net_rx_kbs": rx_kbs, "net_tx_kbs": tx_kbs}


# ─── 스냅샷 수집 ───────────────────────────────────────────────
def collect_snapshot() -> dict:
    """현재 시점 전체 스냅샷 수집."""
    now = datetime.now()
    ts = now.strftime("%Y-%m-%dT%H:%M:%S")

    cpu_pct = collect_cpu()
    mem = collect_memory()
    disk = collect_disk()
    iface = _get_net_interface()
    net = collect_network(iface)

    return {
        "ts": ts,
        "cpu_pct": cpu_pct,
        "mem_pct": mem["mem_pct"],
        "mem_used_gib": mem["mem_used_gib"],
        "mem_total_gib": mem["mem_total_gib"],
        "disk_used_gb": disk["disk_used_gb"],
        "disk_total_gb": disk["disk_total_gb"],
        "disk_pct": disk["disk_pct"],
        "net_rx_kbs": net["net_rx_kbs"],
        "net_tx_kbs": net["net_tx_kbs"],
        "net_iface": iface,
    }


# ─── JSON 파일 저장 ────────────────────────────────────────────
def get_json_path(dt: datetime = None) -> Path:
    """월별 JSON 파일 경로 반환."""
    if dt is None:
        dt = datetime.now()
    return DATA_DIR / f"{dt.strftime('%Y-%m')}.json"


def load_json(path: Path) -> dict:
    """파일 읽기. 없으면 빈 구조 반환."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            backup = path.with_suffix(".json.bak")
            path.rename(backup)
            print(f"[경고] JSON 파일 손상, 백업 생성: {backup}", file=sys.stderr)
    return {"snapshots": []}


def save_json(path: Path, data: dict) -> None:
    """JSON 파일 저장 (원자적 쓰기)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.rename(path)


def append_snapshot(snapshot: dict, dry_run: bool = False) -> Path:
    """스냅샷을 월별 JSON에 추가. dry_run=True 면 저장 생략."""
    path = get_json_path()
    data = load_json(path)
    data["snapshots"].append(snapshot)

    if not dry_run:
        save_json(path, data)
        print(f"[저장] {path}  (총 {len(data['snapshots'])}건)")
    else:
        print(f"[dry-run] 저장 생략. 현재 파일 내 건수: {len(data['snapshots'])}")

    return path


# ─── 메인 ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="자원 스냅샷 수집 → JSON 저장 (psutil 기반)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="수집만 하고 저장하지 않음 (테스트용)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="수집 결과 상세 출력"
    )
    args = parser.parse_args()

    snap = collect_snapshot()

    if args.verbose or args.dry_run:
        print(json.dumps(snap, ensure_ascii=False, indent=2))

    path = append_snapshot(snap, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"[완료] ts={snap['ts']}  cpu={snap['cpu_pct']}%  "
              f"mem={snap['mem_pct']}%({snap['mem_used_gib']}/{snap['mem_total_gib']} GiB)  "
              f"disk={snap['disk_pct']}%({snap['disk_used_gb']}/{snap['disk_total_gb']} GB)  "
              f"net↓{snap['net_rx_kbs']}kB/s ↑{snap['net_tx_kbs']}kB/s")


if __name__ == "__main__":
    main()
