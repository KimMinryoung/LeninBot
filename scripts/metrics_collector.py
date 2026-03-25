#!/usr/bin/env python3
"""
metrics_collector.py — 현재 시점의 자원 스냅샷을 월별 JSON 파일에 누적 저장.

저장 경로: /home/grass/leninbot/data/metrics/YYYY-MM.json
사용법:
    python3 scripts/metrics_collector.py          # 1회 수집 후 저장
    python3 scripts/metrics_collector.py --dry-run # 수집만 하고 저장 안 함 (테스트용)
"""

import json
import os
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path

# ─── 설정 ─────────────────────────────────────────────────────
DATA_DIR = Path("/home/grass/leninbot/data/metrics")
SAMPLE_INTERVAL = 1   # sar 샘플링 간격(초)
SAMPLE_COUNT = 1      # sar 샘플 횟수


# ─── sar 헬퍼 ─────────────────────────────────────────────────
def _sar(args: list[str]) -> list[str]:
    """sar 명령 실행 → stdout 라인 리스트 반환."""
    cmd = ["sar"] + args + [str(SAMPLE_INTERVAL), str(SAMPLE_COUNT)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.splitlines()


def collect_cpu() -> float:
    """CPU 사용률(%) 반환. 100 - %idle."""
    lines = _sar(["-u"])
    for line in lines:
        parts = line.split()
        # "HH:MM:SS AM/PM  all  %user %nice %system %iowait %steal %idle"
        # Average 행 사용 (가장 안정적)
        if parts and parts[0] == "Average:" and len(parts) >= 9 and parts[2] == "all":
            try:
                idle = float(parts[8])
                return round(100.0 - idle, 2)
            except (ValueError, IndexError):
                pass
        # 단일 데이터 행 (Average: 없는 경우 대비)
        if len(parts) >= 9 and parts[2] == "all" and parts[0] != "10:26:57":
            try:
                idle = float(parts[8])
                return round(100.0 - idle, 2)
            except (ValueError, IndexError):
                pass
    return 0.0


def collect_memory() -> dict:
    """메모리 정보 반환: pct, used_gib, total_gib."""
    lines = _sar(["-r"])
    for line in lines:
        parts = line.split()
        # Average 행: "Average: kbmemfree kbavail kbmemused %memused ..."
        if parts and parts[0] == "Average:" and len(parts) >= 6:
            try:
                kb_used = float(parts[3])
                pct = float(parts[4])
                # total = kbmemfree + kbmemused
                kb_free = float(parts[1])
                kb_total = kb_free + kb_used
                return {
                    "mem_pct": round(pct, 2),
                    "mem_used_gib": round(kb_used / 1024 / 1024, 2),
                    "mem_total_gib": round(kb_total / 1024 / 1024, 2),
                }
            except (ValueError, IndexError):
                pass

    # /proc/meminfo 폴백
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for l in f:
                k, v = l.split(":")
                info[k.strip()] = int(v.strip().split()[0])
        total = info["MemTotal"]
        free = info["MemFree"]
        buffers = info.get("Buffers", 0)
        cached = info.get("Cached", 0)
        used = total - free - buffers - cached
        pct = round(used / total * 100, 2)
        return {
            "mem_pct": pct,
            "mem_used_gib": round(used / 1024 / 1024, 2),
            "mem_total_gib": round(total / 1024 / 1024, 2),
        }
    except Exception:
        return {"mem_pct": 0.0, "mem_used_gib": 0.0, "mem_total_gib": 0.0}


def collect_disk_tps() -> float:
    """디스크 전체 tps 합산. loop 장치 제외."""
    lines = _sar(["-d"])
    total_tps = 0.0
    found = False
    for line in lines:
        parts = line.split()
        # Average 행에서 실제 디스크(sda, vda 등)만 합산
        if parts and parts[0] == "Average:" and len(parts) >= 4:
            dev = parts[1]
            if dev in ("DEV",):
                continue
            if dev.startswith("loop"):
                continue
            try:
                tps = float(parts[2])
                total_tps += tps
                found = True
            except (ValueError, IndexError):
                pass
    return round(total_tps, 2) if found else 0.0


def _get_net_interface() -> str:
    """eth0 우선, 없으면 첫 번째 non-lo non-docker non-br 인터페이스."""
    try:
        r = subprocess.run(["ip", "link", "show"], capture_output=True, text=True)
        ifaces = []
        for line in r.stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            # "2: eth0: <...>" 패턴
            if parts[0].endswith(":") and len(parts) > 1:
                name = parts[1].rstrip(":")
                # veth@if 포맷 처리
                name = name.split("@")[0]
                if name not in ("lo",) and not name.startswith("docker") and not name.startswith("br-"):
                    ifaces.append(name)
        if "eth0" in ifaces:
            return "eth0"
        return ifaces[0] if ifaces else "eth0"
    except Exception:
        return "eth0"


def collect_network(iface: str) -> dict:
    """지정 인터페이스의 rx/tx kB/s 반환."""
    lines = _sar(["-n", "DEV"])
    for line in lines:
        parts = line.split()
        # Average 행: "Average: IFACE rxpck/s txpck/s rxkB/s txkB/s ..."
        if parts and parts[0] == "Average:" and len(parts) >= 6 and parts[1] == iface:
            try:
                rx_kbs = float(parts[4])
                tx_kbs = float(parts[5])
                return {
                    "net_rx_kbs": round(rx_kbs, 2),
                    "net_tx_kbs": round(tx_kbs, 2),
                }
            except (ValueError, IndexError):
                pass
    return {"net_rx_kbs": 0.0, "net_tx_kbs": 0.0}


# ─── 스냅샷 수집 ───────────────────────────────────────────────
def collect_snapshot() -> dict:
    """현재 시점 전체 스냅샷 수집."""
    now = datetime.now()
    ts = now.strftime("%Y-%m-%dT%H:%M:%S")

    cpu_pct = collect_cpu()
    mem = collect_memory()
    disk_tps = collect_disk_tps()

    iface = _get_net_interface()
    net = collect_network(iface)

    return {
        "ts": ts,
        "cpu_pct": cpu_pct,
        "mem_pct": mem["mem_pct"],
        "mem_used_gib": mem["mem_used_gib"],
        "mem_total_gib": mem["mem_total_gib"],
        "disk_tps": disk_tps,
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
            # 손상된 파일 대비: 백업 후 새로 시작
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
    parser = argparse.ArgumentParser(description="자원 스냅샷 수집 → JSON 저장")
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
              f"mem={snap['mem_pct']}%  disk={snap['disk_tps']}tps  "
              f"net↓{snap['net_rx_kbs']}kB/s ↑{snap['net_tx_kbs']}kB/s")


if __name__ == "__main__":
    main()
