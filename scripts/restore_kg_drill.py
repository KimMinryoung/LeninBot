#!/usr/bin/env python3
"""Weekly KG restore drill: newest R2 KG backup -> throwaway Neo4j -> validate.

    venv/bin/python scripts/restore_kg_drill.py --from-r2 --max-age-days 2
    venv/bin/python scripts/restore_kg_drill.py --archive data/kg_backups/kg-backup-2026-10-11.tar.gz

Runs on leninbot-standby after the PostgreSQL drill (same unit). Neo4j lives
only on the main VM, so this is the one place a KG backup is proven to load
away from production. The container binds a random 127.0.0.1 port and is
removed afterwards; production Neo4j is never contacted.

Validation: restore_kg.py's distinct-key counts, then label distribution,
extra-property coverage and graphiti's non-null RELATES_TO.episodes, all
compared with the backup JSON. Embeddings are not in the backup (text only
since 2026-09-03), so vector search is out of scope.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "skills" / "kg-maintenance" / "scripts"))

from _r2_backup import R2_CREDENTIAL_PAIRS, promote_systemd_credentials

promote_systemd_credentials(R2_CREDENTIAL_PAIRS)

BUCKET = "cyber-lenin-backups"
KST = timezone(timedelta(hours=9))
# Same digest as leninbot-neo4j on the main VM (neo4j:5-community, 2026-10-10).
DRILL_IMAGE = "neo4j@sha256:40bf5ae9282213087e4d6036aab3ec443fe9c974d3dd4f14a11892c63157238f"
KEY_RE = re.compile(r"^kg-backup-(\d{4}-\d{2}-\d{2})(?:_(\d{6}))?\.tar\.gz$")


class DrillError(RuntimeError):
    pass


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def _fetch_latest_from_r2(dest_dir: Path, max_age_days: int | None) -> Path:
    from _r2_backup import r2_get, r2_list_keys

    try:
        keys = r2_list_keys(BUCKET, "kg-backup")
    except Exception as exc:
        raise DrillError(f"R2 listing failed: {exc}") from exc
    candidates = sorted(
        (m.group(1), m.group(2) or "", key) for key in keys if (m := KEY_RE.match(key))
    )
    if not candidates:
        raise DrillError("no KG backup in R2")
    day, _, key = candidates[-1]
    if max_age_days is not None:
        age = (datetime.now(KST).date() - datetime.strptime(day, "%Y-%m-%d").date()).days
        if age > max_age_days:
            raise DrillError(
                f"newest KG backup in R2 is {key} ({age} days old, limit {max_age_days}); "
                "KG backups have stopped"
            )
    path = dest_dir / key
    print(f"+ R2 download {BUCKET}/{key}", flush=True)
    try:
        r2_get(BUCKET, key, str(path))
    except Exception as exc:
        raise DrillError(f"R2 download failed for {key}: {exc}") from exc
    return path


def _extract(archive: Path, dest_dir: Path) -> dict[str, Path]:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest_dir, filter="data")
    found = {}
    for kind in ("entities", "edges", "mentions"):
        matches = sorted(dest_dir.glob(f"{kind}_*.json"))
        if len(matches) != 1:
            raise DrillError(f"{archive.name}: expected one {kind}_*.json, found {len(matches)}")
        found[kind] = matches[0]
    return found


def _start_container(name: str, image: str, password: str, heap: str, pagecache: str) -> str:
    _run([
        "docker", "run", "-d", "--name", name,
        # Cap the container so a runaway restore cannot starve the PG standby.
        "--memory", "1600m",
        "-p", "127.0.0.1::7687",
        "-e", f"NEO4J_AUTH=neo4j/{password}",
        "-e", f"NEO4J_server_memory_heap_initial__size={heap}",
        "-e", f"NEO4J_server_memory_heap_max__size={heap}",
        "-e", f"NEO4J_server_memory_pagecache_size={pagecache}",
        image,
    ])
    port = _run(["docker", "port", name, "7687/tcp"]).stdout.strip().splitlines()[0]
    return "bolt://" + port


def _wait_for_bolt(uri: str, password: str, timeout_seconds: int = 180) -> None:
    from neo4j import GraphDatabase

    deadline = time.monotonic() + timeout_seconds
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with GraphDatabase.driver(uri, auth=("neo4j", password)) as driver:
                driver.verify_connectivity()
            return
        except Exception as exc:  # noqa: BLE001 - startup races raise several types
            last = exc
            time.sleep(3)
    raise DrillError(f"Neo4j did not accept bolt within {timeout_seconds}s: {last}")


def _validate(uri: str, password: str, files: dict[str, Path]) -> None:
    from neo4j import GraphDatabase

    entities = json.loads(files["entities"].read_text(encoding="utf-8"))
    edges = json.loads(files["edges"].read_text(encoding="utf-8"))

    want_labels = Counter(
        ":".join(sorted(e.get("labels") or [])) for e in {e["uuid"]: e for e in entities}.values()
    )
    want_props = sum(1 for e in entities if e.get("props"))
    want_null_episodes = sum(1 for r in edges if r.get("episodes") is None)

    with GraphDatabase.driver(uri, auth=("neo4j", password)) as driver, driver.session() as s:
        got_labels = Counter({
            ":".join(sorted(r["labels"])): r["c"]
            for r in s.run("MATCH (n:Entity) RETURN labels(n) AS labels, count(*) AS c")
        })
        # backup_kg.py puts every property outside the fixed columns into props.
        got_props = s.run(
            "MATCH (n:Entity) WHERE size([k IN keys(n) WHERE NOT k IN "
            "['uuid','name','summary','group_id','created_at','name_embedding']]) > 0 "
            "RETURN count(n) AS c"
        ).single()["c"]
        got_null_episodes = s.run(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.episodes IS NULL RETURN count(r) AS c"
        ).single()["c"]

    problems = []
    if got_labels != want_labels:
        diff = {k: (want_labels.get(k, 0), got_labels.get(k, 0))
                for k in want_labels.keys() | got_labels.keys()
                if want_labels.get(k, 0) != got_labels.get(k, 0)}
        problems.append(f"label distribution differs (backup, restored): {diff}")
    if got_props != want_props:
        problems.append(f"entities with extra props: backup {want_props}, restored {got_props}")
    if got_null_episodes != want_null_episodes:
        problems.append(
            f"RELATES_TO with NULL episodes: backup {want_null_episodes}, restored {got_null_episodes}"
        )
    if problems:
        raise DrillError("; ".join(problems))
    print(
        f"PASS kg: label_sets={len(got_labels)}, entities_with_props={got_props}, "
        f"null_episodes={got_null_episodes}",
        flush=True,
    )


def _drill(args: argparse.Namespace, work_dir: Path) -> None:
    import restore_kg

    archive = Path(args.archive) if args.archive else _fetch_latest_from_r2(work_dir, args.max_age_days)
    files = _extract(archive, work_dir / "extract")

    password = secrets.token_urlsafe(24)
    name = args.container_name
    if _run(["docker", "container", "inspect", name], check=False).returncode == 0:
        raise DrillError(f"drill container already exists: {name}")
    created = False
    try:
        uri = _start_container(name, args.image, password, args.heap, args.pagecache)
        created = True
        _wait_for_bolt(uri, password)
        # Password through the environment, not argv, so it stays out of ps.
        os.environ["RESTORE_NEO4J_PASSWORD"] = password
        os.environ.pop("NEO4J_URI", None)
        rc = restore_kg.main([
            "--entities", str(files["entities"]),
            "--edges", str(files["edges"]),
            "--mentions", str(files["mentions"]),
            "--target-uri", uri,
            "--clear",
        ])
        if rc != 0:
            raise DrillError(f"restore_kg.py exited {rc}")
        _validate(uri, password, files)
    finally:
        if created and not args.keep_container:
            _run(["docker", "rm", "-f", name], check=False)
        elif created:
            print(f"Kept drill container: {name}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-r2", action="store_true", help="download the newest KG backup from R2")
    src.add_argument("--archive", help="local kg-backup-*.tar.gz instead of R2")
    p.add_argument("--max-age-days", type=int, help="with --from-r2, fail when the newest backup is older")
    p.add_argument("--container-name", default=f"leninbot-kg-restore-drill-{os.getpid()}")
    p.add_argument("--image", default=DRILL_IMAGE)
    p.add_argument("--heap", default="768m")
    p.add_argument("--pagecache", default="256m")
    p.add_argument("--keep-container", action="store_true")
    args = p.parse_args(argv)
    # systemd's TimeoutStartSec sends SIGTERM; turn it into SystemExit so the
    # finally blocks still remove the container and the extracted JSON.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    # /var/tmp, not /tmp: the extracted JSON is ~75 MB and /tmp may be tmpfs.
    with tempfile.TemporaryDirectory(prefix="leninbot-kg-drill-", dir="/var/tmp") as work:
        try:
            _drill(args, Path(work))
        except DrillError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    print("KG DRILL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
