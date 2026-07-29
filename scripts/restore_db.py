#!/usr/bin/env python3
"""Restore or drill LeninBot PostgreSQL backups.

Safe isolated drill:
    venv/bin/python scripts/restore_db.py drill

Actual restore into an already-running PostgreSQL 17/pgvector container:
    venv/bin/python scripts/restore_db.py restore \
        --target-container leninbot-pg-recovery \
        --confirm RECREATE_DATABASES

The restore mode recreates databases. The production container has a separate,
stronger confirmation gate, and any remaining DB client connection blocks the
operation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from secrets_loader import get_secret

PRODUCTION_CONTAINER = "leninbot-pg"
DRILL_IMAGE = "pgvector/pgvector:pg17"
NORMAL_CONFIRMATION = "RECREATE_DATABASES"
PRODUCTION_CONFIRMATION = "RECREATE_LENINBOT_PRODUCTION"


class RestoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackupSpec:
    scope: str
    database: str
    restore_user: str
    pattern: str
    default_dir: Path
    toc_marker: str
    jobs: int


SPECS = {
    "main": BackupSpec(
        "main",
        "leninbot",
        "postgres",
        "main-db-backup-*.dump",
        ROOT / "data" / "main_db_backups",
        "lenin_corpus",
        4,
    ),
    "legacy": BackupSpec(
        "legacy",
        "legacy_game",
        "postgres",
        "legacy-game-db-backup-*.dump",
        ROOT / "data" / "legacy_game_db_backups",
        "story_scenes",
        1,
    ),
    "writer": BackupSpec(
        "writer",
        "writer",
        "writer",
        "writer-db-backup-*.dump",
        ROOT / "data" / "writer_db_backups",
        "writer_manuscripts",
        2,
    ),
}


def _run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    input_text: str | None = None,
    timeout: int = 900,
) -> subprocess.CompletedProcess:
    print(f"+ {shlex.join(args)}", flush=True)
    try:
        return subprocess.run(
            args,
            check=check,
            capture_output=capture_output,
            text=True,
            input=input_text,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        detail = f": {exc.stderr.strip()}" if exc.stderr else ""
        raise RestoreError(f"command failed ({exc.returncode}){detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RestoreError(f"command timed out after {timeout}s: {shlex.join(args)}") from exc


def _resolve_backup(explicit: str | None, spec: BackupSpec) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RestoreError(f"{spec.scope} backup does not exist: {path}")
        return path
    candidates = sorted(spec.default_dir.glob(spec.pattern))
    if not candidates:
        raise RestoreError(
            f"no {spec.scope} backup under {spec.default_dir}; pass --{spec.scope}-backup"
        )
    return candidates[-1].resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _container_exists(name: str) -> bool:
    return _run(["docker", "inspect", name], check=False, capture_output=True).returncode == 0


def _container_running(name: str) -> bool:
    result = _run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _wait_for_postgres(container: str, timeout_seconds: int = 60) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = _run(
            ["docker", "exec", container, "pg_isready", "-U", "postgres", "-d", "postgres"],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    logs = _run(
        ["docker", "logs", "--tail", "100", container],
        check=False,
        capture_output=True,
    )
    raise RestoreError(f"Postgres did not become ready: {logs.stderr or logs.stdout}")


def _create_drill_container(name: str, image: str) -> None:
    _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--shm-size",
            "1g",
            "-e",
            "POSTGRES_PASSWORD=drill_ephemeral_only",
            "-e",
            "POSTGRES_DB=postgres",
            image,
            "postgres",
            "-c",
            "shared_preload_libraries=pg_stat_statements",
            "-c",
            "maintenance_work_mem=512MB",
            "-c",
            "max_wal_size=2GB",
        ]
    )


def _psql(
    container: str,
    database: str,
    sql: str,
    *,
    user: str = "postgres",
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    return _run(
        [
            "docker",
            "exec",
            "-i",
            container,
            "psql",
            "-X",
            "-q",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            user,
            "-d",
            database,
            "-tA",
        ],
        input_text=sql,
        capture_output=capture_output,
    )


def _scalar(container: str, database: str, sql: str, *, user: str = "postgres") -> str:
    result = _psql(container, database, sql, user=user, capture_output=True)
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not values:
        raise RestoreError(f"query returned no value for {database}")
    return values[-1]


def _verify_target(container: str) -> None:
    if not _container_running(container):
        raise RestoreError(f"target container is not running: {container}")
    _wait_for_postgres(container)
    major = int(_scalar(container, "postgres", "SHOW server_version_num;")) // 10000
    if major != 17:
        raise RestoreError(f"PostgreSQL 17 required, target reports major version {major}")
    shm_result = _run(
        ["docker", "inspect", "--format", "{{.HostConfig.ShmSize}}", container],
        capture_output=True,
    )
    shm_size = int(shm_result.stdout.strip())
    if shm_size < 1024 * 1024 * 1024:
        raise RestoreError(
            f"target /dev/shm must be at least 1GiB for HNSW restore; "
            f"container has {shm_size} bytes"
        )


def _verify_archive(container: str, path: Path, marker: str) -> None:
    print(f"Verifying {path} (sha256={_sha256(path)})", flush=True)
    with path.open("rb") as source:
        print(f"+ docker exec -i {container} pg_restore --list", flush=True)
        try:
            result = subprocess.run(
                ["docker", "exec", "-i", container, "pg_restore", "--list"],
                stdin=source,
                capture_output=True,
                check=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as exc:
            raise RestoreError(
                f"archive TOC verification failed: {exc.stderr.decode(errors='replace').strip()}"
            ) from exc
    if marker not in result.stdout.decode(errors="replace"):
        raise RestoreError(f"{marker} missing from archive TOC: {path}")


def _active_connections(container: str, database: str) -> int:
    return int(
        _scalar(
            container,
            "postgres",
            "SELECT count(*) FROM pg_stat_activity "
            f"WHERE datname='{database}' AND pid <> pg_backend_pid();",
        )
    )


def _load_writer_password(path_arg: str | None, *, drill: bool) -> str:
    if path_arg:
        path = Path(path_arg).expanduser().resolve()
        if not path.is_file():
            raise RestoreError(f"writer password file does not exist: {path}")
        if path.stat().st_mode & 0o077:
            raise RestoreError(f"writer password file must not be group/world accessible: {path}")
        password = path.read_text(encoding="utf-8").rstrip("\n")
    else:
        password = get_secret("WRITER_DB_PASSWORD", "") or ""
    if not password and drill:
        password = "drill_writer_ephemeral_only"
    if not password:
        raise RestoreError(
            "writer restore requires WRITER_DB_PASSWORD or --writer-password-file"
        )
    if any(char in password for char in ("\x00", "\r", "\n")):
        raise RestoreError("writer password must be a single non-NUL line")
    return password


def _load_frontend_password(path_arg: str | None, *, drill: bool) -> str | None:
    if path_arg:
        path = Path(path_arg).expanduser().resolve()
        if not path.is_file():
            raise RestoreError(f"frontend password file does not exist: {path}")
        if path.stat().st_mode & 0o077:
            raise RestoreError(
                f"frontend password file must not be group/world accessible: {path}"
            )
        password = path.read_text(encoding="utf-8").rstrip("\n")
    else:
        password = get_secret("FRONTEND_DB_PASSWORD", "") or ""
    if not password and drill:
        password = "drill_frontend_ephemeral_only"
    if password and any(char in password for char in ("\x00", "\r", "\n")):
        raise RestoreError("frontend password must be a single non-NUL line")
    return password or None


def _ensure_frontend_role(container: str, password: str | None) -> None:
    exists = _scalar(
        container,
        "postgres",
        "SELECT count(*) FROM pg_roles WHERE rolname='frontend';",
    )
    if exists == "0" and not password:
        raise RestoreError(
            "fresh main restore requires FRONTEND_DB_PASSWORD or --frontend-password-file"
        )
    if not password:
        return
    escaped_password = password.replace("'", "''")
    if exists == "0":
        role_sql = f"CREATE ROLE frontend LOGIN PASSWORD '{escaped_password}';"
    else:
        role_sql = f"ALTER ROLE frontend LOGIN PASSWORD '{escaped_password}';"
    _psql(container, "postgres", role_sql)


def _recreate_database(
    container: str,
    spec: BackupSpec,
    writer_password: str | None,
) -> None:
    connections = _active_connections(container, spec.database)
    if connections:
        raise RestoreError(
            f"refusing to recreate {spec.database}: {connections} connection(s) remain; "
            "stop DB clients first"
        )
    if spec.scope == "writer":
        if not writer_password:
            raise RestoreError("writer password was not loaded")
        role_exists = _scalar(
            container,
            "postgres",
            "SELECT count(*) FROM pg_roles WHERE rolname='writer';",
        )
        escaped_password = writer_password.replace("'", "''")
        if role_exists == "0":
            role_sql = f"CREATE ROLE writer LOGIN PASSWORD '{escaped_password}';"
        else:
            role_sql = f"ALTER ROLE writer LOGIN PASSWORD '{escaped_password}';"
        _psql(container, "postgres", role_sql)
    _psql(container, "postgres", f"DROP DATABASE IF EXISTS {spec.database};")
    owner = "writer" if spec.scope == "writer" else "postgres"
    _psql(container, "postgres", f"CREATE DATABASE {spec.database} OWNER {owner};")


def _grant_frontend_access(container: str) -> None:
    _psql(
        container,
        "leninbot",
        """
        GRANT CONNECT ON DATABASE leninbot TO frontend;
        GRANT USAGE ON SCHEMA public TO frontend;
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO frontend;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO frontend;
        ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO frontend;
        ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL PRIVILEGES ON SEQUENCES TO frontend;
        """,
    )


def _stage_archive(container: str, path: Path, scope: str) -> str:
    target = f"/tmp/leninbot-{scope}-restore-{os.getpid()}.dump"
    _run(["docker", "cp", str(path), f"{container}:{target}"])
    return target


def _restore_archive(container: str, spec: BackupSpec, staged: str) -> float:
    started = time.monotonic()
    _run(
        [
            "docker",
            "exec",
            container,
            "pg_restore",
            "-U",
            spec.restore_user,
            "-d",
            spec.database,
            "--no-owner",
            "--no-privileges",
            "--exit-on-error",
            f"--jobs={spec.jobs}",
            staged,
        ]
    )
    return time.monotonic() - started


COUNT_AND_SEQUENCE_SQL = r"""
CREATE TEMP TABLE dri_counts(table_name text PRIMARY KEY, row_count bigint NOT NULL);
DO $$
DECLARE r record;
BEGIN
  FOR r IN SELECT tablename FROM pg_tables WHERE schemaname='public' LOOP
    EXECUTE format(
      'INSERT INTO dri_counts SELECT %L, count(*) FROM public.%I',
      r.tablename, r.tablename
    );
  END LOOP;
END $$;
CREATE TEMP TABLE dri_sequences(last_value bigint, max_value bigint);
DO $$
DECLARE r record; lv bigint; mv bigint;
BEGIN
  FOR r IN
    SELECT table_name, column_name,
           pg_get_serial_sequence(format('%I.%I', table_schema, table_name), column_name) seq_name
    FROM information_schema.columns
    WHERE table_schema='public' AND column_default LIKE 'nextval(%'
  LOOP
    IF r.seq_name IS NOT NULL THEN
      EXECUTE format('SELECT last_value FROM %s', r.seq_name) INTO lv;
      EXECUTE format('SELECT max(%I) FROM public.%I', r.column_name, r.table_name) INTO mv;
      INSERT INTO dri_sequences VALUES (lv, mv);
    END IF;
  END LOOP;
END $$;
SELECT concat_ws(
  '|',
  (SELECT count(*) FROM dri_counts),
  (SELECT coalesce(sum(row_count), 0) FROM dri_counts),
  (SELECT count(*) FROM dri_sequences),
  (SELECT count(*) FROM dri_sequences WHERE max_value IS NOT NULL AND last_value < max_value)
);
"""


def _common_validation(container: str, database: str, user: str) -> tuple[int, int, int]:
    summary = _scalar(container, database, COUNT_AND_SEQUENCE_SQL, user=user)
    tables, rows, sequences, behind = map(int, summary.split("|"))
    if behind:
        raise RestoreError(f"{database}: {behind} sequence(s) are behind table data")
    invalid = int(
        _scalar(
            container,
            database,
            """
            SELECT count(*)
            FROM pg_index i
            JOIN pg_class c ON c.oid=i.indexrelid
            JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname='public' AND (NOT i.indisvalid OR NOT i.indisready);
            """,
            user=user,
        )
    )
    if invalid:
        raise RestoreError(f"{database}: {invalid} invalid or unready index(es)")
    return tables, rows, sequences


def _validate_main(container: str) -> None:
    tables, rows, sequences = _common_validation(container, "leninbot", "postgres")
    corpus = int(_scalar(container, "leninbot", "SELECT count(*) FROM public.lenin_corpus;"))
    hnsw = int(
        _scalar(
            container,
            "leninbot",
            """
            SELECT count(*)
            FROM pg_index i
            JOIN pg_class idx ON idx.oid=i.indexrelid
            JOIN pg_am am ON am.oid=idx.relam
            WHERE i.indrelid='public.lenin_corpus'::regclass
              AND am.amname='hnsw' AND i.indisvalid AND i.indisready;
            """,
        )
    )
    plan = _psql(
        container,
        "leninbot",
        """
        SET enable_seqscan=off;
        EXPLAIN (COSTS OFF)
        SELECT id FROM public.lenin_corpus
        ORDER BY embedding <=> (
          SELECT embedding FROM public.lenin_corpus WHERE embedding IS NOT NULL LIMIT 1
        )
        LIMIT 5;
        """,
        capture_output=True,
    ).stdout
    vector_rows = int(
        _scalar(
            container,
            "leninbot",
            """
            SELECT count(*) FROM (
              SELECT id FROM public.lenin_corpus
              ORDER BY embedding <=> (
                SELECT embedding FROM public.lenin_corpus WHERE embedding IS NOT NULL LIMIT 1
              )
              LIMIT 5
            ) q;
            """,
        )
    )
    if corpus <= 0 or hnsw < 1 or "idx_lenin_corpus_embedding" not in plan or vector_rows != 5:
        raise RestoreError(
            f"main vector validation failed: corpus={corpus}, hnsw={hnsw}, rows={vector_rows}"
        )
    not_valid = int(
        _scalar(
            container,
            "leninbot",
            "SELECT count(*) FROM pg_constraint "
            "WHERE connamespace='public'::regnamespace AND NOT convalidated;",
        )
    )
    frontend_tables = int(
        _scalar(
            container,
            "leninbot",
            """
            SELECT count(DISTINCT table_name)
            FROM information_schema.table_privileges
            WHERE table_schema='public' AND grantee='frontend'
              AND privilege_type='SELECT';
            """,
        )
    )
    frontend_sequences = int(
        _scalar(
            container,
            "leninbot",
            """
            SELECT count(DISTINCT object_name)
            FROM information_schema.usage_privileges
            WHERE object_schema='public' AND grantee='frontend'
              AND object_type='SEQUENCE' AND privilege_type='USAGE';
            """,
        )
    )
    frontend_password = int(
        _scalar(
            container,
            "postgres",
            "SELECT count(*) FROM pg_authid "
            "WHERE rolname='frontend' AND rolpassword IS NOT NULL;",
        )
    )
    frontend_corpus = int(
        _scalar(
            container,
            "leninbot",
            "SELECT count(*) FROM public.lenin_corpus;",
            user="frontend",
        )
    )
    if (
        frontend_tables != tables
        or frontend_sequences != sequences
        or frontend_password != 1
        or frontend_corpus != corpus
    ):
        raise RestoreError(
            f"frontend validation failed: tables={frontend_tables}/{tables}, "
            f"sequences={frontend_sequences}/{sequences}, corpus={frontend_corpus}/{corpus}"
        )
    print(
        f"PASS main: tables={tables}, rows={rows}, corpus={corpus}, "
        f"sequences={sequences}, hnsw={hnsw}, not_valid_constraints={not_valid}",
        flush=True,
    )


def _validate_writer(container: str) -> None:
    tables, rows, sequences = _common_validation(container, "writer", "writer")
    manuscripts = _scalar(
        container,
        "writer",
        """
        SELECT concat_ws(
          '|',
          count(*) FILTER (WHERE body IS NOT NULL AND length(body) > 0),
          coalesce(sum(length(body)), 0)
        )
        FROM public.writer_manuscripts;
        """,
        user="writer",
    )
    nonempty, chars = map(int, manuscripts.split("|"))
    wrong_owner = int(
        _scalar(
            container,
            "writer",
            """
            SELECT count(*)
            FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname='public' AND c.relkind IN ('r', 'p', 'S')
              AND pg_get_userbyid(c.relowner) <> 'writer';
            """,
            user="writer",
        )
    )
    if nonempty <= 0 or chars <= 0 or wrong_owner:
        raise RestoreError(
            f"writer validation failed: nonempty={nonempty}, chars={chars}, "
            f"wrong_owner={wrong_owner}"
        )
    print(
        f"PASS writer: tables={tables}, rows={rows}, sequences={sequences}, "
        f"nonempty_manuscripts={nonempty}, manuscript_chars={chars}",
        flush=True,
    )


def _validate_legacy(container: str) -> None:
    tables, rows, sequences = _common_validation(
        container, "legacy_game", "postgres"
    )
    summary = _scalar(
        container,
        "legacy_game",
        """
        SELECT concat_ws(
          '|',
          count(*),
          md5(string_agg(md5(row_to_json(s)::text), '' ORDER BY id)),
          min(id),
          max(id)
        )
        FROM public.story_scenes s;
        """,
    )
    expected = "415|1b3bdc9d7dac48fafc1e216ffd1066f0|1|831"
    if tables != 1 or rows != 415 or sequences != 1 or summary != expected:
        raise RestoreError(
            f"legacy validation failed: tables={tables}, rows={rows}, "
            f"sequences={sequences}, summary={summary}"
        )
    print(
        f"PASS legacy: tables={tables}, rows={rows}, sequences={sequences}, "
        f"story_scenes_digest={summary.split('|')[1]}",
        flush=True,
    )


def _check_restore_confirmation(args: argparse.Namespace) -> None:
    if args.target_container == PRODUCTION_CONTAINER:
        if not args.force_production or args.confirm != PRODUCTION_CONFIRMATION:
            raise RestoreError(
                "production restore requires --force-production and "
                f"--confirm {PRODUCTION_CONFIRMATION}"
            )
    elif args.confirm != NORMAL_CONFIRMATION:
        raise RestoreError(f"restore requires --confirm {NORMAL_CONFIRMATION}")


def _restore_selected(
    container: str,
    specs: list[BackupSpec],
    backups: dict[str, Path],
    writer_password: str | None,
    frontend_password: str | None,
) -> None:
    staged: dict[str, str] = {}
    try:
        # Validate and stage every selected archive before recreating any DB.
        for spec in specs:
            path = backups[spec.scope]
            _verify_archive(container, path, spec.toc_marker)
            staged[spec.scope] = _stage_archive(container, path, spec.scope)
        if any(spec.scope == "main" for spec in specs):
            _ensure_frontend_role(container, frontend_password)
        for spec in specs:
            _recreate_database(container, spec, writer_password)
            elapsed = _restore_archive(container, spec, staged[spec.scope])
            print(f"Restored {spec.database} in {elapsed:.1f}s", flush=True)
            if spec.scope == "main":
                _grant_frontend_access(container)
            validators = {
                "main": _validate_main,
                "legacy": _validate_legacy,
                "writer": _validate_writer,
            }
            validators[spec.scope](container)
            if spec.scope == "legacy":
                _psql(
                    container,
                    "postgres",
                    "ALTER DATABASE legacy_game SET default_transaction_read_only=on;",
                )
                read_only = _scalar(
                    container, "legacy_game", "SHOW default_transaction_read_only;"
                )
                if read_only != "on":
                    raise RestoreError(
                        "legacy_game read-only setting was not restored"
                    )
                print("PASS legacy read-only: on", flush=True)
    finally:
        if _container_running(container):
            for path in staged.values():
                _run(
                    ["docker", "exec", container, "rm", "-f", path],
                    check=False,
                    capture_output=True,
                )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_backup_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--scope", choices=("all", "main", "legacy", "writer"), default="all"
        )
        subparser.add_argument("--main-backup")
        subparser.add_argument("--legacy-backup")
        subparser.add_argument("--writer-backup")
        subparser.add_argument("--writer-password-file")
        subparser.add_argument("--frontend-password-file")

    drill = subparsers.add_parser("drill", help="restore into a disposable container")
    add_backup_args(drill)
    drill.add_argument("--container-name", default=f"leninbot-pg-restore-drill-{os.getpid()}")
    drill.add_argument("--image", default=DRILL_IMAGE)
    drill.add_argument("--keep-container", action="store_true")

    restore = subparsers.add_parser("restore", help="recreate DBs in an existing container")
    add_backup_args(restore)
    restore.add_argument("--target-container", required=True)
    restore.add_argument("--confirm")
    restore.add_argument("--force-production", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "restore":
            _check_restore_confirmation(args)

        specs = (
            [SPECS["main"], SPECS["legacy"], SPECS["writer"]]
            if args.scope == "all"
            else [SPECS[args.scope]]
        )
        writer_selected = any(spec.scope == "writer" for spec in specs)
        writer_password = (
            _load_writer_password(
                args.writer_password_file,
                drill=args.command == "drill",
            )
            if writer_selected
            else None
        )
        main_selected = any(spec.scope == "main" for spec in specs)
        frontend_password = (
            _load_frontend_password(
                args.frontend_password_file,
                drill=args.command == "drill",
            )
            if main_selected
            else None
        )


        backups = {
            spec.scope: _resolve_backup(getattr(args, f"{spec.scope}_backup"), spec)
            for spec in specs
        }

        if args.command == "restore":
            _verify_target(args.target_container)
            _restore_selected(args.target_container, specs, backups, writer_password, frontend_password)
            print("RESTORE PASS", flush=True)
            return 0

        container = args.container_name
        if _container_exists(container):
            raise RestoreError(f"drill container already exists: {container}")
        created = False
        try:
            _create_drill_container(container, args.image)
            created = True
            _wait_for_postgres(container)
            _restore_selected(container, specs, backups, writer_password, frontend_password)
            print("DRILL PASS", flush=True)
            return 0
        finally:
            if created and not args.keep_container:
                _run(["docker", "rm", "-f", container], check=False, capture_output=True)
            elif created:
                print(f"Kept drill container: {container}", flush=True)
    except RestoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
