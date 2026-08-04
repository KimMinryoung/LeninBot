#!/usr/bin/env bash
# run_smokes.sh — run every scripts/smoke_*.py and report one line each.
#
# There was no runner. Each suite was only ever run by whoever had just touched
# the thing it covers, so a suite nobody touched could fail indefinitely without
# anyone seeing it: on 2026-08-02 smoke_llm_routing_search had been asserting a
# routing contract deleted three weeks earlier, and smoke_static_pages had been
# dying at import since April, when static-page rendering moved to the frontend
# repo. Both were found only because the whole set was run at once for the first
# time.
#
# Usage:
#   scripts/run_smokes.sh              # every suite
#   scripts/run_smokes.sh commulingo   # only suites whose name contains this
#
# Secrets: these suites construct real clients, so they need the API keys and
# DB password, which live in the systemd credstore rather than .env (decrypting
# from credstore needs root). If CREDENTIALS_DIRECTORY is not already set, this
# borrows a running service's credential mount, which is group-readable.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

FILTER="${1:-}"
TIMEOUT="${SMOKE_TIMEOUT:-300}"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1090
    source <(grep -vE '^\s*#|^\s*$' .env)
    set +a
fi

if [ -z "${CREDENTIALS_DIRECTORY:-}" ]; then
    for svc in leninbot-api leninbot-telegram leninbot-roleplay novel-writer-api; do
        if [ -r "/run/credentials/${svc}.service/db_password" ]; then
            export CREDENTIALS_DIRECTORY="/run/credentials/${svc}.service"
            break
        fi
    done
fi
if [ -z "${CREDENTIALS_DIRECTORY:-}" ]; then
    echo "warning: no readable credential mount found — suites needing secrets will fail." >&2
    echo "         start a leninbot service, or set CREDENTIALS_DIRECTORY yourself." >&2
fi
export PYTHONUNBUFFERED=1

PY="$ROOT/venv/bin/python"
[ -x "$PY" ] || PY=python3

pass=0
fail=0
failed_names=""

# Hermetic unit suite first (tests/, no secrets/DB needed, ~0.01s). A unit
# failure is reported like any other suite but doesn't block the smokes —
# they may localize the breakage further.
if [ -z "$FILTER" ] || case unit_tests in *"$FILTER"*) true ;; *) false ;; esac; then
    unit_out="$(timeout 60 "$PY" -m unittest discover tests 2>&1)"
    if [ $? -eq 0 ]; then
        printf 'PASS  unit_tests (%s)\n' "$(printf '%s' "$unit_out" | grep -oE 'Ran [0-9]+ tests' | head -1)"
        pass=$((pass + 1))
    else
        printf 'FAIL  unit_tests\n'
        printf '%s\n' "$unit_out" | grep -vE '^\s*$' | tail -6 | sed 's/^/        /'
        fail=$((fail + 1))
        failed_names="$failed_names unit_tests"
    fi
fi
for f in scripts/smoke_*.py; do
    name="$(basename "$f" .py)"
    case "$name" in
        *"$FILTER"*) ;;
        *) continue ;;
    esac
    out="$(timeout "$TIMEOUT" "$PY" "$f" 2>&1)"
    rc=$?
    if [ $rc -eq 0 ]; then
        printf 'PASS  %s\n' "$name"
        pass=$((pass + 1))
    else
        if [ $rc -eq 124 ]; then
            printf 'FAIL  %s  (timed out after %ss)\n' "$name" "$TIMEOUT"
        else
            printf 'FAIL  %s  (rc=%d)\n' "$name" "$rc"
        fi
        printf '%s\n' "$out" | grep -vE '^\s*$' | tail -4 | sed 's/^/        /'
        fail=$((fail + 1))
        failed_names="$failed_names $name"
    fi
done

printf '\n== %d passed, %d failed ==\n' "$pass" "$fail"
if [ $fail -gt 0 ]; then
    printf 'failed:%s\n' "$failed_names"
    exit 1
fi
if [ $pass -eq 0 ]; then
    printf 'no suite matched %s\n' "${FILTER:-(no filter)}"
    exit 1
fi
