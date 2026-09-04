#!/usr/bin/env bash
# finish_alias_repair.sh — the two steps of the 2026-09-04 alias-pollution
# repair that need an operator shell: delete every document-derived edge and
# rebuild the document layer under the trusted-source rules, then restart the
# services so recall/writer pick up the new code.
#
#   skills/kg-maintenance/scripts/finish_alias_repair.sh            # run everything
#   skills/kg-maintenance/scripts/finish_alias_repair.sh --dry-run  # report only, no writes
#   skills/kg-maintenance/scripts/finish_alias_repair.sh --no-restart
#
# Steps
#   1. repair_alias_pollution.py --execute  (backup → strip aliases (idempotent)
#      → delete r.doc_ref edges + orphaned `documents` nodes → fact-text fixes)
#   2. jobs.kg_sync --source documents --full --force  (LLM re-extraction of
#      every document; ~30–60 min; log kept under data/kg_backups/)
#   3. systemctl restart leninbot-api leninbot-telegram
#   4. verify: alias index hits for 소련/미국, recall block for a Soviet question
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"
cd "$ROOT"
DRY=0; RESTART=1
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --no-restart) RESTART=0 ;;
        *) echo "unknown arg: $a" >&2; exit 2 ;;
    esac
done

set -a; source <(grep -vE '^\s*#|^\s*$' .env); set +a
export CREDENTIALS_DIRECTORY="${CREDENTIALS_DIRECTORY:-/run/credentials/leninbot-api.service}"
PY="$ROOT/venv/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/data/kg_backups/finish_alias_repair_${TS}.log"
mkdir -p "$ROOT/data/kg_backups"
exec > >(tee -a "$LOG") 2>&1
echo "== finish_alias_repair $(date '+%F %T') dry_run=$DRY restart=$RESTART  log=$LOG"

echo; echo "## 1. repair_alias_pollution"
if [ "$DRY" = 1 ]; then
    "$PY" skills/kg-maintenance/scripts/repair_alias_pollution.py
else
    "$PY" skills/kg-maintenance/scripts/repair_alias_pollution.py --execute
fi

echo; echo "## 2. documents re-sync (--full --force)"
if [ "$DRY" = 1 ]; then
    LENINBOT_ALLOW_WRITE=1 "$PY" -m jobs.kg_sync --source documents --full --dry-run --json | head -c 1500; echo
else
    START=$(date +%s)
    LENINBOT_ALLOW_WRITE=1 "$PY" -m jobs.kg_sync --source documents --full --force --json \
        | "$PY" -c '
import json, sys
d = json.load(sys.stdin); d = d.get("documents", d)
print(json.dumps({k: d.get(k) for k in ("documents", "by_kind", "llm", "processed", "unchanged", "written", "rejected", "expired", "elapsed_s")}, ensure_ascii=False))
errs = d.get("errors") or []
print(f"errors: {len(errs)}")
for e in errs[:20]: print("  -", e)
'
    echo "re-sync took $(( $(date +%s) - START ))s"
fi

echo; echo "## 3. restart services"
if [ "$DRY" = 1 ] || [ "$RESTART" = 0 ]; then
    echo "skipped"
else
    sudo /usr/bin/systemctl restart leninbot-api leninbot-telegram
    sleep 25
    systemctl is-active leninbot-api leninbot-telegram
    journalctl -u leninbot-api --since "2 minutes ago" --no-pager | grep -E "KG\] init|Traceback" | tail -3 || true
fi

echo; echo "## 4. verify"
KG_ENTITY_GATED_RECALL=1 "$PY" - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from kg_runtime.identity import get_alias_index
from kg_runtime.recall import entity_gated_kg_block
from kg_runtime.search import _run_rows
idx = get_alias_index(); idx.refresh_from_neo4j()
for q in ("소련의 붕괴", "미국의 관세", "정권 교체", "노조 활동", "에너지 전환 정책"):
    print(f"  match {q!r}: {[(h.name, h.key) for h in idx.match(q, broad=False)]}")
b = entity_gated_kg_block("소련은 왜 무너졌지?", "claude")
print("  recall entities:", [l[2:50] for l in b.split("\n") if l.startswith("- ")])
print("  Reference lines in block:", b.count("—Reference→"))
twins = _run_rows("MATCH (n:Entity) WHERE n.name IN ['미국','소련','이란','독일','러시아'] AND n.group_id='documents' RETURN n.name AS name, labels(n) AS l")
print("  Korean-named document twins left:", [(t["name"], t["l"]) for t in twins])
bad = _run_rows("MATCH (n:Entity {name:'United States'}) RETURN size(coalesce(n.aliases,[])) AS na, n.aliases AS al")
print("  United States aliases:", bad[0]["na"], bad[0]["al"])
PYEOF
echo; echo "== done $(date '+%F %T')  (log: $LOG)"
