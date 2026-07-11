#!/usr/bin/env bash
# One-shot apply for the 2026-07-11 body-H1 demotion (11 research_documents rows).
# Fixes: fixed2.csv produced by scripts/demote_research_body_h1.py.
# Backup already saved: data/publication_drafts/research_body_h1_backup_20260711.json
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXES="${1:?usage: apply_h1_demotion_20260711.sh <fixed2.csv>}"

cd "$(dirname "$FIXES")"
"$ROOT/scripts/psql-supabase" <<SQL
BEGIN;
CREATE TABLE _h1_fix (id bigint PRIMARY KEY, markdown text NOT NULL, markdown_en text, sha text NOT NULL);
\copy _h1_fix FROM '$(basename "$FIXES")' WITH (FORMAT csv)
UPDATE research_documents r
   SET markdown = t.markdown,
       markdown_en = NULLIF(coalesce(t.markdown_en, ''), ''),
       content_sha256 = t.sha
  FROM _h1_fix t
 WHERE r.id = t.id;
DROP TABLE _h1_fix;
COMMIT;
SQL

cd "$ROOT"
venv/bin/python - <<'PY'
from runtime_tools.research import _invalidate_cache_sync, _purge_cloudflare_sync
FILES = [
    "20260405_cyber_lenin_com_seo_optimization_strategy_2026.md",
    "20260416_research.md",
    "20260504_2026.md",
    "20260504_anti_communism_01_formation.md",
    "20260508_dpk_analysis.md",
    "caribbean-recolonization-venezuela-cuba-2026.md",
    "kospi-failed-rally-june-12-2026.md",
    "korea-real-economy-briefing-2026-05.md",
    "korea-employment-class-war-may-2026.md",
    "korea-working-class-triple-crush-june-2026.md",
    "korea-employment-class-war-june-2026.md",
]
deleted = 0
for f in FILES:
    deleted += _invalidate_cache_sync(f).get("deleted", 0)
    cf = _purge_cloudflare_sync(f)
    if not cf["ok"]:
        print(f"cloudflare purge FAILED for {f}: {cf.get('reason')}")
print(f"redis keys deleted: {deleted}")
PY
echo "done — verify with: curl -s https://cyber-lenin.com/reports/research/korea-employment-class-war-june-2026 | grep -c '<h1'"
