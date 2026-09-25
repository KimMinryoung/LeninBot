#!/usr/bin/env bash
# Rewrite requirements.lock.txt from the production venv after dependencies
# change (pip freeze). scripts/cloud_setup.sh installs it with --no-deps: the
# production venv holds versions pip's resolver would refuse (e.g. browser-use
# pins an older anthropic), so only an exact replay reproduces it.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
{
    echo "# pip freeze of the production venv; regenerate with scripts/freeze_requirements_lock.sh."
    echo "# Installed with --no-deps by scripts/cloud_setup.sh. requirements.txt stays the source of intent."
    "$ROOT/venv/bin/pip" freeze
} > "$ROOT/requirements.lock.txt.new"
mv "$ROOT/requirements.lock.txt.new" "$ROOT/requirements.lock.txt"
