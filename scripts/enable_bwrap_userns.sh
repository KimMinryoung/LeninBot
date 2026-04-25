#!/usr/bin/env bash
#
# enable_bwrap_userns.sh — Allow bubblewrap (bwrap) to create user namespaces.
#
# Ubuntu 24.04 ships with kernel.apparmor_restrict_unprivileged_userns=1, which
# blocks every bwrap invocation with "setting up uid map: Permission denied".
# That kills the Codex CLI's per-command sandbox even when launched with
# --dangerously-bypass-approvals-and-sandbox (Codex 0.124 still spawns bwrap
# internally for some shell calls). This script installs a narrow AppArmor
# profile that whitelists /usr/bin/bwrap for userns creation while leaving
# every other binary's confinement intact.
#
# Idempotent: re-running it overwrites the profile with the canonical content
# and reloads apparmor. Verifies success with a real bwrap invocation.
#
# Requires sudo. Run from anywhere:
#     sudo ./scripts/enable_bwrap_userns.sh

set -euo pipefail

PROFILE_PATH="/etc/apparmor.d/bwrap"

if [[ $EUID -ne 0 ]]; then
    echo "[!] This script must be run as root (use sudo)." >&2
    exit 1
fi

if [[ ! -x /usr/bin/bwrap ]]; then
    echo "[!] /usr/bin/bwrap not found — install bubblewrap first:" >&2
    echo "    apt install bubblewrap" >&2
    exit 1
fi

echo "[*] Writing AppArmor profile to $PROFILE_PATH ..."
cat > "$PROFILE_PATH" <<'EOF'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap>
}
EOF

echo "[*] Reloading apparmor service ..."
systemctl reload apparmor

echo "[*] Verifying bwrap can create a user namespace ..."
if bwrap --dev /dev --proc /proc --bind / / -- /bin/echo "bwrap-OK" 2>&1 | grep -q "bwrap-OK"; then
    echo "[✓] Success — bwrap user namespace creation is now allowed."
    echo "    Codex CLI subprocesses will pick this up automatically on their next spawn."
    echo "    No leninbot-telegram restart required."
else
    echo "[✗] bwrap test failed even after reload. Check 'journalctl -k -g apparmor' for denials." >&2
    exit 1
fi
