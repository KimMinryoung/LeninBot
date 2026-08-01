#!/usr/bin/env bash
# Deploy the off-VM watchdog Worker (watchdog/).
#
# Idempotent: re-running reuses the existing KV namespace and keeps secrets
# already set. Nothing here touches the main VM's services — after a successful
# deploy you still have to put the printed WATCHDOG_PING_BASE line into .env and
# reinstall the units, or the dead-man's-switch pings stay no-ops.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/watchdog"

# wrangler 3 is the last line that supports the Node 18 on this host; override
# with WRANGLER=wrangler@4 once Node is newer.
WRANGLER="${WRANGLER:-wrangler@3}"
run_wrangler() { npx --yes "$WRANGLER" "$@"; }

if [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
  echo "ERROR: CLOUDFLARE_API_TOKEN is not set." >&2
  echo "  Create one with the 'Edit Cloudflare Workers' template, then:" >&2
  echo "    export CLOUDFLARE_API_TOKEN=...   # this shell only, do not commit" >&2
  exit 1
fi

# --- KV namespace -----------------------------------------------------------
if grep -q 'REPLACE_WITH_KV_NAMESPACE_ID' wrangler.toml; then
  echo "==> creating KV namespace WATCHDOG"
  created="$(run_wrangler kv:namespace create WATCHDOG 2>&1)" || { echo "$created" >&2; exit 1; }
  echo "$created"
  kv_id="$(printf '%s' "$created" | grep -oE 'id\s*=\s*"[0-9a-f]{32}"' | grep -oE '[0-9a-f]{32}' | head -1)"
  if [[ -z "$kv_id" ]]; then
    echo "ERROR: could not parse the namespace id out of wrangler's output." >&2
    echo "  Put it into wrangler.toml by hand and re-run." >&2
    exit 1
  fi
  sed -i "s/REPLACE_WITH_KV_NAMESPACE_ID/$kv_id/" wrangler.toml
  echo "    wrangler.toml id = $kv_id"
else
  echo "==> KV namespace already configured in wrangler.toml"
fi

# --- secrets ----------------------------------------------------------------
# Reuse the values the VM already has so the watchdog talks to the same chat.
read_cred() {
  local name="$1"
  for d in /run/credentials/leninbot-telegram.service /run/credentials/leninbot-api.service; do
    [[ -r "$d/$name" ]] && { cat "$d/$name"; return 0; }
  done
  return 1
}

bot_token="$(read_cred telegram_bot_token || true)"
if [[ -z "$bot_token" ]]; then
  read -rsp "TELEGRAM_BOT_TOKEN: " bot_token; echo
fi

chat_id="$(grep -E '^TELEGRAM_CHAT_ID=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"'"'"' ' || true)"
if [[ -z "$chat_id" ]]; then
  read -rp "TELEGRAM_CHAT_ID: " chat_id
fi

ping_token_file="$ROOT/.watchdog_ping_token"
if [[ -s "$ping_token_file" ]]; then
  ping_token="$(cat "$ping_token_file")"
  echo "==> reusing existing PING_TOKEN from $ping_token_file"
else
  ping_token="$(openssl rand -hex 24)"
  (umask 077; printf '%s' "$ping_token" > "$ping_token_file")
  echo "==> generated a new PING_TOKEN (saved 0600 to $ping_token_file)"
fi

echo "==> putting secrets"
printf '%s' "$bot_token"   | run_wrangler secret put TELEGRAM_BOT_TOKEN
printf '%s' "$chat_id"     | run_wrangler secret put TELEGRAM_CHAT_ID
printf '%s' "$ping_token"  | run_wrangler secret put PING_TOKEN

# --- deploy -----------------------------------------------------------------
echo "==> deploying"
deploy_out="$(run_wrangler deploy 2>&1)"; echo "$deploy_out"
worker_url="$(printf '%s' "$deploy_out" | grep -oE 'https://[a-z0-9.-]+\.workers\.dev' | head -1)"

echo
echo "════════════════════════════════════════════════════════════════"
if [[ -n "$worker_url" ]]; then
  echo "Add this line to $ROOT/.env :"
  echo
  echo "  WATCHDOG_PING_BASE=$worker_url/ping/$ping_token"
  echo
  echo "Then reinstall the units so the pings go live:"
  echo "  sudo cp $ROOT/systemd/* /etc/systemd/system/ && sudo systemctl daemon-reload"
  echo
  echo "Check state any time:"
  echo "  curl $worker_url/status/$ping_token"
else
  echo "Deploy finished but the workers.dev URL could not be parsed;"
  echo "take it from the output above and build WATCHDOG_PING_BASE as"
  echo "  <worker-url>/ping/<PING_TOKEN from $ping_token_file>"
fi
echo "════════════════════════════════════════════════════════════════"
