#!/bin/bash
# Thin wrapper — all logic now lives in scripts/svc
# Kept for backward compatibility (Telegram /deploy calls this path)
exec "$(dirname "$0")/scripts/svc" deploy "$@"
