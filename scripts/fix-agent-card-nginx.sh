#!/usr/bin/env bash
# Fix: proxy A2A endpoints (/.well-known/agent.json, /a2a) to API (port 8000) on cyber-lenin.com
set -euo pipefail

CONF="/etc/nginx/sites-enabled/leninbot-frontend"

if grep -q 'well-known/agent.json' "$CONF" 2>/dev/null; then
    echo "Already configured. Nothing to do."
    exit 0
fi

sed -i '/ssl_certificate_key \/etc\/ssl\/cloudflare\/cyber-lenin.com.key;/a\
\
    # A2A Agent Card + endpoint — proxy to API\
    location = /.well-known/agent.json {\
        proxy_pass http://127.0.0.1:8000;\
        proxy_set_header Host $host;\
        proxy_set_header X-Real-IP $remote_addr;\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\
        proxy_set_header X-Forwarded-Proto $scheme;\
    }\
\
    location = /a2a {\
        proxy_pass http://127.0.0.1:8000;\
        proxy_set_header Host $host;\
        proxy_set_header X-Real-IP $remote_addr;\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\
        proxy_set_header X-Forwarded-Proto $scheme;\
        proxy_read_timeout 120s;\
    }' "$CONF"

nginx -t && systemctl reload nginx

echo "Done. Testing..."
curl -s -o /dev/null -w "/.well-known/agent.json → %{http_code}\n" https://cyber-lenin.com/.well-known/agent.json
curl -s -o /dev/null -w "/a2a → %{http_code}\n" -X POST https://cyber-lenin.com/a2a -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"SendMessage","params":{"message":{"role":"user","parts":[{"text":"ping"}]}},"id":1}'
