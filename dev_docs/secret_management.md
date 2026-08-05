# Secret Management

최종 확인 기준: 2026-08-05 코드 트리.

Production services read secrets from systemd credentials. Local development can still use `.env`. The common access layer is `secrets_loader.py`.

## Read Order

`get_secret("NAME")` resolves values in this order:

1. `$CREDENTIALS_DIRECTORY/name_lower`
2. `os.environ["NAME"]`
3. provided default

At import time, `secrets_loader.py` calls `load_dotenv()` and then bridges every valid credential filename in `$CREDENTIALS_DIRECTORY` into `os.environ` with uppercase names. This lets third-party SDKs that only read environment variables still work under systemd credentials.

Credential values are cached by `get_secret()` for the process lifetime. Rotate credentials by restarting affected services after updating the encrypted store.

## Naming

Credential filename is the lowercased env var name:

| Env var | Credential file |
|---|---|
| `ANTHROPIC_API_KEY` | `anthropic_api_key` |
| `WRITER_ANTHROPIC_API_KEY` | `writer_anthropic_api_key` |
| `WRITER_ACCESS_KEY` | `writer_access_key` |
| `OPENAI_API_KEY` | `openai_api_key` |
| `DEEPSEEK_API_KEY` | `deepseek_api_key` |
| `MOONSHOT_API_KEY` | `moonshot_api_key` |
| `GEMINI_API_KEY` | `gemini_api_key` |
| `KG_GEMINI_API_KEY` | `kg_gemini_api_key` |
| `TAVILY_API_KEY` | `tavily_api_key` |
| `BRAVE_SEARCH_API_KEY` | `brave_search_api_key` |
| `ADMIN_API_KEY` | `admin_api_key` |

The startup bridge only exports credential files whose names match lowercase env-var shape: letters, digits, and underscores, starting with a letter or underscore.

## Classification

| Tier | Store | Examples |
|---|---|---|
| Secret | systemd encrypted credentials in production | API keys, bot tokens, DB passwords, wallet private keys |
| Non-secret config | `.env`, systemd `Environment=`, config files | hostnames, public URLs, feature flags, model base URLs |
| Runtime state | database or local runtime config | `config.json`, task rows, mission state, project state |

Do not add real secrets to repository files. `config.json.example` and `.env` examples should contain placeholders only.

## Management Tooling

Use `scripts/manage_secrets.py` for listing, adding, rotating, and inspecting credential metadata. Production systemd units should use `LoadCredentialEncrypted=`.

Relevant implementation files:

- `secrets_loader.py`
- `scripts/manage_secrets.py`
- `scripts/migrate_secrets_to_credstore.py`
- `scripts/apply_credentials_dropin.sh`
- `systemd/*.service`

## Per-Service Notes

- `leninbot-llm-proxy.service` is the sole production custodian for `anthropic_api_key`, `deepseek_api_key`, `moonshot_api_key`, `openai_api_key`, and `gemini_api_key`. LLM-consuming services receive placeholders and start after the proxy is ready; `scripts/remove_llm_provider_keys.sh` removes stale provider-key mounts. When present in credstore, `scripts/migrate_secrets_to_credstore.py` mounts `writer_anthropic_api_key` and `kg_gemini_api_key` only on the proxy; scoped routes prefer them and fall back to the corresponding shared key.
- `leninbot-roleplay.service` no longer mounts `deepseek_api_key`; its tool/database/search credentials remain service-local and model traffic uses the proxy.
- `novel-writer-api.service` mounts application/tool credentials such as `admin_api_key`, `writer_access_key`, `db_password`, and search keys, but not provider keys. Claude uses proxy route `anthropic-writer`, while DeepSeek/Kimi use their proxy routes. `WRITER_ACCESS_KEY` can protect direct writer API calls separately; if unset, `/writer/*` accepts `ADMIN_API_KEY` through `X-Writer-Key`.
- `leninbot-email-api.service` mounts `admin_api_key`, `db_password`, `email_imap_password`, `email_smtp_password`, and `resend_api_key` for admin-gated `/email/*` review, draft, approval, manual poll, and outbound send paths. The public frontend uses the existing admin login session and injects the backend admin key server-side through `/api/proxy/email/*`.
- `leninbot-email-poller.service` mounts only `db_password` and `email_imap_password`; it runs `scripts/email_poll_once.py` from `leninbot-email-poller.timer` and stores unseen inbound messages in the email bridge tables.
- `leninbot-a2a-api.service` mounts only its non-provider application/tool credentials (`db_password`, `neo4j_password`, `tavily_api_key`, `github_token`, etc.); all LLM provider credentials stay in the proxy. `A2A_ENABLED` remains non-secret config from `.env`/systemd environment.
- `scripts/apply_credentials_dropin.sh` regenerates and installs the drop-ins, then restarts the long-running API, A2A API, Telegram, browser, roleplay, and novel-writer services. Timer-driven services load the credential on their next run.
- `runtime_tools.web_search` reads `TAVILY_API_KEY` and `BRAVE_SEARCH_API_KEY`. `WEB_SEARCH_PROVIDERS` is non-secret provider order (default `tavily,brave`), and `WEB_SEARCH_PROVIDER_COOLDOWN_SECONDS` controls the process-local failure circuit. Missing provider credentials are handled through fallback; `scripts/migrate_secrets_to_credstore.py` emits a credential line only when that encrypted credential exists.

## Operational Notes

- A service restart is required after credential rotation because clients and `get_secret()` cache values in-process.
- Shell exports and `.env` values take precedence during the bridge via `setdefault`, so local development can override credentials intentionally.
- `PROJECT_ROOT` is derived from `secrets_loader.py` location when not set, so clones should not need a machine-specific project-root secret.
