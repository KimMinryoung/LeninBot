# Secret Management

최종 확인 기준: 2026-05-09 코드 트리.

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

- `leninbot-roleplay.service` mounts a minimal credential set via its `.service.d/credentials.conf` drop-in: `deepseek_api_key`, `db_password`, `neo4j_password`, `tavily_api_key` — enough for its four read-only tools and DeepSeek over the Anthropic-compatible endpoint, nothing more. The bot token `ROLEPLAY_BOT_TOKEN` is supplied via `EnvironmentFile=.env` (or a `roleplay_bot_token` credential if migrated). Drop-ins are applied with `scripts/apply_credentials_dropin.sh` and are host-managed (not committed), matching the existing services.
- `novel-writer-api.service` mounts `admin_api_key`, `anthropic_api_key`, `db_password`, `deepseek_api_key`, and `tavily_api_key` for the `/writer` personal fiction workspace. `db_password` is required by `security_gateway/audit.py` because writer tool calls still write `tool_audit_log` in the main PostgreSQL database; writer manuscript/project tables remain on the local writer Postgres via `WRITER_DB_*`. Optional dedicated `writer_anthropic_api_key`, `writer_access_key`, and writer DB credentials should be mounted there too if migrated into the encrypted store. During the compatibility phase, `leninbot-api.service` may also mount writer-related credentials because it still includes `/writer/*` routes. `creative_writer.py` checks `WRITER_ANTHROPIC_API_KEY` first and falls back to `ANTHROPIC_API_KEY`, so a separate paid Claude Fable 5 key can be isolated from normal Claude usage. `WRITER_ACCESS_KEY` can protect direct writer API calls separately; if unset, `/writer/*` accepts `ADMIN_API_KEY` through `X-Writer-Key`. The public frontend writer page normally uses the admin login session and server-side key injection instead of exposing either key to the browser.

## Operational Notes

- A service restart is required after credential rotation because clients and `get_secret()` cache values in-process.
- Shell exports and `.env` values take precedence during the bridge via `setdefault`, so local development can override credentials intentionally.
- `PROJECT_ROOT` is derived from `secrets_loader.py` location when not set, so clones should not need a machine-specific project-root secret.
