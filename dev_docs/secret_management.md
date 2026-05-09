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

## Operational Notes

- A service restart is required after credential rotation because clients and `get_secret()` cache values in-process.
- Shell exports and `.env` values take precedence during the bridge via `setdefault`, so local development can override credentials intentionally.
- `PROJECT_ROOT` is derived from `secrets_loader.py` location when not set, so clones should not need a machine-specific project-root secret.
