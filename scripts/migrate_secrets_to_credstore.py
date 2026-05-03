#!/usr/bin/env python3
"""Migrate Tier A secrets from .env to /etc/credstore.encrypted/ and emit
per-service drop-ins with least-privilege credential scopes.

Usage (as root):
    sudo venv/bin/python scripts/migrate_secrets_to_credstore.py [--force]

Reads .env, encrypts each TIER_A value with `systemd-creds encrypt --name=<lower>`,
writes to /etc/credstore.encrypted/<lower>.cred. Idempotent: skips existing
files unless --force.

Also emits per-service drop-ins at scripts/dropins/leninbot-<svc>.service.d.conf
according to SERVICE_CREDS below. Running apply_credentials_dropin.sh installs
them with `systemctl daemon-reload && systemctl restart`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# Tier A — the full set of secrets that exist in the credstore
# ═══════════════════════════════════════════════════════════════════════════
TIER_A = [
    "ADMIN_API_KEY",
    "AI_DIARY_API_KEY",
    "ALIBABA_API_KEY",
    "ANTHROPIC_API_KEY",
    "DB_PASSWORD",
    "DEEPL_API_KEY",
    "DEEPSEEK_API_KEY",
    "EMAIL_IMAP_PASSWORD",
    "EMAIL_SMTP_PASSWORD",
    "GEMINI_API_KEY",
    "GITHUB_TOKEN",
    "GRAFFITI_API_KEY",
    "HF_TOKEN",
    "MOLTBOOK_API_KEY",
    "NEO4J_PASSWORD",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "R2_CF_API_TOKEN",
    "REPLICATE_API_TOKEN",
    "RESEND_API_KEY",
    "TAVILY_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "X_BEARER_TOKEN",
]

# ═══════════════════════════════════════════════════════════════════════════
# Per-service credential scopes (least privilege).
# api/telegram host the agent and its wide tool surface — they get the full set.
# Narrower services list only what they actually use.
# ═══════════════════════════════════════════════════════════════════════════
_FULL = set(TIER_A)

SERVICE_CREDS: dict[str, set[str]] = {
    # Agent hosts — broad tool access, full Tier A.
    "leninbot-api": _FULL,
    "leninbot-telegram": _FULL,

    # Browser worker runs the task loop and writes task/log rows, so it needs DB.
    # browser-use itself can use Gemini/OpenAI/Anthropic depending on runtime env.
    "leninbot-browser": {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "DB_PASSWORD",
    },

    # T0 autonomous pilot — planning LLMs, KG/DB, telegram notify, razvedchik.
    "leninbot-autonomous": {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "NEO4J_PASSWORD",
        "DB_PASSWORD",
        "TELEGRAM_BOT_TOKEN",
        "MOLTBOOK_API_KEY",
        "TAVILY_API_KEY",
        "GITHUB_TOKEN",
        "X_BEARER_TOKEN",
    },

    # Moltbook patrol timer — writes through the Razvedchik API client and can
    # optionally send Telegram completion notices.
    "leninbot-razvedchik": {
        "MOLTBOOK_API_KEY",
        "TELEGRAM_BOT_TOKEN",
    },

    # Diary writer (daily 00:30) — Gemini for writing, DB/KG for reading activity.
    # Imports bot_config so ANTHROPIC/OPENAI present too.
    "leninbot-experience": {
        "GEMINI_API_KEY",
        "NEO4J_PASSWORD",
        "DB_PASSWORD",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
    },

    # KG backup (daily 03:00) — R2 upload + Neo4j dump.
    "leninbot-kg-backup": {"R2_CF_API_TOKEN", "NEO4J_PASSWORD"},
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDSTORE = Path("/etc/credstore.encrypted")
ENV_PATH = PROJECT_ROOT / ".env"
DROPIN_DIR = PROJECT_ROOT / "scripts" / "dropins"


def parse_env(path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    if not path.is_file():
        return vals
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        if v and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if k:
            vals[k] = v
    return vals


def encrypt_one(cred_name: str, value: str, out_path: Path) -> None:
    subprocess.run(
        ["systemd-creds", "encrypt", f"--name={cred_name}", "-", str(out_path)],
        input=value.encode(),
        check=True,
    )
    os.chmod(out_path, 0o400)


def _which_creds_missing_in_credstore() -> set[str]:
    """Return creds referenced in SERVICE_CREDS but not present in credstore."""
    present = {p.stem for p in CREDSTORE.iterdir() if p.suffix == ".cred"} if CREDSTORE.exists() else set()
    needed: set[str] = set()
    for creds in SERVICE_CREDS.values():
        needed.update(creds)
    return {c for c in needed if c.lower() not in present}


def emit_dropins(missing: set[str]) -> list[Path]:
    """Write per-service drop-in files. Returns list of written paths."""
    DROPIN_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for svc, creds in SERVICE_CREDS.items():
        lines = ["[Service]"]
        for cred in sorted(creds):
            if cred in missing:
                continue
            name = cred.lower()
            lines.append(
                f"LoadCredentialEncrypted={name}:{CREDSTORE}/{name}.cred"
            )
        out = DROPIN_DIR / f"{svc}.conf"
        out.write_text("\n".join(lines) + "\n")
        try:
            os.chmod(out, 0o644)
        except PermissionError:
            pass
        written.append(out)
    return written


def main(argv: list[str]) -> int:
    if os.geteuid() != 0:
        print("ERROR: must run as root (sudo).", file=sys.stderr)
        return 1

    force = "--force" in argv
    skip_encrypt = "--dropins-only" in argv

    values = parse_env(ENV_PATH)
    CREDSTORE.mkdir(mode=0o700, exist_ok=True)

    migrated: list[str] = []
    skipped_existing: list[str] = []
    missing_in_env: set[str] = set()

    if not skip_encrypt:
        for var in TIER_A:
            val = values.get(var, "")
            out = CREDSTORE / f"{var.lower()}.cred"
            if not val:
                if not out.exists():
                    missing_in_env.add(var)
                continue
            if out.exists() and not force:
                skipped_existing.append(var)
                continue
            encrypt_one(var.lower(), val, out)
            migrated.append(var)

        print(f"migrated: {len(migrated)}")
        for v in migrated:
            print(f"  + {v} -> {CREDSTORE}/{v.lower()}.cred")
        if skipped_existing:
            print(f"skipped (use --force to overwrite): {len(skipped_existing)}")
        if missing_in_env:
            print(f"missing from .env and credstore: {len(missing_in_env)}")
            for v in sorted(missing_in_env):
                print(f"  ? {v}")

    # Check which creds referenced by SERVICE_CREDS are absent from credstore.
    absent_in_store = _which_creds_missing_in_credstore()
    if absent_in_store:
        print(f"WARNING: SERVICE_CREDS references {len(absent_in_store)} creds not in credstore:")
        for v in sorted(absent_in_store):
            print(f"  ! {v} (will be omitted from drop-ins)")

    written = emit_dropins(absent_in_store)
    print()
    print(f"drop-ins written: {len(written)}")
    for p in written:
        svc_name = p.stem
        n_creds = len(SERVICE_CREDS[svc_name]) - len(absent_in_store & SERVICE_CREDS[svc_name])
        print(f"  -> {p} ({n_creds} creds)")
    print()
    print("apply with: sudo scripts/apply_credentials_dropin.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
