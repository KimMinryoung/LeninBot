#!/usr/bin/env python3
"""Migrate Tier A secrets from .env to /etc/credstore.encrypted/.

Usage (as root):
    sudo venv/bin/python scripts/migrate_secrets_to_credstore.py [--force]

Reads .env, encrypts each TIER_A value with `systemd-creds encrypt --name=<lower>`,
writes to /etc/credstore.encrypted/<lower>.cred. Idempotent: skips existing files
unless --force.

After this script runs, apply the emitted drop-in snippet to each service's
/etc/systemd/system/<svc>.service.d/override.conf, then:
    sudo systemctl daemon-reload
    sudo systemctl restart leninbot-<svc>
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TIER_A = [
    "ADMIN_API_KEY",
    "AI_DIARY_API_KEY",
    "ALIBABA_API_KEY",
    "ANTHROPIC_API_KEY",
    "DB_PASSWORD",
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
    "RENDER_API_KEY",
    "REPLICATE_API_TOKEN",
    "RESEND_API_KEY",
    "TAVILY_API_KEY",
    "TELEGRAM_BOT_TOKEN",
]

CREDSTORE = Path("/etc/credstore.encrypted")
ENV_PATH = Path("/home/grass/leninbot/.env")
DROPIN_OUT = Path("/home/grass/leninbot/scripts/systemd-credentials.conf")


def parse_env(path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
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


def main(argv: list[str]) -> int:
    if os.geteuid() != 0:
        print("ERROR: must run as root (sudo).", file=sys.stderr)
        return 1

    force = "--force" in argv
    values = parse_env(ENV_PATH)
    CREDSTORE.mkdir(mode=0o700, exist_ok=True)

    migrated: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []

    for var in TIER_A:
        val = values.get(var, "")
        if not val:
            missing.append(var)
            continue
        cred_name = var.lower()
        out = CREDSTORE / f"{cred_name}.cred"
        if out.exists() and not force:
            skipped.append(var)
            continue
        encrypt_one(cred_name, val, out)
        migrated.append(var)

    print(f"migrated: {len(migrated)}")
    for v in migrated:
        print(f"  + {v} -> {CREDSTORE}/{v.lower()}.cred")
    if skipped:
        print(f"skipped (use --force to overwrite): {len(skipped)}")
        for v in skipped:
            print(f"  = {v}")
    if missing:
        print(f"missing from .env: {len(missing)}")
        for v in missing:
            print(f"  ? {v}")

    lines = ["[Service]"]
    for var in TIER_A:
        if var in missing:
            continue
        cred_name = var.lower()
        lines.append(
            f"LoadCredentialEncrypted={cred_name}:{CREDSTORE}/{cred_name}.cred"
        )
    DROPIN_OUT.write_text("\n".join(lines) + "\n")
    os.chmod(DROPIN_OUT, 0o644)
    print()
    print(f"drop-in snippet written to: {DROPIN_OUT}")
    print("apply to each service:")
    print(
        f"  sudo mkdir -p /etc/systemd/system/<svc>.service.d && "
        f"sudo cp {DROPIN_OUT} /etc/systemd/system/<svc>.service.d/credentials.conf"
    )
    print("  sudo systemctl daemon-reload && sudo systemctl restart <svc>")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
