#!/usr/bin/env python3
"""Manage systemd-creds encrypted secrets in /etc/credstore.encrypted/.

Operations (all require root):
  list                — show registered creds (name, age, size, services using it)
  add <NAME>          — add a new credential; value read via hidden prompt or stdin
  rotate <NAME>       — replace an existing credential value
  delete <NAME>       — remove a credential (with confirmation)

Values are never printed. Inputs come from an interactive hidden prompt
(getpass) or piped stdin. After add/rotate/delete, restart services that
mount the credential so the tmpfs reflects the new value.
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CREDSTORE = Path("/etc/credstore.encrypted")
SYSTEMD_SYSTEM = Path("/etc/systemd/system")


def _require_root() -> None:
    if os.geteuid() != 0:
        print("ERROR: must run as root (sudo).", file=sys.stderr)
        sys.exit(1)


def _cred_path(name: str) -> Path:
    return CREDSTORE / f"{name.lower()}.cred"


def _format_size(n: int) -> str:
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KiB"


def _format_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def _services_mounting(cred_name: str) -> list[str]:
    """Scan service drop-ins for `LoadCredentialEncrypted=<cred_name>:` lines."""
    needle = f"LoadCredentialEncrypted={cred_name.lower()}:"
    users: set[str] = set()
    for drop_in in SYSTEMD_SYSTEM.glob("*.service.d/*.conf"):
        try:
            text = drop_in.read_text()
        except OSError:
            continue
        if needle in text:
            users.add(drop_in.parent.name.removesuffix(".service.d"))
    return sorted(users)


def _read_value(prompt: str) -> str:
    """Read a secret value: stdin if piped, else hidden prompt with confirmation."""
    if not sys.stdin.isatty():
        value = sys.stdin.read().rstrip("\n")
        if not value:
            print("ERROR: empty value from stdin", file=sys.stderr)
            sys.exit(1)
        return value
    v1 = getpass.getpass(f"{prompt}: ")
    if not v1:
        print("ERROR: empty value", file=sys.stderr)
        sys.exit(1)
    v2 = getpass.getpass("confirm         : ")
    if v1 != v2:
        print("ERROR: values do not match", file=sys.stderr)
        sys.exit(1)
    return v1


def _encrypt_atomic(cred_name: str, value: str, out_path: Path) -> None:
    """Encrypt value and write .cred atomically (tmp file + rename)."""
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        subprocess.run(
            ["systemd-creds", "encrypt", f"--name={cred_name}", "-", str(tmp)],
            input=value.encode(),
            check=True,
        )
        os.chmod(tmp, 0o400)
        tmp.replace(out_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def cmd_list(args: argparse.Namespace) -> int:
    _require_root()
    if not CREDSTORE.exists():
        print(f"credstore not initialized: {CREDSTORE} does not exist")
        return 1
    creds = sorted(p for p in CREDSTORE.iterdir() if p.suffix == ".cred")
    if not creds:
        print("(no credentials registered)")
        return 0
    rows = []
    for p in creds:
        name_lower = p.stem
        st = p.stat()
        users = _services_mounting(name_lower)
        rows.append((name_lower, st.st_size, st.st_mtime, users))

    w_name = max(len("NAME"), max(len(r[0]) for r in rows))
    print(f"{'NAME':<{w_name}}  {'SIZE':>8}  {'MODIFIED':<16}  USED BY")
    print(f"{'-'*w_name}  {'-'*8}  {'-'*16}  {'-'*7}")
    for name, size, mtime, users in rows:
        used = ", ".join(s.removeprefix("leninbot-") for s in users) or "(unmounted)"
        print(f"{name:<{w_name}}  {_format_size(size):>8}  {_format_mtime(mtime):<16}  {used}")
    print()
    print(f"total: {len(creds)}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    _require_root()
    path = _cred_path(args.name)
    if path.exists():
        print(f"ERROR: {args.name.lower()} already exists. Use `rotate` to replace.", file=sys.stderr)
        return 1
    value = _read_value(f"value for {args.name.upper()}")
    _encrypt_atomic(args.name.lower(), value, path)
    print(f"+ added {args.name.lower()} -> {path}")
    print("Next: add `LoadCredentialEncrypted=...` to the relevant service drop-in and restart.")
    return 0


def cmd_rotate(args: argparse.Namespace) -> int:
    _require_root()
    path = _cred_path(args.name)
    if not path.exists():
        print(f"ERROR: {args.name.lower()} not registered. Use `add` to create.", file=sys.stderr)
        return 1
    users = _services_mounting(args.name)
    value = _read_value(f"new value for {args.name.upper()}")
    _encrypt_atomic(args.name.lower(), value, path)
    print(f"= rotated {args.name.lower()}")
    if users:
        print(f"Restart services to load new value: systemctl restart {' '.join(users)}")
    else:
        print("(no services currently mount this credential)")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    _require_root()
    path = _cred_path(args.name)
    if not path.exists():
        print(f"ERROR: {args.name.lower()} not registered.", file=sys.stderr)
        return 1
    users = _services_mounting(args.name)
    if users and not args.force:
        print(f"WARNING: {args.name.lower()} is still mounted by: {', '.join(users)}")
        print("Those services will FAIL TO START until their drop-ins are updated.")
        print("To retire this credential cleanly:")
        print(f"  1. Remove \"{args.name.upper()}\" from TIER_A / SERVICE_CREDS in")
        print("     scripts/migrate_secrets_to_credstore.py")
        print("  2. sudo scripts/apply_credentials_dropin.sh  (regenerates drop-ins + restart)")
        print(f"  3. Re-run: sudo manage_secrets.py delete {args.name.upper()} -f")
    if not args.force:
        confirm = input(f"Delete {args.name.lower()}? [y/N] ")
        if confirm.strip().lower() != "y":
            print("aborted.")
            return 1
    path.unlink()
    print(f"- deleted {args.name.lower()}")
    if users:
        print("Remove the matching LoadCredentialEncrypted= line from each drop-in, then:")
        print("  systemctl daemon-reload")
        print(f"  systemctl restart {' '.join(users)}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="show registered credentials")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("add", help="add a new credential")
    sp.add_argument("name", help="env-var-style name (e.g. GEMINI_API_KEY)")
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("rotate", help="replace an existing credential's value")
    sp.add_argument("name", help="env-var-style name")
    sp.set_defaults(func=cmd_rotate)

    sp = sub.add_parser("delete", help="remove a credential")
    sp.add_argument("name", help="env-var-style name")
    sp.add_argument("-f", "--force", action="store_true", help="skip confirmation")
    sp.set_defaults(func=cmd_delete)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
