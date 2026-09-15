#!/usr/bin/env python3
"""Stage a reviewable systemd migration as grass; never install or restart.

Copies only changed unit metadata, stripping search-key mounts while retaining
all unrelated local settings. Encrypted key material is never read or copied.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web_gateway.deployment import CONSUMERS

SYSTEM = Path("/etc/systemd/system")
STAGE = ROOT / "systemd/.web-gateway-stage"
KEY_LINE = re.compile(r"^LoadCredentialEncrypted=(tavily_api_key|brave_search_api_key):(/etc/credstore\.encrypted/[a-z_]+\.cred)\s*$")


def main():
    STAGE.mkdir(exist_ok=True)
    # Rebuild this tool-owned staging directory, not any installed unit.
    for item in STAGE.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    mounts = {}
    changes = []
    for service in CONSUMERS:
        unit = service + ".service"
        candidates = [SYSTEM / unit, *sorted((SYSTEM / (unit + ".d")).glob("*.conf"))]
        for path in candidates:
            if not path.is_file():
                continue
            original = path.read_text()
            kept = []
            for line in original.splitlines(keepends=True):
                match = KEY_LINE.fullmatch(line.strip())
                if match:
                    key, source = match.groups()
                    if key in mounts and mounts[key] != source:
                        raise RuntimeError("Conflicting credential source paths; resolve before installation")
                    mounts[key] = source
                else:
                    kept.append(line)
            updated = "".join(kept)
            if updated != original:
                target = STAGE / path.relative_to(SYSTEM)
                target.parent.mkdir(exist_ok=True)
                target.write_text(updated)
                changes.append(str(path))
        dependency = STAGE / (unit + ".d") / "web-gateway.conf"
        dependency.parent.mkdir(exist_ok=True)
        dependency.write_text("[Unit]\nWants=leninbot-web-gateway.service\nAfter=leninbot-web-gateway.service\n")
    if not mounts:
        # Repeat preparation after a completed migration reads the new owner.
        owner = SYSTEM / "leninbot-web-gateway.service.d/credentials.conf"
        if owner.exists():
            for line in owner.read_text().splitlines():
                match = KEY_LINE.fullmatch(line.strip())
                if match:
                    mounts[match[1]] = match[2]
    if not mounts:
        raise RuntimeError("No installed search credential references found")
    shutil.copyfile(ROOT / "systemd/leninbot-web-gateway.service", STAGE / "leninbot-web-gateway.service")
    owner_dir = STAGE / "leninbot-web-gateway.service.d"
    owner_dir.mkdir(exist_ok=True)
    (owner_dir / "credentials.conf").write_text("[Service]\n" + "".join(
        f"LoadCredentialEncrypted={key}:{source}\n" for key, source in sorted(mounts.items())))
    print(json.dumps({"stage": str(STAGE), "search_key_mounts_removed_from": changes,
                      "gateway_credentials": sorted(mounts), "consumers": list(CONSUMERS)}, indent=2))


if __name__ == "__main__":
    main()
