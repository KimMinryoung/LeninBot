"""Private RPC to the single frontend person store; never falls back to Python SQL."""
import json
import os
import subprocess


def call_person_service(request: dict):
    container = os.environ.get("COMMULINGO_FRONTEND_CONTAINER", "leninbot-frontend")
    script = 'commulingo-term-service.js' if request.get('target')=='term' else 'commulingo-person-service.js'
    completed = subprocess.run(
        ["docker", "exec", "-i", container, "node", f"/app/scripts/{script}"],
        input=json.dumps(request, ensure_ascii=False), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45, check=False,
    )
    try:
        response = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("CommuLingo shared store unavailable; no fallback write was attempted") from exc
    if not response.get("ok"):
        raise ValueError(f"{response.get('code') or response.get('status')}: {response.get('error')}")
    return response["result"]


def apply_person_spec(path):
    """Apply an explicit sourced/versioned spec via the atomic Admin upsert CLI."""
    from pathlib import Path
    spec = json.loads(Path(path).read_text())
    container = os.environ.get("COMMULINGO_FRONTEND_CONTAINER", "leninbot-frontend")
    completed = subprocess.run(
        ["docker", "exec", "-i", container, "node", "/app/scripts/commulingo-people-upsert.js", "-"],
        input=json.dumps(spec, ensure_ascii=False), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "CommuLingo Admin upsert failed")
    return completed.stdout.strip()
