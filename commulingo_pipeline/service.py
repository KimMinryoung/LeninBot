"""Private pipeline RPC, with an explicit receipt for every mutation."""
import json
import os
import subprocess


def call(request):
    result = subprocess.run(['docker','exec','-i',
        os.environ.get('COMMULINGO_FRONTEND_CONTAINER','leninbot-frontend'),
        'node','/app/scripts/commulingo-pipeline-service.js'],
        input=json.dumps(request,ensure_ascii=False),text=True,capture_output=True,timeout=45)
    try:
        value = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError('pipeline store unavailable; no fallback write attempted') from exc
    if not value.get('ok'):
        raise ValueError(f"{value.get('code') or value.get('status')}: {value.get('error')}")
    return value['result']
