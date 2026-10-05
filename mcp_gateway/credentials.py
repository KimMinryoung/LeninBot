"""MCP-only, single-secret fallback. Never bridge another service's credentials."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE_CREDENTIALS = Path('/run/credentials')


def bootstrap_kg_credentials() -> dict:
    from dotenv import load_dotenv
    dotenv_denied = False
    try:
        load_dotenv(ROOT / '.env')
    except OSError:
        # Listing/docs must still work when a checkout's .env is unreadable.
        dotenv_denied = True
    # Explicit configuration is authoritative, including an unreadable directory.
    if os.environ.get('CREDENTIALS_DIRECTORY'):
        path = Path(os.environ['CREDENTIALS_DIRECTORY']) / 'neo4j_password'
        try:
            value = path.read_text().rstrip('\n')
        except PermissionError:
            return {'status': 'permission_denied', 'source': 'explicit_credentials'}
        except FileNotFoundError:
            value = ''
        except OSError:
            return {'status': 'unavailable', 'source': 'explicit_credentials'}
        if value:
            os.environ.setdefault('NEO4J_PASSWORD', value)
            return {'status': 'configured', 'source': 'explicit_credentials'}
        return {'status': 'configured' if os.environ.get('NEO4J_PASSWORD') else 'credential_missing', 'source': 'environment'}
    if os.environ.get('NEO4J_PASSWORD'):
        return {'status': 'configured', 'source': 'environment'}
    if os.geteuid() != ROOT.stat().st_uid:
        return {'status': 'permission_denied', 'source': 'repository_owner_required'}
    denied = dotenv_denied
    for unit in ('leninbot-api.service', 'leninbot-telegram.service'):
        try:
            value = (SERVICE_CREDENTIALS / unit / 'neo4j_password').read_text().rstrip('\n')
        except PermissionError:
            denied = True
            continue
        except FileNotFoundError:
            continue
        except OSError:
            denied = True
            continue
        if value:
            os.environ['NEO4J_PASSWORD'] = value
            return {'status': 'configured', 'source': unit}
    return {'status': 'permission_denied' if denied else 'credential_missing', 'source': None}
