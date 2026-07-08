from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from secrets_loader import get_secret

_ADMIN_API_KEY = get_secret("ADMIN_API_KEY", "") or ""
_WRITER_ACCESS_KEY = get_secret("WRITER_ACCESS_KEY", "") or _ADMIN_API_KEY
_WEBCHAT_PROXY_SECRET = get_secret("WEBCHAT_PROXY_SECRET", "") or ""

_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)
_writer_key_header = APIKeyHeader(name="X-Writer-Key", auto_error=False)


async def require_admin(api_key: str = Security(_admin_key_header)):
    """Dependency that enforces admin API key for sensitive endpoints."""
    if not _ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin API key not configured on server")
    if not api_key or api_key != _ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing admin API key")


async def require_writer_access(
    writer_key: str = Security(_writer_key_header),
    admin_key: str = Security(_admin_key_header),
):
    """Allow the writer UI through frontend proxies that strip X-Admin-Key."""
    if not _WRITER_ACCESS_KEY and not _ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Writer access key not configured on server")
    if writer_key and writer_key == _WRITER_ACCESS_KEY:
        return
    if _ADMIN_API_KEY and admin_key and admin_key == _ADMIN_API_KEY:
        return
    raise HTTPException(status_code=403, detail="Invalid or missing writer access key")


def is_admin_request(http_req: Request) -> bool:
    """Non-raising admin check for public endpoints with admin-only extras."""
    if not _ADMIN_API_KEY:
        return False
    key = http_req.headers.get("x-admin-key", "")
    return bool(key) and key == _ADMIN_API_KEY


def trusted_proxy_request(http_req: Request) -> bool:
    """Return True only for headers injected by the trusted frontend proxy."""
    if not _WEBCHAT_PROXY_SECRET:
        return False
    supplied = http_req.headers.get("x-webchat-proxy-secret", "")
    return bool(supplied and supplied == _WEBCHAT_PROXY_SECRET)
