"""Admin user management API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api_security import require_admin
from db import query as db_query, query_one as db_query_one

router = APIRouter()


def _normalize_account_username(raw: str) -> str:
    import re

    value = (raw or "").strip()
    if not 1 <= len(value) <= 30:
        raise HTTPException(status_code=400, detail="username must be 1-30 characters")
    if not re.fullmatch(r"[\w][\w.\- ]*", value, flags=re.UNICODE):
        raise HTTPException(status_code=400, detail="invalid username")
    return value


class AdminUserRenameRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=30)


class AdminUserMergeRequest(BaseModel):
    target_user_id: int = Field(..., gt=0)


@router.get("/admin/users", dependencies=[Depends(require_admin)])
async def list_admin_users(
    search: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Admin account list with authentication and web-chat summary counts."""
    filters = []
    params: list[object] = []
    if search and search.strip():
        filters.append("u.username ILIKE %s")
        params.append(f"%{search.strip()}%")
    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    rows = db_query(
        f"""WITH page_users AS (
                 SELECT u.id, u.username, u.is_admin, u.created_at, u.last_login_at,
                        (u.password_hash IS NOT NULL) AS has_password
                   FROM users u
                    {where_sql}
               ORDER BY u.created_at DESC, u.id DESC
                  LIMIT %s OFFSET %s
             ),
             fingerprint_counts AS (
                 SELECT uf.user_id, COUNT(*)::int AS fingerprint_count
                   FROM user_fingerprints uf
                   JOIN page_users pu ON pu.id = uf.user_id
               GROUP BY uf.user_id
             ),
             passkey_counts AS (
                 SELECT up.user_id, COUNT(*)::int AS passkey_count
                   FROM user_passkeys up
                   JOIN page_users pu ON pu.id = up.user_id
               GROUP BY up.user_id
             ),
             chat_counts AS (
                 SELECT uf.user_id,
                        COUNT(cl.id)::int AS chat_log_count,
                        MAX(cl.created_at) AS last_chat_at
                   FROM user_fingerprints uf
                   JOIN page_users pu ON pu.id = uf.user_id
              LEFT JOIN chat_logs cl ON cl.fingerprint = uf.fingerprint
               GROUP BY uf.user_id
             )
             SELECT pu.id, pu.username, pu.is_admin, pu.created_at, pu.last_login_at,
                    pu.has_password,
                    COALESCE(fc.fingerprint_count, 0) AS fingerprint_count,
                    COALESCE(pc.passkey_count, 0) AS passkey_count,
                    COALESCE(cc.chat_log_count, 0) AS chat_log_count,
                    cc.last_chat_at
               FROM page_users pu
          LEFT JOIN fingerprint_counts fc ON fc.user_id = pu.id
          LEFT JOIN passkey_counts pc ON pc.user_id = pu.id
          LEFT JOIN chat_counts cc ON cc.user_id = pu.id
           ORDER BY pu.created_at DESC, pu.id DESC""",
        tuple(params + [limit, offset]),
    )
    total_row = db_query_one(
        f"SELECT COUNT(*)::int AS total FROM users u {where_sql}",
        tuple(params),
    )
    return {"users": rows, "total": int(total_row["total"]) if total_row else 0}


@router.get("/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def get_admin_user(user_id: int):
    """Admin detail for one public web user account."""
    user = db_query_one(
        """SELECT id, username, is_admin, created_at, last_login_at,
                  (password_hash IS NOT NULL) AS has_password
             FROM users
            WHERE id = %s""",
        (user_id,),
    )
    if not user:
        raise HTTPException(status_code=404, detail="user not found")
    fingerprints = db_query(
        """SELECT uf.fingerprint, uf.bound_at, COUNT(cl.id)::int AS chat_log_count,
                  MAX(cl.created_at) AS last_chat_at
             FROM user_fingerprints uf
        LEFT JOIN chat_logs cl ON cl.fingerprint = uf.fingerprint
            WHERE uf.user_id = %s
         GROUP BY uf.fingerprint, uf.bound_at
         ORDER BY uf.bound_at DESC""",
        (user_id,),
    )
    passkeys = db_query(
        """SELECT id, device_name, backed_up, created_at, last_used_at
             FROM user_passkeys
            WHERE user_id = %s
         ORDER BY created_at DESC""",
        (user_id,),
    )
    chat_summary = db_query_one(
        """SELECT COUNT(cl.id)::int AS chat_log_count,
                  MIN(cl.created_at) AS first_chat_at,
                  MAX(cl.created_at) AS last_chat_at
             FROM user_fingerprints uf
        LEFT JOIN chat_logs cl ON cl.fingerprint = uf.fingerprint
            WHERE uf.user_id = %s""",
        (user_id,),
    )
    return {
        "user": user,
        "fingerprints": fingerprints,
        "passkeys": passkeys,
        "chat_summary": chat_summary or {},
    }


@router.patch("/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def rename_admin_user(user_id: int, request: AdminUserRenameRequest):
    """Rename a user account. Username uniqueness is enforced by the DB."""
    username = _normalize_account_username(request.username)
    try:
        row = db_query_one(
            """UPDATE users
                  SET username = %s
                WHERE id = %s
            RETURNING id, username, is_admin, created_at, last_login_at,
                      (password_hash IS NOT NULL) AS has_password""",
            (username, user_id),
        )
    except Exception as exc:
        if getattr(exc, "pgcode", "") == "23505":
            raise HTTPException(status_code=409, detail="username already exists")
        raise
    if not row:
        raise HTTPException(status_code=404, detail="user not found")
    return {"user": row}


@router.post("/admin/users/{source_user_id}/merge", dependencies=[Depends(require_admin)])
async def merge_admin_user(source_user_id: int, request: AdminUserMergeRequest):
    """Merge a source account into a target account.

    Fingerprints and passkeys move to the target. Existing chat_logs remain
    fingerprint-owned, so the target account can see the merged history through
    the moved fingerprints.
    """
    target_user_id = request.target_user_id
    if source_user_id == target_user_id:
        raise HTTPException(status_code=400, detail="source and target users must differ")
    result = db_query_one(
        """WITH source_user AS (
                 SELECT id, username, is_admin, last_login_at
                   FROM users
                  WHERE id = %s
             ),
             target_user AS (
                 SELECT id, username, is_admin, last_login_at
                   FROM users
                  WHERE id = %s
             ),
             source_fingerprints AS (
                 SELECT fingerprint, bound_at
                   FROM user_fingerprints
                  WHERE user_id = %s
             ),
             inserted_fingerprints AS (
                 INSERT INTO user_fingerprints (user_id, fingerprint, bound_at)
                 SELECT %s, fingerprint, bound_at FROM source_fingerprints
                 WHERE EXISTS (SELECT 1 FROM source_user)
                   AND EXISTS (SELECT 1 FROM target_user)
                   AND COALESCE((SELECT is_admin FROM source_user), TRUE) = FALSE
                   AND COALESCE((SELECT is_admin FROM target_user), TRUE) = FALSE
                 ON CONFLICT (user_id, fingerprint) DO NOTHING
                 RETURNING fingerprint
             ),
             deleted_source_fingerprints AS (
                 DELETE FROM user_fingerprints
                  WHERE user_id = %s
                    AND EXISTS (SELECT 1 FROM target_user)
                    AND COALESCE((SELECT is_admin FROM source_user), TRUE) = FALSE
                    AND COALESCE((SELECT is_admin FROM target_user), TRUE) = FALSE
                 RETURNING fingerprint
             ),
             moved_passkeys AS (
                 UPDATE user_passkeys
                    SET user_id = %s
                  WHERE user_id = %s
                    AND EXISTS (SELECT 1 FROM source_user)
                    AND EXISTS (SELECT 1 FROM target_user)
                    AND COALESCE((SELECT is_admin FROM source_user), TRUE) = FALSE
                    AND COALESCE((SELECT is_admin FROM target_user), TRUE) = FALSE
                 RETURNING id
             ),
             updated_target AS (
                 UPDATE users u
                    SET last_login_at = CASE
                        WHEN u.last_login_at IS NULL THEN s.last_login_at
                        WHEN s.last_login_at IS NOT NULL AND s.last_login_at > u.last_login_at THEN s.last_login_at
                        ELSE u.last_login_at
                    END
                   FROM source_user s
                  WHERE u.id = %s
                    AND EXISTS (SELECT 1 FROM target_user)
                    AND COALESCE(s.is_admin, TRUE) = FALSE
                    AND COALESCE((SELECT is_admin FROM target_user), TRUE) = FALSE
                 RETURNING u.id
             ),
             deleted_source AS (
                 DELETE FROM users
                  WHERE id = %s
                    AND EXISTS (SELECT 1 FROM target_user)
                    AND COALESCE((SELECT is_admin FROM source_user), TRUE) = FALSE
                    AND COALESCE((SELECT is_admin FROM target_user), TRUE) = FALSE
                 RETURNING id
             )
             SELECT
                 (SELECT COUNT(*)::int FROM source_user) AS source_found,
                 (SELECT COUNT(*)::int FROM target_user) AS target_found,
                 (SELECT is_admin FROM source_user) AS source_is_admin,
                 (SELECT is_admin FROM target_user) AS target_is_admin,
                 (SELECT username FROM source_user) AS source_username,
                 (SELECT username FROM target_user) AS target_username,
                 (SELECT COUNT(*)::int FROM inserted_fingerprints) AS inserted_fingerprints,
                 (SELECT COUNT(*)::int FROM deleted_source_fingerprints) AS moved_fingerprints,
                 (SELECT COUNT(*)::int FROM moved_passkeys) AS moved_passkeys,
                 (SELECT COUNT(*)::int FROM updated_target) AS updated_target,
                 (SELECT COUNT(*)::int FROM deleted_source) AS deleted_source""",
        (
            source_user_id,
            target_user_id,
            source_user_id,
            target_user_id,
            source_user_id,
            target_user_id,
            source_user_id,
            target_user_id,
            source_user_id,
        ),
    )
    if not result or not result["source_found"]:
        raise HTTPException(status_code=404, detail="source user not found")
    if not result["target_found"]:
        raise HTTPException(status_code=404, detail="target user not found")
    if result["source_is_admin"]:
        raise HTTPException(status_code=400, detail="admin users cannot be merged here")
    if result["target_is_admin"]:
        raise HTTPException(status_code=400, detail="regular users cannot be merged into admin users here")
    return {"merge": result}


@router.delete("/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def delete_admin_user(
    user_id: int,
    confirm_username: str = Query(..., min_length=1, max_length=30),
):
    """Delete a regular user account. Chat logs are retained by fingerprint."""
    result = db_query_one(
        """WITH target_user AS (
                 SELECT id, username, is_admin
                   FROM users
                  WHERE id = %s AND username = %s
             ),
             target_fingerprints AS (
                 SELECT fingerprint FROM user_fingerprints WHERE user_id = %s
             ),
             chat_count AS (
                 SELECT COUNT(*)::int AS n
                   FROM chat_logs
                  WHERE fingerprint IN (SELECT fingerprint FROM target_fingerprints)
             ),
             deleted_user AS (
                 DELETE FROM users
                  WHERE id = %s
                    AND username = %s
                    AND COALESCE((SELECT is_admin FROM target_user), TRUE) = FALSE
                 RETURNING id, username
             )
             SELECT
                 (SELECT COUNT(*)::int FROM target_user) AS found,
                 (SELECT COALESCE(is_admin, FALSE) FROM target_user) AS is_admin,
                 (SELECT n FROM chat_count) AS retained_chat_logs,
                 (SELECT COUNT(*)::int FROM deleted_user) AS deleted""",
        (user_id, confirm_username, user_id, user_id, confirm_username),
    )
    if not result or not result["found"]:
        raise HTTPException(status_code=404, detail="user not found or confirmation mismatch")
    if result["is_admin"]:
        raise HTTPException(status_code=400, detail="admin users cannot be deleted here")
    if not result["deleted"]:
        raise HTTPException(status_code=400, detail="user was not deleted")
    return {"deleted": True, "retained_chat_logs": result["retained_chat_logs"]}
