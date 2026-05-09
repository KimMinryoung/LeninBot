"""Best-effort x402 payment audit ledger."""

from __future__ import annotations

import json
import logging
from typing import Any

from db import execute

logger = logging.getLogger(__name__)


def ensure_x402_ledger_table() -> None:
    """Create the x402 audit table.

    This is intended for explicit schema migrations, not hot-path calls.
    """
    execute("""
        CREATE TABLE IF NOT EXISTS x402_payment_attempts (
            id              SERIAL PRIMARY KEY,
            direction       VARCHAR(16) NOT NULL,
            stage           VARCHAR(40) NOT NULL,
            status          VARCHAR(24) NOT NULL,
            resource        TEXT,
            method          VARCHAR(12),
            payer           TEXT,
            pay_to          TEXT,
            amount_atomic   BIGINT,
            asset           TEXT,
            network         TEXT,
            tx_hash         TEXT,
            gas_used        BIGINT,
            http_status     INTEGER,
            error           TEXT,
            metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    execute("""
        CREATE INDEX IF NOT EXISTS idx_x402_payment_attempts_created
        ON x402_payment_attempts(created_at DESC)
    """)
    execute("""
        CREATE INDEX IF NOT EXISTS idx_x402_payment_attempts_status_created
        ON x402_payment_attempts(status, created_at DESC)
    """)
    execute("""
        CREATE INDEX IF NOT EXISTS idx_x402_payment_attempts_tx_hash
        ON x402_payment_attempts(tx_hash)
        WHERE tx_hash IS NOT NULL
    """)


def record_x402_attempt(
    *,
    direction: str,
    stage: str,
    status: str,
    resource: str | None = None,
    method: str | None = None,
    payer: str | None = None,
    pay_to: str | None = None,
    amount_atomic: int | None = None,
    asset: str | None = None,
    network: str | None = None,
    tx_hash: str | None = None,
    gas_used: int | None = None,
    http_status: int | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record one x402 client/server event without exposing signatures."""
    try:
        execute(
            """
            INSERT INTO x402_payment_attempts (
                direction, stage, status, resource, method, payer, pay_to,
                amount_atomic, asset, network, tx_hash, gas_used, http_status,
                error, metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s::jsonb
            )
            """,
            (
                direction,
                stage,
                status,
                resource,
                method,
                payer,
                pay_to,
                amount_atomic,
                asset,
                network,
                tx_hash,
                gas_used,
                http_status,
                error,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
    except Exception as exc:
        logger.warning("x402 ledger write failed: %s", exc)
