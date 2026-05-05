"""x402 demo API routes."""

from __future__ import annotations

import base64
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

router = APIRouter()

# Tiny micropayment demo over USDC on Base mainnet via the x402 'exact' scheme.
# The same wallet pays and receives: net USDC change is zero, only gas is spent.
X402_DEMO_AMOUNT_ATOMIC = 1000  # 0.001 USDC (~$0.001)


@router.get("/x402-demo/quote")
async def x402_demo_quote(request: Request):
    """Return a paid demo aphorism after x402 payment verification."""
    from crypto_wallet import x402
    from crypto_wallet.wallet import get_addresses

    addrs = get_addresses()
    pay_to = addrs.get("base")
    if not pay_to:
        raise HTTPException(status_code=503, detail="Server wallet not configured")

    resource = str(request.url)
    requirements = x402.build_payment_requirements(
        pay_to=pay_to,
        amount_atomic=X402_DEMO_AMOUNT_ATOMIC,
        resource=resource,
        description="Cyber-Lenin x402 demo: pay tiny USDC for an aphorism",
        mime_type="application/json",
    )

    payment_header = request.headers.get("PAYMENT-SIGNATURE")
    if not payment_header:
        body = x402.build_402_body(requirements)
        return Response(
            content=json.dumps(body, ensure_ascii=False),
            status_code=402,
            media_type="application/json",
            headers={
                "PAYMENT-REQUIRED": base64.b64encode(
                    json.dumps(requirements, separators=(",", ":")).encode()
                ).decode(),
            },
        )

    try:
        decoded = x402.decode_payment_header(payment_header)
        signer = x402.verify_payment(decoded, requirements)
    except Exception as exc:
        raise HTTPException(status_code=402, detail=f"x402 verification failed: {exc}") from exc

    try:
        settlement = await x402.settle_payment(decoded["payload"])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"x402 settlement failed: {exc}") from exc

    if settlement.get("status") != "success":
        raise HTTPException(status_code=502, detail=f"settlement tx not success: {settlement}")

    aphorism = (
        "정치란 과학이며 예술이다. 그것은 하늘에서 떨어지는 것이 아니라 "
        "노력과 투쟁을 통해 얻어지는 것이다."
    )
    body = {
        "x402Version": x402.X402_VERSION,
        "message": "결제 검증 통과 — Cyber-Lenin의 격언:",
        "aphorism": aphorism,
        "payer": signer,
        "amount_atomic": int(decoded["payload"]["authorization"]["value"]),
        "tx_hash": settlement["tx_hash"],
        "gas_used": settlement["gas_used"],
    }
    return Response(
        content=json.dumps(body, ensure_ascii=False),
        status_code=200,
        media_type="application/json",
        headers={
            "PAYMENT-RESPONSE": x402.encode_settlement_header(settlement),
        },
    )
