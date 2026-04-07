"""crypto_wallet.x402 — x402 payment protocol (Coinbase / x402-foundation).

Implements the EVM `exact` scheme over USDC on Base mainnet using ERC-3009
`transferWithAuthorization`. Both client (sign + retry) and server
(verify + settle) helpers live here so leninbot can play either side
of a self-loop demo.

Spec: https://github.com/x402-foundation/x402 (scheme_exact_evm.md)

Wire format we emit:
  402 response body  : { x402Version, error, accepts: [PaymentRequirements] }
  PAYMENT-SIGNATURE  : base64( JSON({ x402Version, scheme, network, resource,
                                      payload: { authorization, signature } }) )
  PAYMENT-RESPONSE   : base64( JSON({ tx_hash, status, gas_used }) )
"""

import asyncio
import base64
import json
import logging
import os
import secrets
import time
from pathlib import Path

import httpx
from eth_account import Account
from eth_account.messages import encode_typed_data
from web3 import Web3

from crypto_wallet.wallet import get_addresses

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────
X402_VERSION = 2
SCHEME_EXACT = "exact"
NETWORK_BASE = "eip155:8453"  # CAIP-2
BASE_CHAIN_ID = 8453
USDC_DECIMALS = 6

USDC_BASE = Web3.to_checksum_address("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")
USDC_NAME = "USD Coin"  # MUST match on-chain EIP-712 domain (verified against DOMAIN_SEPARATOR)
USDC_VERSION = "2"

USDC_DOMAIN = {
    "name": USDC_NAME,
    "version": USDC_VERSION,
    "chainId": BASE_CHAIN_ID,
    "verifyingContract": USDC_BASE,
}

EIP712_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "TransferWithAuthorization": [
        {"name": "from", "type": "address"},
        {"name": "to", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "validAfter", "type": "uint256"},
        {"name": "validBefore", "type": "uint256"},
        {"name": "nonce", "type": "bytes32"},
    ],
}

# ABI: only what x402 needs (settlement). transactions.py has the rest.
USDC_X402_ABI = [
    {
        "name": "transferWithAuthorization",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "from", "type": "address"},
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "validAfter", "type": "uint256"},
            {"name": "validBefore", "type": "uint256"},
            {"name": "nonce", "type": "bytes32"},
            {"name": "v", "type": "uint8"},
            {"name": "r", "type": "bytes32"},
            {"name": "s", "type": "bytes32"},
        ],
        "outputs": [],
    },
]

# Hard cap (USDC) — separate from `WALLET_AUTO_LIMIT_USD` since x402 calls
# happen at much smaller scale and want a tighter ceiling.
PAY_AND_FETCH_MAX_USDC = float(os.environ.get("X402_MAX_USDC_PER_CALL", "0.05"))


# ── Hex helpers ─────────────────────────────────────────────────────

def _to_0x(b: bytes) -> str:
    return "0x" + b.hex()


def _from_0x(s: str) -> bytes:
    return bytes.fromhex(s[2:] if s.startswith("0x") else s)


# ── EIP-712 typed-data construction ─────────────────────────────────

def _build_typed_data(authorization: dict) -> dict:
    """Build the dict that goes into encode_typed_data() for signing/recovery."""
    return {
        "types": EIP712_TYPES,
        "primaryType": "TransferWithAuthorization",
        "domain": USDC_DOMAIN,
        "message": {
            "from": Web3.to_checksum_address(authorization["from"]),
            "to": Web3.to_checksum_address(authorization["to"]),
            "value": int(authorization["value"]),
            "validAfter": int(authorization["validAfter"]),
            "validBefore": int(authorization["validBefore"]),
            "nonce": _from_0x(authorization["nonce"]),
        },
    }


# ── Server side: build 402 response ─────────────────────────────────

def build_payment_requirements(
    pay_to: str,
    amount_atomic: int,
    resource: str,
    description: str = "",
    mime_type: str = "application/json",
    max_timeout_seconds: int = 60,
) -> dict:
    """Construct one PaymentRequirements entry for the 402 response 'accepts' list."""
    return {
        "scheme": SCHEME_EXACT,
        "network": NETWORK_BASE,
        "maxAmountRequired": str(amount_atomic),
        "resource": resource,
        "description": description,
        "mimeType": mime_type,
        "payTo": Web3.to_checksum_address(pay_to),
        "maxTimeoutSeconds": max_timeout_seconds,
        "asset": USDC_BASE,
        "extra": {
            "name": USDC_NAME,
            "version": USDC_VERSION,
        },
    }


def build_402_body(requirements: dict, error_msg: str = "PAYMENT-SIGNATURE header required") -> dict:
    return {
        "x402Version": X402_VERSION,
        "error": error_msg,
        "accepts": [requirements],
    }


# ── Client side: sign payment ───────────────────────────────────────

def _load_privkey_hex() -> str:
    """Read ETH private key from systemd credential directory. Caller must del after use."""
    cred = os.environ.get("CREDENTIALS_DIRECTORY")
    if not cred:
        raise RuntimeError("CREDENTIALS_DIRECTORY not set — cannot sign x402 payment")
    path = Path(cred) / "eth.privkey"
    if not path.exists():
        raise RuntimeError("eth.privkey not found in CREDENTIALS_DIRECTORY")
    return "0x" + path.read_text().strip()


def sign_payment(requirements: dict, max_atomic: int) -> dict:
    """Sign an EIP-3009 authorization for the given requirements.

    Enforces scheme/network/asset compatibility and the amount cap. Loads
    the private key just-in-time and deletes it from local scope after use.
    Returns: { authorization: {...}, signature: "0x..." }
    """
    if requirements.get("scheme") != SCHEME_EXACT:
        raise ValueError(f"Unsupported scheme: {requirements.get('scheme')}")
    if requirements.get("network") != NETWORK_BASE:
        raise ValueError(f"Unsupported network: {requirements.get('network')}")
    asset = Web3.to_checksum_address(requirements.get("asset", ""))
    if asset != USDC_BASE:
        raise ValueError(f"Unsupported asset: {asset} (expected USDC on Base)")

    amount_str = requirements.get("maxAmountRequired") or requirements.get("amount")
    if amount_str is None:
        raise ValueError("PaymentRequirements missing maxAmountRequired/amount")
    amount = int(amount_str)
    if amount > max_atomic:
        raise ValueError(f"Required amount {amount} exceeds cap {max_atomic} (atomic USDC)")
    if amount <= 0:
        raise ValueError(f"Required amount must be positive (got {amount})")

    addrs = get_addresses()
    payer = addrs.get("base")
    if not payer:
        raise RuntimeError("Base wallet address not available — cannot sign payment")

    pay_to = Web3.to_checksum_address(requirements["payTo"])

    now = int(time.time())
    valid_after = max(now - 60, 0)  # 1min skew tolerance
    timeout = int(requirements.get("maxTimeoutSeconds", 60))
    valid_before = now + max(timeout, 60)
    nonce = secrets.token_bytes(32)

    authorization = {
        "from": Web3.to_checksum_address(payer),
        "to": pay_to,
        "value": str(amount),
        "validAfter": str(valid_after),
        "validBefore": str(valid_before),
        "nonce": _to_0x(nonce),
    }

    signable = encode_typed_data(full_message=_build_typed_data(authorization))
    pk_hex = _load_privkey_hex()
    try:
        signed = Account.sign_message(signable, private_key=pk_hex)
    finally:
        del pk_hex

    return {
        "authorization": authorization,
        "signature": _to_0x(bytes(signed.signature)),
    }


# ── Header encoding ─────────────────────────────────────────────────

def encode_payment_header(requirements: dict, payload: dict, resource: str | None = None) -> str:
    """Build the PAYMENT-SIGNATURE header value."""
    obj = {
        "x402Version": X402_VERSION,
        "scheme": requirements["scheme"],
        "network": requirements["network"],
        "payload": payload,
    }
    if resource:
        obj["resource"] = resource
    return base64.b64encode(json.dumps(obj, separators=(",", ":")).encode()).decode()


def decode_payment_header(header_value: str) -> dict:
    """Decode the PAYMENT-SIGNATURE header value into a dict."""
    return json.loads(base64.b64decode(header_value.encode()).decode())


def encode_settlement_header(settlement: dict) -> str:
    """Build the PAYMENT-RESPONSE header value with on-chain settlement details."""
    return base64.b64encode(json.dumps(settlement, separators=(",", ":")).encode()).decode()


# ── Server side: verify signature ───────────────────────────────────

def verify_payment(payment: dict, requirements: dict) -> str:
    """Verify a decoded PAYMENT-SIGNATURE payload against the requirements
    the server demanded. Returns the recovered signer address. Raises on any
    mismatch (amount, recipient, scheme, network, signature, validity).
    """
    if payment.get("x402Version") != X402_VERSION:
        raise ValueError(f"x402Version mismatch (got {payment.get('x402Version')})")
    if payment.get("scheme") != requirements["scheme"]:
        raise ValueError(f"scheme mismatch")
    if payment.get("network") != requirements["network"]:
        raise ValueError(f"network mismatch")

    payload = payment.get("payload") or {}
    auth = payload.get("authorization") or {}
    sig = payload.get("signature")
    if not auth or not sig:
        raise ValueError("payload missing authorization or signature")

    # Check the signed amount/recipient match what the server demanded
    demanded = int(requirements.get("maxAmountRequired") or requirements.get("amount"))
    if int(auth["value"]) != demanded:
        raise ValueError(f"amount mismatch: signed {auth['value']} vs demanded {demanded}")
    if Web3.to_checksum_address(auth["to"]) != Web3.to_checksum_address(requirements["payTo"]):
        raise ValueError("payTo mismatch")

    # Validity window
    now = int(time.time())
    if now < int(auth["validAfter"]):
        raise ValueError("authorization not yet valid")
    if now > int(auth["validBefore"]):
        raise ValueError("authorization expired")

    # Recover signer
    signable = encode_typed_data(full_message=_build_typed_data(auth))
    recovered = Account.recover_message(signable, signature=sig)
    if recovered.lower() != auth["from"].lower():
        raise ValueError(f"signature recovery mismatch: {recovered} vs claimed {auth['from']}")
    return recovered


# ── Server side: settle on-chain ────────────────────────────────────

async def settle_payment(payload: dict) -> dict:
    """Submit USDC.transferWithAuthorization on Base mainnet using our wallet
    as the gas-paying settler. Returns: { tx_hash, status, gas_used }.

    The settler must have ETH for gas. The signer (auth["from"]) must hold
    sufficient USDC; if not, the on-chain call reverts.
    """
    auth = payload["authorization"]
    sig_bytes = _from_0x(payload["signature"])
    if len(sig_bytes) != 65:
        raise ValueError(f"signature length {len(sig_bytes)} != 65")
    r = sig_bytes[:32]
    s = sig_bytes[32:64]
    v = sig_bytes[64]
    if v < 27:
        v += 27

    nonce_bytes = _from_0x(auth["nonce"])

    # Reuse infra from transactions.py (web3 instance + sign+send helper)
    from crypto_wallet.transactions import _get_w3, _build_sign_send

    w3 = _get_w3()
    usdc = w3.eth.contract(address=USDC_BASE, abi=USDC_X402_ABI)

    addrs = get_addresses()
    settler = Web3.to_checksum_address(addrs.get("base", ""))
    if not settler:
        raise RuntimeError("Base wallet address not available — cannot settle")

    eth_nonce = await asyncio.to_thread(w3.eth.get_transaction_count, settler)
    gas_price = await asyncio.to_thread(lambda: w3.eth.gas_price)

    tx = usdc.functions.transferWithAuthorization(
        Web3.to_checksum_address(auth["from"]),
        Web3.to_checksum_address(auth["to"]),
        int(auth["value"]),
        int(auth["validAfter"]),
        int(auth["validBefore"]),
        nonce_bytes,
        v, r, s,
    ).build_transaction({
        "from": settler,
        "value": 0,
        "nonce": eth_nonce,
        "gasPrice": gas_price,
        "chainId": BASE_CHAIN_ID,
    })

    receipt = await _build_sign_send(tx)
    return {
        "tx_hash": _to_0x(bytes(receipt["transactionHash"])),
        "status": "success" if receipt.get("status") == 1 else "failed",
        "gas_used": int(receipt.get("gasUsed", 0)),
    }


# ── Top-level client flow: pay_and_fetch ────────────────────────────

async def pay_and_fetch(url: str, max_usdc: float = PAY_AND_FETCH_MAX_USDC) -> dict:
    """GET `url`. If the server returns 402, sign an x402 payment within
    `max_usdc` and retry. Returns a dict describing the outcome.

    The hard amount cap (in atomic USDC) = max_usdc * 1e6, capped further by
    the per-call ceiling from sign_payment.
    """
    max_atomic = int(max_usdc * (10 ** USDC_DECIMALS))

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        try:
            resp1 = await client.get(url)
        except Exception as e:
            return {"status": "error", "stage": "initial_get", "error": str(e)}

        if resp1.status_code == 200:
            return {
                "status": "ok",
                "paid": False,
                "http_status": 200,
                "body": resp1.text,
            }
        if resp1.status_code != 402:
            return {
                "status": "error",
                "stage": "initial_get",
                "http_status": resp1.status_code,
                "body": resp1.text[:500],
            }

        # Parse 402 body
        try:
            req_body = resp1.json()
        except Exception:
            return {"status": "error", "stage": "parse_402", "body": resp1.text[:500]}

        accepts = req_body.get("accepts") or req_body.get("accepted") or []
        if not accepts:
            return {"status": "error", "stage": "parse_402", "error": "no accepts list"}

        # Pick first compatible (exact / Base mainnet)
        chosen = None
        for opt in accepts:
            if opt.get("scheme") == SCHEME_EXACT and opt.get("network") == NETWORK_BASE:
                chosen = opt
                break
        if chosen is None:
            return {
                "status": "error",
                "stage": "select_requirement",
                "error": "no exact/eip155:8453 option",
                "accepts": accepts,
            }

        # Sign (enforces cap)
        try:
            payload = await asyncio.to_thread(sign_payment, chosen, max_atomic)
        except (ValueError, RuntimeError) as e:
            return {"status": "rejected", "stage": "sign", "error": str(e)}

        amount_atomic = int(chosen.get("maxAmountRequired") or chosen.get("amount"))
        logger.info(
            "[x402] Paying %s atomic USDC (%.6f USDC) to %s for %s",
            amount_atomic, amount_atomic / 10**USDC_DECIMALS, chosen["payTo"], url,
        )

        header_val = encode_payment_header(chosen, payload, resource=url)

        try:
            resp2 = await client.get(url, headers={"PAYMENT-SIGNATURE": header_val})
        except Exception as e:
            return {"status": "error", "stage": "retry_get", "error": str(e)}

        # Decode settlement details if present
        settlement = None
        ph = resp2.headers.get("PAYMENT-RESPONSE")
        if ph:
            try:
                settlement = json.loads(base64.b64decode(ph.encode()).decode())
            except Exception:
                settlement = {"raw": ph}

        if resp2.status_code != 200:
            return {
                "status": "error",
                "stage": "retry_get",
                "http_status": resp2.status_code,
                "body": resp2.text[:500],
                "settlement": settlement,
            }

        return {
            "status": "ok",
            "paid": True,
            "http_status": 200,
            "amount_usdc": amount_atomic / 10**USDC_DECIMALS,
            "amount_atomic": amount_atomic,
            "settlement": settlement,
            "body": resp2.text,
        }


# ── Tool definition (Anthropic API format) ─────────────────────────

PAY_AND_FETCH_TOOL = {
    "name": "pay_and_fetch",
    "description": (
        "Fetch a URL that requires an x402 payment. Performs the full client "
        "round-trip: GET → 402 PaymentRequirements → sign EIP-3009 USDC "
        "transferWithAuthorization → retry GET with PAYMENT-SIGNATURE header. "
        "Hard-capped per call (default $0.05 USDC). Use only when a normal "
        "fetch_url returns 402 or you specifically need a paid resource. "
        "Returns the response body plus settlement details (tx hash, gas used). "
        "DEMO ENDPOINT: http://localhost:8000/x402-demo/quote — leninbot's own "
        "API exposes a self-loop x402 route that demands 0.001 USDC and returns "
        "an aphorism. Use that URL whenever the user asks for an x402 demo "
        "without naming a specific provider."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch (must support x402 / return 402 with PaymentRequirements).",
            },
            "max_usdc": {
                "type": "number",
                "description": (
                    "Hard cap on the payment amount in USDC. Defaults to 0.05. "
                    "Anything larger is rejected without signing."
                ),
                "default": 0.05,
            },
        },
        "required": ["url"],
    },
}


async def _exec_pay_and_fetch(url: str, max_usdc: float = PAY_AND_FETCH_MAX_USDC) -> str:
    """Tool handler: invoke pay_and_fetch and format result for the LLM."""
    try:
        result = await pay_and_fetch(url, max_usdc=max_usdc)
    except Exception as e:
        logger.error("pay_and_fetch error: %s", e)
        return f"x402 error: {e}"

    status = result.get("status")
    if status == "ok" and result.get("paid"):
        settlement = result.get("settlement") or {}
        return (
            f"x402 payment successful.\n"
            f"  URL: {url}\n"
            f"  Paid: {result.get('amount_usdc'):.6f} USDC ({result.get('amount_atomic')} atomic)\n"
            f"  TX: {settlement.get('tx_hash', '(unknown)')}\n"
            f"  Gas used: {settlement.get('gas_used', '(unknown)')}\n"
            f"  Response body:\n{result.get('body', '')[:2000]}"
        )
    if status == "ok":
        return (
            f"No payment required (server returned 200 without 402).\n"
            f"  URL: {url}\n"
            f"  Response body:\n{result.get('body', '')[:2000]}"
        )
    if status == "rejected":
        return (
            f"x402 payment rejected by client cap.\n"
            f"  URL: {url}\n"
            f"  Reason: {result.get('error')}"
        )
    # error
    return (
        f"x402 fetch failed.\n"
        f"  URL: {url}\n"
        f"  Stage: {result.get('stage', '(unknown)')}\n"
        f"  Error: {result.get('error') or result.get('body', '(none)')}"
    )


PAY_AND_FETCH_TOOL_HANDLER = _exec_pay_and_fetch
