"""crypto_wallet.transactions — Base L2 트랜잭션 (web3.py 기반)

ETH → USDC 스왑 (Uniswap V3) 및 USDC 전송 기능.
개인키: CREDENTIALS_DIRECTORY에서 로드, 서명 후 즉시 삭제.
금액 분기: AUTO_LIMIT_USD 이하는 자율 실행, 초과 시 거부 + 안내.
"""

import asyncio
import logging
import os
from pathlib import Path

import httpx
from web3 import Web3

from crypto_wallet.wallet import get_addresses, _rpc

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════════════
BASE_CHAIN_ID = 8453
WETH = Web3.to_checksum_address("0x4200000000000000000000000000000000000006")
USDC = Web3.to_checksum_address("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")
UNISWAP_V3_ROUTER = Web3.to_checksum_address("0x2626664c2603336E57B271c5C0b26F421741e481")
USDC_DECIMALS = 6
POOL_FEE = 500  # 0.05% — ETH/USDC 메인 풀 on Base

AUTO_LIMIT_USD = float(os.environ.get("WALLET_AUTO_LIMIT_USD", "10.0"))

# ── ABI 정의 ──────────────────────────────────────────────────────────

USDC_ABI = [
    {
        "name": "transfer",
        "type": "function",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
]

# SwapRouter02 — exactInputSingle (no deadline field)
SWAP_ROUTER_ABI = [
    {
        "name": "exactInputSingle",
        "type": "function",
        "inputs": [
            {
                "name": "params",
                "type": "tuple",
                "components": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "recipient", "type": "address"},
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "amountOutMinimum", "type": "uint256"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
            }
        ],
        "outputs": [{"name": "amountOut", "type": "uint256"}],
        "stateMutability": "payable",
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Web3 인스턴스 (lazy init)
# ═══════════════════════════════════════════════════════════════════════
_w3: Web3 | None = None


def _get_w3() -> Web3:
    global _w3
    if _w3 is None:
        _w3 = Web3(Web3.HTTPProvider(_rpc("base")))
    return _w3


# ═══════════════════════════════════════════════════════════════════════
# 개인키 로드 (사용 후 즉시 삭제)
# ═══════════════════════════════════════════════════════════════════════

def _load_privkey_hex() -> str:
    """CREDENTIALS_DIRECTORY에서 ETH 개인키 hex 로드. 호출자가 사용 후 del 할 것."""
    cred = os.environ.get("CREDENTIALS_DIRECTORY")
    if not cred:
        raise RuntimeError("CREDENTIALS_DIRECTORY not set — transaction signing unavailable")
    path = Path(cred) / "eth.privkey"
    if not path.exists():
        raise RuntimeError("eth.privkey not found in CREDENTIALS_DIRECTORY")
    return "0x" + path.read_text().strip()


# ═══════════════════════════════════════════════════════════════════════
# ERC-20 helpers
# ═══════════════════════════════════════════════════════════════════════

async def get_usdc_balance(address: str) -> float:
    """USDC 잔액 조회 (Base)."""
    w3 = _get_w3()
    usdc = w3.eth.contract(address=USDC, abi=USDC_ABI)
    raw = await asyncio.to_thread(usdc.functions.balanceOf(Web3.to_checksum_address(address)).call)
    return raw / (10 ** USDC_DECIMALS)


async def _get_eth_price_usd() -> float:
    """CoinGecko에서 ETH/USD 가격 조회."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "ethereum", "vs_currencies": "usd"},
            )
            resp.raise_for_status()
            return resp.json()["ethereum"]["usd"]
    except Exception:
        return 1800.0  # fallback


# ═══════════════════════════════════════════════════════════════════════
# 트랜잭션 빌드 + 서명 + 전송 (web3.py)
# ═══════════════════════════════════════════════════════════════════════

async def _build_sign_send(tx: dict) -> dict:
    """트랜잭션 빌드 → 서명 → 전송 → 영수증 대기. 개인키 즉시 삭제."""
    w3 = _get_w3()
    pk_hex = _load_privkey_hex()
    try:
        signed = await asyncio.to_thread(w3.eth.account.sign_transaction, tx, pk_hex)
    finally:
        del pk_hex

    tx_hash = await asyncio.to_thread(w3.eth.send_raw_transaction, signed.raw_transaction)
    receipt = await asyncio.to_thread(w3.eth.wait_for_transaction_receipt, tx_hash, timeout=120)
    return receipt


# ═══════════════════════════════════════════════════════════════════════
# 스왑: ETH → USDC (Uniswap V3 SwapRouter02)
# ═══════════════════════════════════════════════════════════════════════

async def swap_eth_to_usdc(amount_eth: float, slippage_pct: float = 1.0) -> dict:
    """ETH → USDC 스왑 실행."""
    addrs = get_addresses()
    my_addr = Web3.to_checksum_address(addrs.get("base", ""))
    if not my_addr:
        raise RuntimeError("Base wallet address not available")

    amount_wei = Web3.to_wei(amount_eth, "ether")

    # 금액 체크
    eth_price = await _get_eth_price_usd()
    usd_value = amount_eth * eth_price
    if usd_value > AUTO_LIMIT_USD:
        return {
            "status": "requires_approval",
            "amount_eth": amount_eth,
            "estimated_usd": round(usd_value, 2),
            "limit_usd": AUTO_LIMIT_USD,
            "message": f"${usd_value:.2f} exceeds auto-execution limit (${AUTO_LIMIT_USD:.2f}). Operator approval required.",
        }

    # 최소 수량 계산 (slippage)
    estimated_usdc = usd_value
    min_usdc_out = int(estimated_usdc * (1 - slippage_pct / 100) * (10 ** USDC_DECIMALS))

    # 컨트랙트 호출 빌드
    w3 = _get_w3()
    router = w3.eth.contract(address=UNISWAP_V3_ROUTER, abi=SWAP_ROUTER_ABI)
    swap_params = (WETH, USDC, POOL_FEE, my_addr, amount_wei, min_usdc_out, 0)

    nonce = await asyncio.to_thread(w3.eth.get_transaction_count, my_addr)
    gas_price = await asyncio.to_thread(w3.eth.gas_price.__int__) if hasattr(w3.eth.gas_price, '__int__') else await asyncio.to_thread(lambda: w3.eth.gas_price)

    tx = router.functions.exactInputSingle(swap_params).build_transaction({
        "from": my_addr,
        "value": amount_wei,
        "nonce": nonce,
        "gasPrice": gas_price,
        "chainId": BASE_CHAIN_ID,
    })

    # 서명 + 전송
    receipt = await _build_sign_send(tx)

    success = receipt.get("status") == 1
    usdc_after = await get_usdc_balance(my_addr)

    return {
        "status": "success" if success else "failed",
        "tx_hash": receipt["transactionHash"].hex(),
        "amount_eth": amount_eth,
        "usdc_balance_after": usdc_after,
        "gas_used": receipt.get("gasUsed", 0),
    }


# ═══════════════════════════════════════════════════════════════════════
# 전송: USDC → 수신자
# ═══════════════════════════════════════════════════════════════════════

async def transfer_usdc(to_address: str, amount_usdc: float) -> dict:
    """USDC 전송 실행."""
    addrs = get_addresses()
    my_addr = Web3.to_checksum_address(addrs.get("base", ""))
    if not my_addr:
        raise RuntimeError("Base wallet address not available")

    to_addr = Web3.to_checksum_address(to_address)

    # 금액 체크
    if amount_usdc > AUTO_LIMIT_USD:
        return {
            "status": "requires_approval",
            "to": to_address,
            "amount_usdc": amount_usdc,
            "limit_usd": AUTO_LIMIT_USD,
            "message": f"${amount_usdc:.2f} exceeds auto-execution limit (${AUTO_LIMIT_USD:.2f}). Operator approval required.",
        }

    # 잔액 확인
    balance = await get_usdc_balance(my_addr)
    if balance < amount_usdc:
        return {
            "status": "insufficient_balance",
            "balance": balance,
            "requested": amount_usdc,
            "message": f"USDC balance ({balance:.2f}) insufficient for transfer ({amount_usdc:.2f}).",
        }

    amount_raw = int(amount_usdc * (10 ** USDC_DECIMALS))

    # 컨트랙트 호출 빌드
    w3 = _get_w3()
    usdc = w3.eth.contract(address=USDC, abi=USDC_ABI)

    nonce = await asyncio.to_thread(w3.eth.get_transaction_count, my_addr)
    gas_price = await asyncio.to_thread(lambda: w3.eth.gas_price)

    tx = usdc.functions.transfer(to_addr, amount_raw).build_transaction({
        "from": my_addr,
        "value": 0,
        "nonce": nonce,
        "gasPrice": gas_price,
        "chainId": BASE_CHAIN_ID,
    })

    # 서명 + 전송
    receipt = await _build_sign_send(tx)

    success = receipt.get("status") == 1
    usdc_after = await get_usdc_balance(my_addr)

    return {
        "status": "success" if success else "failed",
        "tx_hash": receipt["transactionHash"].hex(),
        "to": to_address,
        "amount_usdc": amount_usdc,
        "usdc_balance_after": usdc_after,
        "gas_used": receipt.get("gasUsed", 0),
    }


# ═══════════════════════════════════════════════════════════════════════
# Tool 정의
# ═══════════════════════════════════════════════════════════════════════

SWAP_TOOL = {
    "name": "swap_eth_to_usdc",
    "description": (
        "Swap ETH to USDC on Base L2 via Uniswap V3. "
        "Auto-executes for amounts under $10; requires operator approval above that. "
        "Use when asked to convert ETH to stablecoin."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "amount_eth": {
                "type": "number",
                "description": "Amount of ETH to swap (e.g. 0.005).",
            },
            "slippage_pct": {
                "type": "number",
                "description": "Max slippage percentage (default 1.0).",
                "default": 1.0,
            },
        },
        "required": ["amount_eth"],
    },
}

TRANSFER_TOOL = {
    "name": "transfer_usdc",
    "description": (
        "Transfer USDC to an address on Base L2. "
        "Auto-executes for amounts under $10; requires operator approval above that. "
        "Use when asked to send USDC or make a payment."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to_address": {
                "type": "string",
                "description": "Recipient Ethereum/Base address (0x...).",
            },
            "amount_usdc": {
                "type": "number",
                "description": "Amount of USDC to send (e.g. 5.00).",
            },
        },
        "required": ["to_address", "amount_usdc"],
    },
}


async def _exec_swap(amount_eth: float, slippage_pct: float = 1.0) -> str:
    try:
        result = await swap_eth_to_usdc(amount_eth, slippage_pct)
        if result["status"] == "requires_approval":
            return result["message"]
        if result["status"] == "success":
            return (
                f"Swap complete.\n"
                f"  TX: {result['tx_hash']}\n"
                f"  Swapped: {result['amount_eth']} ETH\n"
                f"  USDC balance: {result['usdc_balance_after']:.2f}\n"
                f"  Gas used: {result['gas_used']}"
            )
        return f"Swap failed. TX: {result.get('tx_hash', 'N/A')}"
    except Exception as e:
        logger.error("swap_eth_to_usdc failed: %s", e)
        return f"Swap error: {e}"


async def _exec_transfer(to_address: str, amount_usdc: float) -> str:
    try:
        result = await transfer_usdc(to_address, amount_usdc)
        if result["status"] == "requires_approval":
            return result["message"]
        if result["status"] == "insufficient_balance":
            return result["message"]
        if result["status"] == "success":
            return (
                f"Transfer complete.\n"
                f"  TX: {result['tx_hash']}\n"
                f"  Sent: {result['amount_usdc']:.2f} USDC to {result['to']}\n"
                f"  USDC balance: {result['usdc_balance_after']:.2f}\n"
                f"  Gas used: {result['gas_used']}"
            )
        return f"Transfer failed. TX: {result.get('tx_hash', 'N/A')}"
    except Exception as e:
        logger.error("transfer_usdc failed: %s", e)
        return f"Transfer error: {e}"


SWAP_TOOL_HANDLER = _exec_swap
TRANSFER_TOOL_HANDLER = _exec_transfer
