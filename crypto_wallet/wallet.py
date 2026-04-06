"""crypto_wallet.wallet — 주소 도출 + 잔액 조회 (ETH/Base/TRX/SOL)

주소 도출: CREDENTIALS_DIRECTORY에서 개인키를 읽어 주소 계산 후 메모리에서 즉시 삭제.
잔액 조회: 공개 주소만으로 RPC 호출. 개인키 불필요.

RPC endpoints는 .env에서 읽음:
  BASE_RPC_URL    (default: https://mainnet.base.org)
  ETH_RPC_URL     (default: https://ethereum-rpc.publicnode.com)
  TRX_API_URL     (default: https://api.trongrid.io)
  SOL_RPC_URL     (default: https://api.mainnet-beta.solana.com)
"""

import asyncio
import hashlib
import logging
import os
from pathlib import Path

import base58
import httpx
from eth_keys import keys as eth_keys
from eth_utils import keccak
import nacl.signing

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# 주소 캐시 (한 번 도출하면 재사용 — 개인키 재읽기 불필요)
# ═══════════════════════════════════════════════════════════════════════
_addresses: dict[str, str] = {}  # chain -> address


def _cred_dir() -> Path | None:
    d = os.environ.get("CREDENTIALS_DIRECTORY")
    return Path(d) if d else None


def _derive_eth_address(pk_bytes: bytes) -> str:
    priv = eth_keys.PrivateKey(pk_bytes)
    return priv.public_key.to_checksum_address()


def _derive_tron_address(pk_bytes: bytes) -> str:
    priv = eth_keys.PrivateKey(pk_bytes)
    pub_bytes = priv.public_key.to_bytes()
    raw_addr = keccak(pub_bytes)[-20:]
    versioned = b"\x41" + raw_addr
    checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
    return base58.b58encode(versioned + checksum).decode()


def _derive_sol_address(seed: bytes) -> str:
    signing_key = nacl.signing.SigningKey(seed)
    return base58.b58encode(bytes(signing_key.verify_key)).decode()


def get_addresses() -> dict[str, str]:
    """캐시된 주소 반환. 최초 호출 시 credential에서 도출."""
    if _addresses:
        return dict(_addresses)

    cred = _cred_dir()
    if not cred:
        logger.warning("CREDENTIALS_DIRECTORY not set — wallet addresses unavailable")
        return {}

    # ── ETH/Base/TRX (같은 개인키) ────────────────────────────────────
    eth_key_path = cred / "eth.privkey"
    if eth_key_path.exists():
        pk_hex = eth_key_path.read_text().strip()
        pk_bytes = bytes.fromhex(pk_hex)
        _addresses["eth"] = _derive_eth_address(pk_bytes)
        _addresses["base"] = _addresses["eth"]  # 같은 주소
        _addresses["tron"] = _derive_tron_address(pk_bytes)
        # 메모리에서 개인키 즉시 제거
        del pk_hex, pk_bytes
    else:
        logger.warning("eth.privkey not found in %s", cred)

    # ── SOL ────────────────────────────────────────────────────────────
    sol_key_path = cred / "sol.keypair"
    if sol_key_path.exists():
        keypair_b58 = sol_key_path.read_text().strip()
        keypair_bytes = base58.b58decode(keypair_b58)
        seed = keypair_bytes[:32]
        _addresses["sol"] = _derive_sol_address(seed)
        del keypair_b58, keypair_bytes, seed
    else:
        logger.warning("sol.keypair not found in %s", cred)

    return dict(_addresses)


# ═══════════════════════════════════════════════════════════════════════
# RPC endpoints
# ═══════════════════════════════════════════════════════════════════════

def _rpc(chain: str) -> str:
    defaults = {
        "base": "https://mainnet.base.org",
        "eth": "https://ethereum-rpc.publicnode.com",
        "tron": "https://api.trongrid.io",
        "sol": "https://api.mainnet-beta.solana.com",
    }
    env_key = f"{chain.upper()}_RPC_URL"
    if chain == "tron":
        env_key = "TRX_API_URL"
    return os.environ.get(env_key, defaults[chain])


# ═══════════════════════════════════════════════════════════════════════
# 잔액 조회 (공개 주소만 사용, 개인키 불필요)
# ═══════════════════════════════════════════════════════════════════════

async def _evm_balance(chain: str, address: str) -> dict:
    """ETH/Base — eth_getBalance JSON-RPC."""
    url = _rpc(chain)
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getBalance",
        "params": [address, "latest"],
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    if "error" in data:
        return {"chain": chain, "error": data["error"].get("message", str(data["error"]))}
    wei = int(data["result"], 16)
    eth_val = wei / 1e18
    return {"chain": chain, "address": address, "balance": eth_val, "unit": "ETH"}


async def _tron_balance(address: str) -> dict:
    """TRX — TronGrid /v1/accounts/{address}."""
    url = f"{_rpc('tron')}/v1/accounts/{address}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    if not data.get("data"):
        # 계정이 활성화되지 않은 경우 (잔액 0)
        return {"chain": "tron", "address": address, "balance": 0.0, "unit": "TRX"}
    account = data["data"][0]
    sun = account.get("balance", 0)
    trx_val = sun / 1e6
    return {"chain": "tron", "address": address, "balance": trx_val, "unit": "TRX"}


async def _sol_balance(address: str) -> dict:
    """SOL — getBalance JSON-RPC."""
    url = _rpc("sol")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBalance",
        "params": [address],
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    if "error" in data:
        return {"chain": "sol", "error": data["error"].get("message", str(data["error"]))}
    lamports = data["result"]["value"]
    sol_val = lamports / 1e9
    return {"chain": "sol", "address": address, "balance": sol_val, "unit": "SOL"}


async def get_balances(chains: list[str] | None = None) -> list[dict]:
    """전 체인 잔액 조회. chains가 None이면 모든 체인."""
    addrs = get_addresses()
    if not addrs:
        return [{"error": "No wallet addresses available (CREDENTIALS_DIRECTORY not set?)"}]

    target = chains or list(addrs.keys())
    tasks = []
    for chain in target:
        addr = addrs.get(chain)
        if not addr:
            continue
        if chain in ("eth", "base"):
            tasks.append(_evm_balance(chain, addr))
        elif chain == "tron":
            tasks.append(_tron_balance(addr))
        elif chain == "sol":
            tasks.append(_sol_balance(addr))

    if not tasks:
        return [{"error": f"No addresses for chains: {target}"}]

    return await asyncio.gather(*tasks, return_exceptions=False)


# ═══════════════════════════════════════════════════════════════════════
# Tool 정의 (Anthropic API format) — telegram_tools.py에서 등록
# ═══════════════════════════════════════════════════════════════════════

WALLET_TOOL = {
    "name": "check_wallet",
    "description": (
        "Check Cyber-Lenin's own crypto wallet. "
        "Returns addresses and balances for Base (L2), Ethereum, Tron, and Solana. "
        "Use this when asked about your wallet, crypto balance, or blockchain address."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chains": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["base", "eth", "tron", "sol"],
                },
                "description": "Chains to query. Omit for all chains. 'base' is the primary chain.",
            },
        },
        "required": [],
    },
}


async def _exec_check_wallet(chains: list[str] | None = None) -> str:
    """Tool handler for check_wallet."""
    addrs = get_addresses()
    if not addrs:
        cred = os.environ.get("CREDENTIALS_DIRECTORY")
        if not cred:
            return ("No crypto wallet credentials loaded. "
                    "The wallet private keys are not injected into this service. "
                    "This means the bot is running without wallet access (e.g. local dev or credentials not configured).")
        return ("Wallet credential directory exists but no key files found. "
                "Expected eth.privkey and/or sol.keypair in CREDENTIALS_DIRECTORY.")

    sections = []

    # 주소 표시
    addr_lines = ["My wallet addresses:"]
    chain_labels = {"base": "Base (L2)", "eth": "Ethereum", "tron": "Tron", "sol": "Solana"}
    for chain in ["base", "eth", "tron", "sol"]:
        addr = addrs.get(chain)
        if addr:
            label = chain_labels.get(chain, chain)
            addr_lines.append(f"  {label}: {addr}")
    sections.append("\n".join(addr_lines))

    # 잔액 조회
    try:
        results = await get_balances(chains)
        bal_lines = ["Balances:"]
        for r in results:
            if isinstance(r, dict) and "error" in r:
                chain = r.get("chain", "?")
                bal_lines.append(f"  {chain}: error — {r['error']}")
            elif isinstance(r, dict):
                chain = r["chain"]
                label = chain_labels.get(chain, chain)
                bal = r["balance"]
                unit = r["unit"]
                bal_str = f"{bal:.18f}".rstrip("0").rstrip(".")
                bal_lines.append(f"  {label}: {bal_str} {unit}")
        sections.append("\n".join(bal_lines))
    except Exception as e:
        logger.error("Balance fetch failed: %s", e)
        sections.append(f"Balance query failed: {e}")

    return "\n\n".join(sections)


WALLET_TOOL_HANDLER = _exec_check_wallet
