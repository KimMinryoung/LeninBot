# x402 Payment Design

> x402 작업 전에 반드시 이 문서를 확인하고, 작업 후에는 변경사항을 반영할 것.

---

## 1. 개요

### 목적

Cyber-Lenin이 HTTP 위에서 USDC 마이크로페이먼트를 자율적으로 처리할 수 있게 한다. 결제·검증·정산을 한 라운드에 끝내는 [x402 프로토콜](https://github.com/x402-foundation/x402)을 Base 메인넷 위에 구현한다.

데모로 검증된 것 (2026-04-07):
- `pay_and_fetch` 도구로 leninbot이 자기 API의 보호된 라우트에 0.001 USDC를 결제하고 컨텐츠를 받아오는 self-loop 한 사이클 — 사람 개입 없이 전 단계 자동
- 첫 성공 TX: `0xfad05f83e786...ddb7a` on Base mainnet, 가스 75,656

### 스택

| 구성요소 | 기술 |
|----------|------|
| 결제 자산 | USDC on Base L2 (`0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`) |
| 결제 메커니즘 | ERC-3009 `transferWithAuthorization` (가스리스, 원샷, EIP-712 서명) |
| 프로토콜 | x402 v2, scheme=`exact`, network=`eip155:8453` (CAIP-2) |
| 서명 라이브러리 | `eth-account` 0.13.7 (`encode_typed_data` + `Account.sign_message`) |
| 트랜잭션 라이브러리 | `web3.py` 7.15.0 |
| 키 보관 | systemd `LoadCredentialEncrypted`로 메모리에만 노출, 사용 후 즉시 삭제 |

---

## 2. 아키텍처

### 파일 구조

```
crypto_wallet/
├── __init__.py            # WALLET_TOOL, SWAP_TOOL, TRANSFER_TOOL, PAY_AND_FETCH_TOOL re-export
├── wallet.py              # 주소 도출 + 잔액 조회 (read-only)
├── transactions.py        # ETH↔USDC swap, USDC transfer (web3.py)
└── x402.py                # x402 프로토콜 — 클라이언트(서명) + 서버(검증/정산) + pay_and_fetch
```

### 메시지 흐름 (self-loop 데모)

```
[telegram_bot.py orchestrator]              [api.py /x402-demo/quote]
  pay_and_fetch tool 호출
  └─→ httpx.GET URL ────────────────────────→ 402
                                              {
                                                "x402Version": 2,
                                                "error": "PAYMENT-SIGNATURE header required",
                                                "accepts": [PaymentRequirements]
                                              }
                                              + PAYMENT-REQUIRED 헤더 (base64 JSON)
  사용 가능한 옵션 선택 (exact / eip155:8453)
  사용자 max_usdc 캡 + 모듈 PAY_AND_FETCH_MAX_USDC 캡 검사
  CREDENTIALS_DIRECTORY/eth.privkey 로드
  EIP-712 typed-data 빌드
  Account.sign_message → 65바이트 ECDSA
  privkey 즉시 del
  base64 인코딩
  └─→ httpx.GET URL                ─────────→ PAYMENT-SIGNATURE 디코드
      headers: PAYMENT-SIGNATURE             EIP-712 서명 검증 (recover 후 from 비교)
                                             금액/recipient/유효기간 일치 확인
                                             USDC.transferWithAuthorization() 호출
                                             receipt 대기 (settler가 가스 부담)
                                              ←  200 OK
                                                {
                                                  "aphorism": "...",
                                                  "payer": "0x...",
                                                  "tx_hash": "0x...",
                                                  "gas_used": 75656
                                                }
                                                + PAYMENT-RESPONSE 헤더 (정산 결과 base64)
  본문 + 정산 정보를 LLM에 반환
```

### EIP-712 도메인

USDC 컨트랙트의 온체인 `DOMAIN_SEPARATOR`와 정확히 일치해야 함 (불일치 시 settle 단계에서 InvalidSignature).

```python
USDC_DOMAIN = {
    "name": "USD Coin",                    # 컨트랙트 name() == "USD Coin", NOT "USDC"
    "version": "2",                        # 컨트랙트 version() == "2"
    "chainId": 8453,                       # Base mainnet
    "verifyingContract": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
}
```

검증: `eth_account.messages.encode_typed_data(...).header.hex()` 결과가 `02fa7265e7c5d81118673727957699e4d68f74cd74b7db77da710fe8a2c7834f` (= 온체인 DOMAIN_SEPARATOR)와 일치하는지 확인. 일치 확인됨 (2026-04-07).

### 와이어 포맷

**402 응답 본문** (서버 → 클라이언트):

```json
{
  "x402Version": 2,
  "error": "PAYMENT-SIGNATURE header required",
  "accepts": [
    {
      "scheme": "exact",
      "network": "eip155:8453",
      "maxAmountRequired": "1000",
      "resource": "http://localhost:8000/x402-demo/quote",
      "description": "Cyber-Lenin x402 demo: pay tiny USDC for an aphorism",
      "mimeType": "application/json",
      "payTo": "0x3cF08...",
      "maxTimeoutSeconds": 60,
      "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
      "extra": { "name": "USD Coin", "version": "2" }
    }
  ]
}
```

**PAYMENT-SIGNATURE 헤더** (클라이언트 재요청 시): base64-encoded JSON of:

```json
{
  "x402Version": 2,
  "scheme": "exact",
  "network": "eip155:8453",
  "resource": "http://localhost:8000/x402-demo/quote",
  "payload": {
    "authorization": {
      "from": "0x3cF08...",
      "to": "0x3cF08...",
      "value": "1000",
      "validAfter": "1775569787",
      "validBefore": "1775569907",
      "nonce": "0x898ab9..."
    },
    "signature": "0xbc4a0a0e35..."
  }
}
```

**PAYMENT-RESPONSE 헤더** (서버 → 클라이언트, 200 응답 시): base64-encoded settlement details (`tx_hash`, `status`, `gas_used`).

---

## 3. 구현 세부

### `crypto_wallet/x402.py` 함수

| 함수 | 역할 |
|---|---|
| `build_payment_requirements(...)` | 서버: 한 개의 PaymentRequirements 객체 생성 |
| `build_402_body(req)` | 서버: 402 응답 JSON envelope |
| `sign_payment(req, max_atomic)` | 클라이언트: cap 검사 + EIP-3009 서명. 개인키 just-in-time 로드, 사용 후 del |
| `encode_payment_header(...)` / `decode_payment_header(...)` | base64 wrap/unwrap |
| `verify_payment(payment, req)` | 서버: 서명/금액/recipient/유효기간 전부 검증, 복원 주소 반환 |
| `settle_payment(payload)` | 서버: USDC.transferWithAuthorization 온체인 호출, receipt 반환 |
| `pay_and_fetch(url, max_usdc)` | 클라이언트 풀 플로우 (GET → 402 → sign → retry → 결과) |
| `_exec_pay_and_fetch(...)` | 도구 핸들러 — LLM에 표시할 형태로 결과 포맷 |

### 안전 가드

| 가드 | 위치 | 동작 |
|---|---|---|
| **per-call hard cap** | `sign_payment` | `int(max_usdc * 1e6)`보다 큰 PaymentRequirements는 서명 거부 |
| **module-level cap** | `PAY_AND_FETCH_MAX_USDC` env (`X402_MAX_USDC_PER_CALL`, 기본 0.05) | 도구 default 인자 |
| **scheme/network/asset 화이트리스트** | `sign_payment` | exact + eip155:8453 + USDC만 통과. 다른 자산 거부 |
| **유효기간** | `validBefore = now + maxTimeoutSeconds` | 짧게 유지 (재사용 차단) |
| **서명 nonce** | `secrets.token_bytes(32)` | 32바이트 랜덤. ERC-3009 컨트랙트가 nonce 재사용 거부하므로 replay 방지는 컨트랙트 레벨 |
| **검증 시 금액/recipient 재확인** | `verify_payment` | 서명한 값이 서버가 요구한 값과 일치하는지 강제 |
| **유효기간 검증** | `verify_payment` | 현재 시각이 `[validAfter, validBefore]` 범위 안에 있어야 함 |
| **개인키 메모리 노출 최소화** | `sign_payment` / `_load_privkey_hex` | 사용 직후 `del pk_hex` |
| **오케스트레이터 화이트리스트** | `telegram_bot.py:_ORCHESTRATOR_TOOLS` | `pay_and_fetch`는 orchestrator 전용. delegated 서브에이전트 호출 불가 |

### systemd credential 분배

| 서비스 | LoadCredentialEncrypted | 이유 |
|---|---|---|
| `leninbot-telegram` | `eth.privkey`, `sol.keypair` | orchestrator가 swap/transfer/pay_and_fetch 도구 호출 |
| `leninbot-api` | `eth.privkey` (override.conf) | x402 라우트가 서버 측에서 settle |
| `leninbot-browser` | (없음) | 결제 안 함 |
| `leninbot-experience` | (없음) | 결제 안 함 |

`leninbot-api`의 override 파일:

```ini
# /etc/systemd/system/leninbot-api.service.d/override.conf
[Service]
LoadCredentialEncrypted=eth.privkey:/etc/credstore.encrypted/eth.privkey.cred
```

---

## 4. 데모 라우트 — `/x402-demo/quote`

`api.py`에 박힌 self-loop 라우트.

| 항목 | 값 |
|---|---|
| 메서드 | GET |
| 가격 | 0.001 USDC (= 1000 atomic units) |
| recipient | leninbot의 자체 Base 지갑 주소 (= signer = settler) |
| 컨텐츠 | "정치란 과학이며 예술이다…" (정적 격언, demo 용도) |
| 인증 | 없음 (외부 노출 시 인증 필요 — 현재는 `127.0.0.1:8000`만 들어옴) |

self-loop 특성상 USDC 순 변동량 = 0, 가스만 소모 (~$0.0001 per call). 외부에서 들어오는 결제로 확장하면 net inflow가 됨.

### LLM 발견 가능성

LLM이 데모 endpoint 존재를 모르면 외부 x402 서비스를 찾으러 가버리기 때문에 두 군데 명시:

1. `crypto_wallet/x402.py:PAY_AND_FETCH_TOOL.description` 마지막 문단
2. `telegram_bot.py:_SYSTEM_PROMPT_TEMPLATE`의 `<tool-strategy>` 블록 안

---

## 5. 알려진 제약과 향후 작업

### 현재 제약

- **단일 라우트**: `/x402-demo/quote`만 존재. 컨텐츠 다양성 없음 (정적 격언 1개)
- **localhost only**: `leninbot-api`는 외부 차단 (Docker 브릿지만 허용). 외부 에이전트가 결제할 수 없음
- **per-resource pricing 없음**: 모든 호출이 0.001 USDC 고정. 컨텐츠 가치 차등 불가
- **인증/abuse 방지 없음**: 외부 노출 시 무한 결제 요청에 취약
- **재정산 캐싱 없음**: 같은 컨텐츠를 두 사람이 결제하면 두 번 정산. 디지털 굿즈 가격 정책 미정
- **facilitator 미사용**: 자체 settle. 외부 x402 호환 facilitator (Coinbase 등) 통합 안 됨

### 다음 단계 옵션 (2026-04-07 시점, 미결정)

1. **컨텐츠 다양화 (Phase 1)**
   - `/x402/aphorism` (현재 quote의 이름 정리, 0.001 USDC)
   - `/x402/brief/<topic>` — 결제 받으면 그 자리에서 LLM 호출해 짧은 지정학 브리핑 생성. 0.05 USDC. 생성 비용 < 결제 금액 검증 필요
   - `/x402/research/<id>` — 미리 만들어둔 리서치 보고서, 캐시. 0.20 USDC

2. **외부 노출**
   - `cyber-lenin.com/x402/*`로 공개. Nginx 프록시로 `/x402/*`만 전달
   - 외부 에이전트가 결제 가능 → 진짜 x402 서비스 운영자 데뷔
   - rate limit, abuse 차단 필수

3. **운영 인프라**
   - audit log: redis 또는 sqlite에 모든 결제 시도 기록 (timestamp, payer, amount, tx_hash, content_id)
   - 일일 누적 cap: redis 카운터. 화이트리스트 + 차단 리스트
   - 정산 모니터링: 실패율, 평균 가스, 수익 추적
   - 텔레그램 알림: 결제 들어올 때마다 사용자에게 보고

4. **x402 SDK 호환성**
   - Coinbase facilitator API와 연동 (현재는 자체 settle)
   - 다른 x402 클라이언트 (TypeScript SDK 등)가 이 서버를 칠 수 있는지 검증

---

## 6. 변경 이력

- **2026-04-07**: 초기 구현. `crypto_wallet/x402.py` 작성, `/x402-demo/quote` 라우트, `pay_and_fetch` 도구, orchestrator 화이트리스트, `leninbot-api` 서비스에 credential 부여. self-loop 데모 1회 성공 (TX `0xfad05f83e786...ddb7a`, 0.001 USDC, gas 75,656).
