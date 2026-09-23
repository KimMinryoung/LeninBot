# x402 결제 데모

현재 구현은 Base 메인넷 USDC의 `exact` 결제를 처리한다. Telegram orchestrator의 `pay_and_fetch`가 보호된 HTTP 리소스를 요청할 수 있고, API의 `GET /x402-demo/quote`가 자체 정산하는 데모를 제공한다. 주 소유 코드는 `crypto_wallet/x402.py`, `crypto_wallet/x402_ledger.py`, `api_routes/x402_demo.py`다. 이 라우터는 `services/api.py`에 등록된다.

## 실행 흐름

1. `pay_and_fetch`가 GET 또는 POST를 보낸다. 200이면 결제 없이 본문을 반환한다.
2. 402이면 응답 본문의 `accepts` 목록에서 `exact`/`eip155:8453`을 고른다. 본문에 목록이 없으면 `PAYMENT-REQUIRED` 또는 `X-PAYMENT-REQUIRED`의 base64 JSON을 시도한다.
3. `sign_payment()`가 자산을 Base USDC(`0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`)로 제한하고, 요청 금액을 호출별 `max_usdc`와 비교한 뒤 systemd credential `eth.privkey`로 ERC-3009 EIP-712 authorization에 서명한다. Nonce는 32바이트 난수이고, 만료 시간은 최소 60초다.
4. 같은 리소스에 `PAYMENT-SIGNATURE` 헤더를 붙여 재요청한다. 데모 서버는 서명·금액·수신자·유효기간을 확인하고 `transferWithAuthorization`으로 온체인 정산한 뒤 성공 시 200과 `PAYMENT-RESPONSE` 헤더를 반환한다.

데모 가격은 `X402_DEMO_QUOTE_USDC` 기본 0.05 USDC, 도구의 기본 호출 상한은 `X402_MAX_USDC_PER_CALL` 기본 0.05 USDC다. 가격이 호출 상한을 넘으면 서명하지 않는다. 데모는 같은 지갑이 지불·수령·정산하므로 USDC 순변동은 없고 gas가 든다. `pay_and_fetch`는 `tool_gateway.profiles.TELEGRAM_ORCHESTRATOR_TOOLS`에 포함되며 specialist agent 목록에는 없다.

## 와이어 형식

402 응답 본문은 `x402Version: 2`, `error`, `accepts: [PaymentRequirements]`이다. 각 requirements에는 `scheme`, `network`, `maxAmountRequired`(USDC atomic units), `resource`, `payTo`, `maxTimeoutSeconds`, `asset`, `extra` 등이 들어간다. 데모는 같은 requirements를 base64로 인코딩해 `PAYMENT-REQUIRED` 헤더에도 보낸다.

현재 `encode_payment_header()`가 만드는 `PAYMENT-SIGNATURE`의 base64 JSON 구조는 다음과 같다. `accepted`는 선택한 requirements 객체 전체이며 `resource`는 URL을 담은 객체다.

```json
{
  "x402Version": 2,
  "payload": {
    "authorization": {
      "from": "0x...",
      "to": "0x...",
      "value": "50000",
      "validAfter": "...",
      "validBefore": "...",
      "nonce": "0x..."
    },
    "signature": "0x..."
  },
  "accepted": { "scheme": "exact", "network": "eip155:8453", "maxAmountRequired": "50000" },
  "resource": { "url": "http://localhost:8000/x402-demo/quote" }
}
```

데모의 200 응답 본문에는 `aphorism`, `payer`, `amount_atomic`, `tx_hash`, `gas_used`가 포함된다. `PAYMENT-RESPONSE`는 정산 결과의 base64 JSON이다. 예시는 구조 설명용이며 `accepted`에는 실제로 선택한 requirements의 나머지 필드도 들어간다.

## 감사·운영 경계

`crypto_wallet/x402_ledger.py`는 `x402_payment_attempts`에 outbound 요청과 inbound 데모 단계를 best-effort로 기록한다. 서명·개인키·raw 결제 헤더는 저장하지 않는다. 테이블은 `venv/bin/python scripts/schema_migrations.py --only x402-ledger`로 적용한다. 결제 요청의 일반 도구 권한·감사는 [security_gateway.md](security_gateway.md)를 따른다.

API 라우트에는 정산용 `eth.privkey` credential이 필요하다. 배포 시 실제 unit의 credential 설정을 확인한다. 현재 코드는 자체 정산 데모 하나만 제공하며 외부 facilitator·공개 유료 콘텐츠·일일 결제 한도는 구현하지 않았다. 외부 공개 여부와 reverse proxy 설정은 코드만으로 단정하지 않는다.
