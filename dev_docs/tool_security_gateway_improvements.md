# Tool/Security Gateway Review and Improvement Backlog

최종 확인 기준: 2026-07-25 코드 트리.

이 문서는 `tool_gateway`, `security_gateway`, public web/A2A tool surface, inbound
MCP gateway를 함께 검토한 결과와 후속 작업을 추적한다. 현재 동작의 source of
truth는 `tool_gateway.md`, `security_gateway.md`,
`tool_allowlist_current_state.md`, `mcp_gateway.md`이며, 이 문서는 위험과 개선
우선순위를 한곳에서 보여 주는 backlog다.

## Review Summary

현재 구조의 장점은 surface별 명시적 allow-list, agent별 `AgentSpec.tools`,
실행 시점 security authorization/audit, MCP inspect/operator 프로필 분리다.
그러나 다음 경계는 추가 보강이 필요하다.

- Telegram orchestrator는 좁은 tool schema와 전체 global handler map을 함께
  전달해, provider가 schema 밖 tool name을 반환할 때 실행 allow-list가
  handler 존재 여부로 대체될 수 있었다.
- public A2A `geopolitical-analysis`가 직접 KG mutation 도구
  `write_kg_structured`를 노출했고, security gateway에는 A2A 전용 read-only
  interface rule이 없었다.
- public web/A2A/MCP에서 쓸 수 있는 `fetch_url`이 scheme, destination IP,
  redirect target을 검증하지 않아 loopback/private/link-local endpoint 접근과
  진단 probe를 통한 내부 port 접근 가능성이 있었다.
- MCP `bounded_query_db`는 SQL의 첫 keyword만 분류하므로 data-modifying CTE와
  multi-statement 우회에 취약하다.
- uncategorized tool과 gateway 내부 오류가 fail-open이고, argument schema/policy,
  recursive audit redaction, atomic rate limiting, side-effect idempotency가 아직
  중앙 경계에 통합되지 않았다.
- MCP calls는 runtime security gateway/audit를 통과하지 않으며 operator profile
  선택도 OS-level authorization과 분리되어 있지 않다.

## Implementation Checklist

### Immediate

- [x] Telegram orchestrator의 visible schema와 executable handler map에 동일한
  allow-list를 적용하고, 목록 밖 handler가 주입되지 않는 회귀 테스트를 추가한다.
- [x] public A2A를 read-only로 전환한다. `write_kg_structured`를 제거하고
  security gateway에 A2A read-only defense-in-depth rule을 추가한다.
- [x] outbound HTTP(S) URL에 공통 SSRF guard를 적용한다. loopback/private/
  link-local/reserved IP, unsafe scheme/port, unsafe redirect를 차단하고
  Playwright subrequest와 failure diagnosis에도 같은 policy를 적용한다.

## Implemented Changes

### 1. Telegram orchestrator execution boundary

- `runtime_tools.allowlists.build_orchestrator_toolset()`이 tool schema와 handler를
  `TELEGRAM_ORCHESTRATOR_TOOLS`로 한 번에 필터링한다.
- `telegram.bot._chat_with_tools()`는 더 이상 전체 `TOOL_HANDLERS`를 orchestrator
  provider loop에 전달하지 않는다. `mission` 같은 동적 handler도 allow-list에
  이름이 있을 때만 추가된다.
- 회귀 테스트는 hidden `transfer_usdc`와 `execute_python` handler 주입이
  거부되는지 확인한다.

### 2. Public A2A read-only boundary

- `a2a.geopolitical-analysis` 프로필에서 `write_kg_structured`를 제거했다.
- public A2A prompt와 geopolitical skill은 KG 갱신 단계를 건너뛰고 새 사실을
  응답에만 포함하도록 안내한다.
- `security_gateway`는 `interface="a2a"`에 대해 shadow/enforce 설정과 무관하게
  `read`, `fetch`, `wallet_read` 외 risk class를 실행 시점에 거부한다.

### 3. Outbound URL SSRF boundary

- `content_fetch/url_security.py`가 HTTP(S) scheme, 허용 port
  (`80`, `443`, `8080`, `8443`), userinfo, DNS answer와 redirect target을
  공통 검증한다.
- literal/DNS 목적지는 모두 global address여야 한다. loopback, private,
  link-local, reserved 및 mixed public/private DNS answer는 거부한다.
- requests redirect는 자동 추적하지 않고 매 hop을 검증한다. Playwright는 main
  navigation과 각 subrequest를 재검증하며 failure diagnosis의 DNS/TCP/HTTP
  probe도 동일한 policy를 사용한다.
- 내부 redirect/subrequest에 공통 validator를 적용할 수 없는 Crawl4AI local
  fallback은 제거했다. Tavily remote extraction과 guarded requests fallback은
  유지한다.

## Verification and Deployment

- `py_compile`: 변경된 gateway, A2A, Telegram, URL fetch 코드 통과
- `scripts/smoke_tool_allowlists.py`: 통과
- `scripts/smoke_security_gateway.py`: 32 passed, 0 failed
- `scripts/smoke_url_security.py`: 통과
- `scripts/smoke_fetch_url_pagination.py`: 통과
- `scripts/smoke_webchat_security.py`: 통과
- `scripts/smoke_mcp_gateway.py`: 통과
- `scripts/smoke_runtime.py`: 통과
- `leninbot-telegram.service`, `leninbot-api.service`,
  `leninbot-a2a-api.service` 재시작 후 모두 active
- main API와 A2A API `/health`: HTTP 200

### Next

- [ ] uncategorized tool과 authorization 내부 오류에 대해 risk-aware fail mode를
  도입한다. read-only는 제한적 fail-open을 허용할 수 있지만
  write/publish/send/pay/execute/admin은 fail-closed로 처리한다.
- [ ] dispatcher에서 JSON Schema를 검증하고 unknown argument를 삭제하지 말고
  거부·감사한다. URL, path, amount, recipient, confirmation nonce 같은
  argument-level policy도 같은 단계에서 평가한다.
- [ ] audit/progress/tool logs에 하나의 recursive redaction 함수를 사용하고,
  nested authorization header/token/cookie와 민감한 본문을 보호한다.
- [ ] Redis rate limit의 count/consume을 Lua 또는 transaction으로 원자화하고,
  Redis 장애를 decision/audit에 명시한다.
- [ ] side-effect tool에 `run_id`, `tool_call_id`, idempotency key를 전달해 provider
  retry나 continuation이 결제·발송·게시를 중복 실행하지 않게 한다.

### MCP and Operations

- [ ] `bounded_query_db`를 raw SQL parser/제한 DB role 없이 mutation 도구로
  사용하지 않는다. 가능하면 domain-specific operator actions로 대체한다.
- [ ] MCP call도 security authorization/audit envelope로 통합하고 inspect와
  operator를 별도 OS wrapper/user/group 및 최소 credential로 분리한다.
- [ ] KG maintenance는 backup command와 backup artifact 검증이 성공한 뒤에만
  mutation을 실행한다.
- [ ] audit DB에 append-only runtime role을 두고
  `UPDATE/DELETE/TRUNCATE` 권한을 회수한다. durable spool과 drop metric도 둔다.
- [ ] systemd unit에 `NoNewPrivileges`, `ProtectSystem`, `ProtectHome`,
  `PrivateTmp`, address-family 제한과 서비스별 최소 credential을 적용한다.

## Required Regression Cases

- orchestrator provider output이 hidden tool name을 반환해도 handler가 없고 실행되지 않음
- public A2A가 모든 write/publish/send/pay/execute/admin class를 거부함
- `127.0.0.1`, RFC1918, `169.254.169.254`, IPv6 loopback/link-local,
  mixed public/private DNS answer, public-to-private redirect가 차단됨
- Playwright navigation/subrequest와 failure diagnosis probe도 동일한 URL policy 사용
- data-modifying CTE와 multi-statement SQL이 operator DB boundary를 우회하지 못함
- nested secret redaction, concurrent rate limit, Redis/policy failure,
  backup failure 후 mutation 미실행을 검증함
