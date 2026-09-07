# Tool/Security Gateway 남은 개선 과제

2026-09-07 코드 기준으로 완료 항목과 과거 배포 로그를 정리했다. 현재 정책은 [tool_gateway.md](tool_gateway.md), [security_gateway.md](security_gateway.md), [tool_allowlist_current_state.md](tool_allowlist_current_state.md), [mcp_gateway.md](mcp_gateway.md)가 소유한다.

## 구현된 경계

orchestrator schema/handler allow-list 일치, public A2A 읽기 제한, 공통 outbound URL SSRF 검증, fail-closed authorization, dispatcher argument validation, atomic Redis rate limiting, loop-local/durable idempotency는 기존 런타임에 구현되어 있다. 이를 미구현 과제로 다시 추가하지 않는다.

감사 행은 `audit_sink.py`를 통해 프록시로 전송한다. INSERT 전용 audit 역할의 설치 스크립트와 proxy unit credential 설정이 있으며, 기존 문서에 2026-09-04 적용 기록이 있다. 공통 durable spool은 별도 미완료 항목이다.

## 남은 작업

- [ ] `security_gateway/audit.py:redact_args`의 top-level masking을 audit/progress/tool logs의 공통 recursive redaction으로 확장한다. 현재 nested dict/list는 JSON 문자열로 잘라 저장한다.
- [ ] 감사 큐·sink 장애 시 durable spool과 drop 지표를 마련한다. 감사 실패가 handler 재실행으로 이어지지 않게 한다.
- [ ] MCP `bounded_query_db`의 mutation 경계를 SQL parser/최소 DB role 또는 domain-specific operator actions로 보강한다. 이름의 “bounded”를 읽기 전용 보장으로 해석하지 않는다.
- [ ] MCP call을 공통 security authorization/audit envelope에 통합하고 inspect/operator를 OS wrapper·user/group·credential 수준에서 분리한다.
- [ ] KG maintenance의 backup 명령 성공뿐 아니라 backup artifact 유효성도 적용 전 검증하는지 보강한다.
- [ ] connector가 지원하는 경우 native idempotency key를 전달한다.
- [ ] systemd hardening을 서비스별로 점검한다. proxy unit에는 이미 `NoNewPrivileges` 등 일부 설정이 있으므로 전체 미적용으로 취급하지 않는다.

## 변경 시 회귀 검증

- hidden tool 호출, public A2A mutation, schema 밖 인자, authorization/Redis 장애는 handler 실행을 차단해야 한다.
- loopback/private/link-local·mixed DNS·unsafe redirect 및 browser subrequest는 같은 URL 정책을 따라야 한다.
- 동일 scoped side effect 성공은 재사용하고 `outcome_unknown`은 자동 재실행하지 않아야 한다.
- SQL 변경에서는 data-modifying CTE/multi-statement, redaction 변경에서는 nested secret, KG maintenance 변경에서는 backup 실패 후 mutation 미실행을 검증한다.

관련 검증 진입점: `scripts/smoke_tool_allowlists.py`, `scripts/smoke_security_gateway.py`, `scripts/smoke_url_security.py`, `scripts/smoke_mcp_gateway.py`, `tests/`. 과거 통과 개수나 서비스 active 기록을 현재 검증 결과로 재사용하지 않는다.
