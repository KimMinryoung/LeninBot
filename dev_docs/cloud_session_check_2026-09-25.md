# 클라우드 세션 단독 실행 점검 (2026-09-25) — 서버 세션 전달용

frontend(BichonWebpage) 의존성 제거(`a5d8db6`~`b8917e6`) 이후, 서버 없이 Claude Code 클라우드 세션에서 이 저장소를 새로 클론해 작업·테스트가 되는지 점검한 결과다. 아래 **서버 세션 요청 사항**을 처리하면 이 문서는 삭제한다(`dev_docs/README.md`: 완료된 인수인계는 보존하지 않음).

점검 기준 커밋: `b8917e6` (main)

## 결론

작업·테스트를 막는 문제는 없다. frontend 체크아웃, `.env`, DB, Redis, Neo4j, 시크릿 없이 전부 동작했다. 이번 점검에서 코드는 바꾸지 않았다.

## 점검 결과

| 항목 | 결과 |
|---|---|
| SessionStart 훅 → `scripts/cloud_setup.sh` | `CLAUDE_CODE_REMOTE=true`에서 자동 실행, `venv/` 생성(Python 3.12.3, 약 1.9GB). 재실행 시 `venv up to date`로 즉시 종료 |
| `scripts/run_unit_tests.sh` | 종료 코드 0, 약 31초. unittest 1267 통과 / 24 skip, pytest 66 통과 / 7 skip |
| skip 사유 | 모두 opt-in 격리 PostgreSQL/RPC 필요 테스트와 `test_copies_match_frontend`(frontend 없음). 클라우드에서는 정상 |
| 로그의 ERROR/WARNING | 실패 경로를 의도적으로 검증하는 테스트의 출력. 실패 아님 |
| 모듈 import 점검 | `tests/`·`scripts/`·`skills/`·`migrations/` 등을 제외한 모듈 294개를 개별 import → 전부 성공. frontend·환경변수 부재로 import 시점에 죽는 곳 없음 |
| `scripts/` 컴파일 | `compileall` 전부 통과 |
| `sync_commulingo_contracts.py` | frontend 부재 시 "nothing to compare" 출력 후 종료 코드 2 (의도대로) |
| 체크인된 config의 호스트 절대경로 | `config/`·`agents/`·`skills/`·`research/`·`writer/`의 JSON/YAML/TOML에 `/home/...` 등 없음. 남은 `/home/grass`는 systemd 유닛, `ops/paths.py`의 `FRONTEND_DIR` 기본값, `skills/kg-maintenance/SKILL.md`의 `PROJECT_ROOT` 기본값뿐이며 모두 서버 전용이라 문제없음 |
| `pip check` | `browser-use 0.12.5`가 `anthropic==0.76.0`, `requests==2.32.5`를 요구하나 lock은 `anthropic 0.116.0`, `requests 2.33.1`. 운영 venv와 같은 상태(`--no-deps` 재현)이며 테스트 영향 없음 |

## 서버 세션 요청 사항

1. **vendored 계약 사본 동기화 확인.** 클라우드 세션은 BichonWebpage 저장소 접근 권한이 없어 `config/commulingo_contracts/`와 frontend 원본의 일치 여부를 확인하지 못했다. 서버에서 실행:

   ```bash
   venv/bin/python scripts/sync_commulingo_contracts.py --check
   ```

   차이가 있으면 인자 없이 실행해 복사하고 커밋한다.

2. **`AGENTS.md`의 `grass` 규칙과 클라우드 환경의 불일치.** `AGENTS.md`는 "root로 실행 중이면 파일 편집과 Git을 `grass`로 하라"고 하지만, 클라우드 세션에는 `grass` 사용자가 없고 세션이 root로 실행된다(`id grass` → no such user). "Cloud sessions" 절에 클라우드에서는 이 규칙이 적용되지 않는다는 한 줄을 추가할지 결정해 반영한다.

## 재현 방법 (클라우드 또는 단독 클론)

```bash
bash scripts/cloud_setup.sh      # 클라우드에서는 SessionStart 훅이 자동 실행
scripts/run_unit_tests.sh
```
