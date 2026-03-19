---
name: code-self-modification
description: Safely modifies leninbot's own Python source code. Handles patch validation, syntax checking, service restart, testing, and git commit/push. Use when asked to fix bugs, add features, or refactor code in telegram_bot.py, shared.py, chatbot.py, or other core files. Keywords: 코드 수정, 버그 수정, 기능 추가, self-modify, patch, code fix, 배포, deploy.
compatibility: Requires git, Python 3.10+, write access to project root, sudo systemctl access. Uses self_modification_core.py.
metadata:
  author: cyber-Lenin
  version: "1.1"
allowed-tools: read_file write_file execute_python
---

# Code Self-Modification Skill

## 핵심 원칙
**잘못된 수정은 봇 전체를 죽인다.** 반드시 아래 순서를 지킬 것.
**할 수 있는 건 직접 한다.** "권한 없다"고 가정하지 말고, 먼저 실행해서 확인할 것.

---

## Step 1 — 현재 코드 파악
```
read_file("파일명.py", line_start=X, line_end=Y)
```
- 수정 대상 파일과 라인 범위를 정확히 확인
- 의존성(import, 함수 호출) 파악

## Step 2 — 안전성 검토
[references/safety-rules.md](references/safety-rules.md) 확인.
금지 패턴이 포함된 수정은 즉시 중단.

## Step 3 — 패치 실행
`self_modification_core.py`의 안전 경로 사용:
```python
# execute_python으로 호출
import sys; sys.path.insert(0, '/home/grass/leninbot')
from self_modification_core import safe_patch_file
result = safe_patch_file(file_path, old_code, new_code)
```
- git 자동 백업 → 구문 검사 → 적용 → 실패 시 자동 롤백

## Step 4 — 구문 검증
```python
import ast
with open("파일명.py") as f:
    ast.parse(f.read())
print("구문 OK")
```

## Step 5 — 서비스 재시작 (변경 사항 즉시 반영)
코드 수정 후 서비스를 **직접 재시작**한다. `/deploy`나 사용자에게 요청하지 말 것.
```python
import subprocess
result = subprocess.run(
    ["sudo", "systemctl", "restart", "leninbot-telegram"],
    capture_output=True, text=True
)
print(result.returncode, result.stdout, result.stderr)
```
- returncode 0이면 성공
- 실패 시 rollback 후 사용자에게 보고

## Step 6 — 테스트
재시작 후 실제 동작을 확인한다:
- `read_server_logs(service="telegram", minutes_back=2)` — 에러 없는지 확인
- 필요 시 기능별 직접 테스트 (KG 연결, tool 호출 등)

## Step 7 — 테스트 통과 시 commit & push
테스트가 통과한 경우에만 원격 저장소에 반영한다:
```python
import subprocess
subprocess.run(["git", "add", "-A"], cwd="/home/grass/leninbot")
subprocess.run(["git", "commit", "-m", "feat: 변경 내용 요약"], cwd="/home/grass/leninbot")
subprocess.run(["git", "push", "origin", "main"], cwd="/home/grass/leninbot")
```
- **테스트 실패 시 push 금지** — 롤백 후 재수정

## Step 8 — 사용자 보고
- 수정된 파일명, 라인 번호, 변경 내용 요약
- 재시작 결과 (성공/실패)
- commit hash

---

## 순서 요약
```
코드 수정 → 구문 검증 → 서비스 재시작 → 테스트 → (통과 시) commit & push
```

## 절대 금지
- 인증/보안 로직 단독 수정 (사용자 승인 필요)
- 프로젝트 루트 외부 파일 수정
- 백업 없는 수정
- `os.system`, `subprocess` 직접 실행 코드 삽입 (자기 자신 코드에)
- 테스트 전 push
- "권한 없다"고 가정하고 사용자에게 떠넘기기
