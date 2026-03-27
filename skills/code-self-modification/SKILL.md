---
name: code-self-modification
description: Safely modifies leninbot's own Python source code. Handles patch validation, syntax checking, service restart, testing, and git commit/push. Use when asked to fix bugs, add features, or refactor code in telegram_bot.py, shared.py, chatbot.py, or other core files. Keywords: 코드 수정, 버그 수정, 기능 추가, self-modify, patch, code fix, 배포, deploy.
compatibility: Requires git, Python 3.10+, write access to project root, sudo systemctl access. Uses self_modification_core.py.
metadata:
  author: cyber-Lenin
  version: "1.2"
allowed-tools: read_file write_file execute_python
---

# Code Self-Modification Skill

## 핵심 원칙
**잘못된 수정은 봇 전체를 죽인다.** 반드시 아래 순서를 지킬 것.
**할 수 있는 건 직접 한다.** "권한 없다"고 가정하지 말고, 먼저 실행해서 확인할 것.
**경로는 .env에서 로드한다. 하드코딩 금지.**

```python
import os
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
```

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
import os, sys
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.environ["PROJECT_ROOT"])
from self_modification_core import self_modify_with_safety
result = self_modify_with_safety(
    filepath=file_path,
    new_content=new_code,
    reason="skill patch",
    request_approval=False,
    skip_tests=False,
)
```
- 내부적으로 라인 단위 패치(generate_line_patch/apply_line_patch_safe) 경로를 사용한다.
- git 자동 백업 → 구문 검사 → 적용 → 실패 시 자동 롤백

## Step 4 — 구문 검증
```python
import ast
with open("파일명.py") as f:
    ast.parse(f.read())
print("구문 OK")
```

## Step 5 — 재시작 전 맥락 인계 (필수!)
**서비스 재시작 = 현재 태스크 사망.** 재시작 전에 반드시 `request_continuation`으로 자식 태스크를 생성해서 나머지 작업(테스트, commit, push)을 위임한다.

```
request_continuation(
    progress_summary="Step 1~4 완료. {수정한 파일명}의 {라인 범위}를 수정함. 구문 검증 통과. 서비스 재시작 예정.",
    next_steps="1. 서버 로그 확인 (에러 없는지)\n2. 기능 테스트\n3. 테스트 통과 시 git commit & push\n4. 사용자에게 결과 보고"
)
```

**progress_summary에 반드시 포함할 것:**
- 수정한 파일명과 변경 내용
- 완료된 단계 (Step 1~4)
- 구문 검증 결과

**next_steps에 반드시 포함할 것:**
- 남은 단계 (테스트 → commit → push → 보고)
- 테스트 시 확인할 구체적 항목

자식 태스크가 생성된 것을 확인한 뒤 재시작한다.

## Step 6 — 서비스 재시작
```python
import subprocess
result = subprocess.run(
    ["sudo", "systemctl", "restart", "leninbot-telegram"],
    capture_output=True, text=True
)
print(result.returncode, result.stdout, result.stderr)
```
- returncode 0이면 성공 → 현재 태스크는 여기서 종료됨
- 실패 시 rollback 후 사용자에게 보고

## Step 7 — 테스트 (자식 태스크가 수행)
재시작 후 자식 태스크가 자동으로 실행되며, 인계받은 맥락을 기반으로:
- `read_server_logs(service="telegram", minutes_back=2)` — 에러 없는지 확인
- 필요 시 기능별 직접 테스트 (KG 연결, tool 호출 등)

## Step 8 — 테스트 통과 시 commit & push
테스트가 통과한 경우에만 원격 저장소에 반영한다:
```python
import os, subprocess
from dotenv import load_dotenv
load_dotenv()
ROOT = os.environ["PROJECT_ROOT"]
subprocess.run(["git", "add", "-A"], cwd=ROOT)
subprocess.run(["git", "commit", "-m", "feat: 변경 내용 요약"], cwd=ROOT)
subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
```
- **테스트 실패 시 push 금지** — 롤백 후 재수정

## Step 9 — 사용자 보고
- 수정된 파일명, 라인 번호, 변경 내용 요약
- 재시작 결과 (성공/실패)
- commit hash

---

## 순서 요약
```
코드 수정 → 구문 검증 → request_continuation → 서비스 재시작 → [자식 태스크] 테스트 → commit & push
```

## 절대 금지
- 인증/보안 로직 단독 수정 (사용자 승인 필요)
- 프로젝트 루트 외부 파일 수정
- 백업 없는 수정
- `os.system`, `subprocess` 직접 실행 코드 삽입 (자기 자신 코드에)
- 테스트 전 push
- "권한 없다"고 가정하고 사용자에게 떠넘기기
- 경로 하드코딩 (반드시 .env에서 PROJECT_ROOT 로드)
