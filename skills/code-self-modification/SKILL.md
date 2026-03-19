---
name: code-self-modification
description: Safely modifies leninbot's own Python source code. Handles patch validation, git backup, syntax checking, and rollback on failure. Use when asked to fix bugs, add features, or refactor code in telegram_bot.py, shared.py, chatbot.py, or other core files. Keywords: 코드 수정, 버그 수정, 기능 추가, self-modify, patch, code fix.
compatibility: Requires git, Python 3.10+, and write access to project root. Uses self_modification_core.py.
metadata:
  author: cyber-lenin
  version: "1.0"
allowed-tools: read_file write_file execute_python
---

# Code Self-Modification Skill

## 핵심 원칙
**잘못된 수정은 봇 전체를 죽인다.** 반드시 아래 순서를 지킬 것.

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

## Step 4 — 검증
```python
# 구문 검사
import ast
with open("파일명.py") as f:
    ast.parse(f.read())
print("구문 OK")
```

## Step 5 — 사용자 보고
- 수정된 파일명, 라인 번호, 변경 내용 요약
- 재시작 필요 여부 명시 (`/deploy` 필요한 경우)

## 절대 금지
- 인증/보안 로직 단독 수정 (사용자 승인 필요)
- 프로젝트 루트 외부 파일 수정
- 백업 없는 수정
- `os.system`, `subprocess` 직접 실행 코드 삽입
