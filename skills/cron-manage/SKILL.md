---
name: cron-manage
description: Safely manages system crontab entries. Loads VENV_PYTHON from .env, validates scripts, standardizes log paths. Use when creating, listing, or deleting cron jobs. Keywords: cron, crontab, 스케줄, 크론, schedule system task, periodic job, 주기 실행.
compatibility: Requires crontab command, .env with VENV_PYTHON set. Python scripts must exist under project root.
metadata:
  author: cyber-Lenin
  version: "1.1"
allowed-tools: execute_python
---

# Cron-Manage Skill

## 핵심 원칙
**Python 스크립트는 반드시 .env의 VENV_PYTHON 경로로 실행한다.** 시스템 python(`/usr/bin/python3`, `python3`)은 절대 사용 금지.

**경로는 .env에서 로드한다. 하드코딩 금지.**

```python
# 모든 Step에서 이 패턴으로 경로를 얻는다
from dotenv import load_dotenv
import os
load_dotenv("/home/grass/leninbot/.env")

VENV_PYTHON = os.environ["VENV_PYTHON"]       # e.g. /home/grass/leninbot/venv/bin/python3
PROJECT_ROOT = str(Path(VENV_PYTHON).parents[2])  # venv/bin/python3 → 프로젝트 루트
LOG_DIR = f"{PROJECT_ROOT}/logs"
```

---

## 언제 사용하나
- 시스템 crontab에 주기적 작업을 등록할 때
- 기존 cron 항목을 조회하거나 삭제할 때
- cron 관련 오류를 디버깅할 때

---

## Step 1 — 환경 로드 및 현재 crontab 확인

```python
import subprocess, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/home/grass/leninbot/.env")
VENV_PYTHON = os.environ["VENV_PYTHON"]
PROJECT_ROOT = str(Path(VENV_PYTHON).parents[2])
LOG_DIR = f"{PROJECT_ROOT}/logs"

print(f"VENV_PYTHON = {VENV_PYTHON}")
print(f"PROJECT_ROOT = {PROJECT_ROOT}")

# python 실행 파일 존재 확인
assert Path(VENV_PYTHON).exists(), f"VENV_PYTHON 경로 없음: {VENV_PYTHON}"

result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)
else:
    print("crontab 비어 있음")
```

등록 전 **반드시** 현재 항목을 확인해서 중복을 방지한다.

---

## Step 2 — 스크립트 존재 및 실행 가능 여부 검증

```python
script = Path(f"{PROJECT_ROOT}/scripts/대상스크립트.py")
assert script.exists(), f"스크립트 없음: {script}"
assert script.suffix == ".py", "Python 스크립트만 지원"

# venv에서 구문 검사
result = subprocess.run(
    [VENV_PYTHON, "-c", f"import ast; ast.parse(open('{script}').read()); print('구문 OK')"],
    capture_output=True, text=True
)
print(result.stdout.strip() or result.stderr.strip())
assert result.returncode == 0, "스크립트 구문 오류"

# dry-run 가능하면 실행
result = subprocess.run(
    [VENV_PYTHON, str(script), "--help"],
    capture_output=True, text=True, timeout=10
)
print(result.stdout[:500])
```

---

## Step 3 — cron 항목 생성

**반드시 아래 템플릿을 사용한다:**

```
# {라벨} — {한 줄 설명}
# 생성: {YYYY-MM-DD}
{cron_expr} {VENV_PYTHON} {PROJECT_ROOT}/{스크립트경로} >> {LOG_DIR}/{로그파일명}.log 2>&1
```

**규칙:**
- Python 경로: `VENV_PYTHON` 값 사용 (고정 경로 직접 입력 금지)
- 로그 파일: `{LOG_DIR}/{스크립트명에서 .py 제거}.log`
- 주석 라벨: 스크립트 이름 또는 목적 (나중에 식별용)
- `>> ... 2>&1`: stdout과 stderr 모두 로그에 추가

```python
from datetime import date

label = "my-task"
description = "작업 설명"
cron_expr = "*/10 * * * *"
script_rel = "scripts/my_script.py"
log_name = "my_script"

new_entry = (
    f"# {label} — {description}\n"
    f"# 생성: {date.today().isoformat()}\n"
    f"{cron_expr} {VENV_PYTHON} "
    f"{PROJECT_ROOT}/{script_rel} "
    f">> {LOG_DIR}/{log_name}.log 2>&1"
)

# 기존 crontab 읽기
result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
current = result.stdout if result.returncode == 0 else ""

# 중복 확인
if script_rel in current:
    print(f"이미 등록됨: {script_rel}")
    print("기존 항목을 삭제하고 다시 등록하려면 Step 5 참고")
else:
    updated = current.rstrip("\n") + "\n\n" + new_entry + "\n"
    proc = subprocess.run(
        ["crontab", "-"], input=updated,
        capture_output=True, text=True
    )
    assert proc.returncode == 0, f"crontab 등록 실패: {proc.stderr}"
    print(f"등록 완료:\n{new_entry}")
```

---

## Step 4 — 등록 후 검증

```python
result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
print(result.stdout)
# 방금 등록한 항목이 보이는지 확인
```

---

## Step 5 — cron 항목 삭제

```python
target_keyword = "삭제할_스크립트명_또는_라벨"

result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
lines = result.stdout.splitlines()

# 타겟 라인과 그 위의 주석 라인 제거
filtered = []
for i, line in enumerate(lines):
    if target_keyword in line and not line.startswith("#"):
        # 바로 위 주석들도 제거
        while filtered and filtered[-1].startswith("#"):
            filtered.pop()
        continue
    filtered.append(line)

updated = "\n".join(filtered) + "\n"
proc = subprocess.run(["crontab", "-"], input=updated, capture_output=True, text=True)
assert proc.returncode == 0, f"삭제 실패: {proc.stderr}"
print(f"'{target_keyword}' 항목 삭제 완료")
```

---

## 절대 금지
- `/usr/bin/python3`, `python3`, `env python3` 등 VENV_PYTHON 외 python 사용
- VENV_PYTHON 경로를 하드코딩 (반드시 .env에서 로드)
- 로그 경로를 `/dev/null`로 보내기 (디버깅 불가)
- 주석 라벨 없이 cron 등록 (나중에 식별 불가)
- 기존 crontab 전체를 덮어쓰기 (항상 기존 내용에 추가)
- 프로젝트 루트 외부 스크립트 등록
