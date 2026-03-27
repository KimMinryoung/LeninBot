---
name: venv-package-check
description: Checks if required Python packages are installed in the leninbot venv, and installs missing ones automatically. Loads VENV_PIP from .env. Use when a script needs a package, before running new code, or when ImportError occurs. Keywords: 패키지 설치, pip install, venv, ImportError, 패키지 확인, package check, install package, 의존성.
compatibility: Requires .env with VENV_PIP set. No sudo needed — venv is user-owned.
metadata:
  author: cyber-Lenin
  version: "1.1"
allowed-tools: execute_python
---

# Venv Package Check Skill

## 언제 사용하나
- 새 코드를 실행하기 전에 의존 패키지가 설치됐는지 확인할 때
- `ImportError` 또는 `ModuleNotFoundError` 발생 시
- 사용자가 특정 패키지 설치를 요청할 때
- 새 스킬/스크립트 작성 전 환경 점검 시

---

## Step 0 — 환경 로드

**모든 Step에서 이 변수들을 사용한다. 경로 하드코딩 금지.**

```python
import subprocess, json, os
from dotenv import load_dotenv

load_dotenv()
PIP = os.environ["VENV_PIP"]
print(f"PIP = {PIP}")
```

---

## Step 1 — 설치 여부 확인

```python
def check_packages(required: list[str]) -> dict:
    """
    required: 확인할 패키지명 리스트 (소문자, import명 기준)
    returns: {"installed": [...], "missing": [...]}
    """
    result = subprocess.run(
        [PIP, "list", "--format=json"],
        capture_output=True, text=True
    )
    installed = {p["name"].lower() for p in json.loads(result.stdout)}

    # import명 → pip명 매핑 (다른 경우만)
    import_to_pip = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "bs4": "beautifulsoup4",
        "yaml": "PyYAML",
        "dotenv": "python-dotenv",
    }

    found, missing = [], []
    for pkg in required:
        pip_name = import_to_pip.get(pkg, pkg).lower()
        if pip_name in installed or pkg.lower() in installed:
            found.append(pkg)
        else:
            missing.append(pkg)

    return {"installed": found, "missing": missing}
```

---

## Step 2 — 없으면 설치

```python
def install_packages(packages: list[str]) -> dict:
    """
    packages: pip 패키지명 리스트
    returns: {"success": [...], "failed": [...]}
    """
    success, failed = [], []
    for pkg in packages:
        result = subprocess.run(
            [PIP, "install", pkg],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            success.append(pkg)
            print(f"✓ {pkg} 설치 완료")
        else:
            failed.append(pkg)
            print(f"✗ {pkg} 설치 실패:\n{result.stderr[-300:]}")
    return {"success": success, "failed": failed}
```

---

## Step 3 — 통합 실행 패턴

한 번에 확인 + 설치:

```python
import subprocess, json, os
from dotenv import load_dotenv

load_dotenv()
PIP = os.environ["VENV_PIP"]

# 1. 현재 설치된 패키지 목록
result = subprocess.run([PIP, "list", "--format=json"], capture_output=True, text=True)
installed = {p["name"].lower() for p in json.loads(result.stdout)}

# 2. 필요한 패키지 정의
required = ["matplotlib", "plotly", "loguru"]  # ← 상황에 맞게 수정

# 3. 누락 확인
missing = [pkg for pkg in required if pkg.lower() not in installed]
print(f"설치됨: {[p for p in required if p not in missing]}")
print(f"누락: {missing}")

# 4. 누락된 것만 설치
for pkg in missing:
    r = subprocess.run([PIP, "install", pkg], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"✓ {pkg} 설치 완료")
    else:
        print(f"✗ {pkg} 실패: {r.stderr[-200:]}")
```

---

## 주의사항

- **경로는 .env에서 로드**: `VENV_PIP` 환경변수 사용. 하드코딩 금지.
  - 시스템 pip(`/usr/bin/pip`) 절대 사용 금지 — 권한 없음
- **버전 지정이 필요하면**: `"package==1.2.3"` 형식 사용
- **설치 후 재시작 불필요**: 다음 `execute_python` 호출부터 바로 import 가능
- **대용량 패키지** (torch, tensorflow 등): 이미 설치돼 있음. 재설치 시도 금지
- **실패 시**: stderr 마지막 300자를 확인해 원인 파악 후 사용자에게 보고
