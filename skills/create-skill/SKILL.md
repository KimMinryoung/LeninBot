---
name: create-skill
description: Creates a new skill by scaffolding a SKILL.md file in the skills/ directory. Handles naming, frontmatter, step-by-step instructions, and validation. Use when asked to create a new skill, add a capability, or extend the skill system. Keywords: 스킬 생성, 스킬 추가, new skill, create skill, add capability, 기능 추가.
compatibility: Requires write access to skills/ directory. skills_loader.py auto-scans on restart.
metadata:
  author: cyber-Lenin
  version: "1.0"
allowed-tools: read_file write_file list_directory execute_python
---

# Create-Skill Skill

## 언제 사용하나
- 사용자가 새 스킬을 만들어달라고 요청할 때
- 반복적인 작업 패턴을 스킬로 추상화할 때
- 기존 스킬의 범위를 벗어난 새 기능이 필요할 때

---

## Step 0 — 기존 스킬 파악
```
list_directory("skills")
```
- 이미 같은 이름/기능의 스킬이 있는지 확인
- 중복이면 사용자에게 알리고 기존 스킬 수정 여부를 판단

---

## Step 1 — 스킬 명세 결정
사용자 요청에서 다음을 추출:

| 항목 | 설명 |
|------|------|
| `name` | kebab-case (예: `web-monitor`, `data-export`) |
| `description` | 한 줄 설명 + 트리거 키워드 포함 (시스템 프롬프트에 노출됨) |
| `allowed-tools` | 이 스킬이 사용할 도구 목록 (공백 구분) |
| 핵심 스텝 | 스킬 실행 절차 (최소 3스텝) |

**이름 규칙:**
- 소문자 kebab-case만 허용: `my-skill` ✓ / `MySkill` ✗ / `my_skill` ✗
- 기능을 동사-명사로 표현: `fetch-metrics`, `export-data`

---

## Step 2 — SKILL.md 작성

파일 경로: `skills/{name}/SKILL.md`

**필수 frontmatter 구조:**
```yaml
---
name: {name}
description: {한 줄 설명. Use when ... Keywords: ...}
compatibility: {의존성/환경 요구사항}
metadata:
  author: cyber-Lenin
  version: "1.0"
allowed-tools: {tool1} {tool2} ...
---
```

**본문 구조 (아래 섹션 순서 권장):**
1. `## 언제 사용하나` — 트리거 조건 명시
2. `## Step N — 단계명` — 구체적 실행 절차 (코드 블록 포함)
3. `## 품질 기준` 또는 `## 절대 금지` — 제약 조건

**좋은 SKILL.md 조건:**
- 각 Step에 실제 실행 가능한 코드/명령 포함
- 실패 시 처리(롤백, 에러 보고) 명시
- 50~150줄 유지 (너무 짧으면 지침 부족, 너무 길면 LLM이 무시)

---

## Step 3 — 파일 저장
```python
# write_file 호출 예시
write_file(
    path="skills/{name}/SKILL.md",
    content="..."
)
```
- 디렉토리는 자동 생성됨 (write_file이 처리)
- 저장 후 `read_file`로 내용 재확인

---

## Step 4 — 유효성 검증
```python
# frontmatter 파싱 테스트 (skills_loader.py 로직 재현)
import re
from pathlib import Path

text = Path("skills/{name}/SKILL.md").read_text()
fm_match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
assert fm_match, "frontmatter 없음!"

meta = {}
for line in fm_match.group(1).splitlines():
    if ":" in line and not line.startswith(" "):
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip().strip('"')

assert "name" in meta, "name 필드 없음"
assert "description" in meta, "description 필드 없음"
print(f"✓ 스킬 '{meta['name']}' 유효성 확인 완료")
```

---

## Step 5 — 시스템 프롬프트 반영 확인
`skills_loader.py`는 **봇 재시작 시** 자동 스캔한다.

```python
# 즉시 반영 확인 (재시작 없이)
import sys; sys.path.insert(0, '/home/grass/leninbot')
from importlib import reload
import skills_loader
skills_loader._skills_loaded = False  # 캐시 초기화
skills = skills_loader.load_skills()
names = [s['name'] for s in skills]
print(f"로드된 스킬: {names}")
assert '{name}' in names, "스킬이 로드되지 않음!"
print("✓ 스킬 등록 확인 완료")
```

재시작 없이 즉시 확인 가능. 실제 시스템 프롬프트 반영은 봇 재시작 후 적용.

---

## Step 6 — 사용자 보고
다음을 포함해 보고:
- 생성된 파일 경로
- 스킬 이름 및 트리거 키워드
- allowed-tools 목록
- 핵심 스텝 요약 (2~3줄)
- "봇 재시작 시 시스템 프롬프트에 자동 반영됩니다"

---

## 절대 금지
- `skills/` 외부에 파일 생성
- 기존 스킬 덮어쓰기 (확인 없이)
- frontmatter 없는 SKILL.md 생성
- 실행 불가능한 추상적 지침만 나열
