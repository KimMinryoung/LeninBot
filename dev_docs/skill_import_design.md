# Skill Import 설계

## 목적

agentskills.io 오픈 표준 기반의 외부 스킬을 leninbot의 `skills/` 디렉토리로 가져온다.
이 디렉토리는 봇 런타임용이며 Codex 개발 스킬과 별개다.

## 현재 스킬과 로딩 경계

| 스킬 | 용도와 참조 |
|------|------------|
| `geopolitical-analysis` | 지정학 분석 지침. `services/a2a_handler.py`가 A2A 프롬프트에 직접 로드 |
| `research-report` | 리서치·보고서 출처 형식. `services/a2a_handler.py`가 A2A 프롬프트에 직접 로드 |
| `kg-maintenance` | KG 관리 지침과 운영 스크립트. 하위 `scripts/`는 MCP 관리 도구와 KG 백업에서도 참조 |

`telegram/bot.py`는 `skills_loader.build_skills_prompt()`로 이름·설명 목록을
orchestrator 프롬프트에 넣는다. 본문은 필요할 때 `read_file`로 읽도록 안내한다.
전문 에이전트에 이 목록이 자동 주입되는 것은 아니다.

`skills_loader.py`는 최초 로드 결과를 프로세스 안에 캐시한다. 스킬 추가·삭제는
해당 프로세스에서 `reload_skills()`를 호출하거나 봇을 재시작해야 목록에 반영된다.
별도 CLI 프로세스에서 목록을 확인해도 실행 중인 봇의 캐시는 갱신되지 않는다.

## 외부 스킬 소스

| 소스 | 형태 | 예시 |
|------|------|------|
| GitHub repo | `owner/repo` 또는 `owner/repo/path` | `anthropics/skills/frontend-design` |
| skills.sh | 레지스트리 검색 → GitHub repo로 해소 | 검색 후 repo URL 획득 |
| 로컬 디렉토리 | 절대/상대 경로 | `/tmp/my-skill/` |

## agentskills.io 표준 vs leninbot 포맷

두 포맷은 거의 동일하다. 차이점은 `allowed-tools`의 tool 이름뿐.

### frontmatter 호환성

| 필드 | agentskills.io | leninbot | 호환 |
|------|----------------|----------|------|
| name | 필수 | 필수 | 동일 |
| description | 필수 | 필수 | 동일 |
| license | 선택 | 없음 | 무시 가능 |
| compatibility | 선택 | 선택 | 동일 |
| metadata | 선택 | 선택 | 동일 |
| allowed-tools | 선택 (Claude Code 형식) | 선택 (leninbot 형식) | **변환 필요** |

### allowed-tools 매핑

외부 스킬은 Claude Code 스타일 tool 이름을 사용한다:

| 외부 (agentskills.io) | leninbot 내부 |
|------------------------|---------------|
| `Bash` / `Bash(*)` | `execute_python` (Python via subprocess) |
| `Read` | `read_file` |
| `Write` / `Edit` | `write_file` |
| `Glob` / `Grep` | `list_directory` |
| `WebSearch` | `web_search` |
| `WebFetch` | `fetch_url` |

매핑에 없는 tool은 그대로 유지 (leninbot이 무시).

### 본문 변환

SKILL.md 본문은 마크다운이므로 변환 없이 그대로 사용 가능.
단, 외부 스킬이 `Bash`, `Read` 등의 Claude Code tool을 직접 호출하는 코드 블록을
포함할 수 있는데, 이는 leninbot에서 `execute_python`으로 실행해야 한다.
본문 자체는 자동 변환하지 않고, import 시 경고만 출력한다.

## Import 흐름

```
1. 소스 해석
   - GitHub URL → git sparse-checkout 또는 API로 디렉토리 다운로드
   - 로컬 경로 → 직접 복사

2. 검증
   - SKILL.md 존재 확인
   - frontmatter 파싱 (name, description 필수)
   - name이 leninbot 기존 스킬과 충돌하는지 확인

3. 변환
   - allowed-tools 매핑 적용
   - metadata.source에 원본 소스 기록
   - metadata.imported_at에 import 시각 기록

4. 설치
   - skills/{name}/ 디렉토리로 복사
   - SKILL.md + 하위 파일(scripts/, references/, assets/) 포함

5. 확인
   - skills_loader.py로 로드 테스트
   - 결과 출력
```

## 디렉토리 구조

```
scripts/
  import_skill.py      ← import 스크립트 (CLI)

skills/
  {imported-skill}/
    SKILL.md            ← 변환된 스킬
    references/         ← 원본 그대로
    scripts/            ← 원본 그대로
    assets/             ← 원본 그대로
```

## 사용법

```bash
# GitHub에서 import
venv/bin/python3 scripts/import_skill.py github anthropics/skills/frontend-design

# 로컬에서 import
venv/bin/python3 scripts/import_skill.py local /tmp/my-skill

# 기존 스킬 덮어쓰기
venv/bin/python3 scripts/import_skill.py github owner/repo/skill-name --force

# 목록 확인
venv/bin/python3 scripts/import_skill.py list
```

## 제한사항

- 외부 스킬의 `scripts/` 내 실행 파일은 보안 검토 없이 자동 실행하지 않음
- leninbot 자체 tool(`execute_python` 등)으로 실행 불가능한 스킬은 경고 출력
- 라이선스 확인은 사용자 책임 (license 필드 출력만 함)
