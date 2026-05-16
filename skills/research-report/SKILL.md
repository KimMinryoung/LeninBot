---
name: research-report
description: Conducts deep research on a topic using multiple sources (web, KG, vector DB) and produces a structured markdown report saved to the research/ directory. Use when asked for deep analysis, background research, or when creating_task for multi-step investigation. Keywords: 리서치, 조사, 연구, 보고서, deep research, report, analysis, background.
compatibility: Requires web search access and write access to research/ directory.
metadata:
  author: cyber-lenin
  version: "1.0"
allowed-tools: web_search knowledge_graph_search vector_search write_file read_file execute_python
---

# Research Report Skill

## 언제 사용하나
- 사용자가 특정 주제에 대해 심층 분석을 요청할 때
- `create_task`로 백그라운드 리서치가 필요할 때
- 기존 `research/` 파일을 업데이트할 때

## Step 1 — 범위 정의
리서치 주제를 명확히 정의:
- 핵심 질문 (최대 3개)
- 시간적 범위
- 지리적/주제적 범위

## Step 2 — 멀티소스 수집
병렬로 실행:
1. `web_search` × 3-5회 (다양한 각도)
2. `knowledge_graph_search` × 2회
3. `vector_search` × 1-2회 (이론 기반)

## Step 3 — 교차 검증
- 소스 간 충돌 사실 식별
- 신뢰도 낮은 정보 제거
- 핵심 팩트만 추출

## Step 4 — 보고서 작성
[assets/research-template.md](assets/research-template.md) 구조 사용.

### 각주/출처 형식 — 웹 렌더링 호환 규칙
본문에서 출처를 가리킬 때는 반드시 Markdown footnote 형식만 사용한다: `[^1]`, `[^2]`, `[^3]`.

문서 끝의 `## 주요 출처` 섹션은 반드시 footnote definition으로 작성하고, 각 항목 안에 실제 URL을 포함한다:

```markdown
본문 문장 끝에 출처를 단다.[^1]

## 주요 출처
[^1]: 매체/기관, "문서 제목", 날짜. https://example.com/source
[^2]: 다른 출처 설명. https://example.com/other
```

금지 형식: 본문 `[1]`, 번호 목록 `1. 설명 URL`, `(출처: URL)`, raw URL만 본문에 삽입, 임의의 별표/괄호 각주.
웹사이트 렌더러는 `[^n]: ... URL` 정의의 URL을 본문 `[^n]` 링크로 연결한다. 새 각주 형식을 만들지 말 것.

## Step 5 — 저장
```
파일명: research/{주제}-{YYYY-MM-DD}.md
```
`write_file`로 저장 후 사용자에게 경로 알림.

## 품질 기준
- 출처 명시 필수
- 본문 각주는 `[^n]`, 출처 목록은 `[^n]: 설명 URL` 형식만 사용
- 추측과 사실 구분
- 결론은 근거 기반
- 500줄 이하 유지 (초과 시 분리)
