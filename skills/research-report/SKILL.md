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

## Step 5 — 저장
```
파일명: research/{주제}-{YYYY-MM-DD}.md
```
`write_file`로 저장 후 사용자에게 경로 알림.

## 품질 기준
- 출처 명시 필수
- 추측과 사실 구분
- 결론은 근거 기반
- 500줄 이하 유지 (초과 시 분리)
