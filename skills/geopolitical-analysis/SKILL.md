---
name: geopolitical-analysis
description: Performs structured geopolitical analysis using dialectical materialist framework. Combines KG entity lookup, ML theory vector search, and live web news into a unified situation report. Use when asked about international conflicts, state relations, wars, sanctions, diplomatic events, or any geopolitical situation. Keywords: 전쟁, 외교, 제재, 분쟁, 지정학, war, conflict, sanctions, diplomacy, geopolitics.
compatibility: Requires Neo4j KG, pgvector DB, and web search access. Designed for Cyber-Lenin (leninbot) agent.
metadata:
  author: cyber-lenin
  version: "1.0"
allowed-tools: knowledge_graph_search vector_search web_search write_kg_structured
---

# Geopolitical Analysis Skill

분석 요청이 들어오면 아래 순서를 반드시 따른다. 순서를 건너뛰지 말 것.

## Step 1 — KG 엔티티 조회
`knowledge_graph_search`로 관련 국가, 조직, 인물, 사건을 검색한다.
- 최소 2회 검색: 행위자(actors) + 사건/관계(events/relations)
- KG에 없으면 Step 2로 넘어간다.

## Step 2 — 이론적 프레임 확보
`vector_search(layer="core_theory")`로 관련 마르크스-레닌주의 이론 발췌를 검색한다.
- 제국주의, 자본주의 모순, 계급 분석 관련 텍스트 우선
- 현대 분석이 필요하면 `layer="modern_analysis"`도 병행

## Step 3 — 최신 정보 보강
`web_search`로 최근 24-72시간 뉴스를 확인한다.
- KG 데이터가 오래됐을 경우 필수
- 검색어는 영어로 작성해 결과 품질 높일 것

## Step 4 — 종합 분석 작성
[assets/report-template.md](assets/report-template.md) 구조를 따라 분석을 작성한다.

## Step 5 — KG 업데이트 (선택)
새로 확인된 중요 사실이 있으면 `write_kg_structured`로 저장한다.
- 이미 KG에 있는 정보는 중복 저장 금지
- 출처가 불명확한 정보는 저장 금지

## 분석 원칙
- **변증법적 유물론**: 물질적 이해관계(자원, 자본, 군사력)를 우선 분석
- **계급 관점**: 국가 행위자 뒤의 계급적 이해관계를 드러낼 것
- **제국주의 분석**: 레닌의 제국주의론([references/imperialism-framework.md](references/imperialism-framework.md)) 적용
- **블러프 금지**: 데이터 없으면 없다고 말할 것

## 엣지 케이스
- KG와 웹 정보가 충돌 → 웹 최신 정보 우선, 충돌 사실 명시
- 분쟁 당사자 모두 자국 유리한 주장 → 양측 주장 병기 후 물질적 증거 기준으로 판단
- 정보 부족 → 추측 금지, 불확실성 명시
