---
name: kg-maintenance
description: Maintains Neo4j Knowledge Graph quality: merges duplicate entities, assigns missing entity types (Person/Organization/Location/Event/Concept), removes orphaned nodes, and validates relationship integrity. Use when KG has duplicate entities, untyped nodes, or after bulk data ingestion. Keywords: KG 정리, 중복 병합, 엔티티 타입, 고아 노드, knowledge graph cleanup, deduplication.
compatibility: Requires Neo4j connection and Python 3.10+. Scripts use neo4j driver and anthropic SDK.
metadata:
  author: cyber-lenin
  version: "1.0"
allowed-tools: kg_query kg_merge_entities kg_delete_episode execute_python read_kg_status
---

# KG Maintenance Skill

## 언제 실행하나
- 대량 데이터 수집 후
- `read_kg_status`에서 untyped 엔티티가 100개 이상일 때
- 중복 엔티티가 의심될 때 (예: "미국" vs "USA" vs "United States")
- 주간 정기 정리 (매주 월요일)

## Step 1 — 현황 파악
```
read_kg_status() 실행 → 총 엔티티 수, 엣지 수, 최근 에피소드 확인
```

## Step 2 — 중복 엔티티 탐지
[scripts/dedup_entities.py](scripts/dedup_entities.py) 실행:
- 이름 유사도 기반 후보 쌍 추출
- 출력: `[(source, target, similarity_score), ...]`

병합 기준:
- 동일 실체, 다른 표기 → 병합 (한국어 우선)
- 유사하지만 다른 실체 → 병합 금지

## Step 3 — 타입 부여
[scripts/assign_types.py](scripts/assign_types.py) 실행:
- 타입 없는 엔티티에 `Person/Organization/Location/Event/Concept` 부여
- LLM 추론 기반, 배치 50개씩 처리

## Step 4 — 고아 노드 정리
[scripts/cleanup_orphans.py](scripts/cleanup_orphans.py) 실행:
- 관계가 0개인 엔티티 탐지
- 에피소드에도 미등장 확인 후 삭제

## Step 5 — 검증
```cypher
MATCH (n) RETURN n.type, count(n) ORDER BY count(n) DESC
```
타입별 분포 확인 후 이상치 재검토.

## 주의사항
- 병합은 되돌릴 수 없다. 확신이 없으면 skip
- 대량 삭제 전 반드시 백업: `scripts/backup_kg.sh`
