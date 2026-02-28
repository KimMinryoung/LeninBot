# Knowledge Graph 설계 문서

> 지식 그래프 관련 작업 전에 반드시 이 문서를 확인하고, 작업 후에는 변경사항을 반영할 것.

---

## 1. 개요

### 목적

Cyber-Lenin의 **정보 에이전트** 기능을 위한 지식 그래프 모듈.
구조화된 엔티티·관계 네트워크를 통해 단순 벡터 검색이 놓치는 연결 관계와 시간축 정보를 캡처한다.

### 스택

| 구성요소 | 기술 |
|----------|------|
| 그래프 프레임워크 | [Graphiti](https://github.com/getzep/graphiti) (바이-템포럴 지식 그래프) |
| 그래프 DB | Neo4j AuraDB (클라우드 관리형) |
| LLM | Gemini 2.5 Flash (추출), Gemini 2.5 Flash-Lite (리랭킹) |
| 임베딩 | gemini-embedding-001 (1024차원) |
| Python 패키지 | `graphiti-core[google-genai]` |

**OpenAI 의존성 없음** — 전부 Gemini 기반.

### chatbot.py 통합 미정 사유

- 지식 그래프 데이터가 아직 테스트 에피소드 수준 (3건)
- 데이터 수집 파이프라인 미구축 (크롤러 → 에피소드 변환 → ingest)
- chatbot 통합 시 어떤 노드에서 그래프를 호출할지 설계 필요 (retrieve? 별도 노드?)
- 그래프 검색 결과와 벡터 검색 결과 병합 전략 미정

→ 독립 모듈로 충분히 성숙시킨 후 통합 예정.

---

## 2. 아키텍처

### 파일 구조

```
graph_memory/
├── __init__.py       # 패키지 exports: GraphMemoryService, ENTITY_TYPES, EDGE_TYPES, etc.
├── entities.py       # 7개 엔티티 타입 정의 (Pydantic) — v2
├── edges.py          # 10개 엣지 타입 정의 (Pydantic) — v2
├── config.py         # EDGE_TYPE_MAP, EXCLUDED_ENTITY_TYPES, EPISODE_SOURCE_MAP
└── service.py        # GraphMemoryService 클래스 (핵심)
```

### GraphMemoryService API

```python
class GraphMemoryService:
    async def initialize() -> None
        # Neo4j 연결, Gemini LLM/Embedder/Reranker 초기화
        # build_indices_and_constraints() 호출

    async def ingest_episode(
        name, body, source_type, reference_time, group_id,
        source_description=None
    ) -> None
        # 에피소드 1건 수집. ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP 자동 주입.
        # source_type → EPISODE_SOURCE_MAP에서 EpisodeType + 설명 자동 매핑.

    async def ingest_episodes_bulk(episodes: list[dict]) -> None
        # 순차 수집 (rate limit 대응). 각 dict는 ingest_episode() 인자.

    async def search(
        query, group_ids=None, edge_types=None, node_labels=None,
        center_node_uuid=None, num_results=10
    ) -> dict  # {"nodes": [...], "edges": [...]}
        # BM25 + cosine + RRF 하이브리드 검색.

    async def generate_briefing(topic, group_ids) -> dict
        # 주제 기반 전략 브리핑 데이터 수집.
        # 메인 검색(limit=30) + 사건 검색(limit=10) + 타임라인 정렬.

    async def close() -> None
        # Graphiti 연결 종료.
```

### 초기화 흐름

```
GraphMemoryService()
  → .initialize()
    → load_dotenv()
    → GeminiClient(gemini-2.5-flash / gemini-2.0-flash-lite)
    → GeminiEmbedder(gemini-embedding-001)
    → GeminiRerankerClient(gemini-2.0-flash-lite)
    → Neo4jDriver(uri, user, password, database)
    → Graphiti(llm_client, embedder, cross_encoder, graph_driver)
    → build_indices_and_constraints()
```

---

## 3. 스키마 (v2 — 레닌 프레임워크 갭 분석 기반 재설계)

> 재설계 근거: `temp_dev/entity_redesign_form.md` 참조

### 엔티티 7종 (53 필드)

| 엔티티 | 필드 수 | 설명 | 주요 필드 |
|--------|---------|------|-----------|
| **Person** | 9 | 개인 인물 | alias, nationality, role, expertise, ideological_alignment, network_role, recruitment_potential, reliability_rating, influence_level |
| **Organization** | 10 | 조직 | org_type, industry, headquarters, country, parent_org, ideological_orientation, alliance_bloc, state_sponsor, threat_classification, known_ttps |
| **Location** | 5 | 지리적 장소 | location_type, coordinates, significance, strategic_resources, geopolitical_bloc |
| **Asset** | 7 | 자산/기술 | asset_type, classification, strategic_value, description_detail, supply_chain_role, dual_use_potential, controlling_entity |
| **Incident** | 9 | 특이사건 | incident_type, severity, occurred_at, detected_at, status, confidence, impact_summary, geopolitical_context, information_source_type |
| **Policy** | 6 | 정책/제도 (신설) | policy_type, issuing_entity, target_scope, status, effective_date, strategic_impact |
| **Campaign** | 7 | 캠페인/작전 (신설) | campaign_type, objective, status, scale, started_at, ideological_framing, effectiveness |

- 모든 필드는 `Optional` — 수집 누적 시 Graphiti 엔티티 해소(entity resolution)가 점진적으로 채움
- `ENTITY_TYPES` 레지스트리: `{"Person": Person, ..., "Policy": Policy, "Campaign": Campaign}`
- v1→v2 변경: Person -clearance_level +3, Organization -employee_count +3, Location +2, Asset +3, Incident +2, Policy 신설, Campaign 신설

### 엣지 10종

| 엣지 | 설명 | 주요 필드 |
|------|------|-----------|
| **Affiliation** | 사람→조직 소속 | position, department, affiliation_type, start_date, end_date, is_current, access_level |
| **PersonalRelation** | 사람↔사람 | relation_type, context, strength, first_observed |
| **OrgRelation** | 조직↔조직 | relation_type, agreement_type, financial_value, strategic_significance |
| **Funding** | 자금 흐름 | funding_type, amount, purpose, is_verified |
| **AssetTransfer** | 기술/자산 이전 | transfer_type, asset_description, legality, export_control |
| **ThreatAction** | 공격/위협 행위 | action_type, technique, target_asset, outcome, confidence |
| **Involvement** | 엔티티→사건/캠페인 관여 | role_in_incident, evidence_basis, confidence |
| **Presence** | 엔티티→장소 관련 | presence_type, frequency, purpose |
| **PolicyEffect** | 정책↔엔티티 영향 (신설) | effect_type, impact_description, compliance_status |
| **Participation** | 엔티티→캠페인 참여 (신설) | role, contribution, commitment_level |

- Graphiti 바이-템포럴 타임스탬프 자동 부여: `valid_at`/`invalid_at`, `created_at`/`expired_at`
- `EDGE_TYPES` 레지스트리: `{"Affiliation": Affiliation, ..., "PolicyEffect": PolicyEffect, "Participation": Participation}`

### EDGE_TYPE_MAP (24 entries)

어떤 엔티티 쌍 사이에 어떤 관계가 허용되는지 정의. 매핑에 없는 쌍은 `RELATES_TO`로 캡처.

| 소스 → 타겟 | 허용 엣지 |
|-------------|-----------|
| Person → Organization | Affiliation, Funding, AssetTransfer, ThreatAction |
| Organization → Person | ThreatAction |
| Person → Person | PersonalRelation, Funding, AssetTransfer |
| Organization → Organization | OrgRelation, Funding, AssetTransfer, ThreatAction |
| Person → Incident | Involvement |
| Organization → Incident | Involvement |
| Organization → Asset | AssetTransfer |
| Person → Asset | AssetTransfer |
| Person → Location | Presence |
| Organization → Location | Presence |
| Incident → Location | Presence |
| Policy → Organization | PolicyEffect |
| Policy → Person | PolicyEffect |
| Policy → Asset | PolicyEffect |
| Policy → Location | PolicyEffect |
| Organization → Policy | PolicyEffect |
| Person → Campaign | Participation, Involvement |
| Organization → Campaign | Participation, Involvement |
| Campaign → Organization | ThreatAction |
| Campaign → Asset | ThreatAction |
| Campaign → Location | Presence |
| Campaign → Incident | Involvement |
| Campaign → Policy | PolicyEffect |
| Entity → Entity (폴백) | Funding, AssetTransfer |

### EPISODE_SOURCE_MAP (12 소스 타입)

| 소스 키 | EpisodeType | 설명 |
|---------|-------------|------|
| osint_news | text | Open source news article |
| osint_social | text | Social media post or thread |
| osint_forum | text | Forum post (dark web / public) |
| cve_feed | json | CVE vulnerability feed entry |
| threat_report | text | Threat intelligence report |
| internal_siem | json | Internal SIEM alert or log |
| internal_report | text | Internal analyst report or memo |
| humint_debrief | text | Human intelligence debrief notes |
| financial_record | json | Financial transaction or filing record |
| patent_filing | json | Patent application or grant record |
| personnel_change | text | Personnel movement notice |
| diplomatic_cable | text | Diplomatic or policy communication |

---

## 4. 인프라 설정

### 환경 변수

| 변수 | 설명 | 예시 |
|------|------|------|
| `NEO4J_URI` | AuraDB 연결 URI | `neo4j+s://0b6bbc24.databases.neo4j.io` |
| `NEO4J_USER` | DB 사용자 | `0b6bbc24` (인스턴스 ID) |
| `NEO4J_PASSWORD` | DB 비밀번호 | `.env` 참조 |
| `NEO4J_DATABASE` | DB 이름 | `0b6bbc24` (인스턴스 ID) |
| `GEMINI_API_KEY` | Gemini API 키 | `.env` 참조 |
| `SEMAPHORE_LIMIT` | Graphiti 동시성 제한 | 기본 20, rate limit 시 1로 낮추기 |

### LLM 모델 구성

| 용도 | 모델 | 클라이언트 |
|------|------|-----------|
| 엔티티/엣지 추출 | gemini-2.5-flash | GeminiClient (model) |
| 리랭킹/속성 추출 | gemini-2.5-flash-lite | GeminiClient (small_model) + GeminiRerankerClient |
| 임베딩 | gemini-embedding-001 | GeminiEmbedder (1024차원 벡터) |

### Neo4j AuraDB 특이사항

1. **DB 이름 = 인스턴스 ID** — `neo4j`가 아닌 인스턴스 ID (예: `0b6bbc24`)를 `NEO4J_DATABASE`로 사용해야 함
2. **`database` 파라미터 필수** — `Neo4jDriver` 생성 시 반드시 `database=` 전달
3. **벡터 인덱스 수동 생성 필요** — `build_indices_and_constraints()`가 벡터 인덱스를 자동 생성하지 않음
4. **AuraDB Free Tier** — 노드/관계 수 제한 있음, 대규모 수집 시 유의

### 벡터 인덱스 수동 생성

`temp_dev/create_vector_indexes.py` 스크립트 사용:

```cypher
-- Entity 이름 임베딩 (1024 dim, cosine)
CREATE VECTOR INDEX entity_name_embedding IF NOT EXISTS
FOR (n:Entity) ON (n.name_embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
}}

-- Edge 사실 임베딩 (1024 dim, cosine)
CREATE VECTOR INDEX edge_fact_embedding IF NOT EXISTS
FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
}}
```

---

## 5. 알려진 제약 / 주의사항

### 검색

- **BM25 + cosine + RRF 리랭커** 조합 사용
- **cross_encoder 리랭커 사용 불가** — Graphiti 기본값인 cross_encoder 리랭커를 사용하면 빈 결과 반환. 반드시 `EdgeReranker.rrf` / `NodeReranker.rrf`로 설정
- SearchConfig에서 `limit=N` 사용 (`num_results` 아님)

### 에피소드 수집

- **`add_episode()`에 `uuid` 파라미터 생략 필수** — uuid를 전달하면 Graphiti가 기존 에피소드 조회를 시도하여 `NodeNotFoundError` 발생
- **`custom_extraction_instructions` 적용** — `CUSTOM_EXTRACTION_INSTRUCTIONS`로 엔티티 영어 통일 + 관계 타입 10종 제한. ~85% 준수율.
- **에피소드당 ~15-20 LLM 호출** — 엔티티 추출 → 노드 해소 → 속성 추출 → 엣지 추출 → 엣지 해소 → 엣지 속성 추출 → 커뮤니티 감지
- **Rate limit 주의** — Gemini Free Tier에서 한 에피소드 수집만으로 429 발생 가능. `SEMAPHORE_LIMIT=1`로 낮추거나 에피소드 간 딜레이 추가 필요
- **벡터 인덱스 없으면 검색 실패** — `entity_name_embedding`, `edge_fact_embedding` 인덱스 수동 생성 필수 (섹션 4 참조)
- **관계 타입은 `r.name` 프로퍼티** — Neo4j에서 `RELATES_TO` 엣지의 관계 타입은 `r.name` 필드에 저장됨 (`r.relation_type` 아님)

### Graphiti 라이브러리 특이사항

- `search_()` 메서드 사용 (밑줄 포함) — `SearchConfig(limit=N)` 전달
- `EpisodeType` enum: `text`, `message`, `json`
- 엔티티 해소(entity resolution)는 Graphiti가 자동 수행 — 동일 엔티티를 다른 에피소드에서 언급하면 자동 병합

---

## 6. 현재 데이터 상태

### AuraDB — 새 인스턴스 (v2 스키마, 2026-02-27 초기화)

> 새 AuraDB 인스턴스로 교체 후 v2 스키마로 초기화 완료.

**통계**: 에피소드 10건, 엔티티 132개, 관계 121개 (2026-02-28 영어 정규화 재수집 후)

| 엔티티 타입 | 수 |
|------------|-----|
| Location | 44 |
| Organization | 44 |
| Asset | 17 |
| Person | 15 |
| Incident | 13 |
| Policy | 9 |
| Entity (기본) | 6 |
| Campaign | 5 |

**에피소드 10건** (2026-02-27 국제 뉴스):

| # | 주제 | source_type | group_id |
|---|------|-------------|----------|
| 1 | 파키스탄-아프간 공개전쟁 선언 | osint_news | geopolitics_conflict |
| 2 | 러-우 전쟁 1464일차 전황 | osint_news | geopolitics_conflict |
| 3 | 러-우 평화협상 교착 | diplomatic_cable | geopolitics_diplomacy |
| 4 | 미-이란 제네바 핵협상 | diplomatic_cable | geopolitics_diplomacy |
| 5 | 수단 내전 제노사이드 경고 | osint_news | geopolitics_conflict |
| 6 | 미 대법원 IEEPA 관세 위헌 판결 | osint_news | geopolitics_economy |
| 7 | 영국 대러 에너지 제재 패키지 | osint_news | geopolitics_economy |
| 8 | 베네수엘라 마두로 체포 후 제재 완화 | osint_news | geopolitics_economy |
| 9 | 트럼프 그린란드 관세 분쟁 | osint_news | geopolitics_economy |
| 10 | UNCTAD 글로벌 무역 보고서 | osint_news | geopolitics_economy |

### group_id 목록

| group_id | 설명 |
|----------|------|
| `geopolitics_conflict` | 전쟁/분쟁 뉴스 |
| `geopolitics_diplomacy` | 외교/협상 뉴스 |
| `geopolitics_economy` | 경제/제재/무역 뉴스 |

### 검증 이력

- 검색 테스트 3건 통과: Pakistan-Afghanistan, Russia-Ukraine, US tariffs (각 노드 5개+엣지 5개 반환)

---

## 7. 변경 이력

| 날짜 | 변경사항 |
|------|----------|
| 2026-02-27 | graph_memory/ 모듈 최초 구축. 5 엔티티, 8 엣지, GraphMemoryService 클래스 완성. |
| 2026-02-27 | Neo4j AuraDB 연결. 벡터 인덱스 수동 생성 (`create_vector_indexes.py`). |
| 2026-02-27 | 테스트 에피소드 3건 수집 (OSINT 뉴스, SIEM 알림, HUMINT 인사정보). 검색 동작 검증 완료. |
| 2026-02-27 | 본 설계 문서 (`knowledge_graph_design.md`) 작성. |
| 2026-02-27 | **v2 스키마 재설계**: 레닌 프레임워크 갭 분석 기반. 엔티티 5→7종(+Policy, +Campaign), 엣지 8→10종(+PolicyEffect, +Participation). 기존 5종 필드 확장. `entity_redesign_form.md` 참조. AuraDB 재수집 필요. |
| 2026-02-27 | **새 AuraDB 인스턴스로 교체**. v2 스키마 초기화. 국제 뉴스 10건 수집 (전쟁 3, 외교 2, 경제 5). 엔티티 153개, 관계 319개. |
| 2026-02-27 | **엔티티 datetime→str 수정**: Incident.occurred_at/detected_at, Policy.effective_date, Campaign.started_at를 `Optional[str]`로 변경. Neo4j DateTime JSON 직렬화 오류 수정. |
| 2026-02-27 | **Graphiti prompt_helpers.py 패치**: `to_prompt_json()`에 `_Neo4jDateTimeEncoder` 추가. Graphiti 내부 타임스탬프(created_at 등)의 Neo4j DateTime 직렬화 오류 해결. |
| 2026-02-27 | **LLM 모델 변경**: small_model/reranker `gemini-2.0-flash-lite` → `gemini-2.5-flash-lite` (2.0-flash-lite rate limit 소진). |
| 2026-02-28 | **엔티티/관계 정규화**: `CUSTOM_EXTRACTION_INSTRUCTIONS` 추가(영어 강제+관계 타입 10종 제한), `NEWS_PREPROCESS_PROMPT_TEMPLATE` 영어 출력 전환. 기존 데이터 전체 삭제 후 재수집. 한국어 엔티티 0개, 관계 타입 85% 정규화 달성. |

---

## 8. 로드맵

### 단기 (데이터 수집)

- [x] AuraDB 새 인스턴스 초기화 + v2 스키마로 뉴스 10건 수집 완료
- [ ] 크롤러 → 에피소드 변환 파이프라인 구축
  - `crawler_modern.py` 출력 → `osint_news` 소스 타입으로 자동 변환
  - 뉴스 기사 단위로 에피소드 분할
- [ ] 대량 수집 스크립트 작성 (rate limit 대응 딜레이 포함)
- [ ] group_id 체계 확정 (산업별? 주제별? 시기별?)

### 중기 (chatbot 통합)

- [ ] chatbot.py 통합 조건 정의
  - 최소 에피소드 수 (예: 50건 이상)
  - 그래프 검색 품질 기준
- [ ] 통합 방식 설계
  - 별도 `graph_retrieve` 노드 vs. 기존 `retrieve` 노드 확장
  - 벡터 검색 결과 + 그래프 검색 결과 병합 전략
- [ ] generate_briefing()을 활용한 전략 브리핑 기능

### 장기 (고도화)

- [ ] 실시간 수집 — 크롤러 스케줄러 + 자동 ingest
- [ ] 그래프 시각화 API
- [ ] 엔티티/관계 통계 대시보드
- [ ] 커뮤니티 감지 기반 클러스터 분석 활용
