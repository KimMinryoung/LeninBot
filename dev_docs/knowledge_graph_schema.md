# 지식그래프 스키마 설계

**버전:** v3.0 (저장소 간 허브)  
**최종 확인 기준:** 2026-09-03 코드 트리  
**기반 프레임워크:** Graphiti + Pydantic

---

## 1. 엔티티 타입 (10 종 + 동기화 전용 1 종)

| 엔티티 | 설명 | 주요 필드 |
|--------|------|-----------|
| **Person** | 개인 인물 (주요인물, 전문가, 위협행위자, 활동가, 정보원 등) | `alias`, `nationality`, `role`, `expertise`, `ideological_alignment`, `network_role`, `recruitment_potential`, `reliability_rating`, `influence_level` |
| **Organization** | 조직 (기업, 정부기관, 군, 연구소, 위협그룹, NGO, 정당, 언론사 등) | `org_type`, `industry`, `headquarters`, `country`, `parent_org`, `ideological_orientation`, `alliance_bloc`, `state_sponsor`, `threat_classification`, `known_ttps` |
| **Location** | 지리적 장소 (시설, 거점, 군사기지, 초크포인트, 분쟁지역 등) | `location_type`, `coordinates`, `significance`, `strategic_resources`, `geopolitical_bloc` |
| **Asset** | 자산 (기술, 제품, IP, 인프라, 무기체계, 공급망 노드 등) | `asset_type`, `classification`, `strategic_value`, `description_detail`, `supply_chain_role`, `dual_use_potential`, `controlling_entity` |
| **Incident** | 사건 (사이버공격, 인사변동, 정책변화, 군사배치, 단속 등) | `incident_type`, `severity`, `occurred_at`, `detected_at`, `status`, `confidence`, `impact_summary`, `geopolitical_context`, `information_source_type` |
| **Policy** | 정책 (제재, 조약, 수출통제, 무역협정, 군사교리, 법률, 행정명령 등) | `policy_type`, `issuing_entity`, `target_scope`, `status`, `effective_date`, `strategic_impact` |
| **Campaign** | 캠페인 (군사작전, 영향력공작, 사이버캠페인, 사회운동, 선전전, 경제전 등) | `campaign_type`, `objective`, `status`, `scale`, `started_at`, `ideological_framing`, `effectiveness` |
| **Concept** | 추상 개념 (이데올로기, 이론, 사회현상, 사회계층, 역사적 시기 등) | `concept_type`, `domain`, `related_thinkers`, `historical_period`, `contemporary_relevance` |
| **Role** | 직책/직위 자체 (그 직책을 점유하는 사람과 분리) | `role_type`, `domain`, `jurisdiction`, `seniority`, `selection_method` |
| **Industry** | 산업, 섹터, 가치사슬 노드 | `sector_type`, `value_chain_position`, `strategic_importance`, `capital_intensity`, `labor_composition` |
| **Document** (동기화 전용) | 다른 저장소에 발행된 문서 — 리서치 문서, 사료 번역 발행본, 자율 프로젝트 종합 노트. `jobs/kg_sync_documents`만 생성하며 `write_kg_structured` 툴과 LLM 추출(`EXTRACTION_ENTITY_TYPES`)에서는 제외 | `doc_kind`, `slug`, `url`, `lang`, `published_at`, `content_sha256` |

### 1.1 정체성 속성 (모든 Entity, 2026-09-03)

graphiti 기본 속성(`uuid`, `name`, `summary`, `group_id`, `created_at`, `name_embedding`) 외에 `kg_runtime/identity.py`가 관리하는 속성. 새 노드는 `EntityNode.attributes`로, 기존 노드는 `upsert_identity_*`로 기록된다.

| 속성 | 타입 | 설명 |
|------|------|------|
| `external_ids` | list[str] | 다른 저장소의 안정 id. `commulingo:person:<slug>`, `commulingo:term:<slug>`, `commulingo:event:<slug>`, `commulingo:office:<id>`, `commulingo:location:<slug>`, `research:<slug>`, `archival:<id>`, `autonote:<id>`, `collection:<kind>` |
| `aliases` | list[str] | 표시용 별칭 (영문명·키릴·한국어 별칭 등) |
| `alias_keys` | list[str] | `normalize_alias_key()` 정규화 키 — name·name_ko·name_en과 **강한** 별칭(완전한 이름 형태). 해석과 AliasIndex 매칭에 사용 |
| `weak_keys` | list[str] | 성(姓)만 있는 별칭 등 단독으로 실체를 특정 못 하는 키("카스트로", "Khrushchev", "레닌"). 해석에는 같은 라벨 노드가 정확히 하나일 때만, 검색 매칭에는 유일할 때만 쓴다 (`is_weak_alias`) |
| `name_ko` / `name_en` | str | 언어별 정식 명칭 |
| `alias_text` | str | aliases를 " / "로 이은 문자열 (풀텍스트 인덱스 `entity_alias_text`) |

엔티티 해석 순서(`resolve_entity_*`): ① `external_ids` 포함 ② 강한 키·정확 이름 일치(그룹 무관, 같은 라벨만; 라벨 불일치는 로그) ③ 약한 키 — 같은 라벨 노드가 정확히 하나일 때만 ④ `KG_RESOLVE_EMBEDDING_NN=1`일 때 이름 임베딩 최근접(cosine ≥ 0.92, 같은 라벨). 들어오는 엔티티의 약한 별칭은 조회에 쓰지 않는다 — 2026-09-03 첫 미러에서 성 별칭이 피델·라울 카스트로 등 40쌍을 한 노드로 묶은 뒤 도입한 규칙.

---

## 2. 엣지 타입 (12 종 + 동기화 전용 1 종)

| 엣지 | 소스 → 타겟 | 설명 | 주요 필드 |
|------|-------------|------|-----------|
| **Affiliation** | Person → Organization | 소속 관계 (고용, 멤버십, 자문, 계약직 등) | `position`, `department`, `affiliation_type`, `start_date`, `end_date`, `is_current`, `access_level` |
| **PersonalRelation** | Person → Person | 대인 관계 (동료, 가족, 멘토, 공모자 등) | `relation_type`, `context`, `strength`, `first_observed` |
| **OrgRelation** | Organization → Organization | 조직 간 관계 (파트너십, 경쟁, 공급망, 합작 등) | `relation_type`, `agreement_type`, `financial_value`, `strategic_significance` |
| **Funding** | Person/Org → Org/Person | 자금 흐름 (투자, 보조금, 기부, 계약금 등) | `funding_type`, `amount`, `purpose`, `is_verified` |
| **AssetTransfer** | Person/Org → Asset/Person/Org | 기술/자산 이전 (기술이전, IP 라이선싱, 인력이동 등) | `transfer_type`, `asset_description`, `legality`, `export_control` |
| **ThreatAction** | Person/Org/Campaign → Org/Person/Asset | 공격/위협 행위 (사이버공격, 첩보, 사보타주 등) | `action_type`, `technique`, `target_asset`, `outcome`, `confidence` |
| **Involvement** | Person/Org → Incident/Campaign | 사건/캠페인 관여 (가해자, 피해자, 목격자, 조사자 등) | `role_in_incident`, `evidence_basis`, `confidence` |
| **Presence** | Person/Org/Incident/Campaign → Location | 위치 관련 (본사, 운영지역, 방문, 주둔 등) | `presence_type`, `frequency`, `purpose` |
| **PolicyEffect** | Policy → Entity / Org → Policy | 정책 효과 (제재, 규제, 면제, 위반 등) | `effect_type`, `impact_description`, `compliance_status` |
| **Participation** | Person/Org → Campaign | 캠페인 참여 (주도, 수행, 지원, 반대, 자금조달 등) | `role`, `contribution`, `commitment_level` |
| **Statement** | Entity → Entity | 발화, 성명, 발표, 비판, 저술, 공개 주장 | `statement_type`, `medium`, `audience`, `statement_date`, `verbatim_excerpt` |
| **Causation** | Entity → Entity | 원인, 기여요인, 촉발, 가속, 완화 같은 분석적 인과 관계 | `causal_type`, `confidence`, `mechanism`, `evidence_basis` |
| **Reference** (동기화 전용) | Document → Entity / Concept ↔ Concept / Person·Organization → Concept / Concept → Incident·Campaign·Person / Incident → Concept | 문서·큐레이션 참조. `reference_type` ∈ about, mentions, collection, related_term, parent_term, category, person_term, event_term, people_group. `validate_fact(allow_sync_predicates=True)`로만 통과 (`REFERENCE_EDGE_PAIRS`, `sync_predicate_allowed`) | `reference_type`, `note` |

모든 동기화 엣지는 `attributes.sync_key`(예: `commulingo:event_person:<event>:<person>`, `doc:research:<slug>:mention:<uuid8>`)를 갖고, 문서 유래 엣지는 `attributes.doc_ref`도 갖는다. 재실행은 sync_key로 멱등이며, 사라진 원본 행은 `expired_at`으로 만료된다(삭제 안 함).

---

## 3. 엣지 타입 매핑 (EDGE_TYPE_MAP)

| 소스 엔티티 | 타겟 엔티티 | 허용 엣지 |
|-------------|-------------|-----------|
| Person | Organization | Affiliation, Funding, AssetTransfer, ThreatAction |
| Organization | Person | ThreatAction |
| Person | Person | PersonalRelation, Funding, AssetTransfer |
| Organization | Organization | OrgRelation, Funding, AssetTransfer, ThreatAction |
| Person | Incident | Involvement |
| Organization | Incident | Involvement |
| Organization | Asset | AssetTransfer |
| Person | Asset | AssetTransfer |
| Person | Location | Presence |
| Organization | Location | Presence |
| Incident | Location | Presence |
| Policy | Organization | PolicyEffect |
| Policy | Person | PolicyEffect |
| Policy | Asset | PolicyEffect |
| Policy | Location | PolicyEffect |
| Organization | Policy | PolicyEffect |
| Person | Campaign | Participation, Involvement |
| Organization | Campaign | Participation, Involvement |
| Campaign | Organization | ThreatAction |
| Campaign | Asset | ThreatAction |
| Campaign | Location | Presence |
| Campaign | Incident | Involvement |
| Campaign | Policy | PolicyEffect |
| Person | Role | Affiliation |
| Role | Organization | Affiliation |
| Role | Location | Presence |
| Organization | Industry | Affiliation |
| Industry | Location | Presence |
| Policy | Industry | PolicyEffect |
| Industry | Asset | AssetTransfer |
| Campaign | Industry | ThreatAction |
| `Entity` fallback | `Entity` fallback | Funding, AssetTransfer, Statement, Causation |

`Entity` fallback에는 `Statement`와 `Causation`이 포함된다. 두 관계는 주제 독립적이므로 특정 entity 쌍에만 묶지 않는다.

`Reference`는 EDGE_TYPE_MAP에 **넣지 않는다** — 그 맵은 graphiti 추출기에 그대로 전달되므로 추출기가 모르는 술어가 섞이면 안 된다. 허용 쌍은 `config.REFERENCE_EDGE_PAIRS`(+ Document → 모든 타입)에 따로 있다.

### 3.1 저장소 미러 매핑 (jobs/kg_sync_commulingo, jobs/kg_sync_documents)

| 원본 | 노드 | 엣지 |
|------|------|------|
| commulingo_people | Person (name=name_ko, aliases=name_en·cyrillic·person_aliases, summary=별칭·생몰·bio·주요 경력 6줄) | Person→Role(역할 범주) Affiliation, Person→Role(기관 계보) Affiliation, Person→Concept(시대 그룹) Reference(people_group) |
| commulingo_offices / office_rows | Role | Person→Role Affiliation (`valid_at`/`invalid_at` = 재임 연도, attributes.position) |
| commulingo_history_events / event_people / locations | Incident, Location | Person→Incident Involvement (role_in_incident=relation_kind), Incident→Location Presence |
| commulingo_terms / term_* | Concept | Concept→Concept Reference(category·parent_term·related_term), Person→Concept Reference(person_term), Concept→Incident Reference(event_term) |
| research_documents(public) / archival manifest / autonomous_project_notes(synthesis) | Document | Document→Concept Reference(collection), Document→Entity Reference(about: manifest 큐레이션 링크, mentions: 별칭 인덱스 매칭·LLM 추출 엔티티), LLM 추출 fact(`KG_DOC_EXTRACT_LLM=1`, attributes.doc_ref) |

경력 17k행은 엣지가 아니라 Person summary에 접힌다. `commulingo_id_redirects`는 병합(`merge_entity_nodes_sync`) 또는 external_ids 추가로 반영된다.

---

## 4. 정보소스 타입 매핑 (EPISODE_SOURCE_MAP)

| 소스 카테고리 | EpisodeType | 설명 |
|---------------|-------------|------|
| `osint_news` | text | 오픈소스 뉴스 기사 |
| `osint_social` | text | 소셜미디어 포스트 |
| `osint_forum` | text | 포럼 게시글 (다크웹/공개) |
| `cve_feed` | json | CVE 취약점 피드 |
| `threat_report` | text | 위협인텔리전스 보고서 |
| `internal_siem` | json | 내부 SIEM 알림/로그 |
| `internal_report` | text | 내부 분석가 보고서 |
| `humint_debrief` | text | 인적정보 보고 (HUMINT) |
| `financial_record` | json | 금융거래/신고 기록 |
| `patent_filing` | json | 특허출원/등록 기록 |
| `personnel_change` | text | 인사변동 공지 |
| `diplomatic_cable` | text | 외교/정책 통신 |

---

## 5. 기술 스택

| 구성 요소 | 역할 |
|-----------|------|
| **Graphiti** | 지식그래프 저장소 (바이템포럴 타임스탬프: valid_at, created_at) |
| **Pydantic** | 엔티티/엣지 스키마 정의 및 유효성 검사 |
| **Python** | 구현 언어 |

---

## 6. 파일 구조

```
graph_memory/
├── config.py         # 엣지 타입 매핑, 동기화 전용 타입·Reference 쌍, 에피소드 소스 매핑
├── entities.py       # 10 개 엔티티 타입 + Document
├── edges.py          # 12 개 엣지 타입 + Reference
├── structured_writer.py  # 결정적 트리플 쓰기 (정체성 힌트, 배치 임베딩, allow_sync_predicates)
└── service.py        # graphiti 서비스 (EXTRACTION_* 부분집합, 에피소드 후 교차 그룹 병합)
kg_runtime/
├── identity.py       # 정규화·해석·정체성 upsert·병합·AliasIndex
├── search.py         # 엔티티 뷰 / 하이브리드 검색 / hydrate / 포맷
├── recall.py         # 엔티티 게이트 회상 블록 (KG_ENTITY_GATED_RECALL)
├── doc_extract.py    # 문서 → Document 노드·참조·LLM fact
└── metrics.py        # 건강 지표
jobs/
├── kg_sync.py        # 진입점 + kg_sync_state 워터마크
├── kg_sync_commulingo.py
└── kg_sync_documents.py
```

---

## 7. 설계 원칙

1. **모든 필드는 Optional** — 수집 시점에 정보가 부족할 수 있음
2. **점진적 해소** — Graphiti 의 entity resolution 이 동일 엔티티 속성을 누적 채움
3. **Incident vs Campaign** — 단발 사건은 Incident, 지속 활동은 Campaign
4. **Incident vs Policy** — 단발 사건은 Incident, 지속 제도는 Policy
5. **ThreatAction vs PolicyEffect** — 물리적/사이버 공격은 ThreatAction, 제도적 효과는 PolicyEffect
6. **Person vs Role** — 실제 개인은 Person, 직책/직위 자체는 Role
7. **Organization vs Industry** — 특정 기관/기업/국가는 Organization, 산업/섹터는 Industry
8. **Statement vs ThreatAction** — 발언·비판·주장은 Statement, 실제 공격/위협 행위는 ThreatAction
9. **Causation** — 단순 선후관계가 아니라 명시적 원인/기여요인 주장에만 사용
10. **정체성은 external_ids·alias_keys로** — 같은 실체는 저장소·언어·group_id가 달라도 노드 하나. 이름 문자열 일치에 기대지 말고 외부 id와 별칭을 실어 보낸다
11. **Document/Reference는 미러 잡 전용** — 에이전트 툴과 자유텍스트 추출에는 노출하지 않는다
