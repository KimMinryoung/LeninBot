# 지식그래프 스키마 설계

**버전:** v2  
**작성일:** 2026-02-27  
**기반 프레임워크:** Graphiti + Pydantic

---

## 1. 엔티티 타입 (7 종 42 필드)

| 엔티티 | 설명 | 주요 필드 |
|--------|------|-----------|
| **Person** | 개인 인물 (주요인물, 전문가, 위협행위자, 활동가, 정보원 등) | `alias`, `nationality`, `role`, `expertise`, `ideological_alignment`, `network_role`, `recruitment_potential`, `reliability_rating`, `influence_level` |
| **Organization** | 조직 (기업, 정부기관, 군, 연구소, 위협그룹, NGO, 정당, 언론사 등) | `org_type`, `industry`, `headquarters`, `country`, `parent_org`, `ideological_orientation`, `alliance_bloc`, `state_sponsor`, `threat_classification`, `known_ttps` |
| **Location** | 지리적 장소 (시설, 거점, 군사기지, 초크포인트, 분쟁지역 등) | `location_type`, `coordinates`, `significance`, `strategic_resources`, `geopolitical_bloc` |
| **Asset** | 자산 (기술, 제품, IP, 인프라, 무기체계, 공급망 노드 등) | `asset_type`, `classification`, `strategic_value`, `description_detail`, `supply_chain_role`, `dual_use_potential`, `controlling_entity` |
| **Incident** | 사건 (사이버공격, 인사변동, 정책변화, 군사배치, 단속 등) | `incident_type`, `severity`, `occurred_at`, `detected_at`, `status`, `confidence`, `impact_summary`, `geopolitical_context`, `information_source_type` |
| **Policy** | 정책 (제재, 조약, 수출통제, 무역협정, 군사교리, 법률, 행정명령 등) | `policy_type`, `issuing_entity`, `target_scope`, `status`, `effective_date`, `strategic_impact` |
| **Campaign** | 캠페인 (군사작전, 영향력공작, 사이버캠페인, 사회운동, 선전전, 경제전 등) | `campaign_type`, `objective`, `status`, `scale`, `started_at`, `ideological_framing`, `effectiveness` |

---

## 2. 엣지 타입 (10 종)

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
| Entity (폴백) | Entity (폴백) | Funding, AssetTransfer |

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
├── __init__.py       # 패키지 초기화
├── config.py         # 엣지 타입 매핑, 에피소드 소스 매핑 설정
├── entities.py       # 7 개 엔티티 타입 정의 (Person, Organization, Location, Asset, Incident, Policy, Campaign)
├── edges.py          # 10 개 엣지 타입 정의
└── service.py        # 지식그래프 서비스 구현
```

---

## 7. 설계 원칙

1. **모든 필드는 Optional** — 수집 시점에 정보가 부족할 수 있음
2. **점진적 해소** — Graphiti 의 entity resolution 이 동일 엔티티 속성을 누적 채움
3. **Incident vs Campaign** — 단발 사건은 Incident, 지속 활동은 Campaign
4. **Incident vs Policy** — 단발 사건은 Incident, 지속 제도는 Policy
5. **ThreatAction vs PolicyEffect** — 물리적/사이버 공격은 ThreatAction, 제도적 효과는 PolicyEffect
