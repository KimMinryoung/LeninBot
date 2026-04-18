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
| 그래프 DB | Neo4j Community (Hetzner Docker, `docker-compose.neo4j.yml`) |
| LLM | Gemini 2.5 Flash (추출), Gemini 2.5 Flash-Lite (리랭킹) |
| 임베딩 | gemini-embedding-001 (1024차원) |
| Python 패키지 | `graphiti-core[google-genai]` |

**OpenAI 의존성 없음** — 전부 Gemini 기반.

### chatbot.py 통합 완료 (2026-03-01)

- Vectorstore 경로: `kg_retrieve` 별도 노드 (retrieve → kg_retrieve → grade_documents)
- Plan 경로: step_executor 내부 per-step KG 검색 + `_merge_kg_contexts()` 중복 제거
- KG 결과는 `kg_context`에 저장, `documents`와 분리 (grade_documents 비적용)
- 남은 과제: 데이터 수집 파이프라인 구축 (크롤러 → 에피소드 변환 → ingest)

---

## 2. 아키텍처

### 파일 구조

```
graph_memory/
├── __init__.py          # 패키지 exports: GraphMemoryService, ENTITY_TYPES, EDGE_TYPES, etc.
├── entities.py          # 7개 엔티티 타입 정의 (Pydantic) — v2
├── edges.py             # 10개 엣지 타입 정의 (Pydantic) — v2
├── config.py            # EDGE_TYPE_MAP, EPISODE_SOURCE_MAP, CUSTOM_EXTRACTION_INSTRUCTIONS
├── graphiti_patches.py  # Graphiti 런타임 몽키패치 (DateTime 직렬화, 엣지 프롬프트)
├── service.py           # GraphMemoryService 클래스 (핵심)
└── news_fetcher.py      # Tavily 뉴스 검색 → 에피소드 수집 유틸리티
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

### 엔티티 10종 (v2.2 — Role/Industry 추가)

| 엔티티 | 필드 수 | 설명 | 주요 필드 |
|--------|---------|------|-----------|
| **Person** | 9 | 개인 인물 (NOT roles) | alias, nationality, role, expertise, ideological_alignment, network_role, recruitment_potential, reliability_rating, influence_level |
| **Organization** | 10 | 조직 (institutional bodies) | org_type, industry, headquarters, country, parent_org, ideological_orientation, alliance_bloc, state_sponsor, threat_classification, known_ttps |
| **Location** | 5 | 지리적 장소 | location_type, coordinates, significance, strategic_resources, geopolitical_bloc |
| **Asset** | 7 | 기술/제품/IP/무기 등 가치 있는 것 | asset_type, classification, strategic_value, description_detail, supply_chain_role, dual_use_potential, controlling_entity |
| **Incident** | 9 | 특정 시점의 단발 사건 | incident_type, severity, occurred_at, detected_at, status, confidence, impact_summary, geopolitical_context, information_source_type |
| **Policy** | 6 | 제도적 수단 (법령/조약/제재) | policy_type, issuing_entity, target_scope, status, effective_date, strategic_impact |
| **Campaign** | 7 | 지속적 조직 활동 (작전/운동) | campaign_type, objective, status, scale, started_at, ideological_framing, effectiveness |
| **Concept** | 5 | 이데올로기·이론·계급·시기 등 추상 | concept_type, domain, related_thinkers, historical_period, contemporary_relevance |
| **Role** ✨ | 5 | 직책·직위 (점유자와 별개) | role_type, domain, jurisdiction, seniority, selection_method |
| **Industry** ✨ | 6 | 경제 산업·섹터 (특정 조직 위 추상) | sector_type, value_chain_position, strategic_importance, geographic_concentration, regulatory_status, capital_composition |

- 모든 필드는 `Optional` — 수집 누적 시 점진적으로 채워짐
- v2.1 → **v2.2** 변경: **Role 신설**, **Industry 신설**. Concept의 enum에 `social_class`, `historical_era`, `movement_doctrine` 추가.
- 미스분류 마이그레이션 (v2.1 → v2.2): Person→Role 12건, Asset→Industry 13건, Concept→Role 2건, Concept→Industry 3건, 내부 노이즈 entities 39건 삭제.

### 엣지 12종 (v2.2 — Statement/Causation 추가)

| 엣지 | 설명 | 주요 필드 |
|------|------|-----------|
| **Affiliation** | 사람→조직, 사람→Role, 조직→Industry 소속 | position, department, affiliation_type, start_date, end_date, is_current, access_level |
| **PersonalRelation** | 사람↔사람 | relation_type, context, strength, first_observed |
| **OrgRelation** | 조직↔조직 | relation_type, agreement_type, financial_value, strategic_significance |
| **Funding** | 자금 흐름 | funding_type, amount, purpose, is_verified |
| **AssetTransfer** | 기술/자산 이전 | transfer_type, asset_description, legality, export_control |
| **ThreatAction** | 공격/위협 행위 (군사·사이버) | action_type, technique, target_asset, outcome, confidence |
| **Involvement** | 엔티티→사건/캠페인 관여 | role_in_incident, evidence_basis, confidence |
| **Presence** | 엔티티→장소 관련 | presence_type, frequency, purpose |
| **PolicyEffect** | 정책↔엔티티 영향 | effect_type, impact_description, compliance_status |
| **Participation** | 엔티티→캠페인 참여 | role, contribution, commitment_level |
| **Statement** ✨ | 발화·성명·인용 (X said about Y) | statement_type, medium, audience, statement_date, verbatim_excerpt |
| **Causation** ✨ | 인과 관계 (X caused Y) | causal_type, confidence, mechanism |

- v2.2 변경: **Statement 신설** (speech acts — 정치적 비판/대립도 여기로), **Causation 신설** (명시적 인과). 모든 entity 쌍에 대해 wildcard로 사용 가능.

### EDGE_TYPE_MAP (32 entries)

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
| **Person → Role** ✨ | Affiliation (X holds role Y) |
| **Role → Organization** ✨ | Affiliation (Role is part of Org) |
| **Role → Location** ✨ | Presence (Role's jurisdiction) |
| **Organization → Industry** ✨ | Affiliation (Org belongs to industry) |
| **Industry → Location** ✨ | Presence (geographic concentration) |
| **Policy → Industry** ✨ | PolicyEffect |
| **Industry → Asset** ✨ | AssetTransfer |
| **Campaign → Industry** ✨ | ThreatAction |
| **Entity → Entity (폴백)** | Funding, AssetTransfer, **Statement** ✨, **Causation** ✨ |

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
| `NEO4J_URI` | Neo4j 연결 URI | `bolt://localhost:7687` |
| `NEO4J_USER` | DB 사용자 | `neo4j` |
| `NEO4J_PASSWORD` | DB 비밀번호 | `.env` 참조 |
| `NEO4J_DATABASE` | DB 이름 | `neo4j` |
| `GEMINI_API_KEY` | Gemini API 키 | `.env` 참조 |
| `SEMAPHORE_LIMIT` | Graphiti 동시성 제한 | 기본 20, rate limit 시 1로 낮추기 |

### LLM 모델 구성

| 용도 | 모델 | 클라이언트 |
|------|------|-----------|
| 엔티티/엣지 추출 | gemini-2.5-flash | GeminiClient (model) |
| 리랭킹/속성 추출 | gemini-2.5-flash-lite | GeminiClient (small_model) + GeminiRerankerClient |
| 임베딩 | gemini-embedding-001 | GeminiEmbedder (1024차원 벡터) |

### Neo4j Local (Docker) 설정

1. **Docker Compose**: `docker-compose.neo4j.yml` — Neo4j 5 Community + APOC
2. **systemd**: `leninbot-neo4j.service` — Docker 컨테이너 자동 시작, `leninbot-api`가 의존
3. **DB 이름**: `neo4j` (기본값)
4. **벡터 인덱스 수동 생성 필요** — `build_indices_and_constraints()`가 벡터 인덱스를 자동 생성하지 않음
5. **AuraDB에서 이전 완료** (2026-03-21) — `scripts/migrate_neo4j.py` 사용

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
- **Rate limit 주의** — Gemini Free Tier에서 한 에피소드 수집만으로 429 발생 가능. 환경변수 `SEMAPHORE_LIMIT`로 동시성 조절, 에피소드 간 딜레이 추가 필요
- **벡터 인덱스 없으면 검색 실패** — `entity_name_embedding`, `edge_fact_embedding` 인덱스 수동 생성 필수 (섹션 4 참조)
- **관계 타입은 `r.name` 프로퍼티** — Neo4j에서 `RELATES_TO` 엣지의 관계 타입은 `r.name` 필드에 저장됨 (`r.relation_type` 아님)

### Graphiti 라이브러리 특이사항

- `search_()` 메서드 사용 (밑줄 포함) — `SearchConfig(limit=N)` 전달
- `EpisodeType` enum: `text`, `message`, `json`
- 엔티티 해소(entity resolution)는 Graphiti가 자동 수행 — 동일 엔티티를 다른 에피소드에서 언급하면 자동 병합

### 런타임 패치 (`graphiti_patches.py`)

`.venv` 파일 수정 대신 런타임 몽키패치로 Render 배포 환경에서도 동작.
`service.py` import 시 자동 적용 (멱등).

| 패치 | 대상 | 효과 |
|------|------|------|
| `to_prompt_json` | `prompt_helpers` + 6개 importing 모듈 | Neo4j DateTime 직렬화 지원 |
| `Edge.relation_type` | Pydantic 모델 (클래스 수준) | SCREAMING_SNAKE_CASE → PascalCase |
| `edge()` 함수 | `prompt_library` VersionWrapper.func | RELATION TYPE RULES 프롬프트 교체 |
| `normalize_entity_names_in_text` | `service.py` ingest_episode() | 에피소드 본문 내 약어→정식명 치환 (US→United States 등) |

### 이름 정규화 (`config.py` NAME_NORMALIZATION)

에피소드 본문이 Graphiti에 전달되기 전에 국가/조직 약어를 정식명으로 치환.
Graphiti의 entity resolution이 짧은 이름(entropy < 1.5)에서 실패하는 문제를 우회.
`NAME_NORMALIZATION` 딕셔너리에 매핑 추가 시 자동 적용.

### 성능 설정

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `SEMAPHORE_LIMIT` | 20 (기본) | Graphiti 내부 LLM/DB 호출 동시성. `service.py`에서 `setdefault`로 설정. |
| `DEFAULT_DELAY_BETWEEN` | 5초 | `news_fetcher.py` 에피소드 간 대기. rate limit 시 증가 가능. |

---

## 5.5. 오염 방어 / Provenance & Trust Tiering (2026-04-08)

> Mythos Preview 같은 고능력 모델이 indirect prompt injection으로 KG에 거짓 사실을 심으려는 공격을 가정한 방어 레이어. Palantir Foundry 류의 4가지 원리 중 LeninBot 규모에 맞는 부분을 발췌 적용.

### 위협 모델

적이 웹페이지/PDF/이메일에 거짓 "사실"을 심는다 → analyst가 fetch_url/convert_document로 읽는다 → "중요한 fact를 KG에 저장하라"는 지시대로 `write_kg`를 호출 → 이후 모든 세션에서 LeninBot이 자기 KG를 신뢰할 만한 1차 출처처럼 인용 → **RAG self-poisoning loop**가 형성되고 매 사이클마다 거짓이 더 단단해진다. 일단 들어가면 출처 추적 수단이 없어 사후 청소가 거의 불가능.

### 4가지 메커니즘 (모두 단일 패스로 wired)

#### (1) Provenance auto-attachment

`shared.ProvenanceBuffer` — per-agent-run 버퍼. `contextvars.ContextVar` 기반이라 parallel `asyncio.gather` tool 호출이 부모 컨텍스트 reference를 공유 (asyncio task가 부모 context를 copy하지만 list 객체 reference가 같으므로 mutation이 양쪽에서 보임).

- 초기화: `claude_loop.chat_with_tools` / `openai_tool_loop.chat_with_tools` 진입 직후 `init_provenance_buffer(agent=..., mission_id=...)` 호출
- `telegram_bot._chat_with_tools`가 `task_id`로 DB 조회해 `agent_type` + `mission_id`를 전달, orchestrator는 `"orchestrator"` 고정
- 자동 기록 hook: `tool_loop_common.execute_tool`이 매 성공 호출 후 `_record_tool_provenance(name, args, result)`를 호출
  - external sources → `buf.external_calls` (fetch_url/web_search/convert_document/check_inbox/network-sourced read_file/search_files)
  - KG 검색 결과 → `buf.kg_reads`
- 실패해도 silent (try/except no-op) — provenance bookkeeping이 tool 실행을 깨지 않음

#### (2) Trust tiering (4단)

`ProvenanceBuffer.infer_trust_tier()` — 최근 external call의 도메인 다양성으로 자동 추론:

| Tier | 조건 |
|------|------|
| `anchor` | operator-asserted (현재는 수동 부여만, 자동 추론 안 함) |
| `corroborated` | ≥2 distinct external 도메인에서 ingest |
| `single` | 1 도메인 또는 local-file source |
| `unverified` | external call 없음 |

- `write_kg`는 자동으로 추론된 tier를 episode 이름에 인코딩: `[T:tier]name`. Graphiti가 sanitize해서 `T-tier-name`으로 저장됨 (round-trip survives).
- Read 시 `search_knowledge_graph`가 두 단계 Cypher pass로:
  1. `MATCH ()-[r:RELATES_TO]->() WHERE r.uuid IN $euuids RETURN r.uuid, r.episodes` — Graphiti edge dict에는 `episodes`가 빠져있어서 직접 조회 필요
  2. `MATCH (e:Episodic) WHERE e.uuid IN $uuids RETURN e.uuid, e.name` — 이름에서 tier 파싱
- 각 fact 라인 앞에 `[T:tier]` 인라인 표시. 옛 facts(이전 시스템)는 `[T:?]`. Edge가 여러 episode를 reference하면 highest tier 채택 (anchor > corroborated > single > unverified).

#### (3) Append-only

엔진 레벨 보장:
- analyst/programmer/scout 화이트리스트에 `kg_delete_episode`, `kg_merge_entities`, `kg_query`, `kg_admin` **모두 부재**
- 변경 가능한 KG 작업은 orchestrator의 self_tools 경로로만 도달 가능 — prompt injection에 노출된 sub-agent가 직접 못 만짐
- 정정/철회는 `write_kg`의 신규 옵셔널 파라미터 `supersedes`로: 새 episode + footer에 `supersedes: <old_name>` 마커. 옛 fact는 절대 silent overwrite/delete 되지 않음 → 사후 롤백 항상 가능.

#### (4) Self-poisoning loop break

`_self_poisoning_overlap()` — 새 content와 같은 agent run 내 최근 KG 검색 결과 사이의 Jaccard 토큰 overlap 계산.

- 임계치 0.55 — 넘으면 `[KG AUDIT] BLOCKED self-poisoning write` 로그 + 거부
- 거부 메시지는 친절하게: "100% overlaps with text you just retrieved... rewrite in your own words with the new source cited"
- 정확히 RAG echo-chamber("내가 이미 알고 있는 걸 다시 KG에 쓴다") 패턴을 끊는다. 이게 가장 흔하고 위험한 자기 강화 메커니즘.

### Provenance footer 형식

`write_kg` 성공 시 episode body에 자동 첨부:

```
<원래 content>

— provenance —
agent: analyst | mission=42
ingested_at: 2026-04-08 21:21 KST
trust_tier: corroborated
sources:
  - url:https://en.wikipedia.org/wiki/...
  - url:https://reuters.com/...
  - web_search:Iran ceasefire 2026
  - document:/home/grass/leninbot/data/downloads/un_report.pdf
supersedes: <optional, when correcting a prior episode>
```

### Audit log 강화

`logger.info("[KG AUDIT] wrote episode | name=... | group=... | tier=... | agent=... | sources=N | supersedes=... | time=... | content_len=...")` — 사후 어떤 fact가 어느 세션 어느 출처에서 들어왔는지 grep으로 추적 가능.

### 의도적으로 미적용

Palantir 스타일 중 LeninBot 규모에서 과잉인 부분:

- **Quarantine 2단 레이어** (pending → canonical) — 단일 사용자 환경에서 review queue 운영 비용이 가치보다 큼
- **Bitemporal modeling** (valid_time / transaction_time 분리) — Graphiti가 `valid_at`/`invalid_at`/`created_at`/`expired_at`을 이미 추적하므로 충분
- **Cell-level classification** — single tenant
- **Anomaly drift detection** — 추후 옵션, 현재는 audit log + 수동 검토로 충분
- **Reality anchors layer** — 별도 group_id로 가능하지만 현재 명시적 분리 안 됨. 필요 시 `group_id="anchor_facts"` + 소스 코드에서 operator-only로 protect.

### 검증 시나리오

| 시나리오 | 결과 |
|----------|------|
| 두 독립 도메인 fetch + write_kg | `[T:corroborated]` tier, 검색 시 인라인 표시 ✅ |
| KG 검색 결과를 그대로 write_kg | "Refused: 100% overlaps..." ✅ |
| External 호출 없이 write_kg | tier=unverified + 친절한 경고 메시지 ✅ |
| 옛 fact 검색 | `[T:?]` 표시 (이전 시스템 산물) ✅ |
| Sub-agent KG 변경 시도 | 화이트리스트에 delete/merge 없음 — 호출 자체 불가 ✅ |

### External-source envelope와의 관계

같은 위협(indirect prompt injection)에 대한 두 레이어 방어:

| 방어 | 막는 것 |
|------|---------|
| `<external source="...">` envelope (`shared.EXTERNAL_SOURCE_RULE` + `_wrap_external`) | 외부 콘텐츠의 *명령형*이 사용자 지시로 둔갑하는 것 |
| Provenance + trust tier + self-poisoning break | 외부 콘텐츠의 *명제*가 KG에 무신경하게 박혀서 자기 강화 루프를 만드는 것 |

Envelope는 모델의 행동을, provenance는 KG의 진실성을 보호한다. 둘은 짝이고 의도적으로 분리됐다 — envelope는 명제 신뢰를 떨어뜨리지 *않고* (Gemini식 paranoia 회피), provenance는 명제를 막지 *않고* tier로 표시만 해서 모델이 가중치를 매기게 한다.

---

## 6. 현재 데이터 상태

### Local Neo4j Docker (v2.1 스키마, 2026-03-29 기준)

**통계**: 에피소드 1,213건, 엔티티 3,279개, 관계 3,261개

| 엔티티 타입 | 수 |
|------------|-----|
| Organization | 901 |
| Asset | 724 |
| Person | 405 |
| Location | 370 |
| Policy | 299 |
| Concept | 274 |
| Incident | 186 |
| Campaign | 97 |
| Entity (미분류) | 23 |

### 에피소드 group_id 분포

| group_id | 에피소드 수 | 설명 |
|----------|-----------|------|
| `diary_news` | 1,053 | LeninBot 자동 수집 뉴스 (3시간 주기) |
| `agent_knowledge` | 59 | 에이전트가 write_kg로 저장한 지식 |
| `geopolitics_conflict` | 53 | 전쟁/분쟁 뉴스 |
| `korea_domestic` | 28 | 한국 국내 뉴스/인물 |
| `economy` | 8 | 경제 뉴스 |
| `geopolitics_economy` | 5 | 경제/제재/무역 뉴스 |
| `diplomacy` | 4 | 외교/협상 뉴스 |
| `geopolitics_diplomacy` | 2 | 외교/협상 뉴스 (레거시) |

### 관계명 분포

| 관계 타입 | 수 |
|----------|-----|
| Involvement | 557 |
| Presence | 470 |
| PolicyEffect | 460 |
| AssetTransfer | 319 |
| NULL (미분류) | 318 |
| ThreatAction | 306 |
| OrgRelation | 297 |
| Participation | 190 |
| Affiliation | 144 |
| Funding | 46 |
| PersonalRelation | 28 |
| 비표준 (~80개) | ~80 |

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
| 2026-02-28 | **런타임 패치**: `graphiti_patches.py` 신규 — .venv 수정 대신 몽키패치로 DateTime 직렬화/엣지 프롬프트 교체. Render 배포 호환. |
| 2026-02-28 | **성능 최적화**: `SEMAPHORE_LIMIT` 1→20, `DEFAULT_DELAY_BETWEEN` 30→5초. `requirements.txt` 프로덕션 전용으로 정리. |
| 2026-03-01 | **한국 국내 뉴스 + 인물 프로파일 수집** (TDD): `temp_dev/ingest_kr_news.py` — Tavily 뉴스 검색 → LLM 인물 추출 → Tavily 프로파일 검색 → KG 수집. 뉴스 3건 + 이재명 프로파일 수집. `group_id="korea_domestic"`. KG에 South Korea, Democratic Party of Korea, Lee Jae-myung, Seongnam, Gyeonggi Province 등 노드 15개, 엣지 15개 추가. |
| 2026-03-01 | **chatbot.py KG 질의 통합**: vectorstore 경로에 `kg_retrieve` 노드, plan 경로에 step_executor 내부 per-step KG 검색 추가. `_merge_kg_contexts()`로 엔티티/팩트 중복 제거. `kg_context`를 strategize/generate 프롬프트에 주입. Neo4j event loop 충돌 해결 (persistent `_kg_loop`). KG 장애 시 graceful degradation 검증 완료. |
| 2026-03-19 | **v2.1 스키마: Concept 타입 신설** — 엔티티 7→8종. 추상 개념/이론/이데올로기/사회현상을 포괄하는 Concept 타입(5 필드) 추가. Gemini 배치 분류로 미분류 엔티티 131개 전량 타입 부여 (Concept 99, Asset 11, Organization 8, Incident 5, Person 5, Location 1, Campaign 1). `scripts/classify_untyped_entities.py` 재사용 가능. |
| 2026-03-21 | **Neo4j AuraDB → Local Docker 이전**: `docker-compose.neo4j.yml` (Neo4j 5 Community + APOC), `systemd/leninbot-neo4j.service`, `scripts/migrate_neo4j.py`. 엔티티 2,116개, 관계 1,826개, 에피소드 761건 전량 이전 완료. AuraDB 연결 끊김 문제 해소. |
| 2026-03-29 | **KG 데이터 품질 개선**: (1) 중복 엔티티 병합 스크립트 `merge_entities.py` — Gemini 배치 분류 + canonical 선택 + 엣지 이전 + duplicate 삭제. (2) 관계명 정규화 스크립트 `normalize_edge_names.py` — 비표준/NULL r.name을 10개 표준 타입으로 분류. (3) KG 백업 스크립트 `backup_kg.py`. (4) 추출 지시문 강화 — 국가/조직/인물 정식명 사용 규칙 추가, 약어 금지. (5) 이름 정규화 패치 — `graphiti_patches.py`에 `NAME_NORMALIZATION` 딕셔너리 + 텍스트 레벨 약어→정식명 치환 (Graphiti entity resolution 실패 방지). |
| 2026-04-08 | **KG 오염 방어 (Palantir-style robustness)**: indirect prompt injection으로 거짓 fact가 KG에 박히는 위협에 대한 4-레이어 방어. (1) `shared.ProvenanceBuffer` + `contextvars` per-agent-run 버퍼; `tool_loop_common.execute_tool`이 external-source 툴 호출과 KG 읽기를 자동 기록; `chat_with_tools` 진입 시 초기화. (2) Trust tiering — 4단(anchor/corroborated/single/unverified), 도메인 다양성으로 자동 추론, episode 이름에 `[T:tier]` 인코딩(graphiti가 `T-tier-`로 sanitize), `search_knowledge_graph`가 두 단계 Cypher pass로 각 fact에 인라인 표시. (3) Append-only — analyst/programmer/scout 화이트리스트에서 delete/merge/query/admin 모두 부재 확인, `write_kg`에 `supersedes` 옵션 추가. (4) Self-poisoning loop break — Jaccard ≥0.55면 거부, friendly 메시지로 fresh source 인용 안내. Provenance footer 자동 첨부 + audit log 보강. 자세한 설계는 §5.5 참조. |
| 2026-04-11 | **KG cleanup + invariant + structured writes + v2.2 schema**: 단일 라운드에 5가지 변화. (1) **데이터 cleanup**: exact-name dup 250건 머지(merge_entities.py 버그 발견 후 recover_lost_edges.py로 277건 복구), orphan 11건 삭제, 비표준 19건 삭제, untyped 6건 분류, fact_embedding 436건 재생성. backup_kg.py가 fact_embedding/episodes 보존하도록 수정. (2) **write-time conformance gate** (`graph_memory/conformance.py`): 모든 `add_episode` 후 자동 검증 — self-loop/non-Entity endpoint auto-fix, non-standard edge name/type-map violation/untyped node 로깅. 동일 validator가 daily scanner에도 재사용 가능. (3) **`write_kg_structured` 도구** (`graph_memory/structured_writer.py`): typed (subject, predicate, object) 직접 단언. graphiti의 LLM extraction 우회, deterministic name+type 매칭, 합성 episode로 trust tier 보존. 6개 agent에 추가, MISSION_GUIDELINES_BLOCK 업데이트. (4) **v2.2 schema**: Role/Industry entity 신설, Statement/Causation predicate 신설. EDGE_TYPE_MAP에 9개 매핑 추가. CUSTOM_EXTRACTION_INSTRUCTIONS에 entity-type 가이드 + 내부 노이즈 필터 추가. (5) **마이그레이션** (`migrate_to_v22_schema.py`): heuristic + Gemini 배치 confirmation으로 Person→Role 12건, Asset→Industry 13건, Concept→Role 2건, Concept→Industry 3건 재분류, 내부 노이즈 entities 39건 삭제. 최종: 3914 entities, 4053 RELATES_TO, 0 untyped/dup/orphan. |

---

## 8. 유지보수 스크립트

### `skills/kg-maintenance/scripts/`

| 스크립트 | 용도 | 사용법 |
|---------|------|--------|
| `backup_kg.py` | Entity/Edge JSON 백업 | `python backup_kg.py` (파괴적 작업 전 필수) |
| `merge_entities.py` | 중복 엔티티 병합 (Gemini) | `python merge_entities.py` (dry-run) / `--execute` |
| `normalize_edge_names.py` | 비표준 관계명 정규화 (Gemini) | `python normalize_edge_names.py` (dry-run) / `--execute` |
| `dedup_entities.py` | 문자열 유사도 기반 중복 탐지 | `python dedup_entities.py` |
| `cleanup_orphans.py` | 고아 엔티티 삭제 | `python cleanup_orphans.py` (dry-run) / `--execute` |
| `assign_types.py` | 미분류 엔티티 타입 부여 | `python assign_types.py` |

### `scripts/`

| 스크립트 | 용도 |
|---------|------|
| `classify_untyped_entities.py` | Gemini 배치 분류 (8 타입, 더 정확) |
| `ingest_reports_to_kg.py` | 텔레그램 태스크 리포트 → KG 수집 |
| `kg_enricher.py` | 엔티티 요약 자동 생성 |
| `backup_kg_to_r2.py` | `backup_kg.py` + tar.gz + R2 업로드 (자동화용) |

### 자동 백업 (2026-04-18~)

- **스케줄:** `leninbot-kg-backup.timer` — 매일 03:00 KST
- **대상:** Cloudflare R2 버킷 `cyber-lenin-backups` (private), 객체 키 `kg-backup-YYYY-MM-DD.tar.gz`
- **보관:** 롤링 2일 (오늘 + 어제). 새 백업 성공 후 2일 전 백업 삭제
- **내용:** entities/edges/mentions JSON (임베딩 포함) 을 tar.gz로 묶음 (~79 MB 압축, 2026-04-18 기준)
- **복구:** R2에서 tar.gz 다운로드 → 풀기 → `restore_kg.py` 등으로 import (복구 경로는 별도 문서화 필요)

---

## 9. 로드맵

### 단기 (데이터 품질 + 수집)

- [x] AuraDB 새 인스턴스 초기화 + v2 스키마로 뉴스 10건 수집 완료
- [x] 한국 국내 뉴스 수집 + 인물 프로파일 보강 파이프라인 (TDD, `temp_dev/ingest_kr_news.py`)
- [x] 데이터 품질 도구: 중복 병합, 관계명 정규화, 백업 스크립트
- [x] 수집 시 약어→정식명 자동 치환 (entity resolution 실패 방지)
- [ ] 서버에서 cleanup 스크립트 실행: backup → merge → normalize → orphan → classify
- [ ] 크롤러 → 에피소드 변환 파이프라인 구축
- [ ] group_id 체계 확정 (산업별? 주제별? 시기별?)

### 중기 (chatbot 통합)

- [x] chatbot.py KG 질의 통합 완료
  - `kg_retrieve` 노드 (vectorstore 경로): retrieve → kg_retrieve → grade_documents
  - plan 경로: step_executor 내부에서 매 단계 KG 검색 (원자화 쿼리), `_merge_kg_contexts()`로 중복 제거
  - KG lazy singleton (`_get_kg_service()`) + persistent event loop (`_kg_loop`)
  - `kg_context`를 strategize/generate 프롬프트에 주입 (casual 제외)
  - KG 장애 시 graceful degradation (파이프라인 중단 없음)
- [ ] generate_briefing()을 활용한 전략 브리핑 기능

### 장기 (고도화)

- [ ] 실시간 수집 — 크롤러 스케줄러 + 자동 ingest
- [ ] 그래프 시각화 API
- [ ] 엔티티/관계 통계 대시보드
- [ ] 커뮤니티 감지 기반 클러스터 분석 활용
