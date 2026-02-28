"""
정보 에이전트 Graphiti 스키마 — 엔티티 정의 v2
================================================

레닌 프레임워크 갭 분석 기반 재설계 (2026-02-27).
v1 대비 변경: Person/Organization/Location/Asset/Incident 필드 확장,
Policy·Campaign 신설. 총 7종 42필드.

수집 대상: 사람, 조직, 장소, 자산, 사건, 정책, 캠페인
Graphiti의 Custom Entity Types API를 사용하여 정의.

모든 필드는 Optional — 수집 시점에 모든 정보가 확보되지 않을 수 있음.
수집이 누적되면서 Graphiti의 엔티티 해소(entity resolution)가
동일 엔티티의 속성을 점진적으로 채워감.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ============================================================
# 1. 사람 (Person)
# ============================================================

class Person(BaseModel):
    """
    개인 인물 엔티티.
    대상: 주요 인물, 의사결정권자, 기술 전문가, 위협 행위자,
    활동가, 정보원, 로비스트 등.
    """
    # --- 기본 식별 ---
    alias: Optional[str] = Field(
        None,
        description="Known aliases, handles, or codenames (comma-separated if multiple)"
    )
    nationality: Optional[str] = Field(
        None,
        description="Nationality or country of origin (ISO 3166-1 alpha-2 code preferred)"
    )

    # --- 직업/역할 ---
    role: Optional[str] = Field(
        None,
        description=(
            "Current primary role or job title. Include informal roles "
            "(broker, advisor, lobbyist, intermediary) in addition to official positions"
        )
    )
    expertise: Optional[str] = Field(
        None,
        description="Domain expertise areas (e.g., 'chip design, EUV lithography')"
    )

    # --- 정치/이데올로기 ---
    ideological_alignment: Optional[str] = Field(
        None,
        description=(
            "Ideological orientation: "
            "communist / socialist / liberal / conservative / nationalist / "
            "islamist / anarchist / techno-libertarian / unknown"
        )
    )

    # --- 네트워크 분석 ---
    network_role: Optional[str] = Field(
        None,
        description=(
            "Structural role in networks: "
            "hub / bridge / gatekeeper / peripheral / isolate / unknown"
        )
    )
    recruitment_potential: Optional[str] = Field(
        None,
        description=(
            "Recruitment or cooperation potential: "
            "high / medium / low / hostile / asset / unknown"
        )
    )

    # --- 평가 ---
    reliability_rating: Optional[str] = Field(
        None,
        description="Source reliability rating: A(reliable) to F(unknown). NATO STANAG 2022 scale."
    )
    influence_level: Optional[str] = Field(
        None,
        description="Estimated influence level: strategic / operational / tactical / minimal"
    )


# ============================================================
# 2. 조직 (Organization)
# ============================================================

class Organization(BaseModel):
    """
    조직 엔티티.
    대상: 기업, 정부기관, 군, 연구소, 위협 그룹, NGO, 정당,
    언론사, 노조, 국제기구, 정보기관, 싱크탱크 등.
    """
    org_type: Optional[str] = Field(
        None,
        description=(
            "Type of organization: "
            "corporation / government_agency / military / "
            "research_institute / threat_group / ngo / consortium / "
            "political_party / media_outlet / labor_union / "
            "international_org / intelligence_agency / think_tank / other"
        )
    )
    industry: Optional[str] = Field(
        None,
        description="Primary industry or sector (e.g., 'semiconductor', 'defense', 'finance')"
    )
    headquarters: Optional[str] = Field(
        None,
        description="Headquarters location (city, country)"
    )
    country: Optional[str] = Field(
        None,
        description="Country of incorporation or primary affiliation (ISO 3166-1 alpha-2)"
    )
    parent_org: Optional[str] = Field(
        None,
        description="Name of parent organization if subsidiary"
    )

    # --- 지정학 ---
    ideological_orientation: Optional[str] = Field(
        None,
        description=(
            "Ideological orientation of the organization (free text). "
            "e.g., 'state capitalism', 'Marxist-Leninist', 'neoliberal', 'techno-nationalism'"
        )
    )
    alliance_bloc: Optional[str] = Field(
        None,
        description=(
            "Geopolitical alliance or bloc affiliation: "
            "NATO / Five_Eyes / SCO / BRICS / AUKUS / EU / ASEAN / "
            "Non-Aligned / none / unknown"
        )
    )
    state_sponsor: Optional[str] = Field(
        None,
        description=(
            "State sponsor or patron if applicable. Country name. "
            "e.g., 'US' for NED, 'Russia' for RT, 'China' for Confucius Institutes"
        )
    )

    # --- 위협 그룹 전용 ---
    threat_classification: Optional[str] = Field(
        None,
        description="For threat groups: APT / cybercrime / hacktivist / state_sponsored / unknown"
    )
    known_ttps: Optional[str] = Field(
        None,
        description="For threat groups: known TTPs summary or MITRE ATT&CK technique IDs"
    )


# ============================================================
# 3. 장소 (Location)
# ============================================================

class Location(BaseModel):
    """
    지리적 장소 엔티티.
    시설, 거점, 군사기지, 초크포인트, 분쟁지역 등 활동의 공간적 맥락.
    """
    location_type: Optional[str] = Field(
        None,
        description=(
            "Type: facility / city / region / country / virtual / "
            "military_base / chokepoint / economic_zone / border_region / conflict_zone"
        )
    )
    coordinates: Optional[str] = Field(
        None,
        description="GPS coordinates if known (lat, lon)"
    )
    significance: Optional[str] = Field(
        None,
        description="Why this location matters (e.g., 'fab site', 'R&D center', 'C2 server location')"
    )
    strategic_resources: Optional[str] = Field(
        None,
        description=(
            "Strategic resources present at this location (comma-separated). "
            "e.g., 'rare earth minerals', 'semiconductor fabs', 'oil reserves', "
            "'naval chokepoint', 'undersea cables'"
        )
    )
    geopolitical_bloc: Optional[str] = Field(
        None,
        description=(
            "Geopolitical sphere of influence: "
            "US-aligned / China-aligned / Russia-aligned / EU-aligned / "
            "non-aligned / contested / neutral / unknown"
        )
    )


# ============================================================
# 4. 자산/기술 (Asset)
# ============================================================

class Asset(BaseModel):
    """
    기술, 제품, 지적재산, 인프라, 무기체계, 공급망 노드 등 가치 있는 대상.
    """
    asset_type: Optional[str] = Field(
        None,
        description=(
            "Type: technology / product / patent / infrastructure / "
            "data / weapon_system / surveillance_tool / "
            "financial_instrument / supply_chain_node / software_platform / other"
        )
    )
    classification: Optional[str] = Field(
        None,
        description="Sensitivity: public / internal / confidential / restricted / top_secret"
    )
    strategic_value: Optional[str] = Field(
        None,
        description="Estimated strategic importance: critical / high / medium / low"
    )
    description_detail: Optional[str] = Field(
        None,
        description="Brief technical description of the asset"
    )
    supply_chain_role: Optional[str] = Field(
        None,
        description=(
            "Position in supply chain: "
            "raw_material / component / manufacturing_equipment / assembly / "
            "distribution / end_product / infrastructure / none"
        )
    )
    dual_use_potential: Optional[str] = Field(
        None,
        description=(
            "Dual-use (military-civilian) potential: "
            "high / medium / low / designated / unknown. "
            "'designated' = officially listed under Wassenaar or similar regime"
        )
    )
    controlling_entity: Optional[str] = Field(
        None,
        description="Name of the entity with effective control (may differ from nominal owner)"
    )


# ============================================================
# 5. 사건 (Incident)
# ============================================================

class Incident(BaseModel):
    """
    특이사건 엔티티.
    정상 패턴에서 벗어난 이벤트를 별도 엔티티로 표현.

    NOTE: Graphiti의 엣지(활동)와 별도로, 단독으로 참조·검색해야 하는
    중요 사건은 엔티티로 승격시킴. 일반 활동은 엣지로 충분.
    """
    incident_type: Optional[str] = Field(
        None,
        description=(
            "Type: cyber_attack / personnel_change / policy_shift / "
            "anomalous_transaction / legal_action / geopolitical_event / "
            "supply_chain_disruption / data_breach / "
            "sanctions_action / military_deployment / protest_uprising / "
            "defection / espionage_discovery / assassination / "
            "election_interference / coup_attempt / other"
        )
    )
    severity: Optional[str] = Field(
        None,
        description="Severity: critical / high / medium / low / informational"
    )
    occurred_at: Optional[str] = Field(
        None,
        description="When the incident occurred (best estimate, ISO 8601 string)"
    )
    detected_at: Optional[str] = Field(
        None,
        description="When the incident was detected or reported (ISO 8601 string)"
    )
    status: Optional[str] = Field(
        None,
        description="Current status: ongoing / resolved / under_investigation / unconfirmed"
    )
    confidence: Optional[str] = Field(
        None,
        description="Information confidence: confirmed / probable / possible / doubtful"
    )
    impact_summary: Optional[str] = Field(
        None,
        description="Brief description of actual or potential impact"
    )
    geopolitical_context: Optional[str] = Field(
        None,
        description=(
            "Higher-level geopolitical context this incident belongs to (free text). "
            "e.g., 'US-China tech war', 'Ukraine conflict', 'Taiwan strait tension'"
        )
    )
    information_source_type: Optional[str] = Field(
        None,
        description=(
            "Intelligence collection method that originally obtained this information: "
            "sigint / osint / humint / techint / finint / cyber / unknown"
        )
    )


# ============================================================
# 6. 정책 (Policy) — 신설
# ============================================================

class Policy(BaseModel):
    """
    국가·국제기구가 시행하는 제도적 수단.
    제재, 조약, 교리, 수출통제, 무역협정, 법률, 행정명령 등.

    Incident와의 구분: Incident는 단발 사건 ("미국이 화웨이를 제재 목록에 추가").
    Policy는 지속적 제도 ("Entity List 수출통제 체제").
    하나의 Policy 아래 여러 Incident가 발생할 수 있음.
    """
    policy_type: Optional[str] = Field(
        None,
        description=(
            "Type: sanction / treaty / export_control / trade_agreement / "
            "military_doctrine / legislation / executive_order / "
            "regulation / alliance_charter / other"
        )
    )
    issuing_entity: Optional[str] = Field(
        None,
        description="Entity that enacted/issued this policy (country or org name)"
    )
    target_scope: Optional[str] = Field(
        None,
        description=(
            "Scope of targets: "
            "bilateral / multilateral / global / sector_specific / entity_specific"
        )
    )
    status: Optional[str] = Field(
        None,
        description="Current status: active / expired / under_negotiation / suspended / proposed"
    )
    effective_date: Optional[str] = Field(
        None,
        description="Date when the policy took or takes effect (ISO 8601 string)"
    )
    strategic_impact: Optional[str] = Field(
        None,
        description="Summary of strategic impact (free text)"
    )


# ============================================================
# 7. 캠페인 (Campaign) — 신설
# ============================================================

class Campaign(BaseModel):
    """
    지속적이고 조직적인 활동.
    군사 작전, 영향력 공작, 사이버 캠페인, 사회 운동, 선전전, 경제전 등.

    Incident와의 구분: Incident는 시점이 명확한 단발 사건.
    Campaign은 시작/끝이 모호하고, 여러 Incident를 포함하며,
    여러 Organization이 참여하고, 목표와 이데올로기적 프레이밍이 있음.
    """
    campaign_type: Optional[str] = Field(
        None,
        description=(
            "Type: military_operation / influence_operation / cyber_campaign / "
            "social_movement / propaganda_campaign / economic_warfare / "
            "espionage_campaign / liberation_movement / sanctions_campaign / "
            "counter_intelligence / other"
        )
    )
    objective: Optional[str] = Field(
        None,
        description="Campaign objective or intent (free text)"
    )
    status: Optional[str] = Field(
        None,
        description="Current status: active / dormant / concluded / escalating / unknown"
    )
    scale: Optional[str] = Field(
        None,
        description="Geographic scale: local / national / regional / global"
    )
    started_at: Optional[str] = Field(
        None,
        description="When the campaign started (best estimate, ISO 8601 string)"
    )
    ideological_framing: Optional[str] = Field(
        None,
        description=(
            "Ideological justification or framing used: "
            "anti-terrorism / democracy_promotion / national_security / "
            "class_struggle / anti-imperialism / religious / humanitarian / none / other"
        )
    )
    effectiveness: Optional[str] = Field(
        None,
        description="Effectiveness assessment: high_impact / moderate_impact / low_impact / backfired / unknown"
    )


# ============================================================
# 엔티티 타입 레지스트리
# ============================================================

ENTITY_TYPES = {
    "Person": Person,
    "Organization": Organization,
    "Location": Location,
    "Asset": Asset,
    "Incident": Incident,
    "Policy": Policy,
    "Campaign": Campaign,
}
