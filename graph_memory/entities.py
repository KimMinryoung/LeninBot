"""
정보 에이전트 Graphiti 스키마 — 엔티티 정의
==============================================

수집 대상: 사람, 조직, 활동, 특이사건
Graphiti의 Custom Entity Types API를 사용하여 정의.

모든 필드는 Optional — 수집 시점에 모든 정보가 확보되지 않을 수 있음.
수집이 누적되면서 Graphiti의 엔티티 해소(entity resolution)가
동일 엔티티의 속성을 점진적으로 채워감.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# ============================================================
# 1. 사람 (Person)
# ============================================================

class Person(BaseModel):
    """
    개인 인물 엔티티.
    대상: 주요 인물, 의사결정권자, 기술 전문가, 위협 행위자 등.
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
        description="Current primary role or job title"
    )
    expertise: Optional[str] = Field(
        None,
        description="Domain expertise areas (e.g., 'chip design, EUV lithography')"
    )
    clearance_level: Optional[str] = Field(
        None,
        description="Known security clearance or access level if applicable"
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
    대상: 기업, 정부기관, 연구소, 위협 그룹, NGO 등.
    """
    org_type: Optional[str] = Field(
        None,
        description=(
            "Type of organization: "
            "corporation / government_agency / military / "
            "research_institute / threat_group / ngo / consortium / other"
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
    employee_count: Optional[int] = Field(
        None,
        description="Approximate number of employees"
    )
    parent_org: Optional[str] = Field(
        None,
        description="Name of parent organization if subsidiary"
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
    시설, 거점, 회의 장소 등 활동의 공간적 맥락.
    """
    location_type: Optional[str] = Field(
        None,
        description="Type: facility / city / region / country / virtual"
    )
    coordinates: Optional[str] = Field(
        None,
        description="GPS coordinates if known (lat, lon)"
    )
    significance: Optional[str] = Field(
        None,
        description="Why this location matters (e.g., 'fab site', 'R&D center', 'C2 server location')"
    )


# ============================================================
# 4. 자산/기술 (Asset)
# ============================================================

class Asset(BaseModel):
    """
    기술, 제품, 지적재산, 인프라 등 가치 있는 대상.
    """
    asset_type: Optional[str] = Field(
        None,
        description=(
            "Type: technology / product / patent / infrastructure / "
            "data / financial_instrument / other"
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
            "supply_chain_disruption / data_breach / other"
        )
    )
    severity: Optional[str] = Field(
        None,
        description="Severity: critical / high / medium / low / informational"
    )
    occurred_at: Optional[datetime] = Field(
        None,
        description="When the incident occurred (best estimate)"
    )
    detected_at: Optional[datetime] = Field(
        None,
        description="When the incident was detected or reported"
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


# ============================================================
# 엔티티 타입 레지스트리
# ============================================================

ENTITY_TYPES = {
    "Person": Person,
    "Organization": Organization,
    "Location": Location,
    "Asset": Asset,
    "Incident": Incident,
}
