"""
정보 에이전트 Graphiti 스키마 — 엣지(관계/연결선) 정의 v2
==========================================================

연결선 = 사람↔조직, 조직↔조직, 사람↔사람 등의 관계와 활동.
Graphiti가 자동으로 바이-템포럴 타임스탬프를 부여:
  - valid_at / invalid_at   : 사실의 실제 유효 기간
  - created_at / expired_at : 시스템 수집/갱신 시점

아래 정의는 그 위에 도메인 특화 속성을 추가하는 것.

v2 변경: PolicyEffect·Participation 신설, Involvement 설명 확장. 총 10종.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# ============================================================
# 1. 소속 관계 (Affiliation)
# ============================================================

class Affiliation(BaseModel):
    """
    사람 → 조직 소속 관계.
    고용, 멤버십, 계약직, 자문 등 모든 소속 형태 포함.
    """
    position: Optional[str] = Field(
        None,
        description="Job title, rank, or role within the organization"
    )
    department: Optional[str] = Field(
        None,
        description="Department or division"
    )
    affiliation_type: Optional[str] = Field(
        None,
        description="Type: employee / contractor / advisor / board_member / agent / informant / member"
    )
    start_date: Optional[datetime] = Field(
        None,
        description="When affiliation began"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="When affiliation ended (None if current)"
    )
    is_current: Optional[bool] = Field(
        None,
        description="Whether this affiliation is currently active"
    )
    access_level: Optional[str] = Field(
        None,
        description="Level of access or authority within the org"
    )


# ============================================================
# 2. 대인 관계 (PersonalRelation)
# ============================================================

class PersonalRelation(BaseModel):
    """
    사람 ↔ 사람 관계.
    직업적·개인적·비공식적 연결 모두 포함.
    """
    relation_type: Optional[str] = Field(
        None,
        description=(
            "Type: colleague / mentor / subordinate / "
            "family / classmate / co_conspirator / "
            "business_partner / rival / handler_asset / unknown"
        )
    )
    context: Optional[str] = Field(
        None,
        description="Context where this relationship exists (e.g., 'MIT alumni network', 'previous co-workers at TSMC')"
    )
    strength: Optional[str] = Field(
        None,
        description="Estimated relationship strength: strong / moderate / weak / unknown"
    )
    first_observed: Optional[datetime] = Field(
        None,
        description="When this relationship was first observed"
    )


# ============================================================
# 3. 조직 간 관계 (OrgRelation)
# ============================================================

class OrgRelation(BaseModel):
    """
    조직 ↔ 조직 관계.
    공식적 파트너십, 경쟁, 공급망, 적대 관계 등.
    """
    relation_type: Optional[str] = Field(
        None,
        description=(
            "Type: subsidiary / partner / supplier / customer / "
            "competitor / joint_venture / investor / "
            "regulatory_target / adversary / allied / unknown"
        )
    )
    agreement_type: Optional[str] = Field(
        None,
        description="Formal agreement if any: MOU / contract / treaty / informal"
    )
    financial_value: Optional[str] = Field(
        None,
        description="Known or estimated financial value of the relationship"
    )
    strategic_significance: Optional[str] = Field(
        None,
        description="Why this relationship matters strategically"
    )


# ============================================================
# 4. 자금 흐름 (Funding)
# ============================================================

class Funding(BaseModel):
    """
    자금 지원, 투자, 재정적 연결.
    사람→조직, 조직→조직, 조직→사람 모두 가능.
    """
    funding_type: Optional[str] = Field(
        None,
        description="Type: investment / grant / donation / contract_payment / subsidy / illicit / unknown"
    )
    amount: Optional[str] = Field(
        None,
        description="Amount with currency (e.g., '$50M', '¥3.2B')"
    )
    purpose: Optional[str] = Field(
        None,
        description="Stated or inferred purpose of funding"
    )
    is_verified: Optional[bool] = Field(
        None,
        description="Whether the funding has been independently verified"
    )


# ============================================================
# 5. 기술 이전 / 자산 이동 (AssetTransfer)
# ============================================================

class AssetTransfer(BaseModel):
    """
    기술, 지적재산, 인력, 장비 등의 이전.
    공급망 보안과 기술 유출 추적의 핵심 관계.
    """
    transfer_type: Optional[str] = Field(
        None,
        description="Type: technology_transfer / ip_licensing / personnel_poaching / equipment_sale / data_sharing / other"
    )
    asset_description: Optional[str] = Field(
        None,
        description="What was transferred"
    )
    legality: Optional[str] = Field(
        None,
        description="Legal status: legal / under_review / sanctioned / illegal / unknown"
    )
    export_control: Optional[str] = Field(
        None,
        description="Relevant export control regime if any (e.g., 'EAR', 'Wassenaar')"
    )


# ============================================================
# 6. 공격/위협 행위 (ThreatAction)
# ============================================================

class ThreatAction(BaseModel):
    """
    위협 행위자 → 대상 간의 공격/적대 활동.
    사이버 공격, 정보공작, 물리적 위협 등.
    """
    action_type: Optional[str] = Field(
        None,
        description=(
            "Type: cyber_attack / espionage / sabotage / "
            "disinformation / reconnaissance / social_engineering / "
            "supply_chain_attack / insider_threat / other"
        )
    )
    technique: Optional[str] = Field(
        None,
        description="Specific technique or MITRE ATT&CK ID if applicable"
    )
    target_asset: Optional[str] = Field(
        None,
        description="What specifically was targeted (system, data, process)"
    )
    outcome: Optional[str] = Field(
        None,
        description="Outcome: successful / partial / failed / unknown / ongoing"
    )
    confidence: Optional[str] = Field(
        None,
        description="Attribution confidence: confirmed / probable / possible / doubtful"
    )


# ============================================================
# 7. 관여 (Involvement)
# ============================================================

class Involvement(BaseModel):
    """
    엔티티 → 사건(Incident) 또는 캠페인(Campaign) 관여.
    누가/어떤 조직이 사건이나 캠페인에 어떻게 관련되었는가.
    Campaign 관여 시에는 Participation 엣지와 병행 사용 가능.
    """
    role_in_incident: Optional[str] = Field(
        None,
        description="Role: perpetrator / victim / witness / investigator / beneficiary / facilitator / unknown"
    )
    evidence_basis: Optional[str] = Field(
        None,
        description="Basis for this association (e.g., 'SIGINT intercept', 'OSINT report', 'insider tip')"
    )
    confidence: Optional[str] = Field(
        None,
        description="Confidence: confirmed / probable / possible / doubtful"
    )


# ============================================================
# 8. 위치 관련 (Presence)
# ============================================================

class Presence(BaseModel):
    """
    엔티티 → 장소 관련.
    거점, 방문, 운영 지역 등.
    """
    presence_type: Optional[str] = Field(
        None,
        description="Type: headquartered / operates_in / visited / resides / deployed_to / transited"
    )
    frequency: Optional[str] = Field(
        None,
        description="Frequency: permanent / regular / occasional / one_time / unknown"
    )
    purpose: Optional[str] = Field(
        None,
        description="Known or inferred purpose of presence"
    )


# ============================================================
# 9. 정책 효과 (PolicyEffect) — 신설
# ============================================================

class PolicyEffect(BaseModel):
    """
    정책(Policy) ↔ 엔티티 간 영향 관계.
    시행↔대상↔영향 삼각관계를 표현.
    예: 수출통제 → 기업 제재, 조직 → 정책 시행/집행.
    """
    effect_type: Optional[str] = Field(
        None,
        description=(
            "Type of effect: "
            "enacts / targets / restricts / enables / exempts / "
            "circumvents / enforces / violates"
        )
    )
    impact_description: Optional[str] = Field(
        None,
        description="Specific description of the impact"
    )
    compliance_status: Optional[str] = Field(
        None,
        description=(
            "Compliance status: "
            "compliant / non_compliant / partially_compliant / exempt / unknown"
        )
    )


# ============================================================
# 10. 캠페인 참여 (Participation) — 신설
# ============================================================

class Participation(BaseModel):
    """
    엔티티 → 캠페인(Campaign) 참여.
    Involvement(사건 관여)와 구분 — Participation은 지속적 활동에 대한 역할.
    """
    role: Optional[str] = Field(
        None,
        description=(
            "Role in campaign: "
            "leads / conducts / participates / supports / opposes / "
            "targets / funds / unknown"
        )
    )
    contribution: Optional[str] = Field(
        None,
        description="Specific contribution or activity within the campaign"
    )
    commitment_level: Optional[str] = Field(
        None,
        description="Level of commitment: full / partial / token / coerced / unknown"
    )


# ============================================================
# 엣지 타입 레지스트리
# ============================================================

EDGE_TYPES = {
    "Affiliation": Affiliation,
    "PersonalRelation": PersonalRelation,
    "OrgRelation": OrgRelation,
    "Funding": Funding,
    "AssetTransfer": AssetTransfer,
    "ThreatAction": ThreatAction,
    "Involvement": Involvement,
    "Presence": Presence,
    "PolicyEffect": PolicyEffect,
    "Participation": Participation,
}
