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
    적대적/공격적 행위 관계.

    Use this for:
    - 사이버 공격, 군사 타격, 침투, 사보타주, 첩보활동, 강압적 위협
    - 특정 행위자가 특정 대상을 실제로 타격/표적화/교란한 경우

    Do NOT use for:
    - 단순 언급, 배경 설명, 일반적 연관성(이 경우 RELATES_TO/MENTIONS로 귀결)
    - 정책/규제의 제도적 영향(이 경우 PolicyEffect)

    Examples:
    - "Russia launched missiles at Kyiv" -> ThreatAction
    - "APT41 infiltrated a semiconductor supplier network" -> ThreatAction
    - "US sanctions affected Huawei" -> PolicyEffect (not ThreatAction)
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
    정책/규제/조약이 엔티티에 미치는 직접적 효과 관계.

    Use this for:
    - 제재, 수출통제, 법령, 행정명령, 조약의 적용/집행/위반/면제
    - 정책 주체(정부/조직)와 정책, 정책과 피영향 대상 사이의 직접 연결

    Do NOT use for:
    - 물리적/사이버 공격 행위 자체(ThreatAction)
    - 단순 조직 간 협력/연관(OrgRelation/RELATES_TO)

    Examples:
    - "Entity List designation blocks exports to Company X" -> PolicyEffect
    - "Ministry Y enforces export controls on dual-use chips" -> PolicyEffect
    - "Group Z attacked refinery A" -> ThreatAction (not PolicyEffect)
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
    엔티티의 캠페인 단위 지속 참여 관계.

    Use this for:
    - 조직/인물이 캠페인을 주도·지원·자금조달·반대 등 지속 역할을 수행

    Do NOT use for:
    - 단일 사건 1회 관여만 표현할 때(Incident + Involvement 사용)
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
# 11. 발화·성명 (Statement) — v2.2 신설
# ============================================================

class Statement(BaseModel):
    """
    공개 발언, 성명, 선언, 인용 관계.

    Use this for:
    - 누가(Person/Organization/Campaign) 무엇에 대해 무엇을 말했는가
    - 발표, 비판, 부인, 옹호, 위협, 약속, 인용
    - 정치적 입장 표명 (특히 ThreatAction에 해당하지 않는 비물리적 비판/대립)

    Do NOT use for:
    - 단순 멘션·연관성 (RELATES_TO/MENTIONS)
    - 물리적/사이버 공격 (ThreatAction)
    - 정책 시행 (PolicyEffect)

    Examples:
    - "Vladimir Putin stated Russia's position on sanctions" -> Statement
    - "Anthropic announced Claude Opus 4.6 release" -> Statement (Org → Asset)
    - "Lee Jae-myung criticized Yoon Suk-yeol's economic policy" -> Statement (Person → Policy)
    - "Marx wrote Capital Volume I" -> Statement (Person → Asset)
    """
    statement_type: Optional[str] = Field(
        None,
        description=(
            "Type: announcement / declaration / claim / accusation / "
            "denial / threat / promise / criticism / endorsement / "
            "quote / authored / commented / other"
        )
    )
    medium: Optional[str] = Field(
        None,
        description=(
            "Where the statement was made: press_conference / social_media / "
            "interview / speech / op_ed / official_release / book / paper / "
            "leak / court_filing / unknown"
        )
    )
    audience: Optional[str] = Field(
        None,
        description="Intended or actual audience"
    )
    statement_date: Optional[str] = Field(
        None,
        description="When the statement was made (ISO 8601)"
    )
    verbatim_excerpt: Optional[str] = Field(
        None,
        description="Short verbatim quote if available"
    )


# ============================================================
# 12. 인과 관계 (Causation) — v2.2 신설
# ============================================================

class Causation(BaseModel):
    """
    인과 관계 — 한 entity/사건이 다른 entity/사건의 원인이거나 기여요인.

    Use this for:
    - "X가 Y를 야기했다" / "X가 Y로 이어졌다"
    - "X는 Y의 기여요인이었다"
    - "X 없이는 Y가 불가능했다"
    - 분석가가 명시적으로 주장하는 인과 사슬

    Do NOT use for:
    - 단순 시간순 (먼저 일어났다고 인과 아님)
    - 정책의 기계적 효과 (PolicyEffect를 사용)
    - 단순 상관관계나 동시 발생

    Examples:
    - "2008 financial crisis triggered global recession" -> Causation
    - "Inflation accelerated by Fed rate cuts" -> Causation
    - "Trump tariffs led to retaliatory Chinese rare-earth restrictions" -> Causation
    """
    causal_type: Optional[str] = Field(
        None,
        description=(
            "Type: direct_cause / contributing_factor / triggered / "
            "enabled / prevented / accelerated / mitigated / counterfactual"
        )
    )
    confidence: Optional[str] = Field(
        None,
        description="Causal claim confidence: confirmed / probable / hypothesized / disputed / unknown"
    )
    mechanism: Optional[str] = Field(
        None,
        description="Brief description of the causal mechanism if known"
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
    "Statement": Statement,
    "Causation": Causation,
}
