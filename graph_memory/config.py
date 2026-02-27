"""
정보 에이전트 Graphiti 스키마 — 엣지 타입 매핑 + 설정 v2
=========================================================

어떤 엔티티 쌍 사이에 어떤 관계가 허용되는지 정의.
Graphiti의 edge_type_map은 LLM의 관계 추출을 안내하는 역할.
매핑에 없는 쌍은 기본 RELATES_TO로 캡처됨 (정보 손실 없음).

v2 변경: Policy·Campaign 엔티티 추가에 따른 매핑 확장.
PolicyEffect·Participation 엣지 추가.
"""

# ============================================================
# 엣지 타입 매핑
# ============================================================
#
# 형식: (소스 엔티티 타입, 타겟 엔티티 타입) → [허용 엣지 타입 리스트]
#
# "Entity"는 모든 타입에 매칭되는 와일드카드.
# 순서: 구체적 매핑이 우선, "Entity" 폴백이 후순위.
# 동일 키가 중복되면 뒤의 것이 덮어쓰므로, 같은 쌍의 엣지는 하나의 리스트로 병합.

EDGE_TYPE_MAP = {
    # ─── 사람 ↔ 조직 ─────────────────────────────────
    ("Person", "Organization"): [
        "Affiliation",       # 소속 (고용, 멤버십, 자문 등)
        "Funding",           # 자금 지원/수령
        "AssetTransfer",     # 기술/자산 이전
        "ThreatAction",      # 내부자 위협 등
    ],
    ("Organization", "Person"): [
        "ThreatAction",      # 조직이 개인을 타겟
    ],

    # ─── 사람 ↔ 사람 ─────────────────────────────────
    ("Person", "Person"): [
        "PersonalRelation",  # 대인 관계 (동료, 가족, 공모 등)
        "Funding",           # 개인 간 자금 흐름
        "AssetTransfer",     # 개인 간 기술/정보 전달
    ],

    # ─── 조직 ↔ 조직 ─────────────────────────────────
    ("Organization", "Organization"): [
        "OrgRelation",       # 파트너십, 경쟁, 공급망 등
        "Funding",           # 투자, 보조금, 계약
        "AssetTransfer",     # 기술 이전, 라이선싱
        "ThreatAction",      # 조직 간 적대 행위
    ],

    # ─── 사건 관여 ───────────────────────────────────
    ("Person", "Incident"): [
        "Involvement",       # 사람의 사건 관여
    ],
    ("Organization", "Incident"): [
        "Involvement",       # 조직의 사건 관여
    ],

    # ─── 자산 관련 ───────────────────────────────────
    ("Organization", "Asset"): [
        "AssetTransfer",     # 조직의 자산 보유/이전
    ],
    ("Person", "Asset"): [
        "AssetTransfer",     # 개인의 자산 접근/이전
    ],

    # ─── 장소 관련 ───────────────────────────────────
    ("Person", "Location"): [
        "Presence",          # 사람의 장소 관련
    ],
    ("Organization", "Location"): [
        "Presence",          # 조직의 장소 관련
    ],
    ("Incident", "Location"): [
        "Presence",          # 사건 발생 장소
    ],

    # ─── 정책 관련 (v2 신설) ──────────────────────────
    ("Policy", "Organization"): [
        "PolicyEffect",      # 정책이 조직에 미치는 영향 (제재 등)
    ],
    ("Policy", "Person"): [
        "PolicyEffect",      # 정책이 개인에 미치는 영향 (입국금지 등)
    ],
    ("Policy", "Asset"): [
        "PolicyEffect",      # 정책이 자산에 미치는 영향 (수출통제 등)
    ],
    ("Policy", "Location"): [
        "PolicyEffect",      # 정책이 지역에 적용 (군사 교리 등)
    ],
    ("Organization", "Policy"): [
        "PolicyEffect",      # 조직이 정책을 시행/집행
    ],

    # ─── 캠페인 관련 (v2 신설) ────────────────────────
    ("Person", "Campaign"): [
        "Participation",     # 인물의 캠페인 참여
        "Involvement",       # 캠페인 관여 (Participation과 병행 가능)
    ],
    ("Organization", "Campaign"): [
        "Participation",     # 조직의 캠페인 참여
        "Involvement",       # 캠페인 관여 (Participation과 병행 가능)
    ],
    ("Campaign", "Organization"): [
        "ThreatAction",      # 캠페인이 조직을 공격/타겟
    ],
    ("Campaign", "Asset"): [
        "ThreatAction",      # 캠페인이 자산을 타겟
    ],
    ("Campaign", "Location"): [
        "Presence",          # 캠페인의 지리적 범위
    ],
    ("Campaign", "Incident"): [
        "Involvement",       # 캠페인 내 개별 사건 연결
    ],
    ("Campaign", "Policy"): [
        "PolicyEffect",      # 캠페인과 정책의 관계
    ],

    # ─── 폴백 ────────────────────────────────────────
    ("Entity", "Entity"): [
        "Funding",           # 모든 엔티티 간 자금 흐름 가능
        "AssetTransfer",     # 모든 엔티티 간 자산 이전 가능
    ],
}


# ============================================================
# 제외 엔티티 타입 (노이즈 방지)
# ============================================================
#
# 정보 에이전트에 불필요한 엔티티 추출을 억제.
# 소스 텍스트에 따라 조정 필요.

EXCLUDED_ENTITY_TYPES = [
    # 예시: 뉴스 기사에서 불필요한 엔티티
    # "Product",   # 일반 상품 정보가 노이즈인 경우
    # "Date",      # Graphiti가 시간을 자체 처리하므로 중복
]


# ============================================================
# 에피소드 소스 타입 매핑
# ============================================================
#
# Graphiti EpisodeType: text, message, json
# 정보 소스별 매핑:

EPISODE_SOURCE_MAP = {
    # 소스 카테고리        → (EpisodeType, source_description)
    "osint_news":          ("text",    "Open source news article"),
    "osint_social":        ("text",    "Social media post or thread"),
    "osint_forum":         ("text",    "Forum post from dark web or public forum"),
    "cve_feed":            ("json",    "CVE vulnerability feed entry"),
    "threat_report":       ("text",    "Threat intelligence report"),
    "internal_siem":       ("json",    "Internal SIEM alert or log"),
    "internal_report":     ("text",    "Internal analyst report or memo"),
    "humint_debrief":      ("text",    "Human intelligence debrief notes"),
    "financial_record":    ("json",    "Financial transaction or filing record"),
    "patent_filing":       ("json",    "Patent application or grant record"),
    "personnel_change":    ("text",    "Personnel movement or organizational change notice"),
    "diplomatic_cable":    ("text",    "Diplomatic or policy communication"),
}
