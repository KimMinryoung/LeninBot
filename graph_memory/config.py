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

    # ─── Role 관련 (v2.2 신설) ────────────────────────
    # Role = 직책/직위 자체. Person이 Role을 점유한다 = Affiliation.
    ("Person", "Role"): [
        "Affiliation",       # Person이 그 Role을 보유 (X holds role Y)
    ],
    ("Role", "Organization"): [
        "Affiliation",       # Role이 어느 조직 안의 자리인가
    ],
    ("Role", "Location"): [
        "Presence",          # Role의 관할 지역 (e.g. Vice Mayor of Livorno)
    ],

    # ─── Industry 관련 (v2.2 신설) ────────────────────
    # Industry = 산업/섹터. 조직이 산업에 속한다 = Affiliation.
    ("Organization", "Industry"): [
        "Affiliation",       # 조직이 그 산업에 속함
    ],
    ("Industry", "Location"): [
        "Presence",          # 산업의 지리적 집중지
    ],
    ("Policy", "Industry"): [
        "PolicyEffect",      # 정책이 산업에 미치는 영향
    ],
    ("Industry", "Asset"): [
        "AssetTransfer",     # 산업이 사용/생산하는 자산
    ],
    ("Campaign", "Industry"): [
        "ThreatAction",      # 캠페인이 산업을 타겟 (e.g. tariff war)
    ],

    # ─── 폴백 ────────────────────────────────────────
    # Statement와 Causation은 모든 entity 쌍에 대해 사용 가능 — 주제 무관성 때문.
    ("Entity", "Entity"): [
        "Funding",           # 모든 엔티티 간 자금 흐름 가능
        "AssetTransfer",     # 모든 엔티티 간 자산 이전 가능
        "Statement",         # X가 Y에 대해/Y에게 발화 (v2.2)
        "Causation",         # X가 Y의 원인 (v2.2)
    ],
}


# ============================================================
# 제외 엔티티 타입 (노이즈 방지)
# ============================================================
#
# 정보 에이전트에 불필요한 엔티티 추출을 억제.
# 소스 텍스트에 따라 조정 필요.

EXCLUDED_ENTITY_TYPES = [
    # NOTE: Graphiti는 등록된 ENTITY_TYPES의 키만 허용.
    # Date/Number/Concept/Event는 우리 스키마에 없으므로 사용 불가.
    # 노이즈 차단은 NEWS_PREPROCESS_PROMPT_TEMPLATE에서 처리.
]


# ============================================================
# 뉴스 전처리/추출 가이드
# ============================================================

NEWS_PREPROCESS_PROMPT_TEMPLATE = """Extract only knowledge-graph-worthy facts from the news article below.

[INCLUDE]
1) Named persons, organizations, countries, locations (use common English names)
2) Policies, sanctions, treaties, laws, export controls — enactment, amendment, enforcement
3) Military actions, cyberattacks, information operations, asset/weapon/technology transfers
4) Time-stamped events (attacks, announcements, agreements, deployments, crackdowns)
5) Use full official names for countries and organizations (e.g., "United States" not "US", "South Korea" not "ROK", "European Union" not "EU")

[EXCLUDE]
1) Anonymous civilians (e.g., "8-year-old child", "local resident", "witness")
2) Article metadata (reporter name, section name, URL slugs)
3) Emotional rhetoric, background descriptions, duplicate sentences
4) Episode identifiers, filenames, campaign day labels

[OUTPUT FORMAT]
- Numbered list (1., 2., 3. ...)
- One fact per line, in English
- Structure: Subject (actor) — Action — Object — Context (time/location)

[ARTICLE]
{article}
"""


# ============================================================
# 에피소드 소스 타입 매핑
# ============================================================
#
# Graphiti EpisodeType: text, message, json
# 정보 소스별 매핑:

CUSTOM_EXTRACTION_INSTRUCTIONS = """\
## LANGUAGE RULES
- ALL entity names MUST be in English. Translate non-English names (e.g., 러시아 → Russia, 미국 → United States).

## ENTITY NAME NORMALIZATION (CRITICAL)
- Countries: ALWAYS use the full official common name, NEVER abbreviations.
  CORRECT: "United States", "South Korea", "North Korea", "United Kingdom", "European Union", "United Nations"
  WRONG: "US", "USA", "U.S.", "ROK", "DPRK", "UK", "EU", "UN"
- Organizations: Use the most recognized full name, not acronyms.
  CORRECT: "International Monetary Fund", "World Health Organization", "Samsung Electronics"
  WRONG: "IMF", "WHO", "Samsung"
  EXCEPTION: Names where the acronym IS the official name (e.g., "NATO", "OPEC", "NVIDIA")
- People: Use "FirstName LastName" (e.g., "Donald Trump" not "Trump", "Vladimir Putin" not "Putin")
- Use "Russia" not "Russian Federation", "China" not "People's Republic of China"

## ENTITY TYPE SELECTION (CRITICAL — common mistakes to avoid)

10 entity types are available. Pick the most semantically accurate one.

- **Person**: actual individual humans only. NOT roles, NOT positions, NOT collectives.
  WRONG: "Senate Minority Leader", "Secretary of Energy", "senior executives", "Multi-home owners"
  RIGHT: "Chuck Schumer", "Jennifer Granholm", "Vladimir Putin"

- **Role**: official titles and positions, distinct from the people who hold them. Roles persist across holders.
  RIGHT: "Senate Minority Leader", "United States Secretary of Energy", "CEO of Anthropic", "Vice Mayor of Livorno"
  Use Affiliation edges to connect a Person to a Role they hold.

- **Organization**: actual institutional bodies (companies, agencies, NGOs, political parties).
  WRONG: "South Korean financial market" (that's a market/system, not an org), "Manufacturing" (that's an industry)
  RIGHT: "Samsung Electronics", "Federal Reserve", "Democratic Party of Korea"

- **Industry**: economic sectors / value chains, one abstraction level above specific organizations.
  RIGHT: "semiconductor industry", "AI industry", "Bitcoin mining", "defense contracting", "fossil fuels"
  Use Affiliation to connect an Organization to the Industry it belongs to.

- **Asset**: technologies, products, IP, weapons, infrastructure. Tangible or intangible things of strategic value.
  WRONG: "Bitcoin mining" (industry, not asset), "Lost Decades" (era, not asset),
         "EBS social inquiry video", "EMAIL_BRIDGE_ENABLED", "Octree", "task #184" (internal noise — DO NOT extract)

- **Concept**: abstract ideas, ideologies, theories, social phenomena, social classes, historical eras.
  RIGHT: "Marxism", "neoliberalism", "Working Class", "Lost Decades", "stagflation", "non-regular labor"
  WRONG: "Anarcho-capitalism" as Policy (it's an ideology = Concept), "Strategic autonomy" as Policy (it's a doctrine = Concept)

- **Incident**: specific time-bounded events with severity/impact. Real, single events.
  WRONG: "Demonstrations" (too generic), "drone attacks" (too generic), "task #184" (internal noise)
  RIGHT: "Minnesota shooting incident on 2026-03-15", "Israel airstrike on Gaza residential complex"

- **Policy**: enacted institutional measures (laws, sanctions, executive orders, treaties, regulations).
  RIGHT: "Reciprocal Tariff Act 2026", "Section 301", "Executive Order 14123"

- **Campaign**: sustained organized activities (military operations, social movements, propaganda campaigns, wars).
  WRONG: "Task #210" (internal noise), "Trade war" (too generic if no specific framing)
  RIGHT: "Belt and Road Initiative", "MAGA movement", "Iran-Iraq War", "Operation Inherent Resolve"

- **Location**: geographic places, facilities, regions.

## INTERNAL NOISE FILTER (CRITICAL — DO NOT EXTRACT)

NEVER create entities for any of these — they pollute the graph:
- Internal task IDs ("Task #184", "task #210")
- Code identifiers ("EMAIL_BRIDGE_ENABLED", "NameError", "Octree", any UPPER_SNAKE_CASE)
- File paths, environment variables, function/class names
- LLM model identifiers when used as software components ("flux_dev", "rd_fast" — UNLESS the article is specifically about that model)
- Generic placeholders ("the user", "the agent", "the system")
- Game/example code references unless central to the article
- Bot operational details (commit hashes, branch names)

If a sentence is about LeninBot internals or programming details, SKIP it entirely.

## RELATION TYPE RULES (OVERRIDE — 12 valid types)

relation_type MUST be one of: Affiliation, OrgRelation, PersonalRelation, Funding,
AssetTransfer, ThreatAction, Involvement, Presence, PolicyEffect, Participation,
**Statement**, **Causation**.

Pick the most specific match. Notes on the new ones:
- **Statement**: use for "X said/announced/criticized/endorsed/quoted Y". Replaces awkward Affiliation/Involvement uses for speech acts. Direction: speaker → topic.
- **Causation**: use for explicit causal claims ("X caused/triggered/enabled Y"). Direction: cause → effect. Do NOT use for mere temporal sequence or correlation.
- **Affiliation** is now also used for: Person → Role (holds), Role → Organization (part of), Organization → Industry (sector membership).

Do NOT invent new types or use SCREAMING_SNAKE_CASE variants.
"""


# ============================================================
# 엔티티 이름 정규화 매핑 (약어 → 정식명)
# ============================================================
#
# Graphiti의 entity resolution이 짧은 이름(US, UK)에서 실패하므로
# 추출 전 텍스트 레벨에서 약어를 정식명으로 치환.
# graphiti_patches.py의 _patch_normalize_entity_names()에서 사용.

NAME_NORMALIZATION = {
    "us": "United States",
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "dprk": "North Korea",
    "rok": "South Korea",
    "prc": "China",
    "roc": "Taiwan",
    "eu": "European Union",
    "un": "United Nations",
    "imf": "International Monetary Fund",
    "who": "World Health Organization",
    "wto": "World Trade Organization",
    "iaea": "International Atomic Energy Agency",
    "uae": "United Arab Emirates",
    "drc": "Democratic Republic of the Congo",
    "ksa": "Saudi Arabia",
}


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
