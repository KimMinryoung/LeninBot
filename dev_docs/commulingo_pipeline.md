# CommuLingo 영속 편집 파이프라인

`commulingo_pipeline/`는 인물·용어의 신규 등록과 보강을 단계별 작업으로 실행한다.
진입점은 `scripts/commulingo_pipeline.py`다. 모듈 import나 조회 명령은 DDL이나
LLM 호출을 하지 않는다. 운영은 `phase=live`, `legacy_shared_budget=true`, `term_editorial_service=true`다.
실행 종료 5분 뒤 타이머가 다음 배치를 재개한다. 한 배치는 기본 12단계·1800초이며, 마지막 단계에서 독립 승인이 끝나면
같은 작업의 공개 저장만 한 단계 추가로 완료한다(최대 13단계, 시간 제한 유지).
독립 검토를 통과한 변경에 별도 일일 반영 건수 제한을 두지 않는다.
기존 작성 스크립트는 롤백을 위해 유지한다. `leninbot-commulingo-batch`는 사건·연결만
실행하고 과거 제안의 review 타이머도 유지한다.

사건 작업은 `knowledge_graph_search`를 사용하므로 `leninbot-commulingo-events.service`에
`neo4j_password` credential이 필요하다. 실행 CLI는 이 값이 없으면 LLM 호출 전에
exit 1로 중단한다(`--print-candidate` 조회는 제외). 유닛의 credential 변경은 설치와
daemon-reload 후 다음 타이머 실행부터 적용된다.

## 대기열과 선정

명시적 migration `scripts/schema_migrations.py --only commulingo-pipeline`이
PostgreSQL 작업·근거·artifact·비용·시범 반영 테이블을 생성한다.
작업은 kind, action, target, topic, baseline, reason, priority와 payload를 가진다.
DB는 같은 대상·주제의 활성 작업을 하나만 허용하며, 계획기는 같은 대상의 활성 작업이
있으면 새 보강 묶음을 만들지 않는다. 조사 시작 시 실제 저장소 revision을 읽고
research artifact에 고정한다. 계획의 baseline은 변경 감지용 타임스탬프이며 저장 권한 토큰이 아니다.

인물은 기본 정보·근거·한영 누락·국적·일화·상세 절, 용어는 정의·역사적 맥락·유사 개념
구별·사례·관계를 선정한다. 12개 절은 상한이며 분량은 품질 점수가 아니다.
운영자가 요청한 gap을 우선하고 네 작업군을 순환 배정한다. 검토·반영 준비 작업은 먼저
처리한다. 인물 보강의 순서는 중요도이며, 현재 기준은 연결된 역사 사건 수다(`priority = 100 - min(사건 수, 79)`,
운영자 결정 2026-09-17). 매 tick의 계획 단계가 미착수 ready 묶음의 priority를 현재 연결 수로 갱신하므로
연결이 늘면 순서가 앞당겨진다. 이전의 생성 순(알파벳) 처리는 폐기했다.
인물 상세 절의 상한은 연결 사건 수에 따른다(0건 2개, 1~2건 3개, 3~5건 5개, 6건 이상 12개;
`store.SECTION_CAP_SQL`). 절 조사 단계는 기존 절 목록을 보고 남은 주요 국면이 없으면 not_applicable로 끝내도록
지시받으며, 그 판단은 기존처럼 180일 유예로 기록된다.
반영 후 재보강 유예: 파이프라인이 승인·반영한 편집(submit artifact status=approved)이 있는 인물은 연결 사건
6건 이상이면 14일, 나머지는 90일 동안 새 보강 후보에서 제외된다(`store.PERSON_IN_GRACE_SQL`, 운영자 결정 2026-09-17).
계획 단계는 유예 중인 인물의 미착수 ready 묶음도 cancelled로 정리하며, 유예가 끝나면 다시 선정된다.
용어 보강(운영자 결정 2026-09-17): 순서는 본문(ko 또는 en)이 빈 용어가 먼저, 그 안에서는 공개 연구 문서가
그 용어를 링크한 수(frontend 렌더 캐시 `data/cache/report-renders.json`의 실제 링크, `commulingo_pipeline/mentions.py`)가
많은 순이다(`store.term_priority`). 파이프라인이 승인·반영한 편집이 있는 용어는 90일 유예, 본문이 이미
ko 2,000자 또는 en 4,500자 이상인 용어(용어 본문에는 schema 상한이 없어 절대값)는 통째 재작성을 피하기 위해 제외,
사건 중복 검사는 **신규 등록에만** 적용한다: gap·발견 후보의 이름이 역사 사건 제목과 같으면 등록하지 않는다
(`term_event_overlap_allow`의 id는 예외, 현재 battle-of-lake-khasan). 기존 용어는 이 규칙으로 빠지지 않는다.
사건 서술을 옮긴 것에 불과해 더 보강하지 않을 기존 용어는 `term_enrichment_exclude`에 명시한다(현재 비어 있음;
2026-09-17 운영자 지시로 doctors-plot, kronstadt-rebellion-1921, leningrad-affair, volga-famine은 사전에서 삭제했다). 계획 단계가 미착수 ready 묶음의 priority를
갱신하고 자격을 잃은 묶음은 cancelled로 정리한다.
발견(discover) 작업은 `config/commulingo_pipeline.json`의 `discovery=false`로 중단했다(운영자 결정
2026-09-17). 꺼져 있으면 공개 자료(보고서·사건·인물 카드)에서 새 항목을 캐는 작업을 만들지 않고, 대기 중이던
자료 작업은 계획 단계에서 cancelled로 정리한다. 명시적 요청(curation gap)은 발견이 아니라 등록이므로 이 스위치와
무관하게 계속 처리된다: 대상 ID가 있으면 일반 선정 경로, 없으면 `gap:` 자료로 slug를 정하는 단일 후보 discover 단계를 거친다.
명시 gap의 kind·label·mention은 실행기가 덮어쓰며 모델의 표기 차이로 거절하지 않는다(과거에는 다섯 번 거절 뒤
빈 결과로 끝나 요청 31건이 pending인 채 방치됐다). 빈 결과에는 20자 이상의 reason이 필요하고 gap은 그 사유와 함께
`skipped`가 된다. 조사 단계가 "대상이 이미 존재"로 끝나면 gap을 resolved_id와 함께 done으로 닫는다. frontend의 유효한 완료 주제와 대기 제안을 존중한다.

보강 후보는 계획 한도를 적용하기 전에 kind·target 기준으로 묶고 `topic=enrichment`,
`payload.topics`에 원래 주제를 보존한다. 활성 작업이 있는 대상은 제외한다.
진행 중 내용의 baseline이 바뀌어도 중복 후보로 계획 한도를 차지하지 않는다.
완료 작업은 기존처럼 baseline이 같은 경우에만 재선정 유예를 적용한다.
명시적 gap과 일반 보강 후보가 같은 대상이면 gap을 우선하고 모든 요청 번호를 보존한다.
인물 기본정보·소개·국적·일화와 용어의 여러 주제는 한 조사·초안·검토로 처리한다.
인물 상세 절은 별도 저장 계약이므로 같은 작업 안에서 카드 주제 처리 후 새 research 단계로
넘어간다. 이때 `remaining_topics`를 단계 artifact와 같은 트랜잭션에서 저장한다.
새 조사는 현재 revision을 다시 읽으며 이전 초안·검토 결과를 다음 절에 재사용하지 않는다.
수집 원문은 같은 작업에 남아 재사용할 수 있지만 독립 검토는 계속 직접 원문을 확인한다.
no-edit 판단은 현재 묶음의 모든 주제에 유효해야 하며 공통 서비스에 원래 주제별로 기록한다.

`consolidate`는 미리보기, `consolidate --apply`는 기존 미착수 보강 작업을 통합한다.
ready/research·attempts=0이며 artifact·수집 원문 연결·비용 이력이 없는 대상만 합친다.
같은 대상에 진행·보류·검토 필요 작업이 있으면 그 대상 전체를 유지한다.
대표 작업 번호를 유지하고 나머지는 cancelled 및 bundled_into로 연결하여 이력을 보존한다.
이 cancelled는 편집 완료가 아니다. 다음 tick에서도 남은 미착수 작업의 통합을 시도한다.

공개 연구 문서·사건·인물·미해결 gap의 내용 hash를 비교해 발견 작업을 만든다.
발견기는 실제 원문 언급을 제시하고 기존 이름·별칭을 검색한다. 한 문서에서 최대 네 후보,
명시적인 gap에서는 요청한 종류·한국어 이름과 일치하는 한 후보만 허용한다.
후보 생성과 처리한 문서 hash는 같은 트랜잭션에서 저장한다. 서로 다른 문서의 언급 수는
신규 후보 사이의 우선순위에만 사용하며, 역사적 중요성이나 출처 독립성을 증명하지 않는다.
동일 문서·내용 hash의 활성 발견 작업도 조회 한도 적용 전에 제외하여 다음 문서가
선정되도록 한다. 내용 hash가 달라지면 새 탐색 후보가 된다.

## 단계와 복구

`discover → research → draft → validate → review → submit → complete` 중 필요한
단계만 실행한다. 편집 불필요 판단은 research 결과를 먼저 저장한 뒤 별도 judge 단계가
동일한 판단으로 현재 보강 주제들의 상태를 기록한다. 각 단계 완료 결과는 별도 artifact다. 120초 lease를 30초마다 갱신하고
단계는 480초로 제한한다. 배치 단계 한도에서 추가하는 저장은 claim 후에도
submit 단계인지 확인하며, 다른 단계로 바뀌었으면 새 조사를 시작하지 않고 반환한다. 소유권을 잃으면 실행을 취소하며 예전 lease의 결과 저장을 거부한다.
실패는 1시간 뒤 해당 단계부터 재시도하고 같은 단계의 3회 실패는 escalated로 남긴다.
DeepSeek가 입력 검열(`Content Exists Risk`, HTTP 400)로 거부하면 같은 소스로는 재시도해도 같으므로
`model_call`이 그 단계를 GPT-5.6 Terra(`provider_fallback=openai`, `gpt56terra`)로 한 번 다시 돌리고, 성공하면 job payload에
`provider_fallback`을 남겨 이후 단계(draft·validate·review)는 처음부터 GPT로 실행한다. attempt·artifact
metrics의 `provider_fallback`으로 식별한다. 다른 오류는 그대로 escalation 규칙을 따른다.

조사기는 field별 주장과 **표시된 문단의 passage 라벨**(`S2@12303` = 출처 핸들@문단 시작 offset, 한 출처의 1~3개)을
제출한다. 실행기는 출처를 표시할 때 문단(빈 줄이 아닌 각 줄, 3,000자 초과는 문장 끝에서 분할)마다 라벨을 앞에 붙이고
그 문단의 범위와 텍스트 다이제스트를 기억하므로(`evidence.Passages.show` — 조사·검토 두 레인이 같은 등록부와 해석 규칙
`Passages.resolve`를 쓴다), claim의 라벨을 문자 범위로 바꾸는 데 복사·매칭·세기가 없다(`evidence.resolve_passages`, 라벨 ≤8개): 여러 출처의 라벨이나 한 범위(6,000자)에 안 들어갈 만큼 떨어진 문단은 같은 field·문장의
claim 여러 개로 나누고(첫 한 시간 거절 15건이 이 두 모양), 표시된 적 없는 라벨은 다른 라벨이 있으면 무시·없으면 claim 번호를 붙여
거절한다. 표시 뒤 바뀐 텍스트도 거절. 길이 하한은 없다(짧은 줄의 근거도 인용 가능; 뒷받침 여부는 Jev 인용 게이트가 판단). 인용문 복사+접기 매칭(2026-09-19 하루)과 240자 chunk 번호(그 전)는
폐기했다 — 복사가 한 글자 어긋나거나 번호가 스냅샷과 어긋나 하루 91~159건이 거절됐다(운영자 결정 2026-09-20: 근거는
형식이 아니라 내용이고, 모델이 읽은 위치는 실행기가 이미 안다). 짧은 핸들(S1)은 내용 hash 기반 영속 ID로 변환해
artifact에 저장한다. 알 수 없는 ID에는 사용 가능한 ID와 URL을 돌려준다. 조사·작성·공통 저장 검증은
근거 개수 상한을 두지 않으며, 개수 때문에 거절하거나 재조사하지 않는다. 독립 검토의 checks도
개수로 거절하지 않는다.
조사는 실제 변경 사실을 뒷받침하면 멈춘다. 출처·주장 개수 목표를 두지 않고 같은 사실의
출처나 인용을 늘리지 않는다. 한 출처를 여러 필드에 재사용하고 추가 검색은 불확실성·상충·누락 해소에 한한다.
한 URL은 한 조사 호출 안에서 **스냅샷 하나**다(`SourcePages`): 오프셋을 나눠 가져온 페이지는 같은
스냅샷 뒤에 이어 붙고(페이지는 내용 hash로 식별) 표시에 `Characters a..b of N`을 적는다. 핸들 S번호도
URL당 하나로 고정된다. 이전 시도에서 남은 같은 URL의 스냅샷들은 조사 시작 시 오래된 순으로 병합하되,
이미 스냅샷 안에 있는 페이지나 현재 텍스트를 접두로 갖는 이전 병합본은 **이어 붙이지 않고 그대로 채택**한다.
2026-09-19까지는 시도마다 저장된 이전 병합본까지 다시 이어 붙여 스냅샷이 시도당 기하급수로 커졌고(job 2523
kosygin: 23 MB→95 MB→453 MB), 6번째 시도가 anon-rss 9 GB에서 OOM-kill되기 전 호스트가 한 번 멎었다.
URL당 스냅샷 상한은 `MAX_SNAPSHOT_CHARS`(200만 자)이며, 저장된 초과 스냅샷은 seed에서 건너뛰고 상한을 넘길
병합은 새 페이지에서 다시 시작한다. 유닛에도 `MemoryHigh=2G`/`MemoryMax=3G`를 두어 다시 생기면 tick만 죽는다.
`tests/test_commulingo_source_pages.py`.
reason이 probe·placeholder·진행 메모("Investigating … before returning")로 시작하는 결과 호출은
길이와 무관하게 거절한다 — 2026-09-19에 그런 호출 두 건이 `sources_unavailable`로 통과해 작업을
90일 미뤘다. 원문 일치·만료·6000자 이내 인용 검증은 유지한다.
추출문에 PostgreSQL text가 수용하지 않는 NUL이 있으면 U+FFFD로 바꾼 뒤 hash와 범위를 계산한다.
인용 위치 확인 뒤 **인용 지지 게이트**(`citation_gate.py`, registry `commulingo_citation_support`, Jev)가 claim마다
발췌가 주장을 뒷받침하는지 판정한다(2026-09-19; 선택지 supports/partially_supports/contradicts/unrelated — partially_supports는 복합 주장의 일부만 덮는 발췌로 통과·기록만). 고신뢰 unrelated/contradicts 또는 봇 확인·동의 안내 같은 boilerplate
발췌는 결과 호출을 거절해 그 claim만 고치게 하고, 판정 수치는 각 claim의 `citation_check`에 남는다(거절 문구는 저장하지 않음). 반박 출처(`stance: disputes`)는 `contradicts`가 정상이다.
`enforce=false`로 shadow, `enabled=false`로 중단, 판정 모델 불가 시 통과. 표본 30쌍 중 조사·검토를 모두 통과한 무관 인용
5건(Britannica 봇 페이지 포함)을 오탐 없이 3건 즉시·2건 유보로 가려냈다(`dev_docs/jev_system_one_adoption.md` 4.5).
집계(`citation_checks`·`citation_rejections`·`citation_unavailable`)는 research artifact의 `metrics`에 남는다. 같은 `metrics`에
`search_triage`(shadow, `commulingo_pipeline/search_triage.py`: web_search hit마다 Jev의 directly/possibly/unrelated 판정과 confidence, 표시는
바꾸지 않음)도 남는다 — `dev_docs/jev_system_one_adoption.md` 4.13.
독립 검토에도 같은 게이트가 있다(`check_review_checks`, registry `commulingo_review_citation_support`): 검토자의
`checks[].quote`가 `finding`이 확인한다고 적은 사실을 담는지 판정해 각 check의 `citation_check`와 review artifact
`metrics`의 `review_citation_*`에 남긴다. 훅은 `make_handlers(..., gate=review_gate(usage))`로 결정이 기록되기 전에 돌며 파이프라인 검토와 검토 타이머의
독립 검토(`scripts/commulingo_person_reviewer.py`) 양쪽에 걸린다. 승인 메모에 넣는 checks에서는 판정 수치를
뺀다(`review_note_checks`). 저장된 검토
40건 기준선에서 무관 인용 2건을 오탐 없이 잡았고(4.8), **`enforce=true`** — 고신뢰
무관·반박 인용을 가진 결정 호출을 그 check만 지목해 거절한다(`enforce=false`면 기록만).
인물 초안에서 `citizenship.code`·`fate.kind`(create·update), 인물 create 초안에서 `groupId`·`role`, 용어 create 초안에서 `category`는 작성기가 아니라 실행기가 Jev로 배정한다(`runtime_tools/commulingo_classify.py`,
`dev_docs/commulingo_editorial.md` "인물 분류 자동 배정"); 판정 수치는 draft artifact `metrics.classification`에 남는다. 작성기는 제한된
초안 도구와 사전 조회만 받는다. 실행기가 6000자 이내의 원문 인용과 기준 revision을 붙인다.
작성 모델에는 evidence/revision을 수정하는 인자가 없다. 인물 상세 절도 같은 경로를 사용한다.
인물 상세 절 초안은 절 하나(slug·heading·body·sortOrder)만 받는다. 조사가 여러 절이나 기존 절 정정을
뒷받침하면 가장 중요한 절 하나를 쓰고 나머지는 결과 도구의 `notes`에 적는다. `notes`는 발행 성공 뒤
frontend 편집 RPC의 `note` 명령으로 `commulingo_editorial_notes`(대상별 작업 메모, 비공개)에 저장되고,
같은 항목의 다음 작업이 `read`로 받는 `current.notes`에 최신 20건이 실린다. 조사 단계는 이를 출발점으로
삼도록 지시받는다. 독립 검토가 escalate/reject로 끝나거나 같은 초안이 두 번 수정되지 않아 보류되면
검토 사유도 같은 저장소에 `검토 <결정> (작업 N)` 메모로 남겨, 다음 작업이 같은 상충에서 다시 시작하지 않게 한다. 절 조사는 body 근거만 수집한다(heading 키 주장은 제목 제안일 뿐이라, 그것만 받은
작성기가 제목을 본문으로 낸 사례가 2026-09-18 bukharin·voroshilov·stucka). 작성 단계는 신설 절의 slug가
인물 id와 같거나, heading이 인물 이름과 같거나, body가 ko 200자·en 300자에 못 미치면 거절하고, 독립
검토에는 사실이 맞더라도 작업 계획 문장은 revise하라고 지시한다. 본문 문구를 패턴 검색해 거르지는
않는다. 2026-09-19 예조프 작업(5432)이 절 셋의 계획을 절 하나로 제출해 공개된 사고와, 9-15 이후 절
101건 중 49건이 인물 id를 slug로 쓰고 26건이 인물 이름을 heading으로 쓴 결과의 재발 방지다.
sortOrder는 절 도구와 같은 YYYYMM 키다.
작성 단계의 사전 조회는 get_person/get_term/get_office/get_event/get_sections만 도구 schema에 노출하며
최대 세 번으로 제한한다. 목록 탐색이나 검색 action을 먼저 보여준 뒤 거절하지 않는다.
검색 전용 q/group_id/status/limit 인자와 설명도 작성 schema에서 제외하고, 각 get action에
해당 ID를 필수로 요구한다. 관련 ID를 모르면 검색을 반복하지 않고 그 관계의 추가를 생략한다.
인물 작성에는 현재 분류 ID·제목·설명을 미리 제공하고 group/groupId를 실제 ID의
enum으로 제한한다. 역할 category/categoryId도 현재 역할 분류 ID·라벨을 제공하고 enum으로 제한한다. 분류명 추측으로 저장 단계의 외래키 오류를 반복하지 않는다.
저장소에서만 검사되던 규칙 일부를 작성 schema로 끌어왔다: `years`는 `person-life-years.js`의 LIFE_YEARS를
그대로 옮긴 `pattern`(도구 schema 공통), 신규 인물은 familyName 또는 givenName 필수, `sortOrder`는
integer만(null은 저장소가 거절하므로 생략이 곧 append). 모델이 초안을 다 쓴 뒤 400으로 알게 되던 오류가
작성 호출 안의 schema 거절로 바뀐다.
목록 전체 교체와 부분 수정(aliases/aliasEdits, career/careerEdits, scenes/sceneEdits)은
동시에 제출할 수 없도록 초안 schema에서 검사하여 같은 작성 호출에서 고친다.
한영 본문 필드에는 schema 상한의 80%를 초안 목표로 제공한다(실제 검증 상한은 유지).
bio·moment·definition은 문자열이 아니라 **문장 배열**로 받는다(`sentence_schema`): 항목 수 상한은
`sentence_budget`과 같은 계산(ko·en 상한 ÷ 밀도 문장 비용), 항목당 길이 상한은 필드 상한의 60%
(1문장 필드는 상한 그대로). 실행기가 공백으로 이어 붙여 저장소에 보내며, 이어 붙인 길이가 상한을
넘으면 문장별 길이와 제거할 항목의 JSON pointer를 돌려준다. 모델은 글자 수를 세지 못하지만 문장은
셀 수 있고 index로 제거할 수 있다: 2026-09-13~19 길이 거절 1,510건 중 890건이 이 세 필드였고 대개
몇 글자씩 깎다 라운드를 소진했다. body·heading·epithet은 문자열 그대로다.
길이 초과 시 핵심 주장을 보존하며 부차적 절·문장을 줄이고 몇 글자씩 반복 제출하지 않도록 안내한다.
`DraftRepair`는 한 작성 호출 안에서 거절된 전체 초안을 보관하고 draft_id와 JSON pointer
repairs로 실패 필드만 교체할 수 있게 한다. 입력 단계의 길이 제한은 로컬 초안 보관까지
유예하며, 완성본은 기존 schema의 원래 길이·필드·분류 조건을 모두 다시 검사한다.
근거·revision은 여전히 실행기가 부착하며, 부분 수정을 이용한 대상·버전 변경은 금지한다.
`draft_id`는 선택 사항이다: 한 호출에는 거절된 초안이 하나뿐이므로 `repairs`만 오면(또는 이전 거절의 ID를
적어 보내도) 현재 초안에 적용한다. 대상·revision 변경 금지와 완성본 재검사는 그대로다(2026-09-19, 주 31건의 ID 불일치 거절 제거).
형식 검사를 통과한 초안도 호출 내에 보관한다. 근거 부착 후 기존 공통 서비스의 비저장 validate를
같은 작성 호출에서 실행하여 연도·역할·분류 등의 오류를 부분 수정으로 고친다. 오래된 draft ID는
적용하지 않고 현재 ID를 반환한다. 근거 누락·revision 충돌은 수정 루프를 끝내고 validate artifact를
거쳐 재조사한다. 근거 누락 시 거절된 초안도 조사 문맥에 보존한다. 신규 필수 사실 필드의 근거가
없으면 작성 모델 호출 전에 이 경로로 보낸다. 정상 초안은 기존 validate·독립 검토·저장 검사를 유지한다.
제공된 조사·현재 스냅샷으로 초안을 일찍 제출하여 길이·형식 오류를 수정할 라운드를 확보한다.
형식 오류는 기존 조사로 최대 두 번 수정한다. 필드별 근거 누락은 이전 주장·검증 오류를
조사기에 전달해 research로 돌리며, 같은 근거 실패가 세 번 누적되면 알림 없이 내부 보류한다.
인물의 `evidence must identify`와 용어의 `evidence required for <field>`를 모두 근거 누락으로
분류한다. 빈 출처 목록 오류도 재작성 대신 재조사로 돌린다. 용어 연도 필드의 근거가 없을 때 같은 초안만 반복 수정하지 않는다.
용어 초안이 현재 값과 같은 `startYear`/`endYear`/`period`를 그대로 되돌려 보내거나 create에서 null 연도를
채운 경우는 편집이 아니므로 검증 전에 키를 제거한다(`drop_unchanged_term_facts`) — 서비스는 키가 있으면 null에도
근거를 요구해 재조사가 끝나지 않았다(2026-09-18, #1470·#1707·#2049·#16869). 실제로 바뀐 연도는 여전히 주장이 필요하다.
재조사는 검증기가 지목한 누락 필드의 주장을 포함해야 ready로 완료된다.
이 재조사는 이전 research의 주장이 아직 살아 있는 소스에 얹혀 있으면 **표적 재조사**로 돈다: 이전 주장은
자동으로 이월되고(`carried_claims`/`merge_claims`), 프롬프트는 누락 필드만 지목하며, 라운드는
`TARGETED_RESEARCH_ROUNDS`(6)로 줄인다. attempt metrics의 `targeted_research`에 대상 필드가 남는다. 이전 소스가
만료됐으면 초안이 어차피 튕기므로 전체 재조사(12라운드)를 한다.
근거를 찾지 못하면 sources_unavailable로 보류하고 주장·인용을 만들지 않는다.
근거 누락과 revision 충돌 횟수는 문장 길이·형식 수정 횟수에 합산하지 않는다.
예산 부족·초안 모드 대기는 실패 재시도 횟수에 포함하지 않는다. 충돌은 새 조사 단계로 돌리며 토큰을 자동 대입하지 않는다.
명시적 gap의 발견 도구는 최대 한 후보와 요청된 kind·label·mention을 schema에 고정하고,
현재 지시도 같은 한 항목으로 제한한다. 일반 문서의 최대 네 후보 탐색과 혼용하지 않는다.
단계 결과를 저장하지 못하면 마지막 handler 거절 사유를 오류에 보존한다.
일일 LLM 예산 부족과 canary 반영 슬롯 소진은 각각의 실제 사유로 기록하며 실패 횟수를 늘리지 않는다.

`sources_unavailable`은 조사 결과를 보존하고 90일 연기한다. complete/not_applicable은
관련 내용이 바뀌지 않으면 180일 동안 재선정을 억제한다. 기존 인물·용어는 공통 저장 서비스에도
그 판단을 기록한다. 글을 늘릴 필요가 없다는 판단은 정상 결과다.

원문은 URL·내용 SHA-256·취득/만료 시각으로 저장한다. authoring 캐시는 도구 이름과
정확한 인자 hash가 같을 때만 공유하며 캐시 적중은 취득 시각을 갱신하지 않는다.
14일 뒤 본문을 지우되 식별자·URL·hash를 보존한다. 저장된 인용은 원문 전체 캐시와 별개다.
독립 검토는 이 캐시를 사용하지 않고 기존 검토 도구로 직접 원문을 가져온다.

모든 새 편집은 검증과 독립 검토를 통과해야 반영된다. 실제 제출과 승인은 별개의
idempotency key를 쓰며 frontend 트랜잭션에 영수증을 저장한다. 프로세스가 저장 직후
죽어도 같은 결과를 되찾는다. 독립 검토가 `revise`를 반환하면 검토 근거를 research와
draft 문맥에 전달한다. `needs_research=false`면 저장된 근거로 draft부터 수정하고,
새 출처가 필요한 경우 또는 이 필드가 없는 기존 결정은 research부터 재개한다.
검토 요청 총횟수로 보류하지 않는다. 근거·revision 메타데이터를 제외한 동일 초안이
두 번의 수정 기회 뒤에도 그대로 다시 거절되면 내부 보류한다. 내용이 바뀌면 이 횟수는
초기화되고 일일 예산·단계별 예약은 계속 적용된다. artifact의 `reviewed_content_hash`로 판정한다.
검토는 필수 사실 교정과 선택적 보강을 구분한다. 추가 날짜·배경이 없어도 현재 변경안이
정확하고 오해를 만들지 않으면 승인할 수 있다. 선택적 보강은 재작성 조건이 아니다.
2026-09-17 운영자 지시로 검토 기준을 완화했다. 핵심 사실이 확인되면 승인이 기본이며, revise는
독자를 오도하는 실질 오류(날짜·이름·직책·사건·분류 오류, 한영 사실 모순, 근거 없는 핵심 주장,
증거에 어긋난 fate)에 한한다. 표현·강조·유보 정도·출처 귀속 방식·일관된 표기 변형·사소한 표기 실수는
reason에 적고 승인한다. 승인 조건에서 인용 출처 전부 커버와 위키백과 밖 출처 요구를 제거했으며,
검증된 인용 check와 모든 risk 해소는 계속 요구한다(`validate_decision`).
미확정 사망 경위는 `fate.kind=""`와 한영 유보 라벨로 표현하며 구금을 망명으로 분류하지 않는다.
검토기는 매번 원문을 직접 가져오며 작성기의 근거 캐시를 승인 근거로 쓰지 않는다.
생몰연도 이설은 출처별 병기로 처리할 수 있으며 임의 확정이나 처형 분류를 강요하지 않는다.
해결되지 않은 초안은 `escalated`라는 기존 DB 상태의 내부 보류 artifact로만 남긴다.
새 pending 제안이나 사용자 검토 요청을 만들지 않는다. 기존 검토 타이머의 미처리 건도
자동 알림·재알림을 보내지 않는다. 이전에 만들어진 pending 제안의 수동 처리 결과는
호환성을 위해 다음 tick에서 작업 상태와 gap 완료 여부에 반영한다.
기존 검토기의 `revise`는 원 제안 ID별 수정 작업으로 이어진다. 수정 작업은 일반 보강
묶음에서 제외하고, 원 변경안과 피드백을 보존하며, 독립 승인 후에만 기존 제안을 대체한다.

## 저장 서비스와 호환성

frontend migration 179와 `commulingo-pipeline-service.js`가 필요하다. 공개 HTTP API는
추가하지 않는다. RPC는 read/validate/submit/review/enrichment이며 submit/review/enrichment에
idempotencyKey를 요구한다. validate는 SAVEPOINT 롤백으로 실제 저장 검증을 실행한다.
서비스가 없거나 실패하면 Python SQL로 우회하지 않는다.

용어 서비스는 field별 evidence, 전체 행·별칭·연결의 revision, 한영 부분 병합,
중복 별칭·분류·상위 용어 검증, 제안·승인·보강 상태를 소유한다. 기존 자료의 근거를
소급 생성하지 않는다. `term_editorial_service=true` 이후 기존 Python 용어 도구도 이 서비스로
연결된다. 이전 버전 없는 용어 제안은 기존 CLI 경로를 유지하며 새 자동 검토에 포함하지 않는다.
새 evidence가 있는 용어 제안은 기존 `/commulingo_review` 명령에서 처리한다.

## 예산과 운영 명령

`config/commulingo_pipeline.json`의 운영 일일 예산은 $10.00, 단계 예약은 $0.20,
검토 확보 비율은 30%다. UTC 호출 시작일 기준으로 PostgreSQL advisory lock 안에서
예약하고 응답 후 정산한다. 예약을 초과한 실제 비용은 overrun으로 보이며 후속 호출을 막는다.
응답 비용이 불명확한 실패는 예약액을 유지한다. 과거 날짜의 미정산분도 costs에서 확인할 수 있다.
예약만으로 외부 제공자의 단일 요청 비용 초과까지 차단하지는 못한다.

`legacy_shared_budget=true`이면 기존 사람·gap·사건의 공용 실행기, 검토기, 연결 생성기도
같은 예산을 예약한다. 배포 전 운영 장애를 막기 위해 기본값은 false이며 새 유료 실행 명령도
이 값이 false이면 거부한다. 기존 ledger 비용과 이 테이블 비용을 중복 합산하지 않는다.

```bash
venv/bin/python scripts/commulingo_pipeline.py plan
venv/bin/python scripts/commulingo_pipeline.py plan --apply
venv/bin/python scripts/commulingo_pipeline.py consolidate
venv/bin/python scripts/commulingo_pipeline.py consolidate --apply
venv/bin/python scripts/commulingo_pipeline.py list
venv/bin/python scripts/commulingo_pipeline.py show 123
venv/bin/python scripts/commulingo_pipeline.py costs
venv/bin/python scripts/commulingo_pipeline.py metrics
venv/bin/python scripts/commulingo_pipeline.py run --limit 1
venv/bin/python scripts/commulingo_pipeline.py run --review --limit 4
venv/bin/python scripts/commulingo_pipeline.py retry 123
venv/bin/python scripts/commulingo_pipeline.py run --job-id 123 --publish --limit 4
```

run은 기본적으로 초안까지만 진행한다. `run --review`는 독립 검토까지 허용하되
승인 결과도 submit 앞에서 멈춘다. 판단 불가 초안은 알림 없이 내부 보류된다. --publish는 phase=canary/live에서만 가능하다.
`--job-id`는 지정한 작업만 정상 lease·예산·검토 제한 아래 재개한다.
run 기본 한도는 6단계, tick은 12단계다. `--limit`로 조정하며 기본 시간 한도는
`--max-seconds 1800`이다. 성공한 다음 단계는 같은 작업 번호로 즉시 재개하고,
완료 후에만 순환 스케줄러로 돌아간다. 지정한 --job-id는 그 작업 종료 시 멈춘다.
단계마다 lease·artifact·예산 예약과 정산을 유지한다. 작성 예산이 부족하면 같은 배치에서 검토와 비유료 validate/judge/submit만 계속 처리한다.
검토 예산까지 부족하면 비유료 단계만 처리한다. 명시적 --job-id, 초안 경계·lease 상실이면
배치를 멈추며 오류·보류는 기존 재시도 시각을 존중한다. 남은 시간이 한 단계의 480초보다
짧으면 새 단계를 시작하지 않는다. service TimeoutStartSec은 정리 여유를 포함해 1950초다.
tick은 현재 예산으로 재개 가능한 `daily budget reserved or spent` 대기만 다시 열고,
완료 검토 정리·미착수 작업 통합·변경분 계획·원문 만료·연속 실행을 수행하며 phase를 따른다.
재개 시점의 검사는 예약이 아니며 실제 호출 전 원장 잠금 아래 다시 예약한다. 오류·근거 부족 보류는 재개하지 않는다.
운영 live에서는 반영 건수 제한이 없다. tick은 과거 canary 한도 소진만을 사유로
보류된 submit 작업을 즉시 다시 열고, 예산·오류에 의한 대기는 그대로 유지한다.
아래 canary 설정은 수동 롤백용으로 남긴다.
canary는 네 작업군별 UTC 하루 한 작업의 반영만 허용한다. 실패·장애 후 같은 작업은
원래 반영 슬롯을 재사용한다. 실패한 작업의 슬롯을 자동 회수하지 않는다.
metrics는 상태·단계별 형식 오류·기록된 비용·캐시 적중, costs는 실제 비용·미정산 예약·초과를 보여준다.

## 검증과 전환

단위/통합 검사: `tests/test_commulingo_pipeline.py`.
`COMMULINGO_PIPELINE_TEST_PORT`를 설정한 경우에만 localhost의
`commulingo_integrity_test` DB를 사용한다. lease 경쟁, 오래된 소유자의 저장, 동시 예산 예약,
영수증 키, 근거 범위·만료, 작성 revision 바인딩, 문서 hash, 발견 결과의 원자성, canary 제한을 검사한다.
frontend의 `test-commulingo-pipeline-db.js`는 별도 DB 가드 아래 실제 저장·검토·롤백을 검사한다.

전환 순서:

1. 기존 작업을 종료시키고 작성·검토 타이머를 정지한다. frontend 179와 Python migration을 적용하고
   검증한 frontend RPC 파일을 배포한다. 기존 자료·제안·검토 결과는 보존한다.
2. 새 UTC 날짜부터 shared budget을 활성화한다. 이전 실행 비용은 자동 소급 수집하지 않으므로
   같은 날 전환하려면 모든 기존 레인의 유휴 상태를 확인하고, UTC 자정부터 중단 시점까지
   journal의 결과 비용과 중복 없는 loop 감사 비용 중 큰 값을 빈 원장에 한 번만 이관한다.
   전환 후 journal을 다시 가져오면 이중 집계되므로 재이관하지 않는다. 사건·연결·기존 검토도 새 예산에 참여시킨다.
3. term_editorial_service를 활성화하고 phase=draft로 고정 사례를 평가한다. 네 작업군의 동일
   입력·출처 조건을 고정하고 정확성·근거 적합성·중요 누락 해소·한영 일치를 기존 결과와 비교한다.
   최초 형식 통과 95%는 목표이며 단위 테스트 통과로 달성했다고 간주하지 않는다.
4. 평가가 기존 이상이면 canary로 바꾸고 새 service/timer를 설치한다. 인물·용어·gap의 기존
   작성 타이머는 중복 실행하지 않는다. 사건·연결과 과거 제안 검토는 유지한다.
5. 중복 반영·revision 우회·예산 누락이 없고 품질 조건을 충족하면 live로 전환한다.

롤백은 새 타이머를 멈추고 진행 작업을 종료한 뒤 기존 작성기를 복구한다. 새 스키마와
artifact를 삭제하지 않는다. 아직 frontend/Python 코드를 되돌리지 않았다면 용어의 새 저장·검토
경계는 유지한다. 새 스키마가 필요한 서비스를 이전 파일과 섞어 실행하지 않는다.

건강 보고서 `scripts/commulingo_lane_health.py`는 과거 레인 비용을 보존하면서 새 파이프라인의
반영·실행·재시도·보류와 UTC 당일 공용 비용·예약을 조회한다. 신규 작업 비용만 기존 journal
합계에 더하며 공용 원장의 기존 레인 비용을 중복 합산하지 않는다. 중단된 인물 작성 레인의
무실행은 장애로 보지 않는다.
보고서는 기간 내 반영 항목의 이름·작업 번호·신규/보강·주제를 먼저 표시하고,
현재 대기열(기간 내 발생 건수 아님)의 단계·사유·확인할 작업 번호를 구분한다.
도구 비성공 비율은 원인별 횟수와 실행 범위 수를 동반하며, 그 숫자만으로 유료 재시도 비용을 추정하지 않는다.
비용은 기록된 LLM 비용으로 명시하고 공용 원장의 레인별 비용·미정산 예약을 별도 분해한다.
유료 검색 비용은 이 집계에 포함하지 않는다. `--since`는 `-Nh` 또는 UTC `today`이며,
bio 길이 분포는 별도로 표시한 7일 창이다. 중단된 작성 레인의 0행은 숨긴다.
긴 정기 알림은 UTF-16 길이를 기준으로 분할해 전송하며 내용을 자르지 않는다.

2026-09-09 전환 시 기존 당일 비용 $0.071381을 유휴 상태에서 한 번 이관했다.
초기 네 사례의 첫 형식 통과는 3/4였으며 95% 목표 달성이나 기존 대비 품질 우위는 아직
입증하지 않았다. 이 관찰은 품질 우위의 증거로 해석하지 않는다. 현재는 운영자 요청에 따라 live로 전환했으며
독립 검토·revision 충돌 검사·공용 비용 예산을 유지하고 반영 건수 제한만 해제했다.

DeepSeek 모델 선택은 큐레이터·이벤트 큐레이터·리뷰어의 스펙과 runtime overlay에서 `deepseek_flash`로 통일한다. 공통 provider registry가 정식 `deepseek-flash`로 해석하며, 타이머 작업은 다음 프로세스 시작부터 읽는다. 옛 `deepseek_pro`는 저장값 호환용으로만 유지한다.

## 단계 호출의 낭비 제어 (2026-09-17)

2026-09-16~17 24시간 창의 원장($8.97, 반영 77건)에서 확인한 낭비와 조치다.
독립 검토·조사 성공의 절반 이상이 12라운드 한도의 강제 마감 호출에서 끝났는데,
강제 마감이 도구 목록을 마감 도구만으로 줄여 보내 캐시 접두사가 깨졌고(캐시 읽기
~5k 대 일반 라운드 ~55k 토큰), 그 뒤 본문 정리용 후속 호출까지 더해 강제 마감 두 호출이
하루 비용의 약 21%였다. 마감 도구 호출이 한 번 거절되면 시도 전체가 실패했고(검토 33건,
$1.40), 초안 단계는 길이 초과를 한 문장씩 깎으며 12라운드를 소진했다(한 작업 3회 시도 $0.21).
재검토는 이전 검토 판정을 받지 못해 매번 새 지적으로 revise를 반복했다(한 인물 10회 검토).

공통 루프(`llm/agent_loop.py`, `llm/claude_loop.py`)의 조치:
- 강제 마감 호출은 도구 목록을 그대로 보내 캐시 접두사를 유지한다. 허용 목록은
  `parse_final`이 강제하며, 허용 밖 호출은 오류 tool_result를 받는다.
- 마감 도구 호출이 모두 거절되면 `FINALIZATION_RETRIES`(2)회까지 오류를 붙여 다시 호출한다.
- 강제 마감에서 terminal tool이 성공하면 일반 라운드처럼 그 결과로 끝나며 후속 텍스트 호출을 하지 않는다.
  terminal tool을 선언한 호출자는 실패 시에도 후속 호출을 받지 않는다.
- `terminal_required=True`(파이프라인 단계 전용)면 텍스트로만 끝난 첫 응답에 한 번 terminal tool 호출을 상기시킨다.
  기존 큐레이터 레인은 텍스트 종료가 정당할 수 있어 기본값 false다.

파이프라인 조치:
- 초안 단계는 `DRAFT_ROUNDS`(8)로 제한하고, 길이 초과 거절에 필드별 문단 길이와 삭감량을 붙인다
  (`DraftRepair.length_guidance`). 같은 필드의 세 번째 초과부터는 횟수를 명시한다.
- 재검토는 같은 묶음의 이전 revise 판정(reason·findings, 인용문 제외)을 `previous_reviews`로 받고,
  요구한 수정이 반영됐는지 먼저 확인하도록 지시한다. 새 revise는 사실 오류·근거 없는 단정에 한한다.
  검토 횟수로 보류하지 않는 원칙은 그대로다.
- 검토 결정의 source_id·행 범위 거절은 사용 가능한 review source ID와 행 수를 함께 돌려준다.

효과는 적용 후 24시간 창의 `llm_audit_log`에서 `Forced-final`·`Forced-final followup` 라벨의
cache_read와 비용, 시도 테이블의 검토·조사 error 비율, 작업당 검토 횟수로 확인한다.

## 런타임 문맥의 출처

단계별 명령은 현재 요청으로 전달하고, job·기존 문서·조사 근거·초안·검토 자료는
출처가 표시된 JSON 문맥으로 감싼다. 기존 source ID/hash/취득·만료 시각, revision,
검증 오류를 보존하며 문자열로 된 가짜 지시가 경계를 닫지 못하도록 구분자를
이스케이프한다. 현재 단계·terminal tool과 단계 결과 미저장 상태는 별도 런타임
메타데이터다. 단계 완료는 발행이 아니며 submit/review 영수증으로 실제 반영을
구분한다. 기존 독립 검토·저장·재시도·비용 정책은 그대로이며 추가 LLM 호출은 없다.

조사 단계는 제공된 근거를 먼저 재사용하고 위임된 주장에 충분한 근거가 있으면 artifact를 반환한다. 추가 검색은 누락·상충·변경 가능성 검증에 한한다. 유료 검색과 Extract는 LLM 예산과 별도로 [공용 web 일일 예산](web_research.md)을 사용한다.

조사 결과의 `claims.field`는 해당 대상·action의 실제 쓰기 schema 필드만 허용한다.
주제명 history/distinctions/examples를 필드로 제출하면 같은 조사 호출 안에서 수정하도록
거절한다. 용어의 역사·구별·사례 근거는 `body`에 연결해야 하며 `definition` 근거로
자동 대체하거나 조사기가 고른 근거의 필드명을 실행기가 임의 변환하지 않는다.

## 시도 계측과 효율화 검증

`commulingo_pipeline_attempts`는 모델 결과가 없는 실패도 기록한다. 단계 시작·종료, 벽시계 실행 시간,
결과 상태·다음 단계·오류·호출 횟수·반복 라운드·초안 검증 횟수를 보존하고 `budget_id`로 공용 원장에 연결한다.
실행 도중 프로세스가 죽으면 미종료 시도로 남는다. 비용 불명 예약은 기존처럼 유지한다.
비용은 시도 metrics를 합산하지 않고 공용 예산 원장만 사용한다.
새 코드를 실행하기 전에 기존 명시적 `commulingo-pipeline` migration으로 시도 테이블을 추가해야 한다.
import/조회는 자동 DDL을 실행하지 않는다.

`venv/bin/python scripts/commulingo_pipeline.py efficiency --since today`는 네 작업군의 기간 내 반영·비용,
미정산 예약, 계측된 실행 시간·미종료 시도·재조사 전환과 첫 저장 검증 결과를 JSON으로 반환한다.
lane health에도 같은 지표를 표시한다. 반영당 비용은 같은 기간의 실패·발견 작업 비용을 포함한 운영 비율이며,
개별 성공 작업의 원가로 해석하지 않는다. 시간당 생산율의 분자는 시도 기록이 있는 반영만 포함한다.
첫 검증 지표는 신규 계측 이후 각 작업의 첫 작성 시도를 대상으로 하며 과거 실패를 소급해서 0으로 보지 않는다.
사건·연결·기존 검토 레인 비용은 기존 공용 원장 분해를 유지한다. 유료 검색 비용은 별도다.

`python scripts/commulingo_efficiency_replay.py FIXTURES.json --baseline-ref ba4e9c8 --output REPORT.json`은
네 작업군별 5건의 저장된 초안을 같은 입력으로 재현한다. 입력은 job_id/kind/action/draft를 가진 20개 JSON 행이다.
모델 호출·공개 저장 없이 schema 호환성과 저장 검증 거절 후 수정 가능한 초안 보존을 비교한다.
이 검사는 모델 품질, 최초 생성 성공률 또는 실제 비용 절감을 입증하지 않는다.
운영 확대 기준은 별도 동일 입력 모델 평가에서 품질 회귀 없음·첫 저장 검증 90% 이상·승인 가능한 사례당
LLM 비용 20% 이상 감소다. 이 기준을 충족하기 전에는 효율 개선을 달성했다고 보고하거나 운영 작업자를 늘리지 않는다.
운영자 지시로 효율화 코드는 main에 병합하고 시도 계측 migration과 함께 운영에 적용한다.
별도 모델 A/B 평가의 품질·비용 기준은 아직 입증하지 않았으며, 이를 배포 승인 조건으로 다시 묻지 않는다.
실제 효과는 적용 후 24시간 창의 생산율·검증 실패·미정산 비용으로 확인한다.
