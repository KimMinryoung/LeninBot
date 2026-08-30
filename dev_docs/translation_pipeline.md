# Translation Pipeline

두 개의 독립된 파이프라인이 있다. 공유하는 것은 LLM 호출 경로(`llm/call_registry.py` → gateway/proxy)와 `scripts/_translation_common.py`의 헬퍼뿐이다.

| | A. 사료(archival) 파이프라인 | B. 사이트 콘텐츠 파이프라인 |
|---|---|---|
| 방향 | RU→KO, ZH→KO | KO→EN |
| 엔진 | `runtime_tools/archival_translation/` | `scripts/translate_*.py` 4종 |
| 단위 | HTML 블록, `[[번호\|태그]]` 마커, 청크(기본 3,500자) | 문서/행 통짜 1회 호출 |
| 용어집 | CommuLingo 스냅샷 + 스펙 `glossary.extra`, 청크 등장 항목만 주입(상한 60) | 프롬프트 내장 소사전 |
| 검증 | 결정론 5종 + 위반 항목 첨부 교정 재시도 | 결정론 검사 + 교정 재시도 1회 |
| 캐시/이어하기 | 내용 해시 JSONL, 청크 단위 | 없음(스킵 조건이 재개 역할) |
| 진입점 | `scripts/translate_archival_documents.py`, `api_routes/archival_translation.py` | systemd `research-document-translation.timer` + 수동 |

설계 원칙은 `~/uploads`로 전달된 번역기 인수인계 문서(2026-08-30)를 따른다: LLM은 청크 번역기로만 쓰고, 용어 일관성·연속성·검증은 결정론적 레이어가 책임진다. 모델 호출은 전부 `call_registry`의 feature 단위 설정(`config/llm_call_sites.json`)을 지난다.

## 사료 파이프라인 실행 (2026-08-30 정리)

- **실행**: `venv/bin/python scripts/translate_archival_documents.py --spec <id>`. 프로바이더 호출은 LLM 게이트웨이 프록시(`:8110`)를 지나고 키는 프록시가 주입하므로 **credstore·sudo·systemd-run이 필요 없다** (예전 `run-archival-translation.sh` 래퍼와 `LoadCredentialEncrypted` 안내는 삭제됨). `--plan`은 모델을 부르지 않고 슬라이싱·청킹·견적만 낸다. 같은 실행은 `POST /admin/archival-translation/run`으로도 된다.
- **모델·max_tokens·thinking은 registry가 결정한다**: `config/llm_call_sites.json`의 `archival_document_translation_ru`/`_zh` 항목(또는 `LLM_SITE_ARCHIVAL_DOCUMENT_TRANSLATION_{RU,ZH}_MODEL`). `call_registry.resolve()`는 항목 값을 호출부 기본값보다 우선하므로 CLI·API·`Options`에는 model 옵션이 **없다** — 예전 `--model`/`--max-tokens`는 조용히 무시되던 죽은 옵션이라 제거했다. 후보 모델은 `--compare provider/model[,+think,+effort=high]`로 같은 청크를 나란히 뽑아 본 뒤 registry 항목을 고친다. `--compare`의 preflight는 변형에 적힌 provider마다 검사한다.
- **청크당 준비물은 한 번만 만든다**: `_prepare_chunk()`가 (프롬프트, 주입 용어 목록, 캐시 키)를 돌려주고, `run()`의 pending 집계와 워커가 같은 것을 쓴다. 사후 용어 검사가 프롬프트에 주입된 목록과 같은 목록을 보는 것도 여기서 보장된다. `Stats` 카운터 갱신은 `Stats.add()`로 락 뒤에서 한다(워커 5개 공유).
- **캐시 키** = `PROMPT_VERSION` + resolve된 provider·model·thinking + 시스템 프롬프트 해시 + 유저 프롬프트 전문. 프롬프트·용어표·tmExamples·register 변경은 자동으로 키를 바꾼다. `PROMPT_VERSION`은 파서·검증기(`parse_response`/`validate`)가 바뀌어 옛 캐시의 판정을 믿을 수 없을 때만 올린다.

## 번역 메모리 (TM)

`runtime_tools/translation_memory.py`. SQLite(`output/translation_memory.sqlite3`)의 단일 테이블:

```
segments(id, lang_pair, source, target, doc_id, block_id,
         status 'machine'|'reviewed', provider, model, created_at)
UNIQUE(lang_pair, doc_id, source, target)
```

- **상태 3단계** (조회 우선순위 machine < published < reviewed): `machine`은 파이프라인 원출력, `published`는 frozen(발행) 스펙에서 온 쌍 — 문서 통독 검수는 거쳤지만 세그먼트 개별 확인은 아니라는 사실을 그대로 남긴 중간 상태, `reviewed`는 그 쌍 자체를 사람이 확인한 것. 같은 쌍이 더 높은 상태로 다시 오면 승격되고 내려가지는 않는다.
- **적재 경로**: 사료 파이프라인 `run()`이 성공한 청크의 블록 쌍을 자동 적재한다(`_record_translation_memory`, 실패는 `tmFailed` 이벤트로만 남고 번역을 깨지 않는다). `translate_db_content.py`는 행 갱신 성공 시 짧은 필드(title 등) 쌍을 적재한다.
- **postEdits 반영**: 캐시에는 모델 원출력이 남지만, 사람이 스펙 `postEdits`로 고친 오역 교정은 적재 전에 같은 치환으로 반영된다(`apply_post_edits`). 따옴표 정규화와 병기 축약은 문서 위치에 묶인 처리라 세그먼트에는 적용하지 않는다.
- **백필**: `python scripts/backfill_translation_memory.py` — 기존 청크 캐시(JSONL)를 스펙 재계획으로 원문과 재정렬해 적재. frozen 스펙은 `published`로 들어간다. API 호출·자격증명 불필요. `--stats`로 집계 확인.
- **조회**: `exact_matches(sources, lang_pair=..., statuses=...)` — 상태 우선순위대로 이기고, 같은 상태면 새 행이 이긴다.
- **완전 일치 재사용**: 사료 `run()`이 시작 전에 검수 등급(published/reviewed) 세그먼트와 완전 일치하는 블록을 모델 없이 채운다(`_tm_prefill`, `tmReuse` 이벤트). machine은 자동 재사용하지 않는다 — 미검수 출력의 자기 강화를 막기 위해서다. 청크 경계는 유지하고 전 블록이 덮인 청크만 건너뛴다: 경계를 다시 자르면 기존 청크 캐시가 무효가 되어 재사용이 비용을 늘리기 때문이다. 유사 세그먼트 예시 주입은 남은 후속이다.

## 검증 레이어

- 사료: `validate()`(마커 누락/원문 반환/한국어 부재/원문 문자 잔존/길이 하한/용어표 준수) + 문서 전체 `stray_cyrillic()`. 실패 사유를 교정 메시지로 붙여 재시도. 용어표 검사는 그 청크에 실제로 주입된 항목만 대상으로 하며(모델은 못 본 표기를 지킬 수 없다), **오탐 방어가 두 겹이다**: 재시도를 소진하고도 용어표 문제만 남으면 실패 대신 경고로 낮춰 통과시키고(`termWarnings` 이벤트·stats, 판단은 발행 전 통독과 postEdits 몫), 다의어라 상습 오탐인 항목은 스펙의 `glossary.noEnforce` 목록에 올려 주입은 하되 강제하지 않는다. 형식·누락·미번역 검사는 오탐이 없으므로 여전히 실패로 처리한다.
- 사이트 공통(`scripts/_translation_common.py`): `field_translation_problems()` — 한글 잔존율(원문이 한국어인 긴 필드), 원문 그대로 반환(짧은 필드), HTML 태그 열 보존, URL 보존. `translate_db_content.py`와 스모크가 사용.
- research markdown: 제목 깊이 열·한글 잔존율·내부 보고서 링크 보존(`_validate_translation`). `translate_markdown_with_retry()`가 검증 실패 사유를 시스템 프롬프트에 붙여 1회 재번역한다(reflection 1회, 반복 자기수정 없음).
- 오프라인 스모크: `scripts/smoke_translation_memory.py`(TM + 공용 검증기, 맨 클론에서 실행 가능), `scripts/smoke_archival_translation.py`(frontend 체크아웃 필요).

## 스타일 자산

- 번역투 금지 목록(writer/prompts.py에서 이식): ~것이다 반복, ~의 사슬, 되어지다 이중 피동, 그/그녀 남용, 한자어+하다 편중 — 사료 시스템 프롬프트(RU·ZH) 양쪽에 있다. 시스템 프롬프트 해시와 유저 프롬프트 전문이 청크 캐시 키에 포함되므로 프롬프트 수정은 캐시를 자동 무효화한다(frozen 스펙은 애초에 재실행이 거부된다). `PROMPT_VERSION`은 파서·검증기 변경 전용이다.
- 문장부호 규칙은 한 곳에 있지 않다: ZH 프롬프트(《》·「」·굽은 따옴표·着重号), 조립기의 `quotes:"curly"` 정규화, 스펙별 `register` 문자열, em-dash 정책(`commulingo_strip_em_dashes.py`).

## 인수인계 체크리스트 대조 (2026-08-30 기준)

✅ 충족 / ⚠️ 부분 / ❌ 미충족. 사료 파이프라인 기준이며 사이트 파이프라인은 괄호로 표시.

| 항목 | 상태 | 비고 |
|---|---|---|
| §2.1 청크 크기 1.5k~3k토큰 | ⚠️ | 문자 기준 3,500자(RU ≈ 1.6k토큰)로 범위 안. 토큰 환산은 `chars_per_token` 추정 (사이트: ❌ 통짜, 60k자 캡) |
| §2.1 표·각주 경계 보존 | ✅ | 블록 단위 청킹이라 중간 절단이 없고, 표 숫자는 모델을 거치지 않는다 |
| §2.1 직전 청크 맥락 주입 | ❌ | 의도적 설계: 청크 독립 + 병렬, 문서 간 규칙(주석 병기 1회, 따옴표)은 조립기가 집행. 도입하려면 병렬성 포기 필요 — 테스트셋(§5-5) 이후 판단 |
| §2.1 청크 ID·재조립·부분 재시도 | ✅ | 마커 + 내용 해시 캐시, 실패분만 재호출 |
| §2.1 한국어 출력 팽창 | ✅ | `SourceLanguage.output_ratio` (RU 0.9 / ZH 1.8) |
| §2.2 코퍼스 단위 용어집 | ⚠️ | CommuLingo DB(상태·출처·리비전 있음)가 코퍼스 용어집, 파이프라인은 스냅샷을 읽기 전용 소비. 스냅샷 갱신은 수동 단계 |
| §2.2 사전 스캔(PREPARE) | ⚠️ | RU: `scripts/scan_archival_terms.py` — 약어·문장 중간 대문자 낱말을 용어집과 대조해 미등재 후보만 보고, 채택·표기는 사람이 결정. ZH: 대소문자 신호가 없어 NER 도입 전까지 수동 |
| §2.2 청크 등장 항목만 주입 | ✅ | `glossary_entries_for()` + 상한 60, 주입 목록과 사후 검사 목록이 동일(`_prepare_chunk`) |
| §2.2 표기 변형 매칭 | ✅ | 러시아어 곡용 변형 + 경계 가드, 중국어 무경계 |
| §2.2 다의어 항목 차단 | ✅ | `glossary.exclude` — 이 문서 문맥에서 거의 항상 다른 뜻인 항목은 **주입 자체를 뺀다**. `noEnforce`는 검사만 끄고 프롬프트엔 여전히 "반드시 쓸 것"으로 들어가, 1925 대회 번역에서 Союз(의원 그룹)→Советский Союз, Октябрьский(인명)→10월 혁명, два лагеря(두 진영론)→두 진영에 씌워진 채 발행됐다(postEdits로 교정). 스냅샷의 성(姓)만 딴 인물 항목과 보통명사가 겹치는 표제어가 상습 원인. exclude는 주입 목록을 바꾸므로 캐시 키가 바뀐다 |
| §2.3 TM 정렬 쌍 저장 | ✅ | 이번 변경. v1은 적재 우선 |
| §2.3 완전 일치 재사용 | ✅ | 검수 등급 한정, 청크 경계 보존(`_tm_prefill`) |
| §2.3 유사 세그먼트 예시 주입 | ✅ | 스펙 고정 방식: `scripts/suggest_tm_examples.py`가 검수 세그먼트를 어휘 겹침으로 추천 → 사람이 스펙 `tmExamples`에 채택 → 청크 프롬프트의 «참고 번역례»로 주입. 동적 주입은 캐시 키를 흔들어 의도적으로 배제 |
| §2.4 번역투 금지 목록 이식 | ✅ | 이번 변경 (RU·ZH 프롬프트) |
| §2.4 스타일 규칙 캐시 위치 | ✅ | 시스템 프롬프트 고정 + 캐시 키 포함 |
| §2.5 문장 수/길이 급감 | ⚠️ | 문장 수 대신 실측 기반 길이 하한(RU 0.25 / ZH 0.7) |
| §2.5 용어집 준수 사후 검사 | ✅ | 주입된 항목이 번역문에 실제로 쓰였는지 확인, 위반은 교정 재시도로 피드백 |
| §2.5 숫자·마크업 보존 | ⚠️ | 표 숫자는 코드 보존, 사이트는 태그 열·URL 검사(이번 추가). 본문 숫자 대조는 없음 |
| §2.5 미번역 잔존 검사 | ✅ | 블록 + 문서 전체(키릴·한자), 사이트는 한글 잔존율 |
| §2.5 위반 항목만 명시 재번역 | ✅ | 사료 원래 있음; research·db_content는 이번 추가 |
| §2.6 위치 지정 편집 정제 | — | 정제 단계 자체가 없다(P5의 회귀 위험이 없는 상태). 필요해지면 diff 반환으로 설계 |
| §2.7 단일 어댑터·언어쌍 설정 | ✅ | `call_registry` + feature 단위 JSON, 핫 리로드. 사료는 `archival_document_translation_ru`/`_zh`로 분리되어 언어쌍별 provider·model 교체 가능(`SourceLanguage.feature`). 교체 전 `--compare`로 검증. CLI·API에는 model 옵션이 없다(registry가 이김) |
| §2.7 Batch API | ❌ | 미지원 — 남은 로드맵(야간 타이머 작업이 후보) |
| §2.7 토큰·비용 기록 | ✅ | `record_llm_call` 감사 + `plan()` 사전 견적 |
| §2.8 고정 테스트셋·자동 지표 | ❌ | 없음 — 남은 로드맵 §5-5 (모델 교체 재평가의 전제) |

## 남은 로드맵 (우선순위 순)

1. ZH 사전 스캔 — RU 스캐너는 있으나 중국어는 NER(GLiNER류) 없이는 소음이라 보류.
2. 언어쌍별 고정 테스트셋 + 청크 크기 실험(1k/3k/8k/통짜) — 모델 라우팅 재평가의 전제.
3. Batch API 경로(야간 타이머 작업 50% 할인).
4. 사이트 파이프라인 청킹 — 60k자 캡을 넘는 문서가 생기면 사료 엔진의 마커 방식 재사용.
