# Translation Pipeline

확인 기준: 2026-09-07 코드와 같은 날 수행한 운영 DB 마이그레이션 검증.

사료 번역과 사이트 영어 번역은 `translation_runtime/`의 실행 함수를 공유하고, 언어·형식별 프롬프트와 검증·조립은 각 어댑터가 맡는다. 이 문서는 현재 구현·운영 경계를 설명한다. 과거 모델 비교와 배치 산출물은 `output/archival_translations/compare-*.md` 등에 있으며, 현재 설정은 `config/llm_call_sites.json`이 기준이다.

## 1. 구성과 소유권

| 대상 | 언어·단위 | 어댑터와 출력 |
|---|---|---|
| 사료 | RU/ZH/EN/DE/FR/IT→KO, HTML 블록·마커, 기본 3,500자 청크 | `runtime_tools/archival_translation/`; 스펙의 `output` HTML fragment |
| 연구문서 DB | KO→EN, Markdown 구조 단위 청크(목표 8,000자) | `scripts/translate_research_documents.py` → Markdown 어댑터; `research_documents` 영어 열 |
| 연구문서 파일 | KO→EN, 같은 Markdown 어댑터 | `scripts/translate_research_markdown.py`; 기본 `research/en/*.md` |
| 기타 DB 콘텐츠 | KO→EN, 행 단위 JSON | `scripts/translate_db_content.py`; posts/ai_diary/hub_curations 영어 열 |
| 정적 페이지 | KO→EN, HTML 또는 문단별 텍스트 노드 | `scripts/static_page_translation_pipeline.py`; 페이지 JSON 영어 필드 |

공통 소유권:

- `translation_runtime/__init__.py`: `generate_translation`, `translate_validated`, `validate_cached`, 번역 오류 타입. 사료 `_translate_chunk`, Markdown `_translate_segment`, DB 콘텐츠 `_call_translator`가 같은 검증·교정 루프를 사용한다.
- `translation_runtime/structure.py`: Markdown 구조·보호 구간, HTML 구조·속성 검사, 숫자·조건 표현 검토 힌트, 청킹·산문 분할.
- `translation_runtime/storage.py`: 원문 해시와 원자적 파일 교체. 기존 파일 권한을 유지하며 새 파일은 0644로 만든다.
- `scripts/_translation_common.py`: 사이트 필드 검증·JSON 파싱과 기존 호출부용 호환 import. 별도 번역 실행 루프를 소유하지 않는다.
- `llm/call_registry.py`: provider 설정 해석, 상세 생성 결과, 정책·사용량 감사. 표준 번역 호출은 registry → LLM gateway → 키 주입 프록시를 지난다.
- 정적 페이지 DeepL 경로는 별도 HTTP 어댑터다. HTML 검증은 공유하지만 LLM registry와 공통 생성 재시도 루프를 사용하지 않는다. 사료 용어 추출·감사도 `terms.py`의 별도 보고 작업이다.

### 호출과 재시도

`translate_validated`는 유효한 캐시 재사용 → 생성 → 파싱 → 검증 → 실패 사유를 첨부한 재번역을 수행한다. 파서·검증기와 교정 지시문 렌더러는 콜백으로 전달한다(사료는 한국어 지시문, 사이트는 플레이스홀더 보존을 포함한 영어 기본문). 검증에 실패한 결과는 성공 캐시에 넣지 않는다. 사료는 기본 총 3회, Markdown·DB JSON은 총 2회 번역 시도를 허용한다.

`generate_translation`은 `generate_detailed`의 결과를 보고 일시적 요청 제한·통신·서버 오류만 최대 3회 호출한다. Retry-After 또는 지수 대기와 지터를 사용하며 대기 상한은 60초다. HTTP 429는 메시지에 `insufficient_quota`가 없는 한 항상 일시적 `rate_limit`이다. Gemini는 분당 제한에도 "Quota exceeded ... billing" 문구를 쓰므로 문자열로 영구 오류를 판정하지 않는다. 인증·정책·할당량 소진(402/456, 잔액 문구)·설정 오류, 빈 응답, 출력 예산 소진은 교정 재시도로 숨기지 않는다. 잘린 텍스트도 번역으로 수락하지 않는다. 이 호출 재시도와 형식 교정 시도, executor 내부 출력 예산 확대는 서로 다른 단계다.

사료는 영구 오류가 나면 실행 중단 플래그로 대기 워커와 통신 재시도의 새 호출을 막는다. 이미 진행 중인 요청은 완료될 수 있다. 연구 DB·기타 DB 배치도 영구 provider 오류에서 멈추며, 성공한 청크·행은 보존한다. 파일 CLI는 대상 파일별 실패를 모아 다음 파일을 처리한다.

`GenerationResult`는 `text`, `error_kind`, `error`, `truncated`, `usage`, `attempts`, `latency_ms`, `retry_after`를 반환한다. 기존 `generate_sync()`의 text/None 인터페이스는 유지한다. 출력 예산 확대 과정에서 버린 응답도 각각 감사하며 상세 `usage`는 합계다. 사료 `Stats`에는 `providerCalls`, `tokensIn`, `tokensOut`이 있고 `usage` 이벤트에는 사용량·지연·오류 종류가 나온다. `compare()`도 이 감사 경로를 사용한다. `--probe`는 진단용 executor 직접 호출이므로 같은 상세 감사 계측을 보장하지 않는다.

## 2. 모델과 실행 진입점

아래는 registry 파일의 기본값이다. model 환경변수 오버라이드와 provider의 출력 예산 확대가 적용될 수 있으므로, 실행 시에는 `call_registry.resolve(feature)`로 확인한다.

| Feature | Provider / model | max_tokens | timeout 설정 | Thinking |
|---|---|---:|---:|---|
| `archival_document_translation_{ru,en,de,fr,it}` | gemini / gemini-3.1-pro-preview | 48,000 | 600초 | 별도 설정 없음 |
| `archival_document_translation_zh` | deepseek_anthropic / deepseek-v4-pro | 48,000 | 600초 | enabled |
| `research_markdown_translation` | deepseek / deepseek-v4-flash | 20,000 | 240초 | enabled |
| `db_content_translation` | deepseek / deepseek-v4-flash | 12,000 | 180초 | disabled, JSON mode |
| `archival_term_extraction` | deepseek / deepseek-v4-flash | 8,000 | 180초 | disabled, JSON mode |

사료 CLI·API·`Options`에는 model/max_tokens 옵션이 없다. 모델 비교는 `--compare`로 명시적으로 수행하고, 채택할 설정은 registry에 반영한다. 현재 모델 선택은 유지한 상태이며, 공통화·검증 개선을 이유로 자동 교체하지 않는다.

### 사료 CLI·API

```bash
venv/bin/python scripts/translate_archival_documents.py --spec <spec-id> --plan
venv/bin/python scripts/translate_archival_documents.py --spec <spec-id>
venv/bin/python scripts/translate_archival_documents.py --spec <spec-id> --reassemble
```

- `--plan`/`--dry-run`은 저본 슬라이싱·용어집·청킹·견적만 계산하며 모델을 호출하지 않는다. 스펙은 `config/archival_translation/<id>.json`이다.
- 기본 동시성은 5, `--retries`는 총 시도 수다. `--limit-chunks N`은 앞 N청크만 처리하고 최종 fragment은 쓰지 않는다.
- `--compare 'provider/model,provider/model'`은 같은 청크를 비교한다. `+think`, `+effort=high` 변형과 `--compare-chunk-ids 2,3`을 지원한다. 번호는 현재 청킹의 0 기반 인덱스이므로 재현 가능한 평가는 아래 고정 평가셋을 사용한다.
- 키 주입은 LLM 프록시가 맡는다. 사료 번역을 위해 provider 실키를 CLI에 전달하거나 DB 자격증명을 마운트할 필요는 없다.
- `api_routes/archival_translation.py`는 `/admin/archival-translation/specs`, `/plan`, `/run`을 소유한다. admin 인증, 요청 필드와 NDJSON 이벤트는 [API Reference](api_reference.md#archival-translation)를 따른다.

### 사이트 실행과 타이머

```bash
venv/bin/python scripts/translate_research_documents.py --limit 2 --dry-run
venv/bin/python scripts/translate_research_documents.py --limit 2
venv/bin/python scripts/translate_research_markdown.py <research-slug>
venv/bin/python scripts/translate_db_content.py --kind all --limit 10 --select-only
venv/bin/python scripts/static_page_translation_pipeline.py status
```

`scripts/static_page_translation_pipeline.py`는 `export`, `import`, `translate-deepl`, `deepl-usage`, `status`를 제공한다. `export`는 외부 번역에 쓸 파일을 만들며, `import`는 반환된 JSON을 검증한다. DeepL은 `DEEPL_API_KEY`를 사용하고 기본 대상 언어는 EN-US다. `--html-mode auto`는 HTML 번역 검증 실패 시 태그를 그대로 보존하는 segments 방식으로 재시도한다. segments 방식은 같은 문단·셀의 원문을 `context`로 전달하고, 코드·pre·script·style 본문은 번역하지 않는다. 요청은 항목 수와 바이트 크기로 나눈다.

저장소의 `deploy/systemd/research-document-translation.{service,timer}` 정의:

- 매일 호스트 시간 04:20, 최대 20분 무작위 지연, `Persistent=true`.
- 연구 DB `--limit 0 --max-chars 0`, 이어서 기타 DB `--kind all --limit 0`을 실행한다. 정적 페이지와 사료는 이 타이머의 대상이 아니다.
- `--max-chars`는 연구 DB에서 **선택할 원문 전체 길이 제한**이며 청크 크기 옵션이 아니다. CLI 기본은 limit 2/60,000자지만 타이머 정의는 두 제한을 해제한다.
- 서비스는 DB credential만 마운트하고 LLM provider 실키는 보유하지 않는다. 두 `ExecStart`에 `-`가 있어 명령 실패를 systemd가 무시하므로, 완료 판단은 번역 로그·실패 수를 함께 본다.

`dry-run`의 의미는 스크립트마다 다르다. 연구 DB의 `--dry-run`과 기타 DB의 `--select-only`는 모델 호출 없이 대상을 조회한다(DB 접속 필요). Markdown 파일·기타 DB·DeepL의 `--dry-run`은 실제 번역·검증을 수행하고 최종 대상 저장을 생략한다. Markdown은 이때도 청크 캐시를 사용할 수 있다. 정적 페이지 `import --dry-run`은 반환 JSON 검증만 수행한다.

## 3. 저본·검증·캐시

### 사료

저본은 `sources.py`의 `militera`, `wikisource`, `stalinism`, `libru`, `marxists`, `html` 어댑터가 처리한다. 범용 `html`은 스펙의 CSS `selector`, `nth`, `drop`을 사용하며 대상이 없으면 오류다. 선언 charset을 존중하고 선언이 없으면 UTF-8 → cp1251로 시도한다. 저본 해시와 문서 범위의 startsWith/endsWith를 확인하고, 여러 저본의 ID 충돌·이동을 피하려면 문서 `band`를 고정한다. 표는 원본 구조·숫자와 번역할 칸 어휘를 분리한다. OCR·스캔 PDF는 먼저 텍스트 저본을 준비해야 한다.

`[[번호|태그]]` 응답에서 중복·누락·추가 마커, 태그 불일치, 원문 그대로 반환, 한국어 부재, 잔존 원문 문자, 대상 밖 문자, 지나친 길이 감소, 마지막 블록의 문장 끊김을 검사한다. 문서 전체 잔존 검사인 `strayCyrillic`는 결과에 보고하는 항목이며, 이 목록이 비어 있다고 의미 정확성이 보장되는 것은 아니다.

청크는 제목 이동 시 빈 청크를 만들지 않는다. 크기 초과 블록은 `plan().oversizedBlocks`로 알리고, 문장·공백 경계로 나누어 부분 번역한다(구분할 곳이 없는 긴 문자열은 길이 기준). 부분 결과는 `.parts.jsonl`에 저장하고, 원래 블록으로 합쳐 검증한 뒤에만 정본 청크 캐시에 넣는다.

| 산출물 | 의미 |
|---|---|
| `output/archival_translations/<id>.jsonl` | 정본 청크 캐시. `--cache`로 변경 가능 |
| 같은 경로의 `.parts.jsonl` | 초과 블록 부분 결과. 재조립이 정본으로 오인하지 않도록 분리 |
| 같은 경로의 `.review.json` | 숫자·조건 검토 힌트와 TM 선택 근거 |
| 스펙 `output` 또는 `--out` | 모든 청크 성공 후 원자적으로 교체하는 HTML fragment |

정본 키는 `PROMPT_VERSION`, resolve된 provider/model/thinking, 시스템 프롬프트 해시, 유저 프롬프트 전문을 반영한다. `_prepare_chunk()` 결과는 pending 집계와 워커가 공유한다. 캐시는 현재 검증기로 재심사하고 실패한 청크만 다시 번역한다. 단, 과거 JSONL에는 원시 응답이 없으므로 이미 덮어써진 중복 마커 자체는 소급 판별할 수 없다.

`postEdits`는 원문→번역의 오역 수정과 조사 보정을 조립 단계에 적용한다. 병기 중복 축약·따옴표·스펙 `register`도 어댑터 소유다. 프롬프트·모델 교체 후 `run()`은 캐시 미스로 재번역할 수 있다. 발행본 손질은 현재 스펙과 산출물의 대응을 확인한 뒤 `--reassemble`을 사용한다. 재조립은 모델 없이 블록 번호로 캐시를 고르고 TM을 덮어쓰며, 같은 청크의 다중 레코드는 나중 것을 사용한다. 현재 청킹과 맞지 않는 고아 레코드를 쓰면 경고한다. `--max-chars`는 번역 당시와 맞춰야 한다. **frozen 스펙은 run과 reassemble 모두 거부한다.** 스펙 밖 수동 편집·병합을 캐시 조립이 되돌리지 않도록 하는 보호다.

### 연구 Markdown과 기타 사이트 콘텐츠

Markdown은 최상위 문단·목록·표·코드 블록 경계로 나눈다. 목표 8,000자는 하드 상한이 아니며 초대형 단일 표·목록은 더 클 수 있다. 호출 전에 언어 태그가 있는 코드 펜스·들여쓰기 코드·인라인 코드·HTML 태그·링크 목적지·참조 ID를 결정론적 플레이스홀더로 보호하고, 출력의 개수를 검사한 뒤 복원한다. **언어 태그가 없는 펜스(또는 `text`/`plain`)는 ASCII 도식으로 보고 번역 대상에 포함한다.** 연구 코퍼스의 태그 없는 펜스 63개 중 59개가 한국어 도식이었고, 보호하면 영어본에 한국어 도식이 남는다. 이 펜스는 내용 대신 줄 수가 같아야 통과하며, 태그 있는 펜스(python·java·yaml 등)의 한국어 주석은 그대로 둔다. 전체 블록 구조, 모든 링크·이미지 목적지, 참조 정의, 각주 ID, 코드, HTML 구조와 한글 잔존율(도식 포함)을 검사한다. 최종 문서 조립 후에도 전체 검증을 통과해야 한다.

연구 청크 캐시는 `output/site_translation_cache/<hash>.json`이다. 키에 원문 청크, 시스템 프롬프트, resolve된 모델 설정, 버전이 들어간다. 성공 청크는 전체 문서가 실패해도 남아 다음 실행에 재사용한다. `--force`는 최종 출력의 스킵 조건을 해제하며 유효한 청크 캐시까지 지우지는 않는다.

기타 DB JSON은 필수 문자열 필드와 필드별 미번역·HTML·URL 보존을 검사한다. 큐레이션의 `source_title_en`만 비어 있어도 통과하며, 이때 UPDATE는 기존 값을 유지한다. 엄격한 JSON 파싱이 실패하면 알려진 키 순서로 문자열 값을 복구한다(값 안의 이스케이프되지 않은 따옴표, 닫는 중괄호 누락). 일기 #438이 이 형태로 매일 밤 실패했다. 행 전체를 호출하며 연구용 청크 캐시는 사용하지 않는다. 정적 페이지는 필수 영어 필드, 안전한 inner HTML, 구조·속성, 한글 잔존율을 검사한다. HTML 검증에서 `alt`, `title`, `aria-label`, `placeholder` 값은 변경 허용 대상이고 다른 속성은 보존해야 한다. Markdown의 태그 보호는 이 속성들도 원문 그대로 복원한다.

숫자·날짜·금액 차이와 일부 부정/조건 표현은 **검토 힌트**이며 자동 수정이나 재시도 조건이 아니다. 연구 DB는 `output/translation_reviews/research-<id>.json`, 파일 번역은 출력 옆 `.review.json`에 기록한다. 기타 DB·정적 페이지는 이 보고서 저장 경로를 사용하지 않는다. 구조 검사는 문단 내부 누락, 주어·목적어 뒤집힘, 조건 범위 오역까지 판정하지 못한다.

## 4. 용어집과 번역 메모리

CommuLingo 인물·용어 스냅샷과 `glossary.extra`를 결합하고, 청크에 등장하는 항목만 기본 최대 60개 주입한다. 러시아어는 격변화와 단어 경계, 중국어는 전체 이름을 사용한다. 라틴 인물사전의 성 단독 항목은 문서에 전체 이름·이니셜 근거가 있어야 주입한다. 직접 지정한 `extra`는 인물사전의 같은 성 때문에 덮이거나 앵커 필터에서 삭제되지 않는다. `glossary.exclude`로 문서별 다의어·동명이인 충돌을 제외한다. 여성 성과 남성 성의 격변화 충돌은 사전 계획에서 경고한다.

용어표 표면 일치로 번역을 강제 검증하지 않는다. Hessen(지명/인물), Союз(나라/단체), Каменева(인물/격변화)처럼 문맥에 따라 달라지는 표기가 있기 때문이다. 사전 스캔 `scan_archival_terms.py --spec <id> --llm`과 사후 `audit_archival_terms.py --spec <id>`는 LLM으로 지시체·실제 번역 표기를 추출해 **보고서와 스펙 수정 제안만** 만든다. 번역 루프에 자동 적용하지 않는다. 결과는 `.terms.jsonl` 캐시와 `.terms-scan.md`/`.terms-audit.md`에 남으며, `--plan`은 모델 호출을 생략한다.

TM은 `runtime_tools/translation_memory.py`가 관리하는 SQLite `output/translation_memory.sqlite3`다.

```text
segments(id, lang_pair, source, target, doc_id, block_id,
         status, provider, model, created_at)
UNIQUE(lang_pair, doc_id, source, target)
status: machine < published < reviewed
```

- 사료 성공 블록과 기타 DB의 짧은 필드 쌍을 적재한다. 사료는 postEdits를 반영하지만 문서 위치에 묶인 병기 축약·따옴표 정규화는 적재하지 않는다. 적재 실패는 번역 자체를 중단하지 않는다.
- `backfill_translation_memory.py`는 기존 사료 캐시를 정렬해 적재하며 frozen 스펙은 published, 나머지는 machine으로 기록한다. API 호출은 없지만 TM 파일을 쓴다. `--stats`는 집계를 조회한다.
- 기본 `exact_matches`는 높은 상태 우선, 같은 상태는 최신 행 우선이다. 자동 재사용은 `reject_conflicts=True, decisions={}`로 published/reviewed만 조회하며, 최고 등급 안에서 복수 번역이면 보류하고 현재 블록 검증도 수행한다.
- 청크 경계를 유지하고 모든 블록이 TM으로 채워진 청크만 호출을 생략한다. 일부만 일치하면 청크를 처리한 뒤 TM이 모델 출력보다 우선한다. 같은 원문도 문맥·문체가 다를 수 있으므로 완전 일치는 무오류 보장이 아니다.
- 스펙 `tmReuse: {enabled: false}`로 전체 재사용을 끄거나 `excludeSources` 목록으로 정확한 원문을 제외한다. 문서 항목 `tmReuse: false`는 해당 문서만 제외한다. `tmSelected`는 상태·출처·세그먼트 ID를, `tmConflict`/`tmInvalid`는 보류 사유를 보고한다.
- `suggest_tm_examples.py`의 유사 번역례 추천은 사람이 `tmExamples`에 고정한다. 실행마다 동적으로 주입해 캐시를 흔들지 않는다.

## 5. 원문 최신성과 DB 적용 상태

`research_documents.markdown_en_source_sha256`는 번역에 사용한 원문의 SHA-256이다. `research_store.upsert_document()`는 원문이 바뀌고 새 영어 번역이 제공되지 않으면 이전 영어 필드와 번역 해시를 무효화한다. 연구 번역기는 public 문서 중 영어 본문이 비었거나 번역 해시가 `content_sha256`과 다른 문서를 선택한다. 이 두 해시를 관리하는 원문 저장 경로가 기준이다.

저장은 `id + 선택 당시 markdown + status='public'` 조건의 UPDATE다. 번역 중 원문이 바뀌거나 비공개가 됐으면 저장을 거부한다. 기록하는 출처 해시는 선택 당시 행의 `content_sha256` 값이다. 직접 SQL로 원문을 고쳐 `content_sha256`이 어긋난 행에 sha256(markdown)을 다시 계산해 넣으면 선택 조건과 영원히 불일치해 매일 밤 재번역된다. title_en/summary_en은 번역 Markdown에서 추출한다. 기타 DB도 선택 당시 원문 필드 값으로 조건부 UPDATE하지만, 연구용 해시 열이나 자동 변경분 탐지 정책은 공유하지 않는다.

파일 번역은 출력 옆 `.translation.json`에 sourceHash/targetHash를 기록한다. sourceHash가 같으면 수동 편집도 보존한다. 보존한 파일이 구조 검증에 실패하면 덮어쓰지 않고 오류로 보고하므로 손으로 고치거나 `--force`로 재번역한다. 원문이 달라지거나 메타데이터가 없으면 다시 처리하며, 생성 중 원문 변경을 확인하면 최종 파일 저장을 거부한다. 성공한 DB 변경 후에는 해당 frontend Redis 캐시를 비운다.

운영 적용 상태:

- **2026-09-07 운영 `leninbot` DB에 research-documents 마이그레이션 적용 완료.** `markdown_en_source_sha256`는 nullable TEXT이며 기본값이 없다. 새 선택 쿼리의 실행도 확인했다.
- 적용 직후 재평가 대상 public 문서는 **173건**(원문 1,639,404자)이었고, 타이머가 워킹트리 스크립트를 그대로 실행하므로 다음 04:31 UTC 실행에서 전부 재번역될 상태였다. 같은 날 검토에서 **기존 번역의 출처 해시를 `content_sha256`으로 백필**(174행, 비공개 1건 포함)하고, 직접 SQL 편집으로 어긋나 있던 `content_sha256` 6행(id 56, 57, 186, 368, 418, 430)을 sha256(markdown)으로 복구했다. 백필 후 재평가 대상은 0건이다.
- 이 작업에서 번역 배치·유료 비교·서비스 재시작은 실행하지 않았다. 타이머 정의는 전체 대상을 선택하므로 수동 실행량 제한은 `--limit`로 정한다.
- 새 환경에서는 `venv/bin/python scripts/schema_migrations.py --only research-documents`로 준비한다. DB credential과 승인된 쓰기 경로가 필요하다. 이번 운영 적용은 기존 DB credential과 해당 프로세스에만 설정한 `LENINBOT_ALLOW_WRITE=1`을 사용했으며, 암호나 전역 쓰기 설정을 파일에 추가하지 않았다.

## 6. 평가·검증과 남은 한계

`tests/fixtures/translation_eval.json`은 기존 RU/ZH/EN/DE/IT 사료 청크 15개와 KO→EN 구조·조건 예제 1개를 원문 해시와 함께 고정한다. 입력·용어 프롬프트를 스냅샷으로 보관해 이후 청크 번호·용어집 변경에 영향받지 않는다. FR 실제 사료는 아직 없다.

```bash
# 모델 호출 없이 목록 확인 또는 이미 생성된 후보 평가
venv/bin/python scripts/evaluate_translation.py
venv/bin/python scripts/evaluate_translation.py --candidates <candidates.json> --output <report.json>

# 오프라인 회귀·스모크
venv/bin/python -m unittest tests.test_translation_pipeline tests.test_call_registry_output_budget
venv/bin/python scripts/smoke_translation_memory.py
venv/bin/python scripts/smoke_archival_translation.py
```

`--generate`를 명시하면 현재 registry 모델을 실제 호출한다. `--id`로 고정 사례를 선택하고 `--context-chars 0|600|1200`으로 원문 맥락 실험을 할 수 있다. 자동 검사는 구조·형식과 검토 힌트를 제공하며, 사람이 누락·주체/객체·부정/조건·용어·자연스러움을 0~3점으로 평가한다. 미평가 점수는 null이고 자동 정답 점수가 아니다.

사료 스펙의 `sourceContextChars`는 기본 0, 범위 0~2000이다. 활성화하면 제목·절 제목·인접 원문을 출력 금지 참고 맥락으로 주입하고 캐시 키에 반영한다. 앞 청크 번역을 기다리지 않으므로 병렬성은 유지된다. 고정 평가셋에서 효과를 확인하기 전 기본 활성화하지 않는다.

2026-09-07 개선 검증에서 관련 단위 테스트 **177개**, 번역 스모크 **2종**이 통과했다. 범위는 공통 실행·사이트 구조·사료 검증·TM·원문 변경·출력 권한·할당량 중단·호출 계측·게이트웨이 호환성이다. 같은 날 검토 후 429 분류·펜스 왕복·관대한 JSON 복구·큐레이션 선택 필드·사료 교정 문구 테스트를 더해 번역 스위트 **58개**, 전체 허메틱 스위트 **529개**가 통과했다. `research/*.md` 84개 파일로 보호·복원 왕복을 확인했다(수정 전에는 펜스 뒤 빈 줄 때문에 10개가 어긋났다). 사료 스모크는 저본·frontend 체크아웃이 필요하며, 고정 평가셋과 공통 회귀 테스트는 외부 API·운영 DB 없이 수행한다.

남은 항목은 실제 모델 후보의 원문 대조 평가와 FR 사례 확장, 청크 크기·맥락 A/B 측정, 초대형 단일 Markdown 표·목록의 하위 분할, 기타 DB 행의 청킹·번역 최신성 확대다. Batch API는 미구현이다. 모델 선택·추가 비용 최적화는 실측과 의미 품질 평가 후 결정한다.
