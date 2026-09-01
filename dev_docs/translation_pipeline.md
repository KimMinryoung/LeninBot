# Translation Pipeline

두 개의 독립된 파이프라인이 있다. 공유하는 것은 LLM 호출 경로(`llm/call_registry.py` → gateway/proxy)와 `scripts/_translation_common.py`의 헬퍼뿐이다.

| | A. 사료(archival) 파이프라인 | B. 사이트 콘텐츠 파이프라인 |
|---|---|---|
| 방향 | RU→KO, ZH→KO | KO→EN |
| 엔진 | `runtime_tools/archival_translation/` | `scripts/translate_*.py` 4종 |
| 단위 | HTML 블록, `[[번호\|태그]]` 마커, 청크(기본 3,500자) | 문서/행 통짜 1회 호출 |
| 용어집 | CommuLingo 스냅샷 + 스펙 `glossary.extra`, 청크 등장 항목만 주입(상한 60) | 프롬프트 내장 소사전 |
| 검증 | 결정론 4종(마커·원문반환·한국어부재/문자잔존·길이) + 사유 첨부 교정 재시도. 용어표는 루프에서 검사 안 함 — 별도 LLM 용어 감사(보고 전용) | 결정론 검사 + 교정 재시도 1회 |
| 캐시/이어하기 | 내용 해시 JSONL, 청크 단위 | 없음(스킵 조건이 재개 역할) |
| 진입점 | `scripts/translate_archival_documents.py`, `api_routes/archival_translation.py` | systemd `research-document-translation.timer` + 수동 |

설계 원칙은 `~/uploads`로 전달된 번역기 인수인계 문서(2026-08-30)를 따른다: LLM은 청크 번역기로만 쓰고, 용어 일관성·연속성·검증은 결정론적 레이어가 책임진다. 모델 호출은 전부 `call_registry`의 feature 단위 설정(`config/llm_call_sites.json`)을 지난다.

## 사료 파이프라인 실행 (2026-08-30 정리)

- **실행**: `venv/bin/python scripts/translate_archival_documents.py --spec <id>`. 프로바이더 호출은 LLM 게이트웨이 프록시(`:8110`)를 지나고 키는 프록시가 주입하므로 **credstore·sudo·systemd-run이 필요 없다** (예전 `run-archival-translation.sh` 래퍼와 `LoadCredentialEncrypted` 안내는 삭제됨). `--plan`은 모델을 부르지 않고 슬라이싱·청킹·견적만 낸다. 같은 실행은 `POST /admin/archival-translation/run`으로도 된다.
- **모델·max_tokens·thinking은 registry가 결정한다**: `config/llm_call_sites.json`의 `archival_document_translation_{ru,zh,en,de,fr,it}` 항목(또는 `LLM_SITE_ARCHIVAL_DOCUMENT_TRANSLATION_<LANG>_MODEL` — model만 덮고 provider는 못 바꾼다). `call_registry.resolve()`는 항목 값을 호출부 기본값보다 우선하므로 CLI·API·`Options`에는 model 옵션이 **없다** — 예전 `--model`/`--max-tokens`는 조용히 무시되던 죽은 옵션이라 제거했다. 후보 모델은 `--compare provider/model[,+think,+effort=high]`로 같은 청크를 나란히 뽑아 본 뒤 registry 항목을 고친다. `--compare`의 preflight는 변형에 적힌 provider마다 검사한다.
- **청크당 준비물은 한 번만 만든다**: `_prepare_chunk()`가 (프롬프트, 캐시 키)를 돌려주고, `run()`의 pending 집계와 워커가 같은 것을 쓴다. `Stats` 카운터 갱신은 `Stats.add()`로 락 뒤에서 한다(워커 5개 공유).
- **캐시 키** = `PROMPT_VERSION` + resolve된 provider·model·thinking + 시스템 프롬프트 해시 + 유저 프롬프트 전문. 프롬프트·용어표·tmExamples·register 변경은 자동으로 키를 바꾼다. `PROMPT_VERSION`은 파서(`parse_response`)가 바뀌어 옛 레코드의 블록 분할을 믿을 수 없을 때만 올린다 — 검증기 강화는 캐시 재심사가 흡수한다.

## 모델 교체·재조립·용어표 충돌 경고 (2026-08-31)

- **프로바이더**: `archival_document_translation_{ru,en,de,fr,it}` = `gemini/gemini-3.1-pro-preview`, `_zh`만 `deepseek_anthropic/deepseek-v4-pro`(thinking on) 유지 — ZH→KO는 DeepSeek 우위라는 사용자 판단. 근거: RU 스펙 2건×2청크 `--compare`(고르바초프 1987 연설·숨가이트 1988 정치국 기록; `output/archival_translations/compare-deepseek-vs-gemini-*.md`). 산문은 Gemini 우위, 표기 규율(이니셜 로마자화·지명 병기·비칭)은 열세였으나 **RU 시스템 프롬프트 말미의 «표기 규칙 보강» 블록**으로 같은 청크 재검증 시 해소(이니셜 45곳 키릴 유지, 지명 병기 0, 비칭 0). Gemini executor는 thinking/output_config 키를 읽지 않으므로 항목에서 뺐다(기본 동적 추론). 모델 id는 프록시 경유 `client.models.list()`로 확인(무료) — "3.5 Pro"는 존재하지 않는다.
- **캐시 키 단절과 `--reassemble`**: 키에 provider·model·system_hash가 들어가므로 교체 뒤 구 문서의 postEdits 손질을 `run()`으로 돌리면 전체 재번역(=발행문 전면 교체)이다. 발행본 손질은 `translate_archival_documents.py --spec <id> --reassemble` — 캐시 레코드를 **블록 번호로** 조립하고 fragment만 다시 쓴다(LLM 호출 0). 규칙: 블록 집합이 현재 청킹의 어느 청크와 일치하는 레코드만 채택(다른 청킹 시절 고아 배제), 같은 청크 다중 레코드는 나중 것, `run()`과 같이 **TM 프리필이 마지막에 덮는다**(빠뜨리면 발행문이 TM 문구에서 캐시 문구로 바뀐다 — 포스펠로프 00485호 인용부 사례). 포스펠로프에서 `run()` 산출과 바이트 동일 재현 확인. `--max-chars`는 번역 당시와 같아야 한다(기본 3500; 일치하는 레코드가 없으면 고아로 채우고 경고).
- **캐시 컴팩션(일회성)**: 교체 직전 구 DeepSeek 키(구 프롬프트=git `552beaf^`의 SYSTEM_PROMPT)로 정본 레코드를 판별해 고아를 제거했다 — nko-order-227 6→3, provisional-government-death-penalty 4→2, stalin-1925-xiv-congress 129→69, sto-kronstadt 2→1. 원본은 `output/archival_translations/backup-20260831-precompaction/`. 구 키로도 완전 매칭이 안 된 14건(1924 헌법, 사면령, 부하린 1928, 21개조, 디미트로프, 비밀보고, 자치화, 4월 테제, 독소조약, nkvd 2종, 1920 조약, 1989 조약 결의, 1928 시베리아, 대전환, 얄타)은 용어표 드리프트로 손대지 않았다 — 재조립 시 다중 버전이면 last-wins라 결과를 통독할 것.
- **용어표 충돌 경고**: `glossary_collision_pairs()` — 남성 성+а(생격·대격 표면)가 한국어 표기가 **다른** 별도 항목(여성형 인물)과 겹치는 쌍. `plan()`이 exclude 적용 뒤, 그 표면이 원문에 실제 있을 때만 stderr로 경고한다(사전 전체가 아니라 이 문서에 해당하는 것만; XXVII съезд/съезда 같은 동일 ko 별칭은 제외). 주입 자체는 건드리지 않는다 — 여성 본인이 등장하지 않는 문서는 `glossary.exclude`에 넣고 시작. 배경: 포스펠로프 보고서에서 인물사전의 예브게니야 예조바가 `Ежова→예조바`로 함께 주입되어 예조프의 생격 8곳이 "예조바"로 발행됨(postEdit로 교정, frontend `f4c9781`). 테스트 `tests/test_glossary_collisions.py`.
- `compare()`/`probe()`가 executor의 `(text, usage[, truncated])` 튜플 반환(8-30 truncation guard 이후)을 문자열로 취급해 죽던 회귀도 같은 날 수정.

## 첫 EN/DE/IT 배치 (2026-08-31)

- 스펙 12건(`marshall-plan-memoranda-1947`, `italy-surrender-instruments-1943-1945`, `cia-poznan-bulletins-1956`, `cia-ore-29-48-1948`, `carter-byrd-letter-1980`, `kohler-telegram-1964-10-18`, `varkiza-agreement-1945`, `german-constitutions-1949`, `fuehrer-directives-16-17-1940`, `fuehrer-directive-41-1942`, `ddr-travel-regulation-1989-11-09`, `gran-consiglio-documents-1928-1943`)을 포크 에이전트 3개(EN/DE/IT)가 병렬 작성, `--plan`으로 검증 뒤 번역. 언어별 단일 스펙이라 이탈리아 묶음은 연합군 측(EN 정본)과 이탈리아 측(IT)으로 나눴다. CIA 문서는 OCR txt에서 문단을 복원한 `-prepared.html`(제목 인식 병합, 배포처·기밀 표시 제거). 짧은 카터 서한은 `output/archival_translations/merge-staging/`로 번역해 데탕트 문헌에 HTML 병합(스펙 frozen).
- **라틴 저본에서 드러난 함정과 수정**: (1) `_decode_page`가 cp1251 외 선언 charset을 무시 → 존중. (2) 표 셀 어휘가 키릴 전용 → 글자 있는 셀 전부. (3) `validate`의 잔존 검사가 로마자 「글자」 비율이라 항목 기호·이니셜·약어가 걸림 → 라틴은 낱말(`stray_word`≥4자) 기준. (4) 끊김 검사가 닫는 괄호로 끝나는 서명부를 오판 → 원문 끝 부호를 `.!?`(+닫는 부호)로 한정. (5) `generic_html` 폴백 경로가 원시 줄바꿈을 블록 줄로 넘김(FRUS) → 블록 태그·빈 줄만 문단 경계. (6) `slice_documents` 블록 캐시가 경로 단위 → source 전체 키. FRUS 잡음은 `drop: ["a.tei-pb1","a.note"]`(쪽 표시·각주 번호).
- **Gemini 선불 크레딧 소진(429 RESOURCE_EXHAUSTED)**: 배치 중반에 크레딧이 떨어져 `german-constitutions-1949`(11청크), `italy-surrender-instruments-1943-1945`(1청크), `marshall-plan-memoranda-1947`(폴백 수정으로 케넌 청크 전부 재번역 필요)이 대기. 충전 후 `--spec <id>`로 재실행하면 성공 청크는 캐시에서 나온다. 빈 응답 재시도 3회는 시간만 쓰므로 429가 보이면 바로 멈추는 게 낫다.
- 발행 뒤 큐 처리는 스크래치 `gaps-done.sql` 형식(UPDATE … status='done', resolved_id, resolution 앞에 처리 메모), manifest 등록은 `manifest-drafts.json` + `publish.py`(HTML이 있는 항목만).

## 라틴계 프롬프트 «표기 규칙 보강» + 고정 함정 청크 비교 (2026-09-01)

- **첫 EN/DE/IT 배치 통독 결과**: 산문은 좋았으나 고유명사·관용어 층에서 반복 실수 — `decreto Reale`→"레알레 칙령"(형용사를 인명으로), `Hessen`→"게센"(러시아어 전사 습관 누출), `Dnjepr`→"드니프로"(1942년 문서에 현대 표기), `Vereine und Gesellschaften`→"사단과 회사", 조항 제목에 임의 낫표 「제25조」(postEdit 29건), 라틴 이니셜 음차 "지. 시안토스 / 제이. 소피아노풀로스". RU 프롬프트의 «표기 규칙 보강» 블록에 상응하는 것이 라틴계 프롬프트에 없었던 것이 원인의 절반.
- `core._latin_prompt`에 공용 «표기 규칙 보강»(이니셜 로마자 유지, 원어 발음 음차, 원문 시대 지명, 낫표 금지, 관용어≠고유명사)을 붙이고 `_latin(..., notation=)`으로 언어별 함정 예시를 덧붙인다(EN: Khrushchev/Kiev, DE: Vereine/Seestreitkräfte, FR: décret/arrêté/직함, IT: decreto Reale). **시스템 프롬프트 해시가 캐시 키에 들어가므로 EN/DE/FR/IT 청크 캐시는 전부 무효** — 발행된 12건은 `run()`을 다시 돌리면 전면 재번역이 된다. 발행본 손질은 `--reassemble`만 쓰고, 새 스펙부터 새 프롬프트가 적용된다.
- `--compare-chunk-ids 2,3`: 문서 앞 N청크 대신 지정한 청크(0부터, 현재 청킹)로 비교한다. 함정이 든 청크를 고정 테스트셋으로 쓰기 위한 것(로드맵 1의 첫 조각). 청크 번호는 `plan()`의 `_chunks`에서 원문 문자열로 찾는다.
- `gran-consiglio-documents-1928-1943`의 `proclama-badoglio` 블록 범위 `[0,2]→[0,4]`: 번역(08-31 23:38)은 `<br>` 재분할 이전 파서로 2블록이었고, 직후 커밋(1f6bce1, 23:39)부터 4블록이라 `plan()`이 "range end moved"로 죽었다. 발행본은 온전하므로 그대로 두되 캐시 레코드는 현 청킹과 안 맞는 고아다(다음 `run()`에서 그 청크만 재번역, `--reassemble`이면 고아 채움 경고).

## 모델 비교 2라운드 — Gemini·GPT·Claude·DeepSeek (2026-09-01)

`--compare-chunk-ids`로 16청크 × 5변형(`gemini/gemini-3.1-pro-preview`, `openai/gpt-5.6-terra+effort=high`, `claude/claude-opus-5`, `claude/claude-sonnet-5`, `deepseek_anthropic/deepseek-v4-pro+think`). 1라운드는 표기 함정 청크(기본법 2·3, 지령41 3, 대평의회법 0, 바르키자 3, 고르바초프 0·1, 숨가이트 0·1), 2라운드는 난문(스탈린 1925 결론연설 60·64, 부하린 1926 26, 통합반대파 40, 콜러 전문 1, 케넌 PPS/1 1, 그란디 결의안 2). 보고서 `output/archival_translations/compare-20260901-*.md`, `compare-20260901-hard-*.md`. 판정은 파일마다 원문 대조 통독(포크 3 + 직접 1).

- **표기 함정은 새 라틴 프롬프트로 5모델 전부 통과**(Reale→왕령, Dnjepr→드네프르, 이니셜 로마자 유지, Vereine→결사와 단체). 이 청크들은 더 이상 변별력이 없다. 예외: GPT만 청크 하나에서 「제10조」 낫표 재발, Sonnet은 `왕령(decreto Reale)` 병기.
- **「게센」의 진짜 원인은 용어표**: 인물사전의 보리스 게센(Hessen) 항목이 라틴 표면 `Hessen`에 걸려 `Hessen→게센`으로 주입됐다. 용어표를 프롬프트 예시보다 우선한 GPT·Opus·Sonnet이 "게센", Gemini·DeepSeek이 "헤센". 스펙 `glossary.exclude: ["Hessen"]`로 처리. 라틴 문서에서 인물 성(姓) 단독 표면 매칭은 같은 사고를 또 낼 수 있다(예조바 사례의 라틴판).
- **GPT-5.6 Terra(effort=high)는 후보 탈락**: 16청크 중 3청크에서 문장 한가운데 벵골 문자(`তথ তথ…` 뒤 블록 5개 무응답)·조지아 문자(`정치국은 სწორედ 이러한`)를 냈다. 산문·정확성은 이론 산문(부하린)에서 1위였으나 이 급 사고가 나면 검증기 없이 못 쓴다 → **`validate()`에 «대상 밖 문자 혼입» 검사 추가**(`foreign_letters()`: 유니코드 범주 L* 중 한글·로마자·키릴·한자·가나·그리스 밖의 글자. 원문자 ①·분수 ¼는 범주 No라 통과). 전 캐시 13,986블록 재심사 오탐 0.
- **난문 판정 요약** (파일별 순위):
  - 스탈린 1925 속기록: Opus > Sonnet > DeepSeek > Gemini > GPT. Gemini는 삽입구 `(Смех, аплодисменты.)` 3곳과 소제목을 키릴로 남김(run()에서는 stray 검사가 재시도로 잡을 유형). GPT는 지노비예프 "전환" 방향 오역 + 청크 붕괴.
  - 통합반대파: Opus > DeepSeek > Gemini(«подкулачников»→"반쿨라크" 오역) > Sonnet(лишенцы→"피선거권 상실자") > GPT.
  - 부하린 1926: GPT > Opus > Gemini > Sonnet > DeepSeek(주어·목적어 뒤집은 순환문). «известную полосу»는 5모델 전부 정답 — 이 함정도 소진.
  - 고르바초프 1987: Opus > Gemini > Sonnet > DeepSeek > GPT. Gemini는 «레닌(Ленин)»·«(ЦК КПСС)» 재병기(어제 재검증에서 0이었던 위반이 재발), Opus는 "동지들/동지 여러분" 혼용·"정치국원/콜호즈원" 용어표 대조 필요.
  - 콜러 전문·케넌 PPS/1(분석 산문): Opus > GPT > Sonnet > Gemini > DeepSeek. Gemini "Again, paradoxically"→"다시 말해 역설적이게도", "feels"→"느낀다" 직역; DeepSeek "apparent belief"→"표면적으로 지녔던 신념".
  - 바르키자(조약): Gemini > Sonnet > DeepSeek > GPT > Opus — Opus가 누적 조건("사면법 적용 대상 **중** 점령기 가담자")을 두 집단으로 분리(자격 범위가 달라지는 실질 오역).
  - 기본법: DeepSeek > Sonnet > Opus("Kein Deutscher"→"어떠한 도이처도") > Gemini(출판의 자유/교육의 자유/망명권 — 법률 관용 표기 약함) > GPT("학문의 자유" 오역, 낫표).
  - 그란디 결의안(한 문장짜리 5단락): Gemini ≥ Opus > GPT ≈ Sonnet(문장 분해) > DeepSeek(관계절 오결합). 지령 41: 5모델 차이 없음.
- **결론**: 어느 모델도 무결점이 아니다. 종합하면 **Opus 5가 산문·문맥(속기록 수사, 분석문, 이론 산문)에서 가장 강하고 결함이 용어 관행 층(동무/동지, 총회/전원회의, 도이처)에 몰려 있어 용어표·postEdits로 잡히는 종류**인 반면, Gemini는 산문 좋고 조약·미사여구에 강하나 다의어(반쿨라크, again)와 병기 규율에서 반복 실수, DeepSeek은 법률문에 안정적이나 구문 오결합·직역투와 청크당 5~16분. 값은 Opus $5/$25가 Gemini $2/$12의 2.5배. 라우팅 변경은 사용자 결정 — 이 근거로 (a) RU 속기록·분석 산문 스펙은 Opus, (b) 조약·법령·지령은 Gemini 유지, 같은 언어쌍 안에서도 문서 종류로 가르려면 `SourceLanguage.feature`가 아니라 스펙 단위 feature 오버라이드가 필요하다(미구현).
- 운영 관찰: `compare()`는 executor를 직접 불러 `llm_audit_log`에 토큰·비용이 안 남는다(프록시 전송 행만 남고 토큰 0). 이번 2라운드 지출은 토큰 실측이 없어 추정치(≈$4~5)만 있다. DeepSeek+think는 청크당 최대 962초 — 스탈린 청크에서 병렬 5워커여도 문서 하나에 시간이 너무 든다.

## 저본 어댑터

`runtime_tools/archival_translation/sources.py`. 문서고마다 어댑터 하나(`militera`, `wikisource`, `stalinism`, `libru`, `marxists`)가 원칙이고, 2026-08-31에 **셀렉터 범위 범용 어댑터 `html`**을 더했다: 스펙의 `source.selector`(CSS 셀렉터)가 문서를 담은 요소를, `source.drop`(선택)이 먼저 버릴 자식(공유 버튼, 내비게이션)을 지정한다. "일반 HTML 파서는 조용히 잘못 자른다"는 원칙은 유지된다 — 이 어댑터는 추측하지 않고 스펙이 지목한 요소만 읽으며, 셀렉터가 아무것도 못 찾으면 빈 문서가 아니라 오류다. sha256·startsWith/endsWith 가드는 그대로 적용된다. 문서 한두 건씩 흩어져 있는 사이트(hrono.ru, doc20vek.ru, kremlin.ru 법령은행, coldwar.ru)를 위한 것이다. 리프 블록 요소(p·h*·blockquote·li·pre)를 문서 순서대로 뽑고, `<br>`로만 나눈 느슨한 본문이 절반을 넘으면 빈 줄 기준으로 다시 나눈다. 표는 wikisource와 같은 방식(rows + 칸 어휘)이다. 같은 셀렉터가 메뉴 칸과 본문 칸에 함께 걸리는 표 레이아웃 사이트를 위해 `source.nth`(0부터, 몇 번째 일치인지)를 둔다. 저장 페이지는 `core._decode_page`가 선언된 charset(windows-1251 등)을 따라 읽고, 선언이 없으면 UTF-8 → cp1251 순으로 물러선다(1989년 관보 전자판이 cp1251). 디스패치는 `sources.parse(source, raw)`. 2026-08-31 실사용: kremlin.ru 법령은행(`div.reader_act_body`), 1000dokumente.de(`div#tab3 div.text-ru`), vedomosti.sssr.su(`body` + 블록 범위로 관보 한 호에서 한 항목 절단), hrono.ru(`td[valign="top"][align="left"]`), istmat.org HTML 노드(`div.field-name-body`), grachev62.narod.ru(`body`). istmat.org의 "문서" 노드와 docs.historyrussia.org(ЭБИД)는 PDF 스캔·이미지 뷰어라 텍스트 어댑터로는 열 수 없다(2026-08-31 확인).

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

- 사료: `validate()`(마커 누락/원문 반환/한국어 부재/원문 문자 잔존/길이 하한/**응답 끊김**) + 문서 전체 `stray_cyrillic()`. 실패 사유를 교정 메시지로 붙여 재시도. 끊김 검사(2026-08-31): 청크의 마지막 비어 있지 않은 블록에서 원문은 문장부호로 끝나는데 번역의 마지막 줄이 그렇지 않으면 실패 — 1925 대회 문서 블록 43·1000017이 "…만들어졌", "…지원"에서 끊긴 채 길이 하한(0.36·0.30 > 0.25)을 통과해 발행된 사고의 후속. 끊김은 마지막 블록에서만 조용히 지나가고(앞에서 끊기면 마커 누락으로 잡힘) 전 스펙 캐시 2,923블록 실측에서 이 제한 아래 오탐 0건. **캐시 재심사**: 캐시에서 꺼낸 청크도 현재 `validate()`를 다시 통과해야 쓰인다(`_cached_blocks`) — 검증기가 엄격해질 때 `PROMPT_VERSION`을 올려 전체를 재번역하는 대신 새 검사에 걸리는 청크만 다시 번역한다(`cacheInvalid` 이벤트, `Stats.revalidated`). **용어표 준수는 검사하지 않는다** (2026-08-30 제거): 표면 일치 검사는 다의어(Союз, Правда, Октябрьский)와 인물 격변화 충돌(Каменева)을 가릴 수 없고, 그 검사가 교정 재시도로 모델의 옳은 첫 번역을 뒤집어 "소비에트 소유즈 (의원 그룹)", "옥탸브리스키 혁명"을 발행시켰다. 용어표는 프롬프트에 참고로 주입될 뿐이며, 표기 통일은 발행 전 통독 + 스펙 postEdits 몫이다.
- **LLM 용어 일관성 (보고 전용, 2026-08-31)** — `runtime_tools/archival_translation/terms.py`. 표면 매칭이 못 가리는 다의어·격변화 판정을 LLM(registry `archival_term_extraction`, deepseek-v4-flash·json_mode·추론 off)에 맡기되, 결과는 **보고서와 붙여넣을 스펙 조각**으로만 나온다. 번역 루프·재시도에는 넣지 않는다(어제 제거한 검사가 뒤집은 것이 바로 그 자리다). 채택은 사람이 스펙(`glossary.extra`/`glossary.exclude`/`postEdits`)을 고쳐서 하고, postEdits 변경은 캐시 키가 유효한 동안은 전 청크 캐시 적중이라 재조립만 일어난다 — 모델·프롬프트 교체 뒤에는 `--reassemble`을 쓴다(위 2026-08-31 절).
  - 사전 스캔 `scripts/scan_archival_terms.py --spec <id> --llm [--plan]`: 번역 청크와 같은 단위로 인명·기관·지명·간행물·정치용어를 lemma+sense로 뽑아 (a) 용어표 미등재 후보 + 제안 표기(`glossary.extra` 조각), (b) 표면 매칭으로 청크에 걸렸지만 그 문맥에서는 뜻이 다른 용어표 항목(misfire — 전 청크 오탐이면 `glossary.exclude` 조각, 일부만이면 두 뜻 공존 표시)을 낸다. 대소문자 신호를 쓰지 않아 **중국어 스펙도 된다**. 정규식 스캔(플래그 없음)은 그대로 남아 있다.
  - 사후 감사 `scripts/audit_archival_terms.py --spec <id> [--plan]`: 청크 캐시(모델 원출력)를 블록 번호로 원문과 정렬해 (원문, 번역) 쌍을 보이고 항목마다 **번역문이 실제 쓴 표기**를 뽑는다. 결정론 집계로 (a) 한 항목에 표기가 둘 이상(불일치 — 다수/소수, 소수의 sense가 다수와 전혀 안 겹치면 "다의어일 수 있음" 표시), (b) 용어표 항목의 뜻(misfire 아님)인데 표기가 다름(이탈)을 보고하고, 스펙 postEdits가 이미 덮는 블록은 "postEdits 적용됨"으로 구분한다. 미처리분만 `postEdits` 제안 조각으로 낸다. 재번역 없음.
  - 추출 결과는 `output/archival_translations/<id>.terms.jsonl`에 (프롬프트 버전·모드·provider·model·프롬프트) 키로 캐시 — 같은 문서 재감사는 호출 0회. 보고서는 `<id>.terms-scan.md` / `<id>.terms-audit.md`.
  - **용어표 연결도 LLM이 한다**: 항목마다 «이 청크에 제시된 용어표 항목 중 같은 것을 가리키는 것»을 `glossary` 필드로 답하게 하고, 이탈 판정·기준 표기는 그 연결이 있을 때만 쓴다. 첫 실행(2026-08-31)에서 정규식으로 lemma를 용어표에 대자 Союз(나라)가 Союз(의원 그룹)에 걸려 "연방 → 소유즈 (의원 그룹)"이 제안됐다 — 표면 매칭을 판정에 쓰면 안 된다는 같은 교훈이다. 기준 표기 = 용어표 표기 > postEdits가 덮고 남은 블록이 가장 많은 표기(동률이면 첫 등장). 다수 표기가 이미 postEdits로 고쳐진 오역("소비에트 소유즈")이면 기준이 되지 않고, 불일치 제안과 이탈 제안이 반대 방향("라린→루리예"/"루리예→라린")으로 나오지 않는다.
  - 집계 키는 lemma 소문자뿐이다. kind(place/org)와 sense는 같은 지시체에도 청크마다 흔들리므로 집계로만 남기고, sense는 사람이 다의어를 가리는 근거로 쓴다. 모델이 지시를 어기고 붙여 온 원어 병기 괄호(사르키스(Саркис))와 복수 '들'은 파서가 결정론으로 뗀다.
  - 첫 실검증(1925 대회, deepseek-v4-flash, 67청크 ≈ $0.1): 발행본에 남아 있던 실제 불일치를 잡았다 — 라린이 3블록에서 "루리예"로 남음(postEdits 키 "루리예)"가 병기 형태만 덮음), Косиор→"코시오р"(키릴 р 혼입), кулак 미번역 1건, смычка "결합" ×3, 협상국/앙탕트, 국가계획위원회/고스플란 등.
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
| §2.2 사전 스캔(PREPARE) | ✅ | `scripts/scan_archival_terms.py` — 정규식(RU 전용: 약어·문장 중간 대문자) 또는 `--llm`(RU·ZH: 문맥 포함 추출 + 용어표 오탐 보고). 채택·표기는 사람이 결정 |
| §2.2 청크 등장 항목만 주입 | ✅ | `glossary_for()` + 상한 60 |
| §2.2 표기 변형 매칭 | ✅ | 러시아어 곡용 변형 + 경계 가드, 중국어 무경계 |
| §2.2 다의어 항목 차단 | ✅ | `glossary.exclude` — 이 문서 문맥에서 거의 항상 다른 뜻인 항목은 주입에서 뺀다 (주입 목록이 바뀌므로 캐시 키가 바뀐다) |
| §2.3 TM 정렬 쌍 저장 | ✅ | 이번 변경. v1은 적재 우선 |
| §2.3 완전 일치 재사용 | ✅ | 검수 등급 한정, 청크 경계 보존(`_tm_prefill`) |
| §2.3 유사 세그먼트 예시 주입 | ✅ | 스펙 고정 방식: `scripts/suggest_tm_examples.py`가 검수 세그먼트를 어휘 겹침으로 추천 → 사람이 스펙 `tmExamples`에 채택 → 청크 프롬프트의 «참고 번역례»로 주입. 동적 주입은 캐시 키를 흔들어 의도적으로 배제 |
| §2.4 번역투 금지 목록 이식 | ✅ | 이번 변경 (RU·ZH 프롬프트) |
| §2.4 스타일 규칙 캐시 위치 | ✅ | 시스템 프롬프트 고정 + 캐시 키 포함 |
| §2.5 문장 수/길이 급감 | ⚠️ | 문장 수 대신 실측 기반 길이 하한(RU 0.25 / ZH 0.7) |
| §2.5 용어집 준수 사후 검사 | ✅ | 루프 안 표면 일치 검사는 제거(2026-08-30, 오탐이 재시도로 옳은 번역을 뒤집음). 대신 `audit_archival_terms.py`가 LLM으로 실제 쓰인 표기를 뽑아 불일치·이탈을 **보고**하고 postEdits 제안을 낸다. 자동 수정·재시도 없음 |
| §2.5 숫자·마크업 보존 | ⚠️ | 표 숫자는 코드 보존, 사이트는 태그 열·URL 검사(이번 추가). 본문 숫자 대조는 없음 |
| §2.5 미번역 잔존 검사 | ✅ | 블록 + 문서 전체(키릴·한자), 사이트는 한글 잔존율 |
| §2.5 위반 항목만 명시 재번역 | ✅ | 사료 원래 있음; research·db_content는 이번 추가 |
| §2.6 위치 지정 편집 정제 | — | 정제 단계 자체가 없다(P5의 회귀 위험이 없는 상태). 필요해지면 diff 반환으로 설계 |
| §2.7 단일 어댑터·언어쌍 설정 | ✅ | `call_registry` + feature 단위 JSON, 핫 리로드. 사료는 `archival_document_translation_ru`/`_zh`로 분리되어 언어쌍별 provider·model 교체 가능(`SourceLanguage.feature`). 교체 전 `--compare`로 검증. CLI·API에는 model 옵션이 없다(registry가 이김) |
| §2.7 Batch API | ❌ | 미지원 — 남은 로드맵(야간 타이머 작업이 후보) |
| §2.7 토큰·비용 기록 | ✅ | `record_llm_call` 감사 + `plan()` 사전 견적 |
| §2.8 고정 테스트셋·자동 지표 | ❌ | 없음 — 남은 로드맵 §5-5 (모델 교체 재평가의 전제) |

## 남은 로드맵 (우선순위 순)

1. 언어쌍별 고정 테스트셋 + 청크 크기 실험(1k/3k/8k/통짜) — 모델 라우팅 재평가의 전제. (2026-09-01: `--compare-chunk-ids`와 난문 16청크 목록이 첫 조각. 표기 함정 청크는 프롬프트 보강으로 변별력을 잃었으니 테스트셋은 난문 위주로.)
2. Batch API 경로(야간 타이머 작업 50% 할인).
3. 사이트 파이프라인 청킹 — 60k자 캡을 넘는 문서가 생기면 사료 엔진의 마커 방식 재사용.
