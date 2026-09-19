# Jev (TypeSafe System One) 도입 계획

작성 2026-09-19. 1~3절은 도입 전 계획, 4.4절부터는 **구현 상태와 실측 기준선**이다. 배포된 것:
`decide()` 인프라(4.4), CommuLingo 조사 인용 게이트(4.6, enforce), route_task 1차 분류기(4.7), CommuLingo
독립 검토 인용 게이트(4.8, enforce). 각 도메인 문서(`llm_call_registry.md`·`llm_gateway.md`·`commulingo_pipeline.md`)가
현재 동작을 기술하고, 이 문서는 후보 목록·기준선 표·채택/기각 근거를 보관한다.

## 1. Jev가 무엇인가

TypeSafe AI(전 OpenAI 연구자 Diogo Almeida 창업)가 2026-09-15 공개한 첫 "System One 모델".
LLM이 아니다 — 텍스트를 생성하지 않고, **상태(state)를 넣으면 타입이 정해진 확률적 판정**을 한
번의 병렬 패스로 돌려준다. RLHF 대신 "Reinforcement Learning for Calibrated Decisions"로
학습해 confidence가 정확도와 같이 움직이도록 맞췄다고 주장한다.

| 항목 | 값 |
|---|---|
| 엔드포인트 | `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer` |
| 모델 | `jev-1.13.0` (= `jev-latest`, `jev-preview`) |
| 입력 | 문자열·JSON·배열. 텍스트만(이미지 불가). 요청당 64k 토큰, 질문당 32k |
| 가격 | 입력 $0.042/MTok, 출력 $0 |
| 지연 | 70~500ms |
| 한도 | 250k tok/s, 1,200 req/min |
| SDK | `pip install typesafe-sdk` (`TypeSafeClient`, `Choice/Score/Noul`), `@typesafe-ai/sdk` |

질문 유형 세 가지. 한 요청에 여러 질문을 섞어 넣을 수 있고 각각 독립 평가된다.

| 유형 | 묻는 것 | 응답 필드 |
|---|---|---|
| `noul` | 참/거짓 확률 | `noul` (0~1). 선택적 `criteria.true/false` 설명 |
| `choice` | 선택지(≤255) 중 하나 | `choice`, `probabilities`, `confidence` |
| `score` | 순서 있는 2~10 단계 척도 위 위치 | `score`(단계 사이 실수), `legend`, `probabilities`, `confidence` |

응답 `usage.input_tokens`와 실제 `model` 버전이 함께 온다. HTTP 422(검증), 429, 529.

### 공식 문서가 인정하는 약점 (`model-jaggedness/jev-1.13`)

- **문자 그대로 읽는다.** 부정·이중부정·함축 조건·범위 한정어를 액면대로 처리.
- 세지 못하고 계산 못 한다. 날짜를 순서 있는 양으로 비교하지 못한다.
- 결정과 무관한 내용이 state에 많을수록 정확도가 떨어진다("context rot") — 코드로 먼저 걸러 필요한 필드만 넣어라.
- **state를 적대적으로 취급하지 않는다.** 주입된 지시나 오도하는 프레이밍이 출력에 영향을 준다.
- `P(noul)+P(¬noul)=1` 같은 구조적 불변식을 보장하지 않는다.
- **영어가 주 학습 언어. CJK 등 다른 언어는 "처리되지만 같은 수준은 아님", 쓰기 전에 직접 테스트하라**고 문서에 명시.
- confidence 문턱값은 도메인별로 자기 데이터로 정하라. 보정 보증은 없다.

이 프로젝트의 판정 대상은 대부분 **한국어·러시아어**다. 마지막 두 항목이 도입 여부를 결정한다.

## 1.1 접근 경로 — 직접 API(현재)와 호스팅 경로 셋

**현재 운영 경로는 직접 API다** (`POST https://api.typesafe.ai/v1/systemone`, Bearer `TYPESAFE_API_KEY`, 모델 `jev-1.13.0`
버전 고정; 별칭 `jev-latest`/`jev-preview`는 릴리스 때 이동하므로 문턱값 튜닝을 지키려면 쓰지 않는다. 한도 250k tok/s·1,200 rpm,
64k 컨텍스트, state+최장 질문 32k. 2026-09-19 16:37 키 마운트, 같은 날 16:39 `scripts/smoke_jev.py` 3/3, 336~730ms, 판정은 OpenRouter
때와 같은 방향 — 교정 noul 0.13/0.73, 라우팅 conf 0.99, 인용 supports 1.0). 아래는 직접 API가 early access 대기열이던
같은 날 오전에 조사한 우회 경로들이며, OpenRouter는 이날 오전~오후 실제 운영 경로였다(기준선 4.5~4.9의 표본은 이 경로로 측정).

| 경로 | 호출 | 모델 ID | 가격 | 컨텍스트 | 비고 |
|---|---|---|---|---|---|
| **OpenRouter Decisions** (2026-09-18 beta 추가) | `POST https://openrouter.ai/api/alpha/decisions`, `Authorization: Bearer <OPENROUTER_API_KEY>`, body `{model, state, questions}` — **native와 거의 동일한 wire format** | `typesafe/jev-1.13`(버전 고정 가능), `~typesafe/jev-latest` | $0.042/M 입력, 출력 $0 (native와 같음) | 32k | alpha 경로라 스키마 변경 가능. `instructions`/`criteria` 값을 문자열로만 검증(구조화 값은 JSON 문자열로). OpenRouter 계정+크레딧 선충전 필요, 대기열 없음 |
| **Cloudflare Workers AI** | `POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run`, body `{"model":"typesafe/jev","input":{state,questions}}`, 응답은 Cloudflare 봉투 `{result,success,errors}` 안 | `typesafe/jev` (항상 최신 alias — **버전 고정 불가**) | 문서에 미기재("대시보드에서 확인"), Workers AI 일반 규칙은 무료 10k neurons/일 + $0.011/1k neurons | 32k | 프로젝트에 **Cloudflare 계정이 이미 있다**(R2·캐시 퍼지, `R2_CF_ACCOUNT_ID`). 단 기존 `R2_CF_API_TOKEN`은 R2 범위일 것이므로 Workers AI 권한의 **별도 토큰** 발급 |
| Vercel AI Gateway | AI SDK 7 `experimental_evaluate` (TypeScript) 중심, 순수 HTTP 형식은 공개 문서에서 확인 못 함 | `typesafe-ai/jev` | 미확인 | — | Python 런타임과 맞지 않아 보류 |

기타: AI/ML API가 제공한다는 언급이 있으나 미확인. 비공식 파이썬 클라이언트 `jevclient`, 공식
`system-one-adapter`(LLM으로 Jev 흉내내는 비교용 어댑터)가 있으나 우리는 registry executor로 직접 HTTP를 친다.

**(당시 판단) shadow 평가는 OpenRouter로 시작.** 이유 — (1) native와 body·응답이 같아 `decide()` executor를
한 번 짜고 base URL만 바꾸면 승인 후 직접 API로 전환된다, (2) `jev-1.13` 버전 고정이 되어 문턱값 튜닝이
유효하다, (3) 가격이 native와 같고 공개돼 있어 `gateway.estimate_cost_usd`에 바로 넣을 수 있다.
Cloudflare는 계정이 있다는 장점이 있지만 봉투 형식이 다르고 alias만 있고 가격이 불투명해 2순위.
둘 다 프록시 라우트(`openrouter`, `cloudflare_ai`)로 등록하고 키는 `leninbot-llm-proxy`만 보유한다는
원칙은 같다. OpenRouter 가입·크레딧 충전은 새 지출이므로 사용자 결정.

## 2. 이 프로젝트에서의 적합 조건

Jev가 맞는 자리는 다음을 모두 만족하는 곳이다.

1. 출력이 **닫힌 집합**(라벨·척도·예/아니오)이고 문장을 만들 필요가 없다.
2. 지금 LLM 한 방(`call_registry` 원샷)으로 JSON을 받아 파싱하고 있거나, 비용 때문에 **판정을 아예 안 하고 있다**.
3. 틀려도 코드 쪽 폴백이 있고, confidence가 낮으면 기존 경로(LLM·사람)로 넘길 수 있다.
4. state를 코드가 미리 잘라 넣을 수 있다(원문 전체가 아니라 관련 필드).

맞지 않는 자리: 요약·번역·초안·KG 엔티티/엣지 추출·댓글 생성 등 **텍스트 생성**. 이들은 그대로 둔다.

## 3. 후보 지점 (우선순위순)

### A. 기존 원샷 분류 호출의 교체 — 지연·파싱 실패 제거

| 호출부 | 현재 | Jev 형태 | 폴백 |
|---|---|---|---|
| `task_routing_advisor` (`self_runtime/tools.py::_classify_route_with_llm`) | deepseek-flash, JSON 8키 | `choice` routing_class 7종 + `choice` recommended_agent(candidates) + `noul` needs_identifier. `reason`/`required_capabilities`/`forbidden_assumptions`는 문장 생성이므로 **Jev로 못 만든다** — 오케스트레이터 판단에 맡기거나(현재 폴백과 동일) 낮은 confidence일 때만 기존 LLM 호출 | 기존 LLM → 오케스트레이터 자체 판단 |
| `scout_kg_classify` (`kg_runtime/scout_ingest.py`) | gpt-5.6-luna, 답변 문자열에 group 포함 여부로 매칭 | `choice` over `KG_GROUP_IDS` 5종, criteria에 각 그룹 설명. confidence < τ → `agent_knowledge` | 현재와 동일 |
| `research_spelling_proofread` (`runtime_tools/research.py::_oneshot_spelling_verdict`) | gpt-5.6-luna, `{"revert":[번호]}` | 교정 항목별 `noul("이 교정은 문맥의 지시 대상과 다른 인물/개념을 가리킨다")` — 번호를 세게 하지 않으므로 Jev의 "counting 약함"을 피한다. 한국어 문맥 판정이라 **A군 중 가장 먼저 shadow 검증** 필요 | 기계 교정 유지(현재) |
| `kg_entity_classification` / `kg_type_assignment` (kg-maintenance scripts) | gemini-3.7-flash / haiku, 배치 JSON | 엔티티별 `choice` over typed schema 타입. 배치 fan-out으로 한 요청에 다수 | 미분류 유지 |
| `kg_entity_merge_grouping` (`skills/kg-maintenance/scripts/merge_entities.py`) | gemini-3.7-flash, 이름 묶기 | 후보 쌍별 `score` 3단계 [병합 / 별개 / 큐레이터 확인] — 공식 cookbook `entity_alignment` 패턴 그대로. 후보 쌍은 코드(정규화·임베딩 근접)가 만든다 | 큐레이터 확인 |

### B. 지금은 비용 때문에 안 하는 판정을 새로 추가 — 실질적 품질 이득

| 지점 | 새 판정 | 왜 지금 못 하나 |
|---|---|---|
| **CommuLingo 인용 지지 검사** (`commulingo_pipeline/evidence.py::locate_claim_quotes` 직후, `validate` 단계 전) | 주장(field, 값)마다 `choice` [supports / contradicts / unrelated] + `noul("인용문이 주장 값을 그대로 담고 있다")`. `unrelated`·`contradicts`가 confidence 높게 나오면 draft 전에 조사 단계로 되돌린다. 공식 cookbook `citation_check` 패턴 | 지금은 인용문의 **존재**만 기계로 확인하고(`locate`), 그 인용문이 주장을 **뒷받침하는지**는 비싼 독립 검토(review 단계)에서야 본다. Jev면 주장당 수백 토큰 ≈ $0.00002 |
| **독립 검토 전 사전 게이트** (`review` 단계 진입 전) | draft 전체에 `score` [바로 승인 가능 / 검토 필요 / 명백한 결함] + `noul`(출처 없는 수치·날짜 포함, 표기 규칙 위반 등 항목별) | 검토 LLM 비용을 줄이는 게 아니라, **명백한 결함은 검토 호출 전에 되돌려** 검토 회차(3회 실패→escalated)를 아끼는 용도. 최종 승인 권위는 계속 독립 검토 |
| **웹챗·A2A 입력/출력 스크리닝** (`services/web_chat.py`, `services/a2a_handler.py`) | 매 메시지 `score` 위험도 + `noul` 항목(프롬프트 주입 시도, 운영 정보 탈취 시도, 공개 부적절 발언) — cookbook `llm_guardrails` | 지금은 도구 allow-list·security_gateway로 **행위**만 막고 **내용**은 안 본다. LLM으로 매 메시지 스크리닝하면 답변 지연이 두 배. Jev는 100ms대. 단, Jev 자체가 주입에 취약하다고 문서가 인정하므로 **단독 차단 권위로 쓰지 않고** 높은 위험은 로그+답변 보수화, 중간은 기존 LLM 재확인 |
| **vector_search 결과 선별** (`runtime_tools` 코퍼스 검색 → 에이전트 컨텍스트 주입 전) | 청크별 `score` 관련성 3단계 — cookbook `classifying_rag_passages`/`rerank` | 지금은 임베딩 순위 그대로 넣는다. 러시아어 사료 청크 30개를 한 요청(≤64k)에 담아 선별 후 상위만 주입하면 에이전트 입력 토큰이 줄어 **에이전트 루프 비용이 실제로 준다** |
| **도구 호출 위험 게이트** (`tool_gateway.dispatcher.execute_tool` 앞, programmer/Codex `bash`·`edit_content` 등) | 인자 JSON에 `noul("파괴적/되돌리기 어려운 변경")`, `noul("작업 지시 범위를 벗어남")` — LangChain `AutoModeMiddleware` 패턴 | security_gateway는 **도구 이름·클래스**로만 판정한다. 인자 내용(`rm -rf`, 대량 UPDATE)은 안 본다. shadow 모드로 로그만 쌓고 enforce는 별도 승인 |
| **자율 프로젝트·에이전트 결과 사후 평가** (`jobs/autonomous_project.py`, `telegram/tasks.py` 완료 보고) | 보고서에 `noul` 항목: 수행하지 않은 행동을 했다고 주장, 도구 결과 없는 검증 주장, 사용자 지시 미이행 | `llm/execution_context.py`의 reality contract는 **프롬프트로만** 금지한다. 위반을 감지하는 판정이 없다. 표본 로그 → 사람 확인 흐름 |

### C. 보류

- `chunk_summary`, `conversation_reflection`, `experience_extraction`, 번역 계열, `kg_document_extraction`: 생성이라 대상 아님.
- Telegram 오케스트레이터의 "직접 답변 vs 위임" 판단: 현재 tool-loop 안에서 이미 이루어지고 Jev를 끼우면 매 턴 라운드트립이 하나 늘어 이득 불명확.

## 4. 통합 구조

원칙은 기존과 같다: **모든 호출은 `llm/call_registry` 경유, 실키는 `llm_proxy`만 보유, `llm/gateway.py` 감사 경유.**

### 4.1 프로바이더 등록

- `llm/call_registry.py::_PROVIDER_CONNECTIONS`에 세 항목: `typesafe`(`TYPESAFE_API_KEY`, `https://api.typesafe.ai`, `/v1/systemone`), `openrouter`(`OPENROUTER_API_KEY`, `https://openrouter.ai`, `/api/alpha/decisions`), `cloudflare_ai`(`CF_AI_API_TOKEN`, `https://api.cloudflare.com/client/v4/accounts/{id}/ai/run`, body를 `input`으로 감싸고 응답 `result`를 벗김). executor는 이 세 transport를 하나의 `Decision`으로 정규화한다.
- `llm_proxy/app.py` 라우트 표에 `typesafe` upstream 추가, systemd credential `typesafe_api_key`를 `leninbot-llm-proxy.service`에만 장착(`secret_management.md` 절차).
- `llm/gateway.py::estimate_cost_usd`에 `jev-*`: 입력 $0.042/M, 출력 0. `infer_provider`에 `jev` 접두.

### 4.2 새 executor — 텍스트가 아니라 판정을 돌려준다

기존 `generate()`는 `str | None`을 돌려주므로 Jev는 별도 진입점이 필요하다.

```python
# llm/call_registry.py (스케치)
@dataclass(frozen=True)
class Decision:
    answers: dict[str, dict]   # {"key": {"type": "choice", "choice": "...", "probabilities": {...}, "confidence": 0.93}, ...}
    model: str
    usage: dict

def decide_sync(feature: str, state, questions: dict, **defaults) -> Decision | None:
    """System One 판정. registry 항목의 provider는 typesafe여야 한다.
    실패·422·429 소진 시 None — 콜사이트가 자기 폴백(기존 LLM 경로·기본 라벨)을 유지한다."""

async def decide(feature, state, questions, **defaults) -> Decision | None: ...
```

- 요청 body는 `{"model": p.model, "state": state, "questions": questions}` 그대로. 질문 dict는 콜사이트가 `{"type":"choice","instructions":...,"criteria":{...}}` 형태로 만든다(SDK 없이도 HTTP만으로 충분하지만, SDK가 재시도·타입을 주므로 `requirements.txt`에 `typesafe-sdk` 추가 검토. 단 `base_url`을 프록시로 바꿀 수 있는지 확인 후 결정).
- `gateway.check_llm_call` → 호출 → `record_llm_call(input_tokens=usage.input_tokens, output_tokens=0)`.
- registry 항목 예: `"citation_support_check": {"provider":"typesafe","model":"jev-1.13.0","timeout":10,"managed":"executor","kind":"system_one","note":"..."}`. 문턱값을 튜닝한 뒤에는 `jev-latest`가 아니라 **버전 고정**(문서 권고). 응답 `model`을 감사 행에 기록.
- `scripts/llm_registry_cli.py list`가 `kind: system_one` 항목을 구분 표시.

### 4.3 confidence 라우팅 규칙

공식 `confidence-routing` 패턴대로 두 축으로 본다: 답이 **무엇**인지와 **행동해도 되는지**.

```
confidence ≥ hi  → 자동 적용
lo ≤ conf < hi   → 기존 LLM 경로로 재판정(또는 로그 후 보수적 기본값)
conf < lo        → 행동하지 않음, 현재 폴백
```

`hi/lo`는 콜사이트별 registry 항목(`thresholds`)에 두고 핫리로드. 초기값은 보수적으로(0.85/0.5) 두고 shadow 로그로 조정.

## 4.4 구현 상태 (2026-09-19)

- 완료: 프록시 라우트 `openrouter`·`typesafe`(둘 다 optional — credential은 드롭인으로 있을 때만 마운트),
  `call_registry` 연결 항목·`decide()/decide_sync()/decide_detailed()`, gateway `SYSTEM_ONE_PRICING`·provider 추론,
  registry 항목 `system_one_smoke`, `scripts/smoke_jev.py`, `tests/test_call_registry_decide.py`.
  `OPENROUTER_API_KEY`는 provider 키로 분류해 api/telegram 드롭인 대상에서 제외(다음 드롭인 재생성 시 반영).
- 첫 호출 결과(사용자 직접 실행, OpenRouter 새 키): HTTP 200, 396ms, 501 입력 토큰, $0.000021.
  한국어 교정 판정 noul 0.17/0.72(방향 정확), 라우팅 choice confidence 0.99. 응답 model `typesafe/jev-1.13-20260917`.
- 프록시 credential 교체 완료(2026-09-19 11:35 프록시 재시작, `/health` providers_without_key 없음). 모든 판정 호출은
  프록시 경유이며 감사 행은 oneshot(토큰·비용)과 proxy(전송) 두 줄이 남고 비용은 oneshot 행에만 있다.
- **직접 API 전환(2026-09-19 16:39)**: `TYPESAFE_API_KEY`를 credstore에 추가하고 드롭인 재생성·프록시 재시작. 프록시
  `typesafe` 라우트로 `GET /v1/models`(무료) 인증 확인 뒤 `system_one_smoke`를 `provider=typesafe, model=jev-1.13.0`으로
  바꿔 스모크 3/3 확인, 이어 Jev 항목 9개 전부 전환. 감사 행은 `provider=typesafe model=jev-1.13.0`(응답 model이 고정
  ID 그대로라 OpenRouter의 날짜 스냅샷 표기가 사라진다). OpenRouter 라우트·`OPENROUTER_API_KEY`는 예비로 남겨 두며,
  `provider`만 되돌리면 폴백된다. 비용 동일.
- **인용 게이트 `partially_supports` 옵션(2026-09-19 저녁)**: 실데이터에서 unrelated 판정 대부분이 긴 복합 claim의 "일부 지지"
  (conf <0.65)였고, Choice는 상대 분포라 그 확률이 unrelated로 샜다(공식 jaggedness "구조적 불변식 없음"·"문자 그대로 읽음").
  선택지를 명명해 통과·기록 결과로 두고 unrelated는 "어느 사실도 다루지 않음"으로 좁혔다. 조사·검토 게이트 모두 적용. 30쌍
  표본 재측정은 보류(표본이 스크립트화돼 있지 않음) — `citation_check` 분포에서 partially_supports 비율과 unrelated 고신뢰
  건수를 관찰한다. 같은 문서 검토에서 나온 다음 후보(미착수): 인물 분류 years를 코드가 시대 버킷(`age_in_1917/1953`,
  기간별 성인 연수)으로 변환해 state에 넣기(jaggedness "날짜 비교는 코드에서"; 시대 경계 불일치 54건의 원인 후보 —
  `logs/commulingo/person_classification_audit_2026-09-19.json`의 `group_boundary` id 목록으로 재측정 가능), 같은 발췌 공유 claim의
  1요청 fan-out, KG 임베딩 후보쌍의 3단계 Score 정렬(entity_alignment 쿡북).
- 재시도: 429·5xx·연결 거부/끊김은 `decide_detailed`가 한 번 더 시도한다(Retry-After 존중, 최대 2초 대기; 항목의
  `retries`로 조정, 0이면 없음; 읽기 타임아웃은 제외). 실패 행은 시도마다 남는다. 이유 — None 한 번의 대가가 크다(게이트는 미검사 통과,
  라우팅은 4배 느린 DeepSeek 폴백).

## 4.5 실데이터 기준선 비교 (2026-09-19, 총 $0.015)

같은 표본에 기존 파이프라인·Jev·정답을 나란히 놓았다. 정답은 원문을 직접 읽어 매김.

**A. 인용 지지 — 조사 단계가 통과시킨 claim↔인용 발췌 30쌍** (S 핵심 사실 뒷받침 / P 일부 절만 / N 무관)

| 정답 | n | 기존 조사 단계 | 하류 독립 검토 | Jev (conf ≥0.85만 행동) |
|---|---|---|---|---|
| S | 20 | 20/20 통과 | 승인 | 19/20 supports (1건 unrelated, conf 0.29 → 유보) |
| N | 5 | 0/5 검출 | 0/5 검출(전부 approve) | 5/5 unrelated; 3건 conf ≥0.85(즉시 차단), 2건 저신뢰(유보) |
| P | 5 | 5/5 통과 | 승인 | 1 supports, 4 unrelated/contradicts — 전부 conf <0.65(유보) |
| 대조군(섞은 쌍) | 29 | — | — | 29/29 unrelated, 오인 0 |

S+N 25건 정확도: 기존 20/25(80%) vs Jev 24/25(96%); 행동 기준 오탐 0, 고신뢰 미탐 0, 저신뢰 미탐 2.
N 5건의 실체: Britannica 봇 확인 페이지(acmeism), 저자 명단 주장↔인터뷰 단락(500-day-program),
1933년 경력↔미학 이론 단락(lunacharsky), 득표수 주장↔숫자 없는 단락(maria-spiridonova), 종결연도↔시기구분(pink-tide).
독립 검토는 자기 인용을 새로 뽑아 사실을 확인하므로 발행 내용은 맞았을 수 있으나, 편집에 저장된 '근거'가 엉뚱한 채 남았고
이를 잡는 단계가 없었다.

**B. 태스크 라우팅 — 실제 위임 태스크 30건, 정답 = 실제 배정 에이전트**

| | 기존 `task_routing_advisor` (deepseek-flash) | Jev |
|---|---|---|
| 정확도 | 26/30 (87%) | 28/30 (93%) |
| 오답 | diary→programmer ×2, programmer→analyst, scout→browser | programmer→analyst ×2 (1건 conf 0.41) |
| 평균 지연 | 1,556 ms | 372 ms |
| 30건 비용 | $0.0082 | $0.0030 |
| 자신감 표시 | 텍스트 "medium"(전부 동일) | 수치 confidence |

둘이 같이 틀린 1건("정치노선 보강" = 프롬프트 파일 수정)은 criteria에 규칙을 적으면 잡히는 유형(4.7에서 반영). 표본이 30건이라 shadow 대신 **confidence 게이트 + DeepSeek 폴백**으로 배포하고(4.7), 감사 로그로 일주일 관찰한다.

## 4.6 인용 지지 게이트 (구현 2026-09-19)

`commulingo_pipeline/citation_gate.py`, registry 항목 `commulingo_citation_support`. 조사 단계 `finish`에서
`resolve_passages` 직후 이번 호출의 claim마다 Jev에 `{field, claim, source_url, excerpt}`를 주고
`support` choice + `specific`·`boilerplate` noul을 받는다(동시 8, claim당 ~$0.00002).
`unrelated/contradicts` conf ≥ `thresholds.reject`(0.85) 또는 `boilerplate` ≥ 0.9면 결과 호출을 거절하고
해당 claim만 지목한 메시지("다른 인용·다른 출처·claim 삭제")를 돌려준다. 판정 수치(support·confidence·specific·
boilerplate)는 각 claim에 `citation_check`로 붙어 artifact에 저장된다 — claim과 함께 움직이므로 targeted research의
carried claims와 섞여도 어긋나지 않고, 거절 문구·모델명은 붙이지 않는다(research artifact가 draft 프롬프트로 전달되므로
shadow 모드의 "claim 삭제" 문구가 작성기를 흔들면 안 된다). 결과 핸들러당 판정 캐시가 있어 고치지 않고 재제출한 claim은
다시 판정하지 않는다(거절 메시지의 "나머지는 그대로 재제출 가능"을 코드가 보장). `stance: disputes` claim은 반박 출처를
인용하는 것이 정상이므로 `contradicts`는 통과, `unrelated`만 결함. `enforce=false`면 기록만(shadow), `enabled=false`면
호출 없음, 판정 불가(None)는 통과+`citation_unavailable` 집계. **배포 상태: `enforce=true`** — 기준선에서 오탐 0이었기
때문. 위 30쌍에 실행하면 정확히 N 3건(고신뢰)만 거절되고 Britannica 페이지는 boilerplate로 잡힌다.

## 4.7 태스크 라우팅 1차 분류기 (구현 2026-09-19)

`self_runtime/tools.py::_classify_route` — `route_task` 도구가 부르는 분류기. 먼저 Jev(registry `task_routing_decision`)에
`{task}`를 주고 `agent` choice(criteria는 `_AGENT_ROUTING_CARDS`에서 생성 + "정치노선·페르소나·프롬프트 텍스트는 코드 저장소
파일이므로 programmer" 규칙), `routing_class` choice(7종), `needs_identifier` noul을 받는다. agent confidence ≥
`thresholds.accept`(0.85)면 그 결과를 기존과 같은 형태(`reason`은 확률 판독문, 생성 문장 아님; `confidence_score` 추가,
`source=jev_classifier`)로 돌려주고, 미만이면 기존 `task_routing_advisor`(DeepSeek)를 부르되 Jev 판독을 `system_one_hint`로
첨부한다. 둘 다 실패하면 저신뢰 Jev 결과를 `low`로 표시해 돌려준다. `route_task` 응답 `classifier.engine`이 `jev`/`llm`/
`jev_low_confidence`를 말한다. `enabled=false`면 바로 DeepSeek.

같은 30건 재실행: 28/30, Jev 24건·폴백 6건. 오답 2건은 **폴백 DeepSeek**의 답이었고 그때 Jev 판독은 각각 programmer 0.82(정답)·
diplomat 0.65(오답)였다 — 문턱값 0.80이면 29/30. 보수적으로 0.85를 두고 `llm_audit_log`의 두 caller 비율과 hint 불일치를
일주일 보고 조정한다.

## 4.8 독립 검토 인용 게이트 — shadow (구현 2026-09-19)

검토자의 `checks[].quote`는 `resolve_review_checks`가 표시된 문단 라벨로 원문에서 붙이지만, 그 문단이 `finding`(이 인용이 무엇을
확인하는지 적은 한국어 문장)을 실제로 담는지는 아무도 보지 않았다. `citation_gate.check_review_checks`가 조사 게이트와
같은 질문(finding↔quote)을 Jev(registry `commulingo_review_citation_support`)에 묻고, 판정은 각 check의 `citation_check`와
review artifact metrics의 `review_citation_*`에 남는다. 훅은 `make_handlers(..., gate=review_gate(usage))` — 결정이 box에 들어가기 전에
돌아 enforce로 바꾸면 그 check만 지목한 ToolRejection으로 검토자에게 되돌아간다. 파이프라인 검토와 검토 타이머의
독립 검토 양쪽에 걸리고, 승인 메모의 checks에서는 판정 수치를 뺀다. **배포 상태: `enforce=true`**(2026-09-19 사용자 지시, 기준선 오탐 0).

**기준선 — 저장된 검토 결정 40건에서 check 하나씩 무작위 추출(최근 21일), 정답은 finding과 quote를 직접 대조**

| 정답 | n | 검토 결정 | Jev |
|---|---|---|---|
| S 인용이 finding을 뒷받침 | 37 | approve 35 · revise 2 | 37/37 supports (conf 0.72~1.0, 0.85 미만 6건) |
| P 일부만 | 1 | approve | supports 0.59 (인용이 앞뒤 잘려 몰로토프 절이 빠짐) |
| N 무관 | 2 | revise 1 · approve 1 | 2/2 unrelated (conf 0.98 · 0.83) — enforce였다면 1건 즉시 거절, 1건 유보 |

오탐 0. 40건 $0.0016, 평균 360ms. N 2건의 실체: 코시긴 사망일 회고를 확인한다는 finding에 1941년 철도 위원회 문장을
인용(revise 결정 안), 피우수트스키의 '폴란드 부흥·사나차 실권자' 서술을 확인한다며 1932년 단치히 구축함 사건 문단을
인용(approve 결정 안). 검토자 인용 40건 중 2건(5%)이 비어 있었으므로 조사 게이트(30건 중 5건)보다 낮지만 0은 아니다.
`review_citation_rejections`가 검토 회차를 눈에 띄게 늘리면 문턱값을 올린다.

## 4.9 기각 — `scout_kg_classify` (평가 2026-09-19)

A군 후보였으나 실제 scout 완료 태스크 30건(최근 90일 202건에서 무작위)을 보니 **전부 메일함 브리핑**이라 분류는
사실상 agent_knowledge/economy 이분법이고 정답 자체가 모호하다(AI 뉴스레터 요약을 economy로 볼지).

| | 현재 `scout_kg_classify` (gpt-5.6-luna) | Jev |
|---|---|---|
| 둘의 일치 | 23/30 | |
| 불일치 7건 | 판정 가능한 2건: 진행 메모→economy(오답 1) | agent_knowledge(정답 1) · 신규 메일 없음→economy 0.50(오답 1, 문턱값 미만이라 폴백됐을 것) · 나머지 5건은 정답 모호 |
| 평균 지연 | 2,275 ms | 360 ms |
| 30건 비용 | $0.0035 | $0.0010 |

월 38회 호출이라 지연·비용 차이가 의미 없고 정확도 이득을 잴 수 없어 교체하지 않는다. 다시 볼 조건: scout가 메일
브리핑 외 정찰(뉴스·SNS 모니터링)을 실제로 맡아 그룹 분류가 갈리기 시작할 때.

## 4.10 scout→KG 사실 필터 (구현 2026-09-19)

4.9에서 scout 보고서 표본을 보다가 다른 문제가 보였다: `process_scout_report_to_kg`는 findings 절의 줄(불릿, 또는 ':'가 있는
20자 이상 줄)을 그대로 "사실"로 KG episode에 쓴다. 실제 scout 보고서 30건에서 추출된 100줄을 직접 라벨링하니 **92줄이
메일함 UID·처리 절차·추론 메모**("INBOX 최고 UID 345로 변동 없음", "Let me reconsider…")였고 전부 KG에 저장되고 있었다.

`_filter_fact_lines`(registry `scout_kg_fact_filter`)가 보고서당 한 요청으로 줄마다 noul("외부 세계에 대한 사실이며 에이전트
자신의 절차 메모가 아니다")을 묻는다. `thresholds.keep`(0.8) 이상만 최대 7줄 저장, 남는 줄이 없으면 episode를 쓰지 않는다.
그룹 분류기(`scout_kg_classify`)는 남은 사실만 본다. 판정 불가 시 전부 유지(기존 동작).

| 정답(직접 라벨) | n | Jev P(사실) — 낮을수록 "절차 메모" 판정 |
|---|---|---|
| 절차·메일함 상태·추론 메모·빈 줄 | 78 | 전부 ≤0.11 |
| 뉴스레터 제목 표 행(제목만, 사실 진술 아님) | 14 | 0.09~0.83 (0.8 이상 1건) |
| 뉴스 사실·사실 헤드라인 | 8 | 6건 ≥0.85, 헤드라인 0.80·문맥 줄 0.69 |

문턱값 0.8: 절차 줄 오저장 0, 사실 8건 중 7건 유지. 30건 보고서 중 사실이 하나라도 남는 것은 6건 — 나머지 24건은 episode를
쓰지 않게 된다. 100줄 판정 비용 $0.0011(보고서당 ~$0.00004).

## 4.11 평가 — CommuLingo 인물 분류 필드 자동 채우기 (2026-09-19, 미채택·후속 결정 대기)

등록 API의 닫힌 집합 필드(`groupId` 9종, `role.category` 10종/`role.officeId` 16종, `fate.kind` 9종, `citizenship.code`·
`nationalOrigin.code` 각 ~190종)를 작성 모델 대신 Jev가 채울 수 있는지, 저장된 인물 60명(무작위)의 bio_ko/bio_en·years·
epithet만 state로 주고 저장값과 비교했다. 인물당 ~9k 토큰(대부분 criteria), 60명 $0.023.

| 필드 | 저장값과 일치 | conf ≥0.85에서 일치 | 판단 |
|---|---|---|---|
| citizenship.code | 57/60 | 49/52 | 채울 수 있음. 불일치는 망명자(soviet↔russia)·체코슬로바키아↔체코 같은 경계 사례 |
| nationalOrigin.code | 56/60 | 52/55 | 채울 수 있음. 불일치는 bio에 출신이 없을 때(belarus→russia) |
| groupId | 45/60 | 36/39 | bio만으로는 부족. 시대 경계(bolshevik↔stalin-era↔thaw)는 편집 판단이고, "scholar"(이 역사를 연구한 사람) 기준을 과학자에게 잘못 적용 |
| role | 36/60 | 25/34 | bio만으로는 부족. office↔category 선택과 party-leadership↔party-secretariat-cadres 같은 경계가 편집 판단 |
| fate.kind | 37/60 | 24/27 | bio에 사망 경위가 없으면 unconfirmed → 저장값 natural과 어긋남. 조사 claim이 필요한 사실 |

단, 고신뢰 불일치에는 **저장값이 틀린 것으로 보이는 사례**가 섞여 있다: 덩화(중국 인민해방군 장성)가 `stalin-era`(Jev:
international-revolutionary 0.94), 이반 파블로프(생리학자)가 `theorist`(Jev: scholar 0.96), 앙드레 마르티가 `non-soviet-revolutionary`
(Jev: comintern 1.0 — 코민테른 서기국원). 자동 채우기보다 **기존 2,341명 분류 감사**(고신뢰 불일치 목록을 큐레이터에게)와
**신규 등록 초안의 분류 불일치 플래그**(작성 모델의 group/role과 Jev 고신뢰 판정이 다르면 검토자에게 확인 요청)가 먼저다.
라벨(`label.ko/en`)은 혼혈·복수 배경을 보존하는 자유 서술이라 코드에서 파생할 수 없고, 코드만 Jev가 채우는 형태가 된다.
`group`·`role`은 update에서 잠겨 있어 실제로 채워야 하는 것은 월 ~75건의 person create뿐이다.

### 4.11.1 전수 감사 실행 (2026-09-19, $0.31)

사용자 지시로 2,341명 전원에 group·role을 판정했다(state에 국적 코드·경력 8건 추가, criteria에 "scholar는 이 역사의
연구자이지 과학자가 아님" 등 규칙 보강; 비소련 인물은 카테고리만 — 첫 실행에서 외국 장군·대통령·동구권 지도자에게 소련
관직을 매긴 오류가 191건이었다). 일치 group 1,932 · role 1,617. conf ≥0.85 불일치 group 137 · role 215를 여덟 묶음으로
나눈 보고서: `logs/commulingo/person_classification_audit_2026-09-19.md`(+ `.json` id 목록, 서버 로컬).

| 묶음 | n | 성격 |
|---|---|---|
| 비소련 국적 + 소련 관직 | 15 (+후계국 국적 8) | Jev 없이 확정되는 규칙 위반 |
| 타국 공산당·사회주의권 지도자가 소련 시대 그룹 | 49 | 정책 결정(그룹 정의대로면 일괄 이동) |
| 그룹 오류 가능성 높음 | 18 | 체르니솁스키·플레하노프가 international-revolutionary, 마니우·미하이 1세·시아누크가 혁명가, 도이처·파이프스가 비연구자 |
| old-regime ↔ 국외 반혁명(백군) | 16 | 감사 criteria 편향, 정책 확인만 |
| 시대 경계 | 54 | 편집 판단, 낮은 우선순위 |
| 역할 오류 가능성 높음 | 42 | 카우츠키·베른슈타인·루카치가 theorist 아님, 장군·물리학자가 theorist, 심리학자·의사가 science-nuclear-space |
| 관직 간 경계 | 89 | 공화국 제1서기의 nationalities-federal ↔ party-leadership |
| 그 밖의 카테고리 겹침 | 77 | 사회주의권 지도자↔비소련 혁명가 등 |

**적용 결과와 확정 규칙 (2026-09-19, 운영자 승인·결정).** 편집 서비스 제출→승인으로 104명 정정(규칙 위반 12, 비소련 인물 그룹 이동
63, 그룹 오류 12, 역할 오류 26, 사하로프 관직 1, 겐딘·이바슈틴 방첩→state-security 2; 보고서 9절). 운영자가 확정한 규칙은 재사용 감사 스크립트 `scripts/commulingo_classification_audit.py`(registry `commulingo_classification_audit`)의 criteria에 들어 있고, 인정된 경계 쌍은 `ACCEPTED_PAIRS`로 보고에서 빠진다. 규칙:
공화국 제1서기·공화국 정부 수반은 `nationalities-federal`, 지방·주 서기와 콤소몰·중앙위 서기는 `party-secretariat-cadres`(Jev의
`party-leadership` 판정은 이 규칙을 모르는 것); `ideology-propaganda`는 친소련 이데올로그 관직(반체제 출판 활동은 해당 없음);
레닌은 `party-leadership`; `head-of-government`·카메네프 comintern·톰스키/루주타크 economic-management·에이헤 agriculture는
애매하므로 Jev 불일치만으로 옮기지 않는다. 비소련 국적 인물은 소련 시대 그룹에 두지 않는다(그룹 정의대로).

**등록 시 자동 배정으로 전환 (2026-09-19, 사용자 결정).** 감사 후 사용자가 "등록할 때부터 Jev가 채우게" 하라고 해서
`runtime_tools/commulingo_classify.py::classify_person`을 초안 단계와 create 도구에 넣었다: 작성 모델은 groupId·role을 내지 않고
실행기가 확정 규칙이 든 criteria로 판정해 채운다(관직은 소련·후계국 국적에만, accept 0.7 미만은 reviewFlags로 검토자 확인, 판정
불가면 작성 모델에 직접 지정 요구). 라이브 확인: 카다르 → international-revolutionary 1.0 / bloc-reform-leader 0.47(플래그),
쿠나예프 → thaw 0.9 / nationalities-federal 0.96. 감사 스크립트는 같은 criteria를 import한다.

**용어 category (2026-09-19).** 같은 방식으로 `classify_term`을 용어 create(월 2,924건)에 넣었다. 저장 용어 1,086건 기준선 —
라벨만 813·지역 우선 규칙 746·주제 우선 규칙 813(채택; conf ≥0.85 628/714, ≥0.95 521/565). 저장값이 작성 모델의 선택이라
서로 모순되는 사례가 많아(포옹 전술=당·국가 기구, RD-107 엔진=현대 자본주의) 75%는 관행 재현율이지 정확도 상한이 아니다.
accept 0.7 미만이고 작성 모델이 값을 줬으면 그 값을 남긴다. 라이브: 프로드날로크 → economy, 헬싱키 최종의정서 → international.

**등록 API의 코드 필드 (2026-09-19).** `citizenship.code`·`fate.kind`를 `classify_person_codes`가 라벨 + 조사 claim 발췌로 채운다.
기준선(최근 인물 초안 60건, 정답 = 작성 모델이 고른 값): citizenship 50/50, fate 32/35(conf ≥0.7 30/33; 불일치에는 claim 없는
초안의 natural을 Jev가 unconfirmed로 본 건이 있어 Jev 쪽이 맞음), nationalOrigin은 규칙 없이 42/49(유대계→israel)였으나
사용자 지적대로 출신 규칙을 instructions에 적자 48/49(conf ≥0.7 46/46)가 되어 **채택**. 생존 인물 fate는 호출 없이 빈 kind.
등록 API의 닫힌 집합 필드(groupId·role·category·citizenship.code·nationalOrigin.code·fate.kind)는 이제 전부 실행기가 채운다.

## 4.12 남은 후보와 관찰 포인트

- 조사 게이트 실데이터 첫 7건(12:06~12:35): 173 claim, unrelated 4건 전부 conf 0.12~0.62의 "일부 지지"(긴 복합 문장의
  절 하나만 인용) → 거절 0. 기준선의 P군과 같은 양상. 복합 claim을 절 단위로 나누게 하거나 `partial` 선택지를 두는
  것은 기준선을 다시 재야 하므로 보류.
- 검토 사전 게이트(3절 B): 최근 14일 revise 사유를 읽으면 전부 "핵심 사실 대부분 검증, 그러나 X가 출처와 충돌"류라
  초안만 보는 판정으로는 못 잡는다. 기각.
- `research_spelling_proofread`: 최근 30일 호출 0건. 보류.
- 남음: vector_search 청크 선별(러시아어 청크 관련성 기준선 필요), 웹챗·A2A 스크리닝(shadow), 도구 인자 위험 게이트
  (shadow), Telegram 태스크 완료 검증 보조. 각각 같은 형식(표본·정답·기존·Jev)의 기준선 먼저.

## 4.13 tool gateway 로그 검토 (2026-09-19~20)

14일 `tool_audit_log` 43,951행: 도구 호출의 94%가 CommuLingo 파이프라인(curator 26.8k·reviewer 13.8k), `fetch_url` 9,750·
`wiki_get` 6,619·`web_search` 4,100. 거절·오류 약 4,000건(9%)은 대부분 schema·프롬프트 문제(`'fields' is a required property`,
`'notes'` 불허, em dash 66건/3일, `placeholder` reason)라 Jev 대상이 아니다.

- **검토 인용문 거절(하루 159건, 09-19)** — 검토 레인의 접기 매칭에 접두 폴백이 없고 오류가 check 번호를 안 알려 결정 전체가 4~5회
  재제출됐다(Postyshev 9개 check 중 2개만 exact 실패, 40자 접두면 통과). 코드 버그였고, 이어 운영자 결정으로 **복사·매칭 자체를 폐기**:
  표시할 때 문단마다 라벨(`S2@12303`/`R…@offset`)을 붙이고 claim·check는 라벨만 인용한다(`evidence.label_passages`·`resolve_passages`,
  `review_policy.review_source`·`resolve_review_checks`). Jev도 문자열도 필요 없다. 그 전에 잰 "Jev 근거 위치 판정" 기준선(저장 claim 40·
  check 40, 창 ~1,500자, 대조군 = 같은 job의 다른 페이지, 2회 $0.16): 정답 창 top-1 73~100%·top-3 95~100%, 정답 P 중앙값 0.93~0.95,
  오답 페이지 오채택(P≥0.7) 10~25%, 요청당 352ms — 결과는 `logs/commulingo/evidence_locator_baseline_2026-09-19*.json`(로컬). 라벨 방식이
  채택돼 구현하지 않았다.
- **페이징 사냥** — fetch 호출의 21%·wiki_get의 24%가 offset>0, 같은 scope·URL 3페이지 이상이 862건(fetch 호출의 40%, 회당 4.7s + LLM 라운드),
  인용 발췌의 15%가 10,000자 이후. 후보(미착수): 첫 fetch 때 저장된 전체 본문을 창으로 잘라 Jev에 "대상·조사 필드에 관한 사실을 담는가"를
  묻고 top-k offset 색인을 도구 결과에 덧붙이기(본문은 숨기지 않음). 기준선은 위 결과가 대용(진술 없이 대상만 줄 때는 재측정 필요).
- **검색 결과 선별(shadow, 2026-09-20 적용)** — 검토당 fetch URL 5.8개 중 인용 3.1개(47% 미인용), 검색은 제공자 순서 그대로 표시.
  `commulingo_pipeline/search_triage.py`(registry `commulingo_search_triage`): 조사 `wrap`·검토 `make_handlers(triage=)`가 `web_search`
  결과의 hit(제목·URL·요약, ≤10)을 한 요청으로 Jev에 주고 hit마다 choice [directly/possibly/unrelated] "대상 항목을 직접 다루는가"를 받아
  usage tracker `search_triage`(artifact metrics)에 `{url, verdict, confidence}`로만 기록한다 — 표시는 그대로. 판정 실패·예외는 검색을
  건드리지 않는다(`search_triage_unavailable`). 1주 뒤 평가 쿼리: research/review artifact metrics의 search_triage를 같은 scope의
  `tool_audit_log` fetch_url URL과 artifact의 인용 URL(claims→sources, checks[].source)에 join해 verdict별 fetch율·인용율 표. directly의
  인용율이 unrelated의 2배 이상이면 렌더링에 표기·정렬을 넣고, 아니면 끈다.
- 로그상 볼륨이 없어 미룸: discover 중복 정렬(이름 포함 쌍 10건, create 23건/14일), 웹챗 스크리닝(web_search 158/14일), vector_search 선별(54),
  도구 인자 게이트(execute 2), task_verifier(주 293라운드 $0.32).

## 5. 롤아웃 단계 (각 단계 시작 전 승인)

**0. 계정·키·shadow 평가 (지출 발생 — 승인 필요).** console.typesafe.ai 가입, 키를 프록시 credential로 설치. 위 4.1~4.2 구현. 그 다음 **실제 동작은 바꾸지 않고** 다음 세 곳에서 기존 호출과 병행 실행해 일치율을 `llm_audit_log`·journald에 남긴다:
   - `research_spelling_proofread` (한국어 문맥 판정 — CJK 성능의 리트머스)
   - `scout_kg_classify` (영어 위주, 기준선)
   - CommuLingo `locate_claim_quotes` 직후 인용 지지 검사(러시아어/영어 출처 + 한국어 주장 — 다국어 혼합)
   한 건에 수백~수천 토큰이므로 일주일 shadow가 **$1 미만**으로 예상. 일치율과 confidence-정확도 곡선을 본 뒤 계속 여부 결정.

**1. A군 교체.** shadow 일치율이 기존 LLM 대비 동등 이상인 호출부만 confidence 라우팅으로 전환. 낮은 confidence는 기존 LLM. 각 전환은 registry 항목 하나 바꾸는 일이므로 되돌리기도 항목 하나.

**2. B군 신설 게이트 — shadow 먼저.** 인용 지지 검사 → 웹챗 스크리닝 → vector_search 선별 → 도구 위험 게이트 순. 각각 로그만 쌓는 shadow 기간을 두고, enforce는 security_gateway와 같은 `gateway_enforce_mode` 식 스위치로 별도 승인.

**3. 문서 흡수.** `llm_call_registry.md`(decide 진입점·kind·재시도), `llm_gateway.md`(프록시 라우트·요율), `commulingo_pipeline.md`(두 게이트)는
갱신됨. 이 문서는 후보·기준선 보관용으로 유지한다.

## 6. 하지 말 것

- Jev 답을 **쓰기 권위**로 쓰지 않는다. CommuLingo 공개 저장의 최종 승인은 독립 검토, 도구 차단의 권위는 security_gateway 정책이다. Jev는 앞단 선별과 낮은 confidence 에스컬레이션.
- 생성 호출을 Jev로 바꾸려 하지 않는다(reason 문장, 요약 등).
- state에 원문 전체를 넣지 않는다. 인용문+주장, 인자 JSON, 메시지 한 건처럼 코드가 잘라 넣는다.
- 부정형 질문("~가 아닌가?")을 피하고 긍정형 서술 + `criteria.true/false`로 쓴다. 세기·날짜 비교를 시키지 않는다(항목별 `noul`로 풀어 쓴다).
- 사용자 입력·외부 문서를 state로 넣는 게이트(웹챗·인용 검사)는 주입 가능성을 전제로 설계한다 — 높은 위험 판정도 "차단"이 아니라 "보수화+기록", 낮은 위험 판정이 통과 근거가 되지 않는다.

## 7. 참고

- OpenRouter: https://openrouter.ai/typesafe , Decisions SDK 문서 https://openrouter.ai/docs/client-sdks/go/sdks/decisions/README
- Cloudflare Workers AI 모델 페이지: https://developers.cloudflare.com/ai/models/typesafe/jev/
- Vercel: https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway
- 문서 인덱스: https://docs.typesafe.ai/llms.txt (HTTP API `api.md`, 약점 `model-jaggedness/jev-1.13.md`, 패턴 `patterns/*.md`, cookbook `cookbooks/citation_check.md`·`llm_guardrails.md`·`classifying_rag_passages.md`·`entity_alignment.md`·`skill_suggestion.md`)
- 개발용 skill: `claude plugin marketplace add typesafe-ai/skills && claude plugin install typesafe@typesafe-ai` (Leninbot 런타임 `skills/`와 무관한 Codex/Claude Code용)
- 발표 글: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- LangChain 하네스 예(`langchain_typesafe` ModelRouterMiddleware·AutoModeMiddleware): https://www.langchain.com/blog/building-a-harness-with-jev
