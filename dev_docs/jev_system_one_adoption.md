# Jev (TypeSafe System One) 도입 계획

작성 2026-09-19. 아직 구현·계약·키 발급 전인 **계획 문서**다. 구현이 끝나면 내용은
`llm_call_registry.md`·`llm_gateway.md`와 각 도메인 문서로 흡수하고 이 파일은 삭제한다.

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

## 1.1 접근 경로 — 직접 API는 대기열, 호스팅 경로 셋

직접 API(`console.typesafe.ai`)는 2026-09-19 기준 early access 대기열이다(사용자 신청 2026-09-19).
대기열 없이 같은 모델을 쓰는 경로가 셋 있고, 조사 결과는 다음과 같다.

| 경로 | 호출 | 모델 ID | 가격 | 컨텍스트 | 비고 |
|---|---|---|---|---|---|
| **OpenRouter Decisions** (2026-09-18 beta 추가) | `POST https://openrouter.ai/api/alpha/decisions`, `Authorization: Bearer <OPENROUTER_API_KEY>`, body `{model, state, questions}` — **native와 거의 동일한 wire format** | `typesafe/jev-1.13`(버전 고정 가능), `~typesafe/jev-latest` | $0.042/M 입력, 출력 $0 (native와 같음) | 32k | alpha 경로라 스키마 변경 가능. `instructions`/`criteria` 값을 문자열로만 검증(구조화 값은 JSON 문자열로). OpenRouter 계정+크레딧 선충전 필요, 대기열 없음 |
| **Cloudflare Workers AI** | `POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run`, body `{"model":"typesafe/jev","input":{state,questions}}`, 응답은 Cloudflare 봉투 `{result,success,errors}` 안 | `typesafe/jev` (항상 최신 alias — **버전 고정 불가**) | 문서에 미기재("대시보드에서 확인"), Workers AI 일반 규칙은 무료 10k neurons/일 + $0.011/1k neurons | 32k | 프로젝트에 **Cloudflare 계정이 이미 있다**(R2·캐시 퍼지, `R2_CF_ACCOUNT_ID`). 단 기존 `R2_CF_API_TOKEN`은 R2 범위일 것이므로 Workers AI 권한의 **별도 토큰** 발급 |
| Vercel AI Gateway | AI SDK 7 `experimental_evaluate` (TypeScript) 중심, 순수 HTTP 형식은 공개 문서에서 확인 못 함 | `typesafe-ai/jev` | 미확인 | — | Python 런타임과 맞지 않아 보류 |

기타: AI/ML API가 제공한다는 언급이 있으나 미확인. 비공식 파이썬 클라이언트 `jevclient`, 공식
`system-one-adapter`(LLM으로 Jev 흉내내는 비교용 어댑터)가 있으나 우리는 registry executor로 직접 HTTP를 친다.

**권장: shadow 평가는 OpenRouter로 시작.** 이유 — (1) native와 body·응답이 같아 `decide()` executor를
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
- 대기: credstore의 `openrouter_api_key.cred`가 401 나는 옛 키라 사용자가 새 키로 교체해야 한다(`systemd-creds encrypt`,
  sudo). 교체 → 유닛 재설치·daemon-reload → 프록시 재시작 뒤 `scripts/smoke_jev.py`가 프록시 경유로 동작한다.

## 5. 롤아웃 단계 (각 단계 시작 전 승인)

**0. 계정·키·shadow 평가 (지출 발생 — 승인 필요).** console.typesafe.ai 가입, 키를 프록시 credential로 설치. 위 4.1~4.2 구현. 그 다음 **실제 동작은 바꾸지 않고** 다음 세 곳에서 기존 호출과 병행 실행해 일치율을 `llm_audit_log`·journald에 남긴다:
   - `research_spelling_proofread` (한국어 문맥 판정 — CJK 성능의 리트머스)
   - `scout_kg_classify` (영어 위주, 기준선)
   - CommuLingo `locate_claim_quotes` 직후 인용 지지 검사(러시아어/영어 출처 + 한국어 주장 — 다국어 혼합)
   한 건에 수백~수천 토큰이므로 일주일 shadow가 **$1 미만**으로 예상. 일치율과 confidence-정확도 곡선을 본 뒤 계속 여부 결정.

**1. A군 교체.** shadow 일치율이 기존 LLM 대비 동등 이상인 호출부만 confidence 라우팅으로 전환. 낮은 confidence는 기존 LLM. 각 전환은 registry 항목 하나 바꾸는 일이므로 되돌리기도 항목 하나.

**2. B군 신설 게이트 — shadow 먼저.** 인용 지지 검사 → 웹챗 스크리닝 → vector_search 선별 → 도구 위험 게이트 순. 각각 로그만 쌓는 shadow 기간을 두고, enforce는 security_gateway와 같은 `gateway_enforce_mode` 식 스위치로 별도 승인.

**3. 문서 흡수.** `llm_call_registry.md`(decide 진입점·kind), `llm_gateway.md`(프록시 라우트·요율), `commulingo_pipeline.md`/`security_gateway.md`/`web_research.md`(각 게이트) 갱신 후 이 문서 삭제.

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
