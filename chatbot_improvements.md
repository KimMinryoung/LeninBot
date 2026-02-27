# `chatbot.py` 개선 제안

## 핵심 개선 포인트

1. **초기화/설정 안정성 강화**
   - 현재 `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `GEMINI_API_KEY`가 없더라도 즉시 명확히 실패하지 않습니다.
   - 앱 시작 시 환경변수 검증 함수를 두고, 누락 시 어떤 키가 빠졌는지 명확한 에러를 주는 것이 좋습니다.
   - 또한 `create_client()` 실패를 별도 처리해 재시도/종료 정책을 분리하면 운영 안정성이 올라갑니다.

2. **상태 타입 불일치 정리 (`datasource`)**
   - `AgentState.datasource` 타입은 `"vectorstore" | "generate"`인데, 실제 런타임에서는 `"plan"`도 사용합니다.
   - 타입 정의와 실제 라우팅 값이 불일치하면 정적 검사·가독성 모두 악화됩니다.
   - `Literal["vectorstore", "generate", "plan"]`로 타입을 맞추는 것을 권장합니다.

3. **JSON 추출 로직 강건화**
   - `_extract_json()`의 `\{[^{}]*\}` 패턴은 중첩 객체를 파싱하지 못해 구조화 응답 실패 가능성이 있습니다.
   - 가능하면 정규식보다:
     - 코드블록 우선 추출,
     - 실패 시 `json.JSONDecoder().raw_decode()` 기반 first-object 파싱,
     - 마지막 fallback으로 모델 재질문
     순서로 바꾸는 것이 안전합니다.

4. **해시 기반 중복제거 안정화**
   - `_deduplicate_docs()`가 파이썬 내장 `hash()`를 사용합니다. 이는 프로세스마다 랜덤 시드가 달라 재현성이 떨어집니다.
   - `hashlib.sha256(page_content.encode(...))` 같은 안정 해시를 사용하면 디버깅/테스트 재현성이 좋아집니다.

5. **rate limit/재시도 정책 중앙화**
   - `time.sleep(1)`가 여러 노드에 산재해 있고, 호출 실패 처리도 함수별로 다릅니다.
   - LLM 호출 유틸(예: `invoke_with_backoff`)을 두어 429/RESOURCE_EXHAUSTED를 지수 백오프로 통합 처리하면 유지보수가 쉬워집니다.

6. **로깅 체계 개선 (`print` → `logging`)**
   - 현재 전역 `print` 기반 로그는 레벨/출력 대상/구조화 분석이 어렵습니다.
   - 표준 `logging`으로 전환해 `INFO/WARNING/ERROR` 구분, node/turn_id 구조화 필드를 넣으면 운영관측(Observability)이 크게 개선됩니다.

7. **프롬프트/모델 구성의 코드 분리**
   - 프롬프트 문자열이 매우 길고 단일 파일에 집중되어 있어 변경 영향 파악이 어렵습니다.
   - `prompts/` 모듈로 분리하고, 모델 설정(`temperature`, `max_output_tokens`)도 설정 파일로 추출하면 A/B 실험이 쉬워집니다.

8. **테스트 추가 (최소 단위 테스트)**
   - 특히 아래 함수들은 회귀에 취약하므로 pytest 기준 스냅샷/유닛 테스트를 권장합니다.
     - `_extract_text_content`
     - `_extract_json`
     - `_deduplicate_docs`
     - `_build_context`
     - `router_logic`, `plan_progress`
   - 외부 API 의존부는 mocking하여 오프라인에서도 테스트 가능하게 구성하는 것이 좋습니다.

9. **문서 길이 절단 전략 개선**
   - `formatted[:500]` 같은 하드 절단은 핵심 문맥을 잃을 수 있습니다.
   - 메타데이터 + 앞/중/뒤 일부를 보존하는 방식이나 토큰 기반 절단(모델 토크나이저 기반)을 추천합니다.

10. **CLI 루프의 장애 격리 강화**
    - 현재 `except Exception`에서 전체를 잡고 계속 진행하므로, 반복 장애의 원인 추적이 어렵습니다.
    - 입력 처리, graph 실행, 출력 렌더링을 분리하고 각 단계별 예외 메시지를 구조화하면 운영시 디버깅이 훨씬 수월합니다.

## 우선순위 권장 (빠르게 효과 보는 순서)

1. 타입 불일치 수정 (`datasource`에 `plan` 반영)
2. 환경변수 검증 + 초기화 실패 메시지 명확화
3. `_extract_json` 강건화
4. 안정 해시 기반 중복 제거
5. 테스트 5~10개 추가

## 참고한 코드 위치

- `AgentState`의 `datasource` 타입과 `analyze_intent_node`의 실제 반환값
- `_extract_json`, `_deduplicate_docs`, `_build_context`
- `retrieve_node`, `grade_documents_node` 내 호출/슬립/예외 처리 패턴
- CLI 실행 루프 (`if __name__ == "__main__": ...`)
