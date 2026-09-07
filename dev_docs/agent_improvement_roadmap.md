# Agent 개선 현황과 남은 검토

2026-09-07 코드 기준. 완료된 구현 계획과 당시 배포·평가 수치는 제거했다. 현재 동작은 아래 주제별 설계 문서가 소유한다.

| 구현된 기능 | 코드 | 현재 문서 |
|---|---|---|
| 공용 tool-loop 엔진 | `llm/agent_loop.py` | [llm_provider_architecture.md](llm_provider_architecture.md) |
| 작업 사후 검증과 bounded retry | `telegram/tasks.py`의 `_run_verification` 호출 | [multi_agent_architecture.md](multi_agent_architecture.md) |
| 보고서 진단→저자 수정 | `llm/reflexion.py`, `telegram/tasks.py` | [multi_agent_architecture.md](multi_agent_architecture.md) |
| 의존 작업 DAG와 결과 전달 | `self_runtime/tools.py`, `telegram/tasks.py` | [multi_agent_architecture.md](multi_agent_architecture.md) |
| 자율 tick planner/critic·편집 진단 | `jobs/autonomous_project.py` | [autonomous_project.md](autonomous_project.md) |
| 작업·자율 문맥의 경험 회상과 실패 기록 | `memory_store/experiential.py` | [multi_agent_architecture.md](multi_agent_architecture.md), [autonomous_project.md](autonomous_project.md) |
| KG 임베딩 pacing/retry, 검색 폴백, entity-gated 회상 | `graph_memory/service.py`, `kg_runtime/` | [knowledge_graph_design.md](knowledge_graph_design.md) |

작업 검증기는 미호출 코드가 아니며 `multi_delegate`는 병렬 fan-out뿐 아니라 `depends_on`도 지원한다. 자율 critic 결과는 진단 피드백이며 프로젝트를 자동 정지시키지 않는다. 기능 구현과 운영 활성화는 구별한다. 설정값은 `config.json`, `config/agent_runtime.json`, 관련 env를 확인한다.

## 남은 검토

- 코퍼스: Mao manifest 선별, Stalin 재등록 완료 여부, legacy chunk metadata 누락을 재감사한다. 과거 관찰만으로 현재 미완료를 단정하지 않는다. 절차는 [vector_corpus_reingestion.md](vector_corpus_reingestion.md).
- 비판 모델의 provider 독립성, planner 품질과 tick 비용은 실제 결과를 통해 평가할 과제다. 이미 해결된 모델 티어·flag 설계를 다시 구현할 작업으로 취급하지 않는다.
- public web chat에 경험 메모리를 주입하려면 공개 가능한 범위부터 설계해야 한다. 현재 entity-gated KG 회상과는 별개다.
- best-of-N은 읽기 전용 작업에 대한 조건부 실험 제안이며 구현·실행 승인이 아니다. 품질 개선이 추가 비용을 정당화할 때만 검토한다. 전체 LATS는 side effect의 복제/롤백과 비용 문제로 보류한다.
