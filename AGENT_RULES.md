# 에이전트 운영 규칙 (2026-03-19)

## 도구 한도 대응

### 체크포인트 시스템
- 모듈: `task_checkpoint.py`
- 체크포인트 파일: `.task_checkpoint.json` (24시간 자동 만료)

```python
from task_checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint

# 세션 시작 시
cp = load_checkpoint("my_task")
if cp:
    print(f"⏩ 이어서: step {cp['step']}/{cp['total_steps']} — {cp['note']}")

# 각 단계 완료 후
save_checkpoint("my_task", step=2, total_steps=4,
    state={"backup_path": "...", "lines_count": 1500},
    note="import 패치 완료")

# 작업 완료 후
clear_checkpoint()
```

### execute_python 규칙
1. **단일 블록 원칙**: 읽기→처리→쓰기를 하나의 블록에
2. **체크포인트 필수**: 블록 시작 시 save_checkpoint() 호출
3. **단계별 저장**: 각 단계 완료 시 step 업데이트

### 복잡한 작업
- 도구 한도 우려 시 → `create_task()`로 백그라운드 위임
- 진행상황 복구 → `recall_experience()` 또는 KG 검색

## /modify 핸들러 (2026-03-19 구현 완료)
- 파일: `telegram_bot.py`
- 백업: `telegram_bot.py.bak_selfmod_integration`
- 사용법: `/modify <파일경로> | <이유> | <새 내용>`
- 승인/거부 InlineKeyboard → `self_modify_with_safety()` 실행
- 5분 만료, 허가된 사용자만
