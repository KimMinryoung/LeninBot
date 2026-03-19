import json
import os
import time
from typing import Any, Optional

CHECKPOINT_PATH = "/home/grass/leninbot/.task_checkpoint.json"


def save_checkpoint(task_name: str, step: int, total_steps: int, state: dict, note: str = ""):
    """현재 진행상황을 파일에 저장. execute_python 블록 시작 시 호출."""
    cp = {
        "task": task_name,
        "step": step,
        "total_steps": total_steps,
        "state": state,
        "note": note,
        "timestamp": time.time(),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S KST"),
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)
    print(f"💾 체크포인트 저장: [{task_name}] step {step}/{total_steps} — {note}")


def load_checkpoint(task_name: Optional[str] = None) -> Optional[dict]:
    """저장된 체크포인트 로드. task_name 지정 시 일치 여부 확인."""
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            cp = json.load(f)
        # 24시간 이상 지난 체크포인트는 무시
        if time.time() - cp.get("timestamp", 0) > 86400:
            print("⚠️ 체크포인트가 24시간 이상 지났어. 무시할게.")
            clear_checkpoint()
            return None
        if task_name and cp.get("task") != task_name:
            print(f"⚠️ 체크포인트 task 불일치: 저장={cp.get('task')}, 요청={task_name}")
            return None
        return cp
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        return None


def clear_checkpoint():
    """작업 완료 후 체크포인트 삭제."""
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("🧹 체크포인트 삭제 완료.")


def show_checkpoint():
    """현재 체크포인트 상태 출력."""
    cp = load_checkpoint()
    if cp is None:
        print("ℹ️ 저장된 체크포인트 없음.")
        return
    print(f"""
📋 체크포인트 상태
━━━━━━━━━━━━━━━━━━━━
작업: {cp['task']}
진행: step {cp['step']}/{cp['total_steps']}
메모: {cp.get('note', '-')}
저장: {cp.get('saved_at', '-')}
상태: {json.dumps(cp.get('state', {}), ensure_ascii=False, indent=2)}
""")
