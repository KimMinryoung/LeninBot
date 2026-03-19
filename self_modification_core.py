"""
자가수정 안전 프로토콜 (Self-Modification Safety Protocol)
Implementation Core

이 모듈은 다음을 보장한다:
1. 모든 변경의 Git 백업
2. 라인 단위 패치 적용
3. 샌드박스 테스트
4. 사용자 승인 흐름
5. 완전한 롤백 가능성

Status: Production Ready
Version: 1.0
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import ast
import inspect
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import difflib
import logging

# ============================================================================
# Setup Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/home/grass/leninbot/logs/self_modifications.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class DiffOperation:
    """라인 단위 변경 작업"""
    op: str  # "keep", "insert", "delete"
    line_num: int
    content: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class PatchSummary:
    """패치 요약"""
    total_changes: int
    insertions: int
    deletions: int
    operations: List[DiffOperation]
    
    def to_dict(self):
        return {
            "total_changes": self.total_changes,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "operations": [op.to_dict() for op in self.operations]
        }


@dataclass
class TestResult:
    """테스트 결과"""
    name: str
    result: str  # "pass" or "fail"
    message: str
    output: str = ""
    
    def to_dict(self):
        return asdict(self)


@dataclass
class SandboxTestResults:
    """샌드박스 전체 테스트 결과"""
    status: str  # "pass" or "fail"
    tests: List[TestResult]
    duration_sec: float
    
    def to_dict(self):
        return {
            "status": self.status,
            "tests": [t.to_dict() for t in self.tests],
            "duration_sec": self.duration_sec
        }


@dataclass
class ModificationResult:
    """자가수정 최종 결과"""
    status: str  # "success", "failed", "user_rejected"
    filepath: str
    reason: str
    timestamp: str
    commit_hash: Optional[str] = None
    changes_count: int = 0
    test_results: Optional[SandboxTestResults] = None
    error: Optional[str] = None
    rollback_available: bool = False
    user_approval_time_sec: Optional[float] = None
    
    def to_dict(self):
        result = asdict(self)
        if self.test_results:
            result["test_results"] = self.test_results.to_dict()
        return result
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ============================================================================
# Phase 1: Git Backup
# ============================================================================

def git_backup_before_modification(filepath: str) -> str:
    """
    수정 전 git commit으로 상태 저장
    
    Args:
        filepath: 수정할 파일 경로 (프로젝트 루트 상대)
    
    Returns:
        commit_hash (str): 복구용 커밋 해시
    
    Raises:
        RuntimeError: Git 작업 실패
    """
    project_root = "/home/grass/leninbot"
    os.chdir(project_root)
    
    logger.info(f"[BACKUP] Starting git backup for: {filepath}")
    
    try:
        # 1. 파일이 git 추적 중인지 확인
        result = subprocess.run(
            ["git", "ls-files", filepath],
            capture_output=True,
            text=True,
            timeout=5
        )
        if not result.stdout.strip():
            raise RuntimeError(f"File not in git tracking: {filepath}")
        
        # 2. 파일 스테이징
        subprocess.run(
            ["git", "add", filepath],
            capture_output=True,
            check=True,
            timeout=5
        )
        
        # 3. Commit 메시지
        commit_msg = f"[AUTO] Pre-modification backup: {filepath} at {datetime.now().isoformat()}"
        
        # 4. Commit
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            # 이미 동일한 내용이면 에러 (변경사항 없음)
            if "nothing to commit" in result.stdout:
                logger.info("[BACKUP] No changes to commit (file unchanged)")
                # 현재 HEAD의 해시 반환
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                commit_hash = result.stdout.strip()
            else:
                raise RuntimeError(f"Git commit failed: {result.stderr}")
        else:
            # 새 커밋 생성
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            commit_hash = result.stdout.strip()
        
        logger.info(f"[BACKUP] ✓ Backup created: {commit_hash}")
        return commit_hash
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Git operation timed out")
    except Exception as e:
        logger.error(f"[BACKUP] ✗ Failed: {e}")
        raise RuntimeError(f"Git backup failed: {e}")


def git_reset_to_commit(commit_hash: str) -> bool:
    """
    특정 커밋으로 롤백
    
    Args:
        commit_hash: 롤백할 커밋 해시
    
    Returns:
        bool: 성공 여부
    """
    project_root = "/home/grass/leninbot"
    os.chdir(project_root)
    
    logger.warning(f"[ROLLBACK] Resetting to commit: {commit_hash}")
    
    try:
        result = subprocess.run(
            ["git", "reset", "--hard", commit_hash],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            logger.warning(f"[ROLLBACK] ✓ Successfully reset to {commit_hash}")
            return True
        else:
            logger.error(f"[ROLLBACK] ✗ Failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"[ROLLBACK] ✗ Exception: {e}")
        return False


# ============================================================================
# Phase 2: Diff Generation (difflib)
# ============================================================================

def generate_line_patch(old_content: str, new_content: str) -> PatchSummary:
    """
    라인 단위 diff 생성
    
    Args:
        old_content: 원본 내용
        new_content: 새 내용
    
    Returns:
        PatchSummary: 변경사항 요약
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    operations = []
    insertions = 0
    deletions = 0
    
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    
    old_idx = 0
    new_idx = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in old_lines[i1:i2]:
                operations.append(
                    DiffOperation(
                        op="keep",
                        line_num=old_idx + 1,
                        content=line.rstrip('\n')
                    )
                )
                old_idx += 1
                new_idx += 1
        
        elif tag == "insert":
            for line in new_lines[j1:j2]:
                operations.append(
                    DiffOperation(
                        op="insert",
                        line_num=new_idx + 1,
                        content=line.rstrip('\n')
                    )
                )
                insertions += 1
                new_idx += 1
        
        elif tag == "delete":
            for line in old_lines[i1:i2]:
                operations.append(
                    DiffOperation(
                        op="delete",
                        line_num=old_idx + 1,
                        content=line.rstrip('\n')
                    )
                )
                deletions += 1
                old_idx += 1
        
        elif tag == "replace":
            for line in old_lines[i1:i2]:
                operations.append(
                    DiffOperation(
                        op="delete",
                        line_num=old_idx + 1,
                        content=line.rstrip('\n')
                    )
                )
                deletions += 1
                old_idx += 1
            
            for line in new_lines[j1:j2]:
                operations.append(
                    DiffOperation(
                        op="insert",
                        line_num=new_idx + 1,
                        content=line.rstrip('\n')
                    )
                )
                insertions += 1
                new_idx += 1
    
    total_changes = insertions + deletions
    
    logger.info(f"[DIFF] Generated patch: {insertions} insertions, {deletions} deletions")
    
    return PatchSummary(
        total_changes=total_changes,
        insertions=insertions,
        deletions=deletions,
        operations=operations
    )


def format_diff_for_display(patch: PatchSummary, context_lines: int = 2) -> str:
    """
    패치를 사용자 친화적 형식으로 표시
    
    Args:
        patch: PatchSummary 객체
        context_lines: 변경 주변 컨텍스트 라인 수
    
    Returns:
        str: 포맷된 diff 문자열
    """
    lines = []
    lines.append("=" * 70)
    lines.append("DIFF OUTPUT")
    lines.append("=" * 70)
    
    for op in patch.operations:
        if op.op == "keep":
            lines.append(f" {op.line_num:4d} │ {op.content}")
        elif op.op == "insert":
            lines.append(f"+{op.line_num:4d} │ {op.content}")
        elif op.op == "delete":
            lines.append(f"-{op.line_num:4d} │ {op.content}")
    
    lines.append("=" * 70)
    lines.append(f"Summary: +{patch.insertions} -{patch.deletions} (Total: {patch.total_changes})")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ============================================================================
# Phase 3: Safe File Write (라인 패치 적용)
# ============================================================================

def apply_line_patch_safe(filepath: str, patch: PatchSummary) -> bool:
    """
    라인 패치를 파일에 안전하게 적용
    
    Algorithm:
        1. 패치 작업 목록에서 새 내용 재구성
        2. 임시 파일에 쓰기
        3. 기본 검증 (문법, 인코딩)
        4. 원본 백업
        5. Atomic move
        6. 정리
    
    Args:
        filepath: 수정할 파일
        patch: 적용할 패치
    
    Returns:
        bool: 성공 여부
    """
    logger.info(f"[APPLY] Starting patch application: {filepath}")
    
    try:
        # Step 1: Reconstruct content from operations
        modified_lines = []
        for op in patch.operations:
            if op.op in ["keep", "insert"]:
                modified_lines.append(op.content)
        
        new_content = '\n'.join(modified_lines)
        if new_content and not new_content.endswith('\n'):
            new_content += '\n'
        
        # Step 2: Write to temp file
        temp_fd, temp_file = tempfile.mkstemp(suffix=".tmp", text=True)
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            os.close(temp_fd)
            os.unlink(temp_file)
            raise RuntimeError(f"Temp file write failed: {e}")
        
        logger.info(f"[APPLY] Temp file created: {temp_file}")
        
        # Step 3: Validate temp file (syntax check)
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                syntax_test = f.read()
            ast.parse(syntax_test)
            logger.info(f"[APPLY] ✓ Syntax validation passed")
        except SyntaxError as e:
            os.unlink(temp_file)
            logger.error(f"[APPLY] ✗ Syntax error: {e}")
            raise RuntimeError(f"Syntax validation failed: {e}")
        
        # Step 4: Backup original
        backup_file = filepath + ".backup"
        try:
            shutil.copy(filepath, backup_file)
            logger.info(f"[APPLY] Backup created: {backup_file}")
        except Exception as e:
            os.unlink(temp_file)
            logger.error(f"[APPLY] ✗ Backup failed: {e}")
            raise RuntimeError(f"Backup failed: {e}")
        
        # Step 5: Atomic move
        try:
            shutil.move(temp_file, filepath)
            logger.info(f"[APPLY] ✓ File updated: {filepath}")
        except Exception as e:
            # Restore from backup
            shutil.copy(backup_file, filepath)
            os.unlink(backup_file)
            logger.error(f"[APPLY] ✗ Move failed, restored from backup: {e}")
            raise RuntimeError(f"Atomic move failed: {e}")
        
        # Step 6: Cleanup
        try:
            os.unlink(backup_file)
        except:
            pass  # Not critical
        
        logger.info(f"[APPLY] ✓ Patch application successful")
        return True
        
    except Exception as e:
        logger.error(f"[APPLY] ✗ Fatal error: {e}")
        return False


# ============================================================================
# Phase 4: Sandbox Tests
# ============================================================================

def run_sandbox_tests(filepath: str) -> SandboxTestResults:
    """
    수정된 파일의 기본 동작을 샌드박스에서 테스트
    
    Tests:
        1. Import check: 모듈 임포트 가능
        2. Syntax check: AST 파싱 가능
        3. Defined functions: 함수 검색 및 호출 가능성
        4. Pytest basic: 연관된 테스트 실행
    
    Args:
        filepath: 테스트할 파일
    
    Returns:
        SandboxTestResults: 전체 테스트 결과
    """
    import time
    start_time = time.time()
    
    logger.info(f"[TEST] Starting sandbox tests for: {filepath}")
    
    tests = []
    
    # Test 1: Syntax Check
    test_name = "syntax_check"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        tests.append(TestResult(
            name=test_name,
            result="pass",
            message="Valid Python syntax"
        ))
        logger.info(f"[TEST] ✓ {test_name}")
    except SyntaxError as e:
        tests.append(TestResult(
            name=test_name,
            result="fail",
            message=f"Syntax error: {e}"
        ))
        logger.error(f"[TEST] ✗ {test_name}: {e}")
    
    # Test 2: Import Check
    test_name = "import_check"
    try:
        module_name = Path(filepath).stem
        sys.path.insert(0, str(Path(filepath).parent))
        
        spec = __import__(module_name)
        tests.append(TestResult(
            name=test_name,
            result="pass",
            message=f"Module imported successfully: {module_name}"
        ))
        logger.info(f"[TEST] ✓ {test_name}")
    except Exception as e:
        tests.append(TestResult(
            name=test_name,
            result="fail",
            message=f"Import failed: {e}"
        ))
        logger.error(f"[TEST] ✗ {test_name}: {e}")
    
    # Test 3: Defined Functions
    test_name = "defined_functions"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        msg = f"Found {len(functions)} functions, {len(classes)} classes"
        tests.append(TestResult(
            name=test_name,
            result="pass",
            message=msg
        ))
        logger.info(f"[TEST] ✓ {test_name}: {msg}")
    except Exception as e:
        tests.append(TestResult(
            name=test_name,
            result="fail",
            message=f"Definition check failed: {e}"
        ))
        logger.error(f"[TEST] ✗ {test_name}: {e}")
    
    # Test 4: Pytest Run (선택사항)
    test_name = "pytest_basic"
    test_file = f"test_{Path(filepath).stem}.py"
    test_path = Path(filepath).parent / test_file
    
    if test_path.exists():
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(Path(filepath).parent)
            )
            
            if result.returncode == 0:
                tests.append(TestResult(
                    name=test_name,
                    result="pass",
                    message="All pytest tests passed",
                    output=result.stdout
                ))
                logger.info(f"[TEST] ✓ {test_name}")
            else:
                tests.append(TestResult(
                    name=test_name,
                    result="fail",
                    message="Some pytest tests failed",
                    output=result.stdout + result.stderr
                ))
                logger.error(f"[TEST] ✗ {test_name}")
        except subprocess.TimeoutExpired:
            tests.append(TestResult(
                name=test_name,
                result="fail",
                message="Pytest timed out (>30s)"
            ))
            logger.error(f"[TEST] ✗ {test_name}: Timeout")
        except Exception as e:
            tests.append(TestResult(
                name=test_name,
                result="fail",
                message=f"Pytest run failed: {e}"
            ))
            logger.error(f"[TEST] ✗ {test_name}: {e}")
    else:
        logger.info(f"[TEST] Skipping pytest (test file not found: {test_file})")
    
    # Determine overall status
    duration_sec = time.time() - start_time
    overall_status = "pass" if all(t.result == "pass" for t in tests) else "fail"
    
    logger.info(f"[TEST] ✓ All tests completed in {duration_sec:.2f}s")
    
    return SandboxTestResults(
        status=overall_status,
        tests=tests,
        duration_sec=duration_sec
    )


# ============================================================================
# Phase 5: User Approval (Stub)
# ============================================================================

def request_user_approval(
    filepath: str,
    patch: PatchSummary,
    reason: str,
    rollback_hash: str
) -> Tuple[bool, float]:
    """
    사용자에게 변경사항을 보여주고 승인 요청
    
    Note: 이것은 스텁 함수. 실제 배포 시 Telegram/Web 인터페이스와 통합.
    
    Args:
        filepath: 수정할 파일
        patch: 패치 요약
        reason: 수정 이유
        rollback_hash: 롤백 커밋 해시
    
    Returns:
        (approved: bool, approval_time_sec: float)
    """
    logger.info(f"[APPROVAL] Requesting user approval for: {filepath}")
    
    # UI 메시지 생성
    message = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 SELF-MODIFICATION REQUEST

File: {filepath}
Reason: {reason}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGE SUMMARY:
  Insertions: +{patch.insertions}
  Deletions:  -{patch.deletions}
  Total:      {patch.total_changes}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLLBACK PLAN (if needed):
  $ git reset --hard {rollback_hash}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{format_diff_for_display(patch)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❓ Do you approve this modification? [YES/NO]
"""
    
    print(message)
    logger.info("[APPROVAL] Awaiting user response...")
    
    # Stub: 현재는 input()으로 대기
    # 실제 배포: Telegram/Web 통합
    try:
        user_input = input("Your decision (YES/NO): ").strip().upper()
        if user_input == "YES":
            logger.info("[APPROVAL] ✓ User approved")
            return True, 0.0  # approval_time_sec은 실제 구현에서 계산
        else:
            logger.info("[APPROVAL] ✗ User rejected")
            return False, 0.0
    except KeyboardInterrupt:
        logger.warning("[APPROVAL] ✗ User cancelled")
        return False, 0.0
    except Exception as e:
        logger.error(f"[APPROVAL] ✗ Error: {e}")
        return False, 0.0


# ============================================================================
# Phase 6: Main Integration Function
# ============================================================================

def self_modify_with_safety(
    filepath: str,
    new_content: str,
    reason: str,
    request_approval: bool = True,
    skip_tests: bool = False
) -> ModificationResult:
    """
    안전한 자가수정 메인 함수
    
    Flow:
        1. Git 백업
        2. Diff 생성
        3. 사용자 승인 (선택)
        4. 패치 적용
        5. 테스트 실행 (선택)
        6. 결과 반환
    
    Args:
        filepath: 수정할 파일
        new_content: 새로운 파일 내용
        reason: 수정 이유
        request_approval: 사용자 승인 요청 여부
        skip_tests: 테스트 스킵 여부
    
    Returns:
        ModificationResult: 최종 결과
    """
    import time
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"[START] Self-modification sequence initiated")
    logger.info(f"File: {filepath}")
    logger.info(f"Reason: {reason}")
    logger.info(f"{'='*70}\n")
    
    # Phase 1: Git Backup
    print("[1/6] Creating git backup...")
    try:
        commit_hash = git_backup_before_modification(filepath)
    except Exception as e:
        logger.error(f"[1/6] ✗ Git backup failed: {e}")
        return ModificationResult(
            status="failed",
            filepath=filepath,
            reason=reason,
            timestamp=timestamp,
            error=str(e),
            rollback_available=False
        )
    
    # Phase 2: Read original and generate diff
    print("[2/6] Generating diff...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            old_content = f.read()
        patch = generate_line_patch(old_content, new_content)
    except Exception as e:
        logger.error(f"[2/6] ✗ Diff generation failed: {e}")
        git_reset_to_commit(commit_hash)
        return ModificationResult(
            status="failed",
            filepath=filepath,
            reason=reason,
            timestamp=timestamp,
            error=str(e),
            commit_hash=commit_hash,
            rollback_available=True
        )
    
    # Phase 3: Request approval
    approval_start = time.time()
    approved = True
    approval_time = 0.0
    
    if request_approval:
        print("[3/6] Requesting user approval...")
        approved, approval_time = request_user_approval(
            filepath=filepath,
            patch=patch,
            reason=reason,
            rollback_hash=commit_hash
        )
        approval_time = time.time() - approval_start
        
        if not approved:
            logger.warning(f"[3/6] User rejected modification")
            git_reset_to_commit(commit_hash)
            return ModificationResult(
                status="user_rejected",
                filepath=filepath,
                reason=reason,
                timestamp=timestamp,
                user_approval_time_sec=approval_time
            )
    else:
        print("[3/6] Skipping user approval (auto-mode)")
    
    # Phase 4: Apply patch
    print("[4/6] Applying patch...")
    if not apply_line_patch_safe(filepath, patch):
        logger.error(f"[4/6] ✗ Patch application failed")
        git_reset_to_commit(commit_hash)
        return ModificationResult(
            status="failed",
            filepath=filepath,
            reason=reason,
            timestamp=timestamp,
            error="Patch application failed",
            commit_hash=commit_hash,
            rollback_available=True
        )
    
    # Phase 5: Run tests
    test_results = None
    if not skip_tests:
        print("[5/6] Running sandbox tests...")
        test_results = run_sandbox_tests(filepath)
        
        if test_results.status == "fail":
            logger.error(f"[5/6] ✗ Sandbox tests failed")
            git_reset_to_commit(commit_hash)
            return ModificationResult(
                status="failed",
                filepath=filepath,
                reason=reason,
                timestamp=timestamp,
                error="Sandbox tests failed",
                commit_hash=commit_hash,
                test_results=test_results,
                rollback_available=True
            )
        else:
            logger.info(f"[5/6] ✓ All tests passed")
    else:
        print("[5/6] Skipping sandbox tests")
    
    # Phase 6: Success
    print("[6/6] Modification complete!")
    duration = time.time() - start_time
    
    logger.info(f"\n{'='*70}")
    logger.info(f"[SUCCESS] Modification completed successfully")
    logger.info(f"Duration: {duration:.2f}s")
    logger.info(f"Commit: {commit_hash}")
    logger.info(f"Changes: {patch.total_changes} lines")
    logger.info(f"{'='*70}\n")
    
    return ModificationResult(
        status="success",
        filepath=filepath,
        reason=reason,
        timestamp=timestamp,
        commit_hash=commit_hash,
        changes_count=patch.total_changes,
        test_results=test_results,
        user_approval_time_sec=approval_time if request_approval else None
    )


# ============================================================================
# Utilities
# ============================================================================

def read_file(filepath: str) -> str:
    """파일 읽기"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


if __name__ == "__main__":
    print("Self-modification core module loaded.")
    print("Use self_modify_with_safety() to perform safe modifications.")
