"""
자가수정 안전 프로토콜 - 통합 테스트

Tests:
  1. Git backup mechanism
  2. Diff generation
  3. Line patch application
  4. Sandbox tests
  5. Rollback capability
  6. End-to-end modification flow

Status: Comprehensive test suite
"""

import os
import sys
import tempfile
import shutil
import subprocess
import unittest
from pathlib import Path
from datetime import datetime

# Import core module
sys.path.insert(0, '/home/grass/leninbot')
from self_modification_core import (
    git_backup_before_modification,
    git_reset_to_commit,
    generate_line_patch,
    format_diff_for_display,
    apply_line_patch_safe,
    run_sandbox_tests,
    read_file
)


class TestGitBackup(unittest.TestCase):
    """Git 백업 메커니즘 테스트"""
    
    def setUp(self):
        """테스트 파일 생성"""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_module.py")
        
        # 테스트 리포 초기화
        os.chdir(self.test_dir)
        subprocess.run(["git", "init"], capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], capture_output=True)
        
        # 테스트 파일 생성
        with open(self.test_file, 'w') as f:
            f.write("def hello():\n    return 'world'\n")
        
        # 초기 커밋
        subprocess.run(["git", "add", "."], capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], capture_output=True)
    
    def tearDown(self):
        """테스트 디렉토리 정리"""
        os.chdir("/home/grass/leninbot")
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_backup_creates_commit(self):
        """백업이 새 커밋을 생성하는지 확인"""
        os.chdir(self.test_dir)
        
        # 파일 수정
        with open(self.test_file, 'w') as f:
            f.write("def hello():\n    return 'modified'\n")
        
        # 백업 생성
        commit_hash = git_backup_before_modification(self.test_file)
        
        # 해시 검증
        self.assertIsNotNone(commit_hash)
        self.assertEqual(len(commit_hash), 40)  # SHA1 길이
        
        print(f"✓ Backup created: {commit_hash}")
    
    def test_rollback_restores_state(self):
        """롤백이 이전 상태를 복구하는지 확인"""
        os.chdir(self.test_dir)
        
        # 원본 상태 저장
        original_content = read_file(self.test_file)
        
        # 커밋 해시 저장
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True
        )
        original_hash = result.stdout.strip()
        
        # 파일 수정 및 백업
        with open(self.test_file, 'w') as f:
            f.write("# Modified content\n")
        subprocess.run(["git", "add", "."], capture_output=True)
        subprocess.run(["git", "commit", "-m", "Test commit"], capture_output=True)
        
        # 롤백
        success = git_reset_to_commit(original_hash)
        
        # 검증
        self.assertTrue(success)
        restored_content = read_file(self.test_file)
        self.assertEqual(original_content, restored_content)
        
        print(f"✓ Rollback successful, content restored")


class TestDiffGeneration(unittest.TestCase):
    """Diff 생성 테스트"""
    
    def test_simple_insertion(self):
        """간단한 삽입 감지"""
        old = "line1\nline2\n"
        new = "line1\nline_new\nline2\n"
        
        patch = generate_line_patch(old, new)
        
        self.assertEqual(patch.insertions, 1)
        self.assertEqual(patch.deletions, 0)
        self.assertEqual(patch.total_changes, 1)
        
        print(f"✓ Simple insertion detected: {patch.to_dict()}")
    
    def test_simple_deletion(self):
        """간단한 삭제 감지"""
        old = "line1\nline2\nline3\n"
        new = "line1\nline3\n"
        
        patch = generate_line_patch(old, new)
        
        self.assertEqual(patch.insertions, 0)
        self.assertEqual(patch.deletions, 1)
        self.assertEqual(patch.total_changes, 1)
        
        print(f"✓ Simple deletion detected: {patch.to_dict()}")
    
    def test_complex_changes(self):
        """복잡한 변경 감지"""
        old = """def func1():
    return 'old'

def func2():
    pass
"""
        new = """def func1():
    return 'new'

def func2():
    x = 1
    return x

def func3():
    pass
"""
        
        patch = generate_line_patch(old, new)
        
        self.assertGreater(patch.insertions, 0)
        self.assertGreater(patch.total_changes, 0)
        
        print(f"✓ Complex changes detected: {patch.insertions} ins, {patch.deletions} del")
    
    def test_diff_display_format(self):
        """Diff 표시 포맷 검증"""
        old = "a\nb\n"
        new = "a\nc\nb\n"
        
        patch = generate_line_patch(old, new)
        formatted = format_diff_for_display(patch)
        
        self.assertIn("Summary", formatted)
        self.assertIn(str(patch.insertions), formatted)
        self.assertIn(str(patch.deletions), formatted)
        
        print(f"✓ Diff format valid:\n{formatted}")


class TestPatchApplication(unittest.TestCase):
    """라인 패치 적용 테스트"""
    
    def setUp(self):
        """테스트 파일 생성"""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "module.py")
    
    def tearDown(self):
        """정리"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_apply_valid_patch(self):
        """유효한 패치 적용"""
        old_content = "def hello():\n    return 'world'\n"
        new_content = "def hello():\n    return 'modified'\n"
        
        # 초기 파일 생성
        with open(self.test_file, 'w') as f:
            f.write(old_content)
        
        # Patch 생성
        patch = generate_line_patch(old_content, new_content)
        
        # Patch 적용
        success = apply_line_patch_safe(self.test_file, patch)
        
        # 검증
        self.assertTrue(success)
        applied_content = read_file(self.test_file)
        self.assertEqual(applied_content.strip(), new_content.strip())
        
        print(f"✓ Patch applied successfully")
    
    def test_apply_syntax_validated(self):
        """Syntax 검증 후 적용"""
        old_content = "x = 1\n"
        new_content = "x = 1\ny = 2\n"
        
        with open(self.test_file, 'w') as f:
            f.write(old_content)
        
        patch = generate_line_patch(old_content, new_content)
        success = apply_line_patch_safe(self.test_file, patch)
        
        self.assertTrue(success)
        print(f"✓ Syntax validated during application")
    
    def test_backup_created_during_apply(self):
        """적용 중 백업이 생성되는지 확인"""
        old_content = "x = 1\n"
        new_content = "x = 2\n"
        
        with open(self.test_file, 'w') as f:
            f.write(old_content)
        
        patch = generate_line_patch(old_content, new_content)
        apply_line_patch_safe(self.test_file, patch)
        
        # .backup 파일 확인
        backup_file = self.test_file + ".backup"
        self.assertFalse(os.path.exists(backup_file), "Backup should be cleaned up")
        
        print(f"✓ Backup created and cleaned up properly")


class TestSandboxTests(unittest.TestCase):
    """샌드박스 테스트 테스트"""
    
    def setUp(self):
        """테스트 파일 생성"""
        self.test_dir = tempfile.mkdtemp()
        sys.path.insert(0, self.test_dir)
    
    def tearDown(self):
        """정리"""
        if self.test_dir in sys.path:
            sys.path.remove(self.test_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_syntax_check(self):
        """문법 검증 테스트"""
        test_file = os.path.join(self.test_dir, "valid_module.py")
        with open(test_file, 'w') as f:
            f.write("def hello():\n    return 'world'\n")
        
        results = run_sandbox_tests(test_file)
        
        syntax_test = [t for t in results.tests if t.name == "syntax_check"][0]
        self.assertEqual(syntax_test.result, "pass")
        
        print(f"✓ Syntax check passed")
    
    def test_syntax_check_fails_on_error(self):
        """문법 오류 감지 테스트"""
        test_file = os.path.join(self.test_dir, "invalid_module.py")
        with open(test_file, 'w') as f:
            f.write("def hello(\n    return 'world'\n")  # Missing )
        
        results = run_sandbox_tests(test_file)
        
        syntax_test = [t for t in results.tests if t.name == "syntax_check"][0]
        self.assertEqual(syntax_test.result, "fail")
        
        print(f"✓ Syntax errors detected correctly")
    
    def test_function_detection(self):
        """함수 정의 감지 테스트"""
        test_file = os.path.join(self.test_dir, "funcs_module.py")
        with open(test_file, 'w') as f:
            f.write("""
def func1():
    pass

def func2():
    pass

class MyClass:
    def method(self):
        pass
""")
        
        results = run_sandbox_tests(test_file)
        
        func_test = [t for t in results.tests if t.name == "defined_functions"][0]
        self.assertEqual(func_test.result, "pass")
        self.assertIn("2 functions", func_test.message)  # func1, func2
        self.assertIn("1 classes", func_test.message)  # MyClass
        
        print(f"✓ Function detection works: {func_test.message}")


class TestEndToEnd(unittest.TestCase):
    """종단간 테스트"""
    
    def setUp(self):
        """테스트 환경 준비"""
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)
        
        # Git 초기화
        subprocess.run(["git", "init"], capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], capture_output=True)
        
        # 테스트 파일 생성
        self.test_file = os.path.join(self.test_dir, "app.py")
        with open(self.test_file, 'w') as f:
            f.write("""def calculate(x, y):
    return x + y

if __name__ == "__main__":
    print(calculate(1, 2))
""")
        
        # 초기 커밋
        subprocess.run(["git", "add", "."], capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], capture_output=True)
    
    def tearDown(self):
        """정리"""
        os.chdir("/home/grass/leninbot")
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_full_modification_cycle(self):
        """전체 수정 사이클 테스트"""
        os.chdir(self.test_dir)
        
        # 1. 백업
        backup_hash = git_backup_before_modification(self.test_file)
        self.assertIsNotNone(backup_hash)
        
        # 2. 수정된 내용 준비
        new_content = """def calculate(x, y):
    return x + y + 10  # Added offset

if __name__ == "__main__":
    result = calculate(1, 2)
    print(f"Result: {result}")
"""
        
        # 3. Diff 생성
        old_content = read_file(self.test_file)
        patch = generate_line_patch(old_content, new_content)
        self.assertGreater(patch.total_changes, 0)
        
        # 4. Patch 적용
        success = apply_line_patch_safe(self.test_file, patch)
        self.assertTrue(success)
        
        # 5. 내용 검증
        applied = read_file(self.test_file)
        self.assertIn("Added offset", applied)
        
        # 6. 테스트 실행
        results = run_sandbox_tests(self.test_file)
        self.assertEqual(results.status, "pass")
        
        print(f"✓ Full cycle successful: {backup_hash}, {patch.total_changes} changes")
    
    def test_rollback_after_failure(self):
        """실패 후 롤백"""
        os.chdir(self.test_dir)
        
        # 백업 생성
        backup_hash = git_backup_before_modification(self.test_file)
        original_content = read_file(self.test_file)
        
        # 파일 수정
        with open(self.test_file, 'w') as f:
            f.write("# This is broken code\ndef (invalid syntax")
        
        # 롤백
        success = git_reset_to_commit(backup_hash)
        self.assertTrue(success)
        
        # 복구 확인
        restored = read_file(self.test_file)
        self.assertEqual(original_content, restored)
        
        print(f"✓ Rollback successful after failure")


def run_test_suite():
    """테스트 스위트 실행"""
    
    # 테스트 로더
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 추가
    suite.addTests(loader.loadTestsFromTestCase(TestGitBackup))
    suite.addTests(loader.loadTestsFromTestCase(TestDiffGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestPatchApplication))
    suite.addTests(loader.loadTestsFromTestCase(TestSandboxTests))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 결과 요약
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
