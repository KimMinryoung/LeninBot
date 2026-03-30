# 자가 수정 안전 규칙

## 🔴 절대 금지 패턴

아래 패턴이 수정 코드에 포함되면 즉시 중단:

```python
# 시스템 명령 실행
os.system(...)
subprocess.call(...)
subprocess.Popen(...)

# 인증 우회
if user_id == ...  # 하드코딩 우회
ADMIN_IDS = [...]  # 권한 목록 직접 수정

# 파괴적 파일 작업
shutil.rmtree(...)
os.remove("/")
open("/etc/passwd", "w")

# 외부 코드 실행
exec(requests.get(...).text)
eval(user_input)

# 환경변수/시크릿 노출
print(os.environ)
logging.info(API_KEY)
```

## 🟡 주의 필요 패턴 (사용자 승인 후 진행)

- `AUTHORIZED_USERS` 또는 `ADMIN_IDS` 수정
- 텔레그램 봇 핸들러 추가/제거
- DB 스키마 변경
- 배포 스크립트(`deploy.sh`) 수정

## 🟢 자유롭게 수정 가능

- 응답 텍스트, 메시지 포맷
- 분석 로직, 프롬프트 내용
- 로깅 추가 (시크릿 제외)
- 버그 수정 (로직 오류, 타입 에러 등)
- 새 유틸리티 함수 추가

## 롤백 절차

수정 실패 시:
```
# git으로 되돌리기
git stash  # 또는
git checkout HEAD -- 파일명.py

# restart_service tool로 재시작 (subprocess 직접 사용 금지)
restart_service(service="telegram")
```
