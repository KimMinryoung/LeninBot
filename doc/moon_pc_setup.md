# MOON PC (로컬 LLM) 운영 가이드

## 개요

Hetzner 서버의 leninbot이 MOON PC(Windows 10, RTX 3060 12GB)의 qwen3.5-9b Q8_0 모델을 SSH 리버스 터널로 사용한다.

터널이 끊기면 Hetzner 로컬 llama-server(qwen3.5-4b)로 자동 폴백.

---

## 1. llama-server 실행

```cmd
C:\Users\DESKTOP\llama-cpp\bin\llama-server.exe ^
  -m C:\Users\DESKTOP\llama-cpp\models\Qwen_Qwen3.5-9B-Q8_0.gguf ^
  --host 0.0.0.0 --port 8080 ^
  -ngl 99 -c 8192
```

| 옵션 | 의미 |
|------|------|
| `-m` | 모델 파일 경로 |
| `--host 0.0.0.0` | 외부 접근 허용 (터널용) |
| `--port 8080` | 서빙 포트 |
| `-ngl 99` | 모든 레이어 GPU 오프로드 |
| `-c 8192` | 컨텍스트 길이 |

### 동작 확인

```cmd
curl http://localhost:8080/health
```

`{"status":"ok"}` 응답이면 정상.

---

## 2. SSH 리버스 터널 연결

llama-server가 뜬 후 실행:

```cmd
ssh -R 8080:localhost:8080 root@37.27.33.127 -N -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes
```

| 옵션 | 의미 |
|------|------|
| `-R 8080:localhost:8080` | Hetzner의 8080 → 이 PC의 8080 포워딩 |
| `-N` | 쉘 없이 터널만 유지 |
| `-o ServerAliveInterval=60` | 60초마다 keepalive (끊김 방지) |
| `-o ExitOnForwardFailure=yes` | 포트 바인딩 실패 시 즉시 종료 |

---

## 3. 전체 시작 순서 (요약)

```
1. llama-server 실행 (위 명령어)
2. curl http://localhost:8080/health 로 확인
3. SSH 리버스 터널 연결
4. Hetzner에서 자동 감지 (llm_client.py)
```

---

## 4. 종료

```cmd
# 터널: Ctrl+C 또는 ssh 프로세스 종료
# llama-server: Ctrl+C 또는
taskkill /F /IM llama-server.exe
```

---

## 5. 트러블슈팅

### llama-server 명령어를 못 찾음
PATH에 없으면 전체 경로 사용:
```cmd
C:\Users\DESKTOP\llama-cpp\bin\llama-server.exe
```

### 터널 연결 시 "port 8080 already in use"
Hetzner에서 이전 터널이 남아있을 수 있음:
```bash
# Hetzner에서 실행
ss -tlnp | grep 8080
# 필요시 이전 ssh 프로세스 종료
```

### GPU 메모리 부족
`-ngl` 값을 줄여서 일부 레이어만 GPU에 올림:
```cmd
-ngl 25   (약 7GB 사용)
```

---

## 6. 경로 정리

| 항목 | 경로 |
|------|------|
| llama-server | `C:\Users\DESKTOP\llama-cpp\bin\llama-server.exe` |
| 모델 (9B Q8_0) | `C:\Users\DESKTOP\llama-cpp\models\Qwen_Qwen3.5-9B-Q8_0.gguf` |
| Hetzner 서버 | `root@37.27.33.127` |
| Hetzner 터널 포트 | `127.0.0.1:8080` |
| Hetzner 폴백 | `127.0.0.1:11435` (로컬 llama-server, qwen3.5-4b) |
