# MOON PC (로컬 LLM) 운영 가이드

## 개요

Hetzner 서버의 leninbot이 MOON PC(Windows 10, RTX 3060 12GB)의 **Qwopus3.5-9B-v3 Q6_K** 모델을 SSH 리버스 터널로 사용한다.

Qwopus는 Qwen3.5-9B를 Claude 4.6 Opus 추론 스타일로 증류한 모델. `<think>` 태그로 구조화된 CoT를 생성하도록 학습됨 → **thinking 모드를 절대 끄지 말 것**.

터널이 끊기면 Hetzner 로컬 llama-server(qwen3.5-4b)로 자동 폴백.

---

## 1. llama-server 실행

`doc/run_llama.bat` 실행 (아래는 참고용 원본 명령어):

```cmd
C:\Users\DESKTOP\llama-cpp\bin\llama-server.exe ^
  -m C:\Users\DESKTOP\llama-cpp\models\Qwopus3.5-9B-v3.Q6_K.gguf ^
  --host 0.0.0.0 --port 8080 ^
  -ngl 99 -c 98304 ^
  --flash-attn on ^
  --cache-type-k q4_0 --cache-type-v q4_0 ^
  --ubatch-size 512 --threads 4 ^
  --parallel 1 --cont-batching ^
  --no-mmap --slots --jinja
```

| 옵션 | 의미 |
|------|------|
| `-m` | 모델 파일 경로 (Qwopus3.5-9B-v3 Q6_K, 7.36GB) |
| `--host 0.0.0.0` | 외부 접근 허용 (터널용) |
| `--port 8080` | 서빙 포트 |
| `-ngl 99` | 모든 레이어 GPU 오프로드 |
| `-c 98304` | 컨텍스트 96k (Q6_K + q4_0 KV cache가 12GB VRAM에 들어가는 한계) |
| `--flash-attn on` | Flash Attention 활성화 (VRAM 절약 + 속도) |
| `--cache-type-k/v q4_0` | KV cache 4-bit 양자화 (긴 컨텍스트용 필수) |
| `--jinja` | 모델 메타데이터의 Jinja 채팅 템플릿 사용 |
| ⚠ `--chat-template-kwargs enable_thinking=false` | **추가 금지.** Qwopus는 thinking으로 학습되어 끄면 출력 품질 붕괴 |

### 동작 확인

```cmd
curl http://localhost:8080/health
```

`{"status":"ok"}` 응답이면 정상.

---

## 2. SSH 리버스 터널 연결

llama-server가 뜬 후 실행:

```cmd
ssh -i "C:\Users\DESKTOP\leninbot" -R 127.0.0.1:8080:127.0.0.1:8080 root@leninbot -N -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes
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
| 현재 모델 (Qwopus 9B Q6_K) | `C:\Users\DESKTOP\llama-cpp\models\Qwopus3.5-9B-v3.Q6_K.gguf` |
| 이전 모델 (보관) | `Qwen3.5-9B-Q4_K_M.gguf`, `Qwen_Qwen3.5-9B-Q8_0.gguf` |
| Hetzner 서버 | `root@37.27.33.127` |
| Hetzner 터널 포트 | `127.0.0.1:8080` |
| Hetzner 폴백 | `127.0.0.1:11435` (로컬 llama-server, qwen3.5-4b) |
