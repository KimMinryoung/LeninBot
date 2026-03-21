# API Reference

> LeninBot FastAPI 서버 (`api.py`) — `https://leninbot.duckdns.org`

---

## 인증

관리자 전용 엔드포인트는 `X-Admin-Key` 헤더가 필요하다.

```
X-Admin-Key: <ADMIN_API_KEY 환경변수 값>
```

키가 없거나 틀리면 `403 Forbidden`, 서버에 키가 미설정이면 `503 Service Unavailable`.

---

## 엔드포인트 목록

### Health Check

| Method | Path | 인증 | 설명 |
|--------|------|------|------|
| GET, HEAD | `/` | 없음 | 서버 상태 확인 |
| GET, HEAD | `/api/health` | 없음 | 서버 상태 확인 (별칭) |

**응답:**
```json
{"status": "ok"}
```

---

### 채팅 (SSE 스트리밍)

| Method | Path | 인증 | 설명 |
|--------|------|------|------|
| POST | `/chat` | 없음 | LangGraph 파이프라인으로 질문 처리, SSE로 실시간 스트리밍 |

**요청 Body:**
```json
{
    "message": "질문 텍스트",
    "session_id": "세션 ID (기본: default)",
    "fingerprint": "브라우저 fingerprint (localStorage UUID)"
}
```

**SSE 이벤트 형식:**
```
data: {"type": "log", "node": "노드명", "content": "로그 메시지"}
data: {"type": "answer", "content": "최종 답변"}
data: {"type": "error", "content": "에러 메시지"}
```

**동시 요청 제한:** 같은 `session_id`로 동시 요청 시 후속 요청은 에러 반환.

**호출자:** BichonWebsite (`public/js/chat.js`)

---

### 대화 기록 조회

| Method | Path | 인증 | 설명 |
|--------|------|------|------|
| GET | `/history` | 없음 | 특정 사용자의 대화 기록 조회 |

**Query Parameters:**

| 파라미터 | 필수 | 기본값 | 설명 |
|----------|------|--------|------|
| `fingerprint` | O | — | 브라우저 fingerprint (UUID) |
| `limit` | X | 50 | 최대 반환 수 (1-200) |

**응답:**
```json
{
    "history": [
        {
            "user_query": "질문",
            "bot_answer": "답변",
            "created_at": "2026-03-21T12:00:00"
        }
    ]
}
```

**호출자:** BichonWebsite (`public/js/chat.js`) — 연결 끊김 시 최근 대화 복구용

---

### 태스크 리포트

| Method | Path | 인증 | 설명 |
|--------|------|------|------|
| GET | `/reports` | 없음 | 완료된 태스크 리포트 목록 |
| GET | `/reports/{report_id}` | 없음 | 개별 리포트 조회 |

**`/reports` Query Parameters:**

| 파라미터 | 필수 | 기본값 | 설명 |
|----------|------|--------|------|
| `limit` | X | 20 | 최대 반환 수 (1-100) |
| `offset` | X | 0 | 페이지네이션 오프셋 |

**`/reports` 응답:**
```json
{
    "reports": [
        {
            "id": 1,
            "content": "태스크 내용",
            "result": "결과 (마크다운)",
            "created_at": "2026-03-21T12:00:00",
            "completed_at": "2026-03-21T12:05:00"
        }
    ],
    "total": 42
}
```

**`/reports/{report_id}` 응답:**
```json
{
    "report": { "id": 1, "content": "...", "result": "...", "created_at": "...", "completed_at": "..." }
}
```

404 시: `{"detail": "Report not found"}`

**호출자:** BichonWebsite (`routes/reports.js`) — 서버 사이드 캐싱 포함

---

### 채팅 로그 (관리자)

| Method | Path | 인증 | 설명 |
|--------|------|------|------|
| GET | `/logs` | `X-Admin-Key` | 전체 채팅 로그 조회 (관리자용) |

**Query Parameters:**

| 파라미터 | 필수 | 기본값 | 설명 |
|----------|------|--------|------|
| `limit` | X | 50 | 최대 반환 수 (1-500) |
| `offset` | X | 0 | 페이지네이션 오프셋 |

**응답:**
```json
{
    "logs": [
        {
            "id": 1,
            "session_id": "abc123",
            "fingerprint": "uuid",
            "user_agent": "...",
            "ip_address": "1.2.3.4",
            "user_query": "질문",
            "bot_answer": "답변",
            "route": "vectorstore",
            "documents_count": 3,
            "web_search_used": false,
            "strategy": null,
            "processing_logs": "[\"로그1\", \"로그2\"]",
            "created_at": "2026-03-21T12:00:00"
        }
    ],
    "count": 50
}
```

**호출자:** BichonWebsite (`/admin/api/logs` 프록시 → 이 엔드포인트)

---

### 세션 관리 (관리자)

| Method | Path | 인증 | 설명 |
|--------|------|------|------|
| DELETE | `/session/{session_id}` | `X-Admin-Key` | 특정 세션의 체크포인트 삭제 |
| DELETE | `/sessions` | `X-Admin-Key` | 전체 세션 체크포인트 삭제 |

**`DELETE /session/{session_id}` 응답:**
```json
{"session_id": "abc123", "cleared": true}
```

**`DELETE /sessions` 응답:**
```json
{"cleared_sessions": 5, "session_ids": ["abc123", "def456", "..."]}
```

**참고:** 체크포인트는 인메모리(`MemorySaver`)이므로 서버 재시작 시 자동 초기화됨. 이 API는 재시작 없이 특정 세션만 리셋할 때 사용.

---

## CORS

허용 오리진:
- `https://bichonwebpage.onrender.com`
- `http://localhost:3000`

---

## 백그라운드 스케줄러

서버 시작 시 자동 실행되는 백그라운드 태스크:

| 스케줄러 | 주기 | 설명 |
|----------|------|------|
| 일기 작성 (`diary_writer`) | 6시간 (0, 6, 12, 18시 KST) | AI 일기 자동 작성 |
| 경험 메모리 (`experience_writer`) | 매일 00:30 KST | 대화 경험 압축 및 저장 |

---

## 환경변수

| 변수 | 필수 | 설명 |
|------|------|------|
| `ADMIN_API_KEY` | O | 관리자 API 인증 키 |
| `RUN_TELEGRAM_IN_API` | X | `true`면 텔레그램 봇도 같은 프로세스에서 실행 (개발용) |

DB, LLM 등 기타 환경변수는 `dev_docs/project_state.md` 참조.
