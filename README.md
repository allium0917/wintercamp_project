## discord id :: cherry_allium

# RAG 기반 PDF 질의응답 시스템

PDF 문서를 업로드하고 자연어 질문으로 문서 내용을 검색하여 LLM이 답변하는 시스템입니다.

## 시스템 아키텍처

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌──────────┐
│   Client    │ ───> │   Backend    │ ───> │ RAG Server  │ ───> │  Ollama  │
│  (curl/API) │      │  (FastAPI)   │      │  (FastAPI)  │      │ (gemma3) │
└─────────────┘      └──────────────┘      └─────────────┘      └──────────┘
                           │                      │
                           │                      │
                           v                      v
                     ┌──────────┐         ┌─────────────┐
                     │  Cookie  │         │  ChromaDB   │
                     │   Auth   │         │  +Vector    │
                     └──────────┘         └─────────────┘
```

## 주요 기능

- **사용자 인증**: 쿠키 기반 로그인/세션 관리
- **PDF 업로드**: PDF 문서를 업로드하고 텍스트 추출
- **RAG 검색**: 질문과 관련된 문서 청크를 벡터 검색
- **LLM 답변**: Ollama의 gemma3 모델을 사용한 한국어 답변 생성

## 기술 스택

### Backend
- FastAPI (Python 3.12)
- PyMuPDF / pymupdf4llm (PDF 처리)
- Cookie 기반 인증

### RAG Server
- FastAPI (Python 3.12)
- ChromaDB (벡터 데이터베이스)
- SentenceTransformer (`jhgan/ko-sroberta-multitask`)
- tiktoken (텍스트 청킹)

### Infrastructure
- Docker & Docker Compose
- Ollama (LLM 서버)
- Network: `rag-network`

## 사전 요구사항

1. **Docker Desktop** 설치 및 실행
2. **Ollama** 설치 및 `gemma3:1b` 모델 다운로드
   ```bash
   # Ollama 설치 (macOS)
   brew install ollama

   # Ollama 서비스 시작
   ollama serve

   # gemma3:1b 모델 다운로드
   ollama pull gemma3:1b
   ```

3. **Docker 네트워크** 생성
   ```bash
   docker network create rag-network
   ```

4. **Ollama를 rag-network에 연결** (Ollama가 Docker 컨테이너로 실행 중인 경우)
   ```bash
   docker network connect rag-network ollama
   ```

## 설치 및 실행

### 1. 프로젝트 클론
```bash
cd /path/to/3day
```

### 2. Docker Compose로 서비스 실행
```bash
docker-compose -f docker-compose up -d --build
```

### 3. 서비스 확인
```bash
docker-compose -f docker-compose ps
```

실행 중인 서비스:
- **Backend**: http://localhost:8000
- **RAG Server**: http://localhost:8888

### 4. 로그 확인
```bash
# 모든 서비스 로그
docker-compose -f docker-compose logs -f

# 특정 서비스 로그
docker logs backend -f
docker logs rag_server -f
```

## API 사용 방법

### 1. 로그인
```bash
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username": "choi", "password": "q1w2e3"}' \
  -c cookies.txt
```

**응답:**
```json
{"ok": true}
```

### 2. PDF 업로드
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/document.pdf" \
  -b cookies.txt
```

**응답:**
```json
{
  "ok": true,
  "user": "choi",
  "chars": 7928,
  "preview": "## 문서 내용 미리보기..."
}
```

### 3. 질문하기
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '"총 매출은?"' \
  -b cookies.txt
```

**응답:**
```json
{
  "ok": true,
  "user": "choi",
  "query": "총 매출은?",
  "llm_response": {
    "response": "총 매출은 4.913억 원입니다."
  }
}
```

## 기본 사용자 계정

| Username | Password |
|----------|----------|
| park     | q1w2e3   |
| choi     | q1w2e3   |

## 프로젝트 구조

```
3day/
├── backend/
│   ├── main.py              # Backend FastAPI 서버
│   ├── requirements.txt     # Python 의존성
│   └── Dockerfile
├── rag_server/
│   ├── main.py              # RAG 서버
│   ├── requirements.txt     # Python 의존성
│   ├── Dockerfile
│   └── chroma_data/         # ChromaDB 데이터 (볼륨)
├── docker-compose           # Docker Compose 설정
└── README.md
```

## API 엔드포인트

### Backend (Port 8000)

| Method | Endpoint  | 설명                | 인증 필요 |
|--------|-----------|---------------------|-----------|
| GET    | /hello    | 헬스 체크           | X         |
| POST   | /login    | 로그인              | X         |
| GET    | /page     | 페이지 접근 확인    | O         |
| POST   | /upload   | PDF 업로드          | O         |
| POST   | /query    | 질문하기            | O         |
| POST   | /summary  | 요약하기           | O         |

### RAG Server (Port 8888)

| Method | Endpoint | 설명                          |
|--------|----------|-------------------------------|
| POST   | /upload  | 텍스트를 ChromaDB에 저장      |
| POST   | /answer  | 질문에 대한 LLM 응답 생성     |

## 동작 원리

1. **PDF 업로드**: 사용자가 PDF 파일을 업로드하면 PyMuPDF로 텍스트 추출
2. **텍스트 청킹**: RAG 서버가 텍스트를 1000 토큰 단위로 분할
3. **벡터 임베딩**: SentenceTransformer로 각 청크를 벡터로 변환
4. **ChromaDB 저장**: 벡터화된 청크를 ChromaDB에 저장
5. **질문 처리**: 사용자 질문이 들어오면 벡터 검색으로 관련 청크 검색 (top 3)
6. **LLM 답변 생성**: 검색된 청크를 컨텍스트로 Ollama에 전달하여 답변 생성

## 트러블슈팅

### 1. 디스크 공간 부족
```bash
# Docker 정리
docker system prune -f

# 사용하지 않는 이미지 삭제
docker image prune -a
```

### 2. Ollama 연결 실패
```bash
# Ollama 서비스 확인
ollama list

# rag-network 연결 확인
docker network inspect rag-network
```

### 3. 모델 로딩 시간이 오래 걸림
RAG 서버 시작 시 SentenceTransformer 모델 다운로드로 1-2분 소요됩니다.
```bash
# 로그로 진행 상태 확인
docker logs rag_server -f
```

### 4. ChromaDB 컬렉션 충돌
```bash
# ChromaDB 데이터 초기화
rm -rf rag_server/chroma_data/*
docker-compose restart rag_server
```

## 서비스 중지

```bash
# 서비스 중지
docker-compose -f docker-compose down

# 서비스 중지 + 볼륨 삭제
docker-compose -f docker-compose down -v
```

## 성능 최적화

- **청크 크기 조정**: `split_text()` 함수의 `chunk_size` 파라미터 조정
- **검색 결과 개수**: `query()` 함수의 `n_results` 파라미터 조정 (현재 3개)
- **LLM 타임아웃**: 긴 응답이 필요한 경우 `timeout` 값 증가

## 라이선스

이 프로젝트는 교육 목적으로 제공됩니다.

## 참고 자료

- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [Ollama 문서](https://ollama.ai/)
- [SentenceTransformers](https://www.sbert.net/)
