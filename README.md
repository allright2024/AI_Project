# AI_Project

성균관대학교 공지사항 검색 및 질의응답 RAG 시스템

## 환경 설정

`.env` 파일에 API 키를 설정해주세요:
```
GOOGLE_API_KEY="Your Google API Key"
OPENAI_API_KEY="Your OpenAI API Key"
```

## 시스템 아키텍처

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  crawl_server   │────▶│   search_server  │◀────│    ai.py    │
│   (Port 8000)   │     │    (Port 8001)   │     │  (Client)   │
└─────────────────┘     └──────────────────┘     └─────────────┘
        │                        │
        ▼                        ▼
  크롤링 + LLM 처리      하이브리드 검색 API
  + 임베딩 파이프라인     (Vector + BM25)
```

## 실행 방법

### 1. 크롤링 서버 실행
```bash
cd server/crawl
python crawl_server.py --days=90
```
- 3시간마다 자동 크롤링 실행
- `POST /crawl` : 수동 크롤링 트리거
- `GET /health` : 서버 상태 확인

### 2. 검색 서버 실행
```bash
cd server/crawl
python search_server.py
```
- `POST /search` : 하이브리드 검색 API

### 3. 클라이언트 실행
```bash
python ai.py
```
- 사용자 질문 입력 → 검색 → GPT 답변 생성

## 폴더 구조

### server/crawl/
| 파일 | 설명 |
|------|------|
| `crawl_server.py` | 크롤링 자동화 서버 (스케줄러) |
| `search_server.py` | 검색 API 서버 (FastAPI) |
| `crawl.py`, `crawl_*.py` | 각 사이트별 크롤러 |
| `llm_processor.py` | LLM 날짜 추출 처리 |
| `embedding_processor.py` | ChromaDB + BM25 인덱싱 |
| `extracted_dates.json` | 날짜가 추출된 json 파일 |

### sbert/
SBERT 모델 학습 및 평가
- `train_sbert.py` : Fine-tuning 스크립트
- `output/` : 학습된 모델 저장 위치
- `compare_models.py` : 기존 baseline 모델과 Fine-tuning 모델의 MAP@100 성능 측정
- `cross_validation.py` : hyper parameter search with K-Folds(K = 3)

### data_augmentation/
SBERT 학습 데이터 생성 (Gemini API 활용)
- `synthetic_aug_api.py` : 질문 자동 생성

## 기술 스택
- **검색**: ChromaDB (Vector) + BM25 (Keyword) + RRF 융합
- **임베딩**: Fine-tuned SBERT (`jhgan/ko-sbert-nli` 기반)
- **LLM**: Google Gemini, OpenAI GPT-4o-mini
- **서버**: FastAPI, APScheduler
