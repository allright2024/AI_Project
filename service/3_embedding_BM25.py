import json
import os
import pickle  # BM25 모델 저장
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from tqdm import tqdm 

AUGMENTED_JSON_PATH = 'skku_augmented.json' # 2_LLM_using_extract_day.py 거친 파일

# 출력 파일 및 디렉터리
CHROMA_DB_PATH = './skku_notice_db'  # Vector DB 저장
BM25_MODEL_PATH = 'bm25_model.pkl'   # BM25 모델 파일
CHUNK_DATA_PATH = 'all_chunks.pkl'   # BM25 매핑을 위한 원본 청크 데이터

# 모델 및 설정
COLLECTION_NAME = 'skku_notices'
EMBEDDING_MODEL = "jhgan/ko-sbert-nli" # pretrained 모델
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_augmented_notices(file_path):
    """
    이전 단계에서 LLM으로 'start_date', 'end_date'가
    추가된 'skku_augmented.json' 파일을 로드합니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notices = json.load(f)
        print(f"✅ [1단계] '{file_path}'에서 {len(notices)}개의 공지사항 로드 완료.")
        return notices
    except FileNotFoundError:
        print(f"❌ [1단계] 오류: '{file_path}' 파일을 찾을 수 없습니다.")
        print("           먼저 LLM으로 날짜를 추출하고 저장하는 코드를 실행해야 합니다.")
        return None
    except json.JSONDecodeError:
        print(f"❌ [1단계] 오류: '{file_path}' 파일이 올바른 JSON 형식이 아닙니다.")
        return None


# --- 3. 텍스트 분할 (Chunking) ---
def run_chunking(augmented_notices):
    """
    공지사항을 의미 단위의 청크(chunk)로 분할합니다.
    (RecursiveCharacterTextSplitter, 500/50 설정)
    """
    print("\n🏛️ [2단계] 텍스트 분할(Chunking) 시작...")
    
    # 1. 분할기 정의 (사용자와 합의한 설정)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], # 문단 -> 줄바꿈 -> 띄어쓰기 순
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_chunks = []
    
    for doc_id, notice in enumerate(tqdm(augmented_notices, desc="  Chunking")):
        text_to_split = notice.get('content', '')
        
        # 2. 각 청크가 상속받을 메타데이터
        # (content는 분할되므로 제외)
        metadata = {
            "url": notice.get("url"),
            "title": notice.get("title"),
            "start_date": notice.get("start_date", "N/A"),
            "end_date": notice.get("end_date", "N/A"),
            # "category": notice.get("category", "일반"), # (카테고리도 추출했다면 포함)
        }
        
        # 3. 텍스트 분할 실행
        chunks_texts = text_splitter.split_text(text_to_split)
        
        # 4. 분할된 청크 + 메타데이터 저장
        for chunk_index, chunk_text in enumerate(chunks_texts):
            chunk_data = {
                "id": f"doc_{doc_id}_chunk_{chunk_index}", # 고유 ID
                "text": chunk_text,                      # 분할된 텍스트
                "metadata": metadata                     # 원본 문서 정보
            }
            all_chunks.append(chunk_data)

    print(f"✅ [2단계] {len(augmented_notices)}개 문서를 {len(all_chunks)}개의 청크로 분할 완료.")
    return all_chunks


# --- 4. 하이브리드 인덱싱 및 저장 ---
def run_hybrid_indexing(all_chunks):
    """
    분할된 청크를 Vector DB(Chroma)와 BM25에 인덱싱하고 *파일로 저장*합니다.
    """
    print("\n🏛️ [3단계] 하이브리드 인덱싱 및 저장 시작...")

    # --- 4-1. Vector Index (ChromaDB) ---
    print("  [3-1] Vector DB (ChromaDB) 인덱싱 중...")
    
    # 1. ChromaDB 클라이언트 초기화 (영구 저장)
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # 2. 임베딩 모델 설정
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device='cpu' # GPU 사용 시 'cuda'
    )

    # 3. 컬렉션 생성 또는 가져오기
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"} # 코사인 유사도
    )

    # 4. ChromaDB에 저장할 형식으로 데이터 준비
    documents = [chunk['text'] for chunk in all_chunks]
    metadatas = [chunk['metadata'] for chunk in all_chunks]
    ids = [chunk['id'] for chunk in all_chunks]

    # 5. DB에 데이터 추가 (Upsert: 덮어쓰기)
    # (배치 크기를 조절하여 메모리 관리)
    batch_size = 5000
    for i in tqdm(range(0, len(ids), batch_size), desc="  Chroma Upsert"):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
    
    print(f"  ✅ [3-1] Vector DB 인덱싱 완료. '{CHROMA_DB_PATH}'에 저장됨.")

    # --- 4-2. Keyword Index (BM25) ---
    print("  [3-2] BM25 인덱싱 중...")

    # 1. BM25용 토큰화된 코퍼스 생성
    # (주의: 간단한 공백 기준. 성능을 높이려면 MeCab, KoNLPy 등 형태소 분석기 권장)
    print("    - BM25 코퍼스 토큰화 중...")
    tokenized_corpus = [chunk['text'].split() for chunk in tqdm(all_chunks)]

    # 2. BM25 모델 초기화 및 학습
    print("    - BM25 모델 학습 중...")
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. BM25 모델과 원본 청크 데이터를 파일로 저장 (pickle)
    with open(BM25_MODEL_PATH, 'wb') as f_bm25:
        pickle.dump(bm25, f_bm25)
        print(f"  ✅ [3-2] BM25 모델 저장 완료. ({BM25_MODEL_PATH})")
        
    with open(CHUNK_DATA_PATH, 'wb') as f_chunks:
        pickle.dump(all_chunks, f_chunks)
        print(f"  ✅ [3-2] BM25 매핑용 청크 데이터 저장 완료. ({CHUNK_DATA_PATH})")
    
    print(f"✅ [3단계] 하이브리드 인덱싱 완료.")


# --- 5. 메인 파이프라인 실행 ---
if __name__ == "__main__":
    
    # 1. LLM으로 강화된 JSON 로드
    augmented_notices = load_augmented_notices(AUGMENTED_JSON_PATH)
    
    if augmented_notices:
        # 2. 텍스트 분할 (Chunking)
        chunks = run_chunking(augmented_notices)
        
        # 3. 하이브리드 인덱싱 및 디스크 저장
        run_hybrid_indexing(chunks)
        
        print("\n🎉 RAG 전처리(인덱싱) 파이프라인이 성공적으로 완료되었습니다.")
        print(f"  - Vector DB: '{CHROMA_DB_PATH}'")
        print(f"  - BM25 Model: '{BM25_MODEL_PATH}'")
        print(f"  - Chunk Data: '{CHUNK_DATA_PATH}'")
        print("이제 이 파일들을 사용하여 쿼리 처리(서비스) 파이프라인을 구축할 수 있습니다.")
