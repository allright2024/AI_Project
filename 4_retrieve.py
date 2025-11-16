import os
import pickle
import json
import re
from datetime import datetime
from dotenv import load_dotenv

# --- LangChain 및 RAG 구성요소 ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field # LLM 출력을 구조화

# --- 하이브리드 검색 구성요소 ---
from rank_bm25 import BM25Okapi
from tqdm import tqdm

TODAY_STR = datetime.now().strftime('%Y-%m-%d') 

CHROMA_DB_PATH = './skku_notice_db'
BM25_MODEL_PATH = 'bm25_model.pkl'
CHUNK_DATA_PATH = 'all_chunks.pkl'

EMBEDDING_MODEL = 'BAAI/bge-base-ko-v1.5'
COLLECTION_NAME = 'skku_notices'
LLM_MODEL = "gemini-1.5-flash-latest" 

VECTOR_SEARCH_K = 50  
BM25_SEARCH_K = 50   
FINAL_TOP_K = 5       
load_dotenv()
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
print(f"✅ [0단계] LLM ({LLM_MODEL}) 로드 완료.")



class QueryFilter(BaseModel):
    """LLM이 사용자의 쿼리를 분석하여 반환할 구조"""
    core_query: str = Field(description="검색 엔진(BM25, Vector)에 사용할 핵심 검색어")
    start_date: str = Field(description="필터링할 시작 날짜 (YYYY-MM-DD). 없으면 'N/A'.")
    end_date: str = Field(description="필터링할 종료 날짜 (YYYY-MM-DD). 없으면 'N/A'.")

def get_query_filter(llm: ChatGoogleGenerativeAI, user_query: str, today: str) -> QueryFilter:
    print(f"\n🏛️ [1단계] 쿼리 변환 시작 (기준일: {today})...")
    print(f"  - 원본 쿼리: \"{user_query}\"")

    structured_llm = llm.with_structured_output(QueryFilter)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
당신은 사용자 쿼리를 분석하여 검색 필터로 변환하는 AI입니다.
오늘 날짜는 {today}입니다.

지침:
1. '오늘', '내일', '이번 주' 같은 상대적 날짜를 {today} 기준으로 'YYYY-MM-DD' 형식의 절대 날짜로 변환하세요.
2. 날짜 범위가 명시되지 않으면 start_date와 end_date 모두 "N/A"로 응답하세요.
3. '오늘까지' = start_date: "N/A", end_date: {today}
4. '오늘부터' = start_date: {today}, end_date: "N/A"
5. '오늘 하는' = start_date: {today}, end_date: {today}
6. 날짜와 무관한 키워드는 'core_query'로 추출하세요.

[쿼리 예시]
"어제 끝난 행사 알려줘" (오늘이 2025-11-17이면)
{{ "core_query": "행사", "start_date": "N/A", "end_date": "2025-11-16" }}

"신청할 수 있는 장학금"
{{ "core_query": "신청 장학금", "start_date": "N/A", "end_date": "N/A" }}
"""),
        ("human", f"쿼리: \"{user_query}\"")
    ])

    query_transformer_chain = prompt | structured_llm

    try:
        filter_obj = query_transformer_chain.invoke({"user_query": user_query})
        print(f"  - 변환 완료: [Query: \"{filter_obj.core_query}\", Start: {filter_obj.start_date}, End: {filter_obj.end_date}]")
        return filter_obj
    except Exception as e:
        print(f"  - 쿼리 변환 실패: {e}. 기본값으로 검색합니다.")
        return QueryFilter(core_query=user_query, start_date="N/A", end_date="N/A")


def load_indexes():
    """디스크에서 BM25, ChromaDB, all_chunks를 로드합니다."""
    print("\n🏛️ [2단계] 로컬 인덱스 로드 중...")
    
    try:
        with open(BM25_MODEL_PATH, 'rb') as f:
            bm25 = pickle.load(f)
        print(f"  - BM25 모델 로드 완료 ({BM25_MODEL_PATH})")
    except FileNotFoundError:
        print(f"❌ 오류: BM25 모델 파일({BM25_MODEL_PATH})을 찾을 수 없습니다.")
        return None, None, None

    try:
        with open(CHUNK_DATA_PATH, 'rb') as f:
            all_chunks = pickle.load(f)
        all_chunks_map = {chunk['id']: chunk for chunk in all_chunks}
        print(f"  - 원본 청크 데이터 로드 완료 ({len(all_chunks)}개 청크)")
    except FileNotFoundError:
        print(f"❌ 오류: 청크 데이터 파일({CHUNK_DATA_PATH})을 찾을 수 없습니다.")
        return None, None, None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        chroma_collection = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        ).as_retriever(search_kwargs={"k": VECTOR_SEARCH_K})
        print(f"  - ChromaDB Vector Store 로드 완료 ({CHROMA_DB_PATH})")
    except Exception as e:
        print(f"❌ 오류: ChromaDB 로드 실패: {e}")
        return None, None, None

    return bm25, all_chunks, all_chunks_map, chroma_collection


def is_date_in_range(chunk_meta: dict, start_filter: str, end_filter: str) -> bool:
    """
    'N/A'를 고려하여 청크의 메타데이터가 날짜 범위 필터를 통과하는지 확인합니다.
    """
    chunk_start = chunk_meta.get('start_date', 'N/A')
    chunk_end = chunk_meta.get('end_date', 'N/A')

    if start_filter != 'N/A':
        if chunk_end != 'N/A' and chunk_end < start_filter:
            return False

    if end_filter != 'N/A':
        if chunk_start != 'N/A' and chunk_start > end_filter:
            return False

    return True


def hybrid_search(query_filter: QueryFilter, collection, bm25, all_chunks, all_chunks_map):
    print("\n🏛️ [3단계] 하이브리드 검색 및 필터링 시작...")
    
    start_f = query_filter.start_date
    end_f = query_filter.end_date
    core_query = query_filter.core_query

    print(f"  - (A) Vector DB 검색 (K={VECTOR_SEARCH_K})...")
    vector_results = collection.invoke(core_query) 
    
    vector_filtered = {}
    for doc in vector_results:
        chunk_id = doc.metadata.get('id', doc.metadata.get('doc_id'))
        if not chunk_id: Warning("Chunk ID not found in metadata")
        
        if is_date_in_range(doc.metadata, start_f, end_f):
            vector_filtered[chunk_id] = doc.metadata.get('_score', 1.0)

    print(f"    - Vector 결과: {len(vector_results)}개 중 {len(vector_filtered)}개 필터 통과.")

    print(f"  - (B) BM25 검색 (전체 {len(all_chunks)}개 스캔)...")
    tokenized_query = core_query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    bm25_filtered = {}
    for score, chunk in zip(bm25_scores, all_chunks):
        if score > 0:
            if is_date_in_range(chunk['metadata'], start_f, end_f):
                bm25_filtered[chunk['id']] = score
    
    print(f"    - BM25 결과: {len(all_chunks)}개 중 {len(bm25_filtered)}개 필터 통과 (점수 > 0).")

    print(f"  - (C) RRF 융합 중...")
    
    vec_ranked = sorted(vector_filtered.items(), key=lambda item: item[1], reverse=True)
    bm25_ranked = sorted(bm25_filtered.items(), key=lambda item: item[1], reverse=True)

    rrf_scores = {}
    k = 60 

    for rank, (chunk_id, score) in enumerate(vec_ranked):
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
        rrf_scores[chunk_id] += 1 / (k + rank)

    for rank, (chunk_id, score) in enumerate(bm25_ranked):
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
        rrf_scores[chunk_id] += 1 / (k + rank)

    fused_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    
    print(f"  - 융합 완료: 최종 {len(fused_results)}개 청크 후보.")
    
    final_chunks_text = []
    for chunk_id, score in fused_results[:FINAL_TOP_K]:
        if chunk_id in all_chunks_map:
            chunk = all_chunks_map[chunk_id]
            chunk_text = f"--- [출처: {chunk['metadata']['title']}]\n"
            chunk_text += f"{chunk['text']}\n"
            final_chunks_text.append(chunk_text)
        
    return "\n\n".join(final_chunks_text)


def generate_answer(llm: ChatGoogleGenerativeAI, context: str, user_query: str):
    print("\n🏛️ [4단계] RAG 답변 생성 시작...")

    if not context:
        print("  - 검색된 컨텍스트가 없습니다. LLM이 자체적으로 답변합니다.")
        context = "검색된 관련 공지사항이 없습니다."

    prompt = ChatPromptTemplate.from_template(f"""
당신은 성균관대학교 공지사항 안내 AI입니다.
제공된 [공지사항 컨텍스트]를 바탕으로 [사용자 쿼리]에 대해 친절하게 답변하세요.

지침:
1. 컨텍스트에 근거하여 답변해야 합니다.
2. 컨텍스트에 내용이 없으면 "관련 공지사항을 찾지 못했습니다."라고 솔직하게 답변하세요.
3. 답변 시, 근거가 된 공지사항의 [출처: 제목]을 명확히 언급해야 합니다.
4. 답변 마지막에, 관련 공지사항의 원본 링크(컨텍스트의 'URL')를 "자세히 보기" 목록으로 제공하세요.

---
[공지사항 컨텍스트]
{context}
---
[사용자 쿼리]
{user_query}
---

[AI 답변]
""")

    rag_chain = prompt | llm | StrOutputParser()
    
    response = rag_chain.invoke({"context": context, "user_query": user_query})
    
    print("✅ [5단계] 최종 답변 생성 완료.")
    return response


if __name__ == "__main__":
    
    USER_QUERY = "오늘 신청할 수 있는 융합연구학점제 공지 있어?"
    
    print("="*50)
    print(f"RAG 파이프라인 시작 (기준일: {TODAY_STR})")
    print(f"사용자 쿼리: \"{USER_QUERY}\"")
    print("="*50)

    bm25_model, all_chunks_list, chunks_map, chroma_retriever = load_indexes()
    
    if bm25_model:
        query_filter = get_query_filter(llm, USER_QUERY, TODAY_STR)
        
        context_string = hybrid_search(
            query_filter, 
            chroma_retriever, 
            bm25_model, 
            all_chunks_list,
            chunks_map
        )
        
        final_response = generate_answer(llm, context_string, USER_QUERY)
        
        print("\n" + "="*50)
        print("[최종 생성 답변]")
        print("="*50)
        print(final_response)
    else:
        print("오류: 인덱스 파일 로드에 실패하여 RAG 파이프라인을 실행할 수 없습니다.")
        print("먼저 build_index.py를 실행했는지 확인하세요.")