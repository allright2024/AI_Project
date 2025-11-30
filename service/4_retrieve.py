import os
import pickle
import json
import re
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from rank_bm25 import BM25Okapi
import numpy as np

load_dotenv()

TODAY_STR = datetime.now().strftime('%Y-%m-%d')

CHROMA_DB_PATH = './skku_notice_db'
BM25_MODEL_PATH = 'bm25_model.pkl'
CHUNK_DATA_PATH = 'all_chunks.pkl'

EMBEDDING_MODEL = "jhgan/ko-sbert-nli"
COLLECTION_NAME = 'skku_notices'
LLM_MODEL = "gemini-2.5-flash-lite"

VECTOR_SEARCH_K = 50
BM25_SEARCH_K = 50
FINAL_TOP_K = 5

llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
print(f"✅ [0단계] LLM ({LLM_MODEL}) 로드 완료.")


class QueryFilter(BaseModel):
    """LLM이 사용자의 쿼리를 분석하여 반환할 구조"""
    core_query: str = Field(description="검색 엔진(BM25, Vector)에 사용할 핵심 검색어")
    start_date: str = Field(description="필터링할 시작 날짜 (YYYY-MM-DD). 없으면 'N/A'.")
    end_date: str = Field(description="필터링할 종료 날짜 (YYYY-MM-DD). 없으면 'N/A'.")

def get_query_filter(llm: ChatGoogleGenerativeAI, user_query: str, today: str) -> QueryFilter:
    print(f"\n🏛️ [1단계] 쿼리 변환 시작 (기준일: {today})...")
    
    structured_llm = llm.with_structured_output(QueryFilter)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
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
-> {{ "core_query": "행사", "start_date": "N/A", "end_date": "2025-11-16" }}

"신청할 수 있는 장학금"
-> {{ "core_query": "신청 장학금", "start_date": "N/A", "end_date": "N/A" }}
"""),
        ("human", "쿼리: \"{user_query}\"")
    ])
    
    query_transformer_chain = prompt | structured_llm
    
    try:
        filter_obj = query_transformer_chain.invoke({
            "user_query": user_query, 
            "today": today
        })
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
        return None, None, None, None

    try:
        with open(CHUNK_DATA_PATH, 'rb') as f:
            all_chunks = pickle.load(f)
        all_chunks_map = {chunk['id']: chunk for chunk in all_chunks}
        print(f"  - 원본 청크 데이터 로드 완료 ({len(all_chunks)}개 청크)")
    except FileNotFoundError:
        print(f"❌ 오류: 청크 데이터 파일({CHUNK_DATA_PATH})을 찾을 수 없습니다.")
        return None, None, None, None

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
        )
        print(f"  - ChromaDB Vector Store 로드 완료 ({CHROMA_DB_PATH})")
    except Exception as e:
        print(f"❌ 오류: ChromaDB 로드 실패: {e}")
        return None, None, None, None

    return bm25, all_chunks, all_chunks_map, chroma_collection


def build_chroma_filter(start_f: str, end_f: str) -> dict:
    """
    ChromaDB 검색용 메타데이터 필터(Where 절)를 생성합니다. (Pre-Filtering용)
    """
    conditions = []

    if start_f != "N/A":
        conditions.append({
            "$or": [
                {"end_date": {"$gte": start_f}},
                {"end_date": {"$eq": "N/A"}}
            ]
        })

    if end_f != "N/A":
        conditions.append({
            "$or": [
                {"start_date": {"$lte": end_f}},
                {"start_date": {"$eq": "N/A"}}
            ]
        })

    if not conditions:
        return None
    
    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def is_date_in_range(chunk_meta: dict, start_filter: str, end_filter: str) -> bool:
    """
    BM25 검색 결과 필터링을 위한 Python 함수 (Post-Filtering용)
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
    print("\n🏛️ [3단계] 하이브리드 검색 시작 (Pre-Filtering 적용)...")
    
    start_f = query_filter.start_date
    end_f = query_filter.end_date
    core_query = query_filter.core_query

    chroma_filter = build_chroma_filter(start_f, end_f)
    print(f"  - Chroma Filter 조건: {chroma_filter}")

    vector_results_with_score = collection.similarity_search_with_score(
        query=core_query,
        k=VECTOR_SEARCH_K, 
        filter=chroma_filter 
    )

    print(f"    - Vector(Pre-filtered) 결과: {len(vector_results_with_score)}개 반환됨.")

    print(f"  - BM25 검색 (전체 {len(all_chunks)}개 스캔)...")
    tokenized_query = core_query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    top_n_indices = np.argsort(bm25_scores)[::-1][:BM25_SEARCH_K * 3]

    bm25_filtered = {}
    for idx in top_n_indices:
        score = bm25_scores[idx]
        if score <= 0: continue
        
        chunk = all_chunks[idx]
        # 날짜 필터링 수행
        if is_date_in_range(chunk['metadata'], start_f, end_f):
            bm25_filtered[chunk['id']] = score

    print(f"    - BM25(Post-filtered) 결과: {len(bm25_filtered)}개 통과.")

    print(f"  - (C) RRF 융합 중...")
    
    rrf_scores = {}
    k = 60 

    for rank, (doc, score) in enumerate(vector_results_with_score):
        chunk_id = doc.metadata.get('id')
        if chunk_id:
            if chunk_id not in rrf_scores: rrf_scores[chunk_id] = 0
            rrf_scores[chunk_id] += 1 / (k + rank)

    bm25_sorted = sorted(bm25_filtered.items(), key=lambda item: item[1], reverse=True)
    for rank, (chunk_id, score) in enumerate(bm25_sorted):
        if chunk_id not in rrf_scores: rrf_scores[chunk_id] = 0
        rrf_scores[chunk_id] += 1 / (k + rank)

    fused_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    print(f"  - 융합 완료: 최종 {len(fused_results)}개 청크 후보.")
    
    final_chunks_text = []
    for chunk_id, score in fused_results[:FINAL_TOP_K]:
        if chunk_id in all_chunks_map:
            chunk = all_chunks_map[chunk_id]
            url = chunk['metadata'].get('url', '')
            
            chunk_text = f"--- [출처: {chunk['metadata']['title']}] ---\n"
            if url: chunk_text += f"[원본 링크: {url}]\n"
            chunk_text += f"{chunk['text']}\n"
            final_chunks_text.append(chunk_text)
        
    return "\n\n".join(final_chunks_text)


def generate_answer(llm: ChatGoogleGenerativeAI, context: str, user_query: str):
    print("\n🏛️ [4단계] RAG 답변 생성 시작...")

    if not context:
        print("  - 검색된 컨텍스트가 없습니다.")
        context = "검색된 관련 공지사항이 없습니다."

    prompt = ChatPromptTemplate.from_template(f"""
당신은 성균관대학교 공지사항 안내 AI입니다.
제공된 [공지사항 컨텍스트]를 바탕으로 [사용자 쿼리]에 대해 친절하게 답변하세요.

지침:
1. 반드시 제공된 [공지사항 컨텍스트]에 기반하여 답변하세요.
2. 컨텍스트에 없는 내용은 지어내지 말고 "관련 공지사항을 찾지 못했습니다"라고 하세요.
3. 답변 시 출처(제목)를 언급하고, 제공된 '원본 링크'가 있다면 답변 끝에 목록으로 정리해 주세요.
4. 사용자 질문이 '오늘 가능한' 것을 묻는다면, 날짜를 꼼꼼히 확인하세요.

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
    
    print("="*60)
    print(f"RAG 파이프라인 시작 (기준일: {TODAY_STR})")
    print(f"사용자 쿼리: \"{USER_QUERY}\"")
    print("="*60)

    # 인덱스 로드
    bm25_model, all_chunks_list, chunks_map, chroma_collection = load_indexes()
    
    if bm25_model and chroma_collection:
        query_filter = get_query_filter(llm, USER_QUERY, TODAY_STR)
        
        context_string = hybrid_search(
            query_filter, 
            chroma_collection, 
            bm25_model, 
            all_chunks_list,
            chunks_map
        )
        
        final_response = generate_answer(llm, context_string, USER_QUERY)
        
        print("\n" + "="*60)
        print("[최종 생성 답변]")
        print("="*60)
        print(final_response)
    else:
        print("오류: 인덱스 파일 로드에 실패하여 RAG 파이프라인을 실행할 수 없습니다.")
