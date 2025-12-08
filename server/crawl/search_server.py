import os
import json
import pickle
import asyncio
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(DIR, 'skku_notice_db')
BM25_MODEL_PATH = os.path.join(DIR, 'bm25_model.pkl')
CHUNK_DATA_PATH = os.path.join(DIR, 'all_chunks.pkl')
COLLECTION_NAME = 'skku_notices'
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli"

app = FastAPI()

chroma_collection = None
bm25_model = None
all_chunks = None
sbert_model = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict

@app.on_event("startup")
async def startup_event():
    global chroma_collection, bm25_model, all_chunks, sbert_model
    
    print("Loading models...")
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device='cpu'
    )
    chroma_collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    
    with open(BM25_MODEL_PATH, 'rb') as f:
        bm25_model = pickle.load(f)
    with open(CHUNK_DATA_PATH, 'rb') as f:
        all_chunks = pickle.load(f)
    print("✅ BM25 & Chunks loaded.")
    
    
    print("🚀 Search Server Ready.")

def extract_date_filter(query: str) -> dict:
    """
    Uses LLM to extract date range from query.
    Returns a filter dict for ChromaDB or Python filtering.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
    Current Date: {current_date}
    User Query: "{query}"
    
    Extract the date range implied by the user query.
    If the user asks for "this week", calculate the start and end date of this week.
    If "today", use today's date.
    If no date is specified, return null.
    
    Return JSON format:
    {{
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD"
    }}
    If no date, return {{ "start_date": null, "end_date": null }}
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        date_info = json.loads(text)
        return date_info
    except Exception as e:
        print(f"LLM Date Extraction Error: {e}")
        return {"start_date": None, "end_date": None}

def hybrid_search(query: str, top_k: int, date_filter: dict) -> List[SearchResult]:
    global chroma_collection, bm25_model, all_chunks
    
    vector_k = top_k * 3
    results = chroma_collection.query(
        query_texts=[query],
        n_results=vector_k
    )
    
    vector_docs = {}
    if results['ids']:
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        
        for i, doc_id in enumerate(ids):
            score = 1 - distances[i] 
            vector_docs[doc_id] = {
                "score": score,
                "metadata": metadatas[i],
                "text": documents[i]
            }

    tokenized_query = query.split()
    bm25_scores = bm25_model.get_scores(tokenized_query)
    
    top_n_indices = np.argsort(bm25_scores)[::-1][:vector_k]
    
    bm25_docs = {}
    for idx in top_n_indices:
        chunk = all_chunks[idx]
        doc_id = chunk['id']
        score = bm25_scores[idx]
        bm25_docs[doc_id] = {
            "score": score,
            "metadata": chunk['metadata'],
            "text": chunk['text']
        }

    if bm25_docs:
        max_bm25 = max(d['score'] for d in bm25_docs.values())
        if max_bm25 > 0:
            for d in bm25_docs.values():
                d['score'] /= max_bm25

    combined_scores = {}
    all_ids = set(vector_docs.keys()) | set(bm25_docs.keys())
    
    for doc_id in all_ids:
        v_score = vector_docs.get(doc_id, {}).get('score', 0)
        b_score = bm25_docs.get(doc_id, {}).get('score', 0)
        
        final_score = (0.7 * v_score) + (0.3 * b_score)
        
        info = vector_docs.get(doc_id) or bm25_docs.get(doc_id)
        
        combined_scores[doc_id] = {
            "score": final_score,
            "info": info
        }

    filtered_results = []
    start_date = date_filter.get("start_date")
    end_date = date_filter.get("end_date")
    
    for doc_id, data in combined_scores.items():
        info = data['info']
        doc_start = info['metadata'].get('start_date')
        doc_end = info['metadata'].get('end_date')
        
        is_valid = True
        if start_date and end_date:
            if doc_start != "N/A" and doc_end != "N/A":
                if doc_end < start_date or doc_start > end_date:
                    is_valid = False
            
        if is_valid:
            filtered_results.append(SearchResult(
                id=doc_id,
                score=data['score'],
                text=info['text'],
                metadata=info['metadata']
            ))
            
    filtered_results.sort(key=lambda x: x.score, reverse=True)
    
    return filtered_results[:top_k]

@app.post("/search", response_model=List[SearchResult])
async def search_endpoint(request: SearchRequest):
    print(f"Search Query: {request.query}")
    
    date_filter = extract_date_filter(request.query)
    print(f"Extracted Date Filter: {date_filter}")
    
    results = hybrid_search(request.query, request.top_k, date_filter)
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
