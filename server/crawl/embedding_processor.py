import json
import os
import glob
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from tqdm import tqdm 

DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_DB_PATH = os.path.join(DIR, 'skku_notice_db')
BM25_MODEL_PATH = os.path.join(DIR, 'bm25_model.pkl')
CHUNK_DATA_PATH = os.path.join(DIR, 'all_chunks.pkl')

COLLECTION_NAME = 'skku_notices'
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_all_augmented_notices():
    """Loads skku_1500.json."""
    pattern = os.path.join(DIR, "skku_1500.json")
    files = glob.glob(pattern)
    
    all_notices = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notices = json.load(f)
                all_notices.extend(notices)
            print(f"✅ Loaded {len(notices)} notices from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Failed to load {os.path.basename(file_path)}: {e}")
            
    print(f"✅ Total {len(all_notices)} notices loaded.")
    return all_notices

def run_chunking(notices):
    """Chunks the notices."""
    print("\n🏛️ Chunking started...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_chunks = []
    
    for doc_id, notice in enumerate(tqdm(notices, desc="  Chunking")):
        text_to_split = notice.get('content', '')
        if not text_to_split:
            continue
            
        metadata = {
            "url": notice.get("post_link", ""), # Changed from 'url' to 'post_link' to match crawler output
            "title": notice.get("title", ""),
            "start_date": notice.get("start_date", "N/A"),
            "end_date": notice.get("end_date", "N/A"),
            "department": notice.get("department", "Unknown"),
            "category": notice.get("category", "")
        }
        
        chunks_texts = text_splitter.split_text(text_to_split)
        
        for chunk_index, chunk_text in enumerate(chunks_texts):
            formatted_text = f"제목: {metadata['title']} | 내용: {chunk_text}"
            
            chunk_data = {
                "id": f"doc_{doc_id}_chunk_{chunk_index}",
                "text": formatted_text,
                "metadata": metadata
            }
            all_chunks.append(chunk_data)

    print(f"✅ {len(notices)} documents split into {len(all_chunks)} chunks.")
    return all_chunks

def run_hybrid_indexing(all_chunks):
    """Indexes chunks into ChromaDB and trains BM25."""
    print("\n🏛️ Hybrid indexing started...")

    print("  [1] Vector DB (ChromaDB) indexing...")
    
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device='cpu' 
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )

    documents = [chunk['text'] for chunk in all_chunks]
    metadatas = [chunk['metadata'] for chunk in all_chunks]
    ids = [chunk['id'] for chunk in all_chunks]

    batch_size = 5000
    for i in tqdm(range(0, len(ids), batch_size), desc="  Chroma Upsert"):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
    
    print(f"  ✅ Vector DB indexing complete. Saved to '{CHROMA_DB_PATH}'.")

    print("  [2] BM25 indexing...")

    print("    - Tokenizing corpus...")
    tokenized_corpus = [chunk['text'].split() for chunk in tqdm(all_chunks)]

    print("    - Training BM25 model...")
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_MODEL_PATH, 'wb') as f_bm25:
        pickle.dump(bm25, f_bm25)
        print(f"  ✅ BM25 model saved to '{BM25_MODEL_PATH}'.")
        
    with open(CHUNK_DATA_PATH, 'wb') as f_chunks:
        pickle.dump(all_chunks, f_chunks)
        print(f"  ✅ Chunk data saved to '{CHUNK_DATA_PATH}'.")

    json_path = os.path.join(DIR, 'final_rag_chunks.json')
    with open(json_path, 'w', encoding='utf-8') as f_json:
        json.dump(all_chunks, f_json, indent=4, ensure_ascii=False)
        print(f"  ✅ Chunk data saved to '{json_path}'.")
    
    print(f"✅ Hybrid indexing complete.")

def run_embedding_pipeline():
    """Runs the full embedding pipeline."""
    print("Starting embedding pipeline...")
    notices = load_all_augmented_notices()
    if notices:
        chunks = run_chunking(notices)
        run_hybrid_indexing(chunks)
        print("Embedding pipeline finished successfully.")
    else:
        print("No notices found to process.")

if __name__ == "__main__":
    run_embedding_pipeline()
