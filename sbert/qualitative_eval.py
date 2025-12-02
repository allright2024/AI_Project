import json
import os
import torch
from sentence_transformers import SentenceTransformer, util

BASE_MODEL_NAME = "jhgan/ko-sbert-nli"
FINE_TUNED_MODEL_PATH = "./output/grid_search/lr_5e-05_bs_128_best"
DATA_FILE = "/Users/yongkim/Desktop/playground/AI_Project/data_augmentation/rag_dataset_with_questions.json"
TOP_K = 5

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    corpus = []
    queries = []
    
    for entry in data:
        metadata = entry.get('metadata', {})
        title = metadata.get('title', '').strip()
        text_content = entry.get('text', '').strip()
        
        if text_content.startswith("제목:") and "| 내용:" in text_content:
            try:
                _, body = text_content.split("| 내용:", 1)
                body = body.strip()
            except ValueError:
                body = text_content
        else:
            body = text_content
            
        full_text = f"[제목]: {title} [본문]: {body}"
        corpus.append(full_text)
        
        if 'question' in entry and entry['question']:
            queries.append(entry['question'])
            
    return corpus, queries

def qualitative_evaluation():
    print(f"Loading Base Model: {BASE_MODEL_NAME}...")
    base_model = SentenceTransformer(BASE_MODEL_NAME)
    
    print(f"Loading Fine-Tuned Model: {FINE_TUNED_MODEL_PATH}...")
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        print(f"Error: Fine-tuned model path '{FINE_TUNED_MODEL_PATH}' does not exist.")
        return
    fine_tuned_model = SentenceTransformer(FINE_TUNED_MODEL_PATH)

    print(f"Loading Data from {DATA_FILE}...")
    corpus, all_queries = load_data(DATA_FILE)
    
    import random
    random.seed(42)
    test_queries = random.sample(all_queries, 5) 
    

    print(f"Encoding Corpus ({len(corpus)} documents)...")
    base_corpus_embeddings = base_model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
    ft_corpus_embeddings = fine_tuned_model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

    print("\n" + "="*80)
    print("QUALITATIVE EVALUATION RESULTS")
    print("="*80)

    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-" * 80)
        
        base_results = util.semantic_search(base_model.encode(query, convert_to_tensor=True), base_corpus_embeddings, top_k=TOP_K)[0]
        
        ft_results = util.semantic_search(fine_tuned_model.encode(query, convert_to_tensor=True), ft_corpus_embeddings, top_k=TOP_K)[0]
        
        print(f"{'[Base Model]':<40} | {'[Fine-Tuned Model]':<40}")
        print("-" * 80)
        
        for i in range(TOP_K):
            base_doc_idx = base_results[i]['corpus_id']
            base_score = base_results[i]['score']
            base_text = corpus[base_doc_idx][:35] + "..."
            
            ft_doc_idx = ft_results[i]['corpus_id']
            ft_score = ft_results[i]['score']
            ft_text = corpus[ft_doc_idx][:35] + "..." 
            
            print(f"{i+1}. {base_text:<35} ({base_score:.4f}) | {i+1}. {ft_text:<35} ({ft_score:.4f})")

if __name__ == "__main__":
    qualitative_evaluation()
