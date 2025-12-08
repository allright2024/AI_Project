import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sklearn.model_selection import train_test_split

BASE_MODEL_NAME = "jhgan/ko-sbert-nli"
FINE_TUNED_MODEL_PATH = "./output/grid_search/lr_5e-05_bs_128_best"
DATA_FILE = "/Users/yongkim/Desktop/playground/AI_Project/data_augmentation/rag_dataset_with_questions_1000.json"
VAL_RATIO = 0.2
RANDOM_STATE = 42

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    all_examples = []
    for entry in raw_data:
        question = entry.get('question', '').strip()
        
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
            
        passage = f"[제목]: {title} [본문]: {body}"
        
        if question and passage:
            all_examples.append(InputExample(texts=[question, passage]))
            
    return all_examples

def evaluate_model(model_name_or_path, val_samples, name_prefix):
    print(f"\nEvaluating {name_prefix} ({model_name_or_path})...")
    
    try:
        model = SentenceTransformer(model_name_or_path)
    except Exception as e:
        print(f"Error loading model {model_name_or_path}: {e}")
        return

    val_corpus = {}
    val_queries = {}
    val_relevant_docs = {}
    
    for idx, ex in enumerate(val_samples):
        q_id = f"val_q_{idx}"
        d_id = f"val_d_{idx}"
        val_queries[q_id] = ex.texts[0]
        val_corpus[d_id] = ex.texts[1]
        val_relevant_docs[q_id] = {d_id}

    evaluator = InformationRetrievalEvaluator(
        val_queries, val_corpus, val_relevant_docs, name=name_prefix
    )
    
    results = evaluator(model)
    
    map_score = 0.0
    if isinstance(results, dict):
        for key in results.keys():
            if 'map@100' in key and 'cosine' in key:
                map_score = results[key]
                break
        if map_score == 0.0:
             for key in results.keys():
                if 'map@100' in key:
                    map_score = results[key]
                    break
    
    print(f"  -> MAP@100 Score: {map_score:.4f}")
    return map_score

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found.")
        return

    print("Loading and splitting data...")
    all_examples = load_data(DATA_FILE)
    
    _, val_samples = train_test_split(all_examples, test_size=VAL_RATIO, random_state=RANDOM_STATE)
    print(f"Validation set size: {len(val_samples)}")

    base_score = evaluate_model(BASE_MODEL_NAME, val_samples, "Base_Model")
    
    ft_score = evaluate_model(FINE_TUNED_MODEL_PATH, val_samples, "Fine_Tuned_Model")
    
    print("\n" + "="*40)
    print("COMPARISON RESULTS (MAP@100)")
    print("="*40)
    print(f"Base Model:       {base_score:.4f}")
    print(f"Fine-Tuned Model: {ft_score:.4f}")
    
    if ft_score > base_score:
        diff = ft_score - base_score
        print(f"Result: Fine-Tuned Model is better by +{diff:.4f}")
    else:
        diff = base_score - ft_score
        print(f"Result: Base Model is better by +{diff:.4f}")

if __name__ == "__main__":
    main()
