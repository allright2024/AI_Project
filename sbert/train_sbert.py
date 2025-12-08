import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import itertools
from samplers import NoDuplicateTitleBatchSampler

BASE_MODEL_NAME = "jhgan/ko-sbert-nli"
TRAIN_DATA_FILE = "./train.json"
OUTPUT_ROOT = "./output/new_data"
NUM_EPOCHS = 10 
MAX_SEQ_LENGTH = 512
VAL_RATIO = 0.1
PHYSICAL_BATCH_SIZE = 32

LEARNING_RATES = [5e-5]
BATCH_SIZES = [128]
BATCH_SIZES = BATCH_SIZES[::-1]
 

def train_sbert():
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"오류: '{TRAIN_DATA_FILE}' 파일이 없습니다.")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with open(TRAIN_DATA_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f">>> 총 {len(raw_data)}개의 데이터를 로드했습니다.")

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
            example = InputExample(texts=[question, passage])
            example.title = title
            all_examples.append(example)

    all_examples_np = np.array(all_examples)

    results = {}

    for lr, target_batch_size in itertools.product(LEARNING_RATES, BATCH_SIZES):
        print(f"\n{'#'*40}")
        print(f"Testing Hyperparameters: LR={lr}, BatchSize={target_batch_size}")
        print(f"{'#'*40}")

        if target_batch_size > PHYSICAL_BATCH_SIZE:
            accumulation_steps = target_batch_size // PHYSICAL_BATCH_SIZE
            actual_batch_size = PHYSICAL_BATCH_SIZE
            print(f"  -> Target Batch {target_batch_size} > Physical {PHYSICAL_BATCH_SIZE}. Using Gradient Accumulation: {accumulation_steps} steps.")
        else:
            accumulation_steps = 1
            actual_batch_size = target_batch_size
            print(f"  -> Using standard Batch Size: {actual_batch_size}")

        train_samples, val_samples = train_test_split(all_examples, test_size=VAL_RATIO, random_state=42)
        
        print(f"  -> Training set: {len(train_samples)}, Validation set: {len(val_samples)}")

        model = SentenceTransformer(BASE_MODEL_NAME)
        model.max_seq_length = MAX_SEQ_LENGTH
        
        train_sampler = NoDuplicateTitleBatchSampler(train_samples, batch_size=actual_batch_size)
        train_dataloader = DataLoader(train_samples, batch_sampler=train_sampler, collate_fn=model.smart_batching_collate)

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
            val_queries, val_corpus, val_relevant_docs, name=f'val_split'
        )

        train_loss = losses.MultipleNegativesRankingLoss(model=model)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        model.to(model.device)
        best_score = -1.0
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss.zero_grad()
            train_loss.train()
            
            epoch_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                features, labels = batch
                
                for feature in features:
                    for key, value in feature.items():
                        feature[key] = value.to(model.device)
                
                if labels is not None and torch.is_tensor(labels):
                    labels = labels.to(model.device)

                loss_value = train_loss(features, labels)
                
                loss_value = loss_value / accumulation_steps
                loss_value.backward()
                
                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss_value.item() * accumulation_steps 

            score = evaluator(model)
            
            if isinstance(score, dict):
                target_metric = None
                for key in score.keys():
                    if 'map@100' in key and 'cosine' in key:
                        target_metric = key
                        break
                
                if target_metric is None:
                    for key in score.keys():
                        if 'map@100' in key:
                            target_metric = key
                            break
                
                if target_metric is None:
                    for key in score.keys():
                        if 'map' in key:
                            target_metric = key
                            break
                            
                if target_metric:
                    score = score[target_metric]
                else:
                    score = list(score.values())[0]

            print(f"    Epoch {epoch+1}: Loss={epoch_loss/len(train_dataloader):.4f}, Val Score={score:.4f}")
            
            if score > best_score:
                best_score = score
                checkpoint_path = os.path.join(OUTPUT_ROOT, f"lr_{lr}_bs_{target_batch_size}_best")
                model.save(checkpoint_path)
                print(f"    -> Saved best model to {checkpoint_path}")
        
        print(f"    -> Best Val Score (MAP@100): {best_score:.4f}")
        results[(lr, target_batch_size)] = best_score

    visualize_results(results)

def visualize_results(results):
    print("\nGenerating Visualization...")
    
    lrs = sorted(list(set(k[0] for k in results.keys())))
    batches = sorted(list(set(k[1] for k in results.keys())))
    
    scores = np.zeros((len(lrs), len(batches)))
    
    for i, lr in enumerate(lrs):
        for j, batch in enumerate(batches):
            scores[i, j] = results.get((lr, batch), 0)
            
    plt.figure(figsize=(10, 6))
    plt.imshow(scores, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='MAP@100 Score')
    
    plt.xticks(np.arange(len(batches)), batches)
    plt.yticks(np.arange(len(lrs)), lrs)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Learning Rate')
    plt.title('Hyperparameter Tuning Results (MAP@100)')
    
    for i in range(len(lrs)):
        for j in range(len(batches)):
            plt.text(j, i, f"{scores[i, j]:.4f}", ha="center", va="center", color="w")
            
    output_img = "hyperparameter_tuning_result.png"
    plt.savefig(output_img)
    print(f"Visualization saved to {output_img}")

if __name__ == "__main__":
    train_sbert()
