import json
import os
import random
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

BASE_MODEL_NAME = "jhgan/ko-sbert-nli" # base sbert 모델
TRAIN_DATA_FILE = "../data_augmentation/rag_dataset_with_questions_1500.json" 
OUTPUT_PATH = "./output/fine-tuned-sbert"   
BATCH_SIZE = 10      
NUM_EPOCHS = 10      
MAX_SEQ_LENGTH = 512  
VAL_RATIO = 0.1       # 검증 데이터 비율 (10%)


def train_sbert():
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"오류: '{TRAIN_DATA_FILE}' 파일이 없습니다.")
        return

    with open(TRAIN_DATA_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f">>> 총 {len(raw_data)}개의 데이터를 로드했습니다.")

    all_examples = []
    for entry in raw_data:
        question = entry.get('question', '').strip()
        passage = f"[제목]: {entry.get('title', '')} [본문]: {entry.get('text', '')}"
        
        if question and passage:
            all_examples.append(InputExample(texts=[question, passage]))

    random.seed(42) 
    random.shuffle(all_examples)

    val_count = int(len(all_examples) * VAL_RATIO)
    train_samples = all_examples[val_count:] # 90%
    val_samples = all_examples[:val_count]   # 10%

    print(f">>> 학습 데이터: {len(train_samples)}개 / 검증 데이터: {len(val_samples)}개")

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)

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
        val_queries, 
        val_corpus, 
        val_relevant_docs,
        name='rag_val_evaluator',
        show_progress_bar=True
    )

    print(f">>> 베이스 모델 '{BASE_MODEL_NAME}' 로드 중...")
    model = SentenceTransformer(BASE_MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    
    train_loss = losses.MultipleNegativesRankingLoss(model=model)


    print(f">>> 학습 시작 (Epochs: {NUM_EPOCHS}) ...")
    
    eval_steps = len(train_dataloader) 
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=int(len(train_dataloader) * NUM_EPOCHS * 0.1),
        output_path=OUTPUT_PATH,
        evaluator=evaluator,
        evaluation_steps=eval_steps,
        save_best_model=True,
        show_progress_bar=True
    )

    print(f"\n>>> [완료] 학습 종료. 최고 성능 모델이 '{OUTPUT_PATH}'에 저장되었습니다.")

if __name__ == "__main__":
    train_sbert()
