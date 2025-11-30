import json
import time
import os
import google.generativeai as genai
from tqdm import tqdm
import random

API_KEY = "" 
INPUT_FILE = "final_rag_chunks.json"
OUTPUT_FILE = "rag_dataset_with_questions.json"
DELAY_SECONDS = 4 

genai.configure(api_key=API_KEY, transport='rest')

model = genai.GenerativeModel('gemini-2.5-flash')

def generate_question_from_chunk(title, text):
    """제목과 본문을 보고 사용자가 검색할 법한 질문 1개 생성"""
    
    prompt = f"""
    너는 대학교 공지사항 검색 서비스의 사용자 시뮬레이터야.
    아래 [공지사항]을 보고, 이 정보를 찾으려는 사용자가 입력할 법한 '검색 쿼리(질문)'를 딱 1개만 만들어줘.
    
    [질문 생성 전략]
    1. **다양성 확보**: 단순히 "마감일이 언제야?" 같은 **구체적인 사실**을 묻는 질문만 만들지 마.
    2. **포괄적 질문**: 본문이 '장학금', '모집', '행사', '특강' 등 굵직한 주제를 다루고 있다면, **"장학금 관련 공지 알려줘"**, **"요즘 취업 관련 프로그램 뭐 있어?"** 같이 **범용적인 질문**도 확률적으로 섞어서 생성해줘.
    3. **자연스러움**: 대학생이 실제 검색창이나 챗봇에 칠 법한 자연스러운 구어체를 사용해.

    [공지사항 정보]
    - 제목: {title}
    - 내용: {text}
    
    [출력 예시 (둘 중 하나 스타일로 생성)]
    - Case A (구체적): "2025년 1학기 성적 우수 장학금 커트라인 몇 점이야?"
    - Case B (포괄적): "이번 학기 장학금 신청 공지사항 보여줘"
    
    [최종 출력 포맷]
    {{ "generated_question": "생성된 질문 텍스트" }}
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        result = json.loads(response.text)
        return result.get("generated_question", "")
    except Exception as e:
        
        print(f" [Error] 질문 생성 실패: {e}")
        return ""

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"오류: '{INPUT_FILE}' 파일이 없습니다.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f">>> 총 {len(data)}개의 청크를 처리합니다.")
    
    random.shuffle(data)
    target_data = data[:100] 
    processed_data = []
    
    print(f">>> 질문 생성 시작... {len(target_data)}개")

    for entry in tqdm(target_data):
        new_entry = entry.copy()
        question = generate_question_from_chunk(entry.get('title', ''), entry.get('text', ''))
        
        if question:
            new_entry['question'] = question
            processed_data.append(new_entry)
        else:
            new_entry['question'] = "" 
            processed_data.append(new_entry)
            
        time.sleep(DELAY_SECONDS)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"\n>>> [완료] '{OUTPUT_FILE}' 파일 생성 완료.")
    
    if processed_data:
        print("\n[결과 샘플]")
        print(json.dumps(processed_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
