import os
import json
import glob
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import time

# 환경 변수 로드 (.env 파일)
load_dotenv()

USER_MODEL_REQUREST = "gemini-2.5-flash-lite" 
JSON_FILE_DIR = "./crawl" # 크롤링한 파일 저장 위치
JSON_FILE_NAME = "skku_*_posts.json" # 크롤링한 파일 이름 형식

# find all files with format skku_*_posts.json
pattern = os.path.join(JSON_FILE_DIR, JSON_FILE_NAME)
files = glob.glob(pattern)

try:
    llm = ChatGoogleGenerativeAI(
        model=USER_MODEL_REQUREST, 
        temperature=0,
        transport="rest" 
    )
    print(f"✅ LLM 모델('{USER_MODEL_REQUREST}') 로드 완료")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit()

def load_notices(file_path):
    """JSON 파일 로드 함수"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # [수정 1] f -> file 로 수정
            notices = json.load(file)
        print(f"✅ 공지사항 데이터 로드 완료: {file_path} ({len(notices)}개)")
        return notices
    except Exception as e:
        # [수정 2] 에러 원인 출력
        print(f"❌ 파일 로드 실패 ('{file_path}'): {e}")
        return None
    
class ExtractedDates(BaseModel):
    """추출된 시작 날짜와 종료 날짜를 담는 구조"""
    start_date: str = Field(description="공지사항의 시작 날짜 (YYYY-MM-DD 형식). 없으면 'N/A'.")
    end_date: str = Field(description="공지사항의 종료 날짜 (YYYY-MM-DD 형식). 없으면 'N/A'.")
    
try:
    structured_llm = llm.with_structured_output(ExtractedDates)
except AttributeError as e:
    print(f"❌ structured output 설정 오류: {e}")
    exit()
    
extraction_template = """
당신은 텍스트에서 날짜 정보를 추출하는 전문 AI입니다.
주어진 [공지사항 텍스트]를 분석하여 '시작 날짜(start_date)'와 '종료 날짜(end_date)'를 추출하십시오.

지침:
1. 날짜는 반드시 'YYYY-MM-DD' 형식으로 정규화해야 합니다. (예: 2025. 11. 15. -> 2025-11-15)
2. 날짜가 명확히 언급되지 않거나 "상시" 등 특정할 수 없는 경우 'N/A'로 응답합니다.
3. '마감일', '~까지' 등은 '종료 날짜(end_date)'로 간주합니다.
4. '신청 기간: A ~ B'인 경우, A는 '시작 날짜', B는 '종료 날짜'입니다.
5. 텍스트에 기준 연도가 명시되지 않으면, [기준 연도]를 참고하여 YYYY를 결정하십시오.
6. 본문의 'dates' 필드에 '최종 수정일'이 있다면, 이는 마감일이 아니라 문서의 수정일이므로 무시합니다.
7. 마감일이 여러 개 나열된 경우(예: 서울대 (마감), 제주대 (마감)), 본문의 주요 마감일을 찾으십시오. 찾기 어려우면 'N/A'로 응답합니다.

---
[기준 연도]
2025년

[공지사항 텍스트]
제목: {title}
본문: {content}
(참고 dates 필드: {dates})
---

[추출 결과 (JSON)]
"""

prompt = ChatPromptTemplate.from_template(extraction_template)
extraction_chain = prompt | structured_llm

def main():
    for file in files:
        notices = load_notices(file)
        notices = notices["posts"]
        
        if notices:
            augmented_results = []
            
            print("\n" + "="*50)
            print(f"--- [단계 3] LLM 기반 날짜 추출 실행 (총 {len(notices)}개) ---")
            
            for i, notice in enumerate(notices): 
                title = notice.get('title', '')
                content = notice.get('content', '')
                dates_field = notice.get('dates', [])
                
                print(f"\n[{i+1}/{len(notices)}] 처리 중: \"{title[:30]}...\"")
                
                try:
                    input_data = {
                        "title": title, 
                        "content": content,
                        "dates": str(dates_field) 
                    }
                    
                    extracted_data = extraction_chain.invoke(input_data)

                    time.sleep(3) # To prevent hitting free-tier rate limit (remove if using paid version)
                    
                    notice['start_date'] = extracted_data.start_date
                    notice['end_date'] = extracted_data.end_date
                    augmented_results.append(notice)
                    
                    print(f"  ▶ [결과] Start: {extracted_data.start_date} / End: {extracted_data.end_date}")
                    break

                except Exception as e:
                    print(f"  ❌ [오류] 문서 처리 실패: {e}")

            output_path = file.replace("posts", "augmented").replace(JSON_FILE_DIR,".")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(augmented_results, f, indent=4, ensure_ascii=False)
                
            print("\n" + "="*50)
            print(f"✅ 모든 작업 완료. 결과가 '{output_path}'에 저장되었습니다.")
        else:
            print("⚠️ 처리할 데이터가 없습니다. 파일 경로를 확인하세요.")

main() # -> 따로 호출하여 수행
