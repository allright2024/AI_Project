import os
import json
import glob
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

USER_MODEL_REQUEST = "gemini-2.5-flash-lite"
DIR = os.path.dirname(os.path.abspath(__file__))

class ExtractedDates(BaseModel):
    """Structure for extracted start and end dates"""
    start_date: str = Field(description="Start date of the notice (YYYY-MM-DD). 'N/A' if not found.")
    end_date: str = Field(description="End date of the notice (YYYY-MM-DD). 'N/A' if not found.")

def get_llm_chain():
    """Initialize LLM and extraction chain"""
    try:
        llm = ChatGoogleGenerativeAI(
            model=USER_MODEL_REQUEST,
            temperature=0,
            transport="rest"
        )
        structured_llm = llm.with_structured_output(ExtractedDates)
        
        extraction_template = """
        당신은 텍스트에서 날짜 정보를 추출하는 전문 AI입니다.
        주어진 [공지사항 텍스트]를 분석하여 '시작 날짜(start_date)'와 '종료 날짜(end_date)'를 추출하십시오.
        
        지침:
        1. 날짜는 반드시 'YYYY-MM-DD' 형식으로 정규화해야 합니다. (예: 2025. 11. 15. -> 2025-11-15)
        2. 날짜가 명확히 언급되지 않거나 "상시" 등 특정할 수 없는 경우 'N/A'로 응답합니다.
        3. **[중요] '마감일', '~까지'만 명시된 경우:** 해당 날짜는 '종료 날짜(end_date)'입니다. 이때 '시작 날짜'가 본문에 없다면, 문서 상단의 **'작성일', '수정일', '게시일'을 찾아 '시작 날짜'로 설정**하십시오.
        4. '신청 기간: A ~ B'인 경우, A는 '시작 날짜', B는 '종료 날짜'입니다.
        5. 텍스트에 기준 연도가 명시되지 않으면, [기준 연도]를 참고하여 YYYY를 결정하십시오.
        6. **[예외 처리]** 본문의 'dates' 필드나 상단의 '최종 수정일'은 원칙적으로 무시하지만, **접수 시작일이 명시되지 않은 경우에 한해 이를 '시작 날짜'로 사용**합니다.
        7. 마감일이 여러 개 나열된 경우(예: 서울대 (마감), 제주대 (마감)), 본문의 주요 마감일을 찾으십시오. 찾기 어려우면 'N/A'로 응답합니다.
        8. [우선순위] 대회, 공모전 등의 경우 '행사 기간'이 아닌 **'접수(신청) 기간'을 우선적으로 추출**하십시오.
        9. **[단일 날짜 처리]** 접수 마감일만 있고 시작일 추론이 불가능할 경우, Start와 End를 동일하게 설정하지 말고 Start는 'N/A' 혹은 (가능하다면) 작성일을 넣으십시오.
        
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
        return prompt | structured_llm
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return None

def process_new_posts():
    """
    Finds skku_*_posts.json files, identifies posts not in skku_*_augmented.json,
    and runs LLM extraction on them.
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting LLM processing for new posts...")
    
    any_updated = False
    
    extraction_chain = get_llm_chain()
    if not extraction_chain:
        return

    pattern = os.path.join(DIR, "skku_*_posts.json")
    files = glob.glob(pattern)

    for posts_file in files:
        augmented_file = posts_file.replace("_posts.json", "_augmented.json")
        
        try:
            with open(posts_file, 'r', encoding='utf-8') as f:
                posts_data = json.load(f)
                posts = posts_data.get("posts", [])
        except Exception as e:
            print(f"❌ Failed to load {posts_file}: {e}")
            continue

        augmented_data = []
        existing_urls = set()
        if os.path.exists(augmented_file):
            try:
                with open(augmented_file, 'r', encoding='utf-8') as f:
                    augmented_data = json.load(f)
                    for p in augmented_data:
                        if "post_link" in p:
                            existing_urls.add(p["post_link"])
            except Exception as e:
                print(f"⚠️ Failed to load {augmented_file}, starting fresh: {e}")

        new_posts = []
        for post in posts:
            if post.get("post_link") not in existing_urls:
                new_posts.append(post)

        if not new_posts:
            print(f"✅ No new posts to process for {os.path.basename(posts_file)}")
            continue

        print(f"🚀 Processing {len(new_posts)} new posts for {os.path.basename(posts_file)}...")

        for i, notice in enumerate(new_posts):
            title = notice.get('title', '')
            content = notice.get('content', '')
            dates_field = notice.get('date', '') # In crawl.py it's 'date', in 2_LLM... it was 'dates' (maybe list?)
            
            print(f"   [{i+1}/{len(new_posts)}] Processing: \"{title[:30]}...\"")
            
            try:
                input_data = {
                    "title": title, 
                    "content": content,
                    "dates": str(dates_field) 
                }
                
                extracted_data = extraction_chain.invoke(input_data)
                
                notice['start_date'] = extracted_data.start_date
                notice['end_date'] = extracted_data.end_date
                
                augmented_data.append(notice)
                
                print(f"     ▶ Start: {extracted_data.start_date} / End: {extracted_data.end_date}")
                
                time.sleep(2) 

            except Exception as e:
                print(f"     ❌ Error processing post: {e}")
                notice['start_date'] = 'N/A'
                notice['end_date'] = 'N/A'
                augmented_data.append(notice)

        try:
            with open(augmented_file, 'w', encoding='utf-8') as f:
                json.dump(augmented_data, f, indent=4, ensure_ascii=False)
            print(f"✅ Saved updated augmented data to {os.path.basename(augmented_file)}")
            any_updated = True
        except Exception as e:
            print(f"❌ Failed to save {augmented_file}: {e}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] LLM processing finished.")
    return any_updated

if __name__ == "__main__":
    process_new_posts()
