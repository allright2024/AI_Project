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
        You are an expert AI that extracts date information from text.
        Analyze the given [Notice Text] and extract the 'Start Date (start_date)' and 'End Date (end_date)'.

        Guidelines:
        1. Dates must be normalized to 'YYYY-MM-DD' format. (e.g., 2025. 11. 15. -> 2025-11-15)
        2. If the date is not clearly mentioned or is "always", respond with 'N/A'.
        3. 'Deadline', 'until', etc. are considered 'End Date (end_date)'.
        4. If 'Application Period: A ~ B', A is 'Start Date', B is 'End Date'.
        5. If the reference year is not specified in the text, refer to [Reference Year] to determine YYYY.
        6. If there is a 'Last Modified Date' in the 'dates' field of the body, ignore it as it is the modification date of the document, not the deadline.
        7. If multiple deadlines are listed, find the main deadline of the text. If difficult to find, respond with 'N/A'.

        ---
        [Reference Year]
        2025

        [Notice Text]
        Title: {title}
        Content: {content}
        (Reference dates field: {dates})
        ---

        [Extraction Result (JSON)]
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
