import json
import random
import os
from collections import defaultdict

def sample_stratified_chunks():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "final_rag_chunks.json")
    output_file = os.path.join(base_dir, "skku_sample_1500.json")
    TARGET_COUNT = 1500
    
    try:
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found.")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        total_items = len(data)
        print(f"Total chunks: {total_items}")
        
        chunks_by_title = defaultdict(list)
        for chunk in data:
            title = chunk.get("metadata", {}).get("title", "Unknown")
            chunks_by_title[title].append(chunk)
            
        unique_titles = list(chunks_by_title.keys())
        print(f"Unique titles: {len(unique_titles)}")
        
        selected_chunks = []
        remaining_pool = []
        
        for title in unique_titles:
            picked = random.choice(chunks_by_title[title])
            selected_chunks.append(picked)
            
            for c in chunks_by_title[title]:
                if c["id"] != picked["id"]:
                    remaining_pool.append(c)
                    
        current_count = len(selected_chunks)
        print(f"Selected 1 per title. Current count: {current_count}")
        
        if current_count < TARGET_COUNT:
            needed = TARGET_COUNT - current_count
            if len(remaining_pool) >= needed:
                print(f"Sampling {needed} more items from remaining pool...")
                additional_chunks = random.sample(remaining_pool, needed)
                selected_chunks.extend(additional_chunks)
            else:
                print(f"Warning: Not enough items to reach {TARGET_COUNT}. Taking all remaining.")
                selected_chunks.extend(remaining_pool)
        else:
            print(f"Note: Unique titles ({current_count}) already exceed or match target {TARGET_COUNT}.")
            
            if current_count > TARGET_COUNT:
                 print(f"Keeping all {current_count} unique title chunks (exceeds 1500).")

        random.shuffle(selected_chunks)
        
        print(f"Final selected count: {len(selected_chunks)}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(selected_chunks, f, indent=4, ensure_ascii=False)
            
        print(f"Saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sample_stratified_chunks()
