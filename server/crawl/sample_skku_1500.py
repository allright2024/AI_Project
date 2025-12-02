import json
import random
import os

def sample_items():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "skku_all.json")
    output_file = os.path.join(base_dir, "skku_1500.json")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        total_items = len(data)
        print(f"Total items in {os.path.basename(input_file)}: {total_items}")
        
        if total_items <= 1500:
            print(f"Count is less than or equal to 1500. Taking all {total_items} items.")
            sampled_data = data
        else:
            print(f"Sampling 1500 items from {total_items}...")
            sampled_data = random.sample(data, 1500)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, indent=4, ensure_ascii=False)
            
        print(f"Saved {len(sampled_data)} items to {os.path.basename(output_file)}")
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sample_items()
