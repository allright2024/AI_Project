import os
import json
import glob

def merge_json_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(base_dir, "skku_all.json")
    
    posts_files = glob.glob(os.path.join(base_dir, "skku_*_posts.json"))
    
    all_posts_map = {}
    
    print(f"Found {len(posts_files)} post files.")
    
    for posts_file in posts_files:
        dept_prefix = posts_file.replace("_posts.json", "")
        augmented_file = f"{dept_prefix}_augmented.json"
        
        filename = os.path.basename(posts_file)
        print(f"Processing {filename}...")
        
        try:
            with open(posts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            pinned_posts = data.get("pinned_posts", [])
            for post in pinned_posts:
                link = post.get("post_link")
                if link:
                    if link not in all_posts_map:
                        if "start_date" not in post:
                            post["start_date"] = "N/A"
                        if "end_date" not in post:
                            post["end_date"] = "N/A"
                        all_posts_map[link] = post
            
            if os.path.exists(augmented_file):
                print(f"  Found augmented file: {os.path.basename(augmented_file)}")
                try:
                    with open(augmented_file, 'r', encoding='utf-8') as f:
                        augmented_data = json.load(f)
                        for post in augmented_data:
                            link = post.get("post_link")
                            if link:
                                all_posts_map[link] = post
                except Exception as e:
                    print(f"  Error reading augmented file: {e}")
                    posts = data.get("posts", [])
                    for post in posts:
                        link = post.get("post_link")
                        if link:
                             if link not in all_posts_map:
                                if "start_date" not in post:
                                    post["start_date"] = "N/A"
                                if "end_date" not in post:
                                    post["end_date"] = "N/A"
                                all_posts_map[link] = post
            else:
                posts = data.get("posts", [])
                for post in posts:
                    link = post.get("post_link")
                    if link:
                        if link not in all_posts_map:
                            if "start_date" not in post:
                                post["start_date"] = "N/A"
                            if "end_date" not in post:
                                post["end_date"] = "N/A"
                            all_posts_map[link] = post
                            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    combined_list = list(all_posts_map.values())
    
    try:
        combined_list.sort(key=lambda x: x.get("date", "0000-00-00"), reverse=True)
    except:
        pass

    print(f"Total unique posts: {len(combined_list)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_list, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    merge_json_files()
