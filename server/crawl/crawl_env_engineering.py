'''
python crawl_env_engineering.py                     ; update jsons
python crawl_env_engineering.py --days={time delta} ; set post crawling range in days
python crawl_env_engineering.py --clean-slate       ; clean slates all jsons, crawl everything again (only when data is corrupt)
python crawl_env_engineering.py --force-update      ; forces update regardless of last update time
'''

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
import os
import json
import argparse
import time

# get command line arguments
parser = argparse.ArgumentParser(description="Crawl SKKU Environmental Engineering notice posts")
parser.add_argument("--days", type=int, default=90, help="post date threshold")
parser.add_argument("--clean-slate", action="store_true", help="get everything regardless of current state")
parser.add_argument("--force-update", action="store_true", help="force update regardless of update time")
args = parser.parse_args()

DIR = os.path.dirname(os.path.abspath(__file__))
DAYS = args.days # post date threshold
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

BASE_URL = "https://cal.skku.edu/index.php?hCode=BOARD&bo_idx=17"
ROOT_URL = "https://cal.skku.edu/"

def parse_date(date_str):
    # Format: 2025-09-08
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt
    except ValueError:
        return None

def get_post_list():
    posts = []
    page = 1
    
    while True:
        url = f"{BASE_URL}&pg={page}"
            
        print(f"Crawling list page: {url}")
        
        try:
            resp = requests.get(url, headers=HEADERS)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            break
            
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Select rows in the board list table
        rows = soup.select("table.board_table tr")
        
        # If only header row exists or no rows, stop
        if not rows or len(rows) <= 1:
            print("No more posts found.")
            break
            
        stop_crawling = False
        
        # Skip header row (first row)
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) < 6:
                continue
                
            # Category (2nd column)
            category = cols[1].get_text(strip=True)
            
            # Title and Link (3rd column)
            title_td = cols[2]
            link_tag = title_td.find("a")
            if not link_tag:
                continue
                
            title = link_tag.get_text(strip=True)
            href = link_tag.get("href")
            post_url = urljoin(ROOT_URL, href)
            
            # Date (6th column)
            date_str = cols[5].get_text(strip=True)
            post_date = parse_date(date_str)
            
            if post_date is None:
                continue
                
            # Check if post is older than threshold
            if datetime.now() - post_date > timedelta(days=DAYS):
                stop_crawling = True
                break
                
            posts.append({
                "url": post_url,
                "title": title,
                "date": post_date.strftime("%Y-%m-%d"),
                "category": category
            })
            
        if stop_crawling:
            break
            
        page += 1
        
    return posts

def get_post(post_info):
    post_url = post_info["url"]
    # print(f"Crawling post: {post_url}")
    
    try:
        resp = requests.get(post_url, headers=HEADERS)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {post_url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Content
    content_div = soup.select_one(".board_content")
    
    # Remove title and header info from content if present (usually they are inside board_view but structured)
    # Based on HTML inspection, board_view contains everything including title.
    # We might want to extract just the body. 
    # Looking at HTML, there is a table inside board_view with class __se_tbl_ext or similar for content body?
    # Or just get text from board_view but exclude the top part?
    # Let's try to get the whole board_view text for now, it might include title again but that's acceptable.
    # Actually, looking at the HTML structure again (from previous steps):
    # The title is in a th tag: <th colspan="3">[한화오션] ...</th>
    # The content seems to be in a td or div below.
    # Let's refine content extraction.
    
    content = ""
    if content_div:
        # Try to find the main content cell. It often is a td with high colspan or just a div.
        # In the inspected HTML, content seems to be after the title row.
        # Let's just take all text from board_view for now to be safe, or try to exclude the title.
        # Title is in `th`.
        for th in content_div.find_all("th"):
            th.decompose() # Remove title from content text
        
        # Also remove the "Attached file" row if it exists
        # It usually has class attachment_parent or similar
        for attach in content_div.select(".attachment_parent"):
            attach.decompose()
            
        content = content_div.get_text(separator="\n", strip=True)
    
    # Images
    images = []
    if content_div:
        for img in content_div.find_all("img", src=True):
            img_url = urljoin(ROOT_URL, img["src"])
            # Filter out common UI icons if possible
            if "common/files.jpg" not in img_url and "common/icon_" not in img_url and "common/x_btn.png" not in img_url:
                 images.append(img_url)
            
    # Files
    files = []
    # Based on inspection: <span class='attach_down'><a href="./NFUpload/nfupload_down.php?..." ...>
    for link in soup.select("span.attach_down a"):
        file_url = link.get("href")
        if file_url:
            full_file_url = urljoin(ROOT_URL, file_url)
            files.append(full_file_url)
            
    # Attached Links (if any in content)
    links = []
    if content_div:
        for a in content_div.find_all("a", href=True):
            link_url = a["href"]
            if link_url not in files and link_url != "#" and not link_url.startswith("javascript:"):
                 # Check if it is absolute or relative
                 full_link_url = urljoin(ROOT_URL, link_url)
                 links.append(full_link_url)

    return {
        "category": post_info["category"],
        "title": post_info["title"],
        "department": "건설환경공학부",
        "post_link": post_url,
        "views": 0, 
        "date": post_info["date"],
        "files": files,
        "images": images,
        "attached_links": links,
        "content": content
    }

def check_outdated(filename):
    filepath = os.path.join(DIR, filename)
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                last_updated = datetime.strptime(data["last_updated"], "%Y-%m-%d %H:%M:%S")
                
                if datetime.now() - last_updated > timedelta(hours=6) or args.force_update or args.clean_slate:
                    print("Updating...")
                    return True
                else:
                    delta = datetime.now() - last_updated
                    hours, remainder = divmod(int(delta.total_seconds()), 3600)
                    minutes, _ = divmod(remainder, 60)
                    print(f"Last updated {hours} hours {minutes} minutes ago, skipping update...")
                    return False
            except (json.JSONDecodeError, KeyError):
                print("File corrupted or invalid format.")
                return True
    else:
        print(f"{filepath} is empty or does not exist.")
        data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pinned_posts": [],
            "posts": []
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True

if __name__ == "__main__":
    filename = "skku_env_posts.json"
    
    if check_outdated(filename):
        print("Fetching post list...")
        post_list = get_post_list()
        
        if args.clean_slate:
            print(f"Found {len(post_list)} posts")
            
        # Load existing data to check for duplicates if not clean slate
        existing_posts = []
        existing_pinned = []
        if not args.clean_slate and os.path.exists(os.path.join(DIR, filename)):
             with open(os.path.join(DIR, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    existing_posts = data.get("posts", [])
                    existing_pinned = data.get("pinned_posts", [])
                except:
                    pass
        
        # Create set of existing URLs for fast lookup
        existing_urls = set()
        for p in existing_posts:
            if "post_link" in p:
                existing_urls.add(p["post_link"])
        
        new_posts_data = []
        new_count = 0
        
        for post_info in post_list:
            # If not clean slate and the post URL is already in existing_urls, skip it
            if not args.clean_slate and post_info["url"] in existing_urls:
                continue
            
            full_post = get_post(post_info)
            if full_post:
                new_posts_data.append(full_post)
                new_count += 1
                
        print(f"Total {new_count} posts updated.")
        
        # Merge posts
        if args.clean_slate:
            final_posts = new_posts_data
        else:
            final_posts = new_posts_data + existing_posts
            
        # Filter outdated posts from final list
        final_posts = [
            p for p in final_posts 
            if datetime.now() - datetime.strptime(p["date"], "%Y-%m-%d") <= timedelta(days=DAYS)
        ]
        
        output = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pinned_posts": existing_pinned,
            "posts": final_posts
        }
        
        with open(os.path.join(DIR, filename), "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
            
        print(f"Updated {filename} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
