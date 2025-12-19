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

parser = argparse.ArgumentParser(description="Crawl SKKU Environmental Engineering notice posts")
parser.add_argument("--days", type=int, default=90, help="post date threshold")
parser.add_argument("--clean-slate", action="store_true", help="get everything regardless of current state")
parser.add_argument("--force-update", action="store_true", help="force update regardless of update time")
args = parser.parse_args()

DIR = os.path.dirname(os.path.abspath(__file__))
DAYS = args.days
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
        
        rows = soup.select("table.board_table tr")
        
        if not rows or len(rows) <= 1:
            print("No more posts found.")
            break
            
        stop_crawling = False
        
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) < 6:
                continue
                
            category = cols[1].get_text(strip=True)
            
            title_td = cols[2]
            link_tag = title_td.find("a")
            if not link_tag:
                continue
                
            title = link_tag.get_text(strip=True)
            href = link_tag.get("href")
            post_url = urljoin(ROOT_URL, href)
            
            date_str = cols[5].get_text(strip=True)
            post_date = parse_date(date_str)
            
            if post_date is None:
                continue
                
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
    
    try:
        resp = requests.get(post_url, headers=HEADERS)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {post_url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    
    content_div = soup.select_one(".board_content")
    
    content = ""
    if content_div:
        for th in content_div.find_all("th"):
            th.decompose()
        for attach in content_div.select(".attachment_parent"):
            attach.decompose()
            
        content = content_div.get_text(separator="\n", strip=True)
    
    images = []
    if content_div:
        for img in content_div.find_all("img", src=True):
            img_url = urljoin(ROOT_URL, img["src"])
            if "common/files.jpg" not in img_url and "common/icon_" not in img_url and "common/x_btn.png" not in img_url:
                 images.append(img_url)
            
    files = []
    for link in soup.select("span.attach_down a"):
        file_url = link.get("href")
        if file_url:
            full_file_url = urljoin(ROOT_URL, file_url)
            files.append(full_file_url)
            
    links = []
    if content_div:
        for a in content_div.find_all("a", href=True):
            link_url = a["href"]
            if link_url not in files and link_url != "#" and not link_url.startswith("javascript:"):
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
        
        existing_urls = set()
        for p in existing_posts:
            if "post_link" in p:
                existing_urls.add(p["post_link"])
        
        new_posts_data = []
        new_count = 0
        
        for post_info in post_list:
            if not args.clean_slate and post_info["url"] in existing_urls:
                continue
            
            full_post = get_post(post_info)
            if full_post:
                new_posts_data.append(full_post)
                new_count += 1
                
        print(f"Total {new_count} posts updated.")
        
        if args.clean_slate:
            final_posts = new_posts_data
        else:
            final_posts = new_posts_data + existing_posts
            
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
