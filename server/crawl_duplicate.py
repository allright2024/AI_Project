import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timedelta
import os
import json
import argparse

# get command line arguments
parser = argparse.ArgumentParser(description="Crawl SKKU notice posts")
parser.add_argument("--days", type=int, default=90, help="post date threshold")
parser.add_argument("--force-update", action="store_true", help="force update regardless of update time")
args = parser.parse_args()

DIR = os.path.dirname(os.path.abspath(__file__))
DAYS = args.days # post date threshold
ARTICLE_NUM = DAYS * 5
LIST_URL = f"?mode=list&&articleLimit={ARTICLE_NUM}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

urls = [
    "https://sw.skku.edu/sw/notice.do",
    "https://cse.skku.edu/cse/notice.do",
]

filenames = [
    "skku_cci_posts.json",
    "skku_cse_posts.json",
]

def load_existing_data(filename):
    filepath = os.path.join(DIR, filename)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                existing_urls = {post["post_link"] for post in data.get('posts', [])}
                return data, existing_urls
            except json.JSONDecodeError:
                pass

    return {"last_updated": "", "posts": []}, set()

def get_post_list(notice_url):
    # send request to site
    resp = requests.get(urljoin(notice_url, LIST_URL), headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    ##################################
    # get list of posts (500 posts)
    ##################################
    posts = []
    # each <li> under <ul class="board-list-wrap">
    for li in soup.select("ul.board-list-wrap > li"):
        # check if date is less than 3 months ago
        info_li = li.select("dd.board-list-content-info ul > li")
        post_date = info_li[2].get_text(strip=True) if len(info_li) >= 3 else ""

        # convert to datetime object
        post_date = datetime.strptime(post_date, "%Y-%m-%d")
        
        # stop searching if it is more than 3 months
        if(datetime.now() - post_date > timedelta(days=DAYS)):
            break

        # reference a tag
        a_tag = li.select_one("a")
        if not a_tag:
            continue

        # clean up url
        href = a_tag.get("href")
        post_url = urljoin(notice_url, href)
        post_url = post_url.replace("&article.offset=0&articleLimit=500", "")

        # add post url to list
        posts.append((post_url, post_date))

    return posts

def get_post(post_url):
    # send request to site
    resp = requests.get(post_url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    ##################################
    # post information
    ##################################

    # reference title container
    title_div = soup.select_one(".board-view-title-wrap")

    if title_div:
        # get <span> as category
        category_tag = title_div.select_one("h4 span")
        category = category_tag.get_text(strip=True).strip("[]") if category_tag else ""

        # get the rest of the text inside <h4> (excluding <span>)
        h4_tag = title_div.select_one("h4")

        # remove <span> to avoid duplication
        if h4_tag and category_tag:
            category_tag.extract()
        title = h4_tag.get_text(strip=True) if h4_tag else ""
    else:
        category = ""
        title = ""

    # other information (department, total views and date posted)
    post_info = soup.select("ul.board-etc-wrap > li") # reference container
    department = post_info[0].get_text(strip=True) if len(post_info) >= 1 else "" # department name
    views_li = post_info[1] if len(post_info) >= 2 else "" # total views
    if views_li:
        span = views_li.select_one("span.board-mg-l10")
        views = int(span.get_text(strip=True)) if span else 0
    else:
        views = 0
    date = post_info[2].get_text(strip=True) if len(post_info) >= 3 else "" # date posted

    ##################################
    # post contents
    ##################################

    # extract file links
    files = []
    for li in soup.select("ul.board-view-file-wrap > li"):
        # select a tag
        file_link = li.select_one("a")
        if not file_link:
            continue
        # clean up url
        href = file_link.get("href")
        file_url = urljoin(post_url, href).replace(f"&article.offset=0&articleLimit={ARTICLE_NUM}", "")
        files.append(file_url)

    # reference content container
    content_div = soup.select_one(".board-view-content-wrap.board-view-txt")

    # extract image URLs
    images = []
    if content_div:
        for img in content_div.find_all("img", src=True):
            img_url = img["src"]
            # convert relative url to absolute
            img_url = urljoin(post_url, img_url)
            images.append(img_url)

    # extract links
    links = []
    if content_div:
        for a in content_div.find_all("a", href=True):
            link_url = a["href"]
            # convert relative url to absolute
            link_url = urljoin(post_url, link_url)
            links.append(link_url)

    # extract content as text
    content = content_div.get_text(separator="\n", strip=True) if content_div else ""

    ##################################
    # assemble information
    ##################################

    # assemble into json format
    post = {
        "category": category,
        "title": title,
        "department": department,
        "post_link": post_url.replace(f"&article.offset=0&articleLimit={ARTICLE_NUM}", ""), # link to original post
        "views": views,
        "date": date,
        "files": files,
        "images": images,
        "attached_links": links,
        "content": content
    }

    return post

# separate crawling for skku homepage
# average post per day is significantly bigger than cci & cse
def get_skku_main(skku_url, existing_urls, last_updated_time):
    posts = []
    offset = 0
    cutoff_date = datetime.now() - timedelta(days=DAYS)
    skipped_count = 0
    while True:
        # set url suffix
        offset_suffix = f"?mode=list&&articleLimit=10&article.offset={offset}"

        # request server
        resp = requests.get(urljoin(skku_url, offset_suffix), headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        li_list = soup.select("ul.board-list-wrap > li")
        if not li_list:
            break  # no more posts
        
        # break condition
        stop_scraping = False
        

        for li in li_list:
            info_li = li.select("dd.board-list-content-info ul > li")
            post_date_str = info_li[2].get_text(strip=True) if len(info_li) >= 3 else ""
            try:
                post_date = datetime.strptime(post_date_str, "%Y-%m-%d")
            except ValueError:
                continue

            if post_date < cutoff_date:
                stop_scraping = True
                break

            a_tag = li.select_one("a")
            if not a_tag:
                continue

            post_url = urljoin(skku_url, a_tag.get("href"))
            post_url = post_url.replace(f"&article.offset={offset}&articleLimit=10", "")

            should_crawl = True

            if post_url in existing_urls:
                if post_date > last_updated_time:
                    should_crawl = True
                else:
                    should_crawl = False
                    skipped_count += 1

            if post_url in existing_urls:
                if skipped_count > 10:
                    stop_scraping = True
                    break
                continue
            
            skipped_count = 0
            # get full post details
            post = get_skku_post(post_url)
            posts.append(post)

        if stop_scraping:
            break

        offset += 10
        print(f"Request for page {offset // 10} complete.")

    return posts

def get_skku_post(post_url):
        # send request to site
    resp = requests.get(post_url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    ##################################
    # post information
    ##################################

    # reference title container
    title_div = soup.find("th", attrs={"scope": "col"})

    if title_div:
        # get <span> as category
        category_tag = title_div.select_one("span.category")
        category = category_tag.get_text(strip=True).strip("[]") if category_tag else ""

        title_text = title_div.select_one("em.ellipsis")
        title = title_text.get_text(strip=True) if title_text else ""
    else:
        category = ""
        title = ""

    date = title_div.select_one("span.date").get_text(strip=True) # date posted

    ##################################
    # post contents
    ##################################

    # extract file links
    files = []
    for li in soup.select("ul.filedown_list > li"):
        # select a tag
        file_link = li.select_one("a")
        if not file_link:
            continue
        
        # clean up url
        href = file_link.get("href")
        file_url = urljoin(post_url, href)
        files.append(file_url)

    # reference content container
    content_div = soup.select_one("table.board_view")

    # extract image URLs
    images = []
    if content_div:
        for img in content_div.find_all("img", src=True):
            img_url = img["src"]
            # convert relative url to absolute
            img_url = urljoin("https://skku.edu", img_url)
            images.append(img_url)

    # extract content as text
    content = content_div.get_text(strip=True) if content_div else ""
    
    ##################################
    # assemble information
    ##################################

    # assemble into json format
    post = {
        "category": category,
        "title": title,
        "department": "SKKU",
        "post_link": post_url, # link to original post
        # "views": views,
        "date": date,
        "files": files,
        "images": images,
        # "attached_links": links,
        "content": content
    }

    return post

def check_outdated(filename):
    # set file path
    filepath = os.path.join(DIR, filename)  

    # check last updated time
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        # read file to check update time
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            last_updated = datetime.strptime(data["last_updated"], "%Y-%m-%d %H:%M:%S")

            # up to date or user request force update
            if datetime.now() - last_updated > timedelta(hours=6) or args.force_update:
                print("Updating...")
                return True
            else:
                # format time delta
                delta = datetime.now() - last_updated
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                minutes, _ = divmod(remainder, 60)

                if hours > 0:
                    print(f"Last updated {hours} hours {minutes} minutes ago, skipping update...")
                else:
                    print(f"Last updated {minutes} minutes ago, skipping update...")

                return False
    else: # if file does not exist
        print(f"{filepath} is empty or does not exist.")
        
        # create dummy data
        data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "posts": []
        }
        # create a new empty file with update time
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return True

if __name__ == "__main__":
    # crawl cci & cse
    for i in range(len(urls)):
        if not check_outdated(filenames[i]):
            continue
        filename = filenames[i]
        existing_data, existing_urls = load_existing_data(filename)

        last_updated_str = existing_data.get("last_updated", "2000-01-01 00:00:00")
        try:
            last_updated_time = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            last_updated_time = datetime(2000, 1, 1)

        # scrape all post links
        _urls = get_post_list(urls[i])
        print(f"Found {len(_urls)} posts")

        # format post information
        posts = []
        updated_urls = set()
        skipped_count = 0
        for post_url, post_date in _urls:
            should_crawl = True

            if post_url in existing_urls:
                if post_date > last_updated_time:
                    updated_urls.add(post_url)
                    should_crawl = True
                else:
                    should_crawl = False

            if not should_crawl:
                skipped_count += 1
                continue

            post = get_post(post_url)
            posts.append(post)

        # output = {
        #     "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        #     "posts": posts
        # }
        
        # save to json
        if posts:
            if updated_urls:
                existing_data['posts'] = [
                    p for p in existing_data['posts'] 
                    if p['post_link'] not in updated_urls
                ]

            existing_data['posts'] = posts + existing_data["posts"]
            existing_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(filenames[i], "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

        print(f"Updated {filenames[i]} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    main_filename = "skku_main_posts.json"
    # crawl skku homepage
    if check_outdated(main_filename) or args.force_update:
        # get all post information
        
        existing_main_data, existing_main_urls = load_existing_data(main_filename)

        last_updated_str = existing_main_data.get("last_updated", "2000-01-01 00:00:00")
        try:
            last_updated_time = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            last_updated_time = datetime(2000, 1, 1)

        skku_main_posts = get_skku_main("https://www.skku.edu/skku/campus/skk_comm/notice01.do", existing_main_urls, last_updated_time)

        if skku_main_posts:
            new_main_urls = {p['post_link'] for p in skku_main_posts}
            existing_main_data['posts'] = [
                p for p in existing_main_data['posts'] 
                if p['post_link'] not in new_main_urls
            ]
            existing_main_data["posts"] = skku_main_posts + existing_main_data["posts"]
            existing_main_data["last_updated"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # save to json
            with open(main_filename, "w", encoding="utf-8") as f:
                json.dump(existing_main_data, f, ensure_ascii=False, indent=4)

            print(f"Updated skku_main_posts.json at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f" No new posts for {main_filename}.")
