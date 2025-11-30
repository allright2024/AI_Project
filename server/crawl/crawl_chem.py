import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
import os
import json
import argparse

from crawl import parse_date, check_outdated

# get command line arguments
parser = argparse.ArgumentParser(description="Crawl SKKU notice posts")
parser.add_argument("--days", type=int, default=90, help="post date threshold")
parser.add_argument("--clean-slate", action="store_true", help="get everything regardless of current state")
parser.add_argument("--force-update", action="store_true", help="force update regardless of update time")
args = parser.parse_args()

DIR = os.path.dirname(os.path.abspath(__file__))
FILENAME = "skku_chem_posts.json"
DAYS = args.days # post date threshold
ARTICLE_NUM = DAYS * 1 # change scrape multiplier

BASE_URL = "https://chem.skku.edu/chem/News/notice.do"
LIST_URL = f"?mode=list&&articleLimit={ARTICLE_NUM}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

def get_post_list():
    # show ARTICLE_NUM posts
    resp = requests.get(urljoin(BASE_URL, LIST_URL), headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    posts = []
    for li in soup.select("ul.noticeList > li"):
        li_content = li.select("ul.noticeInfoList > li")
        for _li in li_content:
            if "DATE" in _li.get_text():
                post_date = parse_date(_li.get_text(strip=True).split(":")[1].strip())
                if datetime.now() - post_date > timedelta(days=DAYS):
                    break
                else:
                    post_url_suffix = li.select("p.noticeDesc > a")[0].get("href")
                    post_url_suffix = post_url_suffix.replace(f"&article.offset=0&articleLimit={ARTICLE_NUM}", "")
                    post_url = urljoin(BASE_URL, post_url_suffix)
                    posts.append(post_url)

    return posts

def get_post(post_url):
    # send request to post link
    resp = requests.get(post_url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    ################################################
    # post information
    ################################################

    # head_div = soup.select("div.noticeViewHead")
    noticeTit = soup.select("h3.noticeTit")[0].get_text(strip=True)

    # category
    try:
        category = re.match(r'^\[([^\]]+)\]', noticeTit).group(1)
    except:
        category = ""

    # title
    title = noticeTit.replace('[' + category + ']', "").strip()

    info_li = soup.select("ul.noticeInfoList > li")
    # department
    department = info_li[1].get_text(strip=True).split(":")[1].strip()

    # views
    views = int(info_li[2].get_text(strip=True).split(":")[1].strip())

    # date
    date = info_li[0].get_text(strip=True).split(":")[1].strip()

    ################################################
    # post contents
    ################################################

    # files
    files = []
    btnlist_div = soup.select("div.noticeViewBtnList > button")
    if btnlist_div:
        for btn in btnlist_div:
            onclick = btn.get("onclick", "")
            suffix = ""
            try:
                suffix = re.search(r"location\.href='([^']+)'", onclick).group(1)
                files.append(urljoin(BASE_URL, suffix))
            except:
                pass

    cont_div = soup.select_one("div.fr-view")

    # images
    images = []
    img_tags = cont_div.find_all("img")
    for img in img_tags:
        img_url_suffix = img.get("src")
        images.append(urljoin(BASE_URL, img_url_suffix))

    # content
    if cont_div:
        # get all text inside, strip extra spaces, and replace <br> with newlines
        content = cont_div.get_text(separator="\n", strip=True)

    ################################################
    # assemble information
    ################################################

    # assemble into json format
    post = {
        "category": category,
        "title": title,
        "department": department,
        "post_link": post_url, # link to original post
        "views": views,
        "date": date,
        "files": files,
        "images": images,
        # "attached_links": links,
        "content": content
    }
    
    return post

if __name__ == "__main__":
    # print(parse_date(datetime.now().strftime("%Y-%m-%d")))
    # print(check_outdated("skku_main_posts.json"))

    # first, check if json is outdated.
    # if outdated or force update option is true, then get the post list
    if check_outdated(FILENAME) or args.force_update or args.clean_slate:
        posts = get_post_list()
        print("Found", len(posts), "posts.")
        new_posts = 0

        # compare each post url to the latest post in json file
        with open(FILENAME, "r", encoding="utf-8") as f:
            data = json.load(f)
            try:
                latest_post = data["posts"][0]["post_link"]
            except:
                latest_post = ""

            updated_posts = []
            for post in posts:
                # if latest post in json != post crawled, add to list
                if post != latest_post or args.clean_slate:
                    updated_posts.append(get_post(post))
                    new_posts += 1
                # else break immediately
                else:
                    break
        
        with open(FILENAME, "w", encoding="utf-8") as _f:
            json.dump({
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                # append the list to existing json data and write to file
                "posts": updated_posts + data.get("posts", [])
            }, _f, ensure_ascii=False, indent=4)

        if args.clean_slate:
            print(f"Total {new_posts} posts added.")    
        else:
            print(f"Total {new_posts} posts updated.")
            
        print(f"Updated {FILENAME} at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    else: # if not outdated, terminate
        pass