import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timedelta
import json
import argparse

# get command line arguments
parser = argparse.ArgumentParser(description="Crawl SKKU notice posts")
parser.add_argument("--filename", type=str, default="skku_cci_posts.json", help="JSON output file name")
parser.add_argument("--url", type=str, default="https://sw.skku.edu/sw/notice.do", help="Base URL to crawl")
args = parser.parse_args()

# arguments sending requests to server
BASE_URL = args.url
ARTICLE_NUM = 500
LIST_URL = f"?mode=list&&articleLimit={ARTICLE_NUM}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

def get_post_list():
    # send request to site
    resp = requests.get(urljoin(BASE_URL, LIST_URL), headers=HEADERS)
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
        if(datetime.now() - post_date > timedelta(days=90)):
            break

        # reference a tag
        a_tag = li.select_one("a")
        if not a_tag:
            continue

        # clean up url
        href = a_tag.get("href")
        post_url = urljoin(BASE_URL, href)
        post_url = post_url.replace("&article.offset=0&articleLimit=500", "")

        # add post url to list
        posts.append(post_url)

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
        category = category_tag.get_text(strip=True) if category_tag else ""

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
        file_url = urljoin(BASE_URL, href)
        file_url = file_url.replace("&article.offset=0&articleLimit=500", "")
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
        "views": views,
        "date": date,
        "files": files,
        "images": images,
        "links": links,
        "content": content
    }

    return post

def main():
    # get list of posts
    post_urls = get_post_list()
    print(f"Found {len(post_urls)} posts\n")

    # collect information from each post
    posts = []
    for post_url in post_urls:
        post = get_post(post_url)
        posts.append(post)
         
    # save to json
    with open(args.filename, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
