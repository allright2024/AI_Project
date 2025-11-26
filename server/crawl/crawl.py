import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
import os
import json
import argparse

# get command line arguments
parser = argparse.ArgumentParser(description="Crawl SKKU notice posts")
parser.add_argument("--days", type=int, default=90, help="post date threshold")
parser.add_argument("--clean-slate", action="store_true", help="get everything regardless of current state")
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
    "https://sw.skku.edu/sw/notice.do", # 소프트웨어융합대학
    "https://cse.skku.edu/cse/notice.do", # 소프트웨어학과
    "https://sco.skku.edu/sco/community/notice.do", # 글로벌융합학부
    "https://intelligentsw.skku.edu/intelligentsw/notice.do", # 지능형소프트웨어학과
    "https://cscience.skku.edu/cscience/community/under_notice.do", # 자연과학대학
    "https://math.skku.edu/math/community/under_notice.do", # 수학과
    "https://physics.skku.ac.kr/physics/notice/notice.do", # 물리학과
    # "https://chem.skku.edu/chem/News/notice.do", # 화학과
    "https://ice.skku.edu/ice/notice.do", # 정보통신대학
    "https://eee.skku.edu/eee/notice.do", # 전자전기공학부
    # "https://semi.skku.edu/semi/community/notice.do", # 반도체시스템공학과 (로그인 필요함)
    "https://skb.skku.edu/mcce/notice.do", # 소재부품융합공학과
    # "https://sce.skku.edu/sce/notice.do", # 반도체융합공학과 (로그인 필요함)
    "https://enc.skku.edu/enc/notice.do", # 공과대학
    # "https://cheme.skku.edu/notice/", # 화학공학부
    "https://amse.skku.edu/AMSE/notice.do", # 신소재공학부
    "https://mech.skku.edu/me/notice.do", # 기계공학부
    # "https://cal.skku.edu/index.php?hCode=BOARD&bo_idx=17", # 건설환경공학부
    "https://sme.skku.edu/iesys/notice.do", # 시스템경영공학과/산업공학과
    "https://arch.skku.edu/arch/NEWS/notice.do", # 건축학과
    # "https://nano.skku.edu/bbs/board.php?tbl=bbs42", # 나노공학과
    "https://qie.skku.edu/qie/notice.do", # 양자정보공학과
    "https://biotech.skku.edu/biotech/community/under_notice.do", # 생명공학대학
    "https://skb.skku.edu/foodlife/community/notice_grad.do", # 식품생명공학과
    "https://skb.skku.edu/biomecha/community/notice.do", # 바이오메카트로닉스학과
    "https://skb.skku.edu/gene/community/under_notice.do", # 융합생명공학과
]

def get_post_list(notice_url):
    # send request to site
    resp = requests.get(urljoin(notice_url, LIST_URL), headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    ################################################
    # get list of posts (total `days` x 5 posts)
    ################################################
    posts = []
    pinned_posts = []
    # each <li> under <ul class="board-list-wrap">
    for li in soup.select("ul.board-list-wrap > li"):
        # check if date is less than 3 months ago
        info_li = li.select("dd.board-list-content-info ul > li")
        post_date_str = info_li[2].get_text(strip=True) if len(info_li) >= 3 else ""
        
        # convert to datetime object
        # post_date = datetime.strptime(post_date, "%Y-%m-%d")
        post_date = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                post_date = datetime.strptime(post_date_str, fmt)
                break
            except ValueError:
                continue
        
        if post_date is None:
            continue
        
        # check if post is a pinned post
        dt_tag = li.select_one("dt.board-list-content-title")
        is_pinned = False
        if dt_tag:
            is_pinned = "board-list-content-top" in dt_tag.get("class", [])

        # reference a tag
        a_tag = li.select_one("a")
        if not a_tag:
            continue

        # clean up url
        href = a_tag.get("href")
        post_url = urljoin(notice_url, href)
        post_url = post_url.replace(f"&article.offset=0&articleLimit={ARTICLE_NUM}", "")
        
        if is_pinned:            
            # move to next post if pinned post is posted more than 3 months ago
            if(datetime.now() - post_date > timedelta(days=DAYS)):
                continue
            
            pinned_posts.append(post_url)
        else:    
            # stop searching if it is posted more than 3 months ago
            if(datetime.now() - post_date > timedelta(days=DAYS)):
                break

            # add post url to list
            posts.append(post_url)

    return pinned_posts, posts

def get_post(post_url, is_pinned=False):
    # send request to site
    resp = requests.get(post_url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    ################################################
    # post information
    ################################################

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

    ################################################
    # post contents
    ################################################

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

    ################################################
    # assemble information
    ################################################

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
def get_skku_main(skku_url, latest_url=None):
    posts = []
    offset = 0
    cutoff_date = datetime.now() - timedelta(days=DAYS)

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

            if post_url == latest_url:
                return posts

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

    ################################################
    # post information
    ################################################

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

    ################################################
    # post contents
    ################################################

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
    
    ################################################
    # assemble information
    ################################################

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
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            last_updated = datetime.strptime(data["last_updated"], "%Y-%m-%d %H:%M:%S")

            # up to date or user request force update
            if datetime.now() - last_updated > timedelta(hours=6) or args.force_update or args.clean_slate:
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
            "pinned_posts": [],
            "posts": []
        }
        # create a new empty file with update time
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return True

if __name__ == "__main__":
    # crawl cci & cse
    for i in range(len(urls)):
        # format file name
        filename = "skku_" + urlparse(urls[i]).hostname.split(".")[0] + "_posts.json"

        # check if outdated
        if not check_outdated(filename):
            continue

        # scrape all post links
        _urls_pinned, _urls = get_post_list(urls[i])
        print(f"Found {len(_urls)} posts")

        # check files to update only
        if not args.clean_slate:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

                # get latest post links
                if data.get("pinned_posts"): # list is not empty
                    latest_pinned_url = data["pinned_posts"][0]["post_link"]
                else:
                    latest_pinned_url = None # empty list

                if data.get("posts"): # list is not empty
                    latest_post_url = data["posts"][0]["post_link"]
                else: # empty list
                    latest_post_url = None

                # remove outdated posts
                while True:
                    if len(data["pinned_posts"]) == 0:
                        break

                    _date = datetime.strptime(data["pinned_posts"][len(data["pinned_posts"]) - 1]["date"], "%Y-%m-%d")
                    if datetime.now() - _date > timedelta(days=DAYS):
                        data["pinned_posts"].pop()
                    else: 
                        break

                while True:
                    if len(data["posts"]) == 0:
                        break

                    _date = datetime.strptime(data["posts"][len(data["posts"]) - 1]["date"], "%Y-%m-%d")
                    if datetime.now() - _date > timedelta(days=DAYS):
                        data["posts"].pop()
                    else: 
                        break

        # format post information
        pinned_posts = []
        posts = []
        new_posts = 0
        # pinned posts
        for url in _urls_pinned:
            # if not a clean slate, add new posts only
            if not args.clean_slate:
                if url == latest_pinned_url:
                    break
                new_posts += 1
            post = get_post(url)
            pinned_posts.append(post)
        # unpinned posts
        for url in _urls:
            # if not a clean slate, add new posts only
            if not args.clean_slate:
                if url == latest_post_url:
                    break
                new_posts += 1

            post = get_post(url)
            posts.append(post)
        
        if not args.clean_slate:
            print(f"Total {new_posts} posts updated.")

        output = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pinned_posts" : pinned_posts if args.clean_slate else pinned_posts + data["pinned_posts"],
            "posts": posts if args.clean_slate else posts + data["posts"]
        }
        
        # save to json
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        print(f"Updated {filename} at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

    # crawl skku homepage
    if check_outdated("skku_main_posts.json") or args.force_update or args.clean_slate:
        if not args.clean_slate:
            with open("skku_main_posts.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                latest = data["posts"][0]["post_link"] if len(data["posts"]) > 0 else []
                skku_main_posts = get_skku_main("https://www.skku.edu/skku/campus/skk_comm/notice01.do",latest_url=latest)
        else:
            # get all post information
            skku_main_posts = get_skku_main("https://www.skku.edu/skku/campus/skk_comm/notice01.do")

        # save to json
        with open("skku_main_posts.json", "w", encoding="utf-8") as f:
            json.dump({
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "posts": skku_main_posts
            }, f, ensure_ascii=False, indent=4)

        print(f"Updated skku_main_posts.json at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")