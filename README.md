Crawls data from last 90 days.

JSON format:
```json
    {
        "category": "전체/학사/입학 ... (string)",
        "title": "소프트웨어학과 공청회 안내 (string)",
        "department": "소프트웨어학과 (string)",
        "views": 0,
        "date": "2025-11-12 (string)",
        "files": [
            "link to file (string)"
        ],
        "images": [
            "link to image (string)"
        ],
        "links": [
            "other links (string)"
        ],
        "content": "text contents"
    },
```

#### Usage

```bash
$ python crawl.py
```
By default, `crawl.py` crawls from https://sw.skku.edu/sw/notice.do and saves as file named `skku_cci_posts.json`

```bash
$ python crawl.py --url="https://cse.skku.edu/cse/notice.do" --filename="skku_cse_posts.json"
```
This command crawls from the SKKU CSE notice page and saves as `skku_cse_posts.json`.