import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

url = "https://cheme.skku.edu/category/notice/"

try:
    resp = requests.get(url, headers=HEADERS)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        print("Success!")
        print(resp.text[:500]) # Print first 500 chars
        
        # Check for article tags
        if "<article" in resp.text:
            print("Found <article> tags")
        else:
            print("No <article> tags found")
            
        # Check for specific class
        if "w-blog-entry" in resp.text:
            print("Found 'w-blog-entry' class")
            
except Exception as e:
    print(f"Error: {e}")
