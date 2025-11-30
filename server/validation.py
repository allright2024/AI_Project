import os
import json
import glob

file_path = "skku_main_augmented.json"
JSON_FILE_NAME = "skku_*_augmented.json" # 크롤링한 파일 이름 형식
files = glob.glob(JSON_FILE_NAME)

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        notices = json.load(f)

    start_valid_cnt = 0
    end_valid_cnt = 0

    for notice in notices:
        if notice["start_date"] != "N/A":
            start_valid_cnt += 1

        if notice["end_date"] != "N/A":
            end_valid_cnt += 1

    print("===\t", file, "\t===")
    print("Valid start dates:", start_valid_cnt)
    print("Valid end dates:", end_valid_cnt)
    print("Total posts:", len(notices))
    print("=" * 43)