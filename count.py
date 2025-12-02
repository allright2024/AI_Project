import json

with open("/Users/yongkim/Desktop/playground/AI_Project/server/crawl/final_rag_chunks.json", "r", encoding="utf-8") as f:
    asd = json.load(f)
print(len(asd))

