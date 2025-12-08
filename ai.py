import openai
import requests


OPENAI_API_KEY = ""  # 🔑 API 키 입력
client = openai.OpenAI(api_key=OPENAI_API_KEY)

SEARCH_SERVER_URL = "http://127.0.0.1:8001/search"
TOP_K = 5 



def retrieve_notices_from_server(user_query: str, top_k: int = TOP_K):
    payload = {"query": user_query, "top_k": top_k}
    try:
        resp = requests.post(SEARCH_SERVER_URL, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] 서버 통신 오류: {e}")
        return []

    notices = []
    for item in data:
        meta = item.get("metadata", {}) or {}
        # URL 가져오기
        final_url = meta.get("url") or meta.get("link") or meta.get("href") or "링크 없음"

        notices.append(
            {
                "title": meta.get("title", "(제목 없음)"),
                "url": final_url,
                "dates": [meta.get("start_date", ""), meta.get("end_date", "")],
                "content": item.get("text", ""),
            }
        )
    return notices


def notices_to_text(filtered_notices):
    blocks = []
    for idx, n in enumerate(filtered_notices, 1):
        dates = n.get("dates", [])
        date_str = " / ".join([d for d in dates if d]) if isinstance(dates, list) else str(dates)
        
        block = [
            f"===== [{idx}] 공지사항 =====",
            f"제목: {n['title']}",
            f"URL: {n['url']}",     
            f"날짜: {date_str}",
            f"내용:\n{n['content'][:800]}" 
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def get_notice_prompt_chatty(filtered_txt: str, user_query: str, top_n: int = 5):
    prompt = f"""
당신은 성균관대학교 재학생을 도와주는 친절하고 유용한 AI 챗봇입니다.
아래에 제공된 {top_n}개의 공지사항 데이터를 바탕으로 사용자에게 답변해 주세요.

[답변 구조]
1. **인사말**: 사용자의 질문 주제(예: 장학금, 수강신청, 졸업 등)를 언급하며 친절하게 시작하세요.
   (예: "안녕하세요! 졸업 요건과 관련된 공지들을 정리해드릴게요. 다음 공지들이 도움이 될 것 같습니다:")

2. **공지 목록**: 검색된 {top_n}개의 공지를 모두 나열하세요. 각 공지는 아래 형식을 지켜주세요.
   
   공지제목
   요약 : (내용을 1~2문장으로 핵심만 요약)
   URL : (제공된 URL 그대로 출력)
   (공지 사이에는 빈 줄 추가)

3. **맺음말**: 따뜻한 마무리 멘트를 해주세요.
   (예: "이 공지들이 도움이 되길 바랍니다! 궁금한 점이 있으면 언제든지 물어보세요.")

[규칙]
- 제공된 {top_n}개의 공지는 **하나도 빠짐없이 순서대로 모두** 출력해야 합니다.
- URL은 절대 변경하거나 생략하지 말고 데이터에 있는 그대로 적으세요. ('링크 없음'이면 그대로 표기)
- 말투는 "해요체"를 사용하여 정중하고 부드럽게 하세요.
- 별표(**)나 번호 매기기(1., 2.) 등 마크다운 리스트 문법은 쓰지 말고, 공지 사이를 빈 줄로만 구분하세요.

--- 사용자 질문 ---
{user_query}

----- 공지사항 데이터 ({top_n}개) -----
{filtered_txt}
----- 데이터 끝 -----
"""
    return prompt

if __name__ == "__main__":
    user_query = input("질문을 입력하세요: ").strip()
    if not user_query: raise SystemExit()

    print("🔍 공지사항을 찾아보고 있습니다...")
    notices = retrieve_notices_from_server(user_query)

    if not notices:
        print("관련된 공지를 찾지 못했어요.")
        raise SystemExit()

    prompt = get_notice_prompt_chatty(notices_to_text(notices), user_query, len(notices))
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 성균관대학교의 친절한 공지사항 안내 봇입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    print("\n" + "="*50 + "\n")
    print(response.choices[0].message.content.strip())
    print("\n" + "="*50)