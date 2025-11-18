from sentence_transformers import SentenceTransformer, util
import torch

# MY_MODEL_PATH = "./output/fine-tuned-sbert"
MY_MODEL_PATH = "jhgan/ko-sbert-nli"

def run_test_cases():
    print(f">>> 모델 로드 중: {MY_MODEL_PATH}")
    try:
        model = SentenceTransformer(MY_MODEL_PATH)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    # 2. 테스트 케이스 정의 (Query, [정답, 오답1, 오답2])
    test_cases = [
        # Case 1: 장학금 (금액 문의) - skku_6.json 기반
        {
            "category": "장학금",
            "query": "인송문화재단 장학금 얼마 지원해줘?",
            "docs": [
                "지원 금액 : 2,000,000원 / 1명당 / 1년 (2022-2학기에 전액 지급)",  # (정답)
                "국가장학금은 소득 분위에 따라 차등 지급됩니다.",                     # (오답: 다른 장학금)
                "기숙사 관리비는 학기당 150만원 내외입니다."                        # (오답: 돈 얘기지만 기숙사)
            ]
        },
        # Case 2: 행사/워크숍 (일정 문의) - skku_5.json 기반
        {
            "category": "행사/교육",
            "query": "AI 영상 만들기 워크숍 언제 해?",
            "docs": [
                "운영일시: 11/20(목) 18:30-21:30, 11/21(금) 18:30-21:30",      # (정답)
                "노션(Notion) 워크숍은 11/18(화) 오후 4시부터 진행됩니다.",         # (오답: Hard Negative - 비슷한 워크숍)
                "중간고사 기간은 10월 20일부터 25일까지입니다."                    # (오답: 전혀 다른 일정)
            ]
        },
        # Case 3: 대학원 모집 (마감일 문의) - skku_3.json 기반
        {
            "category": "입시/모집",
            "query": "국정전문대학원 원서 접수 마감 언제야?",
            "docs": [
                "원서접수: 10/6(월) 10:00 ~ 10/17(금) 17:00까지 접수 가능합니다.", # (정답)
                "유림지도자과정 접수기간: 2014년 5월 26일 ~ 6월 20일",             # (오답: 다른 모집 공고)
                "도서관 대출 도서 반납 기한은 2주입니다."                          # (오답)
            ]
        },
        # Case 4: 기숙사 (공모전 주제) - skku_8.json 기반
        {
            "category": "기숙사/생활",
            "query": "기숙사 공모전 주제가 뭐야?",
            "docs": [
                "주제 (택1): ① 이건 첫 번째 레슨♬ Thank U, dormitory (노래 개사)", # (정답)
                "기숙사 입사 신청 자격은 직전 학기 성적 3.0 이상입니다.",           # (오답: 기숙사 관련이지만 규칙)
                "학교 식당 메뉴는 매주 금요일 업데이트됩니다."                      # (오답)
            ]
        },
        # Case 5: 아르바이트 (근로 장소) - skku_4.json 기반
        {
            "category": "채용/알바",
            "query": "수업 기자재 관리 알바 어디서 일해?",
            "docs": [
                "근 무 지 : 수원시 장안구 천천동 300번지 성균관대학교 자연과학캠퍼스", # (정답)
                "학생인재개발팀 근로 장소는 인문사회캠퍼스 경영관입니다.",            # (오답: 다른 알바 장소)
                "셔틀버스 승차 위치는 정문 앞입니다."                               # (오답)
            ]
        }
    ]

    # 3. 테스트 실행 및 출력
    print("\n" + "="*50)
    print("   SBERT 검색 성능 테스트 결과")
    print("="*50)

    for idx, case in enumerate(test_cases):
        print(f"\n[Case {idx+1}: {case['category']}]")
        print(f"Q: {case['query']}")
        
        query_embedding = model.encode(case['query'])
        doc_embeddings = model.encode(case['docs'])
        
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]

        results = []
        for i, score in enumerate(scores):
            results.append((score.item(), case['docs'][i]))
        
        results.sort(key=lambda x: x[0], reverse=True)

        for rank, (score, doc) in enumerate(results):
            is_correct = "✅" if doc == case['docs'][0] else "❌"
            print(f" {rank+1}등 ({score:.4f}) {is_correct} {doc}")

if __name__ == "__main__":
    run_test_cases()
