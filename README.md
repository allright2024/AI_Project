# AI_Project

## API 키 관련
.env 파일에서
GOOGLE_API_KEY="Your API KEY"
저장하고 실행하시면 됩니다.


## 폴더는 총 4개로 이루어져 있습니다.
### 1. data_augmentation
설명 : sentence embedding에 활용되는 sbert 학습데이터를 gemini api로 만듭니다.
**rag_dataset_with_questions.json** : json 파일 같은 경우는 해당 synthetic_aug_api.py 파일을 실행함으로써 만들어낸 100개의 데이터 셋입니다.
**synthetic_aug_api.py** : 청크된 공지사항에 대하여 질문을 LLM에게 생성하도록 하여 user input - chunked notice 쌍을 만듭니다.

### 2. sbert
설명 : 각각 sbert를 학습 및 평가하는 코드입니다. 
평가 데이터는 현재 5개만 있으며, LLM을 통해 더 추가해볼 예정입니다.

### 3. server
**2_LLM_using_extract_day.py** : 서버에서 크롤링된 데이터의 제목과 내용을 기반으로 LLM API를 활용하여 유효기간을 뽑는 코드입니다. 문제는 기간이 여러 개가 있을 경우 제대로 수행하지 못하는 모습도 보입니다. 공지사항 내에서 행사 마감은 어디까지 신청서는 어디까지 등..(일단 보류)
**3_embedding_BM25.py** : user의  sbert를 활용한 word embedding과 BM25(TF-IDF 활용한 키워드 기반)를 활용하여 2_LLM_using_extract_day.py에서 전처리된 데이터들에 대하여 저장하는 코드입니다.

### 4. service
server폴더에서의 2_LLM_using_extract_day.py, 3_embedding_BM25.py와 중복인데, 해당 코드로 전처리된 파일을 4_retrieve.py의 입력으로 해야 하기 때문에 중복되더라도 포함시켰습니다.
**4_retrieve.py** : 사용자의 쿼리를 통하여 날짜 정보를 추출하고, retrieve를 수행하여 AI의 답변을 만들어냅니다. 이때 활용하는 데이터는 user query와의 유사도를 구해야하기 때문에 3_embedding_BM25.py에서 처리된 벡터 데이터들을 활용해야합니다. 
