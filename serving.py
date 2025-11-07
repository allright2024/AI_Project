import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
print("✅ Gemini LLM 모델 로드 완료")


# --- 2. 로컬 임베딩 모델 및 벡터 DB 로드 ---

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'} # GPU 사용 시 'cuda'
encode_kwargs = {'normalize_embeddings': True}

try:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(f"✅ HuggingFace 임베딩 모델('{model_name}') 로드 완료")
except Exception as e:
    print(f"❌ 임베딩 모델 로드 실패: {e}")
    print("   'pip install langchain-huggingface'를 실행했는지 확인하세요.")
    exit()

index_path = "faiss_law_index"

if not os.path.exists(index_path):
    print(f"❌ 오류: 벡터 스토어('faiss_law_index')를 찾을 수 없습니다.")
    print("   먼저 'embedstore.py'를 실행하여 인덱스를 생성해야 합니다.")
    exit()

try:
    db = FAISS.load_local(
        index_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS 벡터 스토어 로드 완료")
except Exception as e:
    print(f"❌ 벡터 스토어 로드 실패: {e}")
    exit()

# --- 3. RAG 체인 구성 ---

retriever = db.as_retriever(search_kwargs={"k": 10})

template = """
당신은 대한민국 법률을 전문으로 하는 AI 법률 검토 비서입니다.
제시된 [법률 조항]을 근거로 하여, 사용자가 제출한 [기업 문서]의 내용이 위법한지 검토하십시오.

검토 시, 다음 지침을 반드시 따르십시오:
1. [기업 문서]의 내용 중 [법률 조항]에 위배되는 부분을 정확히 식별합니다.
2. 위반 사항이 있다면, 어떤 [법률 조항]의 어느 부분(조, 항, 호)에 근거하여 위배되는지 명확하게 설명합니다.
3. 근거가 되는 [법률 조항]의 'source' (파일명), 'law_name' (법령명), 'clause_number' (조문번호)를 반드시 인용(Citation)해야 합니다.
4. 위반 사항이 없다면, "검토한 문서 내용 중 명백한 법률 위반 사항을 발견하지 못했습니다."라고 결론을 내립니다.
5. [법률 조항]에 없는 내용은 추측하여 답변하지 마십시오.

---
[법률 조항]
{context}
---
[기업 문서]
{input}
---

[검토 결과]
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(
        f"--- 출처: {doc.metadata['source']} ({doc.metadata['law_name']}), {doc.metadata['clause_number']} {doc.metadata['clause_title']} ---\n{doc.page_content}"
        for doc in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# --- 4. RAG 체인 실행 ---
if __name__ == "__main__":
    
    # [검토할 기업 문서를 여기에 입력하세요]
    user_document = """
    [신규 입사자 근로계약서]
    
    제5조 (근로시간)
    1. '을'의 근로시간은 1일 9시간, 1주 45시간을 기준으로 한다.
    2. '갑'은 업무상 필요한 경우 '을'과 합의하여 1주 12시간을 한도로 연장 근로를 명할 수 있다.

    제10조 (퇴직)
    '을'이 임신 또는 출산하는 경우, 이는 자동 퇴직 사유로 간주한다.
    """

    print("="*50)
    print(f"[사용자 문서 원본]\n{user_document}")
    print("="*50)
    print("\n--- AI 법률 검토 시작 ---")

    response = rag_chain.invoke(user_document)

    print(response)
    print("\n--- AI 법률 검토 종료 ---")