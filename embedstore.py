import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

def _get_content_as_string(content_value):
    """
    '항내용', '호내용', '목내용' 등의 값이 리스트(중첩 리스트 포함) 또는 문자열일 수 있으므로,
    이를 모두 단일 문자열로 안전하게 변환합니다.
    """
    if isinstance(content_value, str):
        return content_value
    if isinstance(content_value, list):
        flat_list = []
        
        def flatten(item_list):
            """ 중첩 리스트를 재귀적으로 펼칩니다. """
            for item in item_list:
                if isinstance(item, list):
                    flatten(item)
                else:
                    flat_list.append(str(item))
        
        flatten(content_value)
        return "\n".join(flat_list)
    
    return ""

def _flatten_clause(clause_item):
    """ '항', '호', '목'으로 중첩된 법령 텍스트를 재귀적으로 펼칩니다. (타입 오류 수정됨) """
    text = ""
    
    if "항내용" in clause_item:
        hang_text = _get_content_as_string(clause_item["항내용"])
        text += clause_item.get("항번호", "") + " " + hang_text + "\n"
    
    if "호" in clause_item:
        ho_data = clause_item["호"]
        if isinstance(ho_data, list):
            for ho in ho_data:
                if "호내용" in ho:
                    ho_text = _get_content_as_string(ho["호내용"])
                    text += "  " + ho.get("호번호", "") + " " + ho_text + "\n"
                
                if "목" in ho:
                    mok_data = ho["목"]
                    if isinstance(mok_data, list):
                        for mok in mok_data:
                            if "목내용" in mok:
                                mok_text = _get_content_as_string(mok["목내용"])
                                text += "    " + mok.get("목번호", "") + " " + mok_text + "\n"
    return text

def load_law_documents(directory_path):
    """
    지정된 디렉토리의 모든 법령 JSON 파일을 로드하여 
    '조문' 단위로 Document 객체를 생성하고 메타데이터를 추가합니다.
    (데이터 구조가 list/dict인 경우 모두 처리하도록 수정됨)
    """
    all_docs = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                law_name = data['법령']['기본정보'].get('법령명_한글', '알 수 없음')
                
                
                if '조문' not in data['법령'] or '조문단위' not in data['법령']['조문']:
                    print(f"경고: '{filename}' 파일에 '조문' 또는 '조문단위' 키가 없어 건너뜁니다.")
                    continue

                for article in data['법령']['조문']['조문단위']:
                    if article.get('조문여부') == '조문':
                        clause_number = article.get('조문번호', '')
                        clause_title = article.get('조문제목', '')
                        
                        content_text = article.get('조문내용', '') + "\n"
                        
                        if "항" in article:
                            hang_data = article["항"]
                            
                            # Case 1: "항"이 리스트일 경우 (e.g., 근로기준법 제3조)
                            if isinstance(hang_data, list):
                                for hang_item in hang_data:
                                    content_text += _flatten_clause(hang_item)
                            # Case 2: "항"이 딕셔너리일 경우 (e.g., 근로기준법 제2조)
                            elif isinstance(hang_data, dict):
                                content_text += _flatten_clause(hang_data)
                        
                        metadata = {
                            "source": filename,
                            "law_name": law_name,
                            "clause_number": f"제{clause_number}조",
                            "clause_title": clause_title
                        }
                        
                        doc = Document(
                            page_content=content_text.strip(), # 앞뒤 공백 제거
                            metadata=metadata
                        )
                        all_docs.append(doc)
            
            except json.JSONDecodeError:
                print(f"오류: '{filename}' 파일이 유효한 JSON 형식이 아닙니다.")
            except KeyError as e:
                print(f"오류: '{filename}' 파일에서 필수 키({e})를 찾을 수 없습니다.")
            except Exception as e:
                print(f"오류: '{filename}' 파일 처리 중 예외 발생: {e}")
                    
    return all_docs

            

def build_and_save_vector_store(directory_path, index_path="faiss_law_index"):
    print("1. 법령 데이터 로드 및 전처리 중...")
    law_documents = load_law_documents(directory_path)
    if not law_documents:
        print("로드할 문서가 없습니다.")
        return
    
    print(f"총 {len(law_documents)}개의 법률 조항을 로드했습니다.")
    
    print("2. 텍스트 분할 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    all_splits = text_splitter.split_documents(law_documents)
    print(f"총 {len(all_splits)}개의 텍스트 청크로 분할되었습니다.")
    
    print("3. (무료) HuggingFace 임베딩 모델 로드 중...")
    print("   (최초 실행 시 모델 다운로드가 필요하며, 몇 분 정도 소요될 수 있습니다.)")
    model_name = "jhgan/ko-sbert-nli" # 한국어 성능이 검증된 경량 모델
    model_kwargs = {'device': 'cpu'}  # GPU 사용 시 'cuda'
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(f"'{model_name}' 모델 로드 완료.")
    
    print("4. FAISS 벡터 스토어 생성 중...")
    db = FAISS.from_documents(all_splits, embeddings)
    
    db.save_local(index_path)
    print(f"벡터 스토어가 '{index_path}'에 저장되었습니다.")
    
if __name__ == "__main__":
    data_directory = "./data"
    build_and_save_vector_store(data_directory)