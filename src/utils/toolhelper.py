import numpy as np
import pandas as pd
import re
import faiss
from langchain_openai import OpenAIEmbeddings
from underthesea import word_tokenize
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import SQLiteVec
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from typing import Optional, Union, List

import os
from dotenv import load_dotenv
load_dotenv()

import unicodedata
# VietnameseToneNormalization.md
# https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md

TONE_NORM_VI = {
    'òa': 'oà', 'Òa': 'Oà', 'ÒA': 'OÀ',\
    'óa': 'oá', 'Óa': 'Oá', 'ÓA': 'OÁ',\
    'ỏa': 'oả', 'Ỏa': 'Oả', 'ỎA': 'OẢ',\
    'õa': 'oã', 'Õa': 'Oã', 'ÕA': 'OÃ',\
    'ọa': 'oạ', 'Ọa': 'Oạ', 'ỌA': 'OẠ',\
    'òe': 'oè', 'Òe': 'Oè', 'ÒE': 'OÈ',\
    'óe': 'oé', 'Óe': 'Oé', 'ÓE': 'OÉ',\
    'ỏe': 'oẻ', 'Ỏe': 'Oẻ', 'ỎE': 'OẺ',\
    'õe': 'oẽ', 'Õe': 'Oẽ', 'ÕE': 'OẼ',\
    'ọe': 'oẹ', 'Ọe': 'Oẹ', 'ỌE': 'OẸ',\
    'ùy': 'uỳ', 'Ùy': 'Uỳ', 'ÙY': 'UỲ',\
    'úy': 'uý', 'Úy': 'Uý', 'ÚY': 'UÝ',\
    'ủy': 'uỷ', 'Ủy': 'Uỷ', 'ỦY': 'UỶ',\
    'ũy': 'uỹ', 'Ũy': 'Uỹ', 'ŨY': 'UỸ',\
    'ụy': 'uỵ', 'Ụy': 'Uỵ', 'ỤY': 'UỴ'
    }


# -------------------------
# Load CSV
# -------------------------
def add_column_names_to_values(df : pd.DataFrame) -> pd.DataFrame:
    # Đọc file csv
    
    # Áp dụng cho từng hàng
    def row_with_keys(row):
        return {col: f"{col}: {row[col]}" for col in df.columns}
    
    # Tạo DataFrame mới với giá trị đã thêm tên cột
    new_df = df.apply(row_with_keys, axis=1, result_type="expand")
    return new_df

def load_excel(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    # Normalize current_price
    df["current_price"] = (
        df["current_price"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", np.nan)
    )
    df["current_price"] = df["current_price"].astype(float).dropna().astype("Int64").astype(str) + " vnd"
    
    # df = add_column_names_to_values(df)
    
    return df


def convert_table_to_rows(df: pd.DataFrame, allow_combined_info: bool) -> list:
    result_list = []

    if not allow_combined_info:
        # xóa cột sau cùng trong dữ liệu. cột combined_info
        df = df.iloc[:, :-1]
    
    for _, row in df.iterrows():
        row_string = f", ".join(str(v) for v in row).lower()
        
        row_string = (str(
            row.to_dict())
                      .replace("_id", "món thứ:")
                      .replace("_name", "tên")
                      .replace("type_of_food", "loại món")
                      .replace("main_ingredient", "nguyên liệu chính")
                      .replace("how_to_prepare", "cách chế biến")
                      .replace("taste", "vị giác")
                      .replace("outstanding_fragrance", "mùi vị đặc trưng")
                      .replace("current_price", "giá")
                      .replace("number_of_people_eating", "số người")
                      .replace("combined_info", "tổng hợp thông tin")
                      )
        row_string = bm25_preprocessing_func(row_string)
        row_string = ", ".join(row_string)
        result_list.append(re.sub(r'<[^>]*>|\s+', ' ', row_string).strip())
    return result_list

# -------------------------
# Text preprocessing
# -------------------------
def remove_stopwords_vi(text: str, path_documents_vi: str='stopwords-vietnamese.txt') -> str:
    if not os.path.exists(path_documents_vi):
        raise FileNotFoundError(f"Stopwords file not found: {path_documents_vi}")
    
    tokens = text.split(',')
    id_str, link_str, name_str = tokens[:3]
    content = ','.join(tokens[3:])
    
    stop_words = set(open(path_documents_vi, encoding="utf-8").read().splitlines())
    filtered_tokens = [w.strip() for w in str(word_tokenize(content, format="text")).split(',') if w.strip().lower() not in stop_words]
    
    return f"{id_str},{link_str},{name_str}," + ', '.join(filtered_tokens)

def clean_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'<.*?>', ' ', text)
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    tokens = text.split(',')
    cleaned_tokens = tokens[:3] + [re.sub(r'[^0-9a-zA-ZÀ-Ỹà-ỹ\s]', '', t) for t in tokens[3:]]
    return ' '.join(cleaned_tokens)

def run_load_data_to_embedding(path: str) -> list:
    df = load_excel(path)
    return convert_table_to_rows(df, allow_combined_info=False)

def normalize_vnese(text: str)-> str:
    for i, j in TONE_NORM_VI.items():
        text = text.replace(i, j)
    # Remove control characters (ASCII 0–31, plus DEL 127)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    # normalize spacing
    text = text.replace('\xa0', ' ')
    # Normalize input text to NFC
    text = unicodedata.normalize("NFC", text)
    return text

def clean_vietnamese_text(text: str) -> str:
    """Loại bỏ ký tự không phải chữ cái hoặc số tiếng Việt."""
    VIETNAMESE_CHARS = (
        "QWERTYUIOPASDFGHJKLZXCVBNMMM" # english uppercase
        "qwertyuiopasdfghjklzxcvbnm" # english lowercase
        "àáảãạăằắẳẵặâầấẩẫậ" # vietnamese
        "đ"
        "èéẻẽẹêềếểễệ"
        "ìíỉĩị"
        "òóỏõọôồốổỗộơờớởỡợ"
        "ùúủũụưừứửữự"
        "ỳýỷỹỵ"
        "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
        "Đ"
        "ÈÉẺẼẸÊỀẾỂỄỆ"
        "ÌÍỈĨỊ"
        "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
        "ÙÚỦŨỤƯỪỨỬỮỰ"
        "ỲÝỶỸỴ"
        "0123456789"  # numbers
        "," # keep comma as it is used as a separator
    )
    pattern = f"[^{VIETNAMESE_CHARS}]"
    cleaned_text = re.sub(pattern, " ", text)
    return re.sub(r'\s+', ' ', cleaned_text).strip()

def normalize_record(text: str, fix_inch_heu=False) -> str:
    """Normalize a text record.

    fix_inch_heu is kept for backward compatibility with the Helpers wrapper.
    Currently it is unused but accepted to avoid calling signature mismatches.
    """
    if not text:
        return text
    text = re.sub(r'<.*?>', ' ', text).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\b[nN][aA][nN]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

from pathlib import Path

DEFAULT_STOPWORDS_PATH = Path(__file__).resolve().parent.parent / "store" / "stopwords-vietnamese.txt"

def remove_stopwords_and_not_vi(
        text: str,
        path_documents_vi: Optional[Union[str, Path]] = None,
        keep_nouns: bool = True,
    ) -> str:
        """Loại bỏ stop-word, nhưng LUÔN giữ lại danh từ."""
    
        global DEFAULT_STOPWORDS_PATH
        if not text:
            return text

        sw_path = Path(path_documents_vi or DEFAULT_STOPWORDS_PATH)
        if not sw_path.exists():
            raise FileNotFoundError(f"Stopwords file not found: {sw_path}")
        stop_words = {w.strip().lower() for w in sw_path.read_text(encoding="utf-8").splitlines()}

        # Tách phần CSV (nếu có)
        tokens = text.split(",")
        if len(tokens) >= 4:
            id_str, link_str, name_str = tokens[:3]
            content = ",".join(tokens[3:])
        else:
            id_str = link_str = name_str = ""
            content = text

        # POS tagging
        pos_tags = pos_tag(content)  # nhận str, trả [(word, pos), ...]

        # Lọc: chỉ loại bỏ khi vừa là stop-word vừa KHÔNG phải danh từ
        filtered = [
            word
            for word, pos in pos_tags
            if word.lower() not in stop_words or pos.startswith("N")
        ]

        cleaned = " ".join(filtered)
        if id_str:
            return f"{id_str},{link_str},{name_str},{cleaned}"
        return cleaned

from pyvi import ViTokenizer, ViPosTagger
from typing import List, Dict, Any, Optional, Union, Protocol
from underthesea import word_tokenize, pos_tag

def bm25_preprocessing_func(text: str) -> List[str]:
    """
    First: MÓN GÀ, VỊT & TRỨNG, Vịt kho gừng, Thịt vịt chặt khúc, kho cùng gừng, mắm, đường cho săn, Vịt, gừng, mắm, đường, món mặn, Gừng nồng, mùi vịt kho dậy mùi, 2-3 người
    Final: ['gà', 'vịt trứng', 'vịt kho gừng', 'thịt vịt chặt khúc', 'kho gừng', 'mắm', 'đường săn', 'vịt', 'gừng', 'mắm', 'đường', 'mặn', 'gừng nồng', 'mùi vịt kho dậy mùi', '2 3 người']
    
    First: MÓN GÀ, VỊT & TRỨNG, Trứng chiên thịt, Trứng gà đánh tan, trộn thịt băm, nêm gia vị, chiên vàng, Trứng gà, thịt ba chỉ băm, món béo, mặn, Trứng chiên vàng thơm, hành lá, 2-3 người
    Final: ['gà', 'vịt trứng', 'trứng chiên thịt', 'trứng gà đánh tan', 'trộn thịt băm', 'nêm gia vị', 'chiên vàng', 'trứng gà', 'thịt ba chỉ băm', 'béo', 'mặn', 'trứng chiên vàng thơm', 'hành lá', '2 3 người']

    First: NƯỚC MÁT NHÀ LÀM, Trà đá, Trà nấu, Trà, Nước, Trà, 1 người
    Final: ['nước mát', 'trà đá', 'trà nấu', 'trà', 'nước', 'trà', '1 người']
    
    First: NƯỚC MÁT NHÀ LÀM, Coca / 7 UP, Nước ngọt có gas, Nước ngọt, Soft Drink, Gas, 1 người
    Final: ['nước mát', 'coca 7 up', 'nước ngọt gas', 'nước ngọt', 'soft drink', 'gas', '1 người']
    
    First: NƯỚC MÁT NHÀ LÀM, Nước suối, Nước suối thanh lọc, Nước suối, Nước, Nước, 1 người
    Final: ['nước mát', 'nước suối', 'nước suối thanh lọc', 'nước suối', 'nước', 'nước', '1 người']
    
    First: NƯỚC MÁT NHÀ LÀM, Bia các loại (Tiger, Heineken, Saigon), Bia các loại, Vị lúa mạch, Beer, Bia, 1 người
    Final: ['nước mát', 'bia loại tiger', 'heineken', 'saigon', 'bia loại', 'vị lúa mạch', 'beer', 'bia', '1 người']
    """
    normalized = normalize_vnese(text)
    # print("First:", normalized)
    normalized = clean_vietnamese_text(normalized)
    # print("0.1:", normalized)
    normalized = ' '.join(word_tokenize(normalized))
    # print("0.5:", normalized)
    normalized = ViTokenizer.tokenize(normalized)
    # print("1:", normalized)
    sequences = [str(normalize_record(text=seq)).lower() for seq in normalized.split(" , ")]
    # print("2:", sequences)
    sequences = [remove_stopwords_and_not_vi(text=seq) for seq in sequences]
    # print("3:", sequences)
    sequences = [str(normalize_record(text=seq)).lower() for seq in sequences]

    sequences = [seq.replace("_", " ") for seq in sequences if seq]
    # print("Final:", sequences)
    return sequences

def run_normalization_data(sequences: list[str]) -> list[str]:

    sequences = [' '.join(bm25_preprocessing_func(seq)) for seq in sequences]
    return sequences

# -------------------------
# Init model embedding
# -------------------------

def get_model_qwen(device: str = "cuda:0") -> HuggingFaceEmbeddings:
    
    '''For Local'''

    # Load model embedding (phải giống model lúc insert để đảm bảo tương thích vector dim)
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs = {'device': device}
                )

def get_qwen_embedding_hf_endpoint(base_url: str = 'http://localhost:8080', device: str = "cuda:0") -> HuggingFaceEndpointEmbeddings:

    ''' For Docker '''

    return HuggingFaceEndpointEmbeddings(
        model=base_url,
    )

def get_openai_embedding_base_url(base_url: str = 'http://localhost:8080') -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=base_url,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY_EMBED"),
        # With the `text-embedding-3` class
        # of models, you can specify the size
        # of the embeddings you want returned.
        dimensions=1024
    )

# -------------------------
# Init vector store
# -------------------------

def init_vectorstore_faiss(model, db_folder: str, action: str ='write') -> FAISS:

    if action == "write":
        docs = run_load_data_to_embedding('./src/store/comque_new.csv')
        docs = run_normalization_data(docs)
        docs = [' | '.join(doc.split(', ')) for doc in docs]
        
        dim = len(model.embed_query("hello world"))  # dimension
        index = faiss.IndexFlatL2(dim)

        vector_store = FAISS(
            embedding_function=model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        docs_faiss = [Document(page_content=txt) for txt in docs]
        uuids = [str(uuid4()) for _ in range(len(docs_faiss))]

        # -------------------------------
        # 5. Add to FAISS
        # -------------------------------
        for i in range(0, len(docs_faiss), 32):
            vector_store.add_documents(documents=docs_faiss[i:i+32], ids=uuids[i:i+32])

        # SAVE DATABASE
        vector_store.save_local(db_folder)
    else:
        # load FAISS từ ./data
        vector_store = FAISS.load_local(
            folder_path=db_folder,
            embeddings=model,
            allow_dangerous_deserialization=True  # BẬT lên nếu file do bạn tạo
        )
    
    return vector_store

def init_vectorstore(model, db_folder: str, connection, action:str='write') -> SQLiteVec:

    if action == 'write':
        os.makedirs(db_folder, exist_ok=True)

        db_file = db_folder+"/vec.db"
        vt = SQLiteVec(table="state_union", connection=connection, embedding=model)
        
        # Nếu DB chưa tồn tại, thêm documents
        if not os.path.exists(db_file):
            
            docs = run_load_data_to_embedding('../store/comque_new.csv')
            docs = run_normalization_data(docs)
            list_docs = [Document(page_content=dox, metadata={dox.split(',')[0]}) for dox in docs]
            vt.add_documents(list_docs)
        
    return vt