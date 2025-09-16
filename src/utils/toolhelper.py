import numpy as np
import pandas as pd
import re
import faiss
from langchain_openai import OpenAIEmbeddings
from underthesea import word_tokenize
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import SQLiteVec
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

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

def normalize_vnese(text):
    for i, j in TONE_NORM_VI.items():
        text = text.replace(i, j)
    # Remove control characters (ASCII 0–31, plus DEL 127)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    # normalize spacing
    text = text.replace('\xa0', ' ')
    # Normalize input text to NFC
    text = unicodedata.normalize("NFC", text)
    return text

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


def convert_table_to_rows(df: pd.DataFrame) -> list:
    result_list = []
    for _, row in df.iterrows():
        row_string = ", ".join(str(v) for v in row)
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
    filtered_tokens = [w.strip() for w in word_tokenize(content, format="text").split(',') if w.strip().lower() not in stop_words]
    
    return f"{id_str},{link_str},{name_str}," + ', '.join(filtered_tokens)

def clean_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'<.*?>', ' ', text)
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    tokens = text.split(',')
    cleaned_tokens = tokens[:3] + [re.sub(r'[^0-9a-zA-ZÀ-Ỹà-ỹ\s]', '', t) for t in tokens[3:]]
    return ' '.join(cleaned_tokens)

def normalize_record(text: str, fix_inch_heu=False) -> str:
    if not text:
        return text
    text = re.sub(r'<.*?>', ' ', text).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\b[nN][aA][nN]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_load_data_to_embedding(path: str) -> list:
    df = load_excel(path)
    return convert_table_to_rows(df)

def run_normalization_data(sequences: list, path_stopwords: str="src/stopwords-vietnamese.txt") -> list:
    
    # sequences = [remove_stopwords_vi(seq, path_stopwords) for seq in sequences]
    # sequences = [clean_text(seq) for seq in sequences]
    sequences = [str(normalize_record(seq)).lower() for seq in sequences]
    sequences = [normalize_vnese(seq) for seq in sequences]
    return sequences

def get_model_qwen(device: str = "cuda:0") -> HuggingFaceEmbeddings:
    
    '''For Local'''

    # Load model embedding (phải giống model lúc insert để đảm bảo tương thích vector dim)
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs = {'device': device}
                )

from huggingface_hub import InferenceClient

def get_qwen_embedding_hf_endpoint(base_url: str = 'http://localhost:8080') -> HuggingFaceEndpointEmbeddings:

    ''' For Docker '''

    return HuggingFaceEndpointEmbeddings(
    model=base_url,
    task="feature-extraction",
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
        docs = run_normalization_data(docs, path_stopwords='./src/store/stopwords-vietnamese.txt')
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
        vector_store.add_documents(documents=docs_faiss, ids=uuids)
        
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
            docs = run_normalization_data(docs, path_stopwords='../store/stopwords-vietnamese.txt')
            list_docs = [Document(page_content=dox, metadata={dox.split(',')[0]}) for dox in docs]
            vt.add_documents(list_docs)
        
    return vt



# -------------------------
# CATEGORY SEARCH
# -------------------------

def search_values_in_type(database: pd.DataFrame, name_col: str) -> list[list]:
    """
    Count occurrences of unique values in a column.

    Args:
        name_col (str): Column name to analyze. 
            Must be one of:
            - 'type_of_food'
            - 'name_of_food'
            - 'how_to_prepare'
            - 'main_ingredients'
            - 'taste'
            - 'outstanding_fragrance'
            - 'current_price'
            - 'number_of_people_eating'

    Returns:
        list[list]: A list of [value, count] pairs.
                    Values are lowercased strings, counts are integers.
    """
    allowed_cols = {
        'type_of_food',
        'name_of_food',
        'how_to_prepare',
        'main_ingredients',
        'taste',
        'outstanding_fragrance',
        'current_price',
        'number_of_people_eating'
    }

    if name_col not in allowed_cols:
        raise ValueError(
            f"Invalid column '{name_col}'. Must be one of: {', '.join(allowed_cols)}"
        )

    vals_counts = database[name_col].value_counts()
    return np.column_stack((
        [str(name).lower() for name in vals_counts.index.tolist()],
        vals_counts.values.tolist()
    )).tolist()


def search_type_category_and_people(database: pd.DataFrame, value_type_of_food: str, value_option: str) -> pd.DataFrame:
    """
    Search rows in the database that satisfy three conditions:
    1. The 'type_of_food' column contains the specified food category.
    2. At least one of the other descriptive columns contains the given keyword.
        The following columns are searched:
            - 'name_of_food'
            - 'how_to_prepare'
            - 'main_ingredients'
            - 'taste'
            - 'outstanding_fragrance'
            - 'current_price'
    3. The 'number_of_people_eating' column contains the given number of people.

    Args:
        value_type_of_food (str): Food category filter. 
            Example: "món cá", "món khai vị", "món ăn chơi", "món rau", "món gỏi", "món gà, vịt & trứng", "món tôm & mực", "món xào", "nước mát nhà làm", "lẩu", "món thịt", "món sườn & đậu hũ", "món canh", "các loại khô", "tráng miệng", etc.
        value_option (str): Keyword to search in descriptive columns. 
            Example: "cay", "mặn", "ngọt".

    Returns:
        pd.DataFrame: Subset of the database that matches all three conditions,
                    with the index reset.
    """
    value_option = value_option.lower()

    # 1️⃣ fixed filter on type_of_food
    mask_type = database["type_of_food"].str.lower().str.contains(value_type_of_food.lower(), na=False)

    # 2️⃣ search across other columns (exclude type_of_food + number_of_people_eating)
    search_cols = database.columns.drop(["type_of_food"])
    mask_text = database[search_cols].apply(
        lambda col: col.astype(str).str.lower().str.contains(value_option, na=False)
    )
    mask_any = mask_text.any(axis=1)

    # 4️⃣ combine
    result = database[mask_type & mask_any].reset_index(drop=True)
    return result
