import pandas as pd
import re
from underthesea import word_tokenize
import os
import random
import numpy as np
from rank_bm25 import BM25Okapi
from typing import Optional
from langchain_community.vectorstores import SQLiteVec
from langchain_community.embeddings import HuggingFaceEmbeddings

_MODEL: Optional[HuggingFaceEmbeddings] = None
_DOC_EMBEDDINGS = None
_SEQUENCES: Optional[list] = None

# -------------------------
# Load CSV
# -------------------------
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
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').replace('_', ' ')
    tokens = text.split(',')
    cleaned_tokens = tokens[:3] + [re.sub(r'[^0-9a-zA-ZÀ-Ỹà-ỹ\s]', '', t) for t in tokens[3:]]
    return ','.join(cleaned_tokens)

def normalize_record(text: str, fix_inch_heu=False) -> str:
    if not text:
        return text
    text = re.sub(r'<.*?>', ' ', text).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').replace('_', ' ')
    text = re.sub(r'\b[nN][aA][nN]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_load_data_to_embedding(path: str) -> list:
    df = load_excel(path)
    return convert_table_to_rows(df)

def run_normalization_data(sequences: list, path_stopwords: str="src/stopwords-vietnamese.txt") -> list:
    sequences = [remove_stopwords_vi(seq, path_stopwords) for seq in sequences]
    sequences = [clean_text(seq) for seq in sequences]
    sequences = [normalize_record(seq) for seq in sequences]
    return sequences

# -------------------------
# Embeddings
# -------------------------
def get_model_qwen(device: str = "cuda:0") -> HuggingFaceEmbeddings:
    global _MODEL
    if _MODEL is None:
        _MODEL = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"device": device},
        )
    return _MODEL

def prepare_embeddings(sequences: list, device: str = "cuda:0"):
    global _DOC_EMBEDDINGS, _SEQUENCES
    _MODEL = get_model_qwen(device)
    if _DOC_EMBEDDINGS is None or _SEQUENCES != sequences:
        _DOC_EMBEDDINGS = _MODEL.embed_documents(sequences)
        _SEQUENCES = sequences
    return _DOC_EMBEDDINGS

# -------------------------
# Hybrid Search (BM25 + VectorStore)
# -------------------------
def run_hybrid_search(model: HuggingFaceEmbeddings, query: str, doc_embeddings, k: int = 5):
    path_csv = "src/hoanghamobile.csv"
    path_stopwords = "src/stopwords-vietnamese.txt"
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"CSV file not found: {path_csv}")
    if not os.path.exists(path_stopwords):
        raise FileNotFoundError(f"Stopwords file not found: {path_stopwords}")
    
    # Load & normalize
    sequences = run_load_data_to_embedding(path_csv)
    sequences = run_normalization_data(sequences, path_stopwords)
    
    # BM25
    tokenized_corpus = [doc.split(" ") for doc in sequences]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.split(" ")
    top_n = bm25.get_top_n(tokenized_query, sequences, n=k)
    
    # VectorStore retrieval
    connection = SQLiteVec.create_connection(db_file="../vec.db")
    vector_store = SQLiteVec(table="state_union", db_file="../vec.db", embedding=model, connection=connection)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    vector_docs = retriever.invoke(query)
    
    vector_results = [doc.page_content for doc in vector_docs]
    
    # Combine BM25 + VectorStore results
    combined_results = [{"_id": i, "_content": v} for i, v in enumerate(top_n)]
    combined_results += [{"_id": f"vec_{i}", "_content": v} for i, v in enumerate(vector_results)]
    
    return combined_results
