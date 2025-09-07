import os
from typing import Literal, Optional, List
from langchain.tools import tool
from src.utils.schemas import HybridSearchInput
from src.utils.toolhelper import run_hybrid_search, run_load_data_to_embedding, run_normalization_data
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SQLiteVec
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

search_tool = TavilySearch()

# -------------------------
# Module-level cache
# -------------------------
_MODEL: HuggingFaceEmbeddings
_DOC_EMBEDDINGS = None
_SEQUENCES: list[str] = list()
_STORE: SQLiteVec
# -------------------------
# Model Embeddings
# -------------------------

def prepare_embeddings(sequences: list[str], device: str = "cuda:0"):
    """Encode sequences and cache embeddings."""
    global _DOC_EMBEDDINGS, _SEQUENCES
    _MODEL = get_model_qwen()
    if _DOC_EMBEDDINGS is None or _SEQUENCES != sequences:
        _DOC_EMBEDDINGS = _MODEL.embed_documents(sequences)
        _SEQUENCES = sequences
        
    

    return _DOC_EMBEDDINGS

# -------------------------
# Vector Store (thread-safe, persistent)
# -------------------------
def get_vector_store(
    table_name: str = "state_union",
    db_file: str = "../vec.db",
    sequences = list(),
    device: str = "cuda:0"
) -> SQLiteVec:
    """Create a thread-safe SQLiteVec vector store. Add embeddings in chunks if DB is empty."""
    
    connection = SQLiteVec.create_connection(db_file=db_file)

    _STORE = SQLiteVec(
        table=table_name,
        db_file=db_file,
        embedding=get_model_qwen(),
        connection=connection
    )
    
    if not os.path.exists(db_file):
        _STORE.add_documents(sequences)

    return _STORE

def get_model_qwen() -> HuggingFaceEmbeddings:
    # Load model embedding (phải giống model lúc insert để đảm bảo tương thích vector dim)
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs = {'device': 'cpu'}
                )
    return embeddings




from langchain.schema import Document
# -------------------------
# Hybrid Search Tool
# -------------------------
@tool("hybrid_search", args_schema=HybridSearchInput)
def hybrid_search(
    query: str,
    k: Literal[5, 10, 20] = 5,
    sequences:list[str] = list(),
    doc_embeddings=None,
    use_vector_store: bool = True,
    device: str = "cuda:0",
    ) -> str:
    
    global _DOC_EMBEDDINGS, _SEQUENCES

    
    
    # Nếu DB chưa tồn tại, thêm documents
    if not os.path.exists('../vec.db') and sequences:
        
        path_csv = "src/hoanghamobile.csv"
        path_stopwords = "src/stopwords-vietnamese.txt"
        
        if not os.path.exists(path_csv):
            raise FileNotFoundError(f"Data CSV file not found: {path_csv}")
        if not os.path.exists(path_stopwords):
            raise FileNotFoundError(f"Stopwords file not found: {path_stopwords}")

        sequences = run_load_data_to_embedding(path_csv)
        sequences = run_normalization_data(sequences, path_stopwords=path_stopwords)
        
        db_file = '../vec.db'
        connection = SQLiteVec.create_connection(db_file=db_file)
        
        _STORE = SQLiteVec(
            table= "state_union",
            db_file=db_file,
            embedding=get_model_qwen(),
            connection=connection
        )
        
        docs = [Document(page_content=s) for s in sequences]  # Chuyển str -> Document
        _STORE.add_documents(docs)
        
    else:
        _MODEL = get_model_qwen()
        db_file = '../vec.db'
        
        connection = SQLiteVec.create_connection(db_file=db_file)
        
        _STORE = SQLiteVec(
            table= "state_union",
            db_file=db_file,
            embedding=_MODEL,
            connection=connection
        )
    
    # # --- Prepare embeddings ---
    # valid_sequences = sequences if isinstance(sequences, list) and all(isinstance(s, str) for s in sequences) else []
    # emb = doc_embeddings if doc_embeddings is not None else prepare_embeddings(valid_sequences, device=device)

    # # --- BM25 + embeddings search ---
    # bm25_results = run_hybrid_search(_MODEL, query, emb, k)

    # --- Vector store retrieval ---
    vector_results = []
    if use_vector_store:
        vector_store = get_vector_store(sequences=sequences, device=device)
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        vector_docs = retriever.invoke(query)
        vector_results = [doc.page_content for doc in vector_docs]

    # --- Combine results ---
    combined_results = [{"_id": f"{i+1}", "_content": v} for i, v in enumerate(vector_results)]

    return "\n".join([f"{r.get('_id', 'N/A')} - {r.get('_content', '')}" for r in combined_results])
