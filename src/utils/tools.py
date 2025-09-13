import os
from typing import Literal
from langchain.tools import tool
from src.utils.schemas import HybridSearchInput
from langchain_tavily import TavilySearch
# from langchain_openai import ChatOpenAI
from src.utils.react_constants import *

from langchain_community.vectorstores import SQLiteVec
# import faiss
# from langchain_community.vectorstores import FAISS
from src.utils.toolhelper import run_load_data_to_embedding, run_normalization_data, get_model_qwen, get_qwen_embedding_hf_endpoint, get_openai_embedding_base_url, init_vectorstore_faiss, init_vectorstore

from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch
search_tool = TavilySearch()


# -------------------------
# Module-level cache
# -------------------------

# _MODEL = get_qwen_embedding_hf_endpoint("http://localhost:8080") # chưa chính xác
# _MODEL = get_model_qwen() # chính xác
# _MODEL = get_openai_embedding_base_url() # chưa chính xác
_MODEL = None

# -------------------------
# Hybrid Search Tool
# -------------------------
@tool("hybrid_search", args_schema=HybridSearchInput)
def hybrid_search(
    query: str,
    k: int
) -> str:
    """
    Perform a hybrid semantic search over a local FAISS (or SQLiteVec) vectorstore.
    """

    global _MODEL
    if _MODEL is None:
        _MODEL = get_model_qwen(device='cpu')
    
    path_db_folder = "./src/data"   # fixed: use consistent folder path
    
    os.makedirs(path_db_folder, exist_ok=True)

    db_file = os.path.join(path_db_folder, "vec.db")
    
    connection = SQLiteVec.create_connection(db_file=db_file)
    # vt = init_vectorstore(_MODEL, path_db_folder, connection = connection) # oke
    
    if not os.path.exists('./src/data/index.pkl'):
        vt = init_vectorstore_faiss(_MODEL, db_folder=path_db_folder, action='write') # oke
    else:
        vt = init_vectorstore_faiss(_MODEL, db_folder=path_db_folder, action='load') # oke
        
    results = vt.similarity_search(query, k=k)
    vector_results = [doc.page_content for doc in results]

    combined_results = [
        {"_id": f"{v.split()[0]}", "_content": " ".join(v.split()[1:])}
        for v in vector_results
    ]

    return "\n".join(
        [f"{r.get('_id', 'N/A')} - {r.get('_content', '')}" for r in combined_results]
    )

# exports all tools for agent
all_agent_tools = [hybrid_search]