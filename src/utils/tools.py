import os
from typing import Literal
from langchain.tools import tool
from src.utils.schemas import HybridSearchInput, SearchTypeCategoryAndPeople, SearchValuesInTypeInput
from langchain_tavily import TavilySearch
# from langchain_openai import ChatOpenAI
from src.utils.react_constants import *
from langchain_community.vectorstores import SQLiteVec
# import faiss
# from langchain_community.vectorstores import FAISS
from src.utils.toolhelper import run_load_data_to_embedding, run_normalization_data, get_model_qwen, get_qwen_embedding_hf_endpoint, get_openai_embedding_base_url, init_vectorstore_faiss, init_vectorstore
import pandas as pd
import numpy as np
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
_DATABASE = None
# -------------------------
# Hybrid Search Tool
# -------------------------
@tool("hybrid_search", args_schema=HybridSearchInput)
def hybrid_search(
    query: str,
    k: int
) -> str:
    """
    Perform a semantic search over a local FAISS vectorstore.

    Behavior:
    - Ensures ./src/data exists and opens/creates ./src/data/vec.db (SQLite connection is created but not used for search).
    - Initializes FAISS vectorstore: if ./src/data/index.pkl missing -> build index (action='write'), otherwise load (action='load').
    - Runs semantic similarity search with `vt.similarity_search(query.lower(), k=k)`.
    - Returns a single string where each result line has the format: "<id> - <content>\n".
    - The id is extracted from the first token of the vectorstore document (`doc.page_content.split()[0]`).
    - The content is the rest of the document (`" ".join(tokens[1:])`).
    Notes / Limitations:
    - Despite the function name, this implementation uses FAISS semantic search only (not a hybrid of FAISS+SQLite).
    - Lowercasing the query (`query.lower()`) may affect case-sensitive retrieval.
    - The id extraction method assumes the stored documents begin with an identifier token — change logic if your doc format differs.
    - For robust programmatic use, consider returning a structured object (list[dict]) rather than a single joined string.
    """

    global _MODEL
    if _MODEL is None:
        # _MODEL = get_model_qwen(device='cuda:0') # oke
        _MODEL = get_qwen_embedding_hf_endpoint() # oke
    
    path_db_folder = "./src/data"   # fixed: use consistent folder path
    
    os.makedirs(path_db_folder, exist_ok=True)

    db_file = os.path.join(path_db_folder, "vec.db")
    
    connection = SQLiteVec.create_connection(db_file=db_file)
    vt = init_vectorstore(_MODEL, path_db_folder, connection = connection) # oke
    
    # if not os.path.exists('./src/data/index.pkl'):
    #     vt = init_vectorstore_faiss(_MODEL, db_folder=path_db_folder, action='write') # oke
    # else:
    #     vt = init_vectorstore_faiss(_MODEL, db_folder=path_db_folder, action='load') # oke
        
    results = vt.similarity_search(query.lower(), k=k)
    vector_results = [doc.page_content for doc in results]

    combined_results = [
        {"_id": f"{v.split()[0]}", "_content": " ".join(v.split()[1:])}
        for v in vector_results
    ]

    return "\n".join(
        [f"{r.get('_id', 'N/A')} - {r.get('_content', '')}" for r in combined_results]
    )

# -------------------------
# Category Search Tool
# -------------------------
@tool("search_type_category_and_people", args_schema=SearchTypeCategoryAndPeople)
def search_type_category_and_people(
    value_type_of_food: Literal["món cá", "món khai vị", "món ăn chơi", "món rau", "món gỏi", "món gà, vịt & trứng", "món tôm & mực", "món xào", "nước mát nhà làm", "lẩu", "món thịt", "món sườn & đậu hũ", "món canh", "các loại khô", "tráng miệng"], 
    value_option: str)-> pd.DataFrame:

    """
    Search rows in the food database that satisfy three conditions:
    1. The 'type_of_food' column contains the specified food category (e.g. 'món cá').
    2. At least one of the other descriptive columns contains the given keyword
       (searched case-insensitively across: 'name_of_food', 'how_to_prepare',
        'main_ingredients', 'taste', 'outstanding_fragrance', 'current_price').

    Args:
        value_type_of_food (str): The food category filter, must match one of the predefined categories.
        value_option (str): Keyword to search in descriptive columns.

    Returns:
        pd.DataFrame: A subset of the database that matches all three conditions,
                      with the index reset.
    """
    
    global _DATABASE
    
    if _DATABASE is None:
        _DATABASE = pd.read_csv('./src/store/comque_new.csv')

    value_option = value_option.lower()

    # 1️⃣ fixed filter on type_of_food
    mask_type = _DATABASE["type_of_food"].str.lower().str.contains(value_type_of_food.lower(), na=False)

    # 2️⃣ search across other columns (exclude number_of_people_eating)
    search_cols = _DATABASE.columns.drop("number_of_people_eating")
    mask_text = _DATABASE[search_cols].apply(
        lambda col: col.astype(str).str.lower().str.contains(value_option, na=False)
    )
    mask_any = mask_text.any(axis=1)

    # 4️⃣ combine
    result = _DATABASE[mask_type & mask_any].reset_index(drop=True)
    return result

@tool("search_values_in_type", args_schema=SearchValuesInTypeInput)
def search_values_in_type(
    name_col: Literal[
        "type_of_food",
        "name_of_food",
        "how_to_prepare",
        "main_ingredients",
        "taste",
        "outstanding_fragrance",
        "current_price",
        "number_of_people_eating"
    ]
) -> list[list]:
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

    Example:
        >>> search_values_in_type("type_of_food")
        [['món cá', 12], ['món thịt', 8], ['lẩu', 5]]
    """
    global _DATABASE
    
    if _DATABASE is None:
        _DATABASE = pd.read_csv("./src/store/comque_new.csv")

    vals_counts = _DATABASE[name_col].value_counts()
    return np.column_stack((
        [str(name).lower() for name in vals_counts.index.tolist()],
        vals_counts.values.tolist()
    )).tolist()
# exports all tools for agent
all_agent_tools = [hybrid_search, search_type_category_and_people, search_values_in_type]
