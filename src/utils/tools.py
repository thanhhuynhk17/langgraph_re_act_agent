import os
from typing import Literal
from langchain.tools import tool
from src.utils.schemas import HybridSearchInput
from src.utils.toolhelper import run_hybrid_search
from dotenv import load_dotenv
load_dotenv()


from langchain_tavily import TavilySearch
search_tool = TavilySearch()


@tool("hybrid_search", args_schema=HybridSearchInput)
def hybrid_search(query: str, k: Literal[10, 20] = 10) -> str:
    """
    Query the shop's phone sales database using a hybrid search (semantic + keyword).
    This tool retrieves the most relevant product or sales information based on the query.
    
    Args:
        query (str): The search query text. Can be natural language (e.g., "latest iPhone deals")
                     or keywords (e.g., "Samsung discount").
        k (Literal[10, 20]): The number of top results to return (10 or 20).
    
    Returns:
        str: A formatted list of the most relevant phone sales records.
    """
    # 🔎 Replace with actual hybrid search logic (e.g., BM25 + vector embeddings)
    result = run_hybrid_search(query, k)    
    return "\n".join([f"{id_result['_id'], id_result['_content']}" for id_result in result])