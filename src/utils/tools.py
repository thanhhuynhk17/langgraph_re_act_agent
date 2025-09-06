import os
from typing import Literal
from langchain.tools import tool
from src.utils.schemas import HybridSearchInput
# from src.utils.toolhelper import run_hybrid_search
from dotenv import load_dotenv
load_dotenv()


from langchain_tavily import TavilySearch
search_tool = TavilySearch()



# @tool("hybrid_search", args_schema=HybridSearchInput)
# def hybrid_search(query: str, k):
#     """
#     Performs a hybrid search (semantic + keyword) and returns the most relevant documents.
#     """
#     # 🔎 Replace with actual hybrid search logic (e.g., BM25 + vector embeddings)
#     result = run_hybrid_search(query, k)
    
#     return "\n".join([f"{id_result['_idx'], id_result['_content']}" for id_result in result])