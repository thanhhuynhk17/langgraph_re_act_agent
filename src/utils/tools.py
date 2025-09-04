import os
from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
search_tool = TavilySearch()