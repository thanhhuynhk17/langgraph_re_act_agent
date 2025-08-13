# tools.py
import os
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings  # or other embedding loader

# 1️⃣ Load environment variables from .env file
load_dotenv()

# 2️⃣ Read env variables
neo4j_url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE", "neo4j")
embedding_model_url = os.getenv("EMBEDDINGS_MODEL")
text_embed_dim = int(os.getenv("TEXT_EMBED_DIM", "768"))
index_name = "my_index"  # you can also make this from env

# 3️⃣ Initialize embedding model
embedding_model = OpenAIEmbeddings(
    base_url=f"{embedding_model_url}/v1",
    api_key="dummy_text"
)
# 4️⃣ Define Pydantic schema for tool input
class StoreDocInput(BaseModel):
    doc: str = Field(..., description="Nội dung văn bản cần lưu vào vector store Neo4j")

# 5️⃣ The tool function
@tool("store_doc_in_neo4j", args_schema=StoreDocInput)
def store_doc_in_neo4j(doc: str) -> str:
    """
    Lưu một văn bản (doc) vào vector store Neo4j cho tìm kiếm ngữ nghĩa.
    """
    from langchain.schema import Document

    documents = [Document(page_content=doc)]

    Neo4jVector.from_documents(
        documents=documents,
        embedding=embedding_model,
        url=neo4j_url,
        username=username,
        password=password,
        database=database,
        index_name=index_name,
        # search_type="hybrid"
    )

    return f"✅ Đã lưu thành công văn bản vào Neo4j index '{index_name}'."
