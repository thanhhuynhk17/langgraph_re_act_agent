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


from typing import List
from pydantic import BaseModel, Field, field_validator
from graphdb.utils.helpers import normalize_vnese
from enum import Enum

# Load WORDLIST chỉ chứa từ
with open("4096_output.txt", "r", encoding="utf-8") as f:
    WORDLIST = [
        normalize_vnese(line.strip())
        for line in f
        if line.strip()
    ][:1000]  # 2000 từ đầu tiên

# Tạo Enum động
BagOfWordsEnum = Enum("BagOfWordsEnum", {w: w for w in WORDLIST}, type=str)

class BagOfWordsResponse(BaseModel):
    bag_of_words: List[BagOfWordsEnum] = Field(
        ...,
        description=(
            "Danh sách các từ phù hợp với yêu cầu của người dùng, tối đa 8 từ.\n"
            f"Bộ từ điển các từ được phép sử dụng:\n{', '.join(sorted(WORDLIST))}"
        ),
        max_items=8
    )

    @field_validator("bag_of_words", mode="before")
    @classmethod
    def validate_words(cls, v: List[str]) -> List[str]:
        # Kiểm tra số lượng từ
        if len(v) > 8:
            raise ValueError(f"Số từ quá nhiều: {len(v)}. Chỉ được phép tối đa 8 từ.")
        
        # Kiểm tra từ không hợp lệ
        invalid = [w for w in v if w not in WORDLIST]
        if invalid:
            raise ValueError(
                f"Các từ sau không nằm trong bộ từ điển, phải bị loại bỏ: {', '.join(invalid)}.\nChỉ được phép sử dụng các từ hợp lệ."
            )
        
        return v

@tool(
    args_schema=BagOfWordsResponse,  # input/output validation via Pydantic
    description="Generate up to 8 relevant words from WORDLIST based on the user prompt."
)
def bag_of_words_generator(bag_of_words: BagOfWordsResponse) -> BagOfWordsResponse:
    """
    This tool allows an LLM to populate the `bag_of_words` field automatically.
    The LLM should only return words from WORDLIST and up to 8 items.
    """
    # This function does not manually select words.
    # The LLM is responsible for filling `bag_of_words`.
    return ", ".join(bag_of_words)