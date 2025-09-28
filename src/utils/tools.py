import os
from typing import Literal
from langchain.tools import tool
from src.utils.schemas import (
    HybridSearch, SearchTypeCategory, SearchMultiTypeCategory,
    ColumnValueCount, 
    Dish, CustomerInfo, TakeOrderInput, UpdateOrderInput, DeleteOrderInput
)
from langchain_core.tools.base import ArgsSchema
from typing import Literal, Dict, Optional, Any, List, Type
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch
# from langchain_openai import ChatOpenAI
from src.utils.react_constants import *

from langchain_community.vectorstores import SQLiteVec
from langchain_community.vectorstores import FAISS
from src.utils.toolhelper import bm25_preprocessing_func, get_model_qwen, get_qwen_embedding_hf_endpoint, get_openai_embedding_base_url, init_vectorstore_faiss, init_vectorstore

from dotenv import load_dotenv
import pandas as pd
import re
import numpy as np
from rank_bm25 import BM25Okapi
from typing import ClassVar
from typing import Type
load_dotenv()

from langchain_tavily import TavilySearch
search_tool = TavilySearch()

# -------------------------
# Module-level cache
# -------------------------

# _MODEL = get_qwen_embedding_hf_endpoint("http://localhost:8080") # chưa chính xác
# _MODEL = get_model_qwen() # chính xác
# _MODEL = get_openai_embedding_base_url() # chưa chính xác
_DATABASE: Optional[pd.DataFrame] = None
_MODEL: Optional[Any] = None
_VECTOR_STORE: Optional[FAISS] = None
_BM25: Optional[BM25Okapi] = None

def get_database() -> pd.DataFrame:
    global _DATABASE
    if _DATABASE is None:
        _DATABASE = pd.read_csv('./src/store/comque_new.csv')
    return _DATABASE
# -------------------------
# Hybrid Search Tool
# -------------------------
from pydantic import BaseModel, Field

# @tool("hybrid_search", args_schema=HybridSearch, description=(
#     "Tìm các món ăn liên quan nhất đến các keyword theo mô tả của khách hàng. "
#     "Cú pháp: <mô tả bao quát về một hoặc nhiều món ăn>"
#     "Ví dụ:"
#     " - Món tôm rang muối cay thế nào vậy em ? -> <mô tả món 1>" 
#     " - Món nào vừa cay vừa có cá ? -> <mô tả món 1 và 2>"
#     " - Món nào nhiều rau ít calo ? -> <mô tả món>"
# )
# )
# def hybrid_search(
#     query: str,
#     k: int = 5
# ) -> str:
#     global _MODEL

#     if _MODEL is None:
#         _MODEL = get_qwen_embedding_hf_endpoint("http://localhost:8080", device="cpu")
    
#     path_db_folder = "./src/data"   # fixed: use consistent folder path
    
#     os.makedirs(path_db_folder, exist_ok=True)

#     # db_file = os.path.join(path_db_folder, "vec.db")
#     # connection = SQLiteVec.create_connection(db_file=db_file)
#     # vt = init_vectorstore(_MODEL, path_db_folder, connection = connection) # oke
    
#     if not os.path.exists('./src/data/index.pkl'):
#         vt = init_vectorstore_faiss(_MODEL, db_folder=path_db_folder, action='write') # oke
#     else:
#         vt = init_vectorstore_faiss(_MODEL, db_folder=path_db_folder, action='load') # oke
        
#     results = vt.similarity_search(query, k=k)
#     vector_results = [str(bm25_preprocessing_func(doc.page_content)[0]) + " \n\n " for doc in results]

    
#     combined_results = [
#         {"_id": f"{v.split()[2]}", "_content": " ".join(v.split()[3:]) + " \n\n "}
#         for v in vector_results
#     ]

#     print(f"DEBUG: combined_results = {combined_results}")

#     return "\n".join(
#         [f"Mã món: {r.get('_id')} - {r.get('_content')}" for r in combined_results]
#     )
class HybridSearchTool(BaseTool):
    name: str = "hybrid_search"
    description: str = (
        "Tìm các món ăn liên quan nhất đến các keyword theo mô tả của khách hàng. "
        "Cú pháp: <mô tả bao quát về một hoặc nhiều món ăn>"
        "Ví dụ:"
        " - Món tôm rang muối cay thế nào vậy em ? -> <mô tả món 1>" 
        " - Món nào vừa cay vừa có cá ? -> <mô tả món 1 và 2>"
        " - Món nào nhiều rau ít calo ? -> <mô tả món>"
    )
    args_schema: Type[BaseModel] = HybridSearch
    return_direct: bool = False

    _model: Optional[object] = None  # cache model

    @staticmethod
    def _get_model():
        # singleton
        global _MODEL
        if _MODEL is None:
            _MODEL = get_qwen_embedding_hf_endpoint(
                "http://localhost:8080", device="cuda:0"
            )
        return _MODEL
    
    def _run(self, query: str, k: int) -> str:
        model = self._get_model()

        path_db_folder = "./src/data"
        os.makedirs(path_db_folder, exist_ok=True)

        if not os.path.exists(os.path.join(path_db_folder, "index.pkl")):
            vt = init_vectorstore_faiss(
                model, db_folder=path_db_folder, action="write"
            )
        else:
            vt = init_vectorstore_faiss(
                model, db_folder=path_db_folder, action="load"
            )

        docs = vt.similarity_search(query, k=k)
        combined = [
            {"_id": doc.page_content.split()[0],
             "_content": " ".join(doc.page_content.split()[1:])}
            for doc in docs
        ]
        return "\n".join(
            [f"{r.get('_id', 'N/A')} - {r.get('_content', '')}" for r in combined]
        )

    async def _arun(self, query: str, k: int) -> str:
        # chạy đồng bộ cho đơn giản; có thể chuyển sang asyncio.to_thread nếu cần
        return self._run(query, k)

class SearchTypeCategoryTool(BaseTool):
    name: str = "filter_category"
    description: str = (
        "Tìm món thuộc 1 nhóm duy nhất, có thể thêm 1 từ khóa. "
        "Cú pháp: <nhóm món> [<từ khóa>] "
        "Ví dụ: "
        "- Cá <cay> "
        "- Lẩu cá <cay> "
    )
    args_schema: Type[BaseModel] = SearchTypeCategory
    return_direct: ClassVar[bool] = False

    # ---------- singleton ----------
    _db: ClassVar[Optional[pd.DataFrame]] = None

    @classmethod
    def get_db(cls) -> pd.DataFrame:
        if cls._db is None:
            cls._db = get_database()
        return cls._db

    # ---------- logic ----------
    def _run(
        self,
        value_type_of_food: Literal[
            "món cá", "món khai vị", "món ăn chơi", "món rau", "món gỏi",
            "món gà, vịt & trứng", "món tôm & mực", "món xào", "nước mát nhà làm",
            "lẩu", "món thịt", "món sườn & đậu hũ", "món canh", "các loại khô", "tráng miệng"
        ],
        key_value_option: str
    ) -> str:
        df = self.get_db()

        # 1. Lọc chính xác loại món
        mask_type = df["type_of_food"].str.lower().eq(value_type_of_food.lower())

        # 2. Lọc keyword (nếu có)
        kw = key_value_option.strip()
        if kw:
            cols = df.columns.drop("number_of_people_eating")
            mask_kw = df[cols].apply(
                lambda col: col.astype(str).str.lower().str.contains(kw, na=False)
            ).any(axis=1)
            mask_type &= mask_kw

        # 3. Lấy tên món & clean
        names = df.loc[mask_type, "_name"].dropna().unique()
        cleaned = [re.sub(r"[^\w\s]", "", n).strip() for n in names if n.strip()]
        return "\n".join(cleaned) if cleaned else "Không có món phù hợp."

    async def _arun(self, *args, **kwargs):
        return self._run(*args, **kwargs)

class SearchMultiTypeCategoryTool(BaseTool): # chưa xong
    name: str = "multi_filter_category"
    description: str = (
       "Chọn nhiều nhóm món cùng lúc, mỗi nhóm có thể kèm 1 từ khóa riêng. "
       "Cú pháp: <nhóm món> [<từ khóa>], ... "
       "Ví dụ: "
       "- món cá <>, món lẩu <>, món rau <> "
       "- món cá <không cay>, món lẩu <không cay>, món rau <không cay> "
       )
    args_schema: Type[BaseModel] = SearchMultiTypeCategory
    handle_tool_error: bool = True

    def _run(self, categories: List[str], keywords: List[str]) -> str:
        if len(categories) != len(keywords):
            print("DEBUG: Số lượng categories và keywords KHÔNG bằng nhau.")
            return "Số lượng categories và keywords phải bằng nhau."

        df = get_database()
        print(f"DEBUG: DataFrame shape = {df.shape}")
        print(f"DEBUG: categories = {categories}")
        print(f"DEBUG: keywords   = {keywords}")

        out: List[str] = []

        for i in range(len(categories)):
            cat = categories[i]
            kw  = keywords[i]
            print(f"\nDEBUG --- Vòng {i}: cat='{cat}', kw='{kw}'")

            mask = df["type_of_food"].str.lower().eq(cat.lower())
            if kw.strip():
                cols = [c for c in df.columns if c != "number_of_people_eating"]
                kw_mask = df[cols].apply(
                    lambda col: col.astype(str).str.lower().str.contains(kw.lower(), na=False)
                ).any(axis=1)
                mask &= kw_mask

            names = df.loc[mask, "_name"].dropna().unique().tolist()
            print(f"DEBUG   names tìm được: {names}")
            out.extend(names)

            # ---- in ngay kết quả của cặp này ----
            if names:
                print(f"KẾT QUẢ CẶP {i}: {cat} + '{kw}' → {len(names)} món")
                for n in names:
                    print(f"  - {n}")
            else:
                print(f"KẾT QUẢ CẶP {i}: {cat} + '{kw}' → Không có món")

        seen = dict.fromkeys(out)
        cleaned = [re.sub(r"[^\w\s]", "", n).strip() for n in seen if n.strip()]
        print(f"\nDEBUG: cleaned final = {cleaned}")
        return "\n".join(cleaned) if cleaned else "Không tìm thấy món nào."

    async def _arun(self, *args, **kwargs):
        return self._run(*args, **kwargs)

class ColumnValueCountTool(BaseTool):
    name: str = "column_value_count"
    description: str = (
        "Khi khách hỏi “Menu có gì?” / “Nhà hàng có bao nhiêu món?” thì dùng hàm này trước tiên. "
        "Trả về thống kê: mỗi loại món (cá, lẩu, rau…) kèm số lượng và gợi ý 3-5 món tiêu biểu. "
        "Ví dụ: “Menu hiện tại gồm: món cá (6 món): cá kho tộ, cá hấp xì dầu,…; món lẩu (4 món): lẩu Thái, lẩu cua,…; món rau (3 món): rau muống xào,…”"
    )
    args_schema: Type[BaseModel] = ColumnValueCount
    handle_tool_error: bool = True

    _db: ClassVar[Optional[pd.DataFrame]] = None

    @classmethod
    def get_db(cls) -> pd.DataFrame:
        if cls._db is None:
            cls._db = get_database()
        return cls._db

    def _run(self, name_col: str) -> str:
        df = self.get_db()
        if name_col not in df.columns:
            return f"Cột '{name_col}' không tồn tại."

        # Đếm & gộp tên món
        grouped = (
            df.dropna(subset=[name_col])
            .groupby(name_col, sort=False)
            .agg(count=("_name", "size"),
                foods=("_name", lambda s: ", ".join(dict.fromkeys(s.dropna()))))
            .reset_index()
            .sort_values("count", ascending=False)
        )

        # Format đúng ví dụ trong mô tả
        lines = [
            f"\n\n Type *{str(row[name_col]).lower()}* \n\n Có {row['count']} món: \n\n {row['foods']} "
            for _, row in grouped.iterrows()
        ]

        return "Menu hiện tại có các món:\n" + "\n".join(lines) if lines else "Không có dữ liệu."

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)

# exports all tools for agent
all_agent_tools = [
    HybridSearchTool(), 
    SearchMultiTypeCategoryTool(), 
    ColumnValueCountTool()
    ]