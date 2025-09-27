import os
import re
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type

import numpy as np
import pandas as pd
import pendulum
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.vectorstores import SQLiteVec
from langchain_core.tools import BaseTool, ToolException
from langchain_core.tools.base import ArgsSchema
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field, model_validator
# from utils.tools import init_vectorstore_faiss
from rank_bm25 import BM25Okapi

from react_agent.utils.crud_orders_db import (create_order, delete_order, get_order,
                                      update_order)
from react_agent.utils.react_constants import *
# from langchain_community.vectorstores import FAISS
from react_agent.utils.schemas import (ColumnValueCount, CustomerInfo,
                               DeleteOrderInput, Dish, FoodTypeAndNameInput,
                               HybridSearch, SearchMultiTypeCategory,
                               SearchTypeCategory, TakeOrderInput,
                               UpdateOrderInput)
from react_agent.utils.toolhelper import (  # get_model_qwen,
    get_openai_embedding_base_url, get_qwen_embedding_hf_endpoint,
    init_vectorstore, run_load_data_to_embedding, run_normalization_data)

load_dotenv()


# search_tool = TavilySearch()


# -------------------------
# Module-level cache
# -------------------------

# _MODEL = get_qwen_embedding_hf_endpoint("http://localhost:8080") # chính xác
# _MODEL = get_model_qwen() # chính xác
# _MODEL = get_openai_embedding_base_url() # chưa chính xác
_DATABASE = None
# _MODEL = get_model_qwen()
# _VEC_STORE = init_vectorstore(
#     _MODEL,
#     "./src/data",
#     connection=SQLiteVec.create_connection(db_file="./src/data/vec.db")
# )
# -------------------------
# Hybrid Search Tool
# -------------------------


# class HybridSearchInput(BaseTool):
#     name: str = "hybrid_search"
#     description: str = (
#         "Công cụ tìm kiếm kết hợp (hybrid Search) trên menu quán ăn, "
#         "sử dụng cả tìm kiếm ngữ nghĩa (embedding) và tìm kiếm theo từ khóa (BM25). "
#         "Thích hợp để trả lời các câu hỏi như: số lượng món ăn/đồ uống, "
#         "tên món, thành phần, giá, hoặc thông tin chi tiết trong thực đơn."
#     )
#     args_schema: Optional[ArgsSchema] = HybridSearch
#     return_direct: bool = True

#     _vector_store: Optional[SQLiteVec] = None  # cache nội bộ

#     def _get_vector_store(self) -> SQLiteVec:
#         if self._vector_store is None:
#             print("Initializing vector store...")
#             self._vector_store = init_vectorstore(
#                 _MODEL,
#                 "./src/data",
#                 connection=SQLiteVec.create_connection(
#                     db_file="./src/data/vec.db")
#             )
#         return self._vector_store

#     def _run(
#         self,
#         text_query: str,
#         k: int,
#         # run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:

#         # """Use the tool."""
#         # if self.model_search_embedding_hf is None:
#         #     self.model_search_embedding_hf = get_model_qwen(device='cuda:0') # oke
#         #     # model_search_embedding_hf = get_qwen_embedding_hf_endpoint() # oke

#         docs = run_load_data_to_embedding('./src/store/comque_new.csv')
#         docs = run_normalization_data(
#             docs, path_stopwords='./src/store/stopwords-vietnamese.txt')

#         tokenized_corpus = [doc.split(",") for doc in docs]
#         bm25 = BM25Okapi(tokenized_corpus)

#         # if not os.path.exists('./src/data/index.pkl'):
#         #     vt = init_vectorstore_faiss(_MODEL, db_folder=path_db_folder, action=2) # oke
#         # else:
#         #     vt = init_vectorstore_faiss(_MODEL, db_folder=path_db_folder, action=0) # oke
#         vector_store = self._get_vector_store()

#         results = vector_store.similarity_search(text_query.lower(), k=k)
#         vector_results = [doc.page_content for doc in results]

#         tokenized_query = text_query.split(" ")
#         result_index = list(bm25.get_top_n(tokenized_query, docs, n=k))
#         result_bm25 = [result_index[idx] for idx in range(k)]

#         header = ["Mã món ăn", "Phân loại", "Tên món ăn", "Mô tả ngắn", "Nguyên liệu", "Vị giác nổi bật",
#                   "Hương vị nổi bật", "Giá món ăn (VND)", "Khẩu phần ăn"
#                   ]
#         header_str = ", ".join(header)
#         combined_similarity = list(set(result_bm25 + vector_results))
#         vector_results_str = "\n".join(combined_similarity)

#         return "\n".join([header_str, vector_results_str])

#     async def _arun(self, *args, **kwargs) -> str:
#         """Async entry-point; re-use sync implementation."""
#         return self._run(*args, **kwargs)


# Take order
load_dotenv()

# search_tool = TavilySearch()


# -------------------------
# Singleton cache
# -------------------------
_DATABASE: Optional[pd.DataFrame] = None
# _MODEL: Optional[Any] = None
_VECTOR_STORE: Optional[object] = None
_BM25: Optional[BM25Okapi] = None


def get_database() -> pd.DataFrame:
    global _DATABASE
    print("Current working directory:", os.getcwd())

    if _DATABASE is None:
        csv_path =f"{os.getcwd()}\\react_agent\\store\\comque_new.csv"
        print("csv directory:", csv_path)
        _DATABASE = pd.read_csv(csv_path)
    return _DATABASE


# def get_model():
#     global _MODEL
#     if _MODEL is None:
#         _MODEL = get_openai_embedding_base_url()
#     return _MODEL


# def get_vector_store():
#     global _VECTOR_STORE, _MODEL
#     if _VECTOR_STORE is None:
#         _VECTOR_STORE = init_vectorstore(
#             get_model(),
#             "./src/data",
#             connection=SQLiteVec.create_connection(db_file='./src/data/vec.db')
#         )
#     return _VECTOR_STORE


# def get_bm25() -> BM25Okapi:
#     global _BM25
#     if _BM25 is None:
#         docs = get_database()["name_of_food"].fillna("").astype(str).tolist()
#         tokenized = [doc.split() for doc in docs]
#         _BM25 = BM25Okapi(tokenized)
#     return _BM25

# -------------------------
# Hybrid Search Tool
# -------------------------


# class HybridSearchTool(BaseTool):
#     name: str = "hybrid_search"
#     description: str = (
#         "Tìm kiếm món ăn kết hợp embedding + BM25. "
#         "Input: text_query (str), k (int). "
#         "Output: danh sách món ăn phù hợp nhất."
#     )
#     args_schema: type[BaseModel] = HybridSearch
#     return_direct: ClassVar[bool] = False  # để agent còn suy nghĩ

#     def _run(self, text_query: str, k: int) -> str:

#         if not text_query.strip():
#             return "Không có từ khóa tìm kiếm."

#         db = get_database()
#         bm25 = get_bm25()
#         vs = get_vector_store()

#         # BM25
#         tokenized_q = text_query.lower().split()
#         bm25_idx = bm25.get_top_n(
#             tokenized_q, db["name_of_food"].tolist(), n=k)

#         # Vector
#         vec_docs = vs.similarity_search(text_query.lower(), k=k)
#         vec_names = [d.metadata.get("name", "") for d in vec_docs]

#         # Gộp + uniq
#         merged = list(dict.fromkeys(bm25_idx + vec_names))[:k]

#         return "\n".join(merged) if merged else "Không tìm thấy món nào phù hợp."

#     async def _arun(self, *args, **kwargs):
#         return self._run(*args, **kwargs)


# The class-based tool
class TakeOrder(BaseTool):
    name: str = "take_order"
    description: str = (
        "Handles customer orders or reservations, including validation against the menu. "
        "Useful for processing dine-in or takeaway requests with dish details and notes. "
        "Input should include customer info, order type, time, dishes, and optional note."
    )
    args_schema: Type[BaseModel] = TakeOrderInput
    handle_tool_error: bool = True  # Like Tavily, enable error handling

    # Optional parameters (like Tavily's overrides)
    menu_df: pd.DataFrame = Field(default_factory=get_database)

    def _parse_price(self, raw: str) -> int:
        """Turn '145,000' -> 145000."""
        return int(raw.replace(",", ""))

    def _run(
        self,
        customer: CustomerInfo,
        is_takeaway: bool,
        booking_time: datetime,
        dishes: Optional[List[Dish]] = None,
        note: Optional[str] = None,
        run_manager: Optional[Any] = None,
    ) -> str:
        try:
            if dishes:
                for raw in dishes:
                    # <-- triggers your validator
                    dish = Dish.model_validate(raw)

            # 2. ---- force local TZ if naive ----
            if isinstance(booking_time, str):
                booking_time = pendulum.parse(booking_time)   # str -> datetime
            booking_time = DEFAULT_TZ.convert(booking_time)  # now safe to use

            # 3. ---- build dish list for CRUD ----
            dish_list = [
                {"id": d.id, "name": d.name_of_food, "quantity": d.quantity}
                for d in (dishes or [])
            ]

            # 4. ---- call CRUD layer ----
            if dishes:
                total_cost = float(
                    sum(
                        self._parse_price(self.menu_df.set_index(
                            "ID").loc[d.id, "current_price"]) * d.quantity
                        for d in dishes
                    )
                )
            else:
                total_cost = 0.0

            order_id, table_id = create_order(
                guest_name=customer.name,
                guest_phone_number=customer.phone,
                total_cost=total_cost,
                is_takeaway=is_takeaway,
                dishes=dish_list,
                booking_time=booking_time,
                notes=note,
            )

            # 5. ---- pretty answer ----
            dishes_str = ", ".join(
                [f"{d.name_of_food}×{d.quantity}" for d in dishes]) if dishes else "Chưa chọn"
            return (
                f"✅ Đặt bàn thành công cho {customer.name} ({customer.phone}).\n"
                f"Thời gian: {booking_time.strftime('%Y-%m-%d %H:%M %z')}\n"
                f"Loại: {'Mang đi' if is_takeaway else 'Ăn tại chỗ'}\n"
                f"Mã đơn: {order_id}  |  Bàn: {table_id or 'không có thông tin'}\n"
                f"Món: {dishes_str}\n"
                f"Ghi chú: {note or 'Không'}"
            )

        except Exception as e:
            raise ToolException(f"Order processing failed: {e}") from e

    # ------------------------------------------------------------------
    async def _arun(self, *args, **kwargs) -> str:
        """Async entry-point; re-use sync implementation."""
        return self._run(*args, **kwargs)


# -------------------------
# LangChain tool: update order
# -------------------------
class UpdateOrderTool(BaseTool):
    name: str = "update_order"              # ✅ use type annotation
    description: str = (
        "Update an existing order by its order_id. You can update total_cost or notes."
    )
    args_schema: Type[BaseModel] = UpdateOrderInput
    handle_tool_error: bool = True

    def _run(self, order_id: str, total_cost: Optional[float] = None, notes: Optional[str] = None) -> Dict[str, Any]:
        success = update_order(order_id, total_cost=total_cost, notes=notes)
        if not success:
            return {"success": False, "message": f"Order {order_id} not found or nothing updated."}
        return {"success": True, "order": get_order(order_id)}

    async def _arun(self, order_id: str, total_cost: Optional[float] = None, notes: Optional[str] = None) -> Dict[str, Any]:
        return self._run(order_id, total_cost, notes)


class DeleteOrderTool(BaseTool):
    name: str = "delete_order"              # ✅ type annotation
    description: str = "Delete an order by its order_id."
    args_schema: Type[BaseModel] = DeleteOrderInput
    handle_tool_error: bool = True

    def _run(self, order_id: str) -> Dict[str, Any]:
        success = delete_order(order_id)
        return {"success": success, "order_id": order_id}

    async def _arun(self, order_id: str) -> Dict[str, Any]:
        return self._run(order_id)

# -------------------------
# Category Search Tool
# -------------------------


class SearchTypeCategoryTool(BaseTool):
    name: str = "search_type_category"
    description: str = "Lọc món ăn theo loại và từ khóa tùy chọn"
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
        mask_type = df["type_of_food"].str.lower().eq(
            value_type_of_food.lower())

        # 2. Lọc keyword (nếu có)
        kw = key_value_option.strip()
        if kw:
            cols = df.columns.drop("number_of_people_eating")
            mask_kw = df[cols].apply(
                lambda col: col.astype(
                    str).str.lower().str.contains(kw, na=False)
            ).any(axis=1)
            mask_type &= mask_kw

        # 3. Lấy tên món & clean
        names = df.loc[mask_type, "name_of_food"].dropna().unique()
        cleaned = [re.sub(r"[^\w\s]", "", n).strip()
                   for n in names if n.strip()]
        return "\n".join(cleaned) if cleaned else "Không có món phù hợp."

    async def _arun(self, *args, **kwargs):
        return self._run(*args, **kwargs)

# -------------------------  SearchMultiTypeCategoryTool  -------------------------


class SearchMultiTypeCategoryTool(BaseTool):
    name: str = "search_multi_type_category"
    description: str = (
        "Tìm kiếm nhiều loại món ăn (categories) cùng lúc với từ khóa tương ứng 1-1. "
        "Trả về tên món duy nhất, cách nhau dấu phẩy."
    )
    args_schema: Type[BaseModel] = SearchMultiTypeCategory
    handle_tool_error: bool = True

    def _run(self, categories: List[str], keywords: List[str]) -> str:
        if len(categories) != len(keywords):
            return "Số lượng categories và keywords phải bằng nhau."

        df = get_database()
        out: List[str] = []

        for cat, kw in zip(categories, keywords):
            mask = df["type_of_food"].str.lower().eq(cat.lower())
            if kw.strip():
                cols = [c for c in df.columns if c !=
                        "number_of_people_eating"]
                mask &= df[cols].apply(
                    lambda col: col.astype(str).str.lower(
                    ) # .str.contains(kw.lower(), na=False)
                ).any(axis=1)

            rows = (
                df.loc[mask, ["ID","type_of_food","name_of_food","current_price","number_of_people_eating"]]
                .dropna(subset=["name_of_food"])   # only drop rows without a name
                .drop_duplicates()
            )

            # format each row to a string (customize format as needed)
            names = []
            rows.apply(lambda r: names.append(f"{int(r.ID)}: {r.name_of_food} — {r.current_price} VNĐ"), axis=1)
            out.extend(names)

        # uniq + clean
        header = "ID: Tên món — Giá tiền"
        cleaned = [re.sub(r"[^\w\s]", "", n).strip()
                for n in dict.fromkeys(out)]
        return "\n".join([header, *cleaned]) if cleaned else "Không tìm thấy món nào."

    async def _arun(self, *args, **kwargs):
        return self._run(*args, **kwargs)


class ColumnValueCountTool(BaseTool):
    name: str = "column_value_count"
    description: str = (
        "Đếm tần suất xuất hiện và liệt kê món ăn theo từng giá trị trong cột. "
        'VD: cột "type_of_food" → "món cá-12: cá kho, cá hấp"'
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

        # group nhanh & đếm
        grouped = (
            df.dropna(subset=[name_col])
            .groupby(name_col, sort=False)["name_of_food"]
            .apply(lambda x: ", ".join(dict.fromkeys(x.dropna())))
            .reset_index(name="foods")
            .assign(count=lambda d: d["foods"].str.count(",") + 1)
            .sort_values("count", ascending=False)
        )

        lines = [f"{str(row[name_col]).lower()}-{row['count']}: {row['foods']}"
                 for _, row in grouped.iterrows()]
        return "\n".join(lines) if lines else "Không có dữ liệu."

    async def _arun(self, *args, **kwargs):
        return self._run(*args, **kwargs)


class FoodTypeAndNameTool(BaseTool):
    name: str = "food_type_and_name"
    description: str = "Liệt kê các món theo loại: 'type-food_name'"
    args_schema: Type[BaseModel] = FoodTypeAndNameInput
    handle_tool_error: bool = True

    def _run(self) -> str:
        df = get_database()
        pairs = (
            df[["type_of_food", "name_of_food"]]
            .dropna()
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        return "\n".join(f"{t}-{n}" for t, n in pairs)

    async def _arun(self, *args, **kwargs):
        return self._run(*args, **kwargs)


# def warm_up():
#     """Chạy 1 lần khi worker khởi động"""
#     get_database()
#     get_bm25()
#     get_vector_store()


# exports all tools for agent
all_agent_tools = [
    # HybridSearchTool(),
    TakeOrder(),
    # UpdateOrderTool(),
    DeleteOrderTool(),
    # SearchTypeCategoryTool(),
    SearchMultiTypeCategoryTool(),
    # ColumnValueCountTool()
]
