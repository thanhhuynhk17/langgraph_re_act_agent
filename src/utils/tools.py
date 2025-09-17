import os
from typing import Literal, Dict, Optional
from langchain.tools import tool
from src.utils.schemas import HybridSearch, SearchTypeCategory, SearchMultiTypeCategory, ColumnValueCount, FoodTypeAndNameInput
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
# from utils.tools import init_vectorstore_faiss
from rank_bm25 import BM25Okapi
from langchain_core.tools.base import ArgsSchema
from langchain_core.tools import BaseTool
from typing import ClassVar
from src.utils.toolhelper import run_load_data_to_embedding, run_normalization_data, get_model_qwen, init_vectorstore, get_qwen_embedding_hf_endpoint
import os

load_dotenv()

from langchain_tavily import TavilySearch
search_tool = TavilySearch()


# -------------------------
# Module-level cache
# -------------------------

# _MODEL = get_qwen_embedding_hf_endpoint("http://localhost:8080") # chính xác
# _MODEL = get_model_qwen() # chính xác
# _MODEL = get_openai_embedding_base_url() # chưa chính xác
_DATABASE = None
_MODEL = get_model_qwen()
_VEC_STORE = init_vectorstore(
    _MODEL, 
    "./src/data", 
    connection = SQLiteVec.create_connection(db_file="./src/data/vec.db")
    )
# -------------------------
# Hybrid Search Tool
# -------------------------
class HybridSearchInput(BaseTool):
    name: str = "hybrid_search"
    description: str = (
        "Công cụ tìm kiếm kết hợp (hybrid Search) trên menu quán ăn, "
        "sử dụng cả tìm kiếm ngữ nghĩa (embedding) và tìm kiếm theo từ khóa (BM25). "
        "Thích hợp để trả lời các câu hỏi như: số lượng món ăn/đồ uống, "
        "tên món, thành phần, giá, hoặc thông tin chi tiết trong thực đơn."
    )
    args_schema: Optional[ArgsSchema] = HybridSearch
    return_direct : bool = True
    
    _vector_store: Optional[SQLiteVec] = None  # cache nội bộ

    def _get_vector_store(self) -> SQLiteVec:
        if self._vector_store is None:
            print("Initializing vector store...")
            self._vector_store = init_vectorstore(
                    _MODEL, 
                    "./src/data", 
                    connection = SQLiteVec.create_connection(db_file="./src/data/vec.db")
                )
        return self._vector_store
    
    def _run(
            self, 
            text_query: str,
            k: int,
            # run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
        
        # """Use the tool."""
        # if self.model_search_embedding_hf is None:
        #     self.model_search_embedding_hf = get_model_qwen(device='cuda:0') # oke
        #     # model_search_embedding_hf = get_qwen_embedding_hf_endpoint() # oke

        docs = run_load_data_to_embedding('./src/store/comque_new.csv')
        docs = run_normalization_data(docs, path_stopwords='./src/store/stopwords-vietnamese.txt')
                
        tokenized_corpus = [doc.split(",") for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # if not os.path.exists('./src/data/index.pkl'):
        #     vt = init_vectorstore_faiss(_MODEL, db_folder=path_db_folder, action=2) # oke
        # else:
        #     vt = init_vectorstore_faiss(_MODEL, db_folder=path_db_folder, action=0) # oke
        vector_store = self._get_vector_store()
        
        results = vector_store.similarity_search(text_query.lower(), k=k)
        vector_results = [doc.page_content for doc in results]
        
        tokenized_query = text_query.split(" ")
        result_index = list(bm25.get_top_n(tokenized_query, docs, n=k))
        result_bm25 = [result_index[idx] for idx in range(k)]
    
        header = ["Mã món ăn", "Phân loại", "Tên món ăn", "Mô tả ngắn", "Nguyên liệu", "Vị giác nổi bật", 
            "Hương vị nổi bật", "Giá món ăn (VND)", "Khẩu phần ăn"
        ]
        header_str = ", ".join(header)
        combined_similarity = list(set(result_bm25 + vector_results))
        vector_results_str = "\n".join(combined_similarity)

        return "\n".join([header_str,vector_results_str])
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async entry-point; re-use sync implementation."""
        return self._run(*args, **kwargs)

# Take order
from src.utils.schemas import Dish, CustomerInfo, TakeOrderInput, UpdateOrderInput, DeleteOrderInput
from typing import Any, List, Optional, Type
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
import pandas as pd
from src.utils.react_constants import DEFAULT_TZ
from src.utils.crud_orders_db import create_order, update_order, delete_order, get_order
import pendulum
import hashlib  # For hashing phone to generate table_id

def get_database() -> pd.DataFrame:
    global _DATABASE
    if _DATABASE is None:
        _DATABASE = pd.read_csv('./src/store/comque_new.csv')
    return _DATABASE

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
                    dish = Dish.model_validate(raw)   # <-- triggers your validator

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
                        self._parse_price(self.menu_df.set_index("ID").loc[d.id, "current_price"]) * d.quantity
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
            dishes_str = ", ".join([f"{d.name_of_food}×{d.quantity}" for d in dishes]) if dishes else "Chưa chọn"
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
    description: str = (
        "Tìm kiếm món ăn trong cơ sở dữ liệu dựa trên loại món (category) và từ khóa liên quan. "
        "Công cụ lọc theo nhóm món (ví dụ: món cá, món khai vị, lẩu, tráng miệng, v.v.) "
        "và kết hợp tìm kiếm từ khóa trong các cột khác (tên, nguyên liệu, mô tả, v.v.), "
        "trả về danh sách các món phù hợp với nhu cầu của khách hàng."
    )
    args_schema: Type[BaseModel] = SearchTypeCategory
    handle_tool_error: bool = True
    _database = None

    def _get_database(self) -> pd.DataFrame:
        if self._database is None:
            self._database = pd.read_csv("./src/store/comque_new.csv")
        return self._database
    
    def _run(self, 
            value_type_of_food: Literal["món cá", "món khai vị", "món ăn chơi", "món rau", "món gỏi", "món gà, vịt & trứng", "món tôm & mực", "món xào", "nước mát nhà làm", "lẩu", "món thịt", "món sườn & đậu hũ", "món canh", "các loại khô", "tráng miệng"], 
            key_value_option: str
        ) -> list:
        
        
        _database = get_database()

        value_option = key_value_option.lower()

        # 1️⃣ fixed filter on type_of_food
        mask_type = _database["type_of_food"].str.lower().str.contains(value_type_of_food.lower(), na=False)

        # 2️⃣ search across other columns (exclude number_of_people_eating)
        search_cols = _database.columns.drop("number_of_people_eating")
        mask_text = _database[search_cols].apply(
            lambda col: col.astype(str).str.lower().str.contains(value_option, na=False)
        )
        mask_any = mask_text.any(axis=1)

        # 4️⃣ combine
        result = _database[mask_type & mask_any].reset_index(drop=True)
        print()
        print()
        print(result)
        print()
        print()
        return result.values.tolist()
    
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)

class SearchMultiTypeCategoryTool(BaseTool):
    name: str = "search_multi_type_category"
    description: str = (
        "Tìm kiếm nhiều loại món (categories) cùng lúc với từ khóa 1-1. "
        "Ví dụ: 'cho 1 món mặn và 1 món canh' → "
        "value_types_of_food=['món thịt','món canh'], key_value_options=['mặn','chua']."
    )
    args_schema: Type[BaseModel] = SearchMultiTypeCategory
    handle_tool_error: bool = True

    _database: Optional[pd.DataFrame] = None

    def _get_database(self) -> pd.DataFrame:
        if self._database is None:
            self._database = pd.read_csv("./src/store/comque_new.csv")
        return self._database

    def _filter_category(self, db: pd.DataFrame, category: str, keyword: str) -> List[dict]:
        mask_type = db["type_of_food"].astype(str).str.lower().str.contains(category.lower(), na=False)

        cols = list(db.columns)
        if "number_of_people_eating" in cols:
            cols.remove("number_of_people_eating")

        mask_text = db[cols].apply(
            lambda col: col.astype(str).str.lower().str.contains(keyword.lower(), na=False)
        )
        mask_any = mask_text.any(axis=1)

        filtered = db[mask_type & mask_any].reset_index(drop=True)
        return filtered.to_dict(orient="records")

    def _run(self, value_types_of_food: List[str], key_value_options: List[str]) -> dict:
        db = self._get_database()
        results = {}

        for category, keyword in zip(value_types_of_food, key_value_options):
            results[category] = self._filter_category(db, category, keyword)

        return results

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)

class ColumnValueCountTool(BaseTool):
    """
    Công cụ phân tích tần suất giá trị trong cơ sở dữ liệu món ăn.

    Đọc dữ liệu từ file CSV (comque_new.csv), sau đó đếm số lần xuất hiện 
    của từng giá trị trong một cột cụ thể (ví dụ: 'type_of_food', 'taste', 'outstanding_fragrance'). 

    Kết quả trả về dưới dạng danh sách [value, count, foods]:
    - value: giá trị duy nhất trong cột (chuyển thành lowercase).
    - count: số lần giá trị đó xuất hiện trong dữ liệu.
    - foods: danh sách các món ăn (theo cột 'name_of_food') có giá trị đó.

    Ví dụ:
        Nếu chọn cột "type_of_food", công cụ có thể trả về:
        [
            ["món cá", 12, ["cá kho tộ", "cá hấp gừng", "cá chiên xù", ...]],
            ["món canh", 8, ["canh chua cá lóc", "canh rau đay", ...]]
        ]
    """

    name: str = "column_value_count"
    description: str = (
        "Phân tích dữ liệu menu để đếm số lần xuất hiện của các giá trị "
        "trong một cột được chọn. Ngoài số lượng, công cụ còn liệt kê "
        "danh sách các món ăn tương ứng với mỗi giá trị."
    )
    args_schema: Type[BaseModel] = ColumnValueCount
    handle_tool_error: bool = True

    _database: Optional[pd.DataFrame] = None

    def _get_database(self) -> pd.DataFrame:
        if self._database is None:
            self._database = pd.read_csv("./src/store/comque_new.csv")
        return self._database

    def _run(self, name_col: str) -> List[tuple]:
        """
        Trả về danh sách (value, count, foods) với:
        - value: giá trị trong cột (lowercase)
        - count: số lần xuất hiện (int)
        - foods: list các tên món ('name_of_food') tương ứng
        """
        db = self._get_database()

        if name_col not in db.columns:
            raise ValueError(f"Invalid column '{name_col}'. Available columns: {', '.join(db.columns)}")

        # loại bỏ hàng thiếu giá trị ở cột cần group
        df = db.dropna(subset=[name_col])

        # grouped: dùng named aggregation để tránh trùng tên cột khi reset_index()
        grouped = (
            df.groupby(name_col)
            .agg(foods=('name_of_food', lambda s: s.tolist()))
            .reset_index()
        )

        # đếm số phần tử trong list foods
        grouped['count'] = grouped['foods'].str.len()

        # sắp xếp theo count giảm dần để tiện (tuỳ chọn)
        grouped = grouped.sort_values('count', ascending=False).reset_index(drop=True)

        results = []
        for _, row in grouped.iterrows():
            value = str(row[name_col]).lower()
            count = int(row['count'])
            foods = " ".join(name for name in row['foods'])  # list of strings
            results.append((value, count, foods))

        print()
        print()
        print(results)
        print()
        print()
        
        return results


    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)

class FoodTypeAndNameTool(BaseTool):
    """
    Tool to extract pairs of [type_of_food, name_of_food] from the food database.
    Useful when we want to know which dishes belong to which categories.
    """

    name: str = "food_type_and_name"
    description: str = (
        "Trích xuất danh sách gồm 2 cột: loại món ăn (type_of_food) và tên món ăn (name_of_food) "
        "từ cơ sở dữ liệu. Dùng để liệt kê các món theo loại. Khi khách hỏi menu hay các món ăn chính/nổi tiếng của quán."
        "Liệt kê tên một số món đắt nhất của mỗi loại"
    )
    args_schema: Type[BaseModel] = FoodTypeAndNameInput
    handle_tool_error: bool = True

    _database: Optional[pd.DataFrame] = None

    def _get_database(self) -> pd.DataFrame:
        if self._database is None:
            self._database = pd.read_csv("./src/store/comque_new.csv")
        return self._database

    def _run(self) -> List[List[str]]:
        db = self._get_database()
        result = db[["type_of_food", "name_of_food"]].dropna().reset_index(drop=True)
        return result.values.tolist()

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return self._run()

# exports all tools for agent
all_agent_tools = [HybridSearchInput(), TakeOrder(), UpdateOrderTool(), DeleteOrderTool(), SearchTypeCategoryTool(), SearchMultiTypeCategoryTool(), ColumnValueCountTool(), FoodTypeAndNameTool()]
