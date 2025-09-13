import os
from typing import Literal, Dict
from langchain.tools import tool
from src.utils.schemas import HybridSearchInput, SearchTypeCategoryAndPeople, SearchValuesInTypeInput, menu_df
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
    
    header = ["Mã món ăn", "Phân loại", "Tên món ăn", "Mô tả ngắn",
        "Nguyên liệu", "Vị cay / chua / mặn / ngọt", 
        "Hương vị nổi bật", "Giá món ăn (VNĐ)", "Khẩu phần ăn"
    ]
    header_str = ",".join(header)
    vector_results_str = "\n".join(vector_results)
    # combined_results = [
    #     {"_id": f"{v.split()[0]}", "_content": " ".join(v.split()[1:])}
    #     for v in vector_results
    # ]
    # return "\n".join(
    #     [f"{r.get('_id', 'N/A')} - {r.get('_content', '')}" for r in combined_results]
    # )
    return "\n".join([header_str,vector_results_str]), results
    

# Take order
from src.utils.schemas import Dish, CustomerInfo, TakeOrderInput, UpdateOrderInput, DeleteOrderInput
from typing import Any, List, Optional, Type
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
from src.utils.react_constants import DEFAULT_TZ
from src.utils.crud_orders_db import create_order, update_order, delete_order, get_order
import pendulum
import hashlib  # For hashing phone to generate table_id

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
    menu_df: pd.DataFrame = Field(menu_df)
    """Menu DataFrame (can be overridden if needed)."""

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
all_agent_tools = [hybrid_search, TakeOrder(), UpdateOrderTool(), DeleteOrderTool(), search_type_category_and_people, search_values_in_type]
