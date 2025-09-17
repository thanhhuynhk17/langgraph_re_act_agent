# from copilotkit import CopilotKitState
# from typing import Literal
# from langgraph.managed import IsLastStep, RemainingSteps
from pydantic import BaseModel, Field, model_validator
# from langchain_core.callbacks import (
#     AsyncCallbackManagerForToolRun,
#     CallbackManagerForToolRun,
# )
from langchain_community.vectorstores import SQLiteVec
from typing import Any, Dict, Literal
from datetime import datetime
import pandas as pd
from typing import List, Optional
from src.utils.react_constants import DEFAULT_TZ
import pendulum

# import json

# # FIXME: remove CopilotKitState
# class AgentState(CopilotKitState):
#     language: Literal["english", "vietnamese"] = "vietnamese"

#     is_last_step: IsLastStep

#     remaining_steps: RemainingSteps

#     json_data: Dict[str, Any]

    # @field_validator("json_data", mode="before")
    # def parse_json_string(cls, v):
    #     if isinstance(v, str):
    #         return json.loads(v)
    #     return v
class HybridSearch(BaseModel):
    """
    Input schema for the HybridSearch tool.

    This schema defines the input parameters required to perform a hybrid search 
    (vector-based + BM25 keyword search). It ensures that:
    1. The customer provides a non-empty query string (`text_query`).
    2. The number of results `k` is a positive integer, within a reasonable limit.
    """

    text_query: str = Field(
        ...,
        description="Customer query string used to search across the food database. Example: 'món cá cay'."
    )
    k: Literal[10, 20, 50] = Field(
        ...,
        description="Number of top results to return from the search. Must be >= 10 and recommended <= 50."
    )

    @model_validator(mode="before")
    def validate_inputs(cls, values):
        errors = []

        # validate text_query
        query = values.get("text_query", "").strip()
        if not query:
            errors.append("text_query cannot be empty. Example: 'món cá nướng'.")

        # validate k
        k = values.get("k", 0)
        if not isinstance(k, int) or k <= 0:
            errors.append("k must be a positive integer. Example: 5.")
        elif k > 50:
            errors.append("k is too large. Please choose a value ≤ 50 for performance reasons.")

        if errors:
            raise ValueError("\n".join(errors))

        return values

# FIXME: deduplicate allowed options
VALID_TYPES = [
    "món cá", "món khai vị", "món ăn chơi", "món rau", "món gỏi",
    "món gà, vịt & trứng", "món tôm & mực", "món xào", "nước mát nhà làm",
    "lẩu", "món thịt", "món sườn & đậu hũ", "món canh", "các loại khô", "tráng miệng"
]

class SearchTypeCategory(BaseModel):
    """
    Input schema for the search_type_category tool.
    Finds rows in the food database that satisfy two conditions:
    1. The 'type_of_food' column matches the specified category.
    2. At least one descriptive column contains the keyword.
    """
    
    value_type_of_food: Literal[
        "món cá", "món khai vị", "món ăn chơi", "món rau", "món gỏi",
        "món gà, vịt & trứng", "món tôm & mực", "món xào", "nước mát nhà làm",
        "lẩu", "món thịt", "món sườn & đậu hũ", "món canh", "các loại khô", "tráng miệng"
    ] = Field(
        ...,
        description="Food category filter. Example: 'món cá', 'món khai vị'."
    )

    key_value_option: str = Field(
        ...,
        description="Keyword to search across descriptive columns. Example: 'cay', 'mặn', 'ngọt'."
    )

    @model_validator(mode="before")
    def provide_guidance(cls, values):
        errors = []

        # values could be {} if input is empty
        if "value_type_of_food" not in values:
            errors.append(
                "value_type_of_food is missing. Possible values are:\n" +
                ",".join(f'"{t}"' for t in VALID_TYPES)
            )
        if "key_value_option" not in values or not values.get("key_value_option", "").strip():
            errors.append("key_value_option is missing. Please provide a keyword to search. Example: 'cay', 'mặn', 'ngọt'.")

        if errors:
            raise ValueError("\n".join(errors))

        return values

class SearchMultiTypeCategory(BaseModel):
    """
    Schema cho tool search_multi_type_category.
    Cho phép tìm kiếm đồng thời nhiều category với từ khóa tương ứng (1-1).
    """

    value_types_of_food: List[Literal[
        "món cá", "món khai vị", "món ăn chơi", "món rau", "món gỏi",
        "món gà, vịt & trứng", "món tôm & mực", "món xào", "nước mát nhà làm",
        "lẩu", "món thịt", "món sườn & đậu hũ", "món canh", "các loại khô", "tráng miệng"
    ]] = Field(
        ..., description="Danh sách category. Ví dụ: ['món canh', 'món cá']"
    )

    key_value_options: List[str] = Field(
        ..., description="Danh sách keyword, tương ứng với từng category. "
                         "Ví dụ: ['chua', 'cá hú'] khi value_types_of_food=['món canh','món cá']"
    )

    @model_validator(mode="before")
    def validate_alignment(cls, values):
        types = values.get("value_types_of_food", [])
        keys = values.get("key_value_options", [])
        if not types:
            raise ValueError(
                "value_types_of_food is missing. Possible values are: "
                + ", ".join(VALID_TYPES)
            )
        if not keys:
            raise ValueError(
                "key_value_options is missing. Provide at least one keyword per category."
            )
        if len(types) != len(keys):
            raise ValueError(
                f"Length mismatch: {len(types)} categories but {len(keys)} keywords provided. "
                "Both lists must have the same length."
            )
        return values
    
## Take order

# Load menu
# FIXME: move into Tool class

menu_df = pd.read_csv("./src/store/comque_new.csv", encoding="utf-8")
menu_ids = menu_df["ID"].to_list()
menu_names = menu_df["name_of_food"].to_list()
menu_dict = dict(zip(menu_ids, menu_names))
menu_desc = "\n".join([f"{k}: {v}" for k, v in menu_dict.items()])
id_field_desc = f"Unique identifier of the dish. Must be one of the menu options (id: name):\n{menu_desc}"
name_field_desc = f"Readable name of the dish. Must be one of the menu options (format id: name):\n{menu_desc}"

class Dish(BaseModel):
    id: int = Field(..., description=id_field_desc)
    name_of_food: str = Field(..., description=name_field_desc)
    quantity: int = Field(default=1, ge=1, description="Quantity of the dish (default=1, must be >=1).")

    @model_validator(mode="after")
    def validate_fields(cls, dish):
        errors = []

        if dish.id not in menu_ids:
            errors.append(f"ID '{dish.id}' does not exist in the menu. Please find the **closest matching valid dish** from the menu to replace it.")

        if dish.name_of_food not in menu_names:
            errors.append(f"Dish name '{dish.name_of_food}' is not in the menu. Please find the **closest matching valid dish** from the menu to replace it.")

        if errors:
            raise ValueError("\n".join(errors))
        return dish

class CustomerInfo(BaseModel):
    name: str = Field(..., description="Full name of the customer.")
    phone: str = Field(..., description="Customer's phone number.")
    num_people: int = Field(..., gt=0, description="Number of people (must be >0).")

# Input schema for the tool
class TakeOrderInput(BaseModel):
    """
    Represents a complete order placed by a customer.
    Includes customer info, order details, and optional notes.
    """
    customer: CustomerInfo = Field(..., description="Customer information.")
    is_takeaway: bool = Field(..., description="Whether the order is for takeaway (True) or dine-in (False).")
    booking_time: datetime = Field(..., description="Time the customer wants the booking or pickup.")
    dishes: Optional[List[Dish]] = Field(None, description="List of ordered dishes. Can be empty for reservations.")
    note: Optional[str] = Field(None, description="Optional note from the customer (e.g., no chili, extra soup).")

    @model_validator(mode="after")
    def validate_order(self):
        # 1.  Convert to local TZ (idempotent if already aware)
        self.booking_time = DEFAULT_TZ.convert(self.booking_time)

        # 2.  Past check
        if self.booking_time < DEFAULT_TZ.convert(pendulum.now()):
            raise ValueError("Thời gian đặt bàn không hợp lệ, phải ở hiện tại hoặc tương lai.")

        # 3.  Menu check
        if self.dishes:
            invalid = [
                f"Món id={d.id}: {d.name_of_food}"
                for d in self.dishes
                if d.id not in menu_df["ID"].values
            ]
            if invalid:
                raise ValueError("Danh sách món không tìm thấy trong cơ sở dữ liệu:\n" + "\n".join(invalid))

        return self
    
## UPDATE & DELETE

# -------------------------
# UpdateOrderInput
# -------------------------
class UpdateOrderInput(BaseModel):
    order_id: str = Field(..., description="The short order ID to update.")
    total_cost: Optional[float] = Field(None, description="Updated total cost of the order.")
    notes: Optional[str] = Field(None, description="Updated notes for the order.")


# -------------------------
# DeleteOrderInput
# -------------------------
class DeleteOrderInput(BaseModel):
    order_id: str = Field(..., description="The short order ID to delete.")



# -------------------------
# Column Value Count
# -------------------------
class ColumnValueCount(BaseModel):
    """
    Schema input cho ColumnValueCount.
    Người dùng chỉ định tên cột muốn đếm tần suất giá trị.
    """

    name_col: Literal[
        "type_of_food",
        "name_of_food",
        "how_to_prepare",
        "main_ingredients",
        "taste",
        "outstanding_fragrance",
        "current_price",
        "number_of_people_eating"
    ] = Field(
        ...,
        description="Tên cột để phân tích và đếm giá trị. "
                    "Ví dụ: 'type_of_food', 'taste'."
    )
    
class FoodTypeAndNameInput(BaseModel):
    """
    Input schema for FoodTypeAndNameTool.
    No arguments are needed, the tool will always return
    the two columns 'type_of_food' and 'name_of_food'.
    """
    pass