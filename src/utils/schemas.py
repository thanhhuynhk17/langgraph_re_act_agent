from langgraph.managed import IsLastStep, RemainingSteps
from pydantic import BaseModel, Field, model_validator, field_validator
from langchain_community.vectorstores import SQLiteVec
from typing import Any, Dict, Literal
from datetime import datetime
import pandas as pd
from typing import List, Optional
from src.utils.react_constants import DEFAULT_TZ
import pendulum
from langgraph.graph import MessagesState

class CustomAgentState(MessagesState):
    remaining_steps: RemainingSteps
    is_chitchat: bool
    is_speed: bool

# ------------- SEARCH -------------
class HybridSearch(BaseModel):
    text_query: str = Field(..., min_length=1, description="Từ khóa món ăn, ví dụ: 'cá kho'")
    k: int = Field(5, ge=1, le=20, description="Số món trả về (max 20)")

VALID_TYPES = [
    "món cá", "món khai vị", "món ăn chơi", "món rau", "món gỏi",
    "món gà, vịt & trứng", "món tôm & mực", "món xào", "nước mát nhà làm",
    "lẩu", "món thịt", "món sườn & đậu hũ", "món canh", "các loại khô", "tráng miệng"
]

class SearchTypeCategory(BaseModel):
    category: Literal[tuple(VALID_TYPES)] = Field(..., description="Loại món ăn")
    keyword: str = Field("", description="Từ khóa thêm (VD: 'cay', 'mặn')")

class SearchMultiTypeCategory(BaseModel):
    categories: List[str] = Field(..., description="Danh sách loại món (VD: ['món thịt', 'món canh'])")
    keywords: List[str] = Field(..., description="Từ khóa tương ứng (VD: ['mặn', 'chua'])")

    @field_validator("categories", "keywords")
    def not_empty(cls, v):
        if not v:
            raise ValueError("Không được để trống")
        return v

    @model_validator(mode="after")
    def same_len(self):
        if len(self.categories) != len(self.keywords):
            raise ValueError("Số lượng categories và keywords phải bằng nhau")
        return self
    
## Take order
menu_df = pd.read_csv("./src/store/comque_new.csv", encoding="utf-8")
menu_ids = menu_df["ID"].to_list()
menu_names = menu_df["name_of_food"].to_list()
menu_dict = dict(zip(menu_ids, menu_names))
menu_desc = "\n".join([f"{k}: {v}" for k, v in menu_dict.items()])
id_field_desc = f"Unique identifier of the dish. Must be one of the menu options (id: name):\n{menu_df.to_dict()}"
name_field_desc = f"Readable name of the dish. Must be one of the menu options (format id: name):\n{menu_df.to_dict()}"
# Load menu
# FIXME: move into Tool class

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
            menu_df = pd.read_csv("./src/store/comque_new.csv", encoding="utf-8")
            invalid = [
                f"Món id={d.id}: {d.name_of_food}"
                for d in self.dishes if d.id not in menu_df["ID"].values
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
# For routing incoming message
# -------------------------
class ChitChatCheck(BaseModel):
    is_chitchat: bool = Field(
        ...,
        description="True if the user message is casual chit-chat (greetings, small talk, thanks, etc.), "
                    "False if the user message is about restaurant info (menu, hours, ordering, etc.)."
    )


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
