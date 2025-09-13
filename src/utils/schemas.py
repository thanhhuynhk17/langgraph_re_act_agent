# from copilotkit import CopilotKitState
# from typing import Literal
# from langgraph.managed import IsLastStep, RemainingSteps
from pydantic import BaseModel, Field, model_validator
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
    
class HybridSearchInput(BaseModel):
    """
    Input schema for the hybrid_search tool
    Combines keyword and semantic search for more accurate retrieval
    """
    query: str = Field(
        ...,
        description="The search query text. Can be natural language or keywords"
    )
    k: int = Field(
        ...,
        description="The number of top results to return. Must be either 5, 10, 50, 100 or more than"
    )

class SearchTypeCategoryAndPeople(BaseModel):
    """
    Input schema for the search_type_category_and_people tool.
    Finds rows in the food database that satisfy three conditions:
    1. The 'type_of_food' column matches the specified category.
    2. At least one of the descriptive columns contains the keyword:
        - 'name_of_food'
        - 'how_to_prepare'
        - 'main_ingredients'
        - 'taste'
        - 'outstanding_fragrance'
        - 'current_price'
    """

    value_type_of_food: Literal[
        "món cá",
        "món khai vị",
        "món ăn chơi",
        "món rau",
        "món gỏi",
        "món gà, vịt & trứng",
        "món tôm & mực",
        "món xào",
        "nước mát nhà làm",
        "lẩu",
        "món thịt",
        "món sườn & đậu hũ",
        "món canh",
        "các loại khô",
        "tráng miệng"
    ] = Field(
        ...,
        description="Food category filter. Example: 'món cá', 'món khai vị'."
    )

    value_option: str = Field(
        ...,
        description="Keyword to search across descriptive columns. Example: 'cay', 'mặn', 'ngọt'."
    )
    
    
class SearchValuesInTypeInput(BaseModel):
    """
    Input schema for the search_values_in_type tool.
    Counts occurrences of unique values in a specified column of the food database.
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
        description=(
            "The name of the column to analyze. Must be one of:"
            "- 'type_of_food' (food category)"
            "- 'name_of_food' (name of the dish)"
            "- 'how_to_prepare' (preparation method)"
            "- 'main_ingredients' (main ingredients)"
            "- 'taste' (flavor profile)"
            "- 'outstanding_fragrance' (distinct fragrance)"
            "- 'current_price' (current price)"
            "- 'number_of_people_eating' (number of people suitable for the dish)"
        )
    )


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

