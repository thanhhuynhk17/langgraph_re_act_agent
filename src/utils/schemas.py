# from copilotkit import CopilotKitState
# from typing import Literal
# from langgraph.managed import IsLastStep, RemainingSteps
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal
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
