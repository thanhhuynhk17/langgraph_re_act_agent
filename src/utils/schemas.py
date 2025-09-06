from copilotkit import CopilotKitState
from typing import Literal
from langgraph.managed import IsLastStep, RemainingSteps
from pydantic import BaseModel, Field
from typing import Any, Dict
import json

# FIXME: remove CopilotKitState
class AgentState(CopilotKitState):
    language: Literal["english", "vietnamese"] = "vietnamese" # type: ignore

    is_last_step: IsLastStep

    remaining_steps: RemainingSteps

    json_data: Dict[str, Any]

    # @field_validator("json_data", mode="before")
    # def parse_json_string(cls, v):
    #     if isinstance(v, str):
    #         return json.loads(v)
    #     return v
    
class HybridSearchInput(BaseModel):
    """
    Input schema for the hybrid_search tool.
    Combines keyword and semantic search for more accurate retrieval.
    """
    query: str = Field(
        ...,
        description="The search query text. Can be natural language or keywords."
    )
    k: Literal[10, 20] = Field(
        10,
        description="The number of top results to return. Must be either 10 or 20."
    )

