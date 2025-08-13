from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated
# --- Tool Input Schemas ---

class ComputeLocalMoransIInput(BaseModel):
    """Input schema for compute_local_morans_I tool."""
    filepath: str = Field(".", description="Path to the spatial dataset (e.g., shapefile or GeoJSON)")
    var_name: str = Field(..., description="Variable to compute Local Moran's I on")
    # state: Annotated[dict, InjectedState]
# --- Tool Output Schemas (optional) ---

class ComputeLocalMoransIOutput(BaseModel):
    """Output schema for Local Moran's I results."""
    values: List[float] = Field(..., description="Local Moran's I values")
    pvalues: List[float] = Field(..., description="P-values for each observation")
    clusters: List[int] = Field(..., description="Cluster category index per observation")
    labels: List[str] = Field(..., description="Label names for cluster types")
    colors: List[str] = Field(..., description="Color hex codes for cluster types")

# --- (Optional) ReAct Agent Step Schema Generator ---
# def make_react_step_schema(tool_names: List[str]) -> Type[BaseModel]:
#     ...
from copilotkit import CopilotKitState
from typing import Literal
from langgraph.managed import IsLastStep, RemainingSteps
import json

class AgentState(CopilotKitState):
    language: Literal["english", "vietnamese"] = "vietnamese"

    is_last_step: IsLastStep

    remaining_steps: RemainingSteps

    json_data: Dict[str, Any]

    # @field_validator("json_data", mode="before")
    # def parse_json_string(cls, v):
    #     if isinstance(v, str):
    #         return json.loads(v)
    #     return v

