from copilotkit import CopilotKitState
from typing import Literal
from langgraph.managed import IsLastStep, RemainingSteps
from typing import Any, Dict
import json

# FIXME: remove CopilotKitState
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

