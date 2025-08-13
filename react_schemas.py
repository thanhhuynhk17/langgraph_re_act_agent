from typing import List, Dict, Optional, Literal, Type
from pydantic import BaseModel, Field
from pydantic import model_validator

def make_react_step_schema(tool_names: List[str]) -> Type[BaseModel]:
    """
    Dynamically creates a ReActStep Pydantic v2 model
    with 'action' constrained to the given tool names.
    """

    # Dynamically generate Literal for tool names
    ActionLiteral = Literal[tuple(tool_names)]

    class ReActStep(BaseModel):
        """A single ReAct reasoning step with optional action."""

        thought: str = Field(
            ...,
            description="The agent's reasoning at this step"
        )

        action: Optional[ActionLiteral] = Field(  # type: ignore
            None,
            description=f"Optional: name of the tool to use, must be one of {tool_names}"
        )

        action_input: Optional[Dict] = Field(
            None,
            description="Optional: input to the selected tool"
        )

        # @model_validator(mode="after")
        # def validate_action_input(self) -> "ReActStep":
        #     if self.action and not self.action_input:
        #         raise ValueError("If 'action' is provided, 'action_input' must also be provided.")
        #     return self

    return ReActStep
