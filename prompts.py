PROMPT_REACT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Do not attempt to perform any task yourself.
Always reason step by step and call tools explicitly where needed.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

from pydantic import BaseModel
def generate_tool_prompt(name_for_model: str, 
                        name_for_human: str, 
                        description_for_model: str, 
                        schema: type[BaseModel]) -> str:
    # Get the JSON Schema (Pydantic v2)
    schema_dict = schema.model_json_schema()

    # Extract and format parameter descriptions
    param_descriptions = []
    properties = schema_dict.get("properties", None)
    if properties:
        for field_name, field_info in properties.items():
            desc = field_info.get("description", "No description")
            type_ = field_info.get("type", "unknown")
            required = field_name in schema_dict.get("required", [])
            required_str = "required" if required else "optional"
            param_descriptions.append(f"- `{field_name}` ({type_}, {required_str}): {desc}")
    else: # has no properties
        param_descriptions = ["- This tool takes no parameters."]
    parameters = "\n".join(param_descriptions)

    # Final prompt
    prompt = (
        f"{name_for_model}: Call this tool to interact with the {name_for_human} API.\n"
        f"What is the {name_for_human} API useful for? {description_for_model}\n"
        f"Parameters:\n{parameters}\n\n"
    )

    return prompt


from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from typing import List, Union
import re

def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from the text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def format_react_messages(messages: List[Union[HumanMessage, AIMessage, ToolMessage]]) -> str:
    output_lines = []
    current_tool_calls = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            # output_lines.append(f"Question: {msg.content.strip()}")
            continue

        elif isinstance(msg, AIMessage):
            # Remove <think> blocks
            thought = remove_think_tags(msg.content)
            if msg.tool_calls:
                # AI is calling tools (Action phase)
                current_tool_calls = msg.tool_calls
                output_lines.append(f"Thought: {thought}")
                for call in msg.tool_calls:
                    action = call["name"]
                    args = call["args"]
                    output_lines.append(f"Action: {action}")
                    output_lines.append(f"Action Input: {args}")
            else:
                output_lines.append(f"Thought: {thought}")
                if "Final Answer:" not in thought:
                    output_lines.append("Final Answer: (unknown)")

        elif isinstance(msg, ToolMessage):
            output_lines.append(f"Observation: {msg.content.strip()}")

    return "\n".join(output_lines)


# ---------------------------
# Export list
# ---------------------------

__all__ = [
    "PROMPT_REACT",
    "generate_tool_prompt",
    "format_react_messages"
]