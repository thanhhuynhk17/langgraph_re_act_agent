from react_constants import *

PROMPT_REACT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Do not attempt to perform any task yourself.
Always reason step by step and call tools explicitly where needed.

Use the following format:
{REACT_QUESTION}: The input question you must answer.

{REACT_THOUGHT}: You should always think about what to do.  
{REACT_ACTION}: The action to take, should be one of [{tool_names}].  
{REACT_ACTION_INPUT}: The input to the action. (expect JSON object with double quotes) 
{REACT_OBSERVATION}: (will be inserted by the system — DO NOT GENERATE)  
... (this {REACT_THOUGHT}/{REACT_ACTION}/{REACT_ACTION_INPUT}/{REACT_OBSERVATION} block can repeat zero or more times)

{REACT_THOUGHT}: I now know the final answer.  
{REACT_FINAL_ANSWER}: The final answer to the original input question.
""".strip()


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


# ---------------------------
# Export list
# ---------------------------

__all__ = [
    "PROMPT_REACT",
    "generate_tool_prompt",
    "format_react_messages"
]