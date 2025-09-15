from src.utils.react_constants import *
import re
import json
import uuid
from typing import Tuple, List, Union
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

def load_model(
    model_name="qwen3-30b-a3b",
    base_url="http://localhost:8000/v1",
    api_key="dummy_text"):
    """
    Defines and returns the language model instance, updating the stop sequence
    to use the new XML tag format for observations.
    """
    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0.6,
        top_p=0.95,
        extra_body={"top_k": 20, "min_p": 0.0},
        # IMPORTANT: Updated the stop sequence to the new opening tag format
        stop_sequences=[f"<{TAG_OBSERVATION}"],
        streaming=True,
    )
    return model

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def _detect_tool(text: str) -> Tuple[bool, str, str, str, str]:
    """
    Detects and extracts tool calls from text formatted with XML-style tags.
    """
    # New regex to find <tag>content</tag> patterns
    pattern = r"<({prefix}\w+)>([\s\S]*?)</\1>".format(prefix="react_")
    matches = re.findall(pattern, text, re.DOTALL)

    thought = None
    func_name = None
    func_args = None
    final_answer = None

    for tag_name, content in matches:
        content = content.strip()
        if tag_name == TAG_THOUGHT and not thought:
            thought = content
        elif tag_name == TAG_ACTION and not func_name:
            func_name = content
        elif tag_name == TAG_ACTION_INPUT and not func_args:
            func_args = content
        elif tag_name == TAG_FINAL_ANSWER and not final_answer:
            final_answer = content

    return (func_name is not None), func_name, func_args, thought, final_answer

def create_tool_args(action, action_input):
    """Creates the dictionary for tool call arguments."""
    tool_call_id = uuid.uuid4().hex
    tool_calls = [{
        "index": 0,
        "id": tool_call_id,
        "function": {
            "name": action,
            "arguments": action_input
        },
        "type": "function"
    }]
    return {"tool_calls": tool_calls}


def ignore_before_last_think(text: str) -> str:
    # Split by the tag
    parts = text.rsplit("</think>", 1)
    # Return the part after the last </think>
    return parts[-1].strip()

def process_ai_message(msg: AIMessage, all_tools: List[str]) -> AIMessage:
    """
    Processes the AI message, parsing for the new XML-style tags and
    formatting the output accordingly.
    """
    tool_calls = msg.additional_kwargs.get("tool_calls", None)
    
    # ignore all </think> tags
    msg.content = ignore_before_last_think(msg.content)
    
    has_action, action, action_input, thought, final_answer = _detect_tool(msg.content)

    if has_action and action.lower() not in [t.lower() for t in all_tools]:
        has_action = False

    if tool_calls and msg.content == "":  # LLM already has tool call & arguments
        tool_call = tool_calls[0]["function"]
        action = tool_call["name"]
        action_input = tool_call["arguments"]
        has_action = True
    elif not has_action:
        # If there's no action, format the response with the final answer tag.
        content = ''
        if thought:
            content += f"<{TAG_THOUGHT}>{thought}</{TAG_THOUGHT}>\n\n"
        
        response_content = final_answer if final_answer else msg.content.strip()
        if response_content == "":
            return None
        content += f"<{TAG_FINAL_ANSWER}>{response_content}</{TAG_FINAL_ANSWER}>"
        return AIMessage(content=content)

    if has_action and not action_input:
        action_input = "{}"

    try:  # Ensure action_input is a valid JSON string
        action_input = json.dumps(json.loads(action_input), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        action_input = "{}"

    # Build the response content using the new XML tag format
    thought_text = thought if thought else f"Sử dụng tool {action}"
    content = (
        f"<{TAG_THOUGHT}>{thought_text}</{TAG_THOUGHT}>\n"
        f"<{TAG_ACTION}>{action}</{TAG_ACTION}>\n"
        f"<{TAG_ACTION_INPUT}>{action_input}</{TAG_ACTION_INPUT}>"
    )

    additional_kwargs = create_tool_args(action, action_input)

    return AIMessage(content=content, additional_kwargs=additional_kwargs)