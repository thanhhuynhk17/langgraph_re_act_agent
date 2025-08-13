from langchain_core.messages import convert_to_messages
from langchain_openai import ChatOpenAI

from react_constants import *
def load_model(base_url="http://localhost:8000/v1"):
    # Define the model
    model = ChatOpenAI(
        model="qwen3-30b-a3b",
        base_url=base_url,
        api_key="dummy_key",
        temperature=0.6,
        top_p=0.95,
        extra_body={"top_k": 20, "min_p": 0.0},
        stop_sequences=[REACT_OBSERVATION],
        streaming=True,

    )

    return model

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

from typing import Tuple
import re
def _detect_tool(text: str) -> Tuple[bool, str, str, str]:
    text = re.sub(r"^.*?</think>\n\n", "", text, flags=re.DOTALL)
    
    pattern = r"(\[react_\w+\])[:\s]*([\s\S]*?)(?=\[react_\w+\]?|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    thought = None
    func_name = None
    func_args = None
    final_answer = None
    for tag, content in matches:
        content = content.strip()
        if tag == REACT_THOUGHT and not thought:
            thought = content
        elif tag == REACT_ACTION and not func_name:
            func_name = content
        elif tag == REACT_ACTION_INPUT and not func_args:
            func_args = content
        elif tag == REACT_FINAL_ANSWER and not final_answer:
            final_answer = content

    return (func_name is not None), func_name, func_args, thought, final_answer


import re
import json
import uuid
from langchain_core.messages import AIMessage
from typing import Union, List

def create_tool_args(action, action_input):
    additional_kwargs ={}
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
    additional_kwargs["tool_calls"] = tool_calls
    return additional_kwargs

def process_ai_message(msg: AIMessage, all_tools: List) -> AIMessage:

    tool_calls = msg.additional_kwargs.get("tool_calls", None)
    has_action, action, action_input, thought, final_answer = _detect_tool(msg.content)
    if has_action and action.lower() not in [t.lower() for t in all_tools]:
        has_action = False

    if tool_calls and msg.content == "": # llm already has tool call & arguments, add content
        tool_call = tool_calls[0]["function"]
        # prepare for react agent context
        has_action = True
        action = tool_call["name"]
        action_input = tool_call["arguments"]
    elif not has_action: # do nothing, has no tool call
        print(f'has no action:\n{thought}')
        content = ''
        if thought:
            content += f"{REACT_THOUGHT}: {thought}\n\n"
        if final_answer:
            content += f"{REACT_FINAL_ANSWER}: {final_answer}"
            return AIMessage(content=content)
        return AIMessage(content=f"{REACT_FINAL_ANSWER}: {msg.content.strip()}")

    # in case action found but has no input, by pass with empty dict
    if has_action and not action_input:
        action_input = "{}"

    try: # parsing args
        action_input = json.dumps(json.loads(action_input), ensure_ascii=False)
    except json.JSONDecodeError:
        action_input = "{}"

    content = f'{REACT_THOUGHT}: {thought if thought else f"Sử dụng tool {action}"}\n{REACT_ACTION}: {action}\n{REACT_ACTION_INPUT}: {action_input}'
    # additional_kwargs ={}
    # tool_call_id = uuid.uuid4().hex
    # tool_calls = [{
    #     "index": 0,
    #     "id": tool_call_id,
    #     "function": {
    #         "name": action,
    #         "arguments": action_input
    #     },
    #     "type": "function"
    # }]
    # additional_kwargs["tool_calls"] = tool_calls
    additional_kwargs = create_tool_args(action, action_input)

    return AIMessage(
        content=content,
        additional_kwargs=additional_kwargs)
