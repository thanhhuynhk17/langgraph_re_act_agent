from typing import Tuple
def _detect_tool(text: str) -> Tuple[bool, str, str, str]:
    special_thought_token = '\nThought:'
    special_func_token = '\nAction:'
    special_args_token = '\nAction Input:'

    func_name, func_args, thought = None, None, None

    # Find first occurrences
    i = text.find(special_func_token)
    j = text.find(special_args_token)

    # Basic validation
    if 0 <= i < j:
        # Extract first Thought before Action
        t = text.find(special_thought_token)
        if 0 <= t < i:
            thought = text[t + len(special_thought_token):i].strip()

        # Extract Action and Input
        func_name = text[i + len(special_func_token):j].strip()
        
        # Find where Action Input ends (could be followed by \nObservation: or something else)
        after_args = text[j + len(special_args_token):]
        k = after_args.find('\n')  # look for newline after Action Input
        if k != -1:
            func_args = after_args[:k].strip()
        else:
            func_args = after_args.strip()

    return (func_name is not None), func_name, func_args, thought


import re
import json
import uuid
from langchain_core.messages import AIMessage, HumanMessage
from typing import Union
def process_ai_message(msg: AIMessage) -> Union[AIMessage, HumanMessage]:
    has_action, action, action_input, thought = _detect_tool(msg.content)

    if not has_action: # do nothing, has no tool call
        return None

    content = ""
    additional_kwargs ={}
    try:
        action_input = json.loads(action_input)
        tool_call_id = uuid.uuid4().hex
        tool_calls = [{
            "index": 0,
            "id": tool_call_id,
            "function": {
                "name": action,
                "arguments": json.dumps(action_input)
            },
            "type": "function"
        }]
        content = f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}"
        additional_kwargs["tool_calls"] = tool_calls
    except json.JSONDecodeError as e:
        content = f"Invalid arguments with tool {action}:\n{action_input}\nException error message:{e}"
        
        print(content)
    except Exception as e:
        content = f"Unhandle exception with tool {action}:\n{action_input}\nException error message:{e}"
    finally:
        new_msg = AIMessage(
            content=content,
            additional_kwargs=additional_kwargs)
        return new_msg
