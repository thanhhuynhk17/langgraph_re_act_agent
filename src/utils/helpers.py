import os
from dotenv import load_dotenv
load_dotenv()

from src.utils.react_constants import *
import re
import json
import uuid
from typing import Tuple, List, Union
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

# for routing incoming message
from src.utils.schemas import ChitChatCheck
from langchain_core.messages import SystemMessage, HumanMessage

def load_model(
    model_name="qwen3-30b-a3b",
    base_url="http://localhost:8000/v1",
    api_key="dummy_text",
    temperature=0.7
    ):
    """
    Defines and returns the language model instance, updating the stop sequence
    to use the new XML tag format for observations.
    """
    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        # top_p=0.95,
        # extra_body={"top_k": 20, "min_p": 0.0},
        # IMPORTANT: Updated the stop sequence to the new opening tag format
        streaming=True,
        reasoning_effort="low"
        # max_completion_tokens=4096
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


# Router: check user message is chitchat or ask bussiness info

def chitchat_classify(messages):
    last_msg = messages[-1]
    if not isinstance(last_msg, HumanMessage):  # nếu không phải message của user thì bỏ qua
        return False

    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", None)
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

    model = load_model(
        model_name=str(OPENAI_MODEL_NAME),
        base_url=str(OPENAI_BASE_URL),
        api_key=str(OPENAI_API_KEY),
        temperature=0  # greedy decoding 
    ).with_structured_output(ChitChatCheck)

    system_msg = SystemMessage(content="""
Bạn là một bộ phân loại tin nhắn.  
Nhiệm vụ của bạn: nhận một tin nhắn từ người dùng và quyết định xem nó thuộc loại **chit-chat** (giao tiếp xã giao) hay **restaurant inquiry** (câu hỏi liên quan đến nhà hàng, món ăn, thực đơn, giờ mở cửa, đặt món, hủy/cập nhật đơn).  

- Nếu tin nhắn mang tính chào hỏi, cảm ơn, xã giao, nói chuyện phiếm, khen/chê chung chung → gắn nhãn: is_chitchat = True.  
  Ví dụ:  
  - "Chào bạn"  
  - "Bạn khỏe không?"  
  - "Cảm ơn nhiều nhé!"  
  - "Trời nay đẹp ghê"  

- Nếu tin nhắn yêu cầu thông tin về nhà hàng, món ăn, menu, giá, giờ mở cửa, đặt bàn/đặt món, **hủy đơn**, **cập nhật đơn** → gắn nhãn: is_chitchat = False.  
  Ví dụ:  
  - "Nhà hàng có món chay không?"  
  - "Mở cửa lúc mấy giờ?"  
  - "Tôi muốn đặt bàn cho 4 người"  
  - "Tôi muốn hủy đơn hàng vừa đặt"  
  - "Có thể đổi món trong đơn của tôi được không?"
""".strip())

    raw_output = model.invoke([system_msg, *messages])
    chitchatcheck = ChitChatCheck.parse_obj(raw_output)
    return chitchatcheck.is_chitchat

