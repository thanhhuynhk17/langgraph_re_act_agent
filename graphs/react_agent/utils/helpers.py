import os
# Configure logging based on environment variable
import logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # Default to INFO if not set
logging_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}
logging.basicConfig(
    level=logging_levels.get(LOG_LEVEL, logging.INFO),  # Fallback to INFO if invalid
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Logging configured with level: %s", LOG_LEVEL)

from dotenv import load_dotenv
load_dotenv()

import httpx
import asyncio
from react_agent.utils.react_constants import *
import re
import json
import uuid
from datetime import datetime, timedelta
from typing import Tuple, List, Union, Dict, Any
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

# for routing incoming message
from react_agent.utils.schemas import ChitChatCheck
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
        streaming=True,
        # top_p=0.95,
        # extra_body={"top_k": 20, "min_p": 0.0},
        reasoning_effort="medium",
        stop_sequences=[f"<{TAG_OBSERVATION}"],

        # IMPORTANT: Updated the stop sequence to the new opening tag format
        # reasoning_effort="low", # for openai
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

    # TODO: navigate to graphs\dev.ipynb & make sure loop while action_input is invalid json
    try:  # Ensure action_input is a valid JSON string
        action_input = json.dumps(json.loads(action_input), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError) as e:
        # Keep raw string but mark as invalid for the agent
        action_input = f"__INVALID_JSON__::{action_input}"


    # Build the response content using the new XML tag format
    thought_text = thought if thought else f"Sử dụng tool {action}"
    content = (
        f"<{TAG_THOUGHT}>{thought_text}</{TAG_THOUGHT}>\n"
        f"<{TAG_ACTION}>{action}</{TAG_ACTION}>\n"
        f"<{TAG_ACTION_INPUT}>{action_input}</{TAG_ACTION_INPUT}>"
    )

    additional_kwargs = create_tool_args(action, action_input)

    return AIMessage(content=content, additional_kwargs=additional_kwargs)


async def fetch_customer_orders(customer_id: str) -> Dict[str, Any]:
    """
    Fetch customer order history from the restaurant API for prompt enrichment.

    Args:
        customer_id: The customer's UUID

    Returns:
        Dict containing enriched customer data or empty dict if failed

    Example API response:
    {
        "orders": [
            {
                "order_id": "ABC123",
                "total_cost": 150000,
                "dishes": [{"name": "Pho", "quantity": 2}],
                "created_at": "2024-01-15T10:30:00Z"
            }
        ],
        "total_orders": 5,
        "favorite_dishes": ["Pho", "Bun Bo"],
        "total_spent": 750000
    }
    """
    try:
        url = f"http://localhost:8000/api/customers/{customer_id}/orders"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException) as e:
        # Log error but don't block agent - return empty dict
        logger.warning(f"Failed to fetch customer orders for {customer_id}: {e}")
        return {}
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error fetching customer orders for {customer_id}: {e}")
        return {}


def enrich_customer_prompt(user_name: str, user_uuid: str, orders_data: Dict[str, Any]) -> str:
    """
    Create enriched customer prompt with detailed order history and preferences in formatted receipt style.

    Args:
        user_name: Customer name
        user_uuid: Customer UUID
        orders_data: Order data from API

    Returns:
        Formatted customer prompt string in receipt style
    """
    # Extract order insights
    total_orders = orders_data.get("total_orders", 0)
    total_spent = orders_data.get("total_spent", 0)
    favorite_dishes = orders_data.get("favorite_dishes", [])

    # Get orders array for detailed analysis
    orders = orders_data.get("orders", [])

    # Calculate detailed order statistics
    orders_last_30_days = 0
    orders_last_90_days = 0
    total_dishes_ordered = 0
    dish_frequency = {}
    recent_order_date = None

    if orders:
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)
        ninety_days_ago = now - timedelta(days=90)

        for order in orders:
            # Parse order date
            created_at = order.get("created_at")
            if created_at:
                try:
                    # Handle ISO format dates
                    if 'T' in created_at:
                        order_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        order_date = datetime.strptime(created_at, "%Y-%m-%d")

                    # Count orders in time periods
                    if order_date >= thirty_days_ago:
                        orders_last_30_days += 1
                    if order_date >= ninety_days_ago:
                        orders_last_90_days += 1

                    # Track most recent order
                    if recent_order_date is None or order_date > recent_order_date:
                        recent_order_date = order_date

                except (ValueError, AttributeError):
                    pass

            # Count total dishes and frequency
            dishes = order.get("dishes", [])
            for dish in dishes:
                if isinstance(dish, dict):
                    dish_name = dish.get("name", "")
                    quantity = dish.get("quantity", 1)
                    total_dishes_ordered += quantity
                    if dish_name:
                        dish_frequency[dish_name] = dish_frequency.get(dish_name, 0) + quantity

    # Find most ordered dish
    most_ordered_dish = None
    if dish_frequency:
        most_ordered_dish = max(dish_frequency.items(), key=lambda x: x[1])[0]

    # Calculate average order value
    avg_order_value = 0
    if total_orders > 0 and total_spent > 0:
        avg_order_value = total_spent / total_orders

    # Format with proper spacing using f-strings
    separator = "----------------------------------------"

    formatted_spent = ".0f"
    if total_spent > 0:
        formatted_spent = ",.0f"

    favorite_dishes_str = ", ".join(favorite_dishes[:3]) if favorite_dishes else ""

    # Build formatted receipt
    receipt_lines = [
        "🧾 Thông tin Khách Hàng",
        separator,
        f"Tên:              {user_name}",
        f"UUID:             {user_uuid}",
    ]

    # Add order information
    if total_orders > 0:
        receipt_lines.append(f"Đã đặt:           {total_orders} đơn")

        # Add detailed order statistics
        if orders_last_30_days > 0:
            receipt_lines.append(f"30 ngày gần nhất: {orders_last_30_days} đơn")

        if orders_last_90_days > 0:
            receipt_lines.append(f"90 ngày gần nhất: {orders_last_90_days} đơn")

        if avg_order_value > 0:
            formatted_avg = ",.0f"
            receipt_lines.append(f"Giá trị TB/đơn:   {avg_order_value:{formatted_avg}} vnđ")

        if total_dishes_ordered > 0:
            receipt_lines.append(f"Tổng món đã gọi:  {total_dishes_ordered} món")

        if most_ordered_dish:
            receipt_lines.append(f"Món gọi nhiều:    {most_ordered_dish}")

        if total_spent > 0:
            receipt_lines.append(f"Tổng chi tiêu:    {total_spent:{formatted_spent}} vnđ")

        if favorite_dishes_str:
            receipt_lines.append(f"Món yêu thích:    {favorite_dishes_str}")

        if recent_order_date:
            date_str = recent_order_date.strftime("%Y-%m-%d")
            receipt_lines.append(f"Đơn gần nhất:     {date_str}")
        else:
            receipt_lines.append("Đơn gần nhất:     N/A")
    else:
        receipt_lines.append("Chưa có đơn hàng")

    receipt_lines.append(separator)

    return "\n".join(receipt_lines)


# # Router: check user message is chitchat or ask bussiness info

# def chitchat_classify(messages):
#     last_msg = messages[-1]
#     if not isinstance(last_msg, HumanMessage):  # nếu không phải message của user thì bỏ qua
#         return False

#     OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", None)
#     OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

#     model = load_model(
#         model_name=str(OPENAI_MODEL_NAME),
#         base_url=str(OPENAI_BASE_URL),
#         api_key=str(OPENAI_API_KEY),
#         temperature=0  # greedy decoding 
#     ).with_structured_output(ChitChatCheck)

#     system_msg = SystemMessage(content="""
# Bạn là một bộ phân loại tin nhắn.  
# Nhiệm vụ của bạn: nhận một tin nhắn từ người dùng và quyết định xem nó thuộc loại **chit-chat** (giao tiếp xã giao) hay **restaurant inquiry** (câu hỏi liên quan đến nhà hàng, món ăn, thực đơn, giờ mở cửa, đặt món, hủy/cập nhật đơn).  

# - Nếu tin nhắn mang tính chào hỏi, cảm ơn, xã giao, nói chuyện phiếm, khen/chê chung chung → gắn nhãn: is_chitchat = True.  
#   Ví dụ:  
#   - "Chào bạn"  
#   - "Bạn khỏe không?"  
#   - "Cảm ơn nhiều nhé!"  
#   - "Trời nay đẹp ghê"  

# - Nếu tin nhắn yêu cầu thông tin về nhà hàng, món ăn, menu, giá, giờ mở cửa, đặt bàn/đặt món, **hủy đơn**, **cập nhật đơn** → gắn nhãn: is_chitchat = False.  
#   Ví dụ:  
#   - "Nhà hàng có món chay không?"  
#   - "Mở cửa lúc mấy giờ?"  
#   - "Tôi muốn đặt bàn cho 4 người"  
#   - "Tôi muốn hủy đơn hàng vừa đặt"  
#   - "Có thể đổi món trong đơn của tôi được không?"
# """.strip())

#     raw_output = model.invoke([system_msg, *messages])
#     chitchatcheck = ChitChatCheck.parse_obj(raw_output)
#     return chitchatcheck.is_chitchat
