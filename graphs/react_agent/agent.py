# ================= 2. Prompts =================
CHITCHAT_SYS = "Bạn là nhân viên lễ tân thân thiện của Cơm Quê Dượng Bầu."
SPEED_SYS = (
    "Bạn chỉ biết: địa chỉ (40E Ngô Đức Kế), giờ mở cửa (10-22h), có ship. "
    "Khác 3 thông tin → chỉ được nói 'Chuyển sang agent khác'."
)

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


from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.graph import MessagesState, START, END
from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage
from langgraph.graph.ui import push_ui_message

from pydantic import BaseModel
from typing import Optional

from react_agent.utils.helpers import load_model, fetch_customer_orders, enrich_customer_prompt

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", None)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
model = load_model(
    model_name=OPENAI_MODEL_NAME,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

import httpx
from react_agent.utils.prompts import generate_tool_prompt, PROMPT_REACT
from react_agent.utils.react_constants import *
from react_agent.utils.helpers import process_ai_message
from react_agent.utils.schemas import VALID_TYPES, CustomAgentState
from react_agent.utils.tools import all_agent_tools

from react_agent.utils.interrupt_any_tool import add_human_in_the_loop
import json
import pendulum

client = MultiServerMCPClient({
    "graph_rag": {
        "url": "http://localhost:8000/mcp", 
        "transport": "streamable_http"
    }
})

def build_datetime_prompt() -> str:
    now_vn = pendulum.now(DEFAULT_TZ)
    now_iso = now_vn.to_iso8601_string()
    now_vi = now_vn.format("HH:mm ngày DD/MM/YYYY")  # chuỗi tiếng Việt
    
    return (
        "- Luôn parse thời gian đặt bàn (booking_time) thành ISO datetime với timezone Asia/Ho_Chi_Minh.\n"
        "- Nếu khách nói 'tối nay', 'ngày mai', hãy chuyển thành ngày giờ cụ thể theo lịch hiện tại.\n"
        f"- Hiện tại là: {now_iso} (tức {now_vi}).\n"
    )
async def get_graph(*args):
    # print("get_graph run with config:", args)
    if len(args) == 1:
        config = args[0] if args else {}
        config.setdefault("recursion_limit", 99)
        checkpointer = None
        store = None

    # Tools
    try:
        mcp_tools = await client.get_tools()
        # print("geoda_tools", geoda_tools)
    except httpx.ConnectError as e:
        raise RuntimeError(f"❌ Cannot connect to MCP server. Connection failed.\n{str(e)}")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"❌ MCP server returned HTTP error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        raise RuntimeError(f"❌ Unexpected request error when connecting to MCP server: {str(e)}")

    
    agent_tools = [
        # *all_agent_tools,
        *mcp_tools
    ]

    AGENT_NAME = "GeoDaAgent"

    async def use_geoda_prompt(state, config: RunnableConfig):
        """Prepare the messages for the LLM."""
        logger.debug(f"geoda prompt hooked!")
        logger.debug(f'Chat history len: {len(state["messages"])}')
        tool_msg = state["messages"][-1]

        # Geoda system prompt
        tool_descs = ""
        logger.debug("agent_tools: %s", agent_tools)
        for t in agent_tools:
            schema = getattr(t, "args_schema", None)

            tool_descs += generate_tool_prompt(
                name_for_model=t.name,
                name_for_human=t.name,
                description_for_model=t.description,
                schema=schema
            )
        prompt_react = PROMPT_REACT.format(
            tool_descs=tool_descs,
            tool_names=",".join([t.name for t in agent_tools]),
            TAG_QUESTION=TAG_QUESTION,
            TAG_THOUGHT=TAG_THOUGHT,
            TAG_ACTION = TAG_ACTION,
            TAG_ACTION_INPUT = TAG_ACTION_INPUT,
            TAG_OBSERVATION = TAG_OBSERVATION,
            TAG_FINAL_ANSWER = TAG_FINAL_ANSWER
        )

        datetime_prompt = build_datetime_prompt()
        dish_types = "\n".join(VALID_TYPES)
        menu_prompt = "\n".join([
            "Dưới đây là các danh mục món ăn và thức uống của quán, phải liệt kê đầy đủ cho khách lựa chọn:",
            dish_types,
            "- Hỏi khách cung cấp các danh mục cụ thể."
            ])
        info_prompt = "\n".join([
            "Dựa vào các mẫu trò chuyện sau để giao tiếp với khách",
            "Khách hỏi: Địa chỉ quán ở đâu?\nTrả lời: Dạ Cơm Quê Dượng Bầu ở địa chỉ Lầu 3 - chung cư 40E Ngô Đức Kế, Phường Sài Gòn, TP. HCM ạ, khi đến chung cư anh/chị cứ bấm thang máy lên lầu 3 nha.",
            "Khách hỏi: Quán mở mấy giờ\nTrả lời: Dạ, Cơm Quê Dượng Bầu hoạt động từ 10h trưa đến 22h tối ạ 🥰",
            "Khách hỏi: Có nhận ship không\n Trả lời: Dạ quán em có ship ạ. Mời anh/ chị xem menu và cho em xin list món và em gửi bill anh / chị chuyển khoản xong em lên đơn cho mình nhé ạ 🥰"
        ])
        
        user_name = state.get("user_name",None)
        user_uuid = state.get("user_uuid",None)
        logger.info(f"Username: {user_name} {user_uuid}")
        
        # Enrich customer data with order history if available
        customer_prompt = "Thông tin khách hàng: chưa có thông tin khách hàng"
        if user_uuid:
            # Fetch customer orders for prompt enrichment
            orders_data = await fetch_customer_orders(user_uuid)
            customer_prompt = enrich_customer_prompt(user_name, user_uuid, orders_data)
            logger.info(f"customer_prompt:\n{customer_prompt}")
        system_msg = (
            f"{prompt_react.strip()}\n\n"
            f"{menu_prompt.strip()}\n\n"
            f"{info_prompt.strip()}\n\n"
            f"{datetime_prompt.strip()}\n\n"
            "Thông tin khuyến mãi: chưa có chương trình khuyến mãi.\n"
            "Thông tin chuyển khoản:\n- Anh Vinh quản lý - Ngân hàng VietinBank - Số tài khoản: 105872648804\n- Anh Đức WorldWide - Ngân hàng Techcombank - Số tài khoản: 9808888088\n\n"
            f"{customer_prompt.strip()}\n"
        )

        logger.debug("system_msg: %s", system_msg)
        return [SystemMessage(system_msg), *state["messages"]]

    def use_pre_hook(state, config: RunnableConfig):
        last_msg = state["messages"][-1]
        logger.info("use_pre_hook state: %s", state)

        if isinstance(last_msg.content, str):
            # convert to list of dict
            last_msg.content = [{"type": "text", "text": last_msg.content.strip()}]

        content = last_msg.content[0].get("text", "").strip()
        if isinstance(last_msg, ToolMessage):
            if f"<{TAG_OBSERVATION}>" in content:
                # Đã đúng format => giữ nguyên
                state["messages"][-1].content = [{"type": "text", "text": content}]
                logger.debug("True Tool msg: %s", last_msg.content)
            else:
                # Chưa có => bọc trong cặp thẻ
                state["messages"][-1].content = [{"type": "text", "text": f"<{TAG_OBSERVATION}>{content}</{TAG_OBSERVATION}>"}]
                logger.debug("FA Tool msg: %s", last_msg.content)
            # Push to UI
            logger.info("Push tool message to UI")
            logger.info("Tool msg: %s", state["messages"][-1])
            tool_msg = state["messages"][-1]
            if tool_msg.artifact:
                logger.info("Pushing tool message to UI: %s", tool_msg)
                # only push if tool_msg is not the same as last_msg
                push_ui_message(tool_msg.name, tool_msg.artifact[-1], message=tool_msg)

        elif isinstance(last_msg, HumanMessage):
            if f"<{TAG_QUESTION}>" not in content:
                state["messages"][-1].content = [{"type": "text", "text": f"<{TAG_QUESTION}>{content}</{TAG_QUESTION}>"}]
            # remove duplicate human message
            if len(state["messages"]) > 1 and isinstance(state["messages"][-2], HumanMessage):
                logger.info("remove duplicate human message")
                return {
                    # "json_data": artifact_json,
                    "messages": [RemoveMessage(id=state["messages"][-2].id)],
                }
        return {
            # "json_data": artifact_json
        }

    # Prevent hallucinations:
    def use_post_hook(state, config: RunnableConfig):
        logger.debug("Post hook: check action")

        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage):
            return # do nothing
        all_tools = [t.name for t in agent_tools]
        new_msg = process_ai_message(last_msg, all_tools)
        if not new_msg: # retry logic
            logger.info("retry triggered due to invalid AI message format")
            # goto prev
            return Command(
                goto="pre_model_hook",
                update={"messages": [RemoveMessage(id=last_msg.id)]},
            )

        return {
            # **state,
            "messages": [RemoveMessage(id=last_msg.id), new_msg],
        }

    # subgraph
    geoda_agent = create_react_agent(
                    model,
                    name=AGENT_NAME,
                    tools=agent_tools,
                    pre_model_hook=use_pre_hook,
                    post_model_hook=use_post_hook,
                    prompt=use_geoda_prompt,
                    state_schema=CustomAgentState,
                    debug=False,
                )

    
    # builder = StateGraph(CustomAgentState)
    # def chitchat_router(state):
    #     is_chitchat = chitchat_classify(state["messages"])
    #     if is_chitchat:
    #         print("chitchat hooked!")
    #         res_msg = model.invoke([
    #             SystemMessage(content=CHITCHAT_SYS_PROMPT),
    #             *state["messages"]
    #         ])
    #         res_msg.content = f"<{TAG_FINAL_ANSWER}>{res_msg.content}</{TAG_FINAL_ANSWER}>"

    #         return {
    #             "messages": [res_msg],
    #             "is_chitchat": True,
    #         }
    #     # not chitchat → pass state forward
    #     return {
    #         # "messages": state["messages"],
    #         "is_chitchat": False,
    #     }

    # builder.add_node("chitchat_router", chitchat_router)
    # builder.add_node("geoda_agent", geoda_agent)

    # # Conditional routing: read flag from state
    # def route_condition(state):
    #     return "end" if state["is_chitchat"] else "geoda_agent"

    # builder.add_conditional_edges(
    #     "chitchat_router",
    #     route_condition,
    #     {"end": END, "geoda_agent": "geoda_agent"}
    # )

    # builder.set_entry_point("chitchat_router")
    # graph = builder.compile()

    return geoda_agent



    



# # ================= 4. Graph 3 node =================
# def get_graph(*args) -> StateGraph:
#     config = args[0] if args else {}
#     config.setdefault("recursion_limit", 99)
    
#     builder = StateGraph(CustomAgentState)
#     builder.add_node("chitchat_node", chitchat_node)
#     builder.add_node("speed_node",  speed_node)
#     builder.add_node("geoda_node",  geoda_node)

#     builder.set_entry_point("chitchat_node")

#     # 1. chitchat → speed nếu KHÔNG phải xã giao
#     builder.add_conditional_edges(
#         "chitchat_node",
#         lambda s: END if s.get("is_chitchat") else "speed_node",
#         {"speed_node": "speed_node", END: END}
#     )

#     # 2. speed → geoda nếu có flag "goto"
#     builder.add_conditional_edges(
#         "speed_node",
#         lambda s: "geoda_node" if s.get("goto") == "geoda_node" else END,
#         {"geoda_node": "geoda_node", END: END}
#     )

#     builder.add_edge("geoda_node", END)
#     return builder.compile()
