import os
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage

from pydantic import BaseModel

from src.utils.helpers import load_model, create_tool_args

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", None)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
model = load_model(
    model_name=OPENAI_MODEL_NAME,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

import httpx
from src.utils.prompts import generate_tool_prompt, PROMPT_REACT
from src.utils.react_constants import *
from src.utils.helpers import process_ai_message
from src.utils.schemas import menu_desc
from src.utils.tools import all_agent_tools

from src.utils.interrupt_any_tool import add_human_in_the_loop
from src.utils.logging_setup import logger
import json
from datetime import datetime
import pendulum

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
    # checkpointer = InMemorySaver()
    config = args[0]
    config["recursion_limit"] = 99
    # Tools
    agent_tools = all_agent_tools

    AGENT_NAME = "GeoDaAgent"

    async def use_geoda_prompt(state, config: RunnableConfig):
        """Prepare the messages for the LLM."""
        print(f"geoda prompt hooked!")
        print(f'Chat history len: {len(state["messages"])}')
        tool_msg = state["messages"][-1]

        if isinstance(tool_msg, ToolMessage):
            print(f"[tool_msg]:\n{tool_msg}\n=======")

        # Geoda system prompt
        tool_descs = ""
        for t in agent_tools:
            tool_descs += generate_tool_prompt(
                name_for_model=t.name,
                name_for_human=t.name,
                description_for_model=t.description,
                schema=t.args_schema if isinstance(t.args_schema, type) and issubclass(t.args_schema, BaseModel) else BaseModel
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
        menu_prompt = "\n".join(["Cơm quê Dượng Bầu menu (mã món ăn, tên món ăn):",menu_desc])
        info_prompt = "\n".join([
            "Dựa vào các mẫu trò chuyện sau để giao tiếp với khách",
            "Khách hỏi: Địa chỉ quán ở đâu?\nTrả lời: Dạ Cơm Quê Dượng Bầu ở địa chỉ Lầu 3 - chung cư 40E Ngô Đức Kế, Phường Sài Gòn, TP. HCM ạ, khi đến chung cư anh/chị cứ bấm thang máy lên lầu 3 nha.",
            "Khách hỏi: Quán mở mấy giờ\nTrả lời: Dạ, Cơm Quê Dượng Bầu hoạt động từ 10h trưa đến 22h tối ạ 🥰",
            "Khách hỏi: Có nhận ship không\n Trả lời: Dạ quán em có ship ạ. Mời anh/ chị xem menu và cho em xin list món và em gửi bill anh / chị chuyển khoản xong em lên đơn cho mình nhé ạ 🥰"
        ])
        system_msg = (
            f"{prompt_react.strip()}\n\n"
            f"{menu_prompt.strip()}\n\n"
            f"{info_prompt.strip()}\n\n"
            f"{datetime_prompt.strip()}"
        )

        # print("system_msg", system_msg)
        return [SystemMessage(system_msg), *state["messages"]]

    def use_pre_hook(state, config: RunnableConfig):
        last_msg = state["messages"][-1]
        artifact_json = None
        if isinstance(last_msg, ToolMessage):
            if f"<{TAG_OBSERVATION}>" in last_msg.content:
                # Đã đúng format => giữ nguyên
                state["messages"][-1].content = last_msg.content.strip()
            else:
                # Chưa có => bọc trong cặp thẻ
                state["messages"][-1].content = f"""
<{TAG_OBSERVATION}>{last_msg.content.strip()}</{TAG_OBSERVATION}>
""".strip()

            if last_msg.artifact:
                if isinstance(last_msg.artifact, BaseModel):
                    artifact_data = last_msg.artifact.model_dump()
                else:
                    artifact_data = last_msg.artifact  # assume it's a dict

                artifact_dict = { last_msg.name: artifact_data }
                artifact_json = json.dumps(artifact_dict, ensure_ascii=False)

        elif isinstance(last_msg, HumanMessage):
            if f"<{TAG_QUESTION}>" in last_msg.content:
                state["messages"][-1].content = last_msg.content.strip()
            else:
                state["messages"][-1].content = f"""
<{TAG_QUESTION}>{last_msg.content.strip()}</{TAG_QUESTION}>
""".strip()

            # remove duplicate human message
            if len(state["messages"]) > 1 and isinstance(state["messages"][-2], HumanMessage):
                print("remove msg")
                return {
                    "json_data": artifact_json,
                    "messages": [RemoveMessage(id=state["messages"][-2].id)],
                }
        return {
            "json_data": artifact_json
        }

    # Prevent hallucinations: 
    def use_post_hook(state, config: RunnableConfig):
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage):
            return # do nothing
        print("Post hooked: check action.")
        all_tools = [t.name for t in agent_tools]
        new_msg = process_ai_message(last_msg, all_tools)

        return {
            **state,
            "messages": [RemoveMessage(id=last_msg.id), new_msg],
        }

    geoda_agent = create_react_agent(
                    model,
                    name=AGENT_NAME,
                    tools=agent_tools,
                    pre_model_hook=use_pre_hook,
                    post_model_hook=use_post_hook,
                    prompt=use_geoda_prompt,
                    debug=False
                )

    return geoda_agent
