import os
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage

from pydantic import BaseModel

from my_app.utils.helpers import load_model, create_tool_args

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", None)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
model = load_model(
    model_name=OPENAI_MODEL_NAME,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

import httpx
from my_app.utils.prompts import generate_tool_prompt, PROMPT_REACT
from my_app.utils.react_constants import *
from my_app.utils.helpers import process_ai_message

from my_app.utils.tools import all_agent_tools
from my_app.utils.interrupt_any_tool import add_human_in_the_loop
from my_app.utils.logging_setup import logger

import json

async def get_graph(*args):
    # checkpointer = InMemorySaver()

    # Tools
    agent_tools = agent_tools

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
                schema=t.args_schema
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

        system_msg = f"""{prompt_react}
""".strip()
        # print("system_msg", system_msg)
        return [SystemMessage(system_msg), *state["messages"]]

    def use_pre_hook(state, config: RunnableConfig):
        print("agent_tools", agent_tools)
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
