# agents/re_act_agent/agent.py
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage

from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

# Get current script's parent directory (adjust as needed)
# import sys
# from pathlib import Path
# module_path = Path(__file__).resolve().parent # root
# print(module_path)
# sys.path.append(str(module_path))
from helpers import load_model, create_tool_args
from tools import bag_of_words_generator
# from langgraph_agents.agents.re_act_agent.utils.helpers import load_model  # ✅ Correct

model = load_model(base_url="http://localhost:8000/v1")
namespace = ("agent_memories",)

# # Tools
# client = MultiServerMCPClient({
#     "geoda": {
#         "url": "http://localhost:2025/mcp", 
#         "transport": "streamable_http"
#     },
#     "minio": {
#         "url": "http://localhost:8090/mcp", 
#         "transport": "streamable_http"
#     },
# })

# from utils.memory_manager_agent import manager, namespace as manager_ns, reflection
# from utils.memory_manager_agent import Episode
from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt

import httpx
# from utils.helpers import get_detailed_instruct
from prompts import generate_tool_prompt, PROMPT_REACT
from react_constants import *
from tools import store_doc_in_neo4j
from helpers import process_ai_message
# custom state for communicate with UI
from schemas import AgentState
# emit tool message
from copilotkit.langgraph import copilotkit_customize_config, copilotkit_emit_message

from hybrid_search import run_hybrid_search, hybrid_search
import json

async def get_graph(*args):
    # Determine argument pattern
    if len(args) == 1:
        config = args[0]
        print(type(config))
        config["recursion_limit"] = 99

        checkpointer = config["configurable"]["__pregel_checkpointer"]

    # if not checkpointer:
    #     checkpointer = InMemorySaver()
    # try:
    #     geoda_tools = await client.get_tools()
    #     # print("geoda_tools", geoda_tools)
    # except httpx.ConnectError as e:
    #     raise RuntimeError(f"❌ Cannot connect to MCP server. Connection failed.\n{str(e)}")
    # except httpx.HTTPStatusError as e:
    #     raise RuntimeError(f"❌ MCP server returned HTTP error: {e.response.status_code} - {e.response.text}")
    # except httpx.RequestError as e:
    #     raise RuntimeError(f"❌ Unexpected request error when connecting to MCP server: {str(e)}")
    
    geoda_tools = [bag_of_words_generator]
    config = copilotkit_customize_config(config, emit_tool_calls=[ t.name for t in geoda_tools])

    GEODA_NAME = "GeoDaAgent"

    async def use_geoda_prompt(state, config: RunnableConfig):
        """Prepare the messages for the LLM."""
        print(f"geoda prompt hooked!")
        print(f'Chat history len: {len(state["messages"])}')
        tool_msg = state["messages"][-1]
        search_result_str = "No documents"

        if isinstance(tool_msg, ToolMessage):
            print(f"[tool_msg]:\n{tool_msg}\n=======")

        # Geoda system prompt
        tool_descs = ""
        for t in geoda_tools:
            tool_descs += generate_tool_prompt(
                name_for_model=t.name,
                name_for_human=t.name,
                description_for_model=t.description,
                schema=t.args_schema
            )
        prompt_react = PROMPT_REACT.format(
            tool_descs=tool_descs,
            tool_names=",".join([t.name for t in geoda_tools]),
            TAG_QUESTION=TAG_QUESTION,
            TAG_THOUGHT=TAG_THOUGHT,
            TAG_ACTION = TAG_ACTION,
            TAG_ACTION_INPUT = TAG_ACTION_INPUT,
            TAG_OBSERVATION = TAG_OBSERVATION,
            TAG_FINAL_ANSWER = TAG_FINAL_ANSWER
        )

        system_msg = f"""{prompt_react}
""".strip()
        print("system_msg", system_msg)
        return [SystemMessage(system_msg), *state["messages"]]

    def use_pre_hook(state, config: RunnableConfig):
        with open("prev_messages.md", "w", encoding="utf-8") as f:
            for msg in state["messages"]:
                f.write(f"\n{msg.pretty_repr()}\n")
                if isinstance(msg, ToolMessage):
                    f.write(f"\n{msg.model_dump_json()}\n")

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
        with open("post_messages.md", "w", encoding="utf-8") as f:
            for msg in state["messages"]:
                f.write(f"\n{msg.model_dump_json()}\n")

        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage):
            return # do nothing
        print("Post hooked: check action.")
        all_tools = config["metadata"]["copilotkit:emit-tool-calls"]
        new_msg = process_ai_message(last_msg, all_tools)
        if not new_msg: # has no new message, do nothing!
            return

        # human-in-loop
        tool_calls = new_msg.additional_kwargs.get("tool_calls", None)
        print("tool_calls", tool_calls)
        if tool_calls and tool_calls[0]["function"]["name"] in ["hybrid_search","summary_file", "backward_eliminator", "robust_ols"]:
        # if tool_calls and tool_calls[0]["function"]["name"] in []:
            print("---human_feedback---")
            feedback_args = interrupt({
                "name": tool_calls[0]["function"]["name"],
                "type": "ask",
                "content": f'{tool_calls[0]["function"]["arguments"]}'
            })
            print("before new_msg\n", new_msg)
            print("feedback", feedback_args)
            additional_kwargs = create_tool_args(
                tool_calls[0]["function"]["name"],
                feedback_args
            )
            new_msg = AIMessage(
                content=f"{new_msg.content.split(f'</{TAG_ACTION_INPUT}>')[0]}\nNgười dùng đã cập nhật lại tham số:\n{feedback_args.strip()}</{TAG_ACTION_INPUT}>",
                additional_kwargs=additional_kwargs)
            print("after new_msg\n", new_msg)
        return {
            **state,
            "messages": [RemoveMessage(id=last_msg.id), new_msg],
        }

    try:
        geoda_agent = create_react_agent(
                        model,
                        name=GEODA_NAME,
                        tools=geoda_tools,
                        pre_model_hook=use_pre_hook,
                        post_model_hook=use_post_hook,
                        prompt=use_geoda_prompt,
                        state_schema=AgentState,
                        checkpointer=checkpointer,
                        debug=True
                    )

    except Exception as e:
        print("Assign checkpointer & store failed:\n{e}")
        geoda_agent = create_react_agent(
                        model,
                        name=GEODA_NAME,
                        tools=geoda_tools,
                        pre_model_hook=use_pre_hook,
                        post_model_hook=use_post_hook,
                        prompt=use_geoda_prompt,
                        state_schema=AgentState,
                        debug=True
                    )
    
    return geoda_agent
