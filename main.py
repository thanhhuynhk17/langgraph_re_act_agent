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
# from langgraph_agents.agents.re_act_agent.utils.helpers import load_model  # ✅ Correct

model = load_model(base_url="http://localhost:8000/v1")

# Define embedding
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store 

namespace = ("agent_memories",)
memory_tools = [
    # create_manage_memory_tool(namespace),
    # create_search_memory_tool(namespace, instructions="Find 5 relevant documents where any of the following administrative units are mentioned (province, ward, commune).")
]

# Tools
client = MultiServerMCPClient({
    "geoda": {
        "url": "http://localhost:2025/mcp", 
        "transport": "streamable_http"
    },
    "minio": {
        "url": "http://localhost:8090/mcp", 
        "transport": "streamable_http"
    },
})

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
        store = config["configurable"].get("__pregel_store", None)
    # elif len(args) == 2:
    #     print("checkpointer & store")
    #     checkpointer, store = args
    # else:
    #     raise ValueError("get_graph() expects (config)")
    if not checkpointer:
        checkpointer = InMemorySaver()
    try:
        geoda_tools = await client.get_tools()
        # print("geoda_tools", geoda_tools)
    except httpx.ConnectError as e:
        raise RuntimeError(f"❌ Cannot connect to MCP server. Connection failed.\n{str(e)}")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"❌ MCP server returned HTTP error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        raise RuntimeError(f"❌ Unexpected request error when connecting to MCP server: {str(e)}")

    
    geoda_tools = [hybrid_search, store_doc_in_neo4j]
    # ignored_tools = [
    #     "ask_object", "download_object", "text_to_object", 
    #     'get_bucket_lifecycle', 'get_bucket_replication', 'get_bucket_tags', 
    #     'get_bucket_versioning', 'get_object_metadata', 'get_object_presigned_url',
    #     'get_object_tags', 'get_object_versions', 'set_bucket_tags', 
    #     'set_bucket_versioning', 'set_object_tags', 'upload_object'
    #     ]
    # geoda_tools = [t for t in geoda_tools if t.name not in ignored_tools]
    
    config = copilotkit_customize_config(config, emit_tool_calls=[ t.name for t in geoda_tools])
    print(f"after config:\n{config}")
    if not geoda_tools:
        raise ValueError("No tools available from the MCP client.")

    GEODA_NAME = "GeoDaAgent"
    SEARCH_NAME = "SearchAgent"

    async def use_geoda_prompt(state, config: RunnableConfig):
        """Prepare the messages for the LLM."""
        print(f"geoda prompt hooked!")
        print(f'Chat history len: {len(state["messages"])}')
        tool_msg = state["messages"][-1]
        search_result_str = "No documents"

        if isinstance(tool_msg, ToolMessage):
            print(f"[tool_msg]:\n{tool_msg}\n=======")
            # FIXME: dynamic condition
            if tool_msg.name == "hybrid_search":
                try:
                    kwargs_str = tool_msg.content.split(f"{REACT_OBSERVATION}:")[1].strip()
                    search_result_str = run_hybrid_search(**json.loads(kwargs_str))
                    with open("prev_messages.md", "a", encoding="utf-8") as f:
                        f.write(search_result_str)
                    tool_msg.content = f"{REACT_OBSERVATION}: Truy vấn thông tin thành công. Thông tin nằm trong mục <retrieved_documents></retrieved_documents>.\nNếu không tìm thấy tài liệu liên quan, thử tăng số lượng tài liệu try vấn."

                except Exception as e:
                    print(f"[tool_msg] exception:{e}")

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
            REACT_QUESTION=REACT_QUESTION,
            REACT_THOUGHT=REACT_THOUGHT,
            REACT_ACTION = REACT_ACTION,
            REACT_ACTION_INPUT = REACT_ACTION_INPUT,
            REACT_OBSERVATION = REACT_OBSERVATION,
            REACT_FINAL_ANSWER = REACT_FINAL_ANSWER
        )

#         system_msg = f"""
# You are an agent specializing in geographic and spatial data analysis.

# ## 📚 RETRIEVED DOCUMENTS INSTRUCTIONS:

# You have access to user-relevant documents retrieved via hybrid search tools.

# ### ✅ Your responsibilities:
# - Extract **reliable geographic or spatial facts** from the documents.
# - Ground all answers strictly in retrieved evidence — **do not assume or fabricate** any information.
# - If the documents do **not contain sufficient information**, do **not guess**. Instead, **politely ask the user for more context** to continue.

# ### 📄 RETRIEVED DOCUMENTS:
# <retrieved_documents>
# {search_result_str}
# </retrieved_documents>

# ---

# ## ⚠️ CONSTRAINTS:
# - 🗣️ **Final response to the user must be in Vietnamese.**
# - ❌ **Do not invent, hallucinate, or speculate** outside the content of the retrieved documents.
# - ✅ Use quotes, summaries, or references from the retrieved content to support your reasoning and conclusions.

# ---

# {prompt_react}
# """.strip()
        system_msg = f"""{prompt_react}
## RÀNG BUỘC:
- **Phản hồi cuối cùng cho người dùng phải bằng tiếng Việt.**
- **Không được bịa đặt hoặc suy đoán ngoài nội dung của tài liệu.**
- **Phải sử dụng trích dẫn, tóm tắt hoặc tham chiếu rõ ràng từ nội dung được truy xuất để hỗ trợ lập luận và kết luận.**

## HƯỚNG DẪN SỬ DỤNG TÀI LIỆU ĐÃ TRUY XUẤT:

Bạn có quyền truy cập vào các tài liệu liên quan được truy xuất thông qua các công cụ tìm kiếm kết hợp.

### Nhiệm vụ của bạn:
- Trích xuất **thông tin đáng tin cậy** từ các tài liệu này.
- Chỉ đưa ra câu trả lời dựa trên bằng chứng thu được — **không tự suy đoán hoặc bịa đặt** bất kỳ thông tin nào.
- Nếu các tài liệu **không cung cấp đủ thông tin**, **không được đoán** mà hãy kết luận **không tìm thấy**.

### TÀI LIỆU ĐÃ TRUY XUẤT:
<retrieved_documents>
{search_result_str}
</retrieved_documents>
""".strip()
        return [SystemMessage(system_msg), *state["messages"]]

    def use_pre_hook(state, config: RunnableConfig):
        with open("prev_messages.md", "w", encoding="utf-8") as f:
            for msg in state["messages"]:
                f.write(f"\n{msg.pretty_repr()}\n")
                if isinstance(msg, ToolMessage):
                    f.write(f"\n{msg.model_dump_json()}\n")
        language = state.get("language", "vietnamese")
        print(f"language: {language}")
        last_msg = state["messages"][-1]
        artifact_json = None
        if isinstance(last_msg, ToolMessage):
            state["messages"][-1].content = last_msg.content.strip() if \
                REACT_OBSERVATION in last_msg.content else \
                f"{REACT_OBSERVATION}: {last_msg.content}".strip()
            

            if last_msg.artifact:
                if isinstance(last_msg.artifact, BaseModel):
                    artifact_data = last_msg.artifact.model_dump()
                else:
                    artifact_data = last_msg.artifact  # assume it's a dict

                artifact_dict = { last_msg.name: artifact_data }
                artifact_json = json.dumps(artifact_dict, ensure_ascii=False)

        elif isinstance(last_msg, HumanMessage):
            state["messages"][-1].content = last_msg.content.strip() if \
                REACT_QUESTION in last_msg.content else \
                f"{REACT_QUESTION}: {last_msg.content}".strip()

        return {
            "language": "vietnamese",
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
        if tool_calls and tool_calls[0]["function"]["name"] in ["summary_file", "backward_eliminator", "robust_ols"]:
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
                content=f"{new_msg.content}\nNgười dùng đã cập nhật lại tham số:\n{feedback_args.strip()}",
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
