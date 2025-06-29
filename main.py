
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage,ToolMessage, AIMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage

from prompts import PROMPT_REACT, format_react_messages, generate_tool_prompt
from tools import check_weather, multiply, list_files, allowed_location_dir

llm = ChatOpenAI(
    model = 'qwen3:1.7b',
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    streaming=True,
    extra_body={"include_reasoning": True}
)

# ReACT prompt
tools = [check_weather, multiply, allowed_location_dir, list_files]

from langgraph.prebuilt import create_react_agent

def use_pre_hook(state):
    with open("prev_messages.md", "w", encoding="utf-8") as f:
        f.write("===========================")
        for msg in state["messages"]:
            f.write(f"\n{type(msg)}\n{msg.content}\n")

    if len(state["messages"])==1:
        tool_descs = ""
        for t in tools:
            tool_descs += generate_tool_prompt(
                name_for_model=t.name,
                name_for_human=t.name,
                description_for_model=t.description,
                schema=t.args_schema
            )
        prompt_react = PROMPT_REACT.format(
            tool_descs=tool_descs,
            tool_names=",".join([t.name for t in tools]),
            query=state["messages"][-1].content
        )
        state["messages"][-1].content=prompt_react
        return {
            "messages": state["messages"]
        }

# def use_prompt(state):
#     print(len(state["messages"]))
#     for msg in state["messages"]:
#         msg.pretty_print()
        
#     return {
#         "messages": [SystemMessage("Always reason step by step and call tools explicitly where needed.\nDo not attempt to perform any task yourself.")]
#     }

def use_post_hook(state):
    msg_formated = format_react_messages(state["messages"])
    msg_types = [str(type(msg)) for msg in state["messages"]]
    
    with open("post_messages.md", "w", encoding="utf-8") as f:
        f.write(str(state))
        # f.write(f"{msg_types}")
        # f.write(f"{msg_formated}\n\n")

        # for msg in state["messages"]:
        #     f.write(f"\n{type(msg)}\n{msg}\n")

# System prompt

graph = create_react_agent(
    model=llm,
    tools=tools,
    pre_model_hook=use_pre_hook,
    post_model_hook=use_post_hook,
    prompt=SystemMessage("Always reason step by step and call tools explicitly where needed.\nDo not attempt to perform any task yourself.")
)

