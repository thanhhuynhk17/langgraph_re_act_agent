
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage,ToolMessage, AIMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage
from langchain_core.runnables import RunnableConfig

from prompts import PROMPT_REACT, generate_tool_prompt
from tools import check_weather, multiply, list_files, allowed_location_dir
from react_constants import *

llm = ChatOpenAI(
    model = 'qwen3:1.7b',
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    streaming=True,
    # extra_body={"include_reasoning": True},
    stop_sequences=[REACT_OBSERVATION]
)

# ReACT prompt
tools = [check_weather, multiply, allowed_location_dir, list_files]

from langgraph.prebuilt import create_react_agent

def use_pre_hook(state):
    with open("prev_messages.md", "w", encoding="utf-8") as f:
        f.write("===========================")
        for msg in state["messages"]:
            f.write(f"\n{type(msg)}\n{msg.content}\n")


def use_prompt(state, config: RunnableConfig):
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
        REACT_QUESTION=REACT_QUESTION,
        REACT_THOUGHT=REACT_THOUGHT,
        REACT_ACTION=REACT_ACTION,
        REACT_ACTION_INPUT=REACT_ACTION_INPUT,
        REACT_OBSERVATION=REACT_OBSERVATION,
        REACT_FINAL_ANSWER=REACT_FINAL_ANSWER,
    )
    last_msg = state["messages"][-1]
    # add react question tag
    if isinstance(last_msg, HumanMessage):
        state["messages"][-1].content = last_msg.content.strip() \
            if REACT_QUESTION in last_msg.content \
            else f"{REACT_QUESTION}: {last_msg.content}".strip()

    return [SystemMessage(prompt_react), *state["messages"]]

from helpers import process_ai_message
def use_post_hook(state):
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage):
        return # do nothing
    print("Post hooked: check action.")
    new_msg = process_ai_message(last_msg)
    if not new_msg: # has no new message, do nothing!
        return
    
    with open("post_messages.md", "w", encoding="utf-8") as f:
        for msg in state["messages"]:
            f.write(f"\n{msg.pretty_repr()}\n")
        f.write(f"\nlast_msg:\n{last_msg}\n")
        f.write(f"\nnew_msg:\n{new_msg}\n")

    # replaced the AIMessage with the new one (ReAct architecture format)
    return {
        "messages": [RemoveMessage(id=last_msg.id), new_msg]
    }
    


# System prompt
graph = create_react_agent(
    model=llm,
    tools=tools,
    # pre_model_hook=use_pre_hook,
    post_model_hook=use_post_hook,
    prompt=use_prompt
)

