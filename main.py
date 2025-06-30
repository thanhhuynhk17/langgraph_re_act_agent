
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
    max_tokens=512,
    extra_body={"include_reasoning": True},
    stop=["Observations: "]
)

# ReACT prompt
tools = [check_weather, multiply, allowed_location_dir, list_files]

from langgraph.prebuilt import create_react_agent

def use_pre_hook(state):
    with open("prev_messages.md", "w", encoding="utf-8") as f:
        f.write("===========================")
        for msg in state["messages"]:
            f.write(f"\n{type(msg)}\n{msg.content}\n")



def use_prompt(state, sys_prompt):
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
        tool_names=",".join([t.name for t in tools])
    )

    system_prompt="\n".join([prompt_react, sys_prompt])
    return [SystemMessage(system_prompt), *state["messages"]]

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
        f.write(str(state))
        f.write(f"\nlast_msg:\n{last_msg}\n")
        f.write(f"\nnew_msg:\n{new_msg}\n")

    
    return {
        "messages": [RemoveMessage(id=last_msg.id), new_msg]
    }
    


# System prompt
sys_prompt="Always reason step by step and call tools explicitly where needed.\nDo not attempt to perform any task yourself."
graph = create_react_agent(
    model=llm,
    tools=tools,
    # pre_model_hook=use_pre_hook,
    post_model_hook=use_post_hook,
    prompt=lambda state: use_prompt(state=state, 
                                    sys_prompt=sys_prompt)
)

