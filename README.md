# React Agent with Pre & Post Hooks

![react_agent_mobile](https://github.com/user-attachments/assets/58c88fbc-8227-4f96-8182-f9043d25960f)

This project demonstrates how to use **LangGraph** with a React-based agent, applying **`pre_hook`** and **`post_hook`** to modify the agent's behavior before and after each action.

## 🔹 Hook Usage
- **`pre_hook`** *(after tool call, before LLM call)*  
  Executed after the tool returns but before the LLM processes the result.  
  In my setup, it:
  - Adds `[react_observation]` to maintain consistent instruction-following  
  - Handles tool artifacts before they are passed to the LLM  
  - Can still be used for input sanitization, adding context, or logging

- **`post_hook`** *(after LLM call, before tool call)*  
  Executed after the LLM produces output but before any tool is called.  
  In my setup, it:
  - Handles tool call parsing & argument extraction  
  - Manages interruptions in the reasoning flow  

---

## 📜 ReAct Agent Output Format
My agent uses a structured output format to make its reasoning and tool usage transparent and machine-readable.  
Each response follows these tags in order:

[react_question]: <user question>  
[react_thought]: <high-level reasoning>  
[react_action]: <tool_name_or_"none">  
[react_action_input]: <valid JSON or "{}">  
[react_observation]: <tool result or "none">  
[react_thought]: <updated reasoning>  
[react_final_answer]: <final answer>  

This format keeps the workflow consistent, helps debugging, and makes it easy for `pre_hook` / `post_hook` to adjust inputs and outputs.
