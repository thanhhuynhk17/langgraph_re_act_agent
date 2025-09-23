# React Agent with Pre & Post Hooks

This project demonstrates how to use **LangGraph** with a React-based agent, applying **`pre_hook`** and **`post_hook`** to modify the agent's behavior before and after each action. 

![react_agent_mobile](https://github.com/user-attachments/assets/58c88fbc-8227-4f96-8182-f9043d25960f)

To run in development mode, use:
```cli
BG_JOB_ISOLATED_LOOPS=true langgraph dev --host 0.0.0.0 --allow-blocking
```

```.env
BG_JOB_ISOLATED_LOOPS=true
OPENAI_BASE_URL=https://4922c9699b63.ngrok-free.app/v1
TAVILY_API_KEY=
LOG_LEVEL=INFO

OPENAI_BASE_URL_EMBED=http://localhost:8080
OPENAI_API_KEY_EMBED=<your-api-key>
OPENAI_API_MODEL_NAME_EMBED=qwen3-embed
```

# Project Structure

```
.
├── .env
├── .env.example
├── README.md
├── langgraph.json
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── logs/
│   │   └── react.log
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py
│       ├── interrupt_any_tool.py
│       ├── logging_setup.py
│       ├── prompts.py
│       ├── react_constants.py
│       ├── schemas.py
│       └── tools.py
 
---

## 🔹 Hook Usage
- **`pre_hook`** *(after tool call, before LLM call)*  
  Executed after the tool returns but before the LLM processes the result.  
  In this setup, it:
  - Appends `<react_observation>...</react_observation>` to maintain consistent reasoning flow  
  - Handles tool outputs before passing them to the LLM  
  - Can also be used for input sanitization, context enrichment, or logging  

- **`post_hook`** *(after LLM call, before tool call)*  
  Executed after the LLM generates output but before any tool is called.  
  In this setup, it:
  - Parses tool calls and extracts arguments  
  - Manages interruptions in the reasoning flow  

---

## 📜 ReAct Agent Output Format
The agent follows a **strict, structured format** using XML-like tags.  
This ensures:
- Consistency in reasoning and responses  
- Easier debugging and transparency  
- `pre_hook` / `post_hook` can safely modify inputs & outputs  

### Standard Format
```text
<react_question>
The user’s input question.
</react_question>

<react_thought>
Your reasoning step.
</react_thought>

<react_action>
The action name (must be one of the predefined tools).
</react_action>

<react_action_input>
Valid JSON object for the action input.
</react_action_input>

<react_observation>
(This will be inserted by the system — DO NOT generate it yourself)
</react_observation>

... (this block can repeat multiple times) ...

<react_thought>
I now know the final answer.
</react_thought>

<react_final_answer>
The final answer to the user’s question (MUST be written in Vietnamese).
</react_final_answer>
```


# requirements.txt

```bash
pip install -r requirements.txt
```

restart

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


# server embedding

for gpu

```powershell 
(.venv) PS D:\llama-b6451-bin-win-cuda-12.4-x64> .\llama-server -m "C:\Users\lea26\Downloads\Qwen3-Embedding-0.6B-f16.gguf" --embedding --pooling last -ngl 99 -ub 8192 -c 32768 --threads 16 --threads-batch 16 --flash-attn on --host 0.0.0.0
```

## cURL test

```bash
curl -s https://6be81cf05f50.ngrok-free.app/v1/chat/completions -H "Content-Type: application/json" -d '{
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'

{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","reasoning_content":"Okay, the user sent \"Hello, world!\" which is a classic first program in many programming languages. But since they're interacting with me, the AI, I should respond appropriately. They might be testing if I'm working, or maybe they're new to AI interactions.\n\nFirst, I need to acknowledge their greeting. A simple \"Hello!\" would be good. Then, maybe ask how I can assist them. But I should keep it friendly and open-ended. Let me check if there's any deeper meaning here. Maybe they're quoting the programming phrase, but in this context, it's probably just a greeting.\n\nI should avoid being too technical unless they ask. Since it's a greeting, a warm response is best. Let me make sure to include a smiley to keep it friendly. So, \"Hello! 😊 How can I assist you today?\" That sounds good. Let me confirm there are no typos. Yep, looks good. Keep it short and approachable.","content":"Hello! 😊 How can I assist you today?"}}],"created":1758478262,"model":"gpt-3.5-turbo","system_fingerprint":"b6239-cd36b5e5","object":"chat.completion","usage":{"completion_tokens":215,"prompt_tokens":12,"total_tokens":227},"id":"chatcmpl-B4UovQZgPA4U0n4bnKIYpmRvTVfxdx0K","timings":{"prompt_n":9,"prompt_ms":221.349,"prompt_per_token_ms":24.59433333333333,"prompt_per_second":40.659772576338725,"predicted_n":215,"predicted_ms":4162.873,"predicted_per_token_ms":19.362199999999998,"predicted_per_second":51.647023582030975}}
```
