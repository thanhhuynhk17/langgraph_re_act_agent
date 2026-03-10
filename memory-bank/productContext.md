# Product Context: React Agent with Pre & Post Hooks

## Why This Project Exists
This project exists to demonstrate and implement a sophisticated agent architecture that can handle complex reasoning tasks with structured communication. The need for pre and post hook functions addresses common challenges in agent development where raw tool outputs need to be processed before reasoning and tool calls need to be parsed after reasoning.

## Problems It Solves
1. **Inconsistent Agent Communication**: Without structured formats, agents produce unpredictable outputs that are hard to debug and maintain.
2. **Tool Output Processing**: Raw tool results often need transformation before they can be used in reasoning.
3. **Tool Call Parsing**: Generated text needs to be parsed to extract actual tool calls and parameters.
4. **Reasoning Flow Management**: Complex tasks require multiple reasoning steps that need to be tracked and managed.
5. **Language Support**: Providing final answers in specific languages (Vietnamese) requires consistent formatting.

## How It Should Work
The agent follows a strict, structured format using XML-like tags:
- Each interaction starts with a user question wrapped in <react_question>
- The agent performs reasoning steps wrapped in <react_thought>
- Tool calls are made using <react_action> and <react_action_input> tags
- Tool results are inserted by the system in <react_observation> tags
- The process repeats until the agent has enough information
- Final answers are provided in <react_final_answer> tags in Vietnamese

## User Experience Goals
- **Transparency**: Users can see the complete reasoning process
- **Consistency**: All interactions follow the same structured format
- **Reliability**: Pre and post hooks ensure proper tool handling
- **Maintainability**: Structured format makes debugging and updates easier
- **Localization**: Support for Vietnamese language in final responses

## Technical Benefits
- Easier debugging through structured logging
- Better error handling with pre and post processing
- Modular architecture that can be extended
- Clear separation between reasoning and tool execution
- Support for complex multi-step reasoning tasks