# Project Brief: React Agent with Pre & Post Hooks

## Project Overview
This is a LangGraph-based React agent project that demonstrates the use of pre_hook and post_hook functions to modify agent behavior before and after each action. The project showcases how to implement structured reasoning with XML-like tags for consistent agent responses.

## Core Requirements
- Implement a React agent using LangGraph framework
- Add pre_hook function to modify behavior after tool calls but before LLM processing
- Add post_hook function to parse tool calls and manage interruptions
- Use structured XML-like tags for agent communication:
  - <react_question> for user questions
  - <react_thought> for reasoning steps
  - <react_action> for tool calls
  - <react_action_input> for tool parameters
  - <react_observation> for tool results (system-inserted)
  - <react_final_answer> for final responses (in Vietnamese)

## Technical Stack
- LangGraph for agent orchestration
- React-based agent architecture
- Python for implementation
- XML-like structured communication format
- Pre and post hook functions for behavior modification

## Key Features
- Structured reasoning flow with consistent formatting
- Tool call management with pre and post processing
- Vietnamese language support for final answers
- Development mode with isolated loops
- Integration with external APIs (OpenAI, Tavily)

## Success Criteria
- Agent follows the structured format consistently
- Pre_hook and post_hook functions work as intended
- Tool calls are properly parsed and executed
- Final answers are provided in Vietnamese
- Development environment runs without errors