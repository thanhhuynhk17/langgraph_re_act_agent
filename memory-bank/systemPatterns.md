# System Patterns: React Agent Architecture

## Architecture Overview
The system follows a layered architecture with clear separation of concerns:
- **Agent Layer**: Main orchestration using LangGraph
- **Hook Layer**: Pre and post processing functions
- **Tool Layer**: External tool integrations
- **Communication Layer**: Structured XML-like format for agent responses

## Key Technical Decisions

### Pre and Post Hook Functions
- **Pre_hook**: Executed after tool returns but before LLM processing
  - Appends <react_observation> tags to maintain reasoning flow
  - Handles tool outputs before passing to LLM
  - Can be used for input sanitization, context enrichment, or logging
- **Post_hook**: Executed after LLM call but before tool execution
  - Parses tool calls and extracts arguments
  - Manages interruptions in reasoning flow
  - Ensures tool calls are properly formatted

### Structured Communication Format
The agent uses a strict XML-like format to ensure consistency:
```xml
<react_question>
User's input question
</react_question>

<react_thought>
Agent's reasoning step
</react_thought>

<react_action>
Tool name (predefined)
</react_action>

<react_action_input>
JSON object for tool input
</react_action_input>

<react_observation>
Tool result (system-inserted)
</react_observation>

<react_final_answer>
Final answer in Vietnamese
</react_final_answer>
```

## Design Patterns

### React Pattern
- **State Management**: Agent maintains state through XML tags
- **Action Selection**: Tools are selected based on reasoning context
- **Observation Integration**: Tool results are integrated into reasoning
- **Final Answer Generation**: Consolidated response in target language

### Hook Pattern
- **Pre-processing**: Modify inputs before LLM processing
- **Post-processing**: Parse outputs before tool execution
- **Error Handling**: Manage failures in both directions
- **Context Management**: Maintain consistent reasoning context

## Component Relationships

```
User Input → Pre_hook → LLM → Post_hook → Tool Execution
    ↓              ↓          ↓          ↓            ↓
<react_question> → <react_observation> → <react_action> → Tool Result
```

## Technical Constraints

### XML-like Format Requirements
- Must use exact tag names (case-sensitive)
- Proper nesting and structure required
- Vietnamese language for final answers
- JSON format for tool inputs

### Hook Function Requirements
- Pre_hook must handle tool outputs gracefully
- Post_hook must parse tool calls accurately
- Both hooks must maintain reasoning context
- Error handling must be robust

## Implementation Guidelines

### Agent Development
- Follow the structured format strictly
- Use pre_hook for output transformation
- Use post_hook for tool call parsing
- Test with various tool combinations

### Hook Development
- Keep pre_hook focused on output processing
- Keep post_hook focused on tool call extraction
- Handle edge cases and errors gracefully
- Maintain performance and efficiency

### Tool Integration
- Tools must return consistent output formats
- Tool inputs must be properly validated
- Error handling must be comprehensive
- Performance must be optimized