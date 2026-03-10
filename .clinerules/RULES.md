# Development Rules for React Agent Project

## 0. Working Methodology

### 0.1 React Agent Development
- **USE structured XML-like format** for all agent communication
- Always follow the exact tag structure:
  - <react_question> for user questions
  - <react_thought> for reasoning steps
  - <react_action> for tool calls
  - <react_action_input> for tool parameters
  - <react_observation> for tool results
  - <react_final_answer> for final responses (in Vietnamese)

### 0.2 Hook Function Development
- **Pre_hook**: Process tool outputs before LLM reasoning
- **Post_hook**: Parse tool calls after LLM reasoning
- Both hooks must maintain reasoning context
- Error handling must be robust in both directions

### 0.3 Testing Requirements
- Test hook functions with various tool combinations
- Validate XML-like format parsing
- Test Vietnamese language output
- Verify tool call extraction accuracy

## 1. Code Organization

### 1.1 File Structure
- **agent.py**: Main agent implementation
- **utils/**: Utility functions and helpers
- **hooks/**: Pre and post hook implementations
- **tools/**: Tool definitions and integrations

### 1.2 Hook Implementation
- Keep pre_hook focused on output processing
- Keep post_hook focused on tool call extraction
- Handle edge cases and errors gracefully
- Maintain performance and efficiency

## 2. XML-like Format Rules

### 2.1 Tag Requirements
- Must use exact tag names (case-sensitive)
- Proper nesting and structure required
- Vietnamese language for final answers
- JSON format for tool inputs

### 2.2 Format Validation
- Validate tag structure before processing
- Handle malformed XML gracefully
- Provide meaningful error messages
- Log format violations for debugging

## 3. Hook Function Guidelines

### 3.1 Pre_hook Rules
- Process tool outputs before LLM reasoning
- Append <react_observation> tags
- Handle tool failures gracefully
- Maintain reasoning context

### 3.2 Post_hook Rules
- Parse tool calls after LLM reasoning
- Extract tool names and parameters
- Validate tool call structure
- Handle interruptions in reasoning flow

## 4. Tool Integration

### 4.1 Tool Requirements
- Tools must return consistent output formats
- Tool inputs must be properly validated
- Error handling must be comprehensive
- Performance must be optimized

### 4.2 Tool Call Parsing
- Use JSON format for tool inputs
- Validate tool parameters before execution
- Handle tool failures gracefully
- Log tool execution for debugging

## 5. Error Handling

### 5.1 Hook Errors
- Catch and handle hook function exceptions
- Provide meaningful error messages
- Log errors for debugging
- Maintain agent state on errors

### 5.2 Tool Errors
- Handle tool execution failures
- Provide fallback responses when possible
- Log tool errors for analysis
- Maintain agent reasoning flow

## 6. Performance Guidelines

### 6.1 Hook Performance
- Keep hook functions efficient
- Avoid blocking operations in hooks
- Cache results when appropriate
- Monitor hook execution time

### 6.2 Tool Performance
- Optimize tool execution time
- Use asynchronous operations when possible
- Cache tool results when appropriate
- Monitor tool performance

## 7. Security Considerations

### 7.1 API Security
- Secure storage of API keys
- Request validation and sanitization
- Rate limiting for external services
- Data privacy and compliance

### 7.2 Code Security
- Input validation to prevent injection
- Error handling to avoid information leakage
- Access control for sensitive operations
- Regular security updates and patches

## 8. Testing Requirements

### 8.1 Hook Testing
- Test with various tool combinations
- Validate XML-like format parsing
- Test error handling scenarios
- Verify Vietnamese language output

### 8.2 Tool Testing
- Test tool execution with valid inputs
- Test error handling with invalid inputs
- Verify tool performance under load
- Test tool integration with hooks

## 9. Development Workflow

### 9.1 Implementation Steps
1. Implement hook functions
2. Test hook functionality
3. Implement tool integrations
4. Test tool execution
5. Validate XML-like format
6. Test end-to-end workflows

### 9.2 Debugging Guidelines
- Use structured logging for debugging
- Test hook functions in isolation
- Validate XML-like format parsing
- Monitor hook and tool performance