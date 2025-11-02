# System Patterns - Aegra's Architecture & Design

## System Architecture Overview

```
Client Applications
    ↓ (LangGraph SDK)
FastAPI Server (Agent Protocol)
    ↓ (Graph Management)
LangGraph Runtime
    ↓ (State Management)
PostgreSQL/SQLite (Persistence)
    ↙        ↘
MCP Tools   Local Tools
```

## Core Architectural Patterns

### 1. FastAPI + Agent Protocol Implementation
The backend follows a traditional web service architecture:
- **Controller Layer**: FastAPI routes translate Agent Protocol to LangGraph operations
- **Service Layer**: Business logic abstraction with dependency injection
- **Repository Layer**: Database models and data access via SQLAlchemy
- **Configuration Layer**: JSON-driven graph registration and environment variables

### 2. Graph-Centric Agent Management
Agent graphs serve as first-class citizens:
- **Registry Pattern**: `aegra.json` defines available graphs and their entry points
  ```
  {
    "graphs": {
      "re_act_agent": "./graphs/react_agent/agent.py:get_graph"
    }
  }
  ```
- **Factory Pattern**: Dynamic graph loading via import path resolution and function calls
- **Configuration-driven**: Graphs instantiated with runtime config and dependencies
- **Volume-mediated Loading**: Docker containers load graphs via mounted volumes for hot reload support

### 3. Hook System for Agent Customization
Pre and post processing hooks enable deep customization:
- **Chain of Responsibility**: Sequential message transformation
- **Observer Pattern**: Hooks observe and modify agent state transitions
- **Template Method**: Consistent processing pipeline across graph types

### 4. Multi-Protocol Tool Integration
Hybrid approach to tool management:
- **Adapter Pattern**: MCP tools wrapped as LangChain-compatible interfaces
- **Registry Pattern**: Unified tool collection management
- **Decorator Pattern**: Human-in-the-loop enhancements for any tool

## Key Design Decisions

### Vietnamese Restaurant Agent Example
**Why this specific domain?**
- Demonstrates real production use case beyond simplistic examples
- Shows cultural adaptation (Vietnamese language, business context)
- Illustrates complex workflows (order management, customer history)
- Proves multi-modal capabilities (text, structured data, vector search)

**Technical showcases:**
- Domain-specific prompt engineering
- Cultural context handling
- Business logic integration
- End-to-end user journey

### MCP-First Tool Architecture
**Benefits:**
- Extensibility without code changes
- Protocol-based tool discovery
- Server-side tool updates
- Vendor-neutral extensibility

**Implementation:**
- MultiServerMCPClient maintains persistent connections
- Tool loading failures handled gracefully
- Tool schemas dynamically processed for prompt generation

### ReAct Pattern with Custom Formatting
**Core Design:**
- Thought → Action → Observation → Final Answer cycle
- Vietnamese language optimized prompts
- Tool signature integration into system messages
- Output validation and retry logic

**Customization Points:**
- Tagging system for structured communication
- Message preprocessing for normalization
- Post-processing for response validation

### State Management Extensions
**Custom State Schema:**
```python
class CustomAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_name: Optional[str]
    user_uuid: Optional[str]
    json_data: Optional[Dict[str, Any]]
```

**Rationale:**
- User context persistence across sessions
- Structured data handling for business logic
- Message enrichment for UI features

## Component Relationships

### Graph Execution Flow
```
Graph Creation
    ↓
Tool Discovery (MCP + Local)
    ↓
Hook Setup (Pre/Post processors)
    ↓
LangGraph Compilation
    ↓
Runtime: Message → Process → Tool → Response
```

### Message Processing Pipeline
```
Input Message
    ↓ (Pre-hook - Tagging, Normalization)
    • Add <TAG_QUESTION>...<TAG_QUESTION> wrapper
    • Remove duplicate human messages
    • Normalize tool observations with <TAG_OBSERVATION> tags
    ↓
LLM Processing
    ↓ (Post-hook - Validation, Retries)
    • Parse ReAct format: <TAG_THOUGHT>, <TAG_ACTION>, <TAG_ACTION_INPUT>
    • Validate tool existence and parameters
    • Retry via Command(goto="pre_model_hook") on format violations
    ↓
Tool Execution
    ↓ (Observation Processing)
    • UI push via push_ui_message() for interactive feedback
    • Format tool results with observation tags
    ↓
Response Generation (Final Answer)
    ↓ (UI Push, State Update)
    • Structured <TAG_FINAL_ANSWER> formatting
    • State checkpointing and persistence
    ↓
Final Output
```

### Data Flow Patterns
```
Application Request
    ↓ (FastAPI Routes)
Service Coordination
    ↓ (Business Logic)
Graph Execution
    ↓ (State Transitions)
Tool Operations
    ↓ (External Calls)
Database Persistence
    ↓ (Checkpoint Storage)
Response Formation
```

## Critical Implementation Paths

### 1. Agent Initialization Sequence
**Path:** `get_graph() → MCP Connection → Tool Loading → Hook Setup → Graph Compilation`
- **Critical point**: MCP server availability determines feature set
- **Fallback**: Graceful degradation when MCP unavailable
- **Dependencies**: Network connectivity, server stability

### 2. Message Processing Pipeline
**Path:** `Human Message → Pre-hook → LLM Call → Post-hook → Tool Execution → UI Update`
- **Retry logic**: Post-hook can trigger reprocessing on validation failure
- **State management**: Messages persisted before tool execution
- **Error handling**: Network issues, tool failures, LLM errors

### 3. Human-in-the-Loop Integration
**Path:** `Tool Invocation → Interrupt Detection → User Confirmation → Continuation`
- **Interrupt points**: Configurable per tool via `add_human_in_the_loop`
- **UI integration**: State pushes for real-time user interaction
- **Resume logic**: Command pattern for continuation after interruption

### 4. Persistence and Recovery
**Path:** `Graph Execution → State Checkpoint → Database Write → Recovery Load`
- **Transaction boundaries**: State consistency during complex operations
- **Thread safety**: Checkpointing during concurrent executions
- **Migration support**: Alembic for schema evolution

## Design Pattern Instances

### Command Pattern
Used for graph operations and user interactions:
- `Command(goto="pre_model_hook")` for retry logic
- `RemoveMessage(id=msg_id)` for message management
- Interrupt handling for human-in-the-loop

### Observer Pattern
UI push notifications via push_ui_message():
- Tool execution results pushed to frontend
- Real-time updates during streaming
- Decoupled UI update mechanism

### Strategy Pattern
Multiple agent behaviors:
- Vietnamese restaurant agent vs. general-purpose agents
- Different prompt strategies for different domains
- Configurable tool sets per agent type

### Factory Pattern
Dynamic graph and tool creation:
- Graph factory via `aegra.json` configuration
- Tool factory through MCP discovery
- State factory with custom schema construction

These patterns ensure Aegra remains extensible and maintainable while demonstrating sophisticated agent capabilities through the restaurant chatbot example.
