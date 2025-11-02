# Active Context - Current Project State

## Current Work Focus

### Vietnamese Restaurant Chatbot Agent
The primary development focus is the `re_act_agent` graph, a production-ready chatbot for the Vietnamese restaurant "Cơm Quê Dượng Bầu" that demonstrates:

**Current Capabilities Implemented:**
- ReAct pattern with Vietnamese-optimized prompts and cultural adaptation
- Customer order management with history retrieval
- Menu browsing and recommendation system using Vietnamese dish categories
- Customer profile enrichment with order analytics
- Real-time datetime handling for reservation booking
- Vector-based semantic search for menu items
- Human-in-the-loop capabilities for sensitive operations

**Technical Integration Points:**
- MCP server integration for extensible tooling
- SQLite vector database for local semantic search
- Vietnamese NLP processing with underthesea
- Custom state management with user context persistence
- Comprehensive prompt engineering for restaurant domain

## Recent Changes & Context

### Latest Development Activity
- **Enhanced Customer Profile Enrichment**: Expanded `enrich_customer_prompt` function with detailed order statistics including orders in last 30/90 days, average order value, total dishes ordered, and most ordered dish
- **Zero-Order Customer Handling**: Added appropriate messaging for customers with no order history ("Chưa có đơn hàng")
- **Code Quality Improvements**: Implemented consistent logging infrastructure across all agent components
- **Error Handling Enhancement**: Replaced print() statements with proper logging in helpers, database operations, and agent hooks
- **Observability Improvements**: Enhanced debugging capabilities with structured logging at DEBUG/INFO/WARNING/ERROR levels
- **Refactor/dev-prod-structure**: Current branch indicates restructuring for production deployment
- **Last Commit**: 77068fd8d657393c8880346ab7f5a494f7fe7449 (refactor/dev-prod-structure)
- **Active Working Files**: Focus on agent utilities, database operations, and MCP integration

### Immediate Development Context
- **Agent Complexity**: Production-level Vietnamese chatbot with business logic
- **Tool Ecosystem**: MCP-first architecture with custom tool development
- **State Management**: Custom schemas handling user context and structured data
- **Cultural Adaptation**: Vietnamese language prompts, timezone handling, business etiquette

## Next Steps & Priorities

### Short-term Goals (Next 2-4 weeks)
1. **Production Testing**: Complete end-to-end testing of restaurant chatbot scenarios
2. **Documentation Updates**: Update README with comprehensive agent capabilities
3. **Performance Optimization**: Profile and optimize MCP connection handling
4. **Error Handling**: Improve graceful degradation when MCP server unavailable

### Medium-term Goals (Next 1-3 months)
1. **Multi-Agent Support**: Enable multiple concurrent agent instances
2. **Custom HTTP Endpoints**: Implement custom endpoints beyond Agent Protocol
3. **Redis-backed Streaming**: Upgrade streaming buffers for better performance
4. **Generative UIs**: Add capabilities for dynamic interface generation

### Long-term Vision
1. **Platform Extensions**: Advanced deployment recipes (Kubernetes, load balancing)
2. **Multi-language Support**: Expand beyond Vietnamese to other languages
3. **Custom UI Themes**: Aegra-branded interface components
4. **CLI Tooling**: Aegra CLI for migration and image building

## Active Decisions & Trade-offs

### MCP-First Architecture
**Decision**: Prioritize MCP (Model Context Protocol) for tool integration
- **Rationale**: Enables extensibility without code changes, protocol-based updates
- **Trade-off**: Additional complexity from managing MCP server dependencies
- **Status**: Successfully implemented with graceful fallback

### Vietnamese-Specific Implementation
**Decision**: Build production Vietnamese restaurant agent as primary example
- **Rationale**: Demonstrates real-world complexity beyond toy examples
- **Trade-off**: Domain-specific vs. general-purpose agent framework
- **Mitigation**: Clear separation between domain logic and core platform

### Hook System for Customization
**Decision**: Implement pre/post processing hooks for agent behavior
- **Rationale**: Enables deep customization of agent interactions without graph changes
- **Trade-off**: Additional complexity in message processing pipeline
- **Status**: Successfully enabling Vietnamese prompt engineering and UI integration

### Git Dependency Management
**Decision**: Use Git-based dependency for langchain-mcp-adapters
- **Rationale**: Access to cutting-edge MCP integration features
- **Risk**: Dependency on external GitHub repository stability
- **Monitoring**: Regular updates to track upstream changes

## Important Patterns & Insights

### Vietnamese Language Processing Patterns
- **Normalized Communication**: Consistent "em"/"anh chị" address for restaurant context
- **Business Logic Integration**: Order history and menu knowledge in prompts
- **Cultural Adaptation**: Vietnamese timezone handling, polite communication styles
- **Character Encoding**: Proper UTF-8 handling throughout the pipeline

### Message Processing Pipeline
- **Tagging System**: Structured `<TAG_QUESTION>`, `<TAG_THOUGHT>`, `<TAG_ACTION>` markers
- **Validation Logic**: Post-hook processing to catch LLM hallucinations
- **Retry Mechanisms**: Automated retry on format violations or tool errors
- **UI Integration**: Real-time push updates for tool execution results

### Tool Integration Patterns
- **Dynamic Tool Loading**: Runtime MCP tool discovery with schema processing
- **Human-in-the-Loop**: Configurable interruption points for sensitive operations
- **Fallback Behavior**: Graceful operation when external services unavailable
- **Schema Generation**: Automatic prompt generation from tool specifications

### State Management Insights
- **Context Persistence**: User metadata maintained across conversation sessions
- **Structured Data Handling**: JSON data fields for business logic integration
- **Memory Constraints**: Efficient checkpointing and message management
- **Concurrency Handling**: Thread-safe operations in multi-user environments

## Project Insights & Learnings

### Technical Learnings
1. **Vietnamese NLP Integration**: underthesea library successfully handles Vietnamese text processing
2. **MCP Protocol Benefits**: Protocol-based tool integration reduces coupling and enables runtime updates
3. **Hook System Power**: Pre/post processing hooks enable sophisticated behavior customization
4. **Vector Search for Vietnamese**: BM25 + semantic search effective for menu item retrieval

### Design Learnings
1. **Domain-Specific Agents**: Concrete use cases drive better architecture than abstract frameworks
2. **Cultural Context Matters**: Language and cultural adaptation critical for production agents
3. **Extensibility Through Protocols**: MCP approach scales better than hardcoded tool registration
4. **UI Integration Importance**: Real-time updates transform user experience quality

### Development Process Insights
1. **Test-Driven Development**: Comprehensive e2e tests ensure stability across agent interactions
2. **FastAPI Agent Protocol**: Robust foundation for API-compliant agent services
3. **Database Migration Maturity**: Alembic provides reliable production schema evolution
4. **Containerization Importance**: Docker ensures consistent deployment across environments

This active context ensures that future development maintains focus on production-ready agent capabilities while building toward broader platform extensibility.
