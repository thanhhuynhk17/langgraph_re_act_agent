# Technical Context - Aegra's Technology Stack

## Core Technologies

### Backend Framework
- **FastAPI**: Modern Python web framework for high-performance APIs
- **Python 3.11+**: Minimum version requirement due to async features and performance improvements
- **uv**: Fast Python package installer and resolver for dependency management

### Agent Framework
- **LangGraph**: State management and graph execution for complex agent workflows
- **LangChain**: Integration with language models and external services
- **LangChain Core**: Fundamental abstractions and utilities
- **langgraph-api**: HTTP API layer for LangGraph graphs
- **langgraph-checkpoint-postgres**: PostgreSQL-based persistent checkpoints
- **langgraph-cli**: Command-line tools for development and deployment

### Data Persistence
- **PostgreSQL**: Primary database for conversation state, metadata, and checkpoints
- **SQLAlchemy**: Python SQL toolkit and ORM for database operations
- **Alembic**: Database migration tool with auto-generation capabilities
- **SQLite**: Local development and testing (via sqlite-vec extension for vector search)

### Authentication & Security
- **PyJWT**: JSON Web Token implementation for authentication
- **Custom auth framework**: Extensible authentication middleware ready for OAuth/OpenID

### AI/ML Components
- **OpenAI API**: Primary LLM provider with configurable models and endpoints
- **Vietnamese NLP**: underthesea library for Vietnamese language processing
- **Vector Search**: FAISS CPU for similarity search, sqlite-vec for SQL-based vectors
- **Sentence Transformers**: Text embedding generation for semantic search
- **Torch ecosystem**: PyTorch, TorchVision, TorchAudio for ML model inference

### Tool Integration
- **MCP (Model Context Protocol)**: Custom langchain-mcp-adapters for external tool integration
  - Server location: `http://localhost:8000/mcp`
  - MultiServerMCPClient for persistent connections
  - Runtime tool discovery and schema processing
  - Graceful fallback when MCP server unavailable
- **Graphiti Core**: Knowledge graph and memory management system
- **LangChain MCP Adapters**: Git dependency for MCP compatibility
- **LangChain Tavily**: Web search integration
- **LangChain HuggingFace**: HuggingFace model integration

### Observability & Monitoring
- **Comprehensive Logging**: Environment-controlled logging (DEBUG/INFO/WARNING/ERROR/CRITICAL) across all agent components
- **Langfuse**: Tracing and monitoring integration
- **AsyncPG**: High-performance PostgreSQL driver for async operations
- **HTTpx**: Modern async HTTP client for external API calls

## Development Setup

### Local Development
```bash
# Package management
pip install uv
uv pip install -e .

# Database setup
docker compose up postgres -d
python scripts/migrate.py upgrade

# Run server
python run_server.py
```

### Docker Development
```bash
# Full stack (DB + migrations + server)
docker compose up aegra
```

**Docker Compose Configuration:**
- **PostgreSQL Service**: Persistent data, health checks, environment variables
- **Aegra Service**: Volume-mounted graphs, hot reload support, migration execution
- **Redis (Optional)**: Profile-activated advanced queuing capabilities
- **Volume Mounting**: Graph configs, source code, environment files, database migrations

### Testing Setup
- **pytest**: Comprehensive test framework with asyncio support
- **pytest-asyncio**: Async test support
- **pytest-cov**: Code coverage reporting
- **e2e tests**: End-to-end testing for critical paths

## Technical Constraints

### Runtime Requirements
- **Python 3.11 minimum**: Required for modern async features and type annotations
- **API Keys**: OpenAI API key mandatory for LLM functionality
- **External Services**: MCP server must be running on localhost:8000/mcp for tool access
- **PostgreSQL**: Production deployments require PostgreSQL for persistence

### Environmental Factors
- **Container-ready**: Full Docker containerization for consistent deployments
- **Cross-platform**: Works on Windows/macOS/Linux with platform-specific considerations
- **Memory constraints**: Vector search components (FAISS, transformers) require adequate RAM
- **Network dependencies**: External API calls for OpenAI, MCP server connectivity

### Platform Limitations
- **Language focus**: Custom Vietnamese language support (underthesea) may not extend to other languages
- **Graph complexity**: LangGraph recursive limit (99) bounds maximum agent interaction depth
- **Checkpoint storage**: PostgreSQL-only production persistence (no other databases supported)

### Development Considerations
- **Git dependencies**: langchain-mcp-adapters from custom GitHub repo requires stable upstream
- **Model dependencies**: GPU support optional but beneficial for local transformer inference
- **Migration complexity**: Alembic auto-generation may need manual review for complex schema changes

## Tool Usage Patterns

### Primary Development Tools
- **VS Code**: Primary IDE with Python, YAML, and Markdown support
- **Black + isort**: Code formatting with 88-character line length
- **MyPy**: Type checking with strict settings
- **pre-commit hooks**: Automatic formatting and linting on commits

### Key Dependencies Chain
```
LLM APIs (OpenAI) → LangChain → LangGraph → FastAPI → PostgreSQL/SQLite
                                    ↓
                         MCP Tools ← langchain-mcp-adapters
                                    ↓
                       Vietnamese NLP ← underthesea
                                    ↓
                   Vector Search ← FAISS/sqlite-vec + sentence-transformers
```

This technology stack enables Aegra to deliver production-ready self-hosted agent capabilities while maintaining compatibility with existing LangGraph Platform workflows.
