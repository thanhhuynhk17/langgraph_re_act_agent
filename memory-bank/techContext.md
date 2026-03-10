# Tech Context: React Agent Development

## Technologies Used

### Core Framework
- **LangGraph**: Main agent orchestration framework
- **Python**: Primary implementation language
- **React-based Agent Architecture**: Structured agent design pattern

### Development Tools
- **Visual Studio Code**: Primary IDE
- **Docker**: Containerization for development environment
- **Git**: Version control system
- **Bash/Shell**: Command line interface

### External APIs
- **OpenAI**: LLM services for agent reasoning
- **Tavily**: Search and information retrieval
- **Custom Embedding Server**: Vector embeddings for semantic search

## Development Setup

### Environment Configuration
```bash
# Environment variables
BG_JOB_ISOLATED_LOOPS=true
OPENAI_BASE_URL=https://4922c9699b63.ngrok-free.app/v1
TAVILY_API_KEY=your-api-key
LOG_LEVEL=INFO

# Embedding configuration
OPENAI_BASE_URL_EMBED=http://localhost:8080
OPENAI_API_KEY_EMBED=your-embedding-key
OPENAI_API_MODEL_NAME_EMBED=qwen3-embed
```

### Running the Application
```bash
# Development mode
BG_JOB_ISOLATED_LOOPS=true langgraph dev --host 0.0.0.0 --allow-blocking

# Docker setup
docker-compose up
```

### Dependencies
- **torch**: PyTorch for machine learning
- **torchvision**: Computer vision library
- **torchaudio**: Audio processing library
- **langgraph**: Agent framework
- **langchain**: Language model integration

## Technical Constraints

### Performance Considerations
- **GPU Support**: Required for embedding server
- **Memory Usage**: Agent state management
- **Network Latency**: External API calls
- **Processing Time**: Reasoning and tool execution

### Compatibility Requirements
- **Python Version**: Must be compatible with LangGraph
- **OS Support**: Windows 11 (current environment)
- **Browser Support**: For React-based interfaces
- **API Rate Limits**: External service constraints

## File Structure

```
src/
├── agent.py          # Main agent implementation
├── utils/            # Utility functions
│   ├── helpers.py
│   ├── logging_setup.py
│   ├── prompts.py
│   ├── react_constants.py
│   ├── schemas.py
│   └── tools.py
├── logs/             # Log files
└── data/             # Data storage
```

## Development Workflow

### Code Organization
- **Modular Design**: Separate concerns into different files
- **Consistent Formatting**: Follow project standards
- **Error Handling**: Comprehensive exception management
- **Logging**: Structured logging for debugging

### Testing Approach
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Response time and resource usage
- **Error Scenario Testing**: Failure mode handling

## Deployment Considerations

### Local Development
- **Hot Reload**: Automatic code reloading
- **Debug Mode**: Enhanced error reporting
- **Development Logs**: Detailed logging for troubleshooting

### Production Deployment
- **Environment Variables**: Secure configuration management
- **Resource Limits**: CPU and memory constraints
- **Monitoring**: Performance and error tracking
- **Backup**: Data persistence and recovery

## Security Considerations

### API Security
- **Key Management**: Secure storage of API keys
- **Request Validation**: Input sanitization
- **Rate Limiting**: Prevent abuse of external services
- **Data Privacy**: Handling of sensitive information

### Code Security
- **Dependency Management**: Regular updates and vulnerability scanning
- **Input Validation**: Prevent injection attacks
- **Error Handling**: Avoid information leakage
- **Access Control**: Proper authentication and authorization