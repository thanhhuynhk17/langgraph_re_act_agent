# Aegra - Open Source LangGraph Platform Alternative

## Project Overview
Aegra is an open-source, self-hosted alternative to LangGraph Platform, providing a community-driven solution for AI agent infrastructure. The project delivers identical functionality to LangGraph Platform through a drop-in replacement SDK, enabling developers to maintain full control over their agent backends while avoiding vendor lock-in.

## Core Requirements
- **Self-hosted First**: Complete control over data and infrastructure with zero external dependencies for core functionality
- **SDK Compatibility**: Drop-in replacement for LangGraph Client SDK without code changes
- **Production Ready**: Robust persistence, streaming capabilities, authentication, and monitoring
- **Agent Protocol Compliance**: Full implementation of the open-source Agent Protocol specification
- **UI Compatibility**: Seamless integration with LangChain's Agent Chat UI

## Core Goals
1. Enable self-hosted AI agent deployments with professional-grade reliability
2. Provide zero-migration paths for existing LangGraph Platform users
3. Deliver production-ready features: PostgreSQL persistence, real-time streaming, authentication frameworks
4. Establish a welcoming community-driven alternative to proprietary platforms
5. Demonstrate practical agent capabilities through working examples

## Current Implementation
- FastAPI-based HTTP API server implementing Agent Protocol
- PostgreSQL integration with LangGraph checkpointing for persistent conversation state
- Authentication framework foundation (OAuth/JWT ready)
- Config-driven graph management via JSON configuration
- Docker containerization for easy deployment
- Comprehensive test suite and development tooling

## Example Agent
The project includes a production Vietnamese restaurant chatbot agent that demonstrates:
- ReAct pattern implementation for complex reasoning
- Real-world customer interaction handling
- Order management and customer history integration
- Vietnamese language processing and cultural adaptation
- Tool integration via MCP (Model Context Protocol)
- Human-in-the-loop capabilities

This example serves as both a development reference and proof-of-concept for Aegra's capabilities in handling sophisticated agent workflows.
