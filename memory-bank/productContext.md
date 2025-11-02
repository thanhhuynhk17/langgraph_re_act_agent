# Product Context - Why Aegra Exists

## The Problems Aegra Solves

### Vendor Lock-In
Proprietary AI platforms create dependency on external services, making it difficult to:
- Control cost scaling and infrastructure decisions
- Maintain data sovereignty and privacy
- Customize platforms beyond vendor-provided options
- Migrate between providers when needs change

### Self-Hosting Complexity
Existing open-source agent platforms are often:
- Difficult to deploy and maintain in production
- Missing essential features like persistence and authentication
- Incompatible with established client SDKs and UIs
- Require extensive custom development to achieve reliability

### Migration Pain
Transitioning from proprietary platforms involves:
- Rewriting client applications to use new APIs
- Rebuilding endpoint integrations
- Loss of existing conversation history and state
- Retraining teams on different interfaces

### Community Gap
The AI agent ecosystem lacks:
- Community-driven, production-ready alternatives
- Open-source projects that maintain API compatibility
- Examples of real-world agent implementations

## User Experience Vision

Aegra exists to eliminate these barriers by providing a **plug-and-play self-hosted alternative** that "just works" exactly like the proprietary platforms users know and love.

### Zero Migration User Journey
1. **Current State**: User runs LangGraph Platform with client apps using official SDK
2. **Discovery**: Finds Aegra as community alternative with identical API
3. **Migration**: Swaps platform URL, redeploys - client apps continue working unchanged
4. **Ownership**: Gains full self-hosted control while maintaining all existing functionality

### Development Experience Goals
- **5-minute setup**: Docker-based deployment matching current platform patterns
- **Drop-in compatibility**: Same LangGraph SDK usage, same Agent Chat UI integration
- **Familiar workflows**: PostgreSQL persistence, streaming responses, authentication
- **Easy customization**: JSON config for different agent graphs, extensible architecture

### Production Operations Goals
- **Enterprise ready**: Robust monitoring, health checks, database migrations
- **Developer friendly**: Hot reload, comprehensive testing, clear documentation
- **Scalable architecture**: Multi-tenant support, performance optimization foundation
- **Community driven**: Transparent development, contribution-friendly, Apache 2.0 licensed

## Target User Profiles

### IndividuAL Developers
- Want to experiment with agents without vendor lock-in
- Need production-quality persistence and authentication
- Seek community support rather than paid enterprise tiers

### Startups & Small Businesses
- Require AI agent capabilities for customer interaction
- Cannot afford enterprise vendor pricing
- Need self-hosted solutions for data compliance
- Want full control over their AI operations

### Enterprise Teams
- Built mature agent applications on LangGraph Platform
- Need to move to self-hosted for compliance or cost reasons
- Cannot rewrite applications for new platforms
- Require enterprise features (auth, monitoring, HA)

### Open Source Community
- Contributors seeking to build agent infrastructure
- Teams wanting to customize agent platforms
- Researchers needing self-hosted experimental environments
- Educators teaching agent development

Aegra serves all these users by providing the reliability and compatibility of proprietary platforms with the freedom and control of open-source software.
