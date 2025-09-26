# AG-UI App

A React TypeScript application with integrated AG-UI chat assistant.

## Features

- **Split Layout**: Main content area on the left, chat assistant on the right
- **AG-UI Integration**: Uses HttpAgent for real-time communication with AI agents
- **Event Streaming**: Supports real-time message streaming and tool calls
- **TypeScript**: Full type safety with AG-UI SDK types

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Configure your agent endpoint:
   ```bash
   cp .env.example .env
   # Edit .env to set your REACT_APP_AGENT_URL
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Configuration

The app expects an AG-UI compatible agent service running at the URL specified in `REACT_APP_AGENT_URL`. The agent should:

- Accept POST requests with `RunAgentInput` payload
- Return Server-Sent Events (SSE) stream
- Follow the Agent User Interaction Protocol

## AG-UI Integration

The chat assistant uses:

- **HttpAgent**: For HTTP-based agent communication
- **AgentSubscriber**: For handling streaming events
- **Event Types**: TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, TEXT_MESSAGE_END, etc.

See the [AG-UI documentation](https://docs.ag-ui.com) for more details.