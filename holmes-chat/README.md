# Holmes AG-UI Chat Interface

A React-based chat interface for Holmes GPT that provides real-time streaming, Prometheus graph visualization, and task tracking.

## Features

- **Real-time Chat**: Streaming responses from Holmes with live tool execution updates
- **Prometheus Integration**: 
  - Automatic graph rendering for Prometheus query results
  - Click-to-maximize graphs for detailed viewing
  - Direct links to open queries in Prometheus UI
  - Support for both range and instant queries
- **Task Tracking**: Live investigation task panel showing Holmes' progress
- **Tool Output Display**: Real-time visibility into Holmes' tool executions
- **Event Streaming**: Shows AG-UI event types for debugging

## Prerequisites

- Node.js and npm
- Holmes GPT server running on `http://localhost:5050` (with AG-UI support)
- Prometheus server running on `http://localhost:9090`
  - Note: same one for which HolmesGPT server is configured. Used for Prometheus UI links.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Open [http://localhost:5151](http://localhost:5151) to view the interface

## Usage

1. **Start Holmes Server**: Ensure Holmes GPT server is running with AG-UI support
2. **Ask Questions**: Type questions about your observability data
3. **View Graphs**: Prometheus queries automatically render as interactive charts
4. **Track Progress**: Watch investigation tasks update in real-time in the left panel
5. **Open in Prometheus**: Click the "🔥 Open in Prometheus" button on any graph

## Architecture

- **Frontend**: React app with Chart.js for graph rendering
- **Backend**: Holmes GPT server with AG-UI streaming endpoints
- **Communication**: Server-Sent Events (SSE) for real-time updates
- **Graph Data**: Automatic detection and rendering of Prometheus tool outputs

## Graph Features

- **Time-based Charts**: Proper time axis formatting with timezone display
- **Interactive**: Click to maximize, hover for detailed tooltips  
- **Prometheus Integration**: Direct links to query in Prometheus UI
- **Multiple Formats**: Supports both range queries (time series) and instant queries (single points)

## Development

The interface connects to Holmes GPT server endpoints:
- Chat: `POST /api/agui/chat` (streaming)
- Prometheus Data: `GET /api/prometheus-data/{key}`
- Debug: `GET /api/prometheus-keys`

## Available Scripts

### `npm start`
Runs the app in development mode on [http://localhost:5151](http://localhost:5151)

### `npm run build`
Builds the app for production to the `build` folder

### `npm test`
Launches the test runner in interactive watch mode
