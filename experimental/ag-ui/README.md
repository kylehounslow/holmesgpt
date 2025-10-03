# AG-UI - Experimental HolmesGPT Extension

AG-UI is an experimental extension to HolmesGPT that demonstrates [AG-UI](https://docs.ag-ui.com/introduction) capabilities through a specialized `/api/agui/chat` endpoint and a web-based demo interface. The AG-UI compatible `server.py` is adapted from the [existing server.py implementation](../../server.py)

*⚠️ **Disclaimer**: AG-UI is experimental within HolmesGPT. APIs and interfaces may change as the project evolves. The demonstration server and frontend is not intended for production use.*


## 🛠️ Quick Start

### **Prerequisites**
- **HolmesGPT** instance with AG-UI extensions enabled
- **Data Sources**: Prometheus (`:9090`) and/or OpenSearch (`:9200`)
   - Suggest to run [opentelemetry-demo](https://github.com/open-telemetry/opentelemetry-demo) via docker-compose. 
- **Node.js** 20+ (for frontend demonstration)

### **0. Set up datasources**
e.g. [opentelemetry-demo](https://github.com/open-telemetry/opentelemetry-demo) 
```
cd opentelemetry-demo
docker compose up -d
```

### **1. Start HolmesGPT AG-UI Server on port 5050**

```bash
# Start HolmesGPT AG-UI compatible server
cd holmesgpt
export HOLMES_PORT=5050
poetry run python experimental/ag-ui/server.py
```

### **2. Run Demo Frontend**
Create .env file at `experimental/ag-ui/front-end/.env`. Example below. Replace Prometheus/OpenSearch urls as needed
```
# AG-UI Agent Configuration
HOLMES_PORT=5050
AGENT_URL=http://localhost:${HOLMES_PORT}
# Prometheus Configuration
REACT_APP_PROMETHEUS_URL=http://localhost:9090

# OpenSearch Configuration
REACT_APP_OPENSEARCH_URL=http://localhost:9200
REACT_APP_OPENSEARCH_USER=user
REACT_APP_OPENSEARCH_PASSWORD=pass
```
```bash
cd experimental/ag-ui/front-end
npm install && npm start
```
