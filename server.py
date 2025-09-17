# ruff: noqa: E402
import json
import os
import random
import re
from typing import List, Optional

import litellm
import sentry_sdk
from holmes import get_version, is_official_release
from holmes.utils.cert_utils import add_custom_certificate

ADDITIONAL_CERTIFICATE: str = os.environ.get("CERTIFICATE", "")
if add_custom_certificate(ADDITIONAL_CERTIFICATE):
    print("added custom certificate")

# DO NOT ADD ANY IMPORTS OR CODE ABOVE THIS LINE
# IMPORTING ABOVE MIGHT INITIALIZE AN HTTPS CLIENT THAT DOESN'T TRUST THE CUSTOM CERTIFICATE
from holmes.core import investigation
from holmes.utils.holmes_status import update_holmes_status_in_db
import logging
import uvicorn
import colorlog
import time

from litellm.exceptions import AuthenticationError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from holmes.utils.stream import stream_investigate_formatter, stream_chat_formatter
from holmes.common.env_vars import (
    HOLMES_HOST,
    HOLMES_PORT,
    HOLMES_POST_PROCESSING_PROMPT,
    LOG_PERFORMANCE,
    SENTRY_DSN,
    ENABLE_TELEMETRY,
    DEVELOPMENT_MODE,
    SENTRY_TRACES_SAMPLE_RATE,
)
from holmes.config import Config
from holmes.core.conversations import (
    build_chat_messages,
    build_issue_chat_messages,
    build_workload_health_chat_messages,
)
from holmes.core.models import (
    FollowUpAction,
    InvestigationResult,
    InvestigateRequest,
    WorkloadHealthRequest,
    ChatRequest,
    ChatResponse,
    IssueChatRequest,
    WorkloadHealthChatRequest,
    workload_health_structured_output,
)
from holmes.core.investigation_structured_output import clear_json_markdown
from holmes.plugins.prompts import load_and_render_prompt
from holmes.utils.holmes_sync_toolsets import holmes_sync_toolsets_status
from holmes.utils.global_instructions import add_global_instructions_to_user_prompt

# AG-UI imports
from ag_ui.core import (
    RunAgentInput,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    TextMessageChunkEvent,
)
from ag_ui.encoder import EventEncoder
import uuid

# Store for Prometheus data by random_key
prometheus_data_store = {}


def init_logging():
    logging_level = os.environ.get("LOG_LEVEL", "INFO")
    logging_format = "%(log_color)s%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s"
    logging_datefmt = "%Y-%m-%d %H:%M:%S"

    print("setting up colored logging")
    colorlog.basicConfig(
        format=logging_format, level=logging_level, datefmt=logging_datefmt
    )
    logging.getLogger().setLevel(logging_level)

    httpx_logger = logging.getLogger("httpx")
    if httpx_logger:
        httpx_logger.setLevel(logging.WARNING)

    logging.info(f"logger initialized using {logging_level} log level")


init_logging()
config = Config.load_from_env()
dal = config.dal
# config.toolset_manager.load_toolset_with_status(dal=dal, refresh_status=True, enable_all_toolsets=True)

def sync_before_server_start():
    """Sync Holmes status and toolsets before server startup."""
    try:
        update_holmes_status_in_db(dal, config)
    except Exception:
        logging.error("Failed to update holmes status", exc_info=True)
    try:
        holmes_sync_toolsets_status(dal, config)
    except Exception:
        logging.error("Failed to synchronise holmes toolsets", exc_info=True)


if ENABLE_TELEMETRY and SENTRY_DSN:
    # Initialize Sentry for official releases or when development mode is enabled
    if is_official_release() or DEVELOPMENT_MODE:
        environment = "production" if is_official_release() else "development"
        logging.info(f"Initializing sentry for {environment} environment...")

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            send_default_pii=False,
            traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
            profiles_sample_rate=0,
            environment=environment,
        )
        sentry_sdk.set_tags(
            {
                "account_id": dal.account_id,
                "cluster_name": config.cluster_name,
                "version": get_version(),
                "environment": environment,
            }
        )
    else:
        logging.info(
            "Skipping sentry initialization - not an official release and DEVELOPMENT_MODE not enabled"
        )

app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if LOG_PERFORMANCE:

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            process_time = int((time.time() - start_time) * 1000)

            status_code = "unknown"
            if response:
                status_code = response.status_code
            logging.info(
                f"Request completed {request.method} {request.url.path} status={status_code} latency={process_time}ms"
            )


@app.post("/api/investigate")
def investigate_issues(investigate_request: InvestigateRequest):
    try:
        result = investigation.investigate_issues(
            investigate_request=investigate_request,
            dal=dal,
            config=config,
            model=investigate_request.model,
        )
        return result

    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=e.message)
    except litellm.exceptions.RateLimitError as e:
        raise HTTPException(status_code=429, detail=e.message)
    except Exception as e:
        logging.error(f"Error in /api/investigate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stream/investigate")
def stream_investigate_issues(req: InvestigateRequest):
    try:
        ai, system_prompt, user_prompt, response_format, sections, runbooks = (
            investigation.get_investigation_context(req, dal, config)
        )

        return StreamingResponse(
            stream_investigate_formatter(
                ai.call_stream(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_format=response_format,
                    sections=sections,
                ),
                runbooks,
            ),
            media_type="text/event-stream",
        )

    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        logging.exception(f"Error in /api/stream/investigate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workload_health_check")
def workload_health_check(request: WorkloadHealthRequest):
    try:
        resource = request.resource
        workload_alerts: list[str] = []
        if request.alert_history:
            workload_alerts = dal.get_workload_issues(
                resource, request.alert_history_since_hours
            )

        instructions = request.instructions or []
        if request.stored_instrucitons:
            stored_instructions = dal.get_resource_instructions(
                resource.get("kind", "").lower(), resource.get("name")
            )
            if stored_instructions:
                instructions.extend(stored_instructions.instructions)

        nl = "\n"
        if instructions:
            request.ask = f"{request.ask}\n My instructions for the investigation '''{nl.join(instructions)}'''"

        global_instructions = dal.get_global_instructions_for_account()
        request.ask = add_global_instructions_to_user_prompt(
            request.ask, global_instructions
        )

        ai = config.create_toolcalling_llm(dal=dal, model=request.model)

        system_prompt = load_and_render_prompt(
            request.prompt_template,
            context={
                "alerts": workload_alerts,
                "toolsets": ai.tool_executor.toolsets,
                "response_format": workload_health_structured_output,
                "cluster_name": config.cluster_name,
            },
        )

        ai_call = ai.prompt_call(
            system_prompt,
            request.ask,
            HOLMES_POST_PROCESSING_PROMPT,
            workload_health_structured_output,
        )

        ai_call.result = clear_json_markdown(ai_call.result)

        return InvestigationResult(
            analysis=ai_call.result,
            tool_calls=ai_call.tool_calls,
            instructions=instructions,
            metadata=ai_call.metadata,
        )
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=e.message)
    except litellm.exceptions.RateLimitError as e:
        raise HTTPException(status_code=429, detail=e.message)
    except Exception as e:
        logging.exception(f"Error in /api/workload_health_check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workload_health_chat")
def workload_health_conversation(
    request: WorkloadHealthChatRequest,
):
    try:
        ai = config.create_toolcalling_llm(dal=dal, model=request.model)
        global_instructions = dal.get_global_instructions_for_account()

        messages = build_workload_health_chat_messages(
            workload_health_chat_request=request,
            ai=ai,
            config=config,
            global_instructions=global_instructions,
        )
        llm_call = ai.messages_call(messages=messages)

        return ChatResponse(
            analysis=llm_call.result,
            tool_calls=llm_call.tool_calls,
            conversation_history=llm_call.messages,
            metadata=llm_call.metadata,
        )
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=e.message)
    except litellm.exceptions.RateLimitError as e:
        raise HTTPException(status_code=429, detail=e.message)
    except Exception as e:
        logging.error(f"Error in /api/workload_health_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/issue_chat")
def issue_conversation(issue_chat_request: IssueChatRequest):
    try:
        ai = config.create_toolcalling_llm(dal=dal, model=issue_chat_request.model)
        global_instructions = dal.get_global_instructions_for_account()

        messages = build_issue_chat_messages(
            issue_chat_request=issue_chat_request,
            ai=ai,
            config=config,
            global_instructions=global_instructions,
        )
        llm_call = ai.messages_call(messages=messages)

        return ChatResponse(
            analysis=llm_call.result,
            tool_calls=llm_call.tool_calls,
            conversation_history=llm_call.messages,
            metadata=llm_call.metadata,
        )
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=e.message)
    except litellm.exceptions.RateLimitError as e:
        raise HTTPException(status_code=429, detail=e.message)
    except Exception as e:
        logging.error(f"Error in /api/issue_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def already_answered(conversation_history: Optional[List[dict]]) -> bool:
    if conversation_history is None:
        return False

    for message in conversation_history:
        if message["role"] == "assistant":
            return True
    return False


@app.post("/api/chat")
def chat(chat_request: ChatRequest):
    try:
        ai = config.create_toolcalling_llm(dal=dal, model=chat_request.model)
        global_instructions = dal.get_global_instructions_for_account()
        messages = build_chat_messages(
            chat_request.ask,
            chat_request.conversation_history,
            ai=ai,
            config=config,
            global_instructions=global_instructions,
        )

        # Process tool decisions if provided
        if chat_request.tool_decisions:
            logging.info(
                f"Processing {len(chat_request.tool_decisions)} tool decisions"
            )
            messages = ai.process_tool_decisions(messages, chat_request.tool_decisions)
        follow_up_actions = []
        if not already_answered(chat_request.conversation_history):
            follow_up_actions = [
                FollowUpAction(
                    id="logs",
                    action_label="Logs",
                    prompt="Show me the relevant logs",
                    pre_action_notification_text="Fetching relevant logs...",
                ),
                FollowUpAction(
                    id="graphs",
                    action_label="Graphs",
                    prompt="Show me the relevant graphs. Use prometheus and make sure you embed the results with `<< >>` to display a graph",
                    pre_action_notification_text="Drawing some graphs...",
                ),
                FollowUpAction(
                    id="articles",
                    action_label="Articles",
                    prompt="List the relevant runbooks and links used. Write a short summary for each",
                    pre_action_notification_text="Looking up and summarizing runbooks and links...",
                ),
            ]

        if chat_request.stream:
            return StreamingResponse(
                stream_chat_formatter(
                    ai.call_stream(
                        msgs=messages,
                        enable_tool_approval=chat_request.enable_tool_approval or False,
                    ),
                    [f.model_dump() for f in follow_up_actions],
                ),
                media_type="text/event-stream",
            )
        else:
            llm_call = ai.messages_call(messages=messages)

            # For non-streaming, we need to handle approvals differently
            # This is a simplified version - in practice, non-streaming with approvals
            # would require a different approach or conversion to streaming
            return ChatResponse(
                analysis=llm_call.result,
                tool_calls=llm_call.tool_calls,
                conversation_history=llm_call.messages,
                follow_up_actions=follow_up_actions,
                metadata=llm_call.metadata,
            )
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=e.message)
    except litellm.exceptions.RateLimitError as e:
        raise HTTPException(status_code=429, detail=e.message)
    except Exception as e:
        logging.error(f"Error in /api/chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prometheus-keys")
async def list_prometheus_keys():
    """List all stored Prometheus data keys"""
    return {"keys": list(prometheus_data_store.keys()), "count": len(prometheus_data_store)}


@app.get("/api/prometheus-data/{random_key}")
async def get_prometheus_data(random_key: str):
    """Get stored Prometheus data by random key"""
    logging.info(f"Requested key: {random_key}")
    logging.info(f"Available keys: {list(prometheus_data_store.keys())}")
    
    if random_key in prometheus_data_store:
        logging.info(f"Found data for key: {random_key}")
        return prometheus_data_store[random_key]
    else:
        logging.warning(f"No data found for key: {random_key}")
        raise HTTPException(status_code=404, detail="Prometheus data not found")


@app.post("/api/agui/chat")
async def agui_chat_endpoint(input_data: RunAgentInput, request: Request):
    """AG-UI compatible chat endpoint"""
    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)

    async def event_generator():
        try:
            # Send start event
            yield encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id
                )
            )

            # Convert AG-UI input to ChatRequest format
            user_messages = [msg for msg in input_data.messages if msg.role in ['user', 'assistant']]
            
            # Build conversation history with system message if there are previous messages
            conversation_history = None
            if len(user_messages) > 1:
                conversation_history = [
                    {"role": "system", "content": "You are Holmes, an AI assistant for observability. You use Prometheus metrics, alerts and OpenSearch logs to quickly perform root cause analysis."}
                ]
                conversation_history.extend([
                    {"role": msg.role, "content": msg.content}
                    for msg in user_messages[:-1]
                ])
            
            chat_request = ChatRequest(
                ask=user_messages[-1].content if user_messages and user_messages[-1].role == 'user' else "",
                conversation_history=conversation_history,
                model=getattr(input_data, 'model', None),
                stream=True
            )

            # Use existing chat logic
            ai = config.create_toolcalling_llm(dal=dal, model=chat_request.model)
            global_instructions = dal.get_global_instructions_for_account()
            messages = build_chat_messages(
                chat_request.ask,
                chat_request.conversation_history,
                ai=ai,
                config=config,
                global_instructions=global_instructions,
            )

            message_id = str(uuid.uuid4())
            
            # Get the streaming response and handle it properly
            try:
                stream_response = ai.call_stream(msgs=messages)
                response_content = ""
                
                # Process all chunks and stream ALL events
                for chunk in stream_response:
                    if hasattr(chunk, 'event'):
                        event_type = chunk.event.value if hasattr(chunk.event, 'value') else str(chunk.event)
                        logging.info(f"Streaming chunk: {event_type}")
                    else:
                        event_type = 'unknown'
                        logging.info(f"Streaming chunk: {chunk}")
                    
                    if hasattr(chunk, 'event') and hasattr(chunk, 'data'):
                        # Handle ALL Holmes streaming events
                        if event_type == 'start_tool_calling':
                            tool_info = chunk.data.get('tool_name', 'Unknown Tool')
                            yield encoder.encode(
                                TextMessageChunkEvent(
                                    type=EventType.TEXT_MESSAGE_CHUNK,
                                    message_id=message_id,
                                    delta=f"\n🔧 **Starting:** {tool_info} `[{event_type}]`\n",
                                )
                            )
                        
                        elif event_type == 'tool_calling_result':
                            tool_name = chunk.data.get('tool_name', chunk.data.get('name', 'Tool'))
                            duration = chunk.data.get('duration', 0)
                            
                            # Get the actual tool result - it's in 'result' field, not 'output'
                            result_info = chunk.data.get('result', {})
                            
                            # Extract the actual data from the structured result
                            if isinstance(result_info, dict) and 'data' in result_info:
                                try:
                                    # The data might be a JSON string
                                    data_content = result_info['data']
                                    if isinstance(data_content, str):
                                        output_info = json.loads(data_content)
                                    else:
                                        output_info = data_content
                                except (json.JSONDecodeError, TypeError):
                                    output_info = result_info['data']
                            else:
                                output_info = result_info
                            
                            output_length = len(str(output_info))
                            
                            # Debug logging
                            logging.info(f"Tool result type: {type(result_info)}")
                            logging.info(f"Extracted output length: {output_length}")
                            logging.info(f"Output preview: {str(output_info)[:200]}...")
                            
                            # Show the actual tool output to user
                            output_preview = str(output_info)[:500] + "..." if len(str(output_info)) > 500 else str(output_info)
                            yield encoder.encode(
                                TextMessageChunkEvent(
                                    type=EventType.TEXT_MESSAGE_CHUNK,
                                    message_id=message_id,
                                    delta=f"\n📄 **Tool Output ({output_length} chars):**\n```\n{output_preview}\n```\n",
                                )
                            )
                            
                            # Extract random_key from tool output if it exists (Holmes generates these)
                            random_key = None
                            if isinstance(output_info, dict):
                                if 'random_key' in output_info:
                                    random_key = output_info['random_key']
                                    logging.info(f"Found random_key in output_info: {random_key}")
                                elif 'data' in output_info and isinstance(output_info['data'], dict) and 'random_key' in output_info['data']:
                                    random_key = output_info['data']['random_key']
                                    logging.info(f"Found random_key in output_info.data: {random_key}")
                            elif isinstance(output_info, str) and 'random_key' in output_info:
                                # Try to extract from string output
                                key_match = re.search(r'"random_key":\s*"([^"]+)"', output_info)
                                if key_match:
                                    random_key = key_match.group(1)
                                    logging.info(f"Extracted random_key from string: {random_key}")
                            
                            if not random_key:
                                logging.warning(f"No random_key found in tool output. Tool: {tool_name}")
                                logging.warning(f"Output preview: {str(output_info)[:200]}...")
                            
                            # Check if this is a Prometheus query with actual data
                            if ('prometheus' in tool_name.lower() or 'get_metric' in tool_name.lower() or 'execute_prometheus' in tool_name.lower()) and output_info:
                                try:
                                    prom_data = None
                                    
                                    # Handle different Prometheus response formats
                                    if isinstance(output_info, dict):
                                        if 'data' in output_info and 'result' in output_info['data']:
                                            # This is the actual Prometheus response format
                                            result_data = output_info['data']['result']
                                            if isinstance(result_data, list) and len(result_data) > 0:
                                                # Convert instant query results to range format for graphing
                                                for item in result_data:
                                                    if 'value' in item and 'values' not in item:
                                                        # Instant query: convert single value to values array
                                                        item['values'] = [item['value']]
                                                        del item['value']
                                            prom_data = {'data': output_info['data']}
                                        elif 'result' in output_info:
                                            # Direct result format
                                            result_data = output_info['result']
                                            if isinstance(result_data, list) and len(result_data) > 0:
                                                # Convert instant query results to range format for graphing
                                                for item in result_data:
                                                    if 'value' in item and 'values' not in item:
                                                        # Instant query: convert single value to values array
                                                        item['values'] = [item['value']]
                                                        del item['value']
                                            prom_data = {'data': output_info}
                                        # Skip metric name lists - they don't have actual query data to graph
                                        elif 'data' in output_info and isinstance(output_info['data'], list):
                                            logging.info(f"Skipping metric names list for tool: {tool_name}")
                                            prom_data = None
                                        elif isinstance(output_info, list):
                                            # Direct result array
                                            prom_data = {'data': {'result': output_info}}
                                    
                                    logging.info(f"Processed Prometheus data structure: {type(prom_data.get('data', {}).get('result', []) if prom_data else None)} with {len(prom_data.get('data', {}).get('result', []) if prom_data else [])} items")
                                    
                                    # Only create graph if we have actual prometheus data
                                    if prom_data and prom_data.get('data', {}).get('result'):
                                        # Extract the actual query from the output if available
                                        actual_query = output_info.get('query', f'Query from {tool_name}')
                                        
                                        graph_data = {
                                            'type': 'prometheus_graph',
                                            'tool_name': tool_name,
                                            'query': actual_query,
                                            'data': prom_data['data'],
                                            'metadata': {
                                                'start_time': '2025-09-16T15:00:00Z',
                                                'end_time': '2025-09-16T15:30:00Z',
                                                'step': '60s'
                                            }
                                        }
                                        
                                        # Store data with both generated key and tool name for fallback matching
                                        if not random_key:
                                            random_key = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=4))
                                        
                                        prometheus_data_store[random_key] = graph_data
                                        
                                        # Also store by tool name as fallback
                                        tool_key = f"{tool_name}_{len(prometheus_data_store)}"
                                        prometheus_data_store[tool_key] = graph_data
                                        
                                        logging.info(f"Stored Prometheus data with key: {random_key} and tool_key: {tool_key}")
                                        logging.info(f"Sending Prometheus graph data with {len(prom_data['data']['result'])} series")
                                        
                                        yield encoder.encode(
                                            TextMessageChunkEvent(
                                                type=EventType.TEXT_MESSAGE_CHUNK,
                                                message_id=message_id,
                                                delta=f"\n📊 **GRAPH_DATA:** ```json\n{json.dumps(graph_data, indent=2)}\n```\n",
                                            )
                                        )
                                    else:
                                        logging.info(f"No graphable data found in Prometheus output for tool: {tool_name}")
                                        
                                except Exception as e:
                                    logging.error(f"Error processing Prometheus data: {e}")
                            
                            yield encoder.encode(
                                TextMessageChunkEvent(
                                    type=EventType.TEXT_MESSAGE_CHUNK,
                                    message_id=message_id,
                                    delta=f"✅ **Completed:** {tool_name} ({duration:.2f}s, {output_length} chars) `[{event_type}]`\n",
                                )
                            )
                            # Check if this is a Prometheus query - be more flexible with detection
                            if ('prometheus' in tool_name.lower() or 'execute_prometheus' in tool_name.lower()) and output_info:
                                try:
                                    # Try different possible data structures
                                    graph_data_found = False
                                    prom_data = None
                                    
                                    # If output is a string, try to parse as JSON
                                    if isinstance(output_info, str):
                                        try:
                                            parsed_output = json.loads(output_info)
                                            logging.info(f"Parsed JSON output keys: {list(parsed_output.keys()) if isinstance(parsed_output, dict) else 'Not a dict'}")
                                            output_info = parsed_output
                                        except json.JSONDecodeError:
                                            logging.info(f"Output is not JSON, treating as raw string: {output_info[:200]}...")
                                            # Could be raw Prometheus data or error message
                                            if 'result' not in output_info.lower() and 'metric' not in output_info.lower():
                                                logging.warning(f"Prometheus output doesn't look like data: {output_info[:100]}...")
                                                output_info = None
                                    
                                    # Check if output_info now has the data
                                    if isinstance(output_info, dict):
                                        # Look for common Prometheus response patterns
                                        if 'data' in output_info and 'result' in output_info.get('data', {}):
                                            prom_data = output_info
                                        elif 'result' in output_info:
                                            prom_data = {'data': output_info}
                                        elif 'values' in str(output_info) or 'metric' in str(output_info):
                                            # Raw result format
                                            prom_data = {'data': {'result': output_info if isinstance(output_info, list) else [output_info]}}
                                        
                                        if prom_data:
                                            graph_data = {
                                                'type': 'prometheus_graph',
                                                'tool_name': tool_name,
                                                'query': prom_data.get('query', 'Prometheus Query'),
                                                'data': prom_data.get('data', {}),
                                                'metadata': {
                                                    'start_time': prom_data.get('start_time'),
                                                    'end_time': prom_data.get('end_time'),
                                                    'step': prom_data.get('step')
                                                }
                                            }
                                            
                                            logging.info(f"Sending Prometheus graph data with {len(prom_data.get('data', {}).get('result', []))} series")
                                            
                                            yield encoder.encode(
                                                TextMessageChunkEvent(
                                                    type=EventType.TEXT_MESSAGE_CHUNK,
                                                    message_id=message_id,
                                                    delta=f"\n📊 **GRAPH_DATA:** ```json\n{json.dumps(graph_data, indent=2)}\n```\n",
                                                )
                                            )
                                            graph_data_found = True
                                    
                                    if not graph_data_found:
                                        logging.warning(f"Prometheus tool detected but no graph data found. Output: {str(output_info)[:200] if output_info else 'None'}...")
                                        
                                except Exception as e:
                                    logging.error(f"Error processing Prometheus data: {e}")
                                    logging.error(f"Raw output was: {str(output_info)[:200]}...")
                            
                            yield encoder.encode(
                                TextMessageChunkEvent(
                                    type=EventType.TEXT_MESSAGE_CHUNK,
                                    message_id=message_id,
                                    delta=f"✅ **Completed:** {tool_name} ({duration:.2f}s, {output_length} chars)\n",
                                )
                            )
                        
                        elif event_type == 'ai_message':
                            # This might contain the final response
                            content = chunk.data.get('content', '')
                            if content:
                                yield encoder.encode(
                                    TextMessageChunkEvent(
                                        type=EventType.TEXT_MESSAGE_CHUNK,
                                        message_id=message_id,
                                        delta=f"\n📋 **Analysis:** `[{event_type}]`\n{content}\n",
                                    )
                                )
                                response_content = content
                        
                        elif event_type == 'ai_answer_end':
                            # Handle the final AI response
                            content = chunk.data.get('content', '')
                            if content:
                                logging.info(f"Processing ai_answer_end content with {len(content)} characters")
                                logging.info(f"Available Prometheus keys: {list(prometheus_data_store.keys())}")
                                
                                # Look for Holmes promql embeddings and replace with real data
                                def replace_embedding(match):
                                    try:
                                        embed_data = json.loads(match.group(1))
                                        random_key = embed_data.get('random_key')
                                        tool_name_embed = embed_data.get('tool_name', '')
                                        
                                        logging.info(f"Processing embedding: key={random_key}, tool={tool_name_embed}")
                                        
                                        # First try exact random_key match
                                        if random_key and random_key in prometheus_data_store:
                                            real_graph_data = prometheus_data_store[random_key]
                                            logging.info(f"Found exact match for key: {random_key}")
                                            return f"\n📊 **GRAPH_DATA:** ```json\n{json.dumps(real_graph_data, indent=2)}\n```\n"
                                        
                                        # Fallback: try to match by tool name
                                        if tool_name_embed:
                                            for stored_key, stored_data in prometheus_data_store.items():
                                                if tool_name_embed in stored_key or tool_name_embed in stored_data.get('tool_name', ''):
                                                    logging.info(f"Matched embedding by tool name: {tool_name_embed} -> {stored_key}")
                                                    return f"\n📊 **GRAPH_DATA:** ```json\n{json.dumps(stored_data, indent=2)}\n```\n"
                                        
                                        # Final fallback: return the most recent Prometheus data
                                        if prometheus_data_store:
                                            # Get the last stored data (most recent)
                                            last_key = list(prometheus_data_store.keys())[-1]
                                            last_data = prometheus_data_store[last_key]
                                            # Only use if it has actual graph data
                                            if last_data.get('data', {}).get('result'):
                                                logging.info(f"Using most recent Prometheus data as fallback: {last_key}")
                                                return f"\n📊 **GRAPH_DATA:** ```json\n{json.dumps(last_data, indent=2)}\n```\n"
                                        
                                        logging.warning(f"No data found for random_key: {random_key}, tool: {tool_name_embed}")
                                        logging.warning(f"Available keys: {list(prometheus_data_store.keys())}")
                                        return f"[Graph data not available for key: {random_key}]"
                                    except Exception as e:
                                        logging.error(f"Error processing embedding: {e}")
                                        return match.group(0)
                                
                                # Replace Holmes embeddings with real graph data
                                embedding_pattern = r'<<\s*(\{[^}]*"type"\s*:\s*"promql"[^}]*\})\s*>>'
                                processed_content = re.sub(embedding_pattern, replace_embedding, content)
                                
                                logging.info(f"Processed content length: {len(processed_content)} (original: {len(content)})")
                                
                                yield encoder.encode(
                                    TextMessageChunkEvent(
                                        type=EventType.TEXT_MESSAGE_CHUNK,
                                        message_id=message_id,
                                        delta=f"\n📋 **Final Analysis:** `[{event_type}]`\n{processed_content}\n",
                                    )
                                )
                                response_content = processed_content
                        
                        # Catch any other events and show them
                        else:
                            event_data = str(chunk.data)[:200] + "..." if len(str(chunk.data)) > 200 else str(chunk.data)
                            
                            # Check if this is a task update
                            if 'task' in event_type.lower() or ('content' in chunk.data and 'status' in str(chunk.data)):
                                try:
                                    # Try to extract task information
                                    if isinstance(chunk.data, dict) and 'todos' in chunk.data:
                                        tasks = chunk.data['todos']
                                        task_data = {
                                            'type': 'task_update',
                                            'tasks': tasks
                                        }
                                        yield encoder.encode(
                                            TextMessageChunkEvent(
                                                type=EventType.TEXT_MESSAGE_CHUNK,
                                                message_id=message_id,
                                                delta=f"\n📋 **TASK_UPDATE:** ```json\n{json.dumps(task_data, indent=2)}\n```\n",
                                            )
                                        )
                                    else:
                                        # Regular event display
                                        yield encoder.encode(
                                            TextMessageChunkEvent(
                                                type=EventType.TEXT_MESSAGE_CHUNK,
                                                message_id=message_id,
                                                delta=f"\n🔍 **{event_type}:** {event_data} `[{event_type}]`\n",
                                            )
                                        )
                                except Exception as e:
                                    logging.error(f"Error processing task data: {e}")
                                    yield encoder.encode(
                                        TextMessageChunkEvent(
                                            type=EventType.TEXT_MESSAGE_CHUNK,
                                            message_id=message_id,
                                            delta=f"\n🔍 **{event_type}:** {event_data}\n",
                                        )
                                    )
                            else:
                                yield encoder.encode(
                                    TextMessageChunkEvent(
                                        type=EventType.TEXT_MESSAGE_CHUNK,
                                        message_id=message_id,
                                        delta=f"\n🔍 **{event_type}:** {event_data} `[{event_type}]`\n",
                                    )
                                )
                
                # Fallback if no content was streamed
                if not response_content:
                    yield encoder.encode(
                        TextMessageChunkEvent(
                            type=EventType.TEXT_MESSAGE_CHUNK,
                            message_id=message_id,
                            delta="\n✨ Ready to help with your observability questions!",
                        )
                    )
                    
            except Exception as ai_error:
                logging.error(f"AI streaming error: {ai_error}", exc_info=True)
                yield encoder.encode(
                    TextMessageChunkEvent(
                        type=EventType.TEXT_MESSAGE_CHUNK,
                        message_id=message_id,
                        delta=f"Error processing request: {str(ai_error)}",
                    )
                )

            # Send finish event
            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id
                )
            )

        except Exception as error:
            logging.error(f"Error in /api/agui/chat: {error}", exc_info=True)
            yield encoder.encode(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=str(error)
                )
            )

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type()
    )


@app.get("/api/model")
def get_model():
    return {"model_name": json.dumps(config.get_models_list())}


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = (
        "%(asctime)s %(levelname)-8s %(message)s"
    )
    log_config["formatters"]["default"]["fmt"] = (
        "%(asctime)s %(levelname)-8s %(message)s"
    )
    sync_before_server_start()
    uvicorn.run(app, host=HOLMES_HOST, port=HOLMES_PORT, log_config=log_config, reload=False)
