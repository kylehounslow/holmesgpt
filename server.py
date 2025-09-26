# ruff: noqa: E402
import json
import os
import time
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
from fastapi.responses import StreamingResponse
from holmes.utils.stream import stream_investigate_formatter, stream_chat_formatter, StreamMessage, StreamEvents
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


def sync_before_server_start():
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


# TODO - kylhouns: Move everything to experimental/agui/server.py and include front-end example
#                  This will demonstrate messaging, tool calls, page context etc.
from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import uuid
import asyncio
import json
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from ag_ui.core import (
    RunAgentInput,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    MessagesSnapshotEvent,
    ToolMessage,
    ToolCall,
    AssistantMessage, RunErrorEvent
)
from ag_ui.core.events import TextMessageChunkEvent
from ag_ui.encoder import EventEncoder

@app.get("/api/agui/chat/health")
def agui_chat(request: Request):
    return JSONResponse(content="ok")

@app.post("/api/agui/chat")
def agui_chat(input_data: RunAgentInput, request: Request):
    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)

    async def event_generator():
        try:
            # TODO - kylhouns: Parse and inject input_data.context:Context and input_data.state: dynamicContext and staticContext
            # TODO - kylhouns: Provide prompt instructions on handling various context and state for OSD
            logging.info(f"context: {input_data.context}")
            logging.info(f"state: {input_data.state}")
            yield encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id
                )
            )
            chat_request = _agui_input_to_holmes_chat_request(input_data=input_data)
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

            # Hijack the HolmesGPT stream output and format as AG-UI

            # original for reference:
            # return StreamingResponse(
            #     stream_chat_formatter(
            #         ai.call_stream(
            #             msgs=messages,
            #             enable_tool_approval=chat_request.enable_tool_approval or False,
            #         ),
            #         [f.model_dump() for f in follow_up_actions],
            #     ),
            #     media_type="text/event-stream",
            # )

            hgpt_chat_stream_response: StreamMessage = ai.call_stream(
                msgs=messages,
                enable_tool_approval=chat_request.enable_tool_approval or False)

            for chunk in hgpt_chat_stream_response:
                if hasattr(chunk, 'event'):
                    event_type = chunk.event.value if hasattr(chunk.event, 'value') else str(chunk.event)
                    logging.info(f"Streaming chunk: {event_type}")
                else:
                    event_type = 'unknown'
                    logging.info(f"Streaming chunk: {chunk}")
                if hasattr(chunk, 'data'):
                    tool_name = chunk.data.get('tool_name', chunk.data.get('name', 'Tool'))
                    if event_type in (StreamEvents.AI_MESSAGE, StreamEvents.ANSWER_END, "unknown"):
                        async for event in _stream_agui_text_message_event(
                                message=str(chunk.data.get("content", ""))):
                            yield encoder.encode(event)
                    elif event_type == StreamEvents.START_TOOL:
                        async for event in _stream_agui_text_message_event(
                                message=f"🔧 Using Agent tool: `{tool_name}`..."):
                            yield encoder.encode(event)
                    elif event_type == StreamEvents.TOOL_RESULT:
                        # TODO - kylhouns: Render "TodoWrite" tool_name results prettier.
                        logging.info(f"🔧 TOOL_RESULT received - tool_name: {tool_name}")
                        if _should_graph_timeseries_data(tool_name=tool_name):
                            logging.info(f"🔧 Should graph timeseries data for tool: {tool_name}")
                            ts_data = _parse_timeseries_data(chunk.data)
                            tool_call_id = chunk.data.get("tool_call_id", chunk.data.get("id", "unknown"))
                            async for tool_event in _invoke_front_end_tool(
                                    tool_call_id=tool_call_id,
                                    tool_call_name="graph_timeseries_data",
                                    tool_call_args=ts_data):
                                yield encoder.encode(tool_event)
                        elif _should_execute_suggested_query(tool_name=tool_name):
                            tool_call_id = chunk.data.get("tool_call_id", chunk.data.get("id", "unknown"))
                            async for tool_event in _invoke_front_end_tool(
                                    tool_call_id=tool_call_id,
                                    tool_call_name="execute_ppl_query",
                                    tool_call_args={
                                        "query": _parse_query(chunk.data)
                                    }):
                                yield encoder.encode(tool_event)
                        else:
                            async for event in _stream_agui_text_message_event(
                                    message=f"🔧 {tool_name} result:\n{chunk.data.get("result", "")}"):
                                yield encoder.encode(event)
            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                ))
        except Exception as e:
            logging.error(f"Error in /api/agui/chat: {e}", exc_info=True)
            yield encoder.encode(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=f"Agent encountered an error: {str(e)}"
                )
            )
            if isinstance(e, AuthenticationError):
                raise HTTPException(status_code=401, detail=e.message)
            elif isinstance(e, litellm.exceptions.RateLimitError):
                raise HTTPException(status_code=429, detail=e.message)
            else:
                raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type()
    )

def _should_execute_suggested_query(tool_name: str):
    # Only support ppl query for now.
    return tool_name in ("opensearch_ppl_query_assist")

def _parse_query(data) ->str:
    result_data = data.get("result", {})
    params = result_data.get("params", {})
    query = params.get("query", "")
    return query

def _should_graph_timeseries_data(tool_name: str) -> bool:
    # Only support prometheus timeseries data for now.
    return tool_name in ("execute_prometheus_range_query", "execute_prometheus_instant_query")


def _parse_timeseries_data(data) -> dict:
    try:
        # DEBUG: Log the raw input data
        logging.info(f"🔍 _parse_timeseries_data received data: {data}")
        logging.info(f"🔍 Data type: {type(data)}")
        logging.info(f"🔍 Data keys: {list(data.keys()) if hasattr(data, 'keys') else 'No keys'}")
        
        # Extract the result from chunk.data
        result_data = data.get("result", {})
        params = result_data.get("params", {})
        query = params.get("query", "")
        description = params.get("description")
        tool_name = data.get("tool_name", data.get("name", ""))
        
        logging.info(f"🔍 Extracted - result_data: {result_data}")
        logging.info(f"🔍 Extracted - query: {query}")
        logging.info(f"🔍 Extracted - tool_name: {tool_name}")
        
        # If result is a JSON string, parse it
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
                logging.info(f"🔍 Parsed JSON result_data: {result_data}")
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse result as JSON: {result_data}")
                result_data = {}
        
        # Handle different Prometheus response formats
        prometheus_data = result_data
        result_type = "unknown"
        if "data" in result_data:
            prometheus_data = json.loads(result_data["data"]).get("data")
            result_type = prometheus_data.get("resultType", "unknown")

        # Generate a meaningful title
        title = f"Prometheus Query Results"
        if query:
            # Truncate long queries for display
            display_query = query if len(query) <= 50 else query[:47] + "..."
            title = f"Prometheus: {display_query}"
        elif tool_name:
            title = f"{tool_name} Results"
        
        # Prepare metadata
        metadata = {
            "timestamp": int(time.time()),
            "source": "Prometheus",
            "result_type": result_type,
            "description": description,
            "query": query
        }
        
        return {
            "title": description,
            "query": query,
            "data": prometheus_data,
            "metadata": metadata
        }
        
    except Exception as e:
        logging.error(f"Error parsing timeseries data: {e}", exc_info=True)
        # Return a fallback structure
        return {
            "title": "Prometheus Query Results (Parse Error)",
            "query": data.get("query", ""),
            "data": {
                "result": []
            },
            "metadata": {
                "timestamp": int(time.time()),
                "source": "Prometheus",
                "error": str(e)
            }
        }


async def _invoke_front_end_tool(tool_call_id: str, tool_call_name: str, tool_call_args: dict):
    yield ToolCallStartEvent(
        type=EventType.TOOL_CALL_START,
        tool_call_id=tool_call_id,
        tool_call_name=tool_call_name
    )
    yield ToolCallArgsEvent(
        type=EventType.TOOL_CALL_ARGS,
        tool_call_id=tool_call_id,
        delta=json.dumps(tool_call_args)
    )
    yield ToolCallEndEvent(
        type=EventType.TOOL_CALL_END,
        tool_call_id=tool_call_id
    )


async def _stream_agui_text_message_event(message: str):
    message_id = str(uuid.uuid4())
    yield TextMessageStartEvent(
        type=EventType.TEXT_MESSAGE_START,
        message_id=message_id,
        role="assistant"
    )
    yield TextMessageContentEvent(
        type=EventType.TEXT_MESSAGE_CONTENT,
        message_id=message_id,
        delta=message
    )
    yield TextMessageEndEvent(
        type=EventType.TEXT_MESSAGE_END,
        message_id=message_id
    )


def _agui_input_to_holmes_chat_request(input_data: RunAgentInput) -> ChatRequest:
    # Convert AG-UI input to ChatRequest format
    user_messages = [msg for msg in input_data.messages if msg.role in ['user', 'assistant']]
    # Build conversation history with system message if there are previous messages
    # Add page state/context to conversation history
    osd_query_state = _parse_osd_query_state(input_data.state)
    conversation_history = [{
        "role": "system",
        "content": osd_query_state
    }]
    if len(user_messages) > 1:
        conversation_history.append(
            {"role": "system",
             "content": "You are Holmes, an AI assistant for observability. You use Prometheus metrics, alerts and OpenSearch logs to quickly perform root cause analysis."}
        )
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
    return chat_request
def _parse_osd_query_state(state) -> str:
    static_context = state.get("staticContext", {})
    data = static_context.get("data", {})
    data_context = data.get("dataContext", {})
    query = data_context.get("query", {})
    query_str = query.get("query", "")
    dataset = query.get("dataset", {})
    index_pattern = dataset.get("title", "")
    language = query.get("language", "")
    if language:
        return f"My current {language} query is '{query_str}' with index pattern: '{index_pattern}'"
    return ""

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
    # todo - kylhouns: reload=False was only for pycharm debugging
    uvicorn.run(app, host=HOLMES_HOST, port=HOLMES_PORT, log_config=log_config, reload=False)
