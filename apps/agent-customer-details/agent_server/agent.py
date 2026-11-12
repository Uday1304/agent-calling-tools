
from agents.mcp import MCPServer, MCPServerManager
from typing import AsyncGenerator, List

import mlflow
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.utils import (
    build_mcp_url,
    get_user_workspace_client,
    process_agent_stream_events,
)

# NOTE: this will work for all databricks models OTHER than GPT-OSS, which uses a slightly different API
set_default_openai_api("chat_completions")
set_trace_processors([])  # only use mlflow for trace processing
mlflow.openai.autolog()

# GENERATED

set_default_openai_client(AsyncDatabricksOpenAI(use_ai_gateway=True))
NAME = 'agent-customer-details'
SYSTEM_PROMPT = 'You are a customer address assistant.\n\nYour job is to answer questions about customer address information.\n\nUse the available tools whenever customer address data is required.\n\nDo not invent customer information.\n\nIf a customer cannot be found, clearly tell the user that no matching customer was found.\n\nWhen customer_id is provided, use the customer-address lookup tool.\n\nWhen a user asks for customers by city, use the city lookup tool.'
MODEL = 'system.ai.llama-4-maverick'
MCP_SERVERS = [
    ('UC Function: gizmobox.gold.find_customers_by_billing_city', '/api/2.0/mcp/functions/gizmobox/gold/find_customers_by_billing_city'),
    ('UC Function: gizmobox.gold.get_customer_address', '/api/2.0/mcp/functions/gizmobox/gold/get_customer_address'),
    ('UC Function: gizmobox.gold.get_customer_contact_info', '/api/2.0/mcp/functions/gizmobox/gold/get_customer_contact_info'),
]

# END GENERATED


def get_mcp_user_workspace_client():
    # Uncomment the line below to enable on-behalf-of-user authentication
    # return get_user_workspace_client()
    return None


def init_mcp_servers():
    user_workspace_client = get_mcp_user_workspace_client()
    return [
        McpServer(
            name=name,
            url=build_mcp_url(url, user_workspace_client),
            workspace_client=user_workspace_client,
        )
        for (name, url) in MCP_SERVERS
    ]

def sanitize_input_items(items):
    # The chat UI echoes prior assistant replies back as items with a 'status'
    # key and no 'id', which the SDK chat-completions converter rejects with
    # 'Unhandled item type or structure' on every follow-up query. Rebuild them
    # as simple assistant messages.
    sanitized = []
    for item in items:
        if (
            isinstance(item, dict)
            and item.get('type') == 'message'
            and item.get('role') == 'assistant'
            and 'id' not in item
        ):
            content = item.get('content')
            if isinstance(content, list):
                text = ''.join(
                    part.get('text', '')
                    for part in content
                    if isinstance(part, dict) and part.get('type') == 'output_text'
                )
            else:
                text = content or ''
            sanitized.append({'role': 'assistant', 'content': text})
        else:
            sanitized.append(item)
    return sanitized


def create_agent(mcp_servers: List[MCPServer]) -> Agent:
    return Agent(
        name=NAME,
        instructions=SYSTEM_PROMPT,
        model=MODEL,
        mcp_servers=mcp_servers,
    )


@invoke()
async def invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    mcp_servers = init_mcp_servers()
    async with MCPServerManager(servers = mcp_servers, connect_in_parallel=True) as manager:
        agent = create_agent(manager.active_servers)
        messages = sanitize_input_items([i.model_dump() for i in request.input])
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream(request: dict) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    mcp_servers = init_mcp_servers()
    async with MCPServerManager(servers = mcp_servers, connect_in_parallel=True) as manager:
        agent = create_agent(manager.active_servers)
        messages = sanitize_input_items([i.model_dump() for i in request.input])
        result = Runner.run_streamed(agent, input=messages)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
