import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import toolkits
from agno.db.in_memory import InMemoryDb
from anthropic.types import ContentBlockDeltaEvent, ContentBlockStopEvent, MessageStopEvent, ToolUseBlock
from fastapi import FastAPI

from agents import agent as agent_factory
from agents.claude_refusal import RefusalAwareClaude
from api.routes.v2 import agents
from api.services.models import PullPromptResponse
from db.agent_info_crud import AgentConfig
from tests.test_hitl import _ScriptedClaude
from tests.workspace_suite.helpers import make_mock_client
from toolkits import tasks
from toolkits.tasks import GOOGLE_TASKS
from toolkits.token_cache import TokenCache
from workspace_suite.providers.google_tasks import GoogleTasksProvider

CHAT = "/v2/agents/tasks-agent/chat"
OWNER = {"user_id": "alice", "session_id": "chat-session"}


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(agents.v2_agents_router, prefix="/v2")
    app.dependency_overrides[agents.get_db] = lambda: None
    app.dependency_overrides[agents.get_api_key] = lambda: SimpleNamespace(id=1)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("approve", [True, False])
@pytest.mark.parametrize("stream", [True, False])
async def test_chat_approval_with_real_agno_and_tasks_toolkit(app, monkeypatch, approve, stream):
    proposed_arguments = {"list_id": "list", "task_id": "task", "changes": {"notes": None, "title": "null"}}
    dispatched = []

    def task_request(request):
        assert request.method == "PATCH"
        assert request.url.path == "/tasks/v1/lists/list/tasks/task"
        assert request.headers["Authorization"] == "Bearer synthetic-chat-token"
        dispatched.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "task", "title": "null", "status": "needsAction"})

    provider_http = make_mock_client(task_request)
    monkeypatch.setitem(
        tasks._TASKS_PROVIDERS, GOOGLE_TASKS, lambda config: GoogleTasksProvider(config, http=provider_http)
    )
    monkeypatch.setattr(toolkits, "token_cache", TokenCache())
    scripted = _ScriptedClaude(
        ToolUseBlock(type="tool_use", id="toolu_update", name="update_task", input=proposed_arguments)
    )

    async def create(**kwargs):
        if not scripted.calls:
            tools = {item["name"]: item for item in kwargs["tools"]}
            schema = tools["update_task"]["input_schema"]
            assert schema["required"] == ["list_id", "task_id", "changes"]
            assert schema["properties"]["changes"]["properties"]["notes"]["anyOf"] == [
                {"type": "string"},
                {"type": "null"},
            ]
        return scripted.create(**kwargs)

    @asynccontextmanager
    async def stream_reply(**kwargs):
        message = await create(**kwargs)

        async def events():
            for index, block in enumerate(message.content):
                if block.type == "tool_use":
                    yield ContentBlockStopEvent(type="content_block_stop", index=index, content_block=block)
                else:
                    yield ContentBlockDeltaEvent(
                        type="content_block_delta", index=index, delta={"type": "text_delta", "text": block.text}
                    )
            yield MessageStopEvent(type="message_stop", message=message)

        yield events()

    model = RefusalAwareClaude(id="claude-sonnet-5-5", api_key="offline-test")
    model.get_async_client = lambda: SimpleNamespace(messages=SimpleNamespace(create=create, stream=stream_reply))
    monkeypatch.setenv("AGNO_TELEMETRY", "false")
    db = InMemoryDb()
    monkeypatch.setattr(agent_factory, "get_agent_db", lambda table: db)
    monkeypatch.setattr(agent_factory, "get_knowledge_service", lambda: SimpleNamespace(get_dynamic_kb=lambda: None))
    monkeypatch.setattr(agent_factory, "create_model", lambda **kwargs: model)
    monkeypatch.setattr(agent_factory.api_settings, "enable_guardrails", False)
    prompt = PullPromptResponse(name="tasks", template="Manage tasks.", tags=["available_tools:tasks"])
    config = AgentConfig(enable_memory=False, enable_history=False)
    monkeypatch.setattr(agents, "get_agent_info", lambda db, agent_id: SimpleNamespace(prompt_service_id="one"))
    monkeypatch.setattr(agents, "get_agent_config", lambda db_agent: config)
    monkeypatch.setattr(agents, "_get_agent_prompt", lambda db, agent_id, prompt_id: prompt)
    monkeypatch.setattr(agents, "fetch_access_token", lambda user, key: "synthetic-chat-token")
    monkeypatch.setattr(
        agents, "has_access_tokens_batch", lambda user, keys: {key: key == GOOGLE_TASKS for key in keys}
    )
    monkeypatch.setattr(agents, "record_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(agents, "_agent_cache", {})
    monkeypatch.setattr(agents, "_run_cache", {})

    request = {**OWNER, "stream": stream}

    with provider_http:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as gateway:
            response = await gateway.post(
                CHAT, json={**request, "message": "Update task", "timezone": "UTC", "locale": "en"}
            )
            assert response.status_code == 200, response.text
            if stream:
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                paused = next(event for event in events if event.get("event") == "RunPaused")
            else:
                paused = response.json()
                assert paused["status"].lower() == "paused"
            [tool] = paused["tools"]
            assert tool["tool_args"] == proposed_arguments
            assert dispatched == []

            reply = await gateway.post(
                CHAT + "/commit",
                json={**request, "run_id": paused["run_id"], "updated_tools": [{**tool, "confirmed": approve}]},
            )
            assert reply.status_code == 200, reply.text
            if stream and approve:
                assert "RunCompleted" in reply.text
            else:
                assert reply.json()["status"] == ("completed" if approve else "cancelled")

    assert dispatched == ([{"title": "null", "notes": None}] if approve else [])


@pytest.mark.asyncio
async def test_legacy_routes_exclude_tasks_before_parsing_arguments(app, monkeypatch, caplog):
    toolkit = tasks.TasksToolkit("alice", "tasks", auth=False)
    agent_lookup = AsyncMock(return_value=(SimpleNamespace(tools=[toolkit]), None, "", None))
    monkeypatch.setattr(agents, "get_agent", agent_lookup)
    monkeypatch.setattr(agents, "_toolkit_execution_cache", {"example": {"toolkit": toolkit}})

    path = "/v2/agents/tasks-agent/toolkit"
    tool = {"toolkit_name": "TasksToolkit", "method_name": "delete_task"}
    args = {"task_id": "private-legacy-canary"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as gateway:
        run = await gateway.get(path + "/run", params={**tool, **args, **OWNER, "skip_confirmation": "true"})
        confirm = await gateway.post(
            path + "/confirm",
            json={**tool, "toolkit_name": "CalendarToolkit", "execution_id": "example", "args": args},
        )
    assert run.status_code == confirm.status_code == 404
    agent_lookup.assert_awaited_once()
    assert "private-legacy-canary" not in caplog.text
