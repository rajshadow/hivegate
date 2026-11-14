import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agno.models.response import ToolExecution
from agno.run import RunStatus
from agno.run.agent import RunOutput, RunPausedEvent
from agno.run.requirement import RunRequirement
from fastapi import FastAPI, Header
from httpx import ASGITransport, AsyncClient

from api.routes.v2 import agents
from toolkits.base import BaseToolkit

MODEL = "gemini-2.5-pro"
CHAT = "/v2/agents/assistant/chat"
BODY = {"user_id": "u1", "session_id": "s1", "model": MODEL}
CALL = {"tool_call_id": "write-1", "confirmed": True}


class TestProviderToolkit(BaseToolkit):
    __test__ = False

    def _initialize_service(self):
        self.requires_confirmation_tools = ["write_item"]
        self.register(self.write_item)

    def write_item(self):
        """Write an item with the selected account."""
        return self.service_name


class ScriptedAgent:
    """Produce native Agno pause types and capture the actual route dispatch arguments."""

    session_id = "s1"
    id = "test-agent"
    model = SimpleNamespace(id=MODEL, provider="Test")

    def __init__(self):
        self.tools = [TestProviderToolkit("u1", "google_example")]
        self.requests = []
        self.writes = []
        self.after_claim = None
        self.pause_again = False
        self.reuse_call_id = False

    def arun(self, message, *, stream, run_id, run_context, **kwargs):
        assert run_id == run_context.run_id
        tool = ToolExecution(tool_call_id=CALL["tool_call_id"], tool_name="write_item", requires_confirmation=True)
        tools = [tool]
        if self.pause_again:
            tools.append(ToolExecution(tool_call_id="denied-1", tool_name="write_item", requires_confirmation=True))
        requirements = [RunRequirement(tool_execution=call) for call in tools]
        if stream:

            async def events():
                yield RunPausedEvent(run_id=run_id, tools=tools, requirements=requirements)

            return events()

        async def response():
            return RunOutput(run_id=run_id, status=RunStatus.paused, tools=tools, requirements=requirements)

        return response()

    def acontinue_run(self, *, run_id, run_context, requirements, stream, **kwargs):
        self.requests.append((run_id, requirements, kwargs))
        if self.after_claim:
            self.after_claim()
        for requirement in requirements:
            if requirement.confirmation is True:
                toolkit = run_context.tools[0]
                self.writes.append(toolkit.write_item())

        async def result():
            if self.pause_again:
                [denied] = [req.tool_execution for req in requirements if req.confirmation is False]
                assert denied is not None
                # Agno leaves a denied call in the tool log with no result.
                denied.requires_confirmation = False
                call_id = denied.tool_call_id if self.reuse_call_id else "write-2"
                another = ToolExecution(tool_call_id=call_id, tool_name="write_item", requires_confirmation=True)
                return RunOutput(
                    run_id=run_id,
                    status=RunStatus.paused,
                    tools=[denied, another],
                    requirements=[RunRequirement(tool_execution=another)],
                )
            return RunOutput(run_id=run_id, status=RunStatus.completed, content="Done")

        if stream:

            async def events():
                response = await result()
                if response.is_paused:
                    yield RunPausedEvent(run_id=run_id, tools=response.tools, requirements=response.requirements)

            return events()
        return result()


@pytest.fixture
def gateway(monkeypatch):
    agent = ScriptedAgent()

    async def caller(x_api_key: int = Header(default=1)):
        return SimpleNamespace(id=x_api_key, name="Test")

    app = FastAPI()
    app.include_router(agents.v2_agents_router, prefix="/v2")
    app.dependency_overrides[agents.get_api_key] = caller
    app.dependency_overrides[agents.get_db] = lambda: None
    monkeypatch.setattr(agents, "get_agent", AsyncMock(return_value=(agent, None, "key", None)))
    monkeypatch.setattr(agents, "build_mcp_toolkits", lambda config: [])
    monkeypatch.setattr(agents, "has_access_tokens_batch", lambda user_id, keys: dict.fromkeys(keys, True))
    monkeypatch.setattr(agents, "record_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(agents, "_run_cache", {})
    return agent, app


async def pause(client, stream=False):
    response = await client.post(
        CHAT, json={**BODY, "message": "Write an item", "timezone": "UTC", "locale": "en", "stream": stream}
    )
    assert response.status_code == 200, response.text
    if stream:
        assert "RunPaused" in response.text
        [run_id] = agents._run_cache
        return run_id
    result = response.json()
    assert result["status"] == RunStatus.paused.value
    return result["run_id"]


def commit_body(run_id, *, decision=True, stream=False, **fields):
    return {**BODY, "run_id": run_id, "stream": stream, "updated_tools": [{**CALL, "confirmed": decision}], **fields}


@pytest.mark.parametrize(
    ("path", "fields", "headers"),
    [
        (CHAT, {"user_id": "u2"}, {}),
        (CHAT, {"session_id": "s2"}, {}),
        (CHAT, {"model": "gemini-2.5-flash"}, {}),
        ("/v2/agents/another-agent/chat", {}, {}),
        (CHAT, {}, {"x-api-key": "2"}),
    ],
)
def test_foreign_context_cannot_approve_or_consume_a_pause(gateway, path, fields, headers):
    agent, app = gateway

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            run_id = await pause(client)
            foreign = await client.post(path + "/commit", json=commit_body(run_id, **fields), headers=headers)
            assert foreign.status_code == 410
            await client.post(path + "/commit", json=commit_body(run_id, decision=False, **fields), headers=headers)
            assert not agent.requests
            rightful = await client.post(CHAT + "/commit", json=commit_body(run_id))
            assert rightful.status_code == 200, rightful.text
            assert agent.writes == ["google_example"]

    asyncio.run(go())


@pytest.mark.parametrize("stream", [False, True])
def test_approval_keeps_original_tool_after_dispatch_and_is_consumed(gateway, stream):
    agent, app = gateway
    agent.after_claim = lambda: agent.tools.__setitem__(0, TestProviderToolkit("u1", "microsoft_example"))

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            run_id = await pause(client, stream=stream)
            reply = await client.post(CHAT + "/commit", json=commit_body(run_id, stream=stream))
            assert reply.status_code == 200, reply.text
            again = await client.post(CHAT + "/commit", json=commit_body(run_id, stream=stream))
            assert again.status_code == 410
            assert agent.writes == ["google_example"]

    asyncio.run(go())


@pytest.mark.parametrize("stream", [False, True])
def test_changed_provider_before_commit_requires_new_approval(gateway, stream):
    agent, app = gateway

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            run_id = await pause(client, stream=stream)
            agent.tools = [TestProviderToolkit("u1", "microsoft_example")]
            reply = await client.post(CHAT + "/commit", json=commit_body(run_id, stream=stream))
            assert reply.status_code == 409
            assert not agent.requests
            assert not agent.writes

    asyncio.run(go())


@pytest.mark.parametrize("stream", [False, True])
def test_followup_pause_keeps_the_same_owner(gateway, stream):
    agent, app = gateway
    agent.pause_again = True

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            run_id = await pause(client, stream=stream)
            decisions = [CALL, {"tool_call_id": "denied-1", "confirmed": False}]
            reply = await client.post(
                CHAT + "/commit", json=commit_body(run_id, stream=stream, updated_tools=decisions)
            )
            assert reply.status_code == 200
            assert "write-2" in reply.text
            assert agent.writes == ["google_example"]
            assert list(agents._run_cache) == [run_id]
            replay = await client.post(CHAT + "/commit", json=commit_body(run_id, stream=stream))
            assert replay.status_code == 409
            assert list(agents._run_cache) == [run_id]
            stale_denial = await client.post(CHAT + "/commit", json=commit_body(run_id, updated_tools=[decisions[1]]))
            assert stale_denial.status_code == 409
            assert list(agents._run_cache) == [run_id]
            deny = commit_body(
                run_id,
                updated_tools=[
                    {"tool_call_id": "denied-1", "confirmed": True},
                    {"tool_call_id": "write-2", "confirmed": False},
                ],
            )
            reply = await client.post(CHAT + "/commit", json=deny)
            assert reply.json()["status"] == "cancelled"
            assert not agents._run_cache
            assert agent.writes == ["google_example"]

    asyncio.run(go())


def test_disconnected_toolkit_requires_new_approval(gateway, monkeypatch):
    agent, app = gateway
    monkeypatch.setattr(agents, "has_access_tokens_batch", lambda user_id, keys: dict.fromkeys(keys, False))

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            run_id = await pause(client)
            reply = await client.post(CHAT + "/commit", json=commit_body(run_id))
            assert reply.status_code == 409
            assert not agent.requests
            assert not agent.writes

    asyncio.run(go())


def test_reused_tool_call_id_requires_new_approval(gateway):
    agent, app = gateway
    agent.pause_again = True
    agent.reuse_call_id = True

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            run_id = await pause(client)
            denied = {"tool_call_id": "denied-1", "confirmed": False}
            first = await client.post(CHAT + "/commit", json=commit_body(run_id, updated_tools=[CALL, denied]))
            assert first.status_code == 200, first.text
            reply = await client.post(CHAT + "/commit", json=commit_body(run_id, updated_tools=[denied]))
            assert reply.status_code == 409
            assert "reused" in reply.text
            assert agent.writes == ["google_example"]

    asyncio.run(go())


def test_unrelated_toolkit_disconnect_preserves_approval(gateway, monkeypatch):
    agent, app = gateway

    class ContactsToolkit(BaseToolkit):
        def _initialize_service(self):
            pass

    agent.tools.append(ContactsToolkit("u1", "contacts_example"))
    monkeypatch.setattr(
        agents,
        "has_access_tokens_batch",
        lambda user_id, keys: {key: key != "contacts_example" for key in keys},
    )

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            run_id = await pause(client)
            agent.tools.pop()
            reply = await client.post(CHAT + "/commit", json=commit_body(run_id))
            assert reply.status_code == 200, reply.text
            assert agent.writes == ["google_example"]

    asyncio.run(go())


def test_real_agno_resumes_using_the_original_selection(gateway, monkeypatch):
    from agno.agent import Agent
    from agno.db.in_memory import InMemoryDb

    from agents.claude_refusal import RefusalAwareClaude
    from agents.toolkit_selector import create_multi_toolkit_selector
    from tests.test_hitl import _ScriptedClaude

    _, app = gateway
    writes = []

    class CalendarToolkit(BaseToolkit):
        def _initialize_service(self):
            self.requires_confirmation_tools = ["delete_budget"]
            self.register(self.delete_budget)

        def delete_budget(self, budget_id: int) -> str:
            """Delete a budget from the connected account."""
            writes.append((self.service_name, budget_id))
            return "deleted"

    google = CalendarToolkit("u1", "google_calendar")
    microsoft = CalendarToolkit("u1", "microsoft_calendar")
    selector = create_multi_toolkit_selector(
        user_id="u1",
        timezone="UTC",
        base_tools=[],
        calendar_google=google,
        calendar_microsoft=microsoft,
    )
    model = RefusalAwareClaude(id="claude-sonnet-5-5", api_key="offline-test")
    scripted = _ScriptedClaude()

    async def create(**kwargs):
        return scripted.create(**kwargs)

    model.get_async_client = lambda: SimpleNamespace(messages=SimpleNamespace(create=create))

    class AgentWithProviderSwitch(Agent):
        async def aget_tools(self, *args, **kwargs):
            if self.switch_on_resume:
                self.set_tools([microsoft])
            return await super().aget_tools(*args, **kwargs)

    agent = AgentWithProviderSwitch(
        model=model, pre_hooks=[selector], db=InMemoryDb(), user_id="u1", session_id="s1", telemetry=False
    )
    agent.switch_on_resume = False
    monkeypatch.setattr(agents, "get_agent", AsyncMock(return_value=(agent, None, "key", None)))

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            run_id = await pause(client)
            assert writes == []
            agent.switch_on_resume = True
            body = commit_body(run_id, updated_tools=[{"tool_call_id": "toolu_1", "confirmed": True}])
            reply = await client.post(CHAT + "/commit", json=body)
            assert reply.status_code == 200, reply.text
            assert reply.json()["content"] == "done"
            assert writes == [("google_calendar", 7)]

    asyncio.run(go())
