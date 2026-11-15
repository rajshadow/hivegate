import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
import toolkits

from agents.toolkit_selector import create_multi_toolkit_selector
from api.routes.v2 import tokens
from api.services import access_token, token_refresh
from db import user_token_crud
from tests.workspace_suite.helpers import make_mock_client
from toolkits.tasks import GOOGLE_TASKS, MICROSOFT_TODO, TASKS_WRITE_TOOLS, TasksToolkit
from toolkits.token_cache import TokenCache


@pytest.fixture(autouse=True)
def isolated_token_cache(monkeypatch):
    cache = TokenCache()
    monkeypatch.setattr(toolkits, "token_cache", cache)
    monkeypatch.setattr(tokens, "token_cache", cache)


def connected_toolkit(service, http, *, user="alice", fetch=lambda user, key: "synthetic-token"):
    toolkit = TasksToolkit(user, service, fetch_token_func=fetch)
    toolkit.tasks_service.provider.http = http
    return toolkit


def task_response(google):
    task = {"id": "task", "title": "null", "status": "completed"}
    if google:
        return {**task, "due": "2026-11-20T00:00:00Z", "completed": "2026-11-19T09:15:00Z"}
    return {
        **task,
        "dueDateTime": {"dateTime": "2026-11-20T00:00:00", "timeZone": "UTC"},
        "completedDateTime": {"dateTime": "2026-11-19T09:15:00", "timeZone": "UTC"},
    }


@pytest.mark.parametrize("service", [GOOGLE_TASKS, MICROSOFT_TODO])
def test_toolkit_operations_through_real_provider(service):
    google = service == GOOGLE_TASKS
    list_title = "title" if google else "displayName"
    collection = "items" if google else "value"
    changes = []

    def handler(request):
        assert request.headers["Authorization"] == "Bearer synthetic-token"
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.content:
            changes.append(json.loads(request.content))
            return httpx.Response(200, json=task_response(google))
        if request.url.path.endswith("/lists"):
            return httpx.Response(200, json={collection: [{"id": "list", list_title: "Work"}]})
        return httpx.Response(200, json={collection: [task_response(google)]})

    with make_mock_client(handler) as http:
        toolkit = connected_toolkit(service, http)
        assert {
            name for name, function in toolkit.get_async_functions().items() if function.requires_confirmation
        } == set(TASKS_WRITE_TOOLS)
        assert toolkit.list_task_lists() == {"lists": [{"id": "list", "title": "Work"}]}
        [task] = toolkit.list_tasks("list")["tasks"]
        assert task["due"] == "2026-11-20"
        assert task["completed_at"] == "2026-11-19T09:15:00+00:00"
        assert task["status"] == "completed"
        assert toolkit.create_task("list", {"title": "null"})["id"] == "task"
        assert toolkit.update_task("list", "task", {"title": "true"})["id"] == "task"
        assert toolkit.update_task("list", "task", {"notes": None, "due": None})["id"] == "task"
        assert toolkit.complete_task("list", "task")["status"] == "completed"
        assert toolkit.delete_task("list", "task") == {"deleted": True, "list_id": "list", "task_id": "task"}
    assert changes[:2] == [{"title": "null"}, {"title": "true"}]
    expected_clear = (
        {"notes": None, "due": None}
        if google
        else {"body": {"contentType": "html", "content": ""}, "dueDateTime": None}
    )
    assert changes[2] == expected_clear


def test_invalid_tool_input_stops_before_io():
    def unexpected_fetch(*args):
        pytest.fail("Invalid input reached token lookup")

    toolkit = TasksToolkit("alice", GOOGLE_TASKS, fetch_token_func=unexpected_fetch)
    update = toolkit.get_async_functions()["update_task"]
    assert update.entrypoint is not None
    for changes in ({}, {"due": "20261120"}, {"due": "2026-02-31"}, {"notes": ""}, {"title": None}, {"notes": 12}):
        assert update.entrypoint(list_id="list", task_id="task", changes=changes)["card"] == "error"
    assert update.entrypoint(list_id="list", changes={"notes": None})["card"] == "error"
    assert update.entrypoint(list_id="list", task_id="task", changes={"notes": None}, extra="unused")["card"] == "error"
    assert update.entrypoint(list_id="list", task_id=".", changes={"notes": None})["card"] == "error"
    assert toolkit.create_task("list", {"title": "", "notes": "private-canary"})["card"] == "error"
    disconnected = TasksToolkit("alice", "tasks", auth=False, fetch_token_func=unexpected_fetch)
    assert list(disconnected.get_async_functions()) == ["tasks_auth_required"]
    assert disconnected.create_task("list", {"title": "Draft"})["card"] == "tasks-auth-required"


def test_selector_refresh_reconnect_and_per_user_cache(monkeypatch, caplog):
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    def grant(provider, bearer, *, expired=False):
        return SimpleNamespace(
            provider=provider,
            token_type="oauth2",
            expires_at=now + timedelta(hours=-1 if expired else 1),
            data={"access_token": bearer, "refresh_token": "offline-refresh-grant", "client_id": "offline-client"},
            created_at=now,
        )

    rows = {
        ("alice", GOOGLE_TASKS): grant("google", "expired-alice", expired=True),
        ("alice", MICROSOFT_TODO): grant("microsoft", "unused-outlook"),
        ("bob", GOOGLE_TASKS): grant("google", "bob"),
        ("carol", MICROSOFT_TODO): grant("microsoft", "carol"),
    }
    monkeypatch.setattr(access_token, "get_db", lambda: iter([SimpleNamespace(close=lambda: None)]))
    monkeypatch.setattr(
        user_token_crud, "has_user_tokens_batch", lambda db, user, keys: {key: (user, key) in rows for key in keys}
    )
    # The production path still decides expiry, refreshes, and resolves by user and service.
    monkeypatch.setattr(token_refresh, "get_user_token", lambda db, user, key: rows.get((user, key)))
    monkeypatch.setattr(token_refresh, "get_decrypted_token_data", lambda row: row.data)

    def update_grant(db, user, key, data, expires_at):
        row = rows[user, key]
        row.data, row.expires_at = data, expires_at
        return row

    monkeypatch.setattr(token_refresh, "update_user_token_data", update_grant)

    async def refresh(self, url, data, **kwargs):
        assert httpx.URL(url).host == "oauth2.googleapis.com"
        assert data["refresh_token"] == "offline-refresh-grant"
        return httpx.Response(200, json={"access_token": "refreshed-alice", "expires_in": 3600})

    monkeypatch.setattr(token_refresh.OAuth2TokenRefresher, "_make_refresh_request", refresh)
    requests = []
    fail = False

    def task_http(request):
        requests.append((request.url.host, request.headers["Authorization"]))
        if fail:
            return httpx.Response(401, text="private-vendor-body")
        return httpx.Response(200, json={})

    with make_mock_client(task_http) as http:

        def select(user):
            agent = SimpleNamespace(set_tools=lambda tools: setattr(agent, "tools", tools))
            candidates = TasksToolkit.provider_candidates(user, access_token.fetch_access_token)
            for toolkit in candidates:
                if toolkit.tasks_service is not None:
                    toolkit.tasks_service.provider.http = http
            selector = create_multi_toolkit_selector(
                user,
                "UTC",
                [],
                additional_toolkits={"tasks": candidates},
            )
            selector(agent, None)
            [toolkit] = agent.tools
            return toolkit

        alice, bob, carol, disconnected = (select(user) for user in ("alice", "bob", "carol", "unconnected"))
        assert disconnected.list_task_lists()["context"]["service_name"] is None
        for toolkit in (alice, bob, carol):
            assert toolkit.list_task_lists() == {"lists": []}
        fail = True
        assert alice.list_task_lists()["context"] == {
            "token_valid": False,
            "service_name": GOOGLE_TASKS,
            "available_services": [GOOGLE_TASKS, MICROSOFT_TODO],
        }
        fail = False
        assert alice.list_task_lists() == {"lists": []}  # Refill the old grant before reconnect.

        def store_grant(**kwargs):
            row = grant("google", kwargs["token_data"]["access_token"])
            rows[kwargs["user_id"], kwargs["integration_key"]] = row
            return row

        monkeypatch.setattr(tokens, "create_user_token", store_grant)
        body = tokens.StoreTokenRequest(
            integration_key=GOOGLE_TASKS,
            provider="google",
            token_type="oauth2",
            token_data=tokens.TokenData(access_token="replacement-alice", refresh_token="offline-refresh-grant"),
        )
        asyncio.run(tokens.store_user_token("alice", body, db=None))
        bob.list_task_lists()
        alice.list_task_lists()
    assert requests == [
        ("tasks.googleapis.com", "Bearer refreshed-alice"),
        ("tasks.googleapis.com", "Bearer bob"),
        ("graph.microsoft.com", "Bearer carol"),
        ("tasks.googleapis.com", "Bearer refreshed-alice"),
        ("tasks.googleapis.com", "Bearer refreshed-alice"),
        ("tasks.googleapis.com", "Bearer bob"),
        ("tasks.googleapis.com", "Bearer replacement-alice"),
    ]
    assert "private-vendor-body" not in caplog.text


@pytest.mark.parametrize("failure", [403, 500, "timeout", "malformed"])
def test_task_failure_is_sanitized(failure, caplog):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("private-canary", request=request)
        if failure == "malformed":
            return httpx.Response(200, text="private-canary")
        return httpx.Response(failure, text="private-canary")

    with make_mock_client(handler) as http:
        reply = connected_toolkit(GOOGLE_TASKS, http).create_task("list", {"title": "private-canary"})
    assert reply["card"] == "error"
    assert "private-canary" not in str(reply)
    assert "private-canary" not in caplog.text
    if failure == 403:
        assert "Permission denied" in reply["message"]
