import json
import logging
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
import requests  # noqa: F401  (imported before the CLI so its patch below reaches the same module)

UPDATE = {"list_id": "opaque", "task_id": "one", "changes": {"notes": None, "title": "null"}}
DELETE = {"list_id": "opaque", "task_id": "two"}
CANCELLED = {"status": "cancelled", "content": "Tool execution cancelled by user."}


def _pending(index, name, arguments):
    return {"tool_name": name, "tool_call_id": f"call-{index}", "tool_args": arguments, "requires_confirmation": True}


# A completed read, two pending Tasks writes, and a pending legacy write with its own editor.
COMPLETED = {"tool_name": "list_tasks", "tool_call_id": "done", "requires_confirmation": False, "result": {"ok": 1}}
LEGACY = _pending("legacy", "create_contact", {"name": "Ada"})
PAUSED = {
    "event": "RunPaused",
    "run_id": "run",
    "status": "paused",
    "tools": [COMPLETED, _pending(0, "update_task", UPDATE), _pending(1, "delete_task", DELETE), LEGACY],
}


def _client(monkeypatch, decisions, stream):
    directory = Path(__file__).resolve().parents[1] / "cli"
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setattr("logging.handlers.RotatingFileHandler", lambda *args, **kwargs: logging.NullHandler())
    logger = logging.getLogger("agent_cli")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logger.level)
    select = Mock(side_effect=[SimpleNamespace(execute=lambda action=action: action) for action in decisions])
    terminal_stubs = {
        "InquirerPy": SimpleNamespace(inquirer=SimpleNamespace(select=select)),
        "InquirerPy.base.control": SimpleNamespace(Choice=SimpleNamespace),
        "InquirerPy.separator": SimpleNamespace(Separator=SimpleNamespace),
    }
    with patch.dict(sys.modules, terminal_stubs):
        for name in ("client_ui.dialogs", "client_ui.formatters", "client_ui"):
            sys.modules.pop(name, None)
        cli = runpy.run_path(str(directory / "agent_cli.py"))
    client_class = cli["ConsoleChatClient"]
    client = client_class.__new__(client_class)
    client.dialog_builder = cli["DialogBuilder"](None)
    legacy_editor = Mock(return_value={"action": "skip", "edited_args": {}})
    monkeypatch.setattr(client.dialog_builder, "_build_generic_dialog", legacy_editor)
    client.agent_id, client.user_id, client.session_id = "agent", "alice", "session"
    client.model, client.stream_flag = "gemini-2.5-pro", stream
    client.format_timestamp = lambda: "test"
    client.result_formatter = Mock()
    client._last_tools_payload, client._last_run_id = [], None
    return client, select, legacy_editor


def _sse(*events):
    return "".join(f"event: message\ndata: {json.dumps(event)}\n\n" for event in events)


def _run(monkeypatch, client, stream, commit_reply):
    """Deliver PAUSED to the client and return the commit payloads it sends."""
    submitted = []
    if stream:
        transport_client = httpx.Client

        def transport(request):
            if request.url.path.endswith("/commit"):
                submitted.append(json.loads(request.content))
                if commit_reply == CANCELLED:
                    return httpx.Response(200, json=CANCELLED)
                # A further pause after the commit is out of scope and must not be prompted.
                return httpx.Response(200, text=_sse(PAUSED))
            # A later event carrying tools must not replace the RunPaused batch.
            return httpx.Response(200, text=_sse(PAUSED, {"event": "ToolCallCompleted", "tools": [COMPLETED]}))

        monkeypatch.setattr(
            httpx, "Client", lambda **kwargs: transport_client(transport=httpx.MockTransport(transport))
        )
        client.handle_sse_stream("Update task")
    else:

        def post(url, *, json, timeout):
            submitted.append(json)
            return SimpleNamespace(status_code=200, json=lambda: commit_reply)

        monkeypatch.setattr("requests.post", post)
        client.handle_single_response(PAUSED)
    return submitted


@pytest.mark.parametrize("decisions", [("confirm", "decline"), ("decline", "decline")])
@pytest.mark.parametrize("stream", [False, True])
def test_cli_submits_each_pause_once_and_preserves_pending_decisions(monkeypatch, capsys, decisions, stream):
    client, select, legacy_editor = _client(monkeypatch, decisions, stream)
    all_declined = decisions == ("decline", "decline")
    reply = CANCELLED if all_declined else {**PAUSED, "event": "RunCompleted", "content": "Updated."}

    [payload] = _run(monkeypatch, client, stream, reply)

    assert payload["run_id"] == "run"
    assert payload["stream"] is stream
    # The skipped legacy call is left out of the payload, unlike an explicit decline.
    assert [tool["tool_call_id"] for tool in payload["updated_tools"]] == ["call-0", "call-1"]
    assert [tool["tool_args"] for tool in payload["updated_tools"]] == [UPDATE, DELETE]
    assert [tool["confirmed"] for tool in payload["updated_tools"]] == [decisions[0] == "confirm", False]
    assert select.call_count == 2
    legacy_editor.assert_called_once_with(LEGACY)
    assert client._last_run_id is None
    if all_declined:
        assert CANCELLED["content"] in capsys.readouterr().out


@pytest.mark.parametrize("last", ["skip", "cancel"])
def test_cli_submits_nothing_without_a_decision_and_keeps_the_pause_for_commit(monkeypatch, last):
    client, select, _ = _client(monkeypatch, ("skip", last), stream=False)

    assert _run(monkeypatch, client, False, CANCELLED) == []
    assert client._last_run_id == "run"
    assert [tool["tool_call_id"] for tool in client._last_tools_payload] == ["call-0", "call-1", "call-legacy"]


def test_cli_reviews_other_approval_tools_read_only_and_defaults_to_decline(monkeypatch):
    client, select, legacy_editor = _client(monkeypatch, ("decline",), stream=False)
    arguments = {"prompt": "Fix the build", "repo_path": "."}
    paused = {"run_id": "run", "status": "paused", "tools": [_pending("code", "run_claude_code", arguments)]}
    submitted = []
    monkeypatch.setattr(
        "requests.post",
        lambda url, *, json, timeout: (
            submitted.append(json) or SimpleNamespace(status_code=200, json=lambda: CANCELLED)
        ),
    )

    client.handle_single_response(paused)

    [[], options] = select.call_args
    assert options["default"] == "decline"
    assert [getattr(choice, "value", None) for choice in options["choices"]] == [
        "confirm",
        "decline",
        "skip",
        None,
        "cancel",
    ]
    legacy_editor.assert_not_called()
    [payload] = submitted
    assert payload["updated_tools"] == [{**paused["tools"][0], "confirmed": False}]


@pytest.mark.parametrize("tools", [[], [COMPLETED], [LEGACY], [COMPLETED, LEGACY]])
def test_cli_leaves_batches_of_listed_tools_to_the_existing_flow(monkeypatch, tools):
    client, select, legacy_editor = _client(monkeypatch, (), stream=False)

    assert client._approve_pending_from_metadata({"run_id": "run", "status": "paused", "tools": tools}) is False
    select.assert_not_called()
    legacy_editor.assert_not_called()
    assert client._last_run_id is None


def test_cli_legacy_commit_chain_does_not_open_the_metadata_flow(monkeypatch):
    client, select, legacy_editor = _client(monkeypatch, (), stream=True)

    # The legacy chain passes only the last tool of a commit stream; it stays on the legacy flow.
    client._prompt_and_commit_tools_if_needed({"run_id": "run", "tools": [_pending(0, "update_task", UPDATE)]})

    select.assert_not_called()
    assert client._last_run_id is None


def test_cli_treats_a_commit_stream_that_breaks_after_acceptance_as_submitted(monkeypatch):
    client, select, _ = _client(monkeypatch, ("confirm", "confirm"), stream=True)
    paused = {**PAUSED, "tools": PAUSED["tools"][:3]}

    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield _sse({"event": "RunContinued", "run_id": "run"}).encode()
            raise httpx.ReadError("connection lost")

    def transport(request):
        if request.url.path.endswith("/commit"):
            return httpx.Response(200, stream=BrokenStream())
        return httpx.Response(200, text=_sse(paused))

    transport_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: transport_client(transport=httpx.MockTransport(transport)))
    client.handle_sse_stream("Update task")

    assert select.call_count == 2
    assert (client._last_run_id, client._last_tools_payload) == (None, [])
