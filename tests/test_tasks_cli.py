import json
import logging
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest


@pytest.mark.parametrize("decisions", [("confirm", "decline"), ("decline", "decline")])
@pytest.mark.parametrize("stream", [False, True])
def test_cli_submits_each_pause_once_and_preserves_pending_decisions(monkeypatch, capsys, decisions, stream):
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
    update = {"list_id": "opaque", "task_id": "one", "changes": {"notes": None, "title": "null"}}
    delete = {"list_id": "opaque", "task_id": "two"}
    tools = [
        {
            "tool_name": name,
            "tool_call_id": f"call-{i}",
            "tool_args": arguments,
            "requires_confirmation": True,
        }
        for i, (name, arguments) in enumerate((("update_task", update), ("delete_task", delete)))
    ]
    completed = {"tool_name": "create_task", "tool_call_id": "completed", "requires_confirmation": False}
    legacy = {"tool_name": "create_contact", "tool_call_id": "legacy", "requires_confirmation": True}
    paused = {"event": "RunPaused", "run_id": "run", "status": "paused", "tools": [completed, *tools, legacy]}
    cancelled = {"status": "cancelled", "content": "Tool execution cancelled by user."}
    all_declined = decisions == ("decline", "decline")
    submitted = []

    def sse(*events):
        return "".join(f"event: message\ndata: {json.dumps(event)}\n\n" for event in events)

    if stream:
        transport_client = httpx.Client

        def transport(request):
            if request.url.path.endswith("/commit"):
                submitted.append(json.loads(request.content))
                if all_declined:
                    return httpx.Response(200, json=cancelled)
                return httpx.Response(200, text=sse(paused))
            # A later event carrying tools must not replace the RunPaused batch.
            return httpx.Response(200, text=sse(paused, {"event": "ToolCallCompleted", "tools": [completed]}))

        monkeypatch.setattr(
            httpx, "Client", lambda **kwargs: transport_client(transport=httpx.MockTransport(transport))
        )
        client.handle_sse_stream("Update task")
    else:

        def post(url, *, json, timeout):
            submitted.append(json)
            return SimpleNamespace(status_code=200, json=lambda: cancelled if all_declined else paused)

        monkeypatch.setattr("requests.post", post)
        client.handle_single_response(paused)

    [payload] = submitted
    assert payload["run_id"] == "run"
    assert payload["stream"] is stream
    assert [tool["tool_call_id"] for tool in payload["updated_tools"]] == ["call-0", "call-1"]
    assert [tool["tool_args"] for tool in payload["updated_tools"]] == [update, delete]
    assert [tool["confirmed"] for tool in payload["updated_tools"]] == [decisions[0] == "confirm", False]
    assert select.call_count == 2
    legacy_editor.assert_called_once_with(legacy)
    assert client._last_run_id is None
    if all_declined:
        assert cancelled["content"] in capsys.readouterr().out
