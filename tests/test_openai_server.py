"""LLM_PROVIDER=openai against a stub of a vLLM-style server, over a Unix domain socket.

A real HTTP server speaks the parts of the OpenAI API this provider uses (GET /v1/models, POST /v1/chat/completions,
bearer-token authentication, a reasoning model's habits). The socket is a Unix domain one, which the offline test run
allows, so these tests run in both suites without opening a TCP port.
"""
import http.server
import json
import shutil
import socketserver
import tempfile
import threading
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from rag_lab.config import settings
from rag_lab.pipeline import llm_provider
from rag_lab.pipeline.generation import generate_node
from rag_lab.pipeline.llm_provider import (
    OpenAICompatibleChat,
    get_chat_llm,
    openai_model_name,
    probe_openai,
    strip_thinking,
)

MODEL = "Qwen/Qwen3-8B"
KEY = "sk-stub-secret-4711"


class StubServer:
    """What vLLM does that matters here: bearer auth, a model list, and chat completions."""

    def __init__(self, directory: str, api_key: str = KEY, models=(MODEL,)) -> None:
        self.path = str(Path(directory) / "s.sock")
        self.api_key = api_key
        self.models = list(models)
        self.requests: list[dict] = []
        self.reply = lambda payload: (200, {"choices": [{"message": {"content": "Five minutes."}}]})
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # a Unix-socket peer has no address to log
                pass

            def _authorised(self) -> bool:
                return not stub.api_key or self.headers.get("Authorization") == f"Bearer {stub.api_key}"

            def _send(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                stub.requests.append({"method": "GET", "path": self.path, "auth": self.headers.get("Authorization")})
                if not self._authorised():
                    return self._send(401, {"error": "unauthorized"})
                self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in stub.models]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"method": "POST", "path": self.path, "auth": self.headers.get("Authorization"),
                                      "body": payload})
                if not self._authorised():
                    return self._send(401, {"error": "unauthorized"})
                status, body = stub.reply(payload)
                self._send(status, body)

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

        self._server = Server(self.path, Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def transport(self) -> httpx.HTTPTransport:
        return httpx.HTTPTransport(uds=self.path)

    def chat(self, **kw) -> OpenAICompatibleChat:
        options = {"base_url": "http://vllm.test/v1", "model_name": MODEL, "api_key": KEY, "transport": self.transport} | kw
        return OpenAICompatibleChat(**options)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def server():
    directory = tempfile.mkdtemp(prefix="rs-", dir="/tmp")  # a Unix socket path must be short
    stub = StubServer(directory)
    yield stub
    stub.close()
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def configured(server, monkeypatch):
    """Settings as an operator would set them for LLM_PROVIDER=openai, with the transport pointed at the stub."""
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "local_llm_url", "http://vllm.test/v1")
    monkeypatch.setattr(settings, "local_llm_model", MODEL)
    monkeypatch.setattr(settings, "local_llm_api_key", KEY)
    original = llm_provider._openai_chat
    monkeypatch.setattr(llm_provider, "_openai_chat", lambda transport=None: original(transport=server.transport))
    return server


# --- the request -----------------------------------------------------------------------------------------------------

def test_the_request_is_a_chat_completion_with_the_key_the_model_and_thinking_off(server):
    reply = server.chat().invoke([SystemMessage(content="Be brief."), HumanMessage(content="Rollback SLA?")])
    assert reply.content == "Five minutes."
    request = server.requests[-1]
    assert request["path"] == "/v1/chat/completions" and request["auth"] == f"Bearer {KEY}"
    body = request["body"]
    assert body["model"] == MODEL and body["temperature"] == 0.0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}, "Qwen3 thinks by default on vLLM"


def test_settings_drive_the_provider_the_factory_builds(configured):
    llm = get_chat_llm()
    assert isinstance(llm, OpenAICompatibleChat)
    assert llm.invoke([HumanMessage(content="hi")]).content == "Five minutes."
    assert configured.requests[-1]["body"]["model"] == MODEL and configured.requests[-1]["auth"] == f"Bearer {KEY}"


def test_the_model_id_is_its_own_setting_falling_back_to_the_shared_one(monkeypatch):
    monkeypatch.setattr(settings, "local_llm_model", "")
    monkeypatch.setattr(settings, "chat_model", "qwen3:8b")
    assert openai_model_name() == "qwen3:8b"
    monkeypatch.setattr(settings, "local_llm_model", MODEL)
    assert openai_model_name() == MODEL


def test_template_kwargs_are_configurable_and_can_be_left_out(server, monkeypatch):
    monkeypatch.setattr(settings, "local_llm_template_kwargs", "")
    server.chat(template_kwargs=llm_provider._template_kwargs()).invoke([HumanMessage(content="hi")])
    assert "chat_template_kwargs" not in server.requests[-1]["body"], "for a server that rejects unknown fields"

    monkeypatch.setattr(settings, "local_llm_template_kwargs", '{"thinking": false}')
    server.chat(template_kwargs=llm_provider._template_kwargs()).invoke([HumanMessage(content="hi")])
    assert server.requests[-1]["body"]["chat_template_kwargs"] == {"thinking": False}


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", '"text"'])
def test_bad_template_kwargs_are_a_labelled_error(monkeypatch, bad):
    monkeypatch.setattr(settings, "local_llm_template_kwargs", bad)
    with pytest.raises(ValueError, match="LOCAL_LLM_TEMPLATE_KWARGS"):
        llm_provider._template_kwargs()


# --- the answer ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("returned,expected", [
    ("<think>The user asks about rollback.</think>\nFive minutes.", "Five minutes."),
    ("<THINK>x\ny</THINK>Five minutes.", "Five minutes."),
    ("<think>cut off before it finished", ""),
    ("Five minutes.", "Five minutes."),
])
def test_thinking_left_in_the_text_never_reaches_the_reader(server, returned, expected):
    server.reply = lambda payload: (200, {"choices": [{"message": {"content": returned}}]})
    assert server.chat().invoke([HumanMessage(content="hi")]).content == expected


def test_a_separate_reasoning_field_is_ignored_and_a_null_answer_is_empty_not_the_word_none(server):
    server.reply = lambda payload: (200, {"choices": [{"message": {"content": "Five minutes.", "reasoning_content": "secret chain of thought",
                                                                   "reasoning": "more"}}]})
    assert server.chat().invoke([HumanMessage(content="hi")]).content == "Five minutes."
    server.reply = lambda payload: (200, {"choices": [{"message": {"content": None, "reasoning_content": "only thoughts"}}]})
    assert server.chat().invoke([HumanMessage(content="hi")]).content == ""


def test_strip_thinking_on_its_own():
    assert strip_thinking("a <think>b</think> c <think>d</think> e") == "a  c  e"
    assert strip_thinking("") == ""


# --- failures ------------------------------------------------------------------------------------------------------------------

def test_the_wrong_key_is_an_http_error_that_does_not_echo_the_key(server):
    with pytest.raises(httpx.HTTPStatusError) as caught:
        server.chat(api_key="wrong-key").invoke([HumanMessage(content="hi")])
    assert caught.value.response.status_code == 401 and "wrong-key" not in str(caught.value)


def test_server_errors_and_empty_answers_are_errors(server):
    server.reply = lambda payload: (500, {"error": "boom"})
    with pytest.raises(httpx.HTTPStatusError):
        server.chat().invoke([HumanMessage(content="hi")])
    server.reply = lambda payload: (200, {"choices": []})
    with pytest.raises(ValueError, match="no choices"):
        server.chat().invoke([HumanMessage(content="hi")])


def test_an_unreachable_server_is_a_connection_error(tmp_path):
    llm = OpenAICompatibleChat(base_url="http://vllm.test/v1", transport=httpx.HTTPTransport(uds="/tmp/no-such-socket"))
    with pytest.raises(httpx.ConnectError):
        llm.invoke([HumanMessage(content="hi")])


def test_the_api_key_stays_out_of_repr(server):
    assert KEY not in repr(server.chat()) and KEY not in str(server.chat())


# --- an answer through the pipeline ----------------------------------------------------------------------------------------------

def test_generation_answers_through_the_server(configured):
    configured.reply = lambda payload: (200, {"choices": [{"message": {"content": "<think>checking</think>The rollback SLA is under 5 minutes."}}]})
    state = generate_node({
        "documents": ["Rollback SLA is strictly under 5 minutes."],
        "sanitized_question": "What is the rollback SLA?",
        "degraded": [],
        "telemetry": {},
    })
    assert state["generation"] == "The rollback SLA is under 5 minutes." and "<think>" not in state["generation"]
    assert not state["telemetry"].get("generation_fallback")
    sent = configured.requests[-1]["body"]["messages"]
    assert "Rollback SLA is strictly under 5 minutes." in json.dumps(sent), "the retrieved context reached the server"


def test_a_failing_server_gives_the_labelled_extractive_fallback(configured):
    configured.reply = lambda payload: (503, {"error": "overloaded"})
    state = generate_node({
        "documents": ["Rollback SLA is strictly under 5 minutes."],
        "sanitized_question": "What is the rollback SLA?",
        "degraded": [],
        "telemetry": {},
    })
    assert "5 minutes" in state["generation"] and state["telemetry"]["generation_fallback"] is True
    assert any(note.startswith("generation_fallback: HTTPStatusError") for note in state["degraded"])


# --- the status probe --------------------------------------------------------------------------------------------------------------

def test_probe_lists_the_served_models(server):
    probe = probe_openai("http://vllm.test/v1", KEY, transport=server.transport)
    assert probe.models == {MODEL} and probe.error == ""
    assert server.requests[-1]["path"] == "/v1/models"


def test_probe_says_why_it_failed(server):
    refused = probe_openai("http://vllm.test/v1", "wrong", transport=server.transport)
    assert refused.models is None and "401" in refused.error and "LOCAL_LLM_API_KEY" in refused.error
    gone = probe_openai("http://vllm.test/v1", KEY, transport=httpx.HTTPTransport(uds="/tmp/no-such-socket"))
    assert gone.models is None and "unreachable" in gone.error
    server.models = []
    assert probe_openai("http://vllm.test/v1", KEY, transport=server.transport).models == set()
