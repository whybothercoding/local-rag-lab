"""Pluggable LLM factory supporting Ollama, local OpenAI-compatible engines, and in-process synthesizer.

Ensures the Local Enterprise Hub operates seamlessly:
- Uses local Ollama (default `qwen3:8b`, settings.chat_model) when the daemon is active.
- Connects to local OpenAI-compatible endpoints (vLLM, LM Studio, llama.cpp server) when configured.
- Gracefully falls back to an in-process grounded context synthesizer when no external daemon is running.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from pydantic import Field

from rag_lab.config import settings
from rag_lab.lang import STOPWORDS, abstain_reply, detect_language, same_word as _same_word, words

# Words the extractive synthesizer ignores when matching a question to context sentences.
_EXTRACTOR_IGNORE = STOPWORDS | {"is", "our", "the", "are", "for", "to", "in", "a", "an", "of", "and"}

logger = logging.getLogger("llm_provider")


class DeterministicContextLLM(BaseChatModel):
    """In-process, zero-dependency synthesizer that extracts and constructs answers

    directly from provided context chunks. Used for air-gapped demo runs, testing,
    and automatic fallback when local inference daemons are unreachable.
    """

    model_name: str = "deterministic-context-synthesizer"

    def _extract_answer(self, prompt: str) -> str:
        # Prompt structure expected:
        # Context:\n{context}\n\nQuestion: {q}\n\nAnswer concisely based strictly on the context:
        context = ""
        question = ""

        if "Context:" in prompt and "\n\nQuestion:" in prompt:
            ctx_part, q_part = prompt.rsplit("\n\nQuestion:", 1)
            if ctx_part.startswith("Context:"):
                ctx_part = ctx_part[len("Context:"):].strip()
            else:
                ctx_part = ctx_part.replace("Context:", "", 1).strip()
            # Remove document tags, separators, and markdown headings: only document sentences may be quoted.
            ctx_part = re.sub(r"</?documents?\b[^>]*>", "\n", ctx_part)
            ctx_part = re.sub(r"(?m)^\s*#.*$", "", ctx_part)
            chunks = [c.strip() for c in re.split(r"\n\s*---\s*\n|\n{2,}", ctx_part) if c.strip()]
            context = " ".join(" ".join(c.split()) for c in chunks)

            if "\n\nAnswer" in q_part:
                question = q_part.rsplit("\n\nAnswer", 1)[0].strip()
            elif "Answer" in q_part:
                question = q_part.rsplit("Answer", 1)[0].strip()
            else:
                question = q_part.strip()
        else:
            context = prompt
            question = ""

        if not context:
            return abstain_reply(detect_language(question))

        # Find the most relevant sentences in context matching question tokens
        q_tokens = set(words(question)) - _EXTRACTOR_IGNORE
        sentences = re.split(r"(?<=[.!?])\s+", context)

        matched_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_tokens = set(words(s_clean))
            overlap = sum(1 for q in q_tokens if any(_same_word(q, t) for t in s_tokens))
            if overlap > 0:
                matched_sentences.append((overlap, s_clean))

        if matched_sentences:
            # Keep only the sentences tied for the strongest overlap, so a weak second match cannot pad the answer.
            top_overlap = max(overlap for overlap, _ in matched_sentences)
            best_sentences = [text for overlap, text in matched_sentences if overlap == top_overlap][:2]
            return " ".join(best_sentences)

        # Fallback to the first sentence of context
        first_sentence = sentences[0].strip() if sentences else context[:200]
        return f"Based on available documentation: {first_sentence}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_text = messages[-1].content if messages else ""
        if isinstance(prompt_text, list):
            prompt_text = " ".join(str(p) for p in prompt_text)

        answer = self._extract_answer(str(prompt_text))
        gen = ChatGeneration(message=AIMessage(content=answer))
        return ChatResult(generations=[gen])

    @property
    def _llm_type(self) -> str:
        return "deterministic-context-synthesizer"


# A reasoning model's thinking, if a server leaves it in the text: it must never reach the graders or the reader.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks, and a block that never closed (a cut-off answer is not an answer)."""
    return _THINK_OPEN.sub("", _THINK_BLOCK.sub("", text)).strip()


def _template_kwargs() -> dict[str, Any]:
    """LOCAL_LLM_TEMPLATE_KWARGS as a dict. Empty means send none, for servers that reject unknown request fields."""
    raw = settings.local_llm_template_kwargs.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LOCAL_LLM_TEMPLATE_KWARGS must be a JSON object or empty ({exc}).") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LOCAL_LLM_TEMPLATE_KWARGS must be a JSON object or empty.")
    return parsed


def _openai_role(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, AIMessage):
        return "assistant"
    return "user"


class OpenAICompatibleChat(BaseChatModel):
    """Chat model for any local server that speaks the OpenAI chat-completions API (LM Studio, vLLM, llama.cpp).

    Built on httpx, so it needs no vendor SDK. `transport` lets tests run against an in-memory handler.
    """

    base_url: str
    model_name: str = "qwen3:8b"
    # A secret: kept out of repr() so it cannot reach a log line or a traceback by accident.
    api_key: str = Field(default="local", repr=False)
    # Read when the client is created, so a settings override applies.
    timeout_sec: float = Field(default_factory=lambda: settings.ollama_timeout_sec)
    transport: Any = None
    # Merged into the request as chat_template_kwargs; the default turns Qwen3's thinking off (LOCAL_LLM_TEMPLATE_KWARGS).
    template_kwargs: dict[str, Any] = Field(default_factory=_template_kwargs)

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "temperature": 0.0,
            "messages": [{"role": _openai_role(m), "content": str(m.content)} for m in messages],
        }
        if self.template_kwargs:
            payload["chat_template_kwargs"] = self.template_kwargs
        with httpx.Client(timeout=self.timeout_sec, transport=self.transport) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        response.raise_for_status()
        choices = response.json().get("choices") or []
        if not choices:
            raise ValueError("The chat server returned no choices.")
        # Only `content` is the answer. A server that separates its reasoning (reasoning / reasoning_content) is ignored
        # on purpose, and any <think> block left in the text is removed.
        content = choices[0].get("message", {}).get("content") or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=strip_thinking(str(content))))])


@dataclass(frozen=True)
class OpenAIProbe:
    models: set[str] | None
    error: str = ""


def probe_openai(base_url: str | None = None, api_key: str | None = None, timeout_sec: float = 2.0,
                 transport: Any = None) -> OpenAIProbe:
    """Ask an OpenAI-compatible server which models it serves (GET /models). Never raises; `error` says why not."""
    url = f"{(base_url or settings.local_llm_url).rstrip('/')}/models"
    key = settings.local_llm_api_key if api_key is None else api_key
    try:
        with httpx.Client(timeout=timeout_sec, transport=transport) as client:
            response = client.get(url, headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        return OpenAIProbe(None, f"unreachable ({type(exc).__name__})")
    if response.status_code in (401, 403):
        return OpenAIProbe(None, f"HTTP {response.status_code}: the server refused LOCAL_LLM_API_KEY")
    if response.status_code != 200:
        return OpenAIProbe(None, f"HTTP {response.status_code}")
    try:
        return OpenAIProbe({str(m["id"]) for m in response.json().get("data", [])})
    except (ValueError, KeyError, TypeError, AttributeError):
        return OpenAIProbe(None, "an answer that is not an OpenAI model list")


def openai_model_name() -> str:
    """The model id sent to an OpenAI-compatible server: LOCAL_LLM_MODEL, else the shared chat model setting."""
    return settings.local_llm_model or settings.chat_model


def _openai_chat(transport: Any = None) -> BaseChatModel:
    return OpenAICompatibleChat(
        base_url=settings.local_llm_url,
        model_name=openai_model_name(),
        api_key=settings.local_llm_api_key,
        transport=transport,
    )


def is_ollama_alive(base_url: str | None = None, timeout_sec: float = 0.5) -> bool:
    """Fast check whether local Ollama server is responding."""
    base_url = settings.ollama_base_url if base_url is None else base_url
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        resp = httpx.get(url, timeout=timeout_sec)
        return resp.status_code == 200
    except Exception:
        return False


def ollama_models(base_url: str | None = None, timeout_sec: float = 0.5) -> set[str] | None:
    """Names of the models pulled into the local Ollama server, or None when it is unreachable.

    Names are returned both as listed ("nomic-embed-text:latest") and without a ":latest" tag, so a
    configured name matches either way.
    """
    base_url = settings.ollama_base_url if base_url is None else base_url
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout_sec)
        if resp.status_code != 200:
            return None
        names: set[str] = set()
        for entry in resp.json().get("models", []):
            name = str(entry.get("name") or entry.get("model") or "")
            if name:
                names.add(name)
                if name.endswith(":latest"):
                    names.add(name[: -len(":latest")])
        return names
    except Exception:
        return None


def _ollama_chat() -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        # Reasoning models answer directly: a thinking pass multiplies latency for short policy answers, and
        # its text must never reach the graders. Models without a thinking mode ignore the flag.
        reasoning=False,
        client_kwargs={"timeout": settings.ollama_timeout_sec},
    )


def llm_is_fallback(llm: BaseChatModel, provider: str | None = None) -> bool:
    """True when `llm` is the in-process extractive synthesizer and nobody asked for it.

    Choosing LLM_PROVIDER=deterministic is a deliberate setting. Landing on it because the model
    server is unreachable is a degradation that the reviewer and the user must be told about.
    """
    active = (provider or settings.llm_provider).lower().strip()
    return isinstance(llm, DeterministicContextLLM) and active != "deterministic"


def get_chat_llm(provider: str | None = None) -> BaseChatModel:
    """Factory returning the appropriate chat model based on environment and runtime availability."""
    active = (provider or settings.llm_provider).lower().strip()

    if active == "openai":
        return _openai_chat()

    if active == "deterministic":
        return DeterministicContextLLM()

    if active == "ollama":
        return _ollama_chat()

    # "auto" mode: probe Ollama; if running, use it; otherwise fallback to DeterministicContextLLM
    if is_ollama_alive(settings.ollama_base_url):
        try:
            return _ollama_chat()
        except Exception as exc:
            logger.warning("Could not initialize ChatOllama: %s. Using in-process synthesizer.", exc)

    logger.info("Ollama is offline or unreachable; using in-process context synthesizer.")
    return DeterministicContextLLM()
