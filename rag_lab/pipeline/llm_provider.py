"""Pluggable LLM factory supporting Ollama, local OpenAI-compatible engines, and in-process synthesizer.

Ensures the Local Enterprise Hub operates seamlessly:
- Uses local Ollama (`qwen2.5-coder:7b`) when the daemon is active.
- Connects to local OpenAI-compatible endpoints (vLLM, LM Studio, llama.cpp server) when configured.
- Gracefully falls back to an in-process grounded context synthesizer when no external daemon is running.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from pydantic import Field

from rag_lab.config import settings
from rag_lab.lang import STOPWORDS, words

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
            # Remove separator markers
            chunks = [c.strip() for c in ctx_part.split("---") if c.strip()]
            context = " ".join(chunks)

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
            return "I could not find relevant documentation in this workspace to answer your query."

        # Find the most relevant sentences in context matching question tokens
        q_tokens = set(words(question)) - _EXTRACTOR_IGNORE
        sentences = re.split(r"(?<=[.!?])\s+", context)

        matched_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_tokens = set(words(s_clean))
            overlap = len(q_tokens & s_tokens)
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
    model_name: str = "qwen2.5-coder:7b"
    api_key: str = "local"
    # Read when the client is created, so a settings override applies.
    timeout_sec: float = Field(default_factory=lambda: settings.ollama_timeout_sec)
    transport: Any = None

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
        payload = {
            "model": self.model_name,
            "temperature": 0.0,
            "messages": [{"role": _openai_role(m), "content": str(m.content)} for m in messages],
        }
        with httpx.Client(timeout=self.timeout_sec, transport=self.transport) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(content)))])


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
        return OpenAICompatibleChat(base_url=settings.local_llm_url, model_name=settings.chat_model)

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
