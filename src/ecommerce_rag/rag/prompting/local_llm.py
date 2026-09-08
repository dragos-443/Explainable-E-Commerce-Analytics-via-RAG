"""Local instruction-model backends used for grounded synthesis."""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, List

import requests


class LocalTransformersGenerator:
    provider = "local_transformers"

    def __init__(self, model_name: str, revision: str, max_new_tokens: int = 384):
        self.model_name = model_name
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            torch_dtype=torch.float32,
        )
        self._model.eval()

    def generate(self, messages: List[Dict[str, str]]) -> str:
        self._load()
        import torch

        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(rendered, return_tensors="pt")
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (AttributeError, OSError):
            pass


class OllamaGenerator:
    """Generate constrained JSON through a local Ollama HTTP service."""

    provider = "ollama"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_new_tokens: int = 384,
        timeout_seconds: int = 300,
        keep_alive: str = "0",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self.last_usage: Dict[str, Any] | None = None

    def generate(self, messages: List[Dict[str, str]]) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": {"review_summary": {"type": "string"}},
                    "required": ["review_summary"],
                },
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": 0,
                    "num_predict": self.max_new_tokens,
                    "num_ctx": 4096,
                },
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        self.last_usage = {
            "input_tokens": payload.get("prompt_eval_count"),
            "output_tokens": payload.get("eval_count"),
        }
        try:
            return payload["message"]["content"].strip()
        except (KeyError, TypeError, AttributeError) as exc:
            raise RuntimeError("Ollama returned an invalid chat response") from exc

    def unload(self) -> None:
        # keep_alive=0 releases the model immediately after every generation.
        return None


class OpenAIResponsesGenerator:
    """Generate strict JSON through the OpenAI Responses API."""

    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-5.6-terra",
        api_url: str = "https://api.openai.com/v1/responses",
        max_output_tokens: int = 256,
        timeout_seconds: int = 60,
        reasoning_effort: str = "none",
    ):
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        self.api_key = api_key.strip()
        self.model_name = model_name
        self.api_url = api_url
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self.last_usage: Dict[str, Any] | None = None

    def generate(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "input": messages,
                    "reasoning": {"effort": self.reasoning_effort},
                    "max_output_tokens": self.max_output_tokens,
                    "store": False,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "review_summary",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "review_summary": {"type": "string"}
                                },
                                "required": ["review_summary"],
                                "additionalProperties": False,
                            },
                        }
                    },
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            # Do not include response bodies or request headers: they may contain
            # operational details and the API key must never reach application logs.
            raise RuntimeError("OpenAI Responses API request failed") from exc

        self.last_usage = payload.get("usage")
        texts = []
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    texts.append(content["text"])
        if not texts:
            raise RuntimeError("OpenAI returned no text output")
        return "\n".join(texts).strip()

    def unload(self) -> None:
        return None


class FallbackGenerator:
    """Use a secondary generator only when the primary backend is unavailable."""

    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback
        self.provider = getattr(primary, "provider", "primary")
        self.model_name = getattr(primary, "model_name", None)
        self.last_provider: str | None = None
        self.last_model: str | None = None
        self.last_usage: Dict[str, Any] | None = None

    def generate(self, messages: List[Dict[str, str]]) -> str:
        try:
            output = self.primary.generate(messages)
            self.last_provider = getattr(self.primary, "provider", "primary")
            self.last_model = getattr(self.primary, "model_name", None)
            self.last_usage = getattr(self.primary, "last_usage", None)
            return output
        except Exception:
            output = self.fallback.generate(messages)
            self.last_provider = getattr(self.fallback, "provider", "ollama")
            self.last_model = getattr(self.fallback, "model_name", None)
            self.last_usage = getattr(self.fallback, "last_usage", None)
            return output

    def unload(self) -> None:
        self.primary.unload()
        self.fallback.unload()


def _build_ollama_generator(config: Dict[str, Any]) -> OllamaGenerator:
    return OllamaGenerator(
        config["base_url"],
        config["model"],
        int(config["max_new_tokens"]),
        int(config.get("timeout_seconds", 300)),
        str(config.get("keep_alive", "0")),
    )


def build_generator(config: Dict[str, Any]):
    """Build the configured backend while keeping call sites provider-agnostic."""
    provider = os.getenv("LLM_PROVIDER", str(config.get("provider", ""))).strip().lower()
    if provider == "local_transformers":
        return LocalTransformersGenerator(
            config["model"], config["revision"], int(config["max_new_tokens"])
        )
    if provider == "ollama":
        return _build_ollama_generator(config)
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            # A clone remains usable without cloud credentials.
            return _build_ollama_generator(config)
        primary = OpenAIResponsesGenerator(
            api_key=api_key,
            model_name=os.getenv(
                "OPENAI_MODEL", str(config.get("openai_model", "gpt-5.6-terra"))
            ),
            api_url=str(
                config.get("openai_api_url", "https://api.openai.com/v1/responses")
            ),
            max_output_tokens=int(config.get("openai_max_output_tokens", 256)),
            timeout_seconds=int(config.get("openai_timeout_seconds", 60)),
            reasoning_effort=str(config.get("openai_reasoning_effort", "none")),
        )
        return FallbackGenerator(primary, _build_ollama_generator(config))
    raise ValueError(f"Unsupported LLM provider: {provider!r}")
