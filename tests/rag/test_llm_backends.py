"""Tests for configurable local LLM backends."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from ecommerce_rag.rag.prompting.grounded import assemble_interpretation
from ecommerce_rag.rag.prompting.local_llm import (
    FallbackGenerator,
    OllamaGenerator,
    OpenAIResponsesGenerator,
    build_generator,
)


def test_ollama_generator_requests_deterministic_json_and_unloads_model() -> None:
    response = Mock()
    response.json.return_value = {
        "message": {"content": "  {\"ok\": true}  "},
        "prompt_eval_count": 42,
        "eval_count": 7,
    }
    response.raise_for_status.return_value = None
    messages = [{"role": "user", "content": "test"}]

    with patch("ecommerce_rag.rag.prompting.local_llm.requests.post", return_value=response) as post:
        generator = OllamaGenerator(
            "http://ollama:11434/", "qwen-test:q4", 128, 42, "0"
        )
        output = generator.generate(messages)

    assert output == '{"ok": true}'
    _, kwargs = post.call_args
    assert kwargs["timeout"] == 42
    assert kwargs["json"]["format"]["required"] == ["review_summary"]
    assert kwargs["json"]["stream"] is False
    assert kwargs["json"]["keep_alive"] == "0"
    assert kwargs["json"]["options"]["num_predict"] == 128
    assert kwargs["json"]["options"]["temperature"] == 0
    assert generator.last_usage == {"input_tokens": 42, "output_tokens": 7}


def test_generator_factory_rejects_unknown_provider(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_generator({"provider": "unknown"})


def test_openai_generator_uses_responses_api_and_strict_json() -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "output": [
            {"type": "reasoning", "content": []},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": ' {"review_summary":"Sintesi migliore"} ',
                    }
                ],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    with patch(
        "ecommerce_rag.rag.prompting.local_llm.requests.post",
        return_value=response,
    ) as post:
        generator = OpenAIResponsesGenerator("secret-test-key")
        output = generator.generate([{"role": "user", "content": "test"}])

    assert output == '{"review_summary":"Sintesi migliore"}'
    assert generator.last_usage == {"input_tokens": 10, "output_tokens": 5}
    _, kwargs = post.call_args
    assert post.call_args.args[0] == "https://api.openai.com/v1/responses"
    assert kwargs["headers"]["Authorization"] == "Bearer secret-test-key"
    assert "secret-test-key" not in post.call_args.args[0]
    assert kwargs["json"]["model"] == "gpt-5.6-terra"
    assert kwargs["json"]["reasoning"] == {"effort": "none"}
    assert kwargs["json"]["store"] is False
    output_format = kwargs["json"]["text"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    assert output_format["schema"]["additionalProperties"] is False


def test_fallback_generator_uses_ollama_when_openai_is_unavailable() -> None:
    primary = Mock(provider="openai", model_name="gpt-5.6-terra")
    primary.generate.side_effect = RuntimeError("API unavailable")
    fallback = Mock(provider="ollama", model_name="qwen-test:q4")
    fallback.generate.return_value = '{"review_summary":"Sintesi locale"}'
    fallback.last_usage = {"input_tokens": 20, "output_tokens": 4}

    generator = FallbackGenerator(primary, fallback)
    output = generator.generate([{"role": "user", "content": "test"}])

    assert output == '{"review_summary":"Sintesi locale"}'
    assert generator.last_provider == "ollama"
    assert generator.last_model == "qwen-test:q4"
    assert generator.last_usage == {"input_tokens": 20, "output_tokens": 4}


def test_factory_builds_openai_with_local_fallback(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    config = {
        "provider": "ollama",
        "base_url": "http://ollama:11434",
        "model": "qwen-test:q4",
        "max_new_tokens": 128,
        "openai_model": "gpt-5.6-terra",
    }

    generator = build_generator(config)

    assert isinstance(generator, FallbackGenerator)
    assert isinstance(generator.primary, OpenAIResponsesGenerator)
    assert isinstance(generator.fallback, OllamaGenerator)


def test_factory_without_openai_key_keeps_project_locally_usable(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    generator = build_generator(
        {
            "provider": "ollama",
            "base_url": "http://ollama:11434",
            "model": "qwen-test:q4",
            "max_new_tokens": 128,
        }
    )

    assert isinstance(generator, OllamaGenerator)


def test_review_summary_is_not_extended_with_generic_template_text() -> None:
    payload = assemble_interpretation(
        {
            "review_summary": (
                "I clienti segnalano ordini non consegnati. Una recensione "
                "riferisce che il pacco non è mai arrivato"
            )
        },
        ["a"],
        "explain_rating_drop",
    )

    assert payload["interpretation"] == (
        "Le recensioni recuperate descrivono ordini non consegnati. Una recensione "
        "riferisce che il pacco non è mai arrivato."
    )
    assert "contesto qualitativo" not in payload["interpretation"]
    assert payload["review_ids"] == ["a"]
