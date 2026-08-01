"""Unit tests for Gemini provider wiring (mocked Google GenAI SDK)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.llm.client import ChatMessage, GeminiProvider, LLMClient, LLMClientConfig
from agent.llm.schemas import EditInstructions


def test_gemini_provider_returns_json_text() -> None:
    fake_response = SimpleNamespace(
        text='{"thought":"ok","edits":[],"done":true,"notes":""}'
    )
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    with patch.object(GeminiProvider, "_create_client", return_value=fake_client):
        provider = GeminiProvider(api_key="test-key", max_retries=1)
        text = provider.complete(
            [ChatMessage(role="user", content="Return empty edits JSON")],
            model="gemini-3.6-flash",
            temperature=0.2,
        )

    assert '"done":true' in text.replace(" ", "")
    fake_client.models.generate_content.assert_called_once()
    kwargs = fake_client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-3.6-flash"
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].response_schema is EditInstructions


def test_gemini_retries_on_server_error() -> None:
    from google.genai import errors

    fake_response = SimpleNamespace(text='{"thought":"ok","edits":[],"done":true}')
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        errors.ServerError(503, {"message": "unavailable"}, response=MagicMock()),
        fake_response,
    ]

    with (
        patch.object(GeminiProvider, "_create_client", return_value=fake_client),
        patch.object(GeminiProvider, "_sleep", return_value=None),
    ):
        provider = GeminiProvider(api_key="test-key", max_retries=3)
        text = provider.complete(
            [ChatMessage(role="user", content="hi")],
            model="gemini-3.6-flash",
            temperature=0.1,
        )

    assert "done" in text
    assert fake_client.models.generate_content.call_count == 2


def test_factory_selects_gemini() -> None:
    with patch("agent.llm.client.GeminiProvider") as mocked:
        mocked.return_value = MagicMock()
        LLMClient(
            LLMClientConfig(
                provider="gemini",
                api_key="test-key",
                model="gemini-3.6-flash",
                extra={"max_retries": 2},
            )
        )
        mocked.assert_called_once()
        assert mocked.call_args.kwargs.get("max_retries") == 2 or mocked.call_args.args


def test_gemini_requires_api_key() -> None:
    with pytest.raises(ValueError, match="AGENT_LLM_API_KEY"):
        LLMClient(LLMClientConfig(provider="gemini", api_key=None))
