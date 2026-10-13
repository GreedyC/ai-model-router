import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import asyncio
from unittest.mock import AsyncMock

# Add sdk/python to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modelrouter import (
    ModelRouter,
    AsyncModelRouter,
    ChatMessage,
    ChatCompletion,
    RoutingDecision,
    AuthenticationError,
    RateLimitError,
)


def test_sdk_sync_chat():
    client = ModelRouter(api_key="test-key", base_url="http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "decision": {
            "decision_id": "dec-123",
            "request_id": "req-123",
            "selected_model": "mock-balanced",
            "selected_model_name": "Mock Balanced",
            "provider": "mock",
            "tier": "BALANCED",
            "confidence": 0.95,
            "policy_used": "balanced",
            "reasons": ["High quality"],
            "estimated_cost_usd": 0.0001,
            "estimated_latency_ms": 120.0,
        },
        "response": {
            "content": "Simulated output",
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "model": "mock-balanced",
        },
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.chat.completions.create(prompt="Hello")
        assert res.model == "mock-balanced"
        assert res.choices[0].message.content == "Simulated output"


def test_async_client_context_returns_client_and_closes_on_exit():
    async def run():
        client = AsyncModelRouter(api_key="test-key")
        client.close = AsyncMock()
        async with client as entered:
            assert entered is client
            client.close.assert_not_awaited()
        client.close.assert_awaited_once_with()
    asyncio.run(run())


def test_async_client_context_closes_without_suppressing_errors():
    async def run():
        client = AsyncModelRouter(api_key="test-key")
        client.close = AsyncMock()
        with pytest.raises(ValueError, match="context failure"):
            async with client:
                raise ValueError("context failure")
        client.close.assert_awaited_once_with()
    asyncio.run(run())


def test_async_client_close_is_safe_to_repeat():
    async def run():
        client = AsyncModelRouter(api_key="test-key")
        await client.close()
        await client.close()
        async with client:
            pass
    asyncio.run(run())


@pytest.mark.parametrize("status,expected", [(503, "ProviderUnavailableError"), (504, "GatewayTimeoutError")])
@pytest.mark.parametrize("text,body", [(' {"detail": "upstream failure"}', {"detail": "upstream failure"}), ("upstream failure", "upstream failure")])
def test_sdk_outage_errors_preserve_status_message_and_body(status, expected, text, body):
    import modelrouter
    from modelrouter.client import _handle_error_response

    error_type = getattr(modelrouter, expected)
    assert expected in modelrouter.__all__
    assert issubclass(error_type, modelrouter.APIStatusError)
    with pytest.raises(error_type) as error:
        _handle_error_response(status, text)
    assert error.value.status_code == status
    assert error.value.response_body == body
    assert str(error.value) == f"[{status}] Model Router error ({status}): upstream failure"


@pytest.mark.parametrize("status,expected", [(400, "APIStatusError"), (500, "APIStatusError"), (502, "APIStatusError"), (401, "AuthenticationError"), (403, "AuthenticationError"), (429, "RateLimitError"), (422, "RoutingError")])
def test_sdk_existing_error_mapping_is_unchanged(status, expected):
    import modelrouter
    from modelrouter.client import _handle_error_response

    with pytest.raises(getattr(modelrouter, expected)) as error:
        _handle_error_response(status, '{"detail": "failure"}')
    assert type(error.value) is getattr(modelrouter, expected)
    assert error.value.status_code == status


@pytest.mark.parametrize("status,error_name", [(503, "ProviderUnavailableError"), (504, "GatewayTimeoutError")])
@pytest.mark.parametrize("mode", ["chat", "stream", "health"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_sdk_outage_errors_through_public_requests(status, error_name, mode, asynchronous):
    import httpx
    import modelrouter

    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={"detail": "outage"}))
    error_type = getattr(modelrouter, error_name)
    if asynchronous:
        original = httpx.AsyncClient
        async def run():
            with patch("httpx.AsyncClient", side_effect=lambda **kwargs: original(transport=transport, **kwargs)):
                client = AsyncModelRouter(api_key="test-key", base_url="http://example.test")
                with pytest.raises(error_type) as error:
                    if mode == "health":
                        await client.health()
                    else:
                        result = await client.chat.completions.create(prompt="hello", stream=mode == "stream")
                        if mode == "stream":
                            await anext(result)
                assert error.value.status_code == status
                assert error.value.response_body == {"detail": "outage"}
        asyncio.run(run())
    else:
        original = httpx.Client
        with patch("httpx.Client", side_effect=lambda **kwargs: original(transport=transport, **kwargs)):
            client = ModelRouter(api_key="test-key", base_url="http://example.test")
            with pytest.raises(error_type) as error:
                if mode == "health":
                    client.health()
                else:
                    result = client.chat.completions.create(prompt="hello", stream=mode == "stream")
                    if mode == "stream":
                        next(result)
            assert error.value.status_code == status
            assert error.value.response_body == {"detail": "outage"}
