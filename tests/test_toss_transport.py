import httpx
import pytest

from integrations.toss.config import TossApiConfig
from integrations.toss.transport import (
    TossRequestBlocked,
    TossTransport,
    TossTransportError,
)


@pytest.fixture()
def config():
    return TossApiConfig(
        client_id="client-id",
        client_secret="client-secret",
        account_seq=42,
    )


def _transport(config, handler):
    client = httpx.Client(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    )
    return TossTransport(config=config, client=client)


def test_transport_permits_oauth_token_post(config):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})

    transport = _transport(config, handler)

    payload = transport.request_json(
        "POST",
        "/oauth2/token",
        data={"grant_type": "client_credentials"},
    )

    assert payload["access_token"] == "token"
    assert seen == [("POST", "/oauth2/token")]


def test_transport_permits_allowlisted_observation_get(config):
    def handler(request):
        return httpx.Response(200, json={"result": []})

    transport = _transport(config, handler)

    assert transport.request_json("GET", "/api/v1/holdings") == {"result": []}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/v1/orders"),
        ("POST", "/api/v1/orders/order-1/cancel"),
        ("POST", "/api/v1/orders/order-1/modify"),
        ("DELETE", "/api/v1/orders/order-1"),
        ("PATCH", "/api/v1/orders/order-1"),
        ("GET", "/api/v1/unknown"),
    ],
)
def test_transport_blocks_every_non_allowlisted_request_before_network(
    config, method, path
):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    transport = _transport(config, handler)

    with pytest.raises(TossRequestBlocked, match="blocked by read-only policy"):
        transport.request_json(method, path)

    assert calls == 0


def test_transport_error_does_not_echo_response_body_or_credentials(config):
    def handler(request):
        return httpx.Response(
            401,
            json={
                "echo_client_secret": "client-secret",
                "echo_authorization": "Bearer access-token",
            },
        )

    transport = _transport(config, handler)

    with pytest.raises(TossTransportError) as exc_info:
        transport.request_json("GET", "/api/v1/accounts")

    message = str(exc_info.value)
    assert "status=401" in message
    assert "client-secret" not in message
    assert "access-token" not in message
