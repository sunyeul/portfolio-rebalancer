import httpx
import pytest

from integrations.toss.config import TossApiConfig
from integrations.toss.auth import TossAuthorizedReader, TossTokenProvider
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


def test_token_provider_caches_token_in_memory_until_refresh_window(config):
    token_calls = 0
    now = [100.0]

    def handler(request):
        nonlocal token_calls
        token_calls += 1
        form = dict(httpx.QueryParams(request.content.decode()))
        assert form == {
            "grant_type": "client_credentials",
            "client_id": "client-id",
            "client_secret": "client-secret",
        }
        return httpx.Response(
            200,
            json={
                "access_token": f"token-{token_calls}",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        )

    transport = _transport(config, handler)
    provider = TossTokenProvider(config, transport, clock=lambda: now[0])

    assert provider.access_token() == "token-1"
    assert provider.access_token() == "token-1"
    assert token_calls == 1

    now[0] = 3701.0
    assert provider.access_token() == "token-2"
    assert token_calls == 2
    assert "token-2" not in repr(provider)


def test_authorized_reader_sends_required_headers_without_exposing_them(config):
    seen_headers = {}

    def handler(request):
        if request.url.path == "/oauth2/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "access-token",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        seen_headers.update(request.headers)
        return httpx.Response(200, json={"result": []})

    transport = _transport(config, handler)
    provider = TossTokenProvider(config, transport, clock=lambda: 100.0)
    reader = TossAuthorizedReader(config, transport, provider)

    assert reader.get_json("/api/v1/holdings") == {"result": []}
    assert seen_headers["authorization"] == "Bearer access-token"
    assert seen_headers["x-tossinvest-account"] == "42"
    assert "access-token" not in repr(reader)
    assert "42" not in repr(reader)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"access_token": "", "expires_in": 3600},
        {"access_token": "token", "expires_in": 0},
        {"access_token": "token", "expires_in": "invalid"},
    ],
)
def test_token_provider_rejects_malformed_token_response(config, payload):
    def handler(request):
        return httpx.Response(200, json=payload)

    provider = TossTokenProvider(config, _transport(config, handler))

    with pytest.raises(TossTransportError, match="invalid token response"):
        provider.access_token()
