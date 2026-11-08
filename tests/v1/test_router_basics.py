"""Auth, the version header and the error body, on every /v1 route."""

from __future__ import annotations

import io
import json

import httpx
from callback_shared.contract import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_HEADER,
    ErrorResponse,
)
from fastapi import FastAPI

from tests.v1.conftest import V1_TOKEN, client_for


def assert_wire_error(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status
    assert response.headers[CONTRACT_VERSION_HEADER] == CONTRACT_VERSION
    body = ErrorResponse.model_validate(response.json())
    assert body.contract_version == CONTRACT_VERSION
    assert body.error.code == code
    assert body.error.message


async def test_401_without_a_token(v1_app: FastAPI) -> None:
    async with client_for(v1_app, {CONTRACT_VERSION_HEADER: "1.1.0"}) as client:
        response = await client.get("/v1/users", params={"name": "x"})
    assert_wire_error(response, 401, "unauthorized")


async def test_401_with_a_wrong_token(v1_app: FastAPI) -> None:
    headers = {"Authorization": "Bearer wrong", CONTRACT_VERSION_HEADER: "1.1.0"}
    async with client_for(v1_app, headers) as client:
        response = await client.get("/v1/users/u_eric")
    assert_wire_error(response, 401, "unauthorized")


async def test_auth_is_checked_before_the_version(v1_app: FastAPI) -> None:
    async with client_for(v1_app, {}) as client:
        response = await client.get("/v1/users/u_eric")
    assert_wire_error(response, 401, "unauthorized")


async def test_400_without_the_version_header(v1_app: FastAPI) -> None:
    headers = {"Authorization": f"Bearer {V1_TOKEN}"}
    async with client_for(v1_app, headers) as client:
        response = await client.get("/v1/users/u_eric")
    assert_wire_error(response, 400, "contract_version_mismatch")


async def test_400_for_another_major_version(v1_app: FastAPI) -> None:
    headers = {"Authorization": f"Bearer {V1_TOKEN}", CONTRACT_VERSION_HEADER: "2.0.0"}
    async with client_for(v1_app, headers) as client:
        response = await client.get("/v1/users/u_eric")
    assert_wire_error(response, 400, "contract_version_mismatch")


async def test_400_for_a_version_that_is_not_one(v1_app: FastAPI) -> None:
    headers = {"Authorization": f"Bearer {V1_TOKEN}", CONTRACT_VERSION_HEADER: "abc"}
    async with client_for(v1_app, headers) as client:
        response = await client.get("/v1/users/u_eric")
    assert_wire_error(response, 400, "contract_version_mismatch")


async def test_another_minor_version_is_served_with_one_warning(
    v1_app: FastAPI, log_stream: io.StringIO
) -> None:
    headers = {"Authorization": f"Bearer {V1_TOKEN}", CONTRACT_VERSION_HEADER: "1.0.0"}
    async with client_for(v1_app, headers) as client:
        first = await client.get("/v1/users/u_eric")
        second = await client.get("/v1/users/u_eric")
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers[CONTRACT_VERSION_HEADER] == CONTRACT_VERSION
    warnings = [
        json.loads(line)
        for line in log_stream.getvalue().splitlines()
        if '"v1_contract_version_differs"' in line
    ]
    assert len(warnings) == 1
    assert warnings[0]["presented"] == "1.0.0"


async def test_the_version_header_is_on_every_response(
    v1: httpx.AsyncClient,
) -> None:
    responses = [
        await v1.get("/v1/users", params={"name": "Eric Catalano"}),  # 200
        await v1.get("/v1/users/u_eric"),  # 200
        await v1.get("/v1/users/u_nobody"),  # 404
        await v1.get("/v1/users"),  # 422, name missing
        await v1.get("/v1/no-such-route"),  # 404, routing
        await v1.delete("/v1/users/u_eric"),  # 405, routing
    ]
    for response in responses:
        assert response.headers.get(CONTRACT_VERSION_HEADER) == CONTRACT_VERSION, (
            response.request.url,
            response.status_code,
        )
    for response in responses[2:]:
        ErrorResponse.model_validate(response.json())


async def test_a_missing_query_is_a_422_validation_error(v1: httpx.AsyncClient) -> None:
    response = await v1.get("/v1/users")
    assert_wire_error(response, 422, "validation_error")
    assert "name" in (response.json()["error"]["detail"] or "")


async def test_routing_errors_use_the_wire_body(v1: httpx.AsyncClient) -> None:
    assert_wire_error(await v1.get("/v1/no-such-route"), 404, "not_found")
    assert_wire_error(await v1.delete("/v1/users/u_eric"), 405, "method_not_allowed")


async def test_events_still_use_the_internal_error_body(v1_app: FastAPI) -> None:
    async with client_for(v1_app, {}) as client:
        response = await client.post("/events", json={})
    assert response.status_code == 401
    assert response.json()["contract_version"] == "0.1.0-provisional"
    assert CONTRACT_VERSION_HEADER not in response.headers


# ------------------------------------- M3: auth before anything else


async def test_unauthenticated_malformed_body_is_401_not_422(v1_app: FastAPI) -> None:
    async with client_for(v1_app, {CONTRACT_VERSION_HEADER: "1.1.0"}) as client:
        response = await client.post(
            "/v1/sessions",
            content=b"{bad",
            headers={"Content-Type": "application/json"},
        )
    assert_wire_error(response, 401, "unauthorized")


async def test_unauthenticated_unknown_path_is_401_not_404(v1_app: FastAPI) -> None:
    async with client_for(v1_app, {}) as client:
        response = await client.get("/v1/nothing")
    assert_wire_error(response, 401, "unauthorized")


async def test_unauthenticated_wrong_method_is_401_not_405(v1_app: FastAPI) -> None:
    async with client_for(v1_app, {"Authorization": "Bearer wrong"}) as client:
        response = await client.delete("/v1/users/u_eric")
    assert_wire_error(response, 401, "unauthorized")


async def test_with_a_token_those_still_get_their_own_errors(
    v1: httpx.AsyncClient,
) -> None:
    malformed = await v1.post(
        "/v1/sessions", content=b"{bad", headers={"Content-Type": "application/json"}
    )
    assert_wire_error(malformed, 422, "validation_error")
    assert_wire_error(await v1.get("/v1/nothing"), 404, "not_found")
    assert_wire_error(await v1.delete("/v1/users/u_eric"), 405, "method_not_allowed")


async def test_a_non_ascii_token_is_a_clean_401(v1_app: FastAPI) -> None:
    async with client_for(v1_app, {}) as client:
        response = await client.get(
            "/v1/nothing", headers={"Authorization": b"Bearer t\xe9st"}
        )
    assert_wire_error(response, 401, "unauthorized")


async def test_paths_outside_v1_are_not_affected(v1_app: FastAPI) -> None:
    async with client_for(v1_app, {}) as client:
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/v1beta/anything")).status_code == 404
